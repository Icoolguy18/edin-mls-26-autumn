// Task 4: attention with an online softmax (the idea behind FlashAttention).
// Read docs/task4-attention.md first.
//
// For one query vector q (one token, one head) and the keys/values of the same
// head:
//     score_j = (q . k_j) * scale
//     p_j     = softmax(score)_j
//     out     = sum_j p_j * v_j
//
// The plain way writes the whole [seq_q, seq_k] score matrix to GPU memory and
// reads it back for the softmax and again for p * V. Here nothing but the output
// goes back to memory. Keys and values are streamed through shared memory one
// tile of 32 at a time, and each query keeps a running max m, a running sum l
// and a running output acc (the MaxSum idea from task 2).
//
// Work split:
//   block  = ATTN_WARPS consecutive queries of one head of one batch element
//   warp   = one query
//   lane   = during scoring: one key of the current tile (KV_TILE = 32 = warp size)
//            during p * V:   output dimensions lane, lane + 32, lane + 64, lane + 96
//
// Layouts (as produced by the projections, no transposes):
//   q, out: [batch, seq_q, num_heads,    head_dim]
//   k, v:   [batch, seq_k, num_kv_heads, head_dim]
// The text decoder uses grouped-query attention (GQA): 16 query heads share 4
// K/V heads. The audio encoder has num_kv_heads == num_heads.
#include "asr_kernels.h"
#include "task2_reduce.cuh"  // warp_reduce_max, warp_reduce_sum from task 2

// [AI-assisted change: Claude (Anthropic), 16 Nov 2026] Build-time override so the
// "Try this 1: 1, 2, 8, 16" sweep needs no source edit, e.g. NVCCFLAGS="... -DASR_ATTN_WARPS=...".
// Without the flag the value is 4, exactly as before; the kernel is unchanged.
#ifndef ASR_ATTN_WARPS
#define ASR_ATTN_WARPS 4
#endif
constexpr int ATTN_WARPS = ASR_ATTN_WARPS;          // queries per block; try 2, 8 or 16
constexpr int KV_TILE = WARP_SIZE;     // keys per tile: one per lane
constexpr int MAX_HEAD_DIM = 128;      // head_dim must be a multiple of 32, at most this
constexpr int MAX_DIMS_PER_LANE = MAX_HEAD_DIM / WARP_SIZE;

__global__ void attention_kernel(const float* q, const float* k, const float* v, float* out,
                                 int num_heads, int num_kv_heads, int seq_q, int seq_k,
                                 int head_dim, float scale, int causal) {
  // Dynamic shared memory, sized by the launcher:
  extern __shared__ float smem[];
  const int k_pitch = head_dim + 1;              // +1 avoids bank conflicts when lanes read rows
  float* k_tile = smem;                          // [KV_TILE][k_pitch]
  float* v_tile = k_tile + KV_TILE * k_pitch;    // [KV_TILE][head_dim]
  float* q_all = v_tile + KV_TILE * head_dim;    // [ATTN_WARPS][head_dim]

  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int h = blockIdx.y;                      // query head
  const int b = blockIdx.z;                      // batch element
  const int qi = blockIdx.x * ATTN_WARPS + warp; // this warp's query
  const bool active = qi < seq_q;                // the last block may have spare warps

  int kv_h = 0;  // the K/V head this query head reads
  // ======================== BLANK 4.1 =========================
  // Set kv_h, the K/V head that query head h reads (GQA: groups of consecutive
  // query heads share one K/V head).
  // YOUR CODE HERE
  // ============================================================
  kv_h = h / (num_heads / num_kv_heads);

  const size_t q_stride = static_cast<size_t>(num_heads) * head_dim;      // token to token in q, out
  const size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;  // token to token in k, v
  const float* k_head = k + static_cast<size_t>(b) * seq_k * kv_stride + kv_h * head_dim;
  const float* v_head = v + static_cast<size_t>(b) * seq_k * kv_stride + kv_h * head_dim;
  // key j of this head: k_head + j * kv_stride, value j: v_head + j * kv_stride

  // Each warp copies its query, already multiplied by scale, into shared memory.
  float* qv = q_all + warp * head_dim;
  if (active) {
    const float* q_row = q + (static_cast<size_t>(b) * seq_q + qi) * q_stride + h * head_dim;
    for (int d = lane; d < head_dim; d += WARP_SIZE) qv[d] = q_row[d] * scale;
  }

  // Causal masking: query qi sits at position qi + (seq_k - seq_q) of the key
  // sequence and may only see keys up to that position.
  const int shift = seq_k - seq_q;
  int last_key = causal ? min(seq_k - 1, qi + shift) : seq_k - 1;
  if (!active) last_key = -1;  // spare warps see nothing
  // The block loops until the last key any of its queries needs. Every thread
  // runs the same number of iterations, so the __syncthreads() inside are safe.
  const int block_last_q = min(seq_q - 1, (blockIdx.x + 1) * ATTN_WARPS - 1);
  const int block_last_key = causal ? min(seq_k - 1, block_last_q + shift) : seq_k - 1;

  const int dims = head_dim / WARP_SIZE;  // output dims per lane (2 or 4 here)
  float m = -INFINITY;                    // running max of the scores seen so far
  float l = 0.0f;                         // running sum of exp(score - m)
  float acc[MAX_DIMS_PER_LANE] = {0.0f};  // running sum of exp(score - m) * v, per lane dim

  for (int j0 = 0; j0 <= block_last_key; j0 += KV_TILE) {
    // ======================== BLANK 4.2 =========================
    // Copy keys and values j0 .. j0+KV_TILE-1 of this head into k_tile (rows of
    // k_pitch floats) and v_tile (rows of head_dim floats), with 0 for keys past
    // seq_k. All threads of the block take part.
    // YOUR CODE HERE
    // ============================================================
    for (int idx = threadIdx.x; idx < KV_TILE * head_dim; idx += blockDim.x){
      int r = idx / head_dim;
      int d = idx % head_dim  ;
      int j = j0 + r;

      if (j < seq_k){
        k_tile[r * k_pitch + d] = k_head[static_cast<size_t>(j) * kv_stride + d];
        v_tile[r * head_dim + d] = v_head[static_cast<size_t>(j) * kv_stride + d];
      }
      else{
        k_tile[r * k_pitch + d] = 0.0f;
        v_tile[r * head_dim + d] = 0.0f;
      }
    }
    __syncthreads();

    float s = -INFINITY;  // this lane's score for key j0 + lane
    // ======================== BLANK 4.3 =========================
    // Set s to this query's score for key j0 + lane, if the query may see that
    // key (last_key).
    // YOUR CODE HERE
    // ============================================================
    if (active && j0 + lane <= last_key){
      s = 0.0f;

      for (int d = 0; d < head_dim; d++){
        s += qv[d] * k_tile[lane * k_pitch + d];
      }
    }

    float p = 0.0f;  // this lane's exp(s - m_new): unnormalised probability of key j0 + lane
    // ======================== BLANK 4.4 =========================
    // Update m, l and acc with this tile (guide, section 3) and set p.
    // YOUR CODE HERE
    // ============================================================
    float tile_max = warp_reduce_max(s);
    float m_new = fmaxf(m, tile_max);

    float corr = 0.0f;
    if (m != -INFINITY) {
      corr = expf(m - m_new);
    }

    p = 0.0f;
    if (s != -INFINITY) {
      p = expf(s - m_new);
    }

    l = l * corr + warp_reduce_sum(p);

    #pragma unroll
    for (int t = 0; t < MAX_DIMS_PER_LANE; ++t) {
      acc[t] *= corr;
    }

    m = m_new;

    // ======================== BLANK 4.5 =========================
    // Add this tile's values, weighted by p, to acc.
    // YOUR CODE HERE
    // ============================================================
    for (int r = 0; r < KV_TILE; ++r){
      float prob = __shfl_sync(FULL_MASK, p, r);

      #pragma unroll
      for (int t = 0; t < MAX_DIMS_PER_LANE; ++t){
        if (t < dims){
          int d = lane + t * WARP_SIZE;
          acc[t] += prob * v_tile[r * head_dim + d];
        }
      }
    }

    __syncthreads();  // all warps are done with this tile before the next one overwrites it
  }

  // ======================== BLANK 4.6 =========================
  // Write acc / l to out for this warp's query.
  // YOUR CODE HERE
  // ============================================================
  if (active){
    float* out_row = out + (static_cast<size_t>(b) * seq_q + qi) * q_stride + h * head_dim;

    #pragma unroll
    for (int t = 0; t < MAX_DIMS_PER_LANE; ++t){
      if (t < dims){
        int d = lane + t * WARP_SIZE;
        out_row[d] = (l > 0.0f) ? acc[t] / l : 0.0f;
      }
    }
  }
}

// =================================================================================
// [AI-assisted optimisation: Claude (Anthropic), 16 Nov 2026]
//
// Why: attention_kernel above takes 700 ms of the all-CUDA profile on H200 MIG
// 1g.18gb, against PyTorch's 162 ms. From the call counts (32 encoder layers) and
// the test timing (21.0 ms per encoder call), about 96% of that is the audio encoder:
// non-causal, 1500 queries x 1500 keys, 20 heads, head_dim 64.
// With one query per warp and ATTN_WARPS = 4:
//   (a) every key and value of a head is loaded from global memory 1500 / 4 = 375
//       times per layer (once per block), and
//   (b) the scoring loop does two shared-memory loads (qv[d], k_tile[...]) per
//       multiply-add, so shared-memory bandwidth limits it.
//
// What this kernel changes (non-causal launches only, i.e. the encoder):
//   - each warp handles MQ_QPW queries instead of one. A key element loaded from
//     shared memory is reused for MQ_QPW dot products, and a value element for MQ_QPW
//     outputs. The block covers MQ_WARPS * MQ_QPW queries, so each K/V tile is
//     loaded from global memory MQ_QPW times less often (4x with the defaults);
//   - scores read k_tile and the queries as float4 (4 floats per load). The row
//     pitch is head_dim + 4 (not + 1) to keep each row 16-byte aligned. With
//     head_dim = 64 or 128 the pitch is 4 (mod 32) banks, so the 8 lanes of each
//     float4 phase touch different banks.
//
// Unchanged: the same online-softmax recurrence as BLANK 4.4 (running m, l, acc,
// with corr = 0 when m is still -INFINITY), the same zero-filled tail tile, and the
// same final acc / l. Each query's dot product adds its head_dim terms in the same
// order as attention_kernel. Causal launches (text decoder) still use
// attention_kernel unchanged; -DASR_ATTN_IMPL=0 restores it for every launch.
// =================================================================================
#ifndef ASR_ATTN_IMPL
#define ASR_ATTN_IMPL 1   // 0 = original kernel for every launch (the baseline)
#endif
#ifndef ASR_MQ_WARPS
#define ASR_MQ_WARPS 4    // warps per block in attention_mq_kernel
#endif
#ifndef ASR_MQ_QPW
#define ASR_MQ_QPW 4      // queries per warp in attention_mq_kernel
#endif
constexpr int MQ_WARPS = ASR_MQ_WARPS;
constexpr int MQ_QPW = ASR_MQ_QPW;
constexpr int MQ_QUERIES = MQ_WARPS * MQ_QPW;  // queries per block

template <int QPW>
__global__ void __launch_bounds__(MQ_WARPS * WARP_SIZE)
attention_mq_kernel(const float* __restrict__ q, const float* __restrict__ k,
                    const float* __restrict__ v, float* __restrict__ out, int num_heads,
                    int num_kv_heads, int seq_q, int seq_k, int head_dim, float scale) {
  extern __shared__ __align__(16) float mq_smem[];
  const int kp = head_dim + 4;                    // 16-byte aligned rows (see above)
  float* k_tile = mq_smem;                        // [KV_TILE][kp]
  float* v_tile = k_tile + KV_TILE * kp;          // [KV_TILE][head_dim]
  float* q_all = v_tile + KV_TILE * head_dim;     // [MQ_WARPS * QPW][head_dim], scaled

  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int kv_h = h / (num_heads / num_kv_heads);  // same mapping as BLANK 4.1
  const int q_first = blockIdx.x * (MQ_WARPS * QPW) + warp * QPW;  // this warp's queries

  const size_t q_stride = static_cast<size_t>(num_heads) * head_dim;
  const size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;
  const float* k_head = k + static_cast<size_t>(b) * seq_k * kv_stride + kv_h * head_dim;
  const float* v_head = v + static_cast<size_t>(b) * seq_k * kv_stride + kv_h * head_dim;

  // Each warp copies its QPW queries, multiplied by scale (0 for queries past seq_q).
  float* q_warp = q_all + warp * QPW * head_dim;
#pragma unroll
  for (int t = 0; t < QPW; ++t) {
    const int qi = q_first + t;
    const float* q_row = q + (static_cast<size_t>(b) * seq_q + qi) * q_stride + h * head_dim;
    for (int d = lane; d < head_dim; d += WARP_SIZE)
      q_warp[t * head_dim + d] = (qi < seq_q) ? q_row[d] * scale : 0.0f;
  }

  const int dims = head_dim / WARP_SIZE;
  float m[QPW], l[QPW], acc[QPW][MAX_DIMS_PER_LANE];
#pragma unroll
  for (int t = 0; t < QPW; ++t) {
    m[t] = -INFINITY;
    l[t] = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_DIMS_PER_LANE; ++i) acc[t][i] = 0.0f;
  }

  // Non-causal: every block walks all keys, so every thread runs the same number of
  // iterations and both __syncthreads() below are reached by all threads.
  for (int j0 = 0; j0 < seq_k; j0 += KV_TILE) {
    // Copy keys/values j0 .. j0+31 (zeros past seq_k), as in BLANK 4.2.
    for (int idx = threadIdx.x; idx < KV_TILE * head_dim; idx += blockDim.x) {
      const int r = idx / head_dim;
      const int d = idx % head_dim;
      const int j = j0 + r;
      const bool in = j < seq_k;
      k_tile[r * kp + d] = in ? k_head[static_cast<size_t>(j) * kv_stride + d] : 0.0f;
      v_tile[r * head_dim + d] = in ? v_head[static_cast<size_t>(j) * kv_stride + d] : 0.0f;
    }
    __syncthreads();

    // Scores: lane = key j0 + lane; one float4 of k feeds QPW queries.
    float s[QPW];
#pragma unroll
    for (int t = 0; t < QPW; ++t) s[t] = 0.0f;
    const float* k_row = k_tile + lane * kp;
    for (int d = 0; d < head_dim; d += 4) {
      const float4 k4 = *reinterpret_cast<const float4*>(k_row + d);
#pragma unroll
      for (int t = 0; t < QPW; ++t) {
        const float4 q4 = *reinterpret_cast<const float4*>(q_warp + t * head_dim + d);
        s[t] += q4.x * k4.x;  // same term order as the scalar loop in BLANK 4.3
        s[t] += q4.y * k4.y;
        s[t] += q4.z * k4.z;
        s[t] += q4.w * k4.w;
      }
    }
    const bool key_ok = j0 + lane < seq_k;

    // Online-softmax update per query: the recurrence of BLANK 4.4.
    float p[QPW];
#pragma unroll
    for (int t = 0; t < QPW; ++t) {
      if (!key_ok || q_first + t >= seq_q) s[t] = -INFINITY;
      const float m_new = fmaxf(m[t], warp_reduce_max(s[t]));
      const float corr = (m[t] != -INFINITY) ? expf(m[t] - m_new) : 0.0f;
      p[t] = (s[t] != -INFINITY) ? expf(s[t] - m_new) : 0.0f;
      l[t] = l[t] * corr + warp_reduce_sum(p[t]);
#pragma unroll
      for (int i = 0; i < MAX_DIMS_PER_LANE; ++i) acc[t][i] *= corr;
      m[t] = m_new;
    }

    // p * V: one value element read from shared memory feeds QPW outputs (BLANK 4.5).
    for (int r = 0; r < KV_TILE; ++r) {
      float vv[MAX_DIMS_PER_LANE];
#pragma unroll
      for (int i = 0; i < MAX_DIMS_PER_LANE; ++i)
        vv[i] = (i < dims) ? v_tile[r * head_dim + lane + i * WARP_SIZE] : 0.0f;
#pragma unroll
      for (int t = 0; t < QPW; ++t) {
        const float prob = __shfl_sync(FULL_MASK, p[t], r);
#pragma unroll
        for (int i = 0; i < MAX_DIMS_PER_LANE; ++i)
          if (i < dims) acc[t][i] += prob * vv[i];
      }
    }
    __syncthreads();  // all warps are done with this tile before the next overwrites it
  }

  // Write acc / l, as in BLANK 4.6.
#pragma unroll
  for (int t = 0; t < QPW; ++t) {
    const int qi = q_first + t;
    if (qi >= seq_q) continue;
    float* out_row = out + (static_cast<size_t>(b) * seq_q + qi) * q_stride + h * head_dim;
#pragma unroll
    for (int i = 0; i < MAX_DIMS_PER_LANE; ++i)
      if (i < dims) out_row[lane + i * WARP_SIZE] = (l[t] > 0.0f) ? acc[t][i] / l[t] : 0.0f;
  }
}


ASR_API int launch_attention(const float* q, const float* k, const float* v, float* out,
                             int batch, int num_heads, int num_kv_heads, int seq_q,
                             int seq_k, int head_dim, float scale, int causal,
                             cudaStream_t stream) {
  if (head_dim % WARP_SIZE != 0 || head_dim > MAX_HEAD_DIM || num_heads % num_kv_heads != 0)
    return static_cast<int>(cudaErrorInvalidValue);
  size_t smem = sizeof(float) *
                (KV_TILE * (head_dim + 1) + KV_TILE * head_dim + ATTN_WARPS * head_dim);
  if (smem > 48 * 1024) {  // above 48 KB a kernel must ask for more shared memory
    cudaError_t err = cudaFuncSetAttribute(
        attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (err != cudaSuccess) return static_cast<int>(err);
  }
#if ASR_ATTN_IMPL != 0
  // [AI-assisted] Non-causal launches (the audio encoder) use attention_mq_kernel.
  // Causal launches (text decoder) keep the original kernel below, unchanged.
  if (!causal) {
    const size_t mq_smem = sizeof(float) * (KV_TILE * (head_dim + 4) + KV_TILE * head_dim +
                                            MQ_QUERIES * head_dim);
    if (mq_smem > 48 * 1024) {
      cudaError_t err = cudaFuncSetAttribute(attention_mq_kernel<MQ_QPW>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(mq_smem));
      if (err != cudaSuccess) return static_cast<int>(err);
    }
    dim3 mq_grid(ceil_div(seq_q, MQ_QUERIES), num_heads, batch);
    attention_mq_kernel<MQ_QPW><<<mq_grid, MQ_WARPS * WARP_SIZE, mq_smem, stream>>>(
        q, k, v, out, num_heads, num_kv_heads, seq_q, seq_k, head_dim, scale);
    RETURN_LAUNCH_ERROR();
  }
#endif
  dim3 grid(ceil_div(seq_q, ATTN_WARPS), num_heads, batch);
  attention_kernel<<<grid, ATTN_WARPS * WARP_SIZE, smem, stream>>>(
      q, k, v, out, num_heads, num_kv_heads, seq_q, seq_k, head_dim, scale, causal);
  RETURN_LAUNCH_ERROR();
}
