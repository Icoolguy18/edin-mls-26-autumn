// Task 3: the linear layer as a tiled GEMM. Read docs/task3-gemm.md first.
//
//     y[M, N] = act( x[M, K] * w[N, K]^T + bias[N] )
//
// This is the tiled GEMM from cuda-tutorial/5-tiled-gemm, with three changes:
//   1. w is stored the way PyTorch stores Linear weights: one row per OUTPUT
//      feature, [N, K]. So column n of the result uses row n of w.
//   2. bias is added, and GELU can be applied, before the result is stored
//      (the "epilogue"), so no extra kernel has to read y again.
//   3. every Linear layer of the ASR model and both convolutions run through it.
//
// Block (TILE x TILE threads) computes a TILE x TILE piece of y. Thread (ty, tx)
// computes y[row][col] with row = blockIdx.y*TILE + ty, col = blockIdx.x*TILE + tx.
// The block walks along K one tile at a time, staging a tile of x and a tile of w
// in shared memory so every loaded value is used TILE times.
#include <cstdint>

#include "asr_kernels.h"
#include "common.cuh"
#include "task2_reduce.cuh"  // [AI-assisted] warp_reduce_sum for the small-M kernel below

// ---------------------------------------------------------------------------------
// [AI-assisted change: Claude (Anthropic), 16 Nov 2026]
// Build-time switches, so experiments need no source edits:
//   -DASR_TILE=8|16|32     tile size of the original kernel ("Try this" 1)
//   -DASR_GEMM_IMPL=0      use ONLY the original tiled kernel (the baseline)
//   -DASR_GEMM_IMPL=1      default: shape-based dispatch to the optimised kernels
//   -DASR_GEMM_MIN_BLOCKS_PER_SM=N  tallest register tile that still gives N blocks per SM
// Defaults reproduce the measured baseline when ASR_GEMM_IMPL=0.
// ---------------------------------------------------------------------------------
#ifndef ASR_TILE
#define ASR_TILE 16
#endif
#ifndef ASR_GEMM_MIN_BLOCKS_PER_SM
#define ASR_GEMM_MIN_BLOCKS_PER_SM 2  // row-block height rule below; sweep 1, 2, 4
#endif
#ifndef ASR_GEMM_IMPL
#define ASR_GEMM_IMPL 1
#endif

constexpr int TILE = ASR_TILE;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)

__global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                    float* y, int M, int N, int K, int act) {
  __shared__ float xs[TILE][TILE];      // xs[m][k]: rows of x for this block
  __shared__ float ws[TILE][TILE + 1];  // ws[n][k]: rows of w for this block (+1: see Try this)

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;  // the output row this thread computes
  const int col = blockIdx.x * TILE + tx;  // the output column this thread computes

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // ======================== BLANK 3.1 =========================
    // Load this thread's element of the x tile into xs. Use 0 outside x.
    // YOUR CODE HERE
    // ============================================================
    int input_k = k0 + tx;
    xs[ty][tx] =
      (row < M && input_k < K)?
        x[static_cast<size_t>(row) * K + input_k] :
        0.0f;

    // ======================== BLANK 3.2 =========================
    // Load this thread's element of the w tile, so that ws[n][k] holds
    // w[blockIdx.x * TILE + n][k0 + k]. Use 0 outside w. Threads with consecutive
    // tx must read consecutive addresses of w.
    // YOUR CODE HERE
    // ============================================================
    int weight_row = blockIdx.x * TILE + ty;
    int weight_k = k0 + tx;
    ws[ty][tx] =
      (weight_row < N && weight_k < K)?
        w[static_cast<size_t>(weight_row) * K + weight_k] :
        0.0f;

    // ======================== BLANK 3.3 =========================
    // Add this K step's contribution to acc.
    // YOUR CODE HERE
    // ============================================================
    __syncthreads();

    for (int k = 0; k < TILE; ++k){
      acc += xs[ty][k] * ws[tx][k];
      }
      __syncthreads();
  }

  // ======================== BLANK 3.4 =========================
  // Epilogue: for outputs inside y, add bias[col] (bias may be nullptr), apply
  // gelu() when act == ACT_GELU, and store.
  // YOUR CODE HERE
  // ============================================================
  if (row < M && col < N){
    if (bias != nullptr){
      acc += bias[col];
    }

    if (act == ACT_GELU){
      acc = gelu(acc);
    }

    y[static_cast<size_t>(row) * N + col] = acc;
  }
}

// =================================================================================
// [AI-assisted optimisation: Claude (Anthropic), 16 Nov 2026]
//
// Why: the all-CUDA profile (H200 MIG 1g.18gb) spends 4,966 of 5,691 ms in `linear`.
// The tiled kernel above has two separate bottlenecks, one per kind of shape:
//
//   (a) Large M (audio encoder, decoder prefill). Each thread computes ONE output, so
//       its inner loop does two shared-memory loads per multiply-add
//       (xs[ty][k] and ws[tx][k]). Shared-memory bandwidth, not arithmetic, limits it:
//       ~930 GFLOP/s against cuBLAS's ~6,170 on encoder fc1.
//       -> linear_regtile_kernel: each thread computes a TM x 4 block of outputs held
//          in registers, so every value loaded from shared memory is reused TM or 4
//          times (0.5 loads per multiply-add with TM = 4, instead of 2).
//
//   (b) Small M (token-by-token decoding: M = 1). With 16 x 16 blocks only the ty = 0
//       row has a real output, so 15/16 of the threads do padding work. lm_head reads
//       its 485 MB weight matrix at ~119 GB/s, about 21% of the slice's ~563 GB/s.
//       At M = 1 the job is to stream w once, as fast as possible.
//       -> linear_smallm_kernel: one warp streams SM_COLS rows of w with coalesced
//          float4 loads, keeps one partial sum per (row of x, column) in registers,
//          and combines the 32 lanes with warp_reduce_sum (task 2) at the end.
//
// Correctness notes:
//   - Both kernels zero-pad exactly like the original, apply the same epilogue (bias,
//     then GELU) and write each output once.
//   - linear_regtile_kernel adds the K terms of every output in the same order as
//     the original kernel (k = 0, 1, ..., K-1), so results should match it closely.
//   - linear_smallm_kernel sums in a different order (32 strided partial sums, then
//     a tree), so its rounding differs slightly from a sequential loop. The tests'
//     tolerance (1e-3 absolute and relative) is far above float32 rounding at these K.
//   - float4 loads are used only when K % 4 == 0 and both x and w are 16-byte
//     aligned. Otherwise a scalar version of the same kernel runs (e.g. K = 7, 33, 129).
//   - No library calls: everything below is plain CUDA, as the rules require.
// =================================================================================

// ---- (a) register-tiled kernel for large M ----------------------------------------
constexpr int RT_BN = 64;               // output columns per block
constexpr int RT_BK = 16;               // K step staged in shared memory per iteration
constexpr int RT_TN = 4;                // output columns per thread
constexpr int RT_DIM = 16;              // threads per block side: 16 x 16 = 256 threads
constexpr int RT_THREADS = RT_DIM * RT_DIM;

// TM = output rows per thread, so the block covers BM = 16 * TM rows. TM is a template
// parameter so that acc[TM][RT_TN] stays in registers (fixed loop bounds).
template <int TM>
__global__ void __launch_bounds__(RT_THREADS)
linear_regtile_kernel(const float* __restrict__ x, const float* __restrict__ w,
                      const float* __restrict__ bias, float* __restrict__ y,
                      int M, int N, int K, int act) {
  constexpr int BM = RT_DIM * TM;
  // Stored K-major ([k][m] and [k][n]) so the inner loop reads along m / n.
  // +1 padding keeps the transposed stores below from all hitting one bank.
  __shared__ float xs[RT_BK][BM + 1];
  __shared__ float ws[RT_BK][RT_BN + 1];

  const int tid = threadIdx.x;
  const int tx = tid % RT_DIM;  // this thread's columns: col0 + tx + 16 * j
  const int ty = tid / RT_DIM;  // this thread's rows:    row0 + ty + 16 * i
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * RT_BN;

  float acc[TM][RT_TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < RT_TN; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += RT_BK) {
    // Load the x tile (BM x RT_BK). Consecutive threads read consecutive k of one row
    // of x, so each group of 16 threads reads 64 contiguous bytes (coalesced).
#pragma unroll
    for (int e = 0; e < (BM * RT_BK) / RT_THREADS; ++e) {
      const int idx = tid + e * RT_THREADS;
      const int r = idx / RT_BK;
      const int kk = idx % RT_BK;
      const int gr = row0 + r;
      const int gk = k0 + kk;
      xs[kk][r] = (gr < M && gk < K) ? x[static_cast<size_t>(gr) * K + gk] : 0.0f;
    }
    // Load the w tile (RT_BN x RT_BK) the same way: 16 threads per row of w.
#pragma unroll
    for (int e = 0; e < (RT_BN * RT_BK) / RT_THREADS; ++e) {
      const int idx = tid + e * RT_THREADS;
      const int r = idx / RT_BK;
      const int kk = idx % RT_BK;
      const int gn = col0 + r;
      const int gk = k0 + kk;
      ws[kk][r] = (gn < N && gk < K) ? w[static_cast<size_t>(gn) * K + gk] : 0.0f;
    }
    __syncthreads();  // reached by every thread: no early returns in this kernel

    // Outer product in registers: TM + 4 shared loads feed TM * 4 multiply-adds.
    // Reads of ws[kk][tx + 16 j] are 16 consecutive floats (no bank conflicts);
    // reads of xs[kk][ty + 16 i] are 2 addresses per warp (broadcast).
#pragma unroll
    for (int kk = 0; kk < RT_BK; ++kk) {
      float a[TM], b[RT_TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = xs[kk][ty + RT_DIM * i];
#pragma unroll
      for (int j = 0; j < RT_TN; ++j) b[j] = ws[kk][tx + RT_DIM * j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < RT_TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();  // everyone is done with this tile before it is overwritten
  }

  // Epilogue, identical in meaning to BLANK 3.4. Consecutive tx write consecutive
  // columns, so the stores are coalesced.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = row0 + ty + RT_DIM * i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < RT_TN; ++j) {
      const int col = col0 + tx + RT_DIM * j;
      if (col >= N) continue;
      float v = acc[i][j];
      if (bias != nullptr) v += bias[col];
      if (act == ACT_GELU) v = gelu(v);
      y[static_cast<size_t>(row) * N + col] = v;
    }
  }
}

// ---- (b) weight-streaming kernel for small M (decoding) ---------------------------
constexpr int SM_MAX_M = 8;  // M at or below this uses the small-M kernel
constexpr int SM_COLS = 4;   // columns (rows of w) per warp: each x load is reused 4 times
constexpr int SM_WARPS = 4;  // warps per block (128 threads)

// MR = compile-time upper bound on M (1, 2, 4 or 8) so acc[MR][SM_COLS] stays in
// registers. VEC selects float4 loads (needs K % 4 == 0 and 16-byte alignment).
// There is no __syncthreads() in this kernel, so a warp may return early.
template <int MR, bool VEC>
__global__ void __launch_bounds__(SM_WARPS * WARP_SIZE)
linear_smallm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                     const float* __restrict__ bias, float* __restrict__ y,
                     int M, int N, int K, int act) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int n0 = (blockIdx.x * SM_WARPS + warp) * SM_COLS;  // first column of this warp
  if (n0 >= N) return;                                      // whole warp leaves together

  float acc[MR][SM_COLS];
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int c = 0; c < SM_COLS; ++c) acc[m][c] = 0.0f;

  if (VEC) {
    // Lane l reads float4 number l, l + 32, ... of each row: one warp covers 512
    // contiguous bytes of a row of w per step, so the loads are fully coalesced.
    const int K4 = K / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (int k4 = lane; k4 < K4; k4 += WARP_SIZE) {
      float4 wv[SM_COLS];
#pragma unroll
      for (int c = 0; c < SM_COLS; ++c) {
        const int n = n0 + c;
        wv[c] = (n < N) ? __ldg(&w4[static_cast<size_t>(n) * K4 + k4])
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        if (m >= M) break;
        const float4 xv = __ldg(&x4[static_cast<size_t>(m) * K4 + k4]);
#pragma unroll
        for (int c = 0; c < SM_COLS; ++c) {
          acc[m][c] += xv.x * wv[c].x;
          acc[m][c] += xv.y * wv[c].y;
          acc[m][c] += xv.z * wv[c].z;
          acc[m][c] += xv.w * wv[c].w;
        }
      }
    }
  } else {
    // Scalar version of the same loop, for odd K or unaligned pointers.
    for (int k = lane; k < K; k += WARP_SIZE) {
      float wv[SM_COLS];
#pragma unroll
      for (int c = 0; c < SM_COLS; ++c) {
        const int n = n0 + c;
        wv[c] = (n < N) ? __ldg(&w[static_cast<size_t>(n) * K + k]) : 0.0f;
      }
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        if (m >= M) break;
        const float xv = __ldg(&x[static_cast<size_t>(m) * K + k]);
#pragma unroll
        for (int c = 0; c < SM_COLS; ++c) acc[m][c] += xv * wv[c];
      }
    }
  }

  // Combine the 32 lanes' partial sums (warp_reduce_sum from task 2 leaves the total
  // in every lane), then lane 0 applies the epilogue and stores.
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int c = 0; c < SM_COLS; ++c) acc[m][c] = warp_reduce_sum(acc[m][c]);

  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      if (m >= M) break;
#pragma unroll
      for (int c = 0; c < SM_COLS; ++c) {
        const int n = n0 + c;
        if (n >= N) continue;
        float v = acc[m][c];
        if (bias != nullptr) v += bias[n];
        if (act == ACT_GELU) v = gelu(v);
        y[static_cast<size_t>(m) * N + n] = v;
      }
    }
  }
}

// [AI-assisted] Number of SMs of the current GPU, looked up once. Used only to choose
// a row-block height that still gives every SM some blocks to work on.
static int gemm_num_sms() {
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        sms <= 0)
      sms = 16;  // fall back to the 1g.18gb slice's count; only affects the choice below
  }
  return sms;
}

template <int MR>
static void launch_smallm(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, bool vec, cudaStream_t stream) {
  dim3 grid(ceil_div(N, SM_WARPS * SM_COLS));
  dim3 block(SM_WARPS * WARP_SIZE);
  if (vec)
    linear_smallm_kernel<MR, true><<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
  else
    linear_smallm_kernel<MR, false><<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
}

template <int TM>
static void launch_regtile(const float* x, const float* w, const float* bias, float* y,
                           int M, int N, int K, int act, cudaStream_t stream) {
  dim3 grid(ceil_div(N, RT_BN), ceil_div(M, RT_DIM * TM));
  linear_regtile_kernel<TM><<<grid, RT_THREADS, 0, stream>>>(x, w, bias, y, M, N, K, act);
}

ASR_API int launch_linear(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, cudaStream_t stream) {
#if ASR_GEMM_IMPL == 0
  // Baseline: the original tiled kernel for every shape (unchanged launch).
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  linear_tiled_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
#else
  // [AI-assisted] Shape-based dispatch. The C signature is unchanged, so ops.py and
  // the tests call it exactly as before.
  if (M <= SM_MAX_M) {
    const bool vec = (K % 4 == 0) && (reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
                     (reinterpret_cast<uintptr_t>(w) % 16 == 0);
    if (M <= 1)      launch_smallm<1>(x, w, bias, y, M, N, K, act, vec, stream);
    else if (M <= 2) launch_smallm<2>(x, w, bias, y, M, N, K, act, vec, stream);
    else if (M <= 4) launch_smallm<4>(x, w, bias, y, M, N, K, act, vec, stream);
    else             launch_smallm<8>(x, w, bias, y, M, N, K, act, vec, stream);
  } else {
    // Prefer the tallest row block (most reuse) that still gives at least two blocks
    // per SM. Short, narrow layers (e.g. a 512-wide k_proj at prefill) would otherwise
    // leave most SMs idle.
    const int min_blocks = ASR_GEMM_MIN_BLOCKS_PER_SM * gemm_num_sms();
    const int col_blocks = ceil_div(N, RT_BN);
    if (ceil_div(M, RT_DIM * 4) * col_blocks >= min_blocks)
      launch_regtile<4>(x, w, bias, y, M, N, K, act, stream);
    else if (ceil_div(M, RT_DIM * 2) * col_blocks >= min_blocks)
      launch_regtile<2>(x, w, bias, y, M, N, K, act, stream);
    else
      launch_regtile<1>(x, w, bias, y, M, N, K, act, stream);
  }
#endif
  RETURN_LAUNCH_ERROR();
}
