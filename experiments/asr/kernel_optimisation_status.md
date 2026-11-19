# ASR kernels: frozen main line, cherry-picked optimisations, run plan

[AI-assisted record: Claude (Anthropic), 17 Nov 2026. Numbers come from the saved
logs named here; nothing below is a new GPU measurement.]

Scope: the hand-in kernels stay on `main`. Branch `CUDA-test` (c93c419) is an
extension for the report. Each of its optimisations is cherry-picked onto `main`'s
code as a separate patch (`experiments/asr/variants/`) and benchmarked on its own.
Neither branch's `hw1-asr/` is modified.

## 1. Frozen main line (TODO 1)

| Item | Value |
|---|---|
| Revision | `main` = `65bae974`; suggested tag `baseline-original` (to be created by the author) |
| Kernel source | identical to `9ffa05af`; `git diff 9ffa05a 65bae97 -- hw1-asr` is empty |
| Cluster evidence | `revision.txt` = `9ffa05af6a25...`, empty `source_changes.patch` (experiment log, 17 Nov) |
| Build | supplied Makefile: `-O3 -std=c++17 -lineinfo -diag-suppress 177 -arch=sm_90 -Xcompiler -fPIC`; nvcc from `/opt/cuda-12.8.0` (historical `nvcc --version` not saved) |
| Parameters | `TILE=16`, `ATTN_WARPS=4`, `ROPE_THREADS=256`, `REDUCE_THREADS=256` |
| GPU | H200 MIG 1g.18gb (sm_90) |

| `--cuda` | mean ms (3 runs) | ms/token | linear ms | attention ms | status |
|---|---:|---:|---:|---:|---|
| none (PyTorch) | 1337.6 | 102.9 | 1115.3 | 162.3 | PASS |
| all | 5688.8 | 437.6 | 4965.6 | 700.3 | PASS |
| task1,task2 | 1297.3 | 99.8 | 1115.2 | 208.4 | PASS |

The 3g.71gb all-CUDA run (1640.3 ms) used a different GPU and is not comparable.

## 2. Review of `CUDA-test` and the cherry-picks (TODO 2)

`git diff main CUDA-test` touches only the four task files. The original kernel bodies
are unchanged, and no supplied file differs. The changes split into independent pieces:

| Patch | Contents (code taken verbatim from c93c419) | TODO | Laptop evidence for the same code (RTX 5070 Laptop, 16 Nov) |
|---|---|---|---|
| `01_gemm_smallm` | `linear_smallm_kernel` for M <= 8; larger M keeps `linear_tiled_kernel` | 3 | lm_head 1.917 -> 1.303 ms = cuBLAS; 372 GB/s, bandwidth-bound |
| `02_gemm_regtile` | `linear_regtile_kernel<TM>` for M > 8; M <= 8 keeps the original | 4 | encoder q_proj / fc1 / conv2: 2.4-2.6x; about 46% of cuBLAS |
| `03_gemm_smallm_regtile` | 01 + 02 with the combined dispatch (they edit the same launcher) | 3+4 | as above |
| `04_attention_mq` | `attention_mq_kernel` for non-causal launches; causal keeps the original | 6 | audio encoder 11.795 -> 4.350 ms (2.7x); decoder unchanged |
| `00_tuning_macros` | `#ifndef` overrides for ROPE/REDUCE threads, TILE, ATTN_WARPS (same defaults) | guide sweeps | none: values unchanged |

Checks done here: each patch applies cleanly to `main`'s `csrc`; `03 + 04` reproduces
the `CUDA-test` kernels exactly, apart from its `IMPL` switch plumbing; every variant
compiles for sm_90 and sm_120 with no warnings. Not done: none of the variants has run
on a GPU from these patches. The laptop numbers above come from the `CUDA-test` build
(`IMPL=0` vs default), so the cluster suite must confirm them.

`00` cannot be stacked with `01`-`03`, because both edit the `TILE` line. It is used
only for the guide sweeps on the original kernels.

Observations to carry into the report:
- On the laptop (36 SMs), decoder q_proj/down_proj at M = 64 did not speed up. The
  ">= 2 blocks per SM" rule falls back to TM = 1 for them. On the 16-SM 1g.18gb
  slice it selects TM = 4, so cluster results can differ (`sweep_gemm_min_blocks`).
- Task 1/2 kernels are unchanged. Their laptop differences of about 0.005 ms are
  launch-overhead noise.
- benchmark.py's `+/-` is `np.std` over the timed runs (population SD).

## 3. Resource usage (TODO 5): ptxas, sm_90, `all` variant

Produced by `python3 experiments/asr/resource_usage.py --sm 90 --src <all variant>`
(nvcc 13.5 here; the suite reruns it with the cluster's 12.8). No kernel spills.

| kernel | threads | regs | smem B | blocks/SM | theor. occupancy | limit |
|---|---:|---:|---:|---:|---:|---|
| `linear_tiled_kernel` (main) | 256 | 32 | 2112 | 8 | 100% | regs = threads |
| `linear_regtile_kernel<1>` / `<2>` | 256 | 40 | 5248 / 6272 | 6 | 75% | regs |
| `linear_regtile_kernel<4>` | 256 | 64 | 8320 | 4 | 50% | regs |
| `linear_smallm_kernel<1, float4>` (lm_head) | 128 | 44 | 0 | 10 | 62% | regs |
| `attention_kernel` (main), head_dim 64 | 128 | 40 | 17536 dyn | 12 | 75% | regs = smem |
| `attention_mq_kernel<4>`, head_dim 64 | 128 | 80 | 20992 dyn | 6 | 38% | regs |
| `attention_mq_kernel<8>` (QPW=8), head_dim 64 | 128 | 112 | 25088 dyn | 4 | 25% | regs |

The optimised kernels trade occupancy for data reuse. regtile<4> runs at 50%
occupancy yet is 2.4x faster than the 100%-occupancy original (laptop). The
`sweep_mq_qpw` and `sweep_gemm_min_blocks` results show where that trade stops paying.

## 4. Cluster run (TODO 7-10)

After committing `experiments/asr/` (scripts and patches, not `runs/`) on `main` and
pulling on the cluster:

```bash
srun -p Teaching --gres=gpu:h200_1g.18gb:1 --cpus-per-task=2 --mem=16G --time=02:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate mls-asr
cd ~/edin-mls-26-autumn && git pull
bash experiments/asr/run_report_suite.sh --guide-sweeps 2>&1 | tee suite_console.log
```

Each variant is built from the committed `csrc` (`git archive HEAD`) plus its patches,
in a separate build directory, then gets `make test` and full-model runs:

| variant | patches | ASR `--cuda` configs |
|---|---|---|
| baseline | - | none, all, task1,task2 |
| gemm_smallm | 01 | all |
| gemm_regtile | 02 | all |
| gemm_smallm_regtile | 03 | all |
| attention_mq | 04 | all |
| all | 03 + 04 | all, task1,task2,task4 |

These are followed by sweeps (kernel tests only: min-blocks on 02, MQ_QPW/MQ_WARPS on 04,
and the guide sweeps on 00), `resources_*.md`, and `suite_summary.md`. The summary gives
per-case speed-ups against the baseline, the ASR table against PyTorch, and a warning
if the GPU changed during the suite. To repeat a subset, use `--only "baseline all"`.

**TODO 10:** `suite_summary.md` names the fastest passing configuration. To make it the
hand-in code, apply its patches to `main` and commit. `make lib`, `./run.sh` and the
RAG adapter then build what was measured. Then:

```bash
bash experiments/asr/collect_asr_stats.sh --label final --configs "<chosen>"
ASR_CUDA=<chosen> python rag/scripts/run_audio_query.py --device cpu   # integration check
```

If `main` is to stay the original, report the variants as the extension and leave
`main`'s kernels as they are.
