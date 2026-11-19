# Cherry-picked kernel variants

Unified diffs against `main`'s `hw1-asr/csrc` (65bae974). The code comes from branch
`CUDA-test` (c93c419), split so that each optimisation can be measured on its own.
Apply a patch to a COPY of the sources:

```bash
mkdir /tmp/v && git archive HEAD hw1-asr/csrc | tar -x -C /tmp/v --strip-components=2
patch -p1 -d /tmp/v < experiments/asr/variants/02_gemm_regtile.patch
cd hw1-asr && make SRC=/tmp/v BUILD=build/v_sm_90 test
```

| patch | adds | stack with |
|---|---|---|
| `00_tuning_macros` | `-D` overrides for ROPE/REDUCE threads, TILE, ATTN_WARPS (defaults unchanged) | `04` only (conflicts with 01-03 on the TILE line) |
| `01_gemm_smallm` | `linear_smallm_kernel`, M <= 8 | `04` |
| `02_gemm_regtile` | `linear_regtile_kernel<TM>`, M > 8; `-DASR_GEMM_MIN_BLOCKS_PER_SM` | `04` |
| `03_gemm_smallm_regtile` | 01 + 02 with one combined dispatch | `04` |
| `04_attention_mq` | `attention_mq_kernel`, non-causal; `-DASR_MQ_QPW`, `-DASR_MQ_WARPS` | any one of 00-03 |

`03 + 04` gives the same kernels as `CUDA-test`. `run_report_suite.sh` builds and
measures all of them in one GPU allocation.
