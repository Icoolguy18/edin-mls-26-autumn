#!/usr/bin/env bash
# All ASR measurements for the report in ONE GPU allocation: the frozen main-line
# kernels and each optimisation cherry-picked from branch CUDA-test, applied to
# main ONE AT A TIME, so each change is measured on its own and on the same GPU/MIG
# slice (TODO items 3-10).
#
# [AI-assisted tooling: Claude (Anthropic), 17 Nov 2026]. Not part of the hand-in code.
# It never edits hw1-asr/: every variant is built from a temporary copy of the
# COMMITTED hw1-asr/csrc (git archive HEAD) plus the patches in
# experiments/asr/variants/, via collect_asr_stats.sh --src.
#
# Variants (patches applied to main's csrc):
#   baseline              none (frozen main)          ASR: none; all; task1,task2
#   gemm_smallm           01  (TODO 3: M <= 8 path)   ASR: all
#   gemm_regtile          02  (TODO 4: M > 8 path)    ASR: all
#   gemm_smallm_regtile   03  (01 + 02 together)      ASR: all
#   attention_mq          04  (TODO 6)                ASR: all
#   all                   03 + 04  (= CUDA-test)      ASR: all; task1,task2,task4
# Sweeps (kernel tests only): ASR_GEMM_MIN_BLOCKS_PER_SM on 02, ASR_MQ_QPW and
# ASR_MQ_WARPS on 04; with --guide-sweeps also the task guides' sweeps on 00.
#
# Run on a GPU node, from the main checkout:
#   srun -p Teaching --gres=gpu:h200_1g.18gb:1 --cpus-per-task=2 --mem=16G --time=02:00:00 --pty bash
#   export PATH=/opt/cuda-12.8.0/bin:$PATH
#   source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate mls-asr
#   bash experiments/asr/run_report_suite.sh 2>&1 | tee suite_console.log
#
# Options:
#   --quick           kernel tests + sweeps only, no full-model runs
#   --guide-sweeps    also ROPE/REDUCE threads, TILE and ATTN_WARPS sweeps (original kernels)
#   --runs N          timed transcriptions per ASR configuration (default 5)
#   --only "a b"      run only these steps: variant names above and/or "sweeps",
#                     e.g. --only "baseline all"
#
# Output: experiments/asr/runs/suite_<date>/ : one directory per variant/sweep, the
# variant sources (src/), resources_*.md, and suite_summary.md (compare_runs.py).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PATCHES="$HERE/variants"
QUICK=0; GUIDE=0; RUNS=5; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --guide-sweeps) GUIDE=1; shift ;;
    --runs) RUNS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)"; exit 2 ;;
  esac
done

command -v nvidia-smi >/dev/null || { echo "Run on a GPU node (srun ...)."; exit 1; }
command -v patch >/dev/null || { echo "'patch' not found."; exit 1; }
if [[ -n "$(cd "$REPO" && git status --short -- hw1-asr/csrc)" ]]; then
  echo "NOTE: hw1-asr/csrc has uncommitted changes. They are NOT measured: every variant"
  echo "      starts from the committed csrc (git archive HEAD)."
fi

SUITE="$REPO/experiments/asr/runs/suite_$(date +%Y%m%d-%H%M)"
mkdir -p "$SUITE/src"
nvidia-smi -L > "$SUITE/gpu_start.txt"
{ (cd "$REPO" && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD)
  (cd "$REPO" && git describe --tags --exact-match HEAD 2>/dev/null) || echo "(HEAD is not tagged)"
} > "$SUITE/revision.txt"
SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
echo "Suite: $SUITE (sm_$SM, base $(head -n1 "$SUITE/revision.txt"))"

make_src() {  # name, patch numbers... -> $SUITE/src/<name>, committed csrc + patches
  local name="$1"; shift
  local dir="$SUITE/src/$name"
  mkdir -p "$dir"
  (cd "$REPO" && git archive HEAD hw1-asr/csrc) | tar -x -C "$dir" --strip-components=2
  for n in "$@"; do
    local p; p="$(ls "$PATCHES"/"${n}"_*.patch)"
    patch -s -p1 -d "$dir" < "$p" || { echo "patch $p failed for $name"; return 1; }
    echo "$(basename "$p")" >> "$dir/APPLIED_PATCHES.txt"
  done
  [[ -f "$dir/APPLIED_PATCHES.txt" ]] || echo "(none: frozen main)" > "$dir/APPLIED_PATCHES.txt"
}

want() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }

collect() {  # label, then collect_asr_stats.sh options; moves the run dir into $SUITE
  local label="$1"; shift
  echo; echo "=== $label ($(date +%H:%M:%S))"
  local marker="$SUITE/.$label.start"; touch "$marker"
  bash "$HERE/collect_asr_stats.sh" --label "$label" "$@" > "$SUITE/$label.console.log" 2>&1
  local new; new="$(find "$REPO/experiments/asr/runs" -mindepth 1 -maxdepth 1 -type d \
                    -newer "$marker" -name "*_${label}_*" ! -name 'suite_*' | head -n1)"
  rm -f "$marker" 2>/dev/null
  if [[ -n "$new" ]]; then
    mv "$new" "$SUITE/$label"
    grep -E "Kernel tests:|^  (pytorch|cuda|mixed)" "$SUITE/$label/summary.txt" | head -n 12
  else
    echo "  no run directory produced; see $SUITE/$label.console.log"
  fi
}

ASR=(--runs "$RUNS"); [[ $QUICK -eq 1 ]] && ASR=(--skip-asr)

# ---- variants: kernel tests + full model (items 3, 4, 6, 7, 8, 9) -------------------
#      name                 patches  ASR configurations
VARIANTS=("baseline||none;all;task1,task2"
          "gemm_smallm|01|all"
          "gemm_regtile|02|all"
          "gemm_smallm_regtile|03|all"
          "attention_mq|04|all"
          "all|03 04|all;task1,task2,task4")
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name patches configs <<< "$v"
  want "$name" || continue
  # shellcheck disable=SC2086
  make_src "$name" $patches || continue
  collect "$name" --src "$SUITE/src/$name" --configs "$configs" "${ASR[@]}"
done

# ---- resource usage (item 5): every kernel is in the 'all' source -------------------
echo; echo "=== resource usage"
[[ -d "$SUITE/src/all" ]] || make_src all 03 04
python3 "$HERE/resource_usage.py" --sm "$SM" --src "$SUITE/src/all" > "$SUITE/resources_all.md" 2>&1
python3 "$HERE/resource_usage.py" --sm "$SM" --src "$SUITE/src/all" --defines "-DASR_MQ_QPW=8" \
  > "$SUITE/resources_all_qpw8.md" 2>&1

# ---- tuning sweeps, kernel tests only ------------------------------------------------
if want sweeps; then
  make_src sweep_src_02 02 && collect sweep_gemm_min_blocks --src "$SUITE/src/sweep_src_02" \
    --sweep ASR_GEMM_MIN_BLOCKS_PER_SM=1,2,4
  make_src sweep_src_04 04 && {
    collect sweep_mq_qpw --src "$SUITE/src/sweep_src_04" --sweep ASR_MQ_QPW=2,4,8
    collect sweep_mq_warps --src "$SUITE/src/sweep_src_04" --sweep ASR_MQ_WARPS=2,4,8; }
fi
if [[ $GUIDE -eq 1 ]]; then  # the task guides' "Try this" experiments, original kernels
  make_src sweep_src_00 00 && {
    G=(--src "$SUITE/src/sweep_src_00")
    collect guide_rope_threads "${G[@]}" --sweep ASR_ROPE_THREADS=32,64,128,256,512,1024
    collect guide_reduce_threads "${G[@]}" --sweep ASR_REDUCE_THREADS=32,64,128,256,512,1024
    collect guide_tile "${G[@]}" --sweep ASR_TILE=8,16,32
    collect guide_attn_warps "${G[@]}" --sweep ASR_ATTN_WARPS=1,2,4,8,16; }
fi

nvidia-smi -L > "$SUITE/gpu_end.txt"
cmp -s "$SUITE/gpu_start.txt" "$SUITE/gpu_end.txt" || echo "WARNING: GPU list changed during the suite."

# ---- one comparison table (item 10 input) -------------------------------------------
python3 "$HERE/compare_runs.py" "$SUITE" > "$SUITE/suite_summary.md"
echo; cat "$SUITE/suite_summary.md"
echo; echo "Suite written to $SUITE"
