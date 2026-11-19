#!/usr/bin/env bash
# Collect ASR evidence on a cluster GPU node and write one summary in the same layout
# as experiments/asr/baseline (gpu.txt, tests.log, *_profile.log, revision.txt, ...).
#
# [AI-assisted tooling: Claude (Anthropic), 16 Nov 2026]. Not part of the hand-in
# code; it only calls the supplied Makefile and benchmark.py, never changes them.
#
# Run from a GPU shell with the ASR environment active:
#   srun -p Teaching --gres=gpu:h200_1g.18gb:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
#   export PATH=/opt/cuda-12.8.0/bin:$PATH
#   source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate mls-asr
#   bash experiments/asr/collect_asr_stats.sh --label optimised
#
# Options:
#   --label NAME        name of this run (default: default). Output goes to
#                       experiments/asr/runs/<date>_<label>_<gpu>/
#   --src DIR           build the kernels from DIR instead of hw1-asr/csrc (a patched
#                       copy, e.g. one cherry-picked variant; added 17 Nov). Builds into
#                       a separate build directory; source_changes.patch then records
#                       diff -ru hw1-asr/csrc DIR.
#   --defines "..."     extra compile defines for this build, e.g.
#                       "-DASR_GEMM_IMPL=0" (original GEMM = baseline) or "-DASR_TILE=32".
#                       Builds into a separate build directory, so your normal
#                       build/sm_XX is not touched.
#   --configs "a;b;c"   ASR configurations to profile, separated by ';'
#                       (default: "none;all;task1,task2" = PyTorch, all CUDA, mixed)
#   --sweep NAME=v1,v2  instead of the normal run, rebuild once per value with
#                       -DNAME=value and run `make test` for each (kernel tests only).
#                       Example: --sweep ASR_ATTN_WARPS=1,2,4,8,16
#   --skip-tests        do not run `make test`
#   --skip-asr          do not run the full model
#   --runs N            timed transcriptions per configuration (benchmark default: 3)
#   --no-redact         keep $HOME and $USER in the logs (default: replace them, because
#                       the report, appendices, video and code must be anonymous)
set -uo pipefail

USER="${USER:-$(id -un)}"   # some batch shells do not set $USER; redaction needs it
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ASR="$REPO/hw1-asr"

SRC_DIR=""; LABEL="default"; DEFINES=""; CONFIGS="none;all;task1,task2"; SWEEP=""
SKIP_TESTS=0; SKIP_ASR=0; RUNS=""; REDACT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --defines) DEFINES="$2"; shift 2 ;;
    --src) SRC_DIR="$(cd "$2" && pwd)"; shift 2 ;;
    --configs) CONFIGS="$2"; shift 2 ;;
    --sweep) SWEEP="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-asr) SKIP_ASR=1; shift ;;
    --runs) RUNS="$2"; shift 2 ;;
    --no-redact) REDACT=0; shift ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)"; exit 2 ;;
  esac
done

# ---- checks -----------------------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "No nvidia-smi: run this on a GPU node (srun ...)."; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found: export PATH=/opt/cuda-12.8.0/bin:\$PATH"; exit 1; }
if [[ $SKIP_ASR -eq 0 && -z "$SWEEP" ]]; then
  python -c "import torch" 2>/dev/null || { echo "torch not importable: conda activate mls-asr"; exit 1; }
fi
# A full node-local /tmp broke nvcc before ("Failed to preprocess host compiler properties").
if [[ -z "${TMPDIR:-}" ]] || ! mktemp -p "${TMPDIR:-/tmp}" >/dev/null 2>&1; then
  export TMPDIR="$HOME/mls-tmp"; mkdir -p "$TMPDIR"; export TMP="$TMPDIR" TEMP="$TMPDIR"
fi

SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
GPU_SLUG="$(nvidia-smi -L | grep -o 'MIG [^ ]*' | head -n1 | tr ' ' '-')"
[[ -z "$GPU_SLUG" ]] && GPU_SLUG="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr ' ' '-')"
STAMP="$(date +%Y%m%d-%H%M)"
OUT="$REPO/experiments/asr/runs/${STAMP}_${LABEL}_${GPU_SLUG}"
mkdir -p "$OUT"
BASE_FLAGS="-O3 -std=c++17 -lineinfo -diag-suppress 177"   # the Makefile's defaults
if [[ -n "$DEFINES" || -n "$SRC_DIR" ]]; then
  BUILD_DIR="build/${LABEL}_sm_${SM}"   # relative to hw1-asr, keeps build/sm_XX untouched
else
  BUILD_DIR="build/sm_${SM}"
fi
MAKE_SRC=""; [[ -n "$SRC_DIR" ]] && MAKE_SRC="SRC=$SRC_DIR"
# make cannot see a change of -D flags, so the first build of a directory is forced (-B).
LIB_FORCE=""; [[ $SKIP_TESTS -eq 1 ]] && LIB_FORCE="-B"
echo "Writing to $OUT"

redact() {  # anonymise a log in place
  [[ $REDACT -eq 1 && -f "$1" ]] || return 0
  sed -i -e "s#${HOME}#~#g" -e "s#/${USER}/#/<user>/#g" -e "s#\b${USER}\b#<user>#g" "$1"
}

# ---- configuration evidence -----------------------------------------------------------
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
(cd "$REPO" && git rev-parse HEAD) > "$OUT/revision.txt" 2>&1
{
  echo "# git status --short (empty = clean checkout)"
  (cd "$REPO" && git status --short -- hw1-asr)
  echo "# git diff HEAD -- hw1-asr (empty = measured code equals the commit above)"
  (cd "$REPO" && git diff HEAD -- hw1-asr)
  if [[ -n "$SRC_DIR" ]]; then
    echo "# kernels built from $SRC_DIR: diff -ru hw1-asr/csrc against it"
    (cd "$REPO" && diff -ru hw1-asr/csrc "$SRC_DIR")
  fi
} > "$OUT/source_changes.patch" 2>&1
{
  echo "date: $(date -Iseconds)"
  echo "label: $LABEL"
  echo "host: $(hostname)"
  echo "nvcc: $(nvcc --version | tail -n1)"
  echo "build dir: hw1-asr/$BUILD_DIR"
  echo "NVCCFLAGS: $BASE_FLAGS $DEFINES"
  echo "kernel source: ${SRC_DIR:-hw1-asr/csrc}"
  echo "ASR configs: $CONFIGS"
  python -c "import torch; print('torch:', torch.__version__)" 2>/dev/null
} > "$OUT/build_info.txt"

build_and_test() {  # $1 = build dir, $2 = extra defines, $3 = log file
  (cd "$ASR" && make -s -B BUILD="$1" $MAKE_SRC NVCCFLAGS="$BASE_FLAGS $2" all) > "$3" 2>&1 || {
    echo "BUILD FAILED" >> "$3"; return 1; }
  (cd "$ASR" && make -s BUILD="$1" $MAKE_SRC NVCCFLAGS="$BASE_FLAGS $2" test) >> "$3" 2>&1
}

# ---- sweep mode: kernel tests only, one build per value ----------------------------------
if [[ -n "$SWEEP" ]]; then
  NAME="${SWEEP%%=*}"; VALUES="${SWEEP#*=}"
  IFS=',' read -ra VALS <<< "$VALUES"
  for v in "${VALS[@]}"; do
    echo "  $NAME=$v ..."
    build_and_test "build/sweep_${NAME}_${v}_sm_${SM}" "$DEFINES -D${NAME}=${v}" "$OUT/tests_${NAME}_${v}.log"
    redact "$OUT/tests_${NAME}_${v}.log"
  done
else
  # ---- kernel tests ----------------------------------------------------------------------
  if [[ $SKIP_TESTS -eq 0 ]]; then
    echo "  make test ..."
    build_and_test "$BUILD_DIR" "$DEFINES" "$OUT/tests.log"
    redact "$OUT/tests.log"
  fi
  # ---- full ASR runs (same as ./run.sh, but with this build's library) --------------------
  if [[ $SKIP_ASR -eq 0 ]]; then
    (cd "$ASR" && make -s $LIB_FORCE BUILD="$BUILD_DIR" $MAKE_SRC NVCCFLAGS="$BASE_FLAGS $DEFINES" lib) \
      > "$OUT/build_lib.log" 2>&1 || echo "library build failed, see build_lib.log"
    IFS=';' read -ra CFGS <<< "$CONFIGS"
    for cfg in "${CFGS[@]}"; do
      case "$cfg" in
        none) name="pytorch" ;; all) name="cuda" ;; task1,task2) name="mixed" ;;
        *) name="cuda_$(echo "$cfg" | tr ',' '_')" ;;
      esac
      echo "  ASR --cuda $cfg ..."
      extra=(); [[ -n "$RUNS" ]] && extra=(--runs "$RUNS")
      (cd "$ASR" && ASR_LIB="$ASR/$BUILD_DIR/libasr_kernels.so" \
         python benchmark.py --cuda "$cfg" --profile "${extra[@]}") \
        > "$OUT/${name}_profile.log" 2>&1
      redact "$OUT/${name}_profile.log"
    done
  fi
fi
redact "$OUT/build_info.txt"; redact "$OUT/source_changes.patch"; redact "$OUT/build_lib.log"

# ---- summary in the baseline layout -------------------------------------------------------
python3 "$HERE/summarise_asr_stats.py" "$OUT" > "$OUT/summary.txt"
cat "$OUT/summary.txt"
echo
echo "Summary: $OUT/summary.txt"
