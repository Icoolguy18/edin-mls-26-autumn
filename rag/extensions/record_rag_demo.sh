#!/usr/bin/env bash
set -euo pipefail

cd "${RAG_REPO_ROOT:-$HOME/edin-mls-26-autumn}"
export RAG_ASR_PYTHON="${RAG_ASR_PYTHON:-$HOME/miniconda3/envs/mls-asr/bin/python}"
export ASR_CUDA=none
python="${RAG_DEMO_PYTHON:-$HOME/mls-rag-env/bin/python}"
[[ -x "$python" && -x "$RAG_ASR_PYTHON" ]] || {
    printf '%s\n' 'Set RAG_DEMO_PYTHON and RAG_ASR_PYTHON to the RAG and ASR interpreters.' >&2
    exit 1
}

mkdir -p rag/results
run_dir=$(mktemp -d rag/results/recording.XXXXXX)
query_args=()
if [[ -n "${1:-}" ]]; then
    query_args=(--query-id "$1")
fi

printf 'Practice audio query | PyTorch ASR on GPU | CPU query encoder | top-10\n'
"$python" rag/scripts/run_audio_query.py --split practice --device cpu --k 10 \
    "${query_args[@]}" > "$run_dir/top10.json" 2> "$run_dir/runtime.log" || {
    printf 'Audio run failed. Review %s/runtime.log off camera.\n' "$run_dir" >&2
    exit 1
}

"$python" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
row = json.loads((run_dir / "top10.json").read_text(encoding="utf-8"))
ids = row["retrieval"]["record_ids"]
candidate = {
    "query_id": row["query_id"],
    "split": "practice",
    "audio_file": row["audio_file"],
    "transcript": row["transcript"],
    "retrieved_record_ids": ids,
}
(run_dir / "candidates.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
print(f"Input: {Path(row['audio_file']).name} ({row['query_id']})")
print("Transcript:", row["transcript"])
print("Ranked document IDs:", ", ".join(ids))
print("Evidence characters:", sum(map(len, row["contexts"])))
print("\nTop-10 answer:\n" + row["answer"])
PY

printf '\nBudget-5000 | same transcript and ranked candidates\n'
"$python" rag/extensions/run_budget_context.py \
    --candidates "$run_dir/candidates.jsonl" --budget-chars 5000 \
    --output "$run_dir/budget.jsonl" 2>> "$run_dir/runtime.log" || {
    printf 'Budget replay failed. Review %s/runtime.log off camera.\n' "$run_dir" >&2
    exit 1
}

"$python" - "$run_dir/budget.jsonl" <<'PY'
import json
import sys
from pathlib import Path

row = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("Kept document IDs:", ", ".join(row["retrieved_record_ids"]))
print("Skipped document IDs:", ", ".join(row["skipped_record_ids"]) or "(none)")
print("\nBudget-5000 answer:\n" + row["answer"])
PY
