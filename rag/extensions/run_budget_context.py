"""Replay saved ranked candidates: no ASR, query encoding or GPU required."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.budget_context import BudgetContextBuilder
from src.data import CourseData, load_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--budget-chars", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Save selection diagnostics without ELM requests")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; choose a new filename")
    if args.budget_chars < 1:
        ap.error("--budget-chars must be positive")
    rows = load_jsonl(args.candidates)
    if not rows:
        ap.error("Candidate file is empty")
    data = CourseData(ROOT / "data/rag_corpus")
    seen = set()
    for row in rows:
        qid = row["query_id"]
        if qid in seen:
            ap.error(f"Duplicate query ID: {qid}")
        seen.add(qid)
        if row["split"] not in ("practice", "dev"):
            ap.error(f"Invalid split for {qid}")
        if not isinstance(row["transcript"], str) or not row["transcript"].strip():
            ap.error(f"Empty transcript for {qid}")
        ids = row["retrieved_record_ids"]
        if not ids or len(set(ids)) != len(ids) or any(r not in data.records for r in ids):
            ap.error(f"Invalid candidate record IDs for {qid}")
    builder = BudgetContextBuilder(args.budget_chars)
    generator = None
    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from src.generator import ELMGenerator
        generator = ELMGenerator()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        for row in rows:
            start = time.perf_counter()
            built = builder.build(row["transcript"], row["retrieved_record_ids"], data)
            selection_ms = (time.perf_counter() - start) * 1000
            result = {key: row[key] for key in ("query_id", "split", "transcript", "audio_file")}
            result.update(
                candidate_record_ids=row["retrieved_record_ids"],
                retrieved_record_ids=built["selected_record_ids"],
                skipped_record_ids=built["skipped_record_ids"],
                evidence_chars=built["evidence_chars"], prompt_chars=built["prompt_chars"],
                budget_chars=args.budget_chars, strategy="ranked_whole_document_budget",
                source_candidates=str(args.candidates), dry_run=args.dry_run,
                timing={"selection_ms": selection_ms},
            )
            if generator:
                generated = generator.generate(built["prompt"])
                result["answer"] = generated["answer"]
                result["generator_model"] = generator.model
                result["timing"]["generation_ms"] = generated["latency_ms"]
                result["timing"]["replay_total_ms"] = (time.perf_counter() - start) * 1000
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{row['query_id']}: {len(built['selected_record_ids'])} documents, "
                  f"{built['evidence_chars']} evidence characters", flush=True)


if __name__ == "__main__":
    main()
