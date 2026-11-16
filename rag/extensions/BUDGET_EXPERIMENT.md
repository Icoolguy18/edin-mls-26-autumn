# Context-budget experiment

Question: Can relevance-ranked whole-document evidence selection under a fixed character budget improve the quality/cost trade-off relative to fixed top-k?

Only new extension and test files are used; supplied baseline files stay unchanged. Input is an existing generation JSONL containing ranked retrieved IDs and transcripts. Use the top-10 practice candidate file to avoid fresh ASR, encoding, or retrieval variation.

Selection visits candidates in rank order, includes a whole document if it fits the remaining evidence-character budget, and skips it otherwise. Later shorter documents can still fit. No truncation, reranking, or qrels access occurs in selection. Empty evidence is allowed and uses the baseline insufficient-evidence instruction. The baseline prompt is reused unchanged. Budget counts document text characters only, excluding labels/separators/instructions/question; prompt characters are separately recorded. Neither measure is token usage.

Run from repository root:

```bash
python -m unittest discover -s rag/tests -p test_budget_context.py -v
python rag/extensions/run_budget_context.py --candidates rag/results/variant_k10_30_answers.jsonl --budget-chars 5000 --dry-run --output rag/results/practice_budget5000_selection.jsonl
```

5000 characters is an initial candidate budget motivated by the observed top-5 mean of 4713 characters, not a selected optimum. Inspect document counts, empty selections and evidence sizes on practice before freezing it. Dry-run files deliberately omit answers and cannot be judged as answer candidates.

After inspection, generate using the same command without --dry-run and a new output filename. Uses the supplied ELMGenerator and current rag/.env. No GPU is needed. Keep generator model, endpoint and temperature unchanged from baseline; record any service changes. The runner makes one request per input row and flushes each result. It refuses existing outputs and does not resume partial generation.

Judge answer outputs using the supplied evaluate_answers.py with matching split and --allow-subset. Retain any documented judge-response formatting repairs. Do not use gold relevance judgements to select documents.

Timing includes selection and remote generation only. replay_total_ms excludes ASR, encoding, retrieval and model initialization; never compare it directly to baseline end_to_end_ms. Compare generation_ms with baseline, noting run-to-run API variability. This runner does not record billed tokens. Input candidate timings are not reused as new measurements.

Before a budgeted dev run, freeze all practice decisions. The first 30 dev top-k scores have already been observed; report that exposure and prefer remaining dev queries for a fresh confirmation if feasible. Report subset selection, actual coverage, per-query paired differences, latency distributions, and failures. A budgeted score improvement is a hypothesis, not an established result.
