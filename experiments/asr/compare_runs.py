"""Compare the run directories of one run_report_suite.sh suite in Markdown tables.

[AI-assisted tooling: Claude (Anthropic), 17 Nov 2026]. Read-only: it parses the
logs written by collect_asr_stats.sh, using the parsers in summarise_asr_stats.py.

    python3 experiments/asr/compare_runs.py experiments/asr/runs/suite_<date>

Speed-ups are ratios of reported means from ONE suite (one allocation). Kernel-test
cases below ~0.05 ms are dominated by launch overhead; their ratios are noise. The ASR
"+/-" is benchmark.py's np.std over the timed runs (population standard deviation).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_asr_stats import parse_profile, parse_tests, read  # noqa: E402

KERNEL_RUNS = ["baseline", "gemm_smallm", "gemm_regtile", "gemm_smallm_regtile",
               "attention_mq", "all",   # run_report_suite.sh variants, in this order
               "original", "current", "bigtile"]  # CUDA-test builds (IMPL=0/1/2), 19 Nov
BASE_NAMES = ("baseline", "original")  # the reference column for speed-ups
CONFIG_NAMES = {"pytorch": "none", "cuda": "all", "mixed": "task1,task2"}  # file -> --cuda


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def ratio(a, b):
    a, b = fnum(a), fnum(b)
    return f"{a / b:.2f}x" if a and b else "-"


def tests_of(run):
    cases, _ = parse_tests(read(run / "tests.log") or "")
    return {name: (status, ms) for name, status, err, ms, speed in cases}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    suite = Path(sys.argv[1])
    runs = {p.name: p for p in sorted(suite.iterdir()) if p.is_dir()}
    out = [f"# ASR suite summary: `{suite.name}`", ""]

    # ---- provenance: one GPU, one revision ----------------------------------------
    gpus = {name: (read(p / "gpu.txt") or "").strip() for name, p in runs.items()}
    gpus["(suite start)"] = (read(suite / "gpu_start.txt") or "").strip()
    gpus["(suite end)"] = (read(suite / "gpu_end.txt") or "").strip()
    distinct = {g for g in gpus.values() if g}
    out.append("GPU: " + (" | ".join(sorted(distinct)) or "unknown"))
    out.append("**WARNING: steps ran on different GPUs; do not compare their times.**"
               if len(distinct) > 1 else "All steps report the same GPU identity.")
    revs = {(read(p / "revision.txt") or "").strip() for p in runs.values()}
    out.append(f"Source revision(s): {', '.join(sorted(r for r in revs if r)) or 'unknown'}")
    dirty = [n for n, p in runs.items()
             if re.search(r"^diff --git", read(p / "source_changes.patch") or "", re.M)]
    if dirty:
        out.append(f"Uncommitted csrc changes were present in: {', '.join(dirty)} "
                   "(see their source_changes.patch).")
    out.append("")

    # ---- kernel tests: each variant vs the frozen baseline ----------------------------
    present = [r for r in KERNEL_RUNS if r in runs and (runs[r] / "tests.log").exists()]
    if present:
        tables = {r: tests_of(runs[r]) for r in present}
        names = list(dict.fromkeys(n for r in present for n in tables[r]))
        out += ["## Kernel tests (ms per call; speed-up vs baseline in brackets)", "",
                "| case | " + " | ".join(present) + " |",
                "|---|" + "---:|" * len(present)]
        for n in names:
            cells = []
            ref = next((tables[r] for r in BASE_NAMES if r in tables), {})
            base = ref.get(n, (None, None))[1]
            for r in present:
                status, ms = tables[r].get(n, ("-", None))
                rel = f" ({ratio(base, ms)})" if r not in BASE_NAMES and base and ms else ""
                cells.append(f"{ms or '-'}{rel}{'' if status in ('PASS', '-') else ' **FAIL**'}")
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        for r in present:
            st = [s for s, _ in tables[r].values()]
            out.append(f"\n{r}: {st.count('PASS')}/{len(st)} cases PASS")
        out.append("")

    # ---- full ASR -------------------------------------------------------------------
    rows = []
    for r in KERNEL_RUNS:
        if r not in runs:
            continue
        for p in sorted(runs[r].glob("*_profile.log")):
            d = parse_profile(read(p) or "")
            cfg = p.stem.replace("_profile", "")
            rows.append((r, CONFIG_NAMES.get(cfg, cfg[5:].replace("_", ",") if cfg.startswith("cuda_") else cfg), d))
    if rows:
        ref = next((d.get("mean") for r, c, d in rows if c == "none"), None)
        out += ["## Full ASR (one transcription of test_audio.wav)", "",
                "| kernels | --cuda | mean ms | +/- | ms/token | linear ms | attention ms "
                "| peak mem | vs PyTorch | status |",
                "|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
        for r, c, d in rows:
            lin = d["ops"].get("linear", ("", "-", ""))[1]
            att = d["ops"].get("attention", ("", "-", ""))[1]
            speed = ratio(ref, d.get("mean")) if not c == "none" else "1.00x"
            out.append(f"| {r} | {c} | {d.get('mean', '-')} | {d.get('pm', '-')} | "
                       f"{d.get('per_token', '-')} | {lin} | {att} | {d.get('mem', '-')} | "
                       f"{speed} | {d.get('status', 'MISSING')} |")
        out.append("\n'vs PyTorch' > 1 means faster than PyTorch. linear/attention are from the "
                   "separate profiling transcription, not the timed runs.")
        ok = [(fnum(d.get("mean")), r, c) for r, c, d in rows
              if d.get("status", "").startswith("PASS") and fnum(d.get("mean"))]
        custom = [x for x in ok if not x[2] == "none"]
        if custom:
            best = min(custom)
            out.append(f"\nFastest correct configuration using custom kernels: **{best[1]}, "
                       f"--cuda {best[2]}** ({best[0]:.1f} ms). Differences within a few +/- "
                       "are not meaningful from one clip.")
        out.append("")

    # ---- sweeps -----------------------------------------------------------------------
    for name, p in runs.items():
        logs = sorted(p.glob("tests_*.log"))
        if not logs:
            continue
        values = [l.stem[len("tests_"):] for l in logs]
        parsed = {v: {n: (s, ms) for n, s, e, ms, sp in parse_tests(read(l) or "")[0]}
                  for v, l in zip(values, logs)}
        cases = list(dict.fromkeys(n for v in values for n in parsed[v]))
        out += [f"## Sweep `{name}` (ms per call; * = fastest passing)", "",
                "| case | " + " | ".join(values) + " |", "|---|" + "---:|" * len(values)]
        for n in cases:
            times = {v: fnum(parsed[v].get(n, (None, None))[1]) for v in values}
            good = {v: t for v, t in times.items() if t and parsed[v][n][0] == "PASS"}
            best = min(good, key=good.get) if good else None
            cells = []
            for v in values:
                s, ms = parsed[v].get(n, ("-", None))
                cells.append(f"{ms or '-'}{'*' if v == best else ''}{'' if s in ('PASS', '-') else ' FAIL'}")
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
