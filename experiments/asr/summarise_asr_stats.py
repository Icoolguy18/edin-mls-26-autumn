"""Summarise one ASR evidence directory written by collect_asr_stats.sh.

[AI-assisted tooling: Claude (Anthropic), 16 Nov 2026]

Prints (1) a short table of the key numbers and (2) every raw file, in the same
sectioned layout as the baseline evidence:

    ============================================================
    Configuration: <what it is>
    File: <path>
    ============================================================
    <contents, or FILE NOT FOUND>

Usage: python3 summarise_asr_stats.py <run directory>
It only reads files; it never runs anything.
"""
import re
import sys
from pathlib import Path

RULE = "=" * 60
CASE = re.compile(r"^(\S.*?)\s{2,}(PASS|FAIL)\s+max_err=(\S+)(?:\s+([\d.]+) ms\s+(.*?))?\s*$")
OP = re.compile(r"^\s+(\w+)\s+(cuda|torch)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%")


def read(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


def parse_tests(text):
    """Return (cases, verdicts): cases = [(name, status, max_err, ms, speed)]."""
    cases, verdicts = [], []
    for line in text.splitlines():
        m = CASE.match(line)
        if m:
            cases.append(m.groups())
        elif re.match(r"^Task \d .*: ", line):
            verdicts.append(line.strip())
    return cases, verdicts


def parse_profile(text):
    out = {"ops": {}}
    for line in text.splitlines():
        if line.startswith("GPU:"):
            out["gpu"] = line[4:].strip()
        elif line.startswith("Kernels:"):
            out["kernels"] = line[8:].strip()
        elif line.startswith("Time:"):
            m = re.search(r"([\d.]+) ms per transcription \(\+/- ([\d.]+) ms\), ([\d.]+) ms per token", line)
            if m:
                out["mean"], out["pm"], out["per_token"] = m.groups()
        elif line.startswith("Peak GPU memory:"):
            out["mem"] = line.split(":", 1)[1].strip()
        elif line.startswith("Status:"):
            out["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("Accuracy:"):
            out["accuracy"] = line.split(":", 1)[1].strip()
        else:
            m = OP.match(line)
            if m:
                out["ops"][m.group(1)] = (m.group(2), m.group(4), m.group(5))
    return out


def key_numbers(run):
    lines = ["KEY NUMBERS (parsed from the raw files below; check them against the raw text)", ""]
    info = read(run / "build_info.txt") or ""
    flags = re.search(r"NVCCFLAGS: (.*)", info)
    if flags:
        lines.append(f"Build flags: {flags.group(1).strip()}")
    gpu = (read(run / "gpu.txt") or "").strip().splitlines()
    if gpu:
        lines.append("GPU: " + " | ".join(l.strip() for l in gpu))
    lines.append("")

    tests = read(run / "tests.log")
    if tests:
        cases, verdicts = parse_tests(tests)
        n_pass = sum(c[1] == "PASS" for c in cases)
        lines.append(f"Kernel tests: {n_pass}/{len(cases)} cases PASS")
        lines += ["  " + v for v in verdicts]
        lines.append("")
        lines.append(f"  {'case':38s} {'ms':>9s}  speed")
        for name, status, err, ms, speed in cases:
            if ms:
                lines.append(f"  {name[:38]:38s} {ms:>9s}  {speed}   [{status}]")
        lines.append("")

    sweeps = sorted(run.glob("tests_*.log"))
    if sweeps:
        table, values = {}, []
        for p in sweeps:
            value = p.stem[len("tests_"):]
            values.append(value)
            cases, verdicts = parse_tests(read(p) or "")
            ok = all(c[1] == "PASS" for c in cases) and cases
            for name, status, err, ms, speed in cases:
                table.setdefault(name, {})[value] = (ms or "-") + ("" if status == "PASS" else " FAIL")
            table.setdefault("(all cases pass)", {})[value] = "yes" if ok else "NO"
        lines.append("Sweep: time per call in ms (rows = test case, columns = setting)")
        lines.append("  " + f"{'case':38s}" + "".join(f"{v[:18]:>20s}" for v in values))
        for name, row in table.items():
            lines.append("  " + f"{name[:38]:38s}" + "".join(f"{row.get(v, '-'):>20s}" for v in values))
        lines.append("")

    profiles = sorted(run.glob("*_profile.log"))
    if profiles:
        lines.append(f"  {'ASR run':18s} {'mean ms':>9s} {'+/-':>6s} {'ms/tok':>7s} "
                     f"{'linear ms':>10s} {'attn ms':>9s} {'peak mem':>10s}  status")
        for p in profiles:
            d = parse_profile(read(p) or "")
            lin = d["ops"].get("linear", ("", "-", ""))[1]
            att = d["ops"].get("attention", ("", "-", ""))[1]
            lines.append(f"  {p.stem.replace('_profile', ''):18s} {d.get('mean', '-'):>9s} "
                         f"{d.get('pm', '-'):>6s} {d.get('per_token', '-'):>7s} {lin:>10s} "
                         f"{att:>9s} {d.get('mem', '-'):>10s}  {d.get('status', 'MISSING')}")
        gpus = {parse_profile(read(p) or "").get("gpu") for p in profiles}
        if len(gpus) > 1:
            lines.append("  WARNING: these runs report different GPUs; do not compare their times.")
        lines.append("")
    return "\n".join(lines)


def section(run, name, what):
    path = run / name
    text = read(path)
    rel = f"experiments/asr/runs/{run.name}/{name}"
    body = text.rstrip("\n") if text is not None else "FILE NOT FOUND"
    return f"\n{RULE}\nConfiguration: {what}\nFile: {rel}\n{RULE}\n{body}\n"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    parts = [key_numbers(run)]
    parts.append(section(run, "gpu.txt", "Recorded GPU identity (nvidia-smi -L); verify against each profile"))
    parts.append(section(run, "build_info.txt", "Build flags, compiler and environment"))
    if (run / "tests.log").exists() or not list(run.glob("tests_*.log")):
        parts.append(section(run, "tests.log", "CUDA correctness: all four supplied kernel suites (make test)"))
    for p in sorted(run.glob("tests_*.log")):
        parts.append(section(run, p.name, f"Sweep build {p.stem[len('tests_'):]} (make test)"))
    named = {"pytorch_profile.log": "PyTorch ASR (--cuda none --profile)",
             "cuda_profile.log": "All-CUDA ASR (--cuda all --profile)",
             "mixed_profile.log": "Mixed ASR (--cuda task1,task2 --profile)"}
    for name, what in named.items():
        if (run / name).exists():
            parts.append(section(run, name, what))
    for p in sorted(run.glob("cuda_*_profile.log")):
        parts.append(section(run, p.name, f"ASR, CUDA groups {p.stem[5:-8]} (--profile)"))
    parts.append(section(run, "revision.txt", "Recorded source revision"))
    parts.append(section(run, "source_changes.patch", "Uncommitted source changes captured at measurement time"))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
