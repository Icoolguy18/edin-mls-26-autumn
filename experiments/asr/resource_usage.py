"""Registers, shared memory and theoretical occupancy of every ASR kernel.

[AI-assisted tooling: Claude (Anthropic), 17 Nov 2026]. Not part of the hand-in code;
it only compiles hw1-asr/csrc with `nvcc -Xptxas -v` and reads what ptxas reports.

    python3 experiments/asr/resource_usage.py                  # SM of this GPU (nvidia-smi)
    python3 experiments/asr/resource_usage.py --sm 90 --sm 120 # no GPU needed
    python3 experiments/asr/resource_usage.py --sm 90 --src <patched csrc copy> --defines "-DASR_MQ_QPW=8"

Theoretical occupancy = resident warps per SM / maximum warps per SM, limited by
registers, shared memory, threads and blocks (the CUDA occupancy-calculator rules:
registers are allocated per warp in units of 256, each block reserves 1 KB of
shared memory on sm_80+). It is an upper bound, not a measurement: it says how many
blocks CAN be resident, not whether the grid has enough blocks to fill the GPU
(see `waves` below) or how fast the kernel runs.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "hw1-asr" / "csrc"
FILES = ["task1_rope.cu", "task2_reduce.cu", "task3_gemm.cu", "task4_attention.cu"]

# Per-SM limits from the CUDA C++ Programming Guide (compute capabilities table).
LIMITS = {  # regs, max threads, max blocks, smem bytes per SM, smem reserved per block
    75: (65536, 1024, 16, 64 * 1024, 0),
    80: (65536, 2048, 32, 164 * 1024, 1024),
    86: (65536, 1536, 16, 100 * 1024, 1024),
    89: (65536, 1536, 24, 100 * 1024, 1024),
    90: (65536, 2048, 32, 228 * 1024, 1024),
    120: (65536, 1536, 32, 100 * 1024, 1024),
}


def defines_dict(defines):
    out = {}
    for tok in defines.split():
        m = re.match(r"-D(\w+)(?:=(\S+))?", tok)
        if m:
            out[m.group(1)] = m.group(2) or "1"
    return out


def block_threads(kernel, d):
    """Threads per block, as chosen by the launchers (keep in sync with csrc)."""
    g = lambda k, v: int(d.get(k, v))
    if kernel.startswith("rope"):
        return g("ASR_ROPE_THREADS", 256)
    if kernel.startswith(("layernorm", "rmsnorm", "softmax")):
        return g("ASR_REDUCE_THREADS", 256)
    if kernel.startswith("linear_tiled"):
        return g("ASR_TILE", 16) ** 2
    if kernel.startswith(("linear_regtile", "linear_bigtile")):
        return 256
    if kernel.startswith("linear_smallm"):
        return 128
    if kernel.startswith("attention_mq"):
        return g("ASR_MQ_WARPS", 4) * 32
    if kernel.startswith("attention_kernel"):
        return g("ASR_ATTN_WARPS", 4) * 32
    return None


def dynamic_smem(kernel, d, head_dim):
    """Dynamic shared memory requested by launch_attention (bytes)."""
    if kernel.startswith("attention_mq"):
        q = int(d.get("ASR_MQ_WARPS", 4)) * int(d.get("ASR_MQ_QPW", 4))
        return 4 * (32 * (head_dim + 4) + 32 * head_dim + q * head_dim)
    if kernel.startswith("attention_kernel"):
        return 4 * (32 * (head_dim + 1) + 32 * head_dim + int(d.get("ASR_ATTN_WARPS", 4)) * head_dim)
    return 0


def occupancy(sm, regs, threads, smem):
    max_regs, max_threads, max_blocks, smem_sm, reserve = LIMITS[sm]
    warps = -(-threads // 32)
    regs_per_warp = -(-(max(regs, 1) * 32) // 256) * 256
    by_regs = max_regs // (regs_per_warp * warps)
    by_threads = max_threads // (warps * 32)
    by_smem = smem_sm // (smem + reserve) if smem + reserve > 0 else max_blocks
    blocks = min(by_regs, by_threads, by_smem, max_blocks)
    limiter = " = ".join(n for v, n in ((by_regs, "registers"), (by_threads, "threads"),
                                         (by_smem, "smem"), (max_blocks, "block limit"))
                         if v == blocks)
    return blocks, blocks * warps / (max_threads // 32), limiter


def ptxas(sm, defines, nvcc, src):
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in FILES:
            cmd = [nvcc, "-O3", "-std=c++17", f"-arch=sm_{sm}", "-I", str(src), "-Xptxas", "-v",
                   *defines.split(), "-c", str(src / f), "-o", str(Path(tmp) / "k.o")]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                sys.exit(f"nvcc failed on {f}:\n{res.stderr}")
            name = None
            for line in res.stderr.splitlines():
                m = re.search(r"Compiling entry function '(\w+)'", line)
                if m:
                    name, spill = m.group(1), 0
                m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
                if m and name:
                    spill = int(m.group(1)) + int(m.group(2))
                m = re.search(r"Used (\d+) registers(?:.*?(\d+) bytes smem)?", line)
                if m and name:
                    rows.append((f, name, int(m.group(1)), int(m.group(2) or 0), spill))
                    name = None
    return rows


def demangle(names):
    if not shutil.which("c++filt"):
        return {n: n for n in names}
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout
    return dict(zip(names, (s.split("(")[0] for s in out.splitlines())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sm", type=int, action="append", help="compute capability, e.g. 90 (repeatable)")
    ap.add_argument("--defines", default="", help='extra -D flags, e.g. "-DASR_GEMM_MAX_TM=8"')
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--src", type=Path, default=SRC, help="kernel source directory (default hw1-asr/csrc)")
    ap.add_argument("--head-dim", type=int, action="append",
                    help="attention head_dim for dynamic smem (default: 64 and 128)")
    args = ap.parse_args()
    sms = args.sm
    if not sms:
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.split()
        if not cap:
            sys.exit("No GPU found: pass --sm 90 (H200) or --sm 120 (RTX 50xx).")
        sms = [int(cap[0].replace(".", ""))]
    head_dims = args.head_dim or [64, 128]
    d = defines_dict(args.defines)

    print(f"Source: {args.src}\nDefines: {args.defines or '(none: source defaults)'}")
    for sm in sms:
        if sm not in LIMITS:
            sys.exit(f"No limits table for sm_{sm}; add it to LIMITS.")
        rows = ptxas(sm, args.defines, args.nvcc, args.src.resolve())
        names = demangle([r[1] for r in rows])
        print(f"\n## sm_{sm}\n")
        print("| kernel | threads/block | regs/thread | static smem B | dyn smem B | spill B "
              "| blocks/SM | occupancy | limited by |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for f, raw, regs, smem, spill in rows:
            kname = names[raw].replace("void ", "")
            threads = block_threads(kname, d)
            if threads is None:
                continue
            dyns = [dynamic_smem(kname, d, h) for h in head_dims] if kname.startswith("attention") else [0]
            for h, dyn in zip(head_dims if len(dyns) > 1 else [None], dyns):
                blocks, occ, lim = occupancy(sm, regs, threads, smem + dyn)
                label = kname + (f" (head_dim {h})" if h else "")
                print(f"| `{label}` | {threads} | {regs} | {smem} | {dyn} | {spill} | {blocks} "
                      f"| {occ:.0%} | {lim} |")
    print("\nWaves = grid blocks / (blocks per SM x SMs). A kernel with fewer blocks than one "
          "wave leaves SMs idle whatever its occupancy (e.g. decoder GEMMs at M = 64).")


if __name__ == "__main__":
    main()
