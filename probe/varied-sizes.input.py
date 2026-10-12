# varied-sizes.input.py - is it the SIZE DISTRIBUTION of the retained device allocations?
# Usage:  .venv312\Scripts\python.exe probe\varied-sizes.input.py [uniform|varied|varied-big] [total_gb]
# Why (2026-10-11 16:55): the shard reader faults (two processes, one card each, copy into preallocated device tensors,
# retained) but the uniform storm never does - not at 12/16/19/21/23/26/30 GiB retained, not with a fresh host buffer
# per copy, not with a single 2400 MiB tensor (the 72B's embedding is 2376 MiB, just over the 2 GiB signed boundary, so
# that was worth ruling out on its own). The one difference left is that the model's tensors are not all the same size.
# Measured from the 72B's first two shards, 308 tensors: min 0.002 MB, median 0.91 MB, max 2376 MB, and the spread is
#   82 under 0.1 MB | 83 at 0.1-1 MB | 83 at 1-10 MB | 24 at 10-100 MB | 36 over 100 MB
# so this cycles allocations through that distribution instead of one size, retains them all, and runs two processes
# with one card each and no affinity mask - everything else identical to the storm that passes.
import os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("varied-sizes.output.txt")

# MB sizes in the measured proportions, cycled in checkpoint order (small norms between big weights)
VARIED = [0.002, 0.004, 0.06, 0.25, 0.9, 3.0, 8.0, 0.06, 0.9, 32.0, 121.0, 0.002, 0.25, 243.0]
VARIED_BIG = VARIED + [2376.0]          # include the embedding-sized one
UNIFORM = [16.0]


def child(mode: str, device: int, total_gb: float) -> None:
    import torch
    sizes = {"uniform": UNIFORM, "varied": VARIED, "varied-big": VARIED_BIG}[mode]
    kept, moved, i = [], 0.0, 0
    t0 = time.time()
    while moved < total_gb * 1024:
        mb = sizes[i % len(sizes)]
        i += 1
        n = max(1, int(mb * (1 << 20)) // 2)
        src = torch.empty(n, dtype=torch.float16)
        src.fill_(1.0)
        dst = torch.empty(n, dtype=torch.float16, device=f"xpu:{device}")
        dst.copy_(src)                      # copy INTO a preallocated device tensor
        kept.append(dst)                    # and keep it, as a loader does
        moved += mb
    torch.xpu.synchronize()
    print(f"    [{os.environ['VARIED_TAG']}] {mode}: {i} allocations, {moved/1024:.1f} GiB retained, "
          f"{time.time()-t0:.1f} s", flush=True)


def run(mode: str, total_gb: float) -> str:
    procs = []
    for i, d in enumerate((0, 1)):
        e = dict(os.environ)
        e["VARIED_CHILD"] = "1"; e["VARIED_TAG"] = f"p{i}"; e["PYTHONFAULTHANDLER"] = "1"
        e.pop("ZE_AFFINITY_MASK", None)
        procs.append(subprocess.Popen([sys.executable, str(HERE), mode, str(d), str(total_gb)],
                                      env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    codes, notes = [], []
    for p in procs:
        out, _ = p.communicate(); codes.append(p.returncode)
        notes += [l for l in out.splitlines() if "retained" in l]
    bad = [c for c in codes if c != 0]
    if bad:
        c = bad[0] & 0xFFFFFFFF
        v = f"FAULT 0x{c:08X}" + (" (ACCESS VIOLATION)" if c == 0xC0000005 else "")
    else:
        v = "OK"
    line = f"{mode:<11} {total_gb:>4} GiB retained each   {v}"
    print(line, flush=True)
    for n in notes:
        print(n, flush=True)
    return line


if __name__ == "__main__":
    if os.environ.get("VARIED_CHILD"):
        child(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
        sys.exit(0)
    modes = [sys.argv[1]] if len(sys.argv) > 1 else ["uniform", "varied", "varied-big"]
    gb = float(sys.argv[2]) if len(sys.argv) > 2 else 19.0
    lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  two processes, one card each, no affinity mask, "
             f"copy into preallocated device tensors and retain"]
    print(lines[0], flush=True)
    for m in modes:
        lines.append(run(m, gb))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
