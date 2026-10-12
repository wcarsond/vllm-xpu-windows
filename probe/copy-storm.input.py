# copy-storm.input.py - characterise the concurrent host-to-device fault: how big, how fast, and which cards?
# Usage:  .venv312\Scripts\python.exe probe\copy-storm.input.py            (writes probe\copy-storm.output.txt)
# Why (2026-10-11 15:30): mmap-readers.input.py showed that two processes each copying a large sustained volume of host
# memory to their OWN card kill one of themselves with 0xC0000005, while one process doing identical work is fine and two
# processes reading the same bytes without any device copy are fine. That is enough to know the workaround (serialize
# weight loading) but NOT enough to file a bug or to claim inference is safe. Three questions this answers, with no model
# and no vLLM - just torch tensors:
#   1. THRESHOLD: at what per-copy size does it start? vLLM's all_reduce moves 7-14 KB per call and its big path tops out
#      at a 32 MB slot; weight loading moves ~19 GB. If small copies are safe at any rate, inference is safe by argument
#      rather than by "it has not crashed yet".
#   2. SAME CARD: do two processes both targeting card 0 fault, or does it need one card each?
#   3. VOLUME vs RATE: does a given size fail on total bytes moved, or only above some rate?
# Each case runs in its own pair of child processes so a fault kills only that case. A case is OK only if every child
# exits 0; 0xC0000005 is reported as such.
import os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("copy-storm.output.txt")
TOTAL_GB = float(os.environ.get("STORM_TOTAL_GB", "8"))     # bytes each child pushes per case


def child(size_mb: float, device: int, total_gb: float) -> None:
    import torch
    n = int(size_mb * (1 << 20)) // 2                        # float16 elements
    # STORM_PINNED=1 allocates through the driver (pinned) instead of ordinary pageable
    # memory. A pageable copy has to be staged by the driver through an internal buffer;
    # a pinned one does not. If pageable faults and pinned does not, the bug is in that
    # staging path and the fix is to give each process its own pinned staging rather than
    # to serialize - which is also why the exchange (host USM) has never faulted.
    if os.environ.get("STORM_PINNED"):
        src = torch.empty(n, dtype=torch.float16, pin_memory=True)
    else:
        src = torch.empty(n, dtype=torch.float16)            # ordinary pageable host memory
    src.fill_(1.0)
    # STORM_FILL_GB: hold this much DEVICE memory before the copies start. vLLM's 72B
    # workers each hold 19.39 GiB of a 23.33 GiB card while loading, where every model
    # that loads fine concurrently tops out at 14.5 GiB. This tests "nearly full card +
    # concurrent copies" without downloading another 40 GB checkpoint.
    ballast = None
    fill_gb = float(os.environ.get("STORM_FILL_GB", "0"))
    if fill_gb:
        ballast = torch.empty(int(fill_gb * (1 << 30)) // 2, dtype=torch.float16,
                              device=f"xpu:{device}")
        torch.xpu.synchronize()
        print(f"    [{os.environ['STORM_TAG']}] holding {fill_gb} GiB on xpu:{device}", flush=True)
    iters = max(1, int(total_gb * (1 << 30) / (n * 2)))
    t0 = time.time()
    # STORM_GAP_MS: pause between copies. Size turned out not to matter (15 KB faults as
    # readily as 1 GB) and the same card is always fine, so the remaining difference
    # between this storm and real inference - which runs two workers on two cards for
    # hours without faulting - is how densely the copies are issued.
    gap = float(os.environ.get("STORM_GAP_MS", "0")) / 1000.0
    # STORM_SLICE=half|stride: copy a SLICE of the host buffer rather than the whole thing.
    # This is what tensor parallel does and pipeline parallel does not: both workers read
    # the same weight and each copies its own half. The 72B loads fine under a pipeline
    # split at the identical 19.5 GB per card and faults under a tensor split, so the
    # slicing is the remaining difference (2026-10-11 16:05).
    slice_mode = os.environ.get("STORM_SLICE")
    half = src.numel() // 2
    rank_hi = os.environ.get("STORM_TAG", "p0").endswith("1")
    # STORM_RETAIN=1 KEEPS every copy on the card instead of freeing it, which is what
    # weight loading does and what every earlier version of this probe did not. The
    # difference matters: thousands of retained allocations grow the driver's per-process
    # allocation table, where a transient one reuses the same block over and over.
    retain = bool(os.environ.get("STORM_RETAIN"))
    kept = []
    for i in range(iters):
        # STORM_FRESH_SRC=1 allocates a NEW host buffer every iteration instead of reusing
        # one. The model-shaped probe clones each weight (fresh host allocation, varied
        # sizes) and faults; this storm reused a single buffer and never did. 16:45.
        if os.environ.get("STORM_FRESH_SRC"):
            src = torch.empty(n, dtype=torch.float16)
            src.fill_(1.0)
        if slice_mode == "half":        # contiguous half, as a row-split weight gives
            piece = src[half:] if rank_hi else src[:half]
        elif slice_mode == "stride":    # non-contiguous, as a column split gives
            piece = src[1::2] if rank_hi else src[0::2]
        else:
            piece = src
        g = piece.to(f"xpu:{device}", non_blocking=False)
        if retain:
            kept.append(g)
        else:
            del g
        if gap:
            time.sleep(gap)
    if retain:
        print(f"    [{os.environ['STORM_TAG']}] retained {len(kept)} allocations", flush=True)
        kept.clear()
    torch.xpu.synchronize()
    dt = time.time() - t0
    print(f"    [{os.environ['STORM_TAG']}] {iters} x {size_mb} MB to xpu:{device} "
          f"= {iters*n*2/2**30:.1f} GiB in {dt:.1f} s ({iters*n*2/2**30/dt:.2f} GiB/s)", flush=True)


def run_case(label: str, size_mb: float, devices: list[int], total_gb: float) -> str:
    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"
    procs = []
    for i, d in enumerate(devices):
        e = dict(env)
        e["STORM_CHILD"] = "1"; e["STORM_TAG"] = f"p{i}"
        e["ZE_AFFINITY_MASK"] = str(d)
        # the child sees exactly one card under the mask, so it is always xpu:0 inside
        procs.append(subprocess.Popen([sys.executable, str(HERE), str(size_mb), "0", str(total_gb)],
                                      env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    codes, out = [], []
    for pr in procs:
        o, _ = pr.communicate()
        codes.append(pr.returncode)
        out += [l for l in o.splitlines() if l.strip()][-3:]
    bad = [c for c in codes if c != 0]
    if not bad:
        verdict = "OK"
    else:
        c = bad[0] & 0xFFFFFFFF
        verdict = f"FAULT 0x{c:08X}" + (" (ACCESS VIOLATION)" if c == 0xC0000005 else "")
    line = f"{label:<52} {verdict}"
    print(line, flush=True)
    for l in out:
        if "GiB in" in l:
            print(l, flush=True)
    return line


if __name__ == "__main__":
    if os.environ.get("STORM_CHILD"):
        child(float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
        sys.exit(0)

    lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  each child pushes {TOTAL_GB} GiB per case, "
             f"pageable host memory, blocking copies"]
    print(lines[0], flush=True)

    # 1. threshold sweep, one card each (the configuration that faults)
    lines.append("")
    lines.append("-- two processes, one card each (the failing shape), by per-copy size:")
    print(lines[-1], flush=True)
    for mb in (0.015, 0.25, 4, 32, 256, 1024):
        lines.append(run_case(f"   {mb} MB per copy, xpu:0 + xpu:1", mb, [0, 1], TOTAL_GB))

    # 2. same card, to separate "two processes" from "two cards"
    lines.append("")
    lines.append("-- two processes, BOTH on card 0:")
    print(lines[-1], flush=True)
    for mb in (32, 1024):
        lines.append(run_case(f"   {mb} MB per copy, xpu:0 + xpu:0", mb, [0, 0], TOTAL_GB))

    # 3. one process control at the largest size
    lines.append("")
    lines.append("-- one process (control):")
    print(lines[-1], flush=True)
    lines.append(run_case("   1024 MB per copy, xpu:0 alone", 1024, [0], TOTAL_GB))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {OUT}", flush=True)
