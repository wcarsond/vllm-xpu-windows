# copy-bw.input.py - how fast does one B60 copy between its memory and host memory on Windows? Same test as the Ubuntu side's
# ~/scripts/copy-bw/copy-bw.input.py so the numbers line up: plain torch, one card, a bf16 tensor on xpu and a CPU tensor,
# time copy_ both ways at 7 KB, 1 MB, 14.7 MB, 256 MB. Writes probe\copy-bw.output.txt.
# Usage:  .venv312\Scripts\python.exe probe\copy-bw.input.py [card]          (card 0 default)
# Why (2026-10-13): Linux measured 542 us for 14.7 MB card->host (28.4 GB/s) and 26-28 GB/s both ways from 1 MB up. The
# Windows exchange log (server\exchange-test.output.txt, 2026-10-11) shows ~3.8 ms for the same 14.7 MB card->pinned step,
# ~3.9 GB/s. If raw copies are that slow on Windows, the driver's copy path is the biggest single Windows prefill loss.
# Each size runs four ways: host memory pinned vs pageable, and the copy issued blocking vs non_blocking (then synced),
# because a "pinned" tensor that the driver does not really pin would show up as pinned == pageable.
import os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("copy-bw.output.txt")
card = int(sys.argv[1]) if len(sys.argv) > 1 else 0

import torch

dev = torch.device(f"xpu:{card}")
SIZES = [("7 KB", 7 * 1024), ("1 MB", 1 << 20), ("14.7 MB", 14_680_064), ("256 MB", 256 << 20)]
lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  torch {torch.__version__}  card {card}: {torch.xpu.get_device_name(card)}",
         "size     | host     | d2h blocking      | d2h non_blocking  | h2d blocking      | h2d non_blocking"]


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.xpu.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.xpu.synchronize(dev)
    return (time.perf_counter() - t0) / reps


def fmt(sec, nbytes):
    return f"{sec*1e6:9.1f} us {nbytes/sec/1e9:5.1f} GB/s"


for label, nbytes in SIZES:
    n = nbytes // 2
    reps = 2000 if nbytes < 1 << 20 else 200 if nbytes < 64 << 20 else 20
    g = torch.randn(n, dtype=torch.bfloat16, device=dev)
    for pinned in (True, False):
        h = torch.empty(n, dtype=torch.bfloat16, pin_memory=pinned)
        d2h_b = timed(lambda: h.copy_(g, non_blocking=False), reps)
        d2h_n = timed(lambda: h.copy_(g, non_blocking=True), reps)
        h2d_b = timed(lambda: g.copy_(h, non_blocking=False), reps)
        h2d_n = timed(lambda: g.copy_(h, non_blocking=True), reps)
        s = (f"{label:8} | {'pinned' if pinned else 'pageable':8} | {fmt(d2h_b, nbytes)} | {fmt(d2h_n, nbytes)} | "
             f"{fmt(h2d_b, nbytes)} | {fmt(h2d_n, nbytes)}")
        print(s, flush=True)
        lines.append(s)
        del h
    del g
    torch.xpu.empty_cache()
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
