# threads-hang.input.py - where do two threads, one card each, hang in eager torch on Windows XPU? One case per run:
#   E   elementwise only (norm, silu, add; no matmul)
#   M   matmul only
#   W   the full step, but each card runs it once from the MAIN thread first (warm: kernels compiled, oneDNN primitives
#       built, allocator pools made), then the two threads
#   S   the full step, two threads, started one after the other with a lock around each thread's FIRST step
#   F   the full step, cold, two threads (the case that hung in threads-gil eager B)
# If a case is not done in 60 s, every thread's Python stack is dumped to the .err log (faulthandler) and the run exits 9.
# Usage:  .venv312\Scripts\python.exe probe\threads-hang.input.py <E|M|W|S|F>   (threads-hang.input.cmd runs them all)
# Why (2026-10-13): a one-process executor needs one thread per card (each rank blocks its host on its own card, e.g.
# sampling's .tolist(), while that card waits for the other's half), and two threads hung in eager with no output.
import faulthandler, pathlib, sys, threading, time

import torch

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("threads-hang.output.txt")
LAYERS, STEPS, H, F = 28, 100, 3584, 2304
CASE = sys.argv[1]
faulthandler.dump_traceback_later(60, exit=True)


def make(dev):
    torch.xpu.set_device(dev)
    d = torch.device("xpu", dev)
    w1 = torch.randn(H, F, device=d, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn(F, H, device=d, dtype=torch.bfloat16) * 0.01
    x = torch.randn(1, H, device=d, dtype=torch.bfloat16)
    return d, w1, w2, x


def step(w1, w2, x):
    for _ in range(LAYERS):
        if CASE == "E":
            y = x * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
            y = torch.nn.functional.silu(y)
        elif CASE == "M":
            y = (x @ w1) @ w2
        else:
            y = x * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
            y = torch.nn.functional.silu(y @ w1) @ w2
        x = x + y
    return x


first_lock = threading.Lock()


def run(dev, res):
    d, w1, w2, x = make(dev)
    if CASE == "S":
        with first_lock:
            step(w1, w2, x)
            torch.xpu.synchronize(d)
    for _ in range(5):
        step(w1, w2, x)
    torch.xpu.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step(w1, w2, x)
    torch.xpu.synchronize(d)
    res[dev] = 1e6 * (time.perf_counter() - t0) / STEPS


if __name__ == "__main__":
    print(f"{CASE} starting", flush=True)
    if CASE == "W":
        for dev in (0, 1):
            d, w1, w2, x = make(dev)
            step(w1, w2, x)
            torch.xpu.synchronize(d)
        print("W: both cards warm from the main thread", flush=True)
    res = {}
    ts = [threading.Thread(target=run, args=(dev, res), name=f"card{dev}") for dev in (0, 1)]
    for t in ts: t.start()
    for t in ts: t.join()
    faulthandler.cancel_dump_traceback_later()
    line = f"{CASE}: completed, us per step card0 {res[0]:.0f}, card1 {res[1]:.0f}"
    print(line, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
