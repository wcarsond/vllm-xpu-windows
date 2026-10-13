# threads-gil.input.py - what does Python's global lock cost if one process drives both cards from two threads, as a
# single-process two-card executor would? A synthetic eager decode step (28 layers x a handful of small kernels, the shape of
# a 7B decode on half a card) timed three ways:
#   A  one thread, one card                      (the per-card launch cost alone)
#   B  two threads in ONE process, one card each (what a one-process executor would do)
#   C  two PROCESSES, one card each              (what vLLM does today)
# and the same with the step recorded as an XPU graph and replayed (one Python call per step instead of ~170).
# Usage:  .venv312\Scripts\python.exe probe\threads-gil.input.py          (writes probe\threads-gil.output.txt)
# Why (2026-10-13): a helper-free decode exchange needs both cards in one process (probe/one-process-pingpong: 14 us vs
# 28 us); host-memory IPC between processes is refused (probe/host-ipc). If two threads double the eager step time, the
# exchange saving is eaten by the lock unless decode runs from graphs.
import multiprocessing as mp, pathlib, threading, time

import torch

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("threads-gil.output.txt")
LAYERS, STEPS, H, F = 28, 200, 3584, 2304


def make(dev):
    torch.xpu.set_device(dev)
    d = torch.device("xpu", dev)
    w1 = torch.randn(H, F, device=d, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn(F, H, device=d, dtype=torch.bfloat16) * 0.01
    x = torch.randn(1, H, device=d, dtype=torch.bfloat16)
    return d, w1, w2, x


def step(w1, w2, x):
    for _ in range(LAYERS):
        y = x * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + 1e-6).to(x.dtype)   # norm (4-5 kernels)
        y = torch.nn.functional.silu(y @ w1) @ w2                                                  # 2 matmuls + act
        x = x + y                                                                                  # residual
    return x


def run_eager(dev, steps, out=None, key=None):
    d, w1, w2, x = make(dev)
    for _ in range(10):
        step(w1, w2, x)
    torch.xpu.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(steps):
        step(w1, w2, x)
    torch.xpu.synchronize(d)
    us = 1e6 * (time.perf_counter() - t0) / steps
    if out is not None:
        out[key] = us
    return us


def run_graph(dev, steps, out=None, key=None):
    d, w1, w2, x = make(dev)
    s = torch.xpu.Stream(device=d)
    s.wait_stream(torch.xpu.current_stream(d))
    with torch.xpu.stream(s):
        for _ in range(3):
            step(w1, w2, x)
    torch.xpu.current_stream(d).wait_stream(s)
    torch.xpu.synchronize(d)
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        y = step(w1, w2, x)
    torch.xpu.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(steps):
        g.replay()
    torch.xpu.synchronize(d)
    us = 1e6 * (time.perf_counter() - t0) / steps
    if out is not None:
        out[key] = us
    return us


def proc_entry(fn_name, dev, steps, q):
    q.put((dev, (run_eager if fn_name == "eager" else run_graph)(dev, steps)))


def two_threads(fn):
    res = {}
    ts = [threading.Thread(target=fn, args=(dev, STEPS, res, dev)) for dev in (0, 1)]
    for t in ts: t.start()
    for t in ts: t.join()
    return res[0], res[1]


def two_procs(name):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=proc_entry, args=(name, dev, STEPS, q)) for dev in (0, 1)]
    for p in ps: p.start()
    r = dict(q.get(timeout=600) for _ in ps)
    for p in ps: p.join()
    return r[0], r[1]


if __name__ == "__main__":
    # one case per invocation (2026-10-13: the all-in-one run hung with no output, so each case now runs alone under its
    # own timeout from threads-gil.input.cmd and appends one line):  <eager|graph> <A|B|C>
    import sys
    name, case = sys.argv[1], sys.argv[2]
    fn = run_eager if name == "eager" else run_graph
    print(f"{name} {case} starting", flush=True)
    if case == "A":
        r = f"{fn(0, STEPS):8.0f}"
    elif case == "B":
        b0, b1 = two_threads(fn)
        r = f"{b0:8.0f} {b1:8.0f}"
    else:
        c0, c1 = two_procs(name)
        r = f"{c0:8.0f} {c1:8.0f}"
    line = f"{name:5} {case}: {r} us per step  ({ {'A': 'one thread, one card', 'B': 'two threads, one process', 'C': 'two processes'}[case]})"
    print(line, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
