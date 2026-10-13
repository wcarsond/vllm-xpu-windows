# one-process-torch.input.py - can ONE process drive both B60s through torch, done carefully? Three questions, one per run:
#   T1  is torch.xpu's current device per thread or per process? (a thread sets device 1; the main thread reads its own)
#   T2  two threads, one card each, every tensor and stream named explicitly, no set_device: does eager work complete, and
#       how does the step time compare with one card alone?
#   T3  ONE thread, two graphs: record the step on card 0, then on card 1 (one after the other), then replay both each step
#       from the same thread. Replays are asynchronous, so one thread may keep both cards busy with no threads at all.
# Usage:  .venv312\Scripts\python.exe probe\one-process-torch.input.py <T1|T2|T3>     (one line appended to the output)
# Why (2026-10-13): threads-gil showed two threads in one process deadlocking (eager) and capturing an empty graph for one
# card ("captured on wrong device or stream") - possibly because the current device is process-wide and the threads switch
# it under each other, not because one process cannot drive two cards. These runs separate the two.
import pathlib, sys, threading, time

import torch

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("one-process-torch.output.txt")
LAYERS, STEPS, H, F = 28, 200, 3584, 2304


def weights(d):
    w1 = torch.randn(H, F, device=d, dtype=torch.bfloat16) * 0.01
    w2 = torch.randn(F, H, device=d, dtype=torch.bfloat16) * 0.01
    x = torch.randn(1, H, device=d, dtype=torch.bfloat16)
    return w1, w2, x


def step(w1, w2, x):
    for _ in range(LAYERS):
        y = x * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
        y = torch.nn.functional.silu(y @ w1) @ w2
        x = x + y
    return x


def say(s):
    print(s, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def t1():
    torch.xpu.set_device(0)
    seen = {}

    def other():
        torch.xpu.set_device(1)
        seen["thread"] = torch.xpu.current_device()

    th = threading.Thread(target=other)
    th.start(); th.join()
    say(f"T1 current device: thread set 1 and reads {seen['thread']}; main thread (set 0) now reads {torch.xpu.current_device()} "
        f"-> {'PER THREAD' if torch.xpu.current_device() == 0 else 'PER PROCESS (a thread changes it for everyone)'}")


def one_card_eager(d, s, res, key):
    w1, w2, x = weights(d)
    with torch.xpu.stream(s):
        for _ in range(10):
            step(w1, w2, x)
        s.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step(w1, w2, x)
        s.synchronize()
    res[key] = 1e6 * (time.perf_counter() - t0) / STEPS


def t2():
    devs = [torch.device("xpu", i) for i in (0, 1)]
    streams = [torch.xpu.Stream(device=d) for d in devs]
    alone = {}
    one_card_eager(devs[0], streams[0], alone, 0)
    res = {}
    ths = [threading.Thread(target=one_card_eager, args=(devs[i], streams[i], res, i)) for i in (0, 1)]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join(timeout=150)
    hung = [i for i, t in enumerate(ths) if t.is_alive()]
    if hung:
        say(f"T2 two threads, explicit devices and streams: HUNG (threads {hung} still running after 150 s)")
        return
    say(f"T2 two threads, explicit devices and streams: completed; us per step card0 {res[0]:.0f}, card1 {res[1]:.0f} "
        f"(one card alone {alone[0]:.0f})")


def t3():
    devs = [torch.device("xpu", i) for i in (0, 1)]
    graphs, outs = [], []
    for d in devs:                        # record one graph per card, one after the other, from this one thread
        w1, w2, x = weights(d)
        s = torch.xpu.Stream(device=d)
        with torch.xpu.device(d):
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
        graphs.append(g)
        outs.append((y, w1, w2, x))
    # reference: card 0's graph alone
    t0 = time.perf_counter()
    for _ in range(STEPS):
        graphs[0].replay()
    torch.xpu.synchronize(devs[0])
    alone = 1e6 * (time.perf_counter() - t0) / STEPS
    # both graphs replayed from this one thread, every step
    torch.xpu.synchronize(devs[0]); torch.xpu.synchronize(devs[1])
    t0 = time.perf_counter()
    for _ in range(STEPS):
        graphs[0].replay()
        graphs[1].replay()
    torch.xpu.synchronize(devs[0]); torch.xpu.synchronize(devs[1])
    both = 1e6 * (time.perf_counter() - t0) / STEPS
    # correctness: each card's replayed output matches an eager run of the same step on that card
    ok = []
    for d, (y, w1, w2, x) in zip(devs, outs):
        with torch.xpu.device(d):
            ref = step(w1, w2, x)
            torch.xpu.synchronize(d)
        diff = (y.float() - ref.float()).abs().max().item()
        ok.append(f"{'match' if torch.allclose(y.float(), ref.float(), atol=1e-2, rtol=1e-2) else 'DIFFER'} "
                  f"(max abs diff {diff:.3g}, |ref| max {ref.float().abs().max().item():.3g}, nan in y {bool(torch.isnan(y).any())})")
    say(f"T3 one thread, two graphs: card0 alone {alone:.0f} us per step; both cards replayed from one thread {both:.0f} us per "
        f"step ({'in parallel' if both < 1.3 * alone else 'NOT in parallel'}); outputs match eager: {ok}")


if __name__ == "__main__":
    def t3b():
        # T3 with each card's graph checked right after it is captured and replayed ALONE, before the other card's capture,
        # to tell "the second capture disturbs the first card's graph" from "card 0's graph is wrong by itself"
        for i in (0, 1):
            d = torch.device("xpu", i)
            w1, w2, x = weights(d)
            s = torch.xpu.Stream(device=d)
            with torch.xpu.device(d):
                s.wait_stream(torch.xpu.current_stream(d))
                with torch.xpu.stream(s):
                    for _ in range(3):
                        step(w1, w2, x)
                torch.xpu.current_stream(d).wait_stream(s)
                torch.xpu.synchronize(d)
                g = torch.xpu.XPUGraph()
                with torch.xpu.graph(g):
                    y = step(w1, w2, x)
                g.replay()
                torch.xpu.synchronize(d)
                ref = step(w1, w2, x)
                torch.xpu.synchronize(d)
            diff = (y.float() - ref.float()).abs().max().item()
            say(f"T3b card {i} alone: graph vs eager max abs diff {diff:.3g} (|ref| max {ref.float().abs().max().item():.3g})")

    {"T1": t1, "T2": t2, "T3": t3, "T3b": t3b}[sys.argv[1]]()
