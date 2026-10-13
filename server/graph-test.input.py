# graph-test.input.py - is the two-card exchange graph-safe? Two processes, one B60 each: record decode-size all_reduces
# into an XPU graph, replay it many times with fresh inputs, mixed with eager small and big all_reduces in between (as
# vLLM's prefill and decode alternate), and check every result bit for bit. Then time a replayed "layer stack" against the
# same calls made eagerly. Writes server\graph-test.output.txt.
# Usage:  .venv312\Scripts\python.exe server\graph-test.input.py [replays]          (default 200)
# Why (2026-10-13): eager decode loses ~225 us per layer to launches (0.5B one card: 118 -> 319 tok/s with graphs), and
# two-card graphs were off because the exchange's host-side bookkeeping could not be recorded. The exchange now keeps its
# decode-size sequence on the card and its helper finds jobs by itself; this is the test that it replays correctly.
import datetime, os, pathlib, sys, time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

OUT = pathlib.Path(__file__).with_name("graph-test.output.txt")
H = 3584          # the 7B's hidden size: a one-token decode all_reduce is H bf16 = 7 KB
LAYERS = 56       # all_reduces per captured "token" (the 7B has 28 layers x 2)


def worker(rank, replays, q):
    lines = []

    def log(s):
        s = f"[rank {rank}] {s}"
        print(s, flush=True)
        lines.append(s)

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29573"
        dist.init_process_group("gloo", rank=rank, world_size=2)
        torch.xpu.set_device(rank)
        dev = torch.device("xpu", rank)
        from vllm.distributed.device_communicators.xpu_gpu_exchange import GpuExchange
        ex = GpuExchange(rank, 2, dist.group.WORLD, dev)
        log("exchange up; big path " + ("direct" if ex.ext.direct() else "helper memcpy"))
        scale = 1.0 if rank == 0 else 2.0

        # static inputs, one per layer; the graph reads them and writes its outputs into static outputs
        xs = [torch.zeros(H, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]
        side = torch.xpu.Stream(device=dev)
        side.wait_stream(torch.xpu.current_stream(dev))
        with torch.xpu.stream(side):
            for x in xs:                     # warm-up on the capture stream, eager
                ex.all_reduce(x)
        torch.xpu.current_stream(dev).wait_stream(side)
        torch.xpu.synchronize(dev)
        dist.barrier()

        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            ys = [ex.all_reduce(x) for x in xs]
        dist.barrier()
        log(f"captured {LAYERS} all_reduces; exchange counts (small, recorded, big) = {ex.ext.counts()}")

        wrong = 0
        big = torch.empty(512 * H, dtype=torch.bfloat16, device=dev)     # a 3.7 MB prefill-size all_reduce
        for r in range(replays):
            base = float((r % 50) + 1)
            for i, x in enumerate(xs):
                x.fill_(scale * (base + i % 7))
            g.replay()
            torch.xpu.synchronize(dev)
            for i, y in enumerate(ys):
                want = 3.0 * (base + i % 7)
                if not bool((y == want).all()):
                    wrong += 1
                    if wrong <= 5:
                        log(f"WRONG replay {r} layer {i}: got {y[:3].float().tolist()} want {want}; {ex.ext.debug()}")
            if r % 10 == 5:                  # eager calls between replays, small and big, as prefill and decode alternate
                e = ex.all_reduce(torch.full((H,), scale * 5.0, dtype=torch.bfloat16, device=dev))
                b = ex.all_reduce(big.fill_(scale * 2.0))
                torch.xpu.synchronize(dev)
                if not bool((e == 15.0).all()) or not bool((b == 6.0).all()):
                    wrong += 1
                    log(f"WRONG eager after replay {r}: small {e[:2].float().tolist()} big {b[:2].float().tolist()}")
        log(f"{replays} replays x {LAYERS} all_reduces + eager calls between: {wrong} wrong")

        # timing: one replay of the 56-exchange graph against the same 56 calls made eagerly
        dist.barrier()
        torch.xpu.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(100):
            g.replay()
        torch.xpu.synchronize(dev)
        t1 = time.perf_counter()
        dist.barrier()
        torch.xpu.synchronize(dev)
        t2 = time.perf_counter()
        for _ in range(100):
            for x in xs:
                ex.all_reduce(x)
        torch.xpu.synchronize(dev)
        t3 = time.perf_counter()
        log(f"{LAYERS} all_reduces: replayed graph {1e6 * (t1 - t0) / 100 / LAYERS:.1f} us each, eager "
            f"{1e6 * (t3 - t2) / 100 / LAYERS:.1f} us each; status {ex.ext.status()}")
        log("PASS" if wrong == 0 else "FAIL")
        ex.close()
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        log(f"EXCEPTION {type(e).__name__}: {e}")
    q.put("\n".join(lines))


if __name__ == "__main__":
    replays = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, replays, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = []
    deadline = time.time() + 300
    while len(results) < 2 and time.time() < deadline:
        try:
            results.append(q.get(timeout=5))
        except Exception:  # noqa: BLE001
            if not any(p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    header = f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  replays {replays}"
    OUT.write_text(header + "\n" + "\n".join(results) + ("\n(timed out or crashed)" if len(results) < 2 else "") + "\n", encoding="utf-8")
    print(header)
