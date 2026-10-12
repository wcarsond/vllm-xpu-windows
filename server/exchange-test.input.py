# exchange-test.input.py - unit test of the two-card GPU-timeline exchange (_xpu_exchange.pyd) outside vLLM: two processes,
# one B60 each, gloo for the one-time name broadcast, every all_reduce checked bit for bit against the host's answer.
# Usage:  .venv312\Scripts\python.exe server\exchange-test.input.py [rounds]      (writes server\exchange-test.output.txt)
# Why (2026-10-11 09:10): the first vLLM run with the exchange failed its warm-up check (bf16 256 elements came back as the
# stale fp32 contents of the previous call's buffer). A 20-second test that replays mixed sizes and dtypes beats a
# 90-second engine start per experiment. Expected value: rank 0 holds v, rank 1 holds 2v; the sum rounds exactly as the
# kernel rounds (float add, round to nearest even to the dtype), so equality is exact.
import datetime, os, pathlib, sys, time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

OUT = pathlib.Path(__file__).with_name("exchange-test.output.txt")
SIZES = [256, 128, 1792, 65536, 256, 3584, 9408, 16384, 1792, 1 << 20, 1792, 7340032, 1792, 256, 16 << 20, 128, 65536, 1792]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def expected(numel, dtype, device):
    v = (torch.arange(numel, device=device, dtype=torch.float32) % 977) * 0.0625 - 20.0
    a0 = v.to(dtype)
    a1 = (2.0 * v).to(dtype)
    return a0, a1, (a0.float() + a1.float()).to(dtype)


def worker(rank, rounds, q):
    lines = []

    def log(s):
        s = f"[rank {rank}] {s}"
        print(s, flush=True)
        lines.append(s)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29571"
    os.environ.setdefault("VLLM_XPU_EXCHANGE_STATS", "0")
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.xpu.set_device(rank)
    dev = torch.device("xpu", rank)
    from vllm.distributed.device_communicators.xpu_gpu_exchange import GpuExchange

    t0 = time.time()
    try:
        ex = GpuExchange(rank, 2, dist.group.WORLD, dev)
    except Exception as e:  # noqa: BLE001
        log(f"FAIL at construction/warm-up: {e}")
        q.put("\n".join(lines))
        return
    log(f"exchange up and warm-up checked in {time.time() - t0:.1f} s")
    fails = 0
    n = 0
    for r in range(rounds):
        for dtype in DTYPES:
            for numel in SIZES:
                a0, a1, want = expected(numel, dtype, dev)
                t = (a0 if rank == 0 else a1).clone()
                if not ex.accepts(t):
                    continue
                t = ex.all_reduce(t)
                torch.xpu.synchronize(rank)
                n += 1
                bad = int((t.view(torch.int16 if t.element_size() == 2 else torch.int32) != want.view(torch.int16 if t.element_size() == 2 else torch.int32)).sum())
                if bad:
                    fails += 1
                    if fails <= 5:
                        log(f"WRONG round {r} {numel} x {dtype}: {bad} of {numel} elements differ; first got {t[:4].float().tolist()} want {want[:4].float().tolist()}; {ex.ext.debug()}")
    dist.barrier()
    st = ex.stats(8)
    log(f"{n} all_reduces, {fails} wrong; last stage times us (seq, bytes, wait_card, memcpy_in, wait_peer, memcpy_out, enqueue): "
        + str([(s, b, round(a), round(c), round(d), round(e), round(f)) for s, b, a, c, d, e, f in st]))
    # decode-size timing: 2000 back-to-back 7 KB all_reduces, main-thread time per call and wall per call
    t = torch.ones(3584, dtype=torch.bfloat16, device=dev)
    dist.barrier()
    torch.xpu.synchronize(rank)
    t1 = time.perf_counter()
    for _ in range(2000):
        t = ex.all_reduce(t)
    t2 = time.perf_counter()
    torch.xpu.synchronize(rank)
    t3 = time.perf_counter()
    log(f"2000 x 7 KB back to back: {1e6 * (t2 - t1) / 2000:.1f} us per call on the main thread, {1e6 * (t3 - t1) / 2000:.1f} us per all_reduce wall")
    log("PASS" if fails == 0 else "FAIL")
    ex.close()
    dist.destroy_process_group()
    q.put("\n".join(lines))


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, rounds, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = []
    deadline = time.time() + 600
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
    header = f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  rounds {rounds}"
    OUT.write_text(header + "\n" + "\n".join(results) + "\n", encoding="utf-8")
    print(header)
