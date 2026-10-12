# allreduce-bench.input.py - how long does one all_reduce between the two B60s take, at the sizes a tensor split moves?
# Usage:  .venv312\Scripts\python.exe server\allreduce-bench.input.py [gloo|shm]      (writes server\allreduce-bench.output.txt)
# Why (2026-10-10, two cards faster than one): a tensor split of the 7B does 56 all_reduces per token (28 layers x 2), each
# ~7 KB in decode. The present relay (xpu_ram_relay.py, gloo over loopback) was measured at 0.13 ms per one-way hop; this
# measures the collective itself, as vLLM calls it, so the fast exchange that replaces it has an honest "before".
# Engines: gloo = xpu_ram_relay.all_reduce (today's path); shm = xpu_shm_exchange (pinned host + shared memory + flags).
# Sizes: decode 1 token (7 KB), 32 tokens (224 KB, batch of seats), 1000-token prefill (7 MB). Median of 200 / 200 / 20.
import datetime, importlib.util, os, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMM = ROOT / "src" / "vllm" / "vllm" / "distributed" / "device_communicators"
OUT = pathlib.Path(__file__).with_name("allreduce-bench.output.txt")
PORT = 29519
ENGINE = sys.argv[1] if len(sys.argv) > 1 else "gloo"
SIZES = [("1 token  (7 KB)", (1, 3584), 200), ("32 tokens (224 KB)", (32, 3584), 200), ("1000-token prefill (7 MB)", (1000, 3584), 20)]


def load(name):
    spec = importlib.util.spec_from_file_location(name, COMM / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker(rank, q):
    import torch, torch.distributed as dist
    torch.xpu.set_device(rank)
    dev = torch.device(f"xpu:{rank}")
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=2)
    group = dist.group.WORLD
    lines = [f"rank {rank}: {torch.xpu.get_device_name(rank)}, engine {ENGINE}"]
    if ENGINE == "gloo":
        relay = load("xpu_ram_relay")
        reduce = lambda t: relay.all_reduce(t, group)
        close = lambda: None
    else:
        shm = load("xpu_shm_exchange")
        ex = shm.ShmExchange(rank, 2, group, dev)
        reduce = ex.all_reduce
        close = ex.close
    for label, shape, reps in SIZES:
        t = torch.full(shape, float(rank + 1), dtype=torch.bfloat16, device=dev)
        # correctness first: 1 + 2 = 3 everywhere, identical on both ranks
        s = reduce(t.clone()); torch.xpu.synchronize()
        ok = bool((s.float() == 3.0).all())
        times = []
        dist.barrier(group)
        for _ in range(reps):
            x = t.clone(); torch.xpu.synchronize()
            t0 = time.perf_counter()
            r = reduce(x); torch.xpu.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2] * 1e6; p90 = times[int(len(times) * 0.9)] * 1e6
        nbytes = t.numel() * 2
        lines.append(f"rank {rank}: {label}: all_reduce median {med:8.1f} us, p90 {p90:8.1f} us  ({nbytes/1e9/(med/1e6):.2f} GB/s)  sum correct: {ok}")
    if rank == 0:
        per_token = 56 * (sorted([tt for tt in times])[0] and 0)  # placeholder keeps flake8 quiet about reuse
    close()
    dist.destroy_process_group()
    q.put((rank, lines))


if __name__ == "__main__":
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    out = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}  engine {ENGINE}"]
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(r, q)) for r in range(2)]
    for p in ps:
        p.start()
    results = {}
    for _ in ps:
        rank, lines = q.get(timeout=600)
        results[rank] = lines
    for p in ps:
        p.join(timeout=30)
    for r in sorted(results):
        out += results[r]
    # the number that matters for the 7B tensor split: 56 all_reduces of the decode size per token
    for line in results.get(0, []):
        if "1 token" in line:
            us = float(line.split("median")[1].split("us")[0])
            out.append(f"=> 56 x {us:.0f} us = {56*us/1000:.1f} ms of exchange per token for the 7B at TP=2 (one card computes a token in ~34 ms)")
    text = "\n".join(out); print(text); OUT.write_text(text + "\n", encoding="utf-8")
