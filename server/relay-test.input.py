# relay-test.input.py - does the Windows card-to-card relay move bf16 tensors between the two B60s intact, and how fast?
# Usage:  .venv312\Scripts\python.exe server\relay-test.input.py      (writes server\relay-test.output.txt)
#
# Why (2026-10-10, rung 6): vLLM's XPU communicator on Windows relays through host RAM over gloo
# (src\vllm\vllm\distributed\device_communicators\xpu_ram_relay.py). That module imports only torch, so it is tested here
# by file path, without importing vllm (which cannot load until the kernels exist). Two spawned processes, one per card,
# gloo over loopback TCP. Checks: send_tensor_dict/recv_tensor_dict round trip bit-exact; all_reduce equals the CPU sum;
# timings for a pipeline hand-off of one decode token for a 7B model (hidden 3584 + residual, bf16 = 14 KB), a 1000-token
# prefill (14 MB) and the rung-1 size (84 MB). Compare with rung 1: 84 MB card->RAM->card in-process was ~10 ms.
import datetime, importlib.util, os, pathlib, sys, time

RELAY = pathlib.Path(__file__).resolve().parents[1] / "src" / "vllm" / "vllm" / "distributed" / "device_communicators" / "xpu_ram_relay.py"
OUT = pathlib.Path(__file__).with_name("relay-test.output.txt")
PORT = 29517
SIZES = [("1 token, 7B (hidden+residual)", (2, 1, 3584)), ("1000-token prefill", (2, 1000, 3584)), ("84 MB, rung-1 size", (2, 6000, 3584))]


def load_relay():
    spec = importlib.util.spec_from_file_location("xpu_ram_relay", RELAY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker(rank, q):
    import torch, torch.distributed as dist
    relay = load_relay()
    torch.xpu.set_device(rank)
    dev = torch.device(f"xpu:{rank}")
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=2)
    group = dist.group.WORLD
    lines = [f"rank {rank}: {torch.xpu.get_device_name(rank)}, gloo up"]
    peer = 1 - rank

    # 1. tensor dict round trip, bit-exact
    torch.manual_seed(1234 + rank)
    d = {"hidden_states": torch.randn(7, 3584, dtype=torch.bfloat16, device=dev), "residual": torch.randn(7, 3584, dtype=torch.bfloat16, device=dev), "step": 42, "empty": torch.empty(0, dtype=torch.bfloat16, device=dev)}
    if rank == 0:
        relay.send_tensor_dict(d, peer, group)
        back = relay.recv_tensor_dict(peer, group, dev)
        torch.xpu.synchronize()
        ok = all(torch.equal(back[k].cpu(), d[k].cpu()) for k in ("hidden_states", "residual")) and back["step"] == 42 and back["empty"].numel() == 0
        lines.append(f"rank 0: tensor dict there and back, bit-exact: {ok}")
    else:
        got = relay.recv_tensor_dict(peer, group, dev)
        torch.xpu.synchronize()
        lines.append(f"rank 1: received keys {list(got)}; hidden on {got['hidden_states'].device}")
        relay.send_tensor_dict(got, peer, group)

    # 2. all_reduce equals the CPU sum
    a = torch.full((1024, 1024), float(rank + 1), dtype=torch.bfloat16, device=dev)
    s = relay.all_reduce(a, group)
    torch.xpu.synchronize()
    lines.append(f"rank {rank}: all_reduce of 1s and 2s -> {s[0,0].item():.0f} (want 3), all equal: {bool((s == 3).all())}")

    # 2b. all_gather in the shape the tensor split uses for logits (rows x half-vocab); result[r] must be rank r's rows.
    #     Added 2026-10-11 after the first TP run died in gloo with "invalid tensor size ... expected (256, 76032)".
    g_in = torch.full((256, 76032), float(rank + 1), dtype=torch.bfloat16, device=dev)
    g = relay.all_gather(g_in, 2, group)
    torch.xpu.synchronize()
    ok_g = tuple(g.shape) == (2, 256, 76032) and bool((g[0] == 1).all()) and bool((g[1] == 2).all())
    lines.append(f"rank {rank}: all_gather (256 x 76032) -> shape {tuple(g.shape)}, rows by rank correct: {ok_g}")

    # 3. timings: rank 0 sends, rank 1 receives and sends back; median of 7 for the round trip, halved = one hop
    for label, shape in SIZES:
        t = torch.randn(shape, dtype=torch.bfloat16, device=dev)
        nbytes = t.numel() * 2
        times = []
        for _ in range(8):
            dist.barrier(group)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            if rank == 0:
                relay.send(t, peer, group)
                r = relay.recv(t.shape, t.dtype, peer, group, dev)
            else:
                r = relay.recv(t.shape, t.dtype, peer, group, dev)
                relay.send(r, peer, group)
            torch.xpu.synchronize()
            times.append(time.perf_counter() - t0)
        times = sorted(times[1:])
        hop_ms = times[len(times) // 2] / 2 * 1000
        if rank == 0:
            lines.append(f"rank 0: {label}: {nbytes/1e6:.2f} MB, one hop card->RAM->other card {hop_ms:.2f} ms median ({nbytes/1e9/(hop_ms/1000):.1f} GB/s), round trip intact: {torch.equal(r.cpu(), t.cpu())}")
    dist.destroy_process_group()
    q.put((rank, lines))


if __name__ == "__main__":
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    out = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}  relay {RELAY}"]
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(r, q)) for r in range(2)]
    for p in ps:
        p.start()
    results = {}
    for _ in ps:
        rank, lines = q.get(timeout=300)
        results[rank] = lines
    for p in ps:
        p.join(timeout=30)
    for r in sorted(results):
        out += results[r]
    text = "\n".join(out)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
