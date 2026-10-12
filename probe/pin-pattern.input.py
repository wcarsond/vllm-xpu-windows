# pin-pattern.input.py - replay vLLM's offloader pattern (card tensor -> .to("cpu") -> .pin_memory() -> card view -> kernel)
# with the 14B's real parameter shapes, each test in a FRESH process, naming the step that fails.
# Usage:  .venv312\Scripts\python.exe probe\pin-pattern.input.py      (writes probe\pin-pattern.output.txt; card 0 only)
# Why (2026-10-11 13:27): pin-limit.input.py pinned 8 GiB in one piece and 16 GiB of 256 MiB chunks, yet vLLM's offloader
# dies in cpu_data.pin_memory() (uva.py:114) with OUT_OF_DEVICE_MEMORY. First version (13:24, one process) saw: pin-only
# pattern fails at tensor 125 after 9.23 GiB; then EVERYTHING failed, because torch's caching host allocator keeps freed
# pinned blocks locked (del does not unlock), so test B inherited test A's 9+ GiB. Hence one process per test here.
# Qwen2.5-14B: hidden 5120, intermediate 13824, 40 q / 8 kv heads x 128, 48 layers.
import os, pathlib, subprocess, sys, time
os.environ.setdefault("ZE_AFFINITY_MASK", "0")
HERE = pathlib.Path(__file__).resolve()
out = HERE.with_name("pin-pattern.output.txt")
TESTS = ["A", "B", "C"]
if len(sys.argv) == 1:
    lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')} - each test in its own process"]
    for t in TESTS:
        r = subprocess.run([sys.executable, str(HERE), t], capture_output=True, text=True)
        got = [l for l in r.stdout.splitlines() if l.startswith(t + " ")]
        if not got:
            tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1][:140]
            got = [f"{t} produced no result line, exit {r.returncode}: {tail}"]
        for l in got:
            print(l, flush=True); lines.append(l)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0)

import torch
import vllm_xpu_kernels._C  # noqa: F401  registers torch.ops._C.get_xpu_view_from_cpu_tensor
GiB = 1 << 30
H, I, QKV = 5120, 13824, 5120 + 2 * 1024
layer = [("qkv_proj.weight", (QKV, H)), ("qkv_proj.bias", (QKV,)), ("o_proj.weight", (H, H)),
         ("gate_up_proj.weight", (2 * I, H)), ("down_proj.weight", (H, I)), ("input_layernorm", (H,)), ("post_layernorm", (H,))]

def replay(view: bool, label: str, budget: int):
    held, total, n, step, name, shape, li = [], 0, 0, "", "", (), 0
    t0 = time.time()
    try:
        for li in range(48):
            for name, shape in layer:
                if total >= budget:
                    raise StopIteration
                step = "card alloc"; dev = torch.empty(shape, dtype=torch.bfloat16, device="xpu")
                step = "to cpu"; cpu = dev.to("cpu"); del dev
                step = "pin_memory"; cpu = cpu.pin_memory()
                if view:
                    step = "card view"; cpu = torch.ops._C.get_xpu_view_from_cpu_tensor(cpu)
                    step = "kernel through the view"; cpu.zero_(); torch.xpu.synchronize()
                held.append(cpu); total += cpu.numel() * 2; n += 1
    except StopIteration:
        print(f"{label}: budget {budget/GiB:.0f} GiB reached, {n} tensors (layer {li}), {time.time()-t0:.1f} s, no failure", flush=True)
    except Exception as e:
        print(f"{label}: FAIL at tensor {n+1} step '{step}' (layer {li}, {name} {tuple(shape)}, "
              f"{torch.Size(shape).numel()*2/2**20:.1f} MiB) after {n} ok / {total/GiB:.2f} GiB nominal: {str(e).splitlines()[0][:110]}", flush=True)

t = sys.argv[1]
if t == "A":
    replay(False, "A pin only, 14B shapes, 10 GiB budget", 10 * GiB)
elif t == "B":
    replay(True, "B pin + card view + kernel through it, 14B shapes, 4 GiB budget", 4 * GiB)
elif t == "C":
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    try:
        cpu = torch.zeros(64 << 20, dtype=torch.float32).pin_memory()          # 256 MiB
        view = get_accelerator_view_from_cpu_tensor(cpu)
        view.fill_(3.0); view.add_(1.0); torch.xpu.synchronize()
        ok = bool((cpu == 4.0).all())
        w_cpu = torch.randn(4096, 4096, dtype=torch.float16).pin_memory()
        w_view = get_accelerator_view_from_cpu_tensor(w_cpu)
        x = torch.randn(16, 4096, dtype=torch.float16, device="xpu")
        w_dev = w_cpu.to("xpu")
        y_host = x @ w_view.T; y_card = x @ w_dev.T; torch.xpu.synchronize()
        t0 = time.time()
        for _ in range(20): y_host = x @ w_view.T
        torch.xpu.synchronize(); t_host = (time.time() - t0) / 20
        t0 = time.time()
        for _ in range(20): y_card = x @ w_dev.T
        torch.xpu.synchronize(); t_card = (time.time() - t0) / 20
        print(f"C card fills host memory through the view, CPU reads 4.0: {'ok' if ok else 'WRONG'}; matmul 16x4096x4096 fp16 with "
              f"weights in host memory {t_host*1e3:.2f} ms ({32e6/1e9/t_host:.1f} GB/s of weights) vs on the card {t_card*1e3:.3f} ms; "
              f"max |diff| {(y_host-y_card).abs().max().item():.4f}", flush=True)
    except Exception as e:
        print(f"C view + kernels: FAIL {str(e).splitlines()[0][:140]}", flush=True)
