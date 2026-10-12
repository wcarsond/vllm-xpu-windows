# linear-shapes2.input.py - F.linear on the B60 under the conditions vLLM's step adds that probe 1 did not have.
# Usage:  .venv312\Scripts\python.exe probe\linear-shapes2.input.py      (writes probe\linear-shapes2.output.txt)
# Why (2026-10-10): linear-shapes.input.py passed all 72 plain cases, yet inside vLLM the same F.linear launch fails with
# UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY (oneDNN "gemm_kernel", 17 args). vLLM's call differs in: a bias (Qwen2.5 q/k/v
# projections have one; a bias post-op is a different oneDNN kernel), a non-contiguous activation slice, a card already
# holding ~10 GB (weights + KV cache; oneDNN scratch is allocated at launch), and a long run of layers. Each is tried alone
# and then all together, M=13 (the failing prefill) and M=1 (decode), bf16, Qwen-0.5B shape.
import datetime, pathlib, torch, torch.nn.functional as F

out = pathlib.Path(__file__).with_name("linear-shapes2.output.txt")
dev = torch.device("xpu:0"); dt = torch.bfloat16; K, N = 896, 896
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}  free before: {torch.xpu.mem_get_info(0)[0]/2**30:.1f} GiB"]

def attempt(label, fn):
    try:
        y = fn(); torch.xpu.synchronize()
        lines.append(f"ok    {label}  (out {tuple(y.shape)})")
    except Exception as e:
        lines.append(f"FAIL  {label}  {type(e).__name__}: {str(e)[:110]}")

w = (torch.randn(N, K, dtype=dt, device=dev) * 0.02)
b = torch.randn(N, dtype=dt, device=dev) * 0.02
big = torch.empty(4 * 896 * 4096, dtype=dt, device=dev).view(4096, 4 * 896)   # a wider buffer to slice from
for M in (13, 1):
    x = torch.randn(M, K, dtype=dt, device=dev)
    xs = big[:M, 896:2 * 896]                                                  # non-contiguous row slice
    attempt(f"M={M} plain",                     lambda: F.linear(x, w))
    attempt(f"M={M} + bias",                    lambda: F.linear(x, w, b))
    attempt(f"M={M} non-contiguous x",          lambda: F.linear(xs, w))
    attempt(f"M={M} non-contiguous x + bias",   lambda: F.linear(xs, w, b))
    attempt(f"M={M} bias, 24 layers in a row",  lambda: [F.linear(x, w, b) for _ in range(24)][-1])
# memory pressure: hold most of the card, like weights + KV cache, then repeat the bias case
free = torch.xpu.mem_get_info(0)[0]
hold = torch.empty(int(free * 0.80) // 2, dtype=dt, device=dev)
lines.append(f"holding {hold.numel()*2/2**30:.1f} GiB; free now {torch.xpu.mem_get_info(0)[0]/2**30:.1f} GiB")
for M in (13, 1):
    x = torch.randn(M, K, dtype=dt, device=dev)
    attempt(f"M={M} + bias under memory pressure",            lambda: F.linear(x, w, b))
    attempt(f"M={M} plain under memory pressure",             lambda: F.linear(x, w))
    attempt(f"M={M} bias, 24 layers under memory pressure",   lambda: [F.linear(x, w, b) for _ in range(24)][-1])
text = "\n".join(lines); print(text); out.write_text(text + "\n", encoding="utf-8")
