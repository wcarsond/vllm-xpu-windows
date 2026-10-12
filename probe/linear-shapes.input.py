# linear-shapes.input.py - does torch's own F.linear (oneDNN GEMM) launch on the B60 under Windows for small M?
# Usage:  .venv312\Scripts\python.exe probe\linear-shapes.input.py      (writes probe\linear-shapes.output.txt)
# Why (2026-10-10): first light died at the first real step inside torch.nn.functional.linear; the UR trace names the
# failing launch as oneDNN's native "gemm_kernel" (global 896, local 64) returning UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY.
# The profiling pass (large M) and rung 1's 4096x4096 matmul worked, so the suspicion is the small-M GEMM kernel oneDNN
# picks for a 13-token prefill. This runs F.linear for Qwen-0.5B and Coder-7B shapes across M and dtypes, one at a time,
# synchronizing after each, and records pass/fail per shape. No vLLM involved: if it fails here it is torch/oneDNN/driver.
import datetime, pathlib, time, torch, torch.nn.functional as F

out = pathlib.Path(__file__).with_name("linear-shapes.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}  {torch.xpu.get_device_name(0)}"]
dev = torch.device("xpu:0")
shapes = [("qwen0.5b K=896 N=896", 896, 896), ("qwen0.5b K=896 N=4864 (mlp up)", 896, 4864), ("coder7b K=3584 N=3584", 3584, 3584), ("coder7b K=3584 N=18944 (mlp up)", 3584, 18944)]
for label, K, N in shapes:
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        w = torch.randn(N, K, dtype=dtype, device=dev) * 0.02
        for M in (1, 13, 25, 64, 256, 2048):
            x = torch.randn(M, K, dtype=dtype, device=dev)
            try:
                t0 = time.perf_counter(); y = F.linear(x, w); torch.xpu.synchronize(); dt = (time.perf_counter() - t0) * 1000
                ref = (x.float().cpu() @ w.float().cpu().T)
                err = (y.float().cpu() - ref).abs().max().item()
                lines.append(f"ok    {label:34s} {str(dtype):16s} M={M:<5d} {dt:7.2f} ms  max|err| {err:.3g}")
            except Exception as e:
                lines.append(f"FAIL  {label:34s} {str(dtype):16s} M={M:<5d} {type(e).__name__}: {str(e)[:120]}")
                try: torch.xpu.synchronize()
                except Exception as e2: lines.append(f"      device state after failure: {type(e2).__name__}: {str(e2)[:100]}")
        del w
text = "\n".join(lines); print(text); out.write_text(text + "\n", encoding="utf-8")
