# int4-gemv.input.py - how close does the int4 decode kernel get to the B60's memory bandwidth? Times
# torch.ops._xpu_C.int4_gemm_w4a16 (what vLLM's XPUwNa16LinearKernel calls for GPTQ/AWQ int4) at real decode shapes, and a
# bf16 matmul at the same shape, and reports the weight bytes read per second against the card's 456 GB/s.
# Usage:  .venv312\Scripts\python.exe probe\int4-gemv.input.py [card]          (writes probe\int4-gemv.output.txt)
# Why (2026-10-13): the 4-bit models decode at ~59% of the bandwidth ceiling on two cards (Granite 30B int4), the 16-bit 7B
# at 77% and one card of it at ~100%. If the int4 kernel alone reads well under bf16's GB/s at decode shapes (1-8 tokens),
# it is the loss, and the fix is in the kernel (tile shape, dequant cost, split-K for skinny M), not in the exchange.
# Weights are laid out exactly as XPUwNa16LinearKernel leaves them for a symmetric GPTQ checkpoint (group 128).
import pathlib, sys, time

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C, as vllm/platforms/xpu.py does)

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("int4-gemv.output.txt")
card = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dev = torch.device(f"xpu:{card}")
PEAK = 456.0
GROUP = 128
# (label, K = input features, N = output features): per-card shapes of a two-card tensor split
SHAPES = [
    ("7B qkv (TP2)", 3584, 2304), ("7B o (TP2)", 1792, 3584), ("7B gate_up (TP2)", 3584, 18944), ("7B down (TP2)", 9472, 3584),
    ("32B qkv (TP2)", 5120, 3584), ("32B gate_up (TP2)", 5120, 27648), ("32B down (TP2)", 13824, 5120),
    ("one card, 5120 x 27648", 5120, 27648), ("one card, 8192 x 8192", 8192, 8192),
]
TOKENS = [1, 4, 8]
lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  torch {torch.__version__}  card {card}  group {GROUP}  peak {PEAK} GB/s",
         f"{'shape':24} {'K':>6} {'N':>6} {'tok':>3} | {'int4 us':>8} {'GB/s':>6} {'% peak':>6} | {'bf16 us':>8} {'GB/s':>6} {'% peak':>6} | int4 speedup"]


def timed(fn, reps=200):
    for _ in range(10):
        fn()
    torch.xpu.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.xpu.synchronize(dev)
    return (time.perf_counter() - t0) / reps


for label, K, N in SHAPES:
    # GPTQ checkpoint layout: qweight [K/8, N] int32 (8 nibbles per word along K), scales [K/group, N]
    qweight = torch.randint(-(2**31), 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
    scales = (torch.rand(K // GROUP, N, device=dev, dtype=torch.float32) * 0.01 + 0.001).to(torch.bfloat16)
    # what process_weights_after_loading does for GPTQ: qweight transposed to [N, K/8] contiguous, scales unchanged,
    # zero point a single int8 8 (symmetric)
    w_q = qweight.t().contiguous()
    w_zp = torch.tensor([8], dtype=torch.int8, device=dev)
    w_bf16 = torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.02
    int4_bytes = K * N / 2 + scales.numel() * 2
    bf16_bytes = K * N * 2
    for M in TOKENS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        try:
            t4 = timed(lambda: torch.ops._xpu_C.int4_gemm_w4a16(x, w_q.t(), None, scales, w_zp, GROUP, None))
            s4 = f"{t4*1e6:8.1f} {int4_bytes/t4/1e9:6.0f} {100*int4_bytes/t4/1e9/PEAK:5.0f}%"
        except Exception as e:  # noqa: BLE001
            t4 = None
            s4 = f"FAILED {type(e).__name__}: {str(e)[:60]}"
        tb = timed(lambda: x @ w_bf16)
        sb = f"{tb*1e6:8.1f} {bf16_bytes/tb/1e9:6.0f} {100*bf16_bytes/tb/1e9/PEAK:5.0f}%"
        sp = f"{tb/t4:5.2f}x (ideal 4x)" if t4 else "-"
        s = f"{label:24} {K:6} {N:6} {M:3} | {s4} | {sb} | {sp}"
        print(s, flush=True)
        lines.append(s)
    del qweight, scales, w_q, w_bf16
    torch.xpu.empty_cache()
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
