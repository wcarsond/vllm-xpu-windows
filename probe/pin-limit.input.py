# pin-limit.input.py - how much host memory can one process pin for the cards on this driver, and in what chunk sizes?
# Usage:  .venv312\Scripts\python.exe probe\pin-limit.input.py      (writes probe\pin-limit.output.txt; card 0 only)
# Why (2026-10-11 13:25): vLLM's cpu_offload_gb pins every spilled weight (offloader/uva.py:114, cpu_data.pin_memory()) and
# died with UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY at 10, 12 and 16 GB of spill (7B and 14B alike). Three questions:
#   A. is there a size limit on ONE pinned allocation?   B. is there a limit on the TOTAL pinned by one process?
#   C. can a kernel on the card read pinned host memory through vLLM's UVA view (the zero-copy path the offloader uses)?
# Cumulative test stops at 16 GB so the machine (31 GB) keeps its head.
import os, pathlib, time
os.environ.setdefault("ZE_AFFINITY_MASK", "0")
import torch
out = pathlib.Path(__file__).with_name("pin-limit.output.txt")
lines = []
def log(s):
    print(s, flush=True); lines.append(s)

log(f"torch {torch.__version__}, card {torch.xpu.get_device_name(0)}, {time.strftime('%Y-%m-%d %H:%M:%S')}")
GB = 1 << 30
# A. single allocations, growing
for gb in (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8):
    t0 = time.time()
    try:
        t = torch.empty(int(gb * GB), dtype=torch.uint8).pin_memory()
        log(f"A single {gb:>4} GB: ok ({time.time()-t0:.2f} s)")
        del t
    except Exception as e:
        log(f"A single {gb:>4} GB: FAIL {str(e).splitlines()[0][:110]}")
# B. cumulative 256 MB chunks
held, total_mb = [], 0
while total_mb < 16 * 1024:
    try:
        held.append(torch.empty(256 << 20, dtype=torch.uint8).pin_memory()); total_mb += 256
    except Exception as e:
        log(f"B cumulative 256 MB chunks: FAIL after {total_mb/1024:.2f} GB: {str(e).splitlines()[0][:110]}")
        break
else:
    log(f"B cumulative 256 MB chunks: {total_mb/1024:.0f} GB pinned at once, no failure (stopped by the cap)")
del held
# C. the UVA view: card writes through the view, CPU reads it back
try:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    cpu = torch.zeros(64 << 20, dtype=torch.float32).pin_memory()          # 256 MB
    view = get_accelerator_view_from_cpu_tensor(cpu)
    view.fill_(3.0); view.add_(1.0); torch.xpu.synchronize()
    ok = bool((cpu == 4.0).all())
    log(f"C UVA view (card fills pinned host memory, CPU reads 4.0 back): {'ok' if ok else 'WRONG'}  view.device={view.device}")
    # and a real matmul reading weights from host memory, against the same weights on the card
    w_cpu = torch.randn(4096, 4096, dtype=torch.float16).pin_memory()
    w_view = get_accelerator_view_from_cpu_tensor(w_cpu)
    x = torch.randn(16, 4096, dtype=torch.float16, device="xpu")
    y_host = x @ w_view.T; y_card = x @ w_cpu.to("xpu").T; torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(20): y_host = x @ w_view.T
    torch.xpu.synchronize(); t_host = (time.time()-t0)/20
    w_dev = w_cpu.to("xpu"); t0 = time.time()
    for _ in range(20): y_card = x @ w_dev.T
    torch.xpu.synchronize(); t_card = (time.time()-t0)/20
    log(f"C matmul 16x4096x4096 fp16, weights in host memory via UVA: {t_host*1e3:.2f} ms ({32/1e9/t_host:.1f} GB/s of weights); same weights on the card: {t_card*1e3:.3f} ms; max |diff| {(y_host-y_card).abs().max().item():.4f}")
except Exception as e:
    log(f"C UVA view: FAIL {str(e).splitlines()[0][:160]}")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
