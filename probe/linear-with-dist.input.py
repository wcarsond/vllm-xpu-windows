# linear-with-dist.input.py - F.linear after the pieces of vLLM's worker setup: gloo process group, device set, a stream.
# Usage:  .venv312\Scripts\python.exe probe\linear-with-dist.input.py      (writes probe\linear-with-dist.output.txt)
# Why (2026-10-10): the small-M bf16 GEMM launches in a plain process and fails inside vLLM's engine core. vLLM's worker
# does three things a plain script does not before its first forward: torch.distributed.init_process_group("gloo") with
# world size 1 (Windows has no xccl), torch.xpu.set_device, and work on torch.xpu streams. Each is added in turn.
import datetime, os, pathlib, torch, torch.nn.functional as F, torch.distributed as dist

out = pathlib.Path(__file__).with_name("linear-with-dist.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}"]
dt = torch.bfloat16

def gemm(dev, label):
    w = torch.randn(896, 896, dtype=dt, device=dev) * 0.02; b = torch.randn(896, dtype=dt, device=dev); x = torch.randn(13, 896, dtype=dt, device=dev)
    for variant, fn in (("no bias", lambda: F.linear(x, w)), ("bias", lambda: F.linear(x, w, b))):
        try:
            fn(); torch.xpu.synchronize(); lines.append(f"ok    {label} / {variant}")
        except Exception as e:
            lines.append(f"FAIL  {label} / {variant}: {type(e).__name__}: {str(e)[:100]}")

dev = torch.device("xpu:0")
gemm(dev, "1 plain")
torch.xpu.set_device(dev)
gemm(dev, "2 after torch.xpu.set_device")
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29519", rank=0, world_size=1)
gemm(dev, "3 after gloo process group (world 1)")
s = torch.xpu.Stream(device=dev)
with torch.xpu.stream(s):
    gemm(dev, "4 on a non-default torch.xpu.Stream")
with torch.inference_mode():
    gemm(dev, "5 under inference_mode")
dist.destroy_process_group()
text = "\n".join(lines); print(text); out.write_text(text + "\n", encoding="utf-8")
