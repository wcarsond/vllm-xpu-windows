# many-allocs.input.py - does a small-M F.linear still launch once the process holds many separate device allocations?
# Usage:  .venv312\Scripts\python.exe probe\many-allocs.input.py      (writes probe\many-allocs.output.txt)
# Why (2026-10-10): the failing oneDNN gemm_kernel launch in vLLM had USM_INDIRECT_ACCESS set, which makes Level Zero
# reference every USM allocation of the process in that submission; Windows' GPU memory manager (WDDM) limits how many
# allocations one submission may reference. A plain probe holds a handful of allocations and passes; vLLM holds hundreds
# (one per weight tensor, KV blocks). This grows the count in steps and runs the M=13 bf16 GEMM after each step. The
# torch caching allocator pools small tensors into segments, so distinct 4 MiB tensors are used to force one Level Zero
# allocation each (the allocator's large-block path).
import datetime, pathlib, torch, torch.nn.functional as F

out = pathlib.Path(__file__).with_name("many-allocs.output.txt")
dev = torch.device("xpu:0"); dt = torch.bfloat16
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}  free {torch.xpu.mem_get_info(0)[0]/2**30:.1f} GiB"]
w = torch.randn(896, 896, dtype=dt, device=dev) * 0.02
b = torch.randn(896, dtype=dt, device=dev) * 0.02
x = torch.randn(13, 896, dtype=dt, device=dev)
held = []
def gemm_ok():
    try:
        y = F.linear(x, w, b); torch.xpu.synchronize(); return "ok"
    except Exception as e:
        try: torch.xpu.synchronize()
        except Exception: pass
        return f"FAIL {type(e).__name__}: {str(e)[:90]}"
lines.append(f"held {len(held):5d} allocations: {gemm_ok()}")
for target in (100, 250, 500, 1000, 1500, 2000, 3000, 4000):
    while len(held) < target:
        held.append(torch.empty(4 * 2**20 // 2, dtype=dt, device=dev))   # 4 MiB each, distinct
    r = gemm_ok()
    lines.append(f"held {len(held):5d} allocations ({len(held)*4/1024:.1f} GiB): {r}")
    print(lines[-1], flush=True)
    if r.startswith("FAIL"):
        break
text = "\n".join(lines); print(text); out.write_text(text + "\n", encoding="utf-8")
