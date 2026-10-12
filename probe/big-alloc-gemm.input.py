# big-alloc-gemm.input.py - does one huge device allocation break the next oneDNN GEMM on this Windows driver? Each case in its own process.
# Usage:  .venv312\Scripts\python.exe probe\big-alloc-gemm.input.py      (writes probe\big-alloc-gemm.output.txt)
# Why (2026-10-10): vLLM main allocates the whole KV cache as ONE tensor (v1/worker/utils.py allocate_kv_cache: 9.54 GiB
# for the 0.5B smoke). The first real step then dies at the one launch that has USM_INDIRECT_ACCESS (oneDNN gemm_kernel),
# which makes the driver pin every allocation at once. Every GEMM probe so far passed, but none had a 9.5 GiB tensor alive.
# Cases: GEMM alone; one 3.5 GiB tensor; one 5 GiB (over 4 GiB); one 9.5 GiB; 24 x 400 MiB (same total, small pieces).
# A case runs in a child process so one driver fault cannot colour the next case.
import datetime, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("big-alloc-gemm.output.txt")
GIB = 1 << 30
CASES = {
    "A_gemm_alone":        [],
    "B_one_3.5GiB":        [int(3.5 * GIB)],
    "C_one_5GiB":          [5 * GIB],
    "D_one_9.5GiB":        [int(9.5 * GIB)],
    "E_24x400MiB":         [400 << 20] * 24,
    "F_one_9.5GiB_empty":  ["empty", int(9.5 * GIB)],
}


def gemm(label):
    import torch
    x = torch.randn(13, 896, dtype=torch.bfloat16, device="xpu:0")
    w = torch.randn(896, 896, dtype=torch.bfloat16, device="xpu:0")
    y = torch.nn.functional.linear(x, w)
    torch.xpu.synchronize()
    return f"{label}: gemm ok, sum {float(y.float().sum()):.1f}"


def run_case(name):
    import torch
    plan = CASES[name]
    keep = []
    out = []
    out.append(gemm("before"))
    mode = "zeros"
    for item in plan:
        if item == "empty":
            mode = "empty"
            continue
        t = (torch.empty if mode == "empty" else torch.zeros)(item, dtype=torch.int8, device="xpu:0")
        keep.append(t)
    torch.xpu.synchronize()
    held = sum(t.numel() for t in keep) / GIB
    out.append(f"holding {len(keep)} tensor(s), {held:.2f} GiB ({mode}); torch reports allocated {torch.xpu.memory_allocated(0)/GIB:.2f} GiB")
    out.append(gemm("after"))
    if mode == "empty":
        keep[0].zero_(); torch.xpu.synchronize()
        out.append(gemm("after fill"))
    return out


if len(sys.argv) > 1:
    try:
        print("\n".join(run_case(sys.argv[1])))
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    sys.exit(0)

lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"]
try:
    import torch
    p = torch.xpu.get_device_properties(0)
    lines.append(f"device {p.name}, total {p.total_memory/GIB:.2f} GiB, torch {torch.__version__}")
    for attr in ("max_mem_alloc_size", "max_global_mem_alloc_size"):
        if hasattr(p, attr):
            lines.append(f"  {attr} = {getattr(p, attr)/GIB:.2f} GiB")
except Exception as e:  # noqa: BLE001
    lines.append(f"device query failed: {e}")
for name in CASES:
    r = subprocess.run([sys.executable, str(HERE), name], capture_output=True, text=True, timeout=600)
    err = r.stderr.strip().splitlines()
    body = r.stdout.strip() or (err[-1] if err else "(no output)")
    lines.append(f"== {name} (exit {r.returncode})")
    lines.extend("   " + l for l in body.splitlines())
text = "\n".join(lines)
print(text)
OUT.write_text(text + "\n", encoding="utf-8")
