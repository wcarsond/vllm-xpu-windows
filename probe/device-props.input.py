# device-props.input.py - everything torch and the OpenCL side of the same driver report about the B60s: SLM limit, work-group limits, driver version.
# Usage:  .venv312\Scripts\python.exe probe\device-props.input.py      (writes probe\device-props.output.txt)
# Why (2026-10-10): attention kernels fail at launch with OUT_OF_RESOURCES and no allocation; the driver returns that code when a
# kernel's SLM request exceeds the device's local memory size. The kernels were compiled for Linux's idea of the B60; this prints
# Windows's idea, so the two can be compared number for number.
import datetime, pathlib, subprocess, torch

out = pathlib.Path(__file__).with_name("device-props.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}"]
for i in range(torch.xpu.device_count()):
    p = torch.xpu.get_device_properties(i)
    lines.append(f"-- xpu:{i}")
    for a in sorted(dir(p)):
        if not a.startswith("_"):
            try:
                lines.append(f"   {a} = {getattr(p, a)}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"   {a} = <{e}>")
# The compiler's sycl-ls sees the same driver through OpenCL; --verbose prints "Local memory size" and work-group limits.
sycl_ls = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\sycl-ls.exe"
try:
    r = subprocess.run([sycl_ls, "--verbose"], capture_output=True, text=True, timeout=120)
    keep = ("Name", "Driver", "Local memory", "Max work group", "Max compute units", "Sub-group", "Global mem", "Max mem alloc", "Platform", "Version", "Device [")
    lines.append("-- sycl-ls --verbose (lines of interest)")
    lines.extend("   " + l.strip() for l in r.stdout.splitlines() if any(k in l for k in keep))
    if r.stderr.strip():
        lines.append("   stderr: " + r.stderr.strip().splitlines()[-1][:200])
except Exception as e:  # noqa: BLE001
    lines.append(f"sycl-ls failed: {e}")
text = "\n".join(lines); print(text); out.write_text(text + "\n", encoding="utf-8")
