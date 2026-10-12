# arch-list.input.py - which card architectures does this torch build know, and what does it call the B60?
# Usage:  .venv\Scripts\python.exe probe\arch-list.input.py     (writes probe\arch-list.output.txt)
# Why (2026-10-10, rung 4): the kernel build compiles ahead-of-time for every name in TORCH_XPU_ARCH_LIST;
# Intel's Dockerfile sets it to bmg-g21 for the B60. Confirm the name this torch uses before setting it,
# because a wrong name means a build for the wrong card and a right-but-wide list means hours of extra compiling.
import datetime, pathlib, torch

out = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}"]
out.append(f"torch.xpu.get_arch_list(): {torch.xpu.get_arch_list()}")
for i in range(torch.xpu.device_count()):
    p = torch.xpu.get_device_properties(i)
    fields = {k: getattr(p, k) for k in dir(p) if not k.startswith('_') and not callable(getattr(p, k))}
    out.append(f"xpu:{i} " + ", ".join(f"{k}={v}" for k, v in fields.items()))
text = "\n".join(out)
print(text)
pathlib.Path(__file__).with_name("arch-list.output.txt").write_text(text + "\n", encoding="utf-8")
