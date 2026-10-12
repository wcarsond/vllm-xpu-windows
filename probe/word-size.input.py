"""word-size.input.py - which number formats does the B60 compute in natively?

    C:\\Users\\claude\\scripts\\vllm-xpu-windows\\.venv\\Scripts\\python.exe word-size.input.py

JavaScript numbers are 64-bit floats. This checks whether the card can do 64-bit float and 64-bit
integer math, or only 32-bit. Appends to word-size.output.txt beside this file.
"""
import datetime
import pathlib

import torch

here = pathlib.Path(__file__).parent
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}"]

for name, dtype, value in [
    ("float32", torch.float32, 16_777_217.0),          # first whole number a 32-bit float cannot hold
    ("float64", torch.float64, 7_515_755_912.5),       # World record tour length, plus a half
    ("int32", torch.int32, 2_000_000_000),
    ("int64", torch.int64, 7_515_755_912),
]:
    try:
        x = torch.tensor([value], dtype=dtype, device="xpu:0")
        back = (x + 1 - 1)[0].item()
        lines.append(f"{name:8s} on card: ok, {value!r} came back as {back!r}, exact={back == value}")
    except Exception as error:  # the point of the probe is to record what fails
        lines.append(f"{name:8s} on card: FAILED: {str(error).splitlines()[0][:160]}")

print("\n".join(lines))
with open(here / "word-size.output.txt", "a", encoding="utf-8") as out:
    out.write("\n".join(lines) + "\n\n")
