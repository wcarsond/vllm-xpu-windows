"""xpu-probe.input.py - the call. Run with the project venv:

    C:\\Users\\claude\\scripts\\vllm-xpu-windows\\.venv\\Scripts\\python.exe xpu-probe.input.py

Writes xpu-probe.output.txt beside this file.
"""
import datetime
import importlib.util
import pathlib
import sys

here = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("xpu_probe", here / "xpu-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

import torch

lines = []


def say(text):
    print(text)
    lines.append(text)


say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  python {sys.version.split()[0]}  torch {torch.__version__}")

cards = probe.devices()
say(f"cards PyTorch can use: {len(cards)}")
for index, name in cards:
    say(f"  xpu:{index}  {name}  {probe.memory_gb(index):.1f} GB")

for index, name in cards:
    seconds = probe.matmul_seconds(f"xpu:{index}")
    say(f"matmul 4096x4096 float16 on xpu:{index}: {seconds * 1000:.2f} ms")

# Sizes that matter for splitting one model across two cards (hidden size ~5120, float16):
# one token = ~10 KB, a 512-token batch = ~5 MB, an 8192-token prompt = ~84 MB.
if len(cards) >= 2:
    for label, nbytes in [("10 KB (1 token)", 10 * 1024), ("5 MB (512 tokens)", 5 * 1024**2), ("84 MB (8192 tokens)", 84 * 1024**2)]:
        for src, dst in [("xpu:0", "xpu:1"), ("xpu:1", "xpu:0")]:
            seconds, intact = probe.relay_seconds(src, dst, nbytes)
            rate = nbytes / seconds / 1024**2
            say(f"relay {label:20s} {src} -> RAM -> {dst}: {seconds * 1000:8.3f} ms  {rate:8.0f} MB/s  intact={intact}")
else:
    say("relay test skipped: needs two cards")

with open(here / "xpu-probe.output.txt", "a", encoding="utf-8") as out:  # append: every run stays on record
    out.write("\n".join(lines) + "\n\n")
