"""card-free.input.py - read how much memory is free on each B60, straight from the cards. Takes a few seconds,
allocates nothing on the cards, and exits.

    C:\\Users\\claude\\scripts\\vllm-xpu-windows\\.venv\\Scripts\\python.exe card-free.input.py

Why (2026-10-02): Windows' GPU memory counters read 0 on both B60s, so they cannot say whether llama-server's
loaded model is on a card or in system RAM. The cards themselves can.
"""
import torch

for card in range(torch.xpu.device_count()):
    free, total = torch.xpu.mem_get_info(card)
    print(f"xpu:{card}  {torch.xpu.get_device_name(card)}  used {(total - free) / 2**30:5.1f} GB of {total / 2**30:5.1f} GB, "
          f"free {free / 2**30:5.1f} GB")
