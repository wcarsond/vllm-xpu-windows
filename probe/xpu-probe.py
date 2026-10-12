"""xpu-probe.py - what PyTorch on native Windows can do with the Intel cards.

First rung of porting vLLM's Intel (XPU) path to Windows: can PyTorch see both B60s, do
half-precision math on each, and how long does a card -> system RAM -> other card copy take.
That copy is the simple stand-in for oneCCL (Intel's Linux-only multi-card library).

No side effects on import. The call is xpu-probe.input.py; results go to xpu-probe.output.txt.
"""
import time

import torch


def devices():
    """[(index, name)] for every Intel card PyTorch can use."""
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return []
    return [(i, torch.xpu.get_device_name(i)) for i in range(torch.xpu.device_count())]


def memory_gb(index):
    """Total card memory PyTorch reports, in GB."""
    return torch.xpu.get_device_properties(index).total_memory / 1024**3


def matmul_seconds(dev, size=4096, reps=10):
    """Seconds for one size x size half-precision matrix multiply on dev."""
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    (a @ b).sum().item()  # warm up
    torch.xpu.synchronize(dev)
    start = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    torch.xpu.synchronize(dev)
    c[0, 0].item()  # read one number back, so the timing cannot stop before the card has finished
    return (time.perf_counter() - start) / reps


def relay_seconds(src, dst, nbytes, reps=50):
    """Seconds to move nbytes of float16 from src card to system RAM to dst card, and check it arrived intact."""
    x = torch.randn(nbytes // 2, device=src, dtype=torch.float16)
    y = x.cpu().to(dst)  # warm up
    torch.xpu.synchronize(dst)
    start = time.perf_counter()
    for _ in range(reps):
        y = x.cpu().to(dst)
        y[-1].item()  # read the last number back from dst every time: proves this copy landed
    seconds = (time.perf_counter() - start) / reps
    intact = bool(torch.equal(x.cpu(), y.cpu()))
    return seconds, intact
