# pin-ceiling.input.py - the hard ceiling on pinned (card-lockable) host memory for one process on this machine.
# Usage:  .venv312\Scripts\python.exe probe\pin-ceiling.input.py      (writes probe\pin-ceiling.output.txt; card 0 only)
# Why (2026-10-11 13:35): pin-pattern.input.py failed after 9.23 GiB of odd-sized pins while 16 GiB of 256 MiB pins passed.
# torch's caching host allocator rounds each pin up to a power of two, so odd sizes lock up to 2x their bytes. Hypothesis:
# the ceiling is Windows' "shared GPU memory" budget, about half of RAM (31.3 GB here -> ~15.6 GiB). Pins 256 MiB chunks
# until the driver refuses; stops at 19 GiB regardless so the machine keeps its head (checks free RAM first).
import os, pathlib, time
os.environ.setdefault("ZE_AFFINITY_MASK", "0")
import ctypes, torch
out = pathlib.Path(__file__).with_name("pin-ceiling.output.txt")
lines = []
def log(s):
    print(s, flush=True); lines.append(s)

class MEMSTAT(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ax", ctypes.c_ulonglong)]
def ram():
    m = MEMSTAT(); m.len = ctypes.sizeof(m); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / 2**30, m.avail / 2**30
total, avail = ram()
log(f"torch {torch.__version__}, card {torch.xpu.get_device_name(0)}, {time.strftime('%Y-%m-%d %H:%M:%S')}, RAM {total:.1f} GiB, free {avail:.1f} GiB")
cap = min(19.0, avail - 3.0)
held, pinned = [], 0.0
while pinned < cap:
    try:
        held.append(torch.empty(256 << 20, dtype=torch.uint8).pin_memory()); pinned += 0.25
    except Exception as e:
        log(f"ceiling: {pinned:.2f} GiB pinned, the next 256 MiB refused: {str(e).splitlines()[0][:100]}")
        log(f"that is {pinned/total*100:.0f}% of RAM (half of RAM would be {total/2:.2f} GiB)")
        break
else:
    log(f"no refusal up to the {cap:.2f} GiB cap")
del held
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
