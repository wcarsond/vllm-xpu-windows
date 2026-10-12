# d2h-blocks.input.py - can the copy engine read every block the XPU caching allocator hands out? One card, no vLLM.
# Usage:  .venv312\Scripts\python.exe probe\d2h-blocks.input.py [expandable]      (writes probe\d2h-blocks.output.txt)
# Why (2026-10-11 11:45): under torch.compile on two cards the relay's copy-to-host of one tensor fails with
# UR_RESULT_ERROR_UNKNOWN while a device-side clone of the same tensor works; the tensor sits at the first byte of a fresh
# 32 MB allocator block (ptr ...12000000) and tensors from other blocks copy fine. So: walk the allocator through many fresh
# blocks (allocate and keep growing tensors), and for each new block try a kernel read (clone), a copy-engine D2H (.cpu()),
# and a kernel D2H (index_select into pinned memory is still a copy; use a tiny SYCL-free stand-in: .to('cpu') of a view).
# With "expandable" the allocator runs with expandable segments, the mode Inductor-style pools may use.
import datetime, os, pathlib, sys

if len(sys.argv) > 1 and sys.argv[1] == "expandable":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

out = pathlib.Path(__file__).with_name("d2h-blocks.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}  PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF', '(unset)')}"]

def log(s):
    print(s, flush=True); lines.append(s)

if __name__ == "__main__":
    dev = torch.device("xpu", 0)
    keep = []
    seen_blocks = set()
    fails = 0
    # sizes that force new segments: 1 MB, 20 MB, 40 MB, 128 MB, then many 3.5 MB ones
    sizes_mb = [1, 20, 40, 128] + [3.5] * 24
    for i, mb in enumerate(sizes_mb):
        n = int(mb * 1024 * 1024 // 2)
        t = torch.empty(n, dtype=torch.bfloat16, device=dev)
        t.fill_(float(i + 1))
        keep.append(t)
        block = t.data_ptr() >> 25   # 32 MB granule
        new = block not in seen_blocks
        seen_blocks.add(block)
        head = t[:512 * 3584] if n >= 512 * 3584 else t
        torch.xpu.synchronize()
        try:
            c = head.clone(); torch.xpu.synchronize(); clone = "clone ok"
        except Exception as e:  # noqa: BLE001
            clone = f"clone FAILED {type(e).__name__}"
        try:
            h = head.cpu(); ok = bool((h == float(i + 1)).all()); cpu = "cpu ok" if ok else "cpu WRONG DATA"
        except Exception as e:  # noqa: BLE001
            cpu = f"cpu FAILED {str(e)[:60]}"; fails += 1
        log(f"alloc {i:2d} {mb:6.1f} MB ptr {t.data_ptr():#x} {'NEW block' if new else 'same block'}: {clone}; {cpu}")
    log(f"{fails} failures over {len(sizes_mb)} allocations, {len(seen_blocks)} distinct 32 MB granules")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
