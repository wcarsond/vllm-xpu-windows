# strided-copy.input.py - is a STRIDED host-to-device copy what faults two cards at once?
# Usage:  .venv312\Scripts\python.exe probe\strided-copy.input.py        (writes probe\strided-copy.output.txt)
# Why (2026-10-11 16:15, Carson: "your trying to generate the answer instead of uncover it ... its there"): reading the
# crash's FULL stack instead of its first two frames put the fault under
#   linear.py:1329 load_weights  ->  the weights generator  ->  safetensors get_tensor
# and the code right above that line is what tensor parallel does and pipeline parallel does not:
#     loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
#     param_data.copy_(loaded_weight)            # param_data is ON THE CARD
# i.e. copy INTO a pre-allocated device tensor FROM a narrowed host view. Narrowing dim 0 stays contiguous; narrowing
# dim 1 - which every row-parallel layer does - does not. Earlier probes only ever did `tensor.to(device)`, which
# allocates a fresh destination and may make the source contiguous first, and none of them reproduced the fault.
# Four cases, two processes, one card each, no affinity mask (so this cannot be the separate masked-process bug).
import os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("strided-copy.output.txt")
ROWS, COLS = 8192, 8192          # 128 MiB of float16, the shape class of a 72B weight
ITERS = int(os.environ.get("STRIDED_ITERS", "48"))


def child(mode: str, device: int) -> None:
    import torch
    host = torch.empty(ROWS, COLS, dtype=torch.float16)      # the "checkpoint" tensor
    host.fill_(1.0)
    half = COLS // 2
    hi = os.environ.get("STRIDED_TAG", "p0").endswith("1")
    if mode == "whole":                                      # pipeline parallel: whole tensor
        src_fn = lambda: host
        dst = torch.empty(ROWS, COLS, dtype=torch.float16, device=f"xpu:{device}")
    elif mode == "narrow0":                                  # column-parallel: contiguous slice
        off = (ROWS // 2) if hi else 0
        src_fn = lambda: host.narrow(0, off, ROWS // 2)
        dst = torch.empty(ROWS // 2, COLS, dtype=torch.float16, device=f"xpu:{device}")
    else:                                                    # narrow1 = row-parallel: STRIDED slice
        off = half if hi else 0
        src_fn = lambda: host.narrow(1, off, half)
        dst = torch.empty(ROWS, half, dtype=torch.float16, device=f"xpu:{device}")
    torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        dst.copy_(src_fn())                                  # copy INTO the device tensor
    torch.xpu.synchronize()
    print(f"    [{os.environ['STRIDED_TAG']}] {mode}: {ITERS} copies in {time.time()-t0:.1f} s", flush=True)


def run(mode: str) -> str:
    procs = []
    for i, d in enumerate((0, 1)):
        e = dict(os.environ)
        e["STRIDED_CHILD"] = "1"; e["STRIDED_TAG"] = f"p{i}"
        e["PYTHONFAULTHANDLER"] = "1"
        e.pop("ZE_AFFINITY_MASK", None)                      # as a vLLM worker runs
        procs.append(subprocess.Popen([sys.executable, str(HERE), mode, str(d)],
                                      env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    codes, notes = [], []
    for p in procs:
        out, _ = p.communicate(); codes.append(p.returncode)
        notes += [l for l in out.splitlines() if "copies in" in l]
    bad = [c for c in codes if c != 0]
    if bad:
        c = bad[0] & 0xFFFFFFFF
        v = f"FAULT 0x{c:08X}" + (" (ACCESS VIOLATION)" if c == 0xC0000005 else "")
    else:
        v = "OK"
    line = f"{mode:<9} (two processes, one card each, no mask)   {v}"
    print(line, flush=True)
    for n in notes:
        print(n, flush=True)
    return line


if __name__ == "__main__":
    if os.environ.get("STRIDED_CHILD"):
        child(sys.argv[1], int(sys.argv[2]))
        sys.exit(0)
    lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  {ROWS}x{COLS} fp16 (128 MiB), {ITERS} copies per child",
             "whole   = copy the whole tensor          (what a pipeline split does)",
             "narrow0 = contiguous half, dim 0         (column-parallel)",
             "narrow1 = STRIDED half, dim 1            (row-parallel: o_proj, down_proj)"]
    for l in lines:
        print(l, flush=True)
    for mode in ("whole", "narrow0", "narrow1"):
        lines.append(run(mode))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
