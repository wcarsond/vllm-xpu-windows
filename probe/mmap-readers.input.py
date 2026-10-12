# mmap-readers.input.py - can N processes memory-map and read the same safetensors shards at once on this box?
# Usage:  .venv312\Scripts\python.exe probe\mmap-readers.input.py <model-folder> [readers] [passes]
#         e.g.  ... mmap-readers.input.py Qwen2.5-72B-Instruct-AWQ 2 1      (writes probe\mmap-readers.output.txt)
# Why (2026-10-11 15:15): loading Qwen2.5-72B-Instruct-AWQ with tensor_parallel_size=2 kills a worker with
# 0xC0000005 (access violation). PYTHONFAULTHANDLER put the crash in
#   torch/storage.py __getitem__  <-  weight_utils.safetensors_weights_iterator  (f.get_tensor(name))
# i.e. reading tensor bytes out of the memory-mapped checkpoint. The SAME files read by ONE process are fine: the
# one-card run read all 11 shards and only then failed with a clean torch OutOfMemoryError on the card. Ruled out
# already: host RAM at the start, the commit limit, AWQ group/TP divisibility, and shard size (Qwen3.5-27B has a
# 5.3 GB shard and loads; the 72B's largest is 4.0 GB).
# So the remaining difference is TWO processes mapping the same files at the same time. This strips everything else
# away: no vLLM, no GPU, no torch device work - just N processes doing safetensors get_tensor over every shard.
# If this faults, the problem is the OS/driver/filesystem under concurrent mapping and vLLM is innocent.
# If it does not, the fault needs something vLLM does as well, and the next suspect is the device copy alongside it.
import os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("mmap-readers.output.txt")


_KEPT = []
_HELD = [0]


def read_all(folder: pathlib.Path, tag: str, passes: int) -> int:
    from safetensors import safe_open
    shards = sorted(folder.glob("*.safetensors"))
    total = 0
    for p in range(passes):
        for i, st in enumerate(shards, 1):
            with safe_open(str(st), framework="pt") as f:
                for ti, name in enumerate(f.keys()):  # noqa: SIM118
                    # MMAP_DISJOINT=1: reader r takes only tensors with index % 2 == r, so the two
                    # readers never touch the same file pages - a pipeline split's access pattern.
                    # Default: both read everything - a tensor split's. Host RAM at the vLLM
                    # fault was 14.5 GB free with 2.8 GB working sets (18:45), so the question
                    # is no longer pressure but whether SHARING the pages is the trigger.
                    if os.environ.get("MMAP_DISJOINT") and ti % 2 != int(tag[-1]):
                        continue
                    t = f.get_tensor(name)          # the call that faulted
                    # Touch one byte per 4 KB page so every page of the mapping is really
                    # faulted in. Reading only element 0 (the first version of this probe)
                    # finished "41.6 GB" in 1.5 s, which is the giveaway that mmap had
                    # handed back addresses nobody had read yet.
                    import torch
                    flat = t.reshape(-1).view(torch.uint8) if t.dtype != torch.uint8 else t.reshape(-1)
                    total += int(flat[::4096].sum().item() != 0)
                    # MMAP_DEVICE=<index>: also copy the tensor to that card, which is the one
                    # thing vLLM's loader does that reading alone does not. Two readers reading
                    # 41.6 GB concurrently was clean (15:13), so if adding the device copy
                    # reproduces 0xC0000005 the fault is the driver touching host memory that
                    # is a file mapping, not the mapping or the concurrency by itself.
                    dev = os.environ.get("MMAP_DEVICE")
                    if dev is not None:
                        # MMAP_CLONE=1 copies from ordinary heap memory instead of straight
                        # out of the file mapping. Same bytes, same volume, same two cards:
                        # the only difference is whether the driver's source pointer is a
                        # mapped file. That is what separates "this driver cannot copy from
                        # a file mapping" from "this driver cannot take this much traffic".
                        src = t.clone() if os.environ.get("MMAP_CLONE") else t
                        if os.environ.get("MMAP_COPY_INTO"):
                            # What vLLM actually does: copy_ INTO a pre-allocated device
                            # tensor from the mapped file, rather than .to() which
                            # allocates a fresh destination. Neither earlier probe had
                            # both halves - the shard reader used .to() on mapped memory,
                            # the strided test used copy_ on heap memory. 16:20.
                            dst = torch.empty(src.shape, dtype=src.dtype, device=f"xpu:{dev}")
                            dst.copy_(src)
                            # MMAP_RETAIN=1 KEEPS it, as a loader does. This is the one cell
                            # the probe matrix never filled: mapped source AND copy_ into a
                            # preallocated device tensor AND the result retained. 16:35.
                            # MMAP_RETAIN_HALF=1 keeps every second tensor only: the mapping
                            # covers the whole checkpoint but ~20 GB stays on the card, which is
                            # vLLM's tensor split exactly (every tensor read, half kept, no
                            # overflow - measured 19.55 of 23.33 GiB at 17:20).
                            # MMAP_RETAIN_CAP_GB=<n> keeps tensors on the card until n GiB are
                            # held, then frees the rest. The mapping still covers the whole
                            # checkpoint. n=20 is vLLM's tensor split (19.55 GiB measured, no
                            # overflow) and faults; sweeping n downward finds the threshold,
                            # which decides whether loading can run in parallel up to it and
                            # serialize only the remainder (Carson's look-ahead carry, 17:25).
                            cap = os.environ.get("MMAP_RETAIN_CAP_GB")
                            if cap is not None:
                                if _HELD[0] < float(cap) * 2**30:
                                    _KEPT.append(dst)
                                    _HELD[0] += dst.numel() * dst.element_size()
                                else:
                                    del dst
                            elif os.environ.get("MMAP_RETAIN"):
                                _KEPT.append(dst)
                            else:
                                del dst
                        else:
                            g = src.to(f"xpu:{dev}", non_blocking=False)
                            del g
                        del src
            if os.environ.get("MMAP_TRIM_WS"):
                # Mitigation candidate: give the file pages back after each shard so the
                # process never holds a large file-backed working set beside its device
                # allocations. If the fault needs both, this should prevent it.
                import ctypes
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            print(f"[{tag}] pass {p+1} shard {i}/{len(shards)} ok", flush=True)
    return total


if __name__ == "__main__":
    if os.environ.get("MMAP_CHILD"):
        folder = pathlib.Path(sys.argv[1])
        t0 = time.time()
        try:
            read_all(folder, os.environ["MMAP_CHILD"], int(sys.argv[2]))
            print(f"[{os.environ['MMAP_CHILD']}] DONE in {time.time()-t0:.1f} s", flush=True)
        except Exception as e:
            print(f"[{os.environ['MMAP_CHILD']}] EXCEPTION {type(e).__name__}: {e}", flush=True)
            sys.exit(3)
        sys.exit(0)

    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-72B-Instruct-AWQ"
    readers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    folder = pathlib.Path(r"C:\Users\claude\models-hf") / model
    if not folder.is_dir():
        print(f"no model at {folder}")
        sys.exit(2)
    shards = sorted(folder.glob("*.safetensors"))
    gb = sum(s.stat().st_size for s in shards) / 1e9
    lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  {model}  {len(shards)} shards, {gb:.1f} GB  "
             f"{readers} concurrent readers x {passes} pass(es)"]
    print(lines[0], flush=True)

    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"
    procs = []
    t0 = time.time()
    for r in range(readers):
        e = dict(env); e["MMAP_CHILD"] = f"reader{r}"
        # one card per reader, as vLLM gives one card per tensor-parallel worker
        if os.environ.get("MMAP_DEVICE_PER_READER"):
            e["MMAP_DEVICE"] = str(r)
            # MMAP_NO_MASK=1 leaves ZE_AFFINITY_MASK unset so each reader sees both cards
            # and selects one, exactly as a vLLM worker does. Setting the mask is itself
            # enough to fault this driver (copy-storm), so with the mask on, this probe
            # was only ever reproducing THAT bug, not vLLM's. 2026-10-11 16:10.
            if not os.environ.get("MMAP_NO_MASK"):
                e["ZE_AFFINITY_MASK"] = str(r)
        procs.append(subprocess.Popen([sys.executable, str(HERE), str(folder), str(passes)],
                                      env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    for r, pr in enumerate(procs):
        out, _ = pr.communicate()
        tail = [l for l in out.splitlines() if l.strip()][-6:]
        code = pr.returncode
        verdict = "OK" if code == 0 else f"FAILED exit {code}" + (
            f" (0x{code & 0xFFFFFFFF:08X} = ACCESS VIOLATION)" if code & 0xFFFFFFFF == 0xC0000005 else
            f" (0x{code & 0xFFFFFFFF:08X})" if code not in (0, 3) else "")
        lines.append(f"reader{r}: {verdict}")
        lines += ["    " + l for l in tail]
        print(lines[-1 - len(tail)], flush=True)
    lines.append(f"wall {time.time()-t0:.1f} s")
    print(lines[-1], flush=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
