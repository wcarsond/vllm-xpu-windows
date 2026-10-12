# image-kind.input.py - what device code a SYCL executable, DLL or .pyd carries (zebin AOT vs SPIR-V JIT) and, for zebin, how each kernel is compiled.
# Usage:  .venv312\Scripts\python.exe probe\image-kind.input.py <exe-or-dll> [more...]      (prints; also writes probe\image-kind.output.txt)
# Why (2026-10-10): a one-step icx build warned "-Xsycl-target-backend ... unused" yet produced a running program, so it is not
# obvious whether a binary was compiled ahead of time for bmg or will be JIT-compiled by the driver. The image's first bytes say.
# For zebin images, ocloc disasm yields .ze_info: grf_count, simd, slm, barriers, dpas, stack calls, inline data... per kernel,
# which is what the driver reads to set up a launch. Comparing a working binary with a faulting one names the odd field out.
import pathlib, re, subprocess, sys, tempfile

EXTRACT = r"C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\compiler\clang-offload-extract.exe"
OCLOC = r"C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\bin\ocloc.exe"
OUT = pathlib.Path(__file__).with_name("image-kind.output.txt")
SKIP_KEYS = {"actual_kernel_start_offset", "offset_to_skip_per_thread_data_load", "offset_to_skip_set_ffid_gp"}
lines = []


def say(s):
    print(s); lines.append(s)


def ze_info_rows(img: pathlib.Path):
    dump = img.with_suffix(img.suffix + ".dump")
    dump.mkdir(exist_ok=True)
    subprocess.run([OCLOC, "disasm", "-file", str(img), "-dump", str(dump), "-device", "bmg"], capture_output=True, text=True)
    rows = []
    for ze in dump.rglob("*ze_info*"):
        text = ze.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"\n\s*-\s+name:\s*", text)[1:]:
            name = block.splitlines()[0].strip()
            m = re.search(r"execution_env:\s*\n((?:[ \t]+\S.*\n)+)", block + "\n")
            if not m or "Intel_Symbol_Table" in name:
                continue
            env = dict(re.findall(r"^[ \t]+([A-Za-z_0-9]+):[ \t]*(.*?)[ \t]*$", m.group(1), re.M))
            rows.append((name, {k: v for k, v in env.items() if k not in SKIP_KEYS}))
    return rows


for target in sys.argv[1:]:
    t = pathlib.Path(target)
    if not t.exists():
        say(f"{t.name}: not found at {t}")
        continue
    with tempfile.TemporaryDirectory() as td:
        # clang-offload-extract prints raw section names (bytes such as 0x90); cp1252 decoding of that threw in a reader thread
        r = subprocess.run([EXTRACT, f"--output={td}\\img", str(t)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        imgs = sorted(pathlib.Path(td).glob("img.*"), key=lambda p: int(p.suffix[1:]))
        if not imgs:
            tail = (r.stderr or r.stdout).strip().splitlines()
            say(f"{t.name}: no device images ({tail[-1] if tail else 'nothing extracted'})")
            continue
        kinds = {}
        all_rows = []
        for im in imgs:
            head = im.read_bytes()[:4]
            kind = "zebin" if head == b"\x7fELF" else "SPIR-V" if head == b"\x03\x02\x23\x07" else f"unknown {head.hex()}"
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind == "zebin":
                all_rows.extend((im.name, n, env) for n, env in ze_info_rows(im))
        say(f"== {t.name}  ({t.stat().st_size/1e6:.1f} MB): {len(imgs)} device images: " + ", ".join(f"{k} x{v}" for k, v in kinds.items()))
        if all_rows:
            keys = sorted({k for *_, env in all_rows for k in env})
            say(f"   {len(all_rows)} kernels; execution_env keys: {', '.join(keys)}")
            # value distribution per key: shows at a glance what every kernel in this binary shares
            for k in keys:
                vals = {}
                for *_, env in all_rows:
                    vals[env.get(k, "(absent)")] = vals.get(env.get(k, "(absent)"), 0) + 1
                say(f"   {k:<40} " + "  ".join(f"{v}:{n}" for v, n in sorted(vals.items(), key=lambda x: -x[1])[:6]))
            say("   first kernels:")
            for im, n, env in all_rows[:4]:
                say(f"     {im} {n[:90]}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
