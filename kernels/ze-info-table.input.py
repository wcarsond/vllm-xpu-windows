# ze-info-table.input.py - every chunk-prefill kernel in the extracted attention images with its policy, its five flags and its full execution_env.
# Usage:  .venv312\Scripts\python.exe kernels\ze-info-table.input.py      (reads kernels\inspect-attn-dll.work\*.dump; writes kernels\ze-info-table.output.txt)
# Why (2026-10-10): attn-direct shows Local=true non-paged kernels run correctly and every Local=false or Paged=true kernel faults
# on its first launch. They sit in the same DLL with the same flags, so the difference must be in the kernel itself. The zebin's
# execution_env is what the driver reads to set the launch up (barriers, stack calls, inline data, walk order, ...). Side by side,
# the fields that only the faulting kernels carry name the suspect. Also prints which flags are compiled at all (the config set).
import pathlib, re, subprocess

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "inspect-attn-dll.work"
OUT = HERE / "ze-info-table.output.txt"
CXXFILT = pathlib.Path(r"C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\compiler\llvm-cxxfilt.exe")

rows = []
for ze in sorted(WORK.glob("img.*.dump/**/*ze_info*")):
    img = ze.parts[len(WORK.parts)].split(".dump")[0]
    text = ze.read_text(encoding="utf-8", errors="replace")
    for block in re.split(r"\n\s*-\s+name:\s*", text)[1:]:
        name = block.splitlines()[0].strip()
        if "Intel_Symbol_Table" in name:
            continue
        m = re.search(r"execution_env:\s*\n((?:\s+\S.*\n)+?)(?=\s*\S+:\s*$|\s*\S+:\s*\n\s+-|\Z)", block + "\n", re.M)
        env = {}
        if m:
            for line in m.group(1).splitlines():
                kv = re.match(r"\s+([A-Za-z_0-9]+):\s*(.*)", line)
                if kv:
                    env[kv.group(1)] = kv.group(2).strip()
        if not env:
            continue  # the second listing of each kernel (functions/payload only) carries no execution_env
        # policy and flags from the mangled kernel type name: ...chunk_policy_head64_b16... and the trailing Lb0ELb1E... run
        pol = re.search(r"\d+(chunk_policy_\w+?)E", name)
        flags = re.findall(r"Lb([01])E", name)
        rows.append((img, pol.group(1) if pol else "?", "".join("t" if f == "1" else "f" for f in flags), name, env))

keys = sorted({k for *_, env in rows for k in env})
lines = [f"{len(rows)} kernels with execution_env across {len({r[0] for r in rows})} images", "keys: " + ", ".join(keys), ""]
lines.append("-- compiled (policy, flags) set:")
for pol, fl in sorted({(r[1], r[2]) for r in rows}):
    lines.append(f"   {pol:<28} {fl}")
lines.append("")
lines.append("-- per kernel (flags = Paged Causal Local Sink LSE; '.' = key absent):")
hdr = f"{'img':<8} {'policy':<28} {'flags':<7} " + " ".join(f"{k[:14]:>14}" for k in keys)
lines.append(hdr)
for img, pol, fl, name, env in sorted(rows, key=lambda r: (r[1], r[2], r[0])):
    lines.append(f"{img:<8} {pol:<28} {fl:<7} " + " ".join(f"{env.get(k, '.')[:14]:>14}" for k in keys))
if CXXFILT.exists():
    lines.append("")
    lines.append("-- one demangled name per (policy, flags) for orientation:")
    seen = set()
    for img, pol, fl, name, env in rows:
        if (pol, fl) in seen:
            continue
        seen.add((pol, fl))
        try:
            d = subprocess.run([str(CXXFILT), name], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            d = f"<{e}>"
        lines.append(f"   {pol} {fl}: {d[:600]}")
text = "\n".join(lines); print(text); OUT.write_text(text + "\n", encoding="utf-8")
