# ze-info-summary.input.py - one line per kernel from a zebin .ze_info dump: GRF count, spill, private, SLM, barriers, simd.
# Usage:  .venv312\Scripts\python.exe kernels\ze-info-summary.input.py <path to .ze_info>     (prints; inspect-attn-dll.input.cmd calls it)
# Why (2026-10-10): the YAML is thousands of lines; the questions are only "did 256 GRF apply" and "which kernels spill how much".
import re, sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
# Kernel blocks start with "- name:" under "kernels:"; fields of interest sit in execution_env.
blocks = re.split(r"\n\s*-\s+name:\s*", text)
rows = []
for b in blocks[1:]:
    name = b.splitlines()[0].strip()
    def f(key):
        m = re.search(rf"^\s*{key}:\s*(\S+)", b, re.M)
        return m.group(1) if m else "-"
    rows.append((name, f("grf_count"), f("spill_size"), f("private_size"), f("slm_size"), f("simd_size"), f("has_barriers"), f("required_work_group_size")))
print(f"   {len(rows)} kernels   grf  spill     private   slm     simd barriers")
for name, grf, spill, priv, slm, simd, bar, _ in rows:
    short = name if len(name) <= 70 else name[:34] + "..." + name[-33:]
    print(f"   {grf:>4} {spill:>8} {priv:>9} {slm:>7} {simd:>4} {bar:>4}  {short}")
grfs = sorted({r[1] for r in rows})
spills = [int(r[2]) for r in rows if r[2].isdigit()]
print(f"   grf values seen: {grfs};  spill bytes/thread max {max(spills) if spills else 0}, kernels spilling: {sum(1 for s in spills if s > 0)} of {len(rows)}")
