# bisect-tp.input.py - cut the FAILING case down instead of building a passing one up.
# Usage:  .venv312\Scripts\python.exe probe\bisect-tp.input.py [case ...]   (writes probe\bisect-tp.output.txt)
# Why (2026-10-11 16:25, Carson: "random is the same method as binary search but with better methods ... i cut till it
# breaks, put back, cut more, until i have it"): ten hand-built probes all passed, because building upward from a
# minimal case only ever samples the space randomly. The failure is a known point - Qwen2.5-72B-AWQ, tensor split, two
# cards, load lock off, 0xC0000005 every time - so bisect from THERE by removing one thing at a time.
# Each case is a whole vLLM engine in its own process, so a fault kills only that case. Ordered as cuts:
#   full      the known failure, as a control
#   dummy     same shapes, same device allocations, NO file reading at all   <- cuts the whole loader out
#   gloo      the host-RAM relay instead of the GPU exchange                 <- cuts our exchange out
#   nolock    == full (the lock is off everywhere here; kept for symmetry)
# A case that still faults keeps the cut; a case that stops faulting names the half that matters.
import os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("bisect-tp.output.txt")
MODEL = os.environ.get("BISECT_MODEL", r"C:\Users\claude\models-hf\Qwen2.5-72B-Instruct-AWQ")

SNIP = """
import os, sys
from vllm import LLM, SamplingParams
kw = dict(model=r'{model}', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.97,
          enforce_eager=True, tensor_parallel_size=2, distributed_executor_backend='mp')
{extra}
llm = LLM(**kw)
print('ENGINE UP', flush=True)
o = llm.generate(['Count to five.'], SamplingParams(temperature=0.0, max_tokens=8))[0]
print('CHILD OK', repr(o.outputs[0].text[:30]))
"""

# LAYERS=<n>: load only the first n of the 72B's 80 layers, via hf_overrides. Same files, same
# loader, same tensor split, same quantization - only the amount read and held changes. This is the
# knob to binary search on (16:30): 80 faults, so halve until it stops.
if os.environ.get("BISECT_LAYERS"):
    CASES = {"layers": (f"first {os.environ['BISECT_LAYERS']} layers only",
                        "kw['hf_overrides'] = {'num_hidden_layers': %s}" % os.environ["BISECT_LAYERS"], {})}
else:
    CASES = {}
CASES.update({
    "full":   ("the known failure, unchanged", "", {}),
    "dummy":  ("dummy weights: no file reading at all", "kw['load_format'] = 'dummy'", {}),
    "gloo":   ("host-RAM relay instead of the GPU exchange", "", {"VLLM_XPU_EXCHANGE": "gloo"}),
})


def run(name: str) -> str:
    why, extra, env = CASES[name]
    e = dict(os.environ)
    e.update(env)
    e["VLLM_XPU_NO_LOAD_LOCK"] = "1"          # the lock is what hides the fault; off for all cases
    e["PYTHONFAULTHANDLER"] = "1"
    e["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    e.pop("ZE_AFFINITY_MASK", None)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", SNIP.format(model=MODEL, extra=extra)],
                       env=e, capture_output=True, text=True)
    blob = p.stdout + p.stderr
    code = p.returncode
    av = (code & 0xFFFFFFFF) == 0xC0000005
    got_up = "ENGINE UP" in blob
    if code == 0:
        verdict = "OK"
    elif av or "died before signalling READY: exit code 3221225477" in blob:
        verdict = "FAULT 0xC0000005" + ("  (after engine up)" if got_up else "  (during load)")
    else:
        last = [l for l in blob.splitlines() if l.strip()][-1:]
        verdict = f"other exit {code}: {last[0][:70] if last else ''}"
    line = f"{name:<7} {why:<46} {verdict}   [{time.time()-t0:.0f} s]"
    print(line, flush=True)
    return line


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(CASES)
    lines = [f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  model {pathlib.Path(MODEL).name}  "
             f"tensor_parallel_size=2, load lock OFF, no affinity mask"]
    print(lines[0], flush=True)
    for name in wanted:
        lines.append(run(name))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
