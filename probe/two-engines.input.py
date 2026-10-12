# two independent vLLM engines at once: one per card, or both on the same card.
# TWO_ENGINES_SELECT=mask   (default) each child gets ZE_AFFINITY_MASK=<card>  - the Level Zero DRIVER hides the other card
# TWO_ENGINES_SELECT=oneapi each child gets ONEAPI_DEVICE_SELECTOR=level_zero:<card> and NO mask - the SYCL runtime, one
#                           layer above the driver, filters the device list; the driver itself still sees both cards.
# Why (2026-10-11 22:40): the masked pair faults (trigger 1, at any copy size, heap or pinned source); the unmasked pair
# that picks a card in-process is clean at 20 GiB each. Per-item batch work wants one worker per card with no exchange, so
# the question is whether choosing the card above the driver keeps us in the clean cell. PyTorch enumerates XPU devices
# through SYCL, which honours ONEAPI_DEVICE_SELECTOR, so torch should then see one card as xpu:0 with the driver unmasked.
import os, pathlib, subprocess, sys, time
ROOT = pathlib.Path(r"C:\Users\claude\scripts\vllm-xpu-windows")
PY_EXE = ROOT / ".venv312/Scripts/python.exe"
MODEL = r"C:\Users\claude\models-hf\Qwen2.5-0.5B-Instruct"
SNIP = (
    "import os,sys,torch\n"
    "from vllm import LLM, SamplingParams\n"
    "llm = LLM(model=r'%s', dtype='bfloat16', max_model_len=2048, gpu_memory_utilization=0.35, enforce_eager=True)\n"
    "o = llm.generate(['Count to five.'], SamplingParams(temperature=0.0, max_tokens=32))[0]\n"
    "print('CHILD OK mask=%%s sel=%%s ndev=%%d dev0=%%s' %% (os.environ.get('ZE_AFFINITY_MASK'),"
    " os.environ.get('ONEAPI_DEVICE_SELECTOR'), torch.xpu.device_count(), torch.xpu.get_device_name(0)),"
    " repr(o.outputs[0].text[:40]))\n" % MODEL
)
SELECT = os.environ.get("TWO_ENGINES_SELECT", "mask")
def run(label, masks):
    procs = []
    for m in masks:
        e = dict(os.environ); e["PYTHONFAULTHANDLER"] = "1"
        e.pop("ZE_AFFINITY_MASK", None); e.pop("ONEAPI_DEVICE_SELECTOR", None)
        if SELECT == "oneapi":
            e["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{m}"
        else:
            e["ZE_AFFINITY_MASK"] = str(m)
        e["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        procs.append(subprocess.Popen([str(PY_EXE), "-c", SNIP], env=e,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    codes, notes = [], []
    for p in procs:
        out, _ = p.communicate(); codes.append(p.returncode)
        notes += [l for l in out.splitlines() if "CHILD OK" in l or "fatal" in l.lower()][:1]
    bad = [c for c in codes if c != 0]
    v = "OK" if not bad else f"FAULT 0x{bad[0]&0xFFFFFFFF:08X}"
    print(f"{label:<46} {v}   {notes}", flush=True)
print(f"select={SELECT}", flush=True)
run("two 0.5B engines, SAME card (0 and 0)", [0, 0])
run("two 0.5B engines, one card each (0 and 1)", [0, 1])
