# oneproc-profile.input.py - where does a decode step's card time go? torch's profiler (CPU + XPU kernels) around one
# 64-token greedy generation, after a warm-up, with the engine in this process so the profiler sees every card.
# Prints the top kernels by total card time, with call counts and time per call.
# Usage:  .venv312\Scripts\python.exe server\oneproc-profile.input.py <oneproc|one> [model folder]
#         oneproc = two cards, one-process executor; one = one card (uni executor). Full decode graphs either way.
#         (appends to server\oneproc-profile.output.txt)
# Why (2026-10-13): gpt-oss-20b decodes at 14 ms/token in one process against 10 ms with two processes and 11 ms on one
# card, while each rank's host work is only ~1.9 ms (VLLM_XPU_ONEPROC_TIMING): the extra time is on the cards.
import os, pathlib, sys, time

MODE = sys.argv[1] if len(sys.argv) > 1 else "oneproc"
MODEL = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\claude\models-hf\gpt-oss-20b"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
if MODE == "one":
    os.environ["ZE_AFFINITY_MASK"] = "0"
else:
    os.environ.pop("ZE_AFFINITY_MASK", None)
OUT = pathlib.Path(__file__).with_name("oneproc-profile.output.txt")


def say(s):
    print(s, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def run():
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams

    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, max_num_batched_tokens=2048, max_num_seqs=16,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    # PROF_EAGER=1: no graphs - the profiler does not see kernels inside a replayed graph, so per-kernel card times of
    # the decode step itself need an eager run (its wall time is not comparable; its kernel times are)
    if os.environ.get("PROF_EAGER") == "1":
        kw.pop("compilation_config")
        kw["enforce_eager"] = True
    if MODE == "oneproc":
        kw.update(tensor_parallel_size=2, distributed_executor_backend="vllm.v1.executor.oneproc_executor.OneProcExecutor")
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
    prompt = "Write a Python function that parses an INI file into a nested dict."
    llm.generate([prompt], sp)
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        llm.generate([prompt], sp)
    wall = time.time() - t0
    rows = []
    for e in prof.key_averages():
        dev = getattr(e, "device_time_total", 0) or getattr(e, "xpu_time_total", 0)
        if dev > 0 and e.device_type is not None and "xpu" in str(e.device_type).lower():
            rows.append((dev, e.count, e.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    eager = " EAGER" if os.environ.get("PROF_EAGER") == "1" else ""
    say(f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  {os.path.basename(MODEL)}, {MODE}{eager}: 64 tokens in {wall:.2f} s "
        f"({64 / wall:.1f} tok/s incl. prefill); card time recorded {total / 1e3:.1f} ms in total, all cards")
    for dev, n, key in rows[:18]:
        say(f"  {dev / 1e3:9.2f} ms  {n:6d} calls  {dev / n:8.1f} us/call  {key[:90]}")


if __name__ == "__main__":
    import ctypes, traceback
    code = 0
    try:
        run()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(-1), code)   # see oneproc-smoke.input.py: terminate()
