# determinism-test.input.py - does the served model give the same answer to the same request every time? The 7B on two
# cards (one-process executor, full decode graphs), one prompt sent 20 times, greedy, 128 tokens, with the chosen token's
# log-probability: counts distinct token sequences and the largest log-probability difference between runs, and times
# decode. Run once as is and once with VLLM_XPU_DETERMINISTIC=1 (torch's deterministic algorithms in the workers).
# Usage:  .venv312\Scripts\python.exe server\determinism-test.input.py      (appends to server\determinism-test.output.txt)
# Why (2026-10-13): probe/graph-determinism found the fake decode step differing from itself in the last bf16 bit in
# ~7 runs of 100 (eager and graphs alike; 0 of 100 with deterministic algorithms). Whether that reaches a served answer
# is measured here, on vLLM's own output.
import os, pathlib, sys, time

os.environ.pop("ZE_AFFINITY_MASK", None)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
OUT = pathlib.Path(__file__).with_name("determinism-test.output.txt")
RUNS = int(os.environ.get("DET_RUNS", "20"))
PROMPT = ("Write a Python function that parses an INI file into a nested dict, with comments explaining each step. "
          "Then explain how you would test it.")


def say(s):
    print(s, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def run():
    from vllm import LLM, SamplingParams

    llm = LLM(model=r"C:\Users\claude\models-hf\Qwen2.5-Coder-7B-Instruct", dtype="bfloat16", tensor_parallel_size=2,
              distributed_executor_backend="vllm.v1.executor.oneproc_executor.OneProcExecutor", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_prefix_caching=False, max_num_batched_tokens=2048, max_num_seqs=16,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    sp = SamplingParams(temperature=0.0, max_tokens=128, logprobs=0, ignore_eos=True)
    llm.generate([PROMPT], sp)
    seqs, lps, times = [], [], []
    for _ in range(RUNS):
        t0 = time.time()
        o = llm.generate([PROMPT], sp)[0].outputs[0]
        times.append(time.time() - t0)
        seqs.append(tuple(o.token_ids))
        lps.append([next(iter(d.values())).logprob for d in o.logprobs])
    distinct = len(set(seqs))
    ref = seqs[0]
    first_div = min((next((k for k, (a, b) in enumerate(zip(ref, s)) if a != b), None) for s in seqs if s != ref),
                    default=None)
    same_len = [l for s, l in zip(seqs, lps) if s == ref]
    lp_diff = max(max(abs(a - b) for a, b in zip(same_len[0], l)) for l in same_len)
    lp_runs = sum(1 for l in same_len if l != same_len[0])
    say(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  deterministic={os.environ.get('VLLM_XPU_DETERMINISTIC', '0')}: {RUNS} runs, "
        f"{distinct} distinct token sequences (first divergence at token {first_div}); among runs with the first run's "
        f"tokens, {lp_runs} have different log-probabilities (largest difference {lp_diff:.3g}); "
        f"{128 / (sum(times) / len(times)):.1f} tok/s average incl. prefill")


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
