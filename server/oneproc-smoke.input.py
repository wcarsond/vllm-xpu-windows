# oneproc-smoke.input.py - the 7B on both cards through the one-process executor (v1/executor/oneproc_executor.py),
# against the same run through the two-process executor: same prompts, greedy, token ids compared one for one.
# Usage:  .venv312\Scripts\python.exe server\oneproc-smoke.input.py <oneproc|mp> [eager|graphs|compile]
#         (writes server\oneproc-smoke-<executor>-<mode>.output.txt: timings, decode tok/s, and the token ids as JSON)
#         oneproc-smoke.input.cmd runs both executors and compares them.
# Why (2026-10-13): stage 3 of the one-process executor - does vLLM run on it, and give the same tokens.
import faulthandler, json, os, pathlib, sys, time

EXEC = sys.argv[1] if len(sys.argv) > 1 else "oneproc"
MODE = sys.argv[2] if len(sys.argv) > 2 else "eager"
os.environ.pop("ZE_AFFINITY_MASK", None)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# the engine core in this process: one stack trace for everything while the executor is new
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
faulthandler.dump_traceback_later(900, repeat=True)

MODEL = os.environ.get("SMOKE_MODEL", r"C:\Users\claude\models-hf\Qwen2.5-Coder-7B-Instruct")
# SMOKE_MODEL (another model than the 7B) puts the model folder in the file name: oneproc-smoke-<model>-<exec>-<mode>
TAG = (os.path.basename(os.path.normpath(MODEL)) + "-") if "SMOKE_MODEL" in os.environ else ""
OUT = pathlib.Path(__file__).with_name(f"oneproc-smoke-{TAG}{EXEC}-{MODE}.output.txt")
PROMPTS = [
    "Write a Python function that parses an INI file into a nested dict, with comments explaining each step.",
    "Explain in three sentences why the sky is blue.",
]
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    from vllm import LLM, SamplingParams

    backend = "vllm.v1.executor.oneproc_executor.OneProcExecutor" if EXEC == "oneproc" else "mp"
    log(f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  {os.path.basename(MODEL)}, two cards, executor {EXEC}, {MODE}")
    # graphs: whole-step decode graphs with no torch.compile (mode 0; one Dynamo cache shared by two ranks of one process
    # is a separate question); compile: vLLM's default torch.compile + graphs
    extra = {}
    if MODE == "graphs":
        extra["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    t0 = time.time()
    llm = LLM(model=MODEL, dtype=os.environ.get("SMOKE_DTYPE", "bfloat16"), tensor_parallel_size=2, distributed_executor_backend=backend,
              max_model_len=int(os.environ.get("SMOKE_MAX_LEN", "4096")), gpu_memory_utilization=float(os.environ.get("SMOKE_GPU_UTIL", "0.85")),
              enforce_eager=MODE == "eager", enable_prefix_caching=False, max_num_batched_tokens=2048,
              max_num_seqs=int(os.environ.get("SMOKE_MAX_SEQS", "16")), **extra)
    log(f"engine up in {time.time() - t0:.1f} s")
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    llm.generate(PROMPTS[:1], SamplingParams(temperature=0.0, max_tokens=8))   # warm-up
    ids = []
    for p in PROMPTS:
        t1 = time.time()
        o = llm.generate([p], sp)[0]
        dt = time.time() - t1
        n = len(o.outputs[0].token_ids)
        ids.append(list(o.outputs[0].token_ids))
        log(f"prompt {len(o.prompt_token_ids)} tokens -> {n} tokens in {dt:.2f} s ({n / dt:.1f} tok/s incl. prefill): "
            + o.outputs[0].text.strip().replace("\n", " ")[:100])
    # decode rate: one long greedy generation, timed against a 1-token generation of the same prompt
    one = SamplingParams(temperature=0.0, max_tokens=1)
    t1 = time.time(); llm.generate(PROMPTS[:1], one); tp = time.time() - t1
    from vllm.distributed.device_communicators import xpu_oneproc
    b = xpu_oneproc.BATON
    h0, w0 = (b.handoffs, list(b.wait_s)) if b else (0, [0.0, 0.0])
    t2 = time.time(); o = llm.generate(PROMPTS[:1], SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True))[0]
    tt = time.time() - t2
    if b:
        nt = len(o.outputs[0].token_ids)
        log(f"baton over that generation: {(b.handoffs - h0) / nt:.1f} hand-overs per token, waiting per token: rank 0 "
            f"{1e3 * (b.wait_s[0] - w0[0]) / nt:.2f} ms, rank 1 {1e3 * (b.wait_s[1] - w0[1]) / nt:.2f} ms "
            f"(of {1e3 * tt / nt:.2f} ms per token)")
    n = len(o.outputs[0].token_ids)
    log(f"decode {(n - 1) / max(tt - tp, 1e-6):.1f} tok/s over {n} tokens (prefill {tp:.3f} s subtracted)")
    log("ids " + json.dumps(ids))


def terminate(code):
    """End with TerminateProcess, as the two-process executor ends its workers, on every path: with the engine in this
    process, a normal exit runs the attention DLL's unload code, which spins forever in the OpenCL runtime and leaves a
    process no kill reaches (2026-10-13 17:57; multiproc_executor.WorkerProc._hard_exit_if_windows_xpu). After an error
    too (18:19: a failed 27B run sat there for 15 minutes)."""
    import ctypes, logging
    for name in ("vllm", ""):
        for h in logging.getLogger(name).handlers:
            try:
                h.flush()
            except Exception:  # noqa: BLE001
                pass
    sys.stdout.flush(); sys.stderr.flush()
    ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(-1), code)


if __name__ == "__main__":
    # max_num_seqs 16 (SMOKE_MAX_SEQS): the default 256 asks a hybrid model for 256 Mamba cache blocks, which the 27B at
    # util 0.85 does not have (143), and graph capture refuses; one request at a time needs one
    code = 0
    try:
        run()
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    faulthandler.cancel_dump_traceback_later()
    terminate(code)
