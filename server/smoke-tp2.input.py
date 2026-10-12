# smoke-tp2.input.py - two cards, tensor split: Qwen2.5-Coder-7B-Instruct with half of every layer on each B60 (tensor_parallel_size=2).
# Usage:  .venv312\Scripts\python.exe server\smoke-tp2.input.py      (writes server\smoke-tp2.output.txt)
# Why (2026-10-10): the pipeline split (smoke-pp2) buys room, not speed: 26.0 tok/s against 29.4 on one card. A tensor split
# halves the compute per card but needs 56 all_reduces per token on this model; through the gloo relay that is ~15 ms and
# the split only breaks even, through the shared-memory exchange ~2-3 ms and the target is ~50 tok/s. Same prompt and
# budget as the one-card race so the three numbers (1 card, PP2, TP2) sit side by side. Both cards visible, Triton attention.
import datetime, os, pathlib, sys, time

os.environ.pop("ZE_AFFINITY_MASK", None)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# SMOKE_MODEL overrides the model (2026-10-11 03:55: the 0.5B is enough to check that the two workers exit cleanly,
# and it fits in RAM beside the MoE compile; the 7B is the speed number).
model = os.environ.get("SMOKE_MODEL", r"C:\Users\claude\models-hf\Qwen2.5-Coder-7B-Instruct")
prompt = "Write a Python function that parses an INI file into a nested dict, with comments explaining each step."
# 2026-10-11 13:55: a model other than the 7B gets its folder name in the output file (smoke-tp2-<model>.output.txt) so
# results sit side by side; SMOKE_MAX_LEN trims the context window for a model that only just fits (the 72B-AWQ).
MODEL_TAG = "-" + os.path.basename(os.path.normpath(model)) if "SMOKE_MODEL" in os.environ else ""
MAX_LEN = int(os.environ.get("SMOKE_MAX_LEN", "4096"))
out = pathlib.Path(__file__).with_name(f"smoke-tp2{MODEL_TAG}.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  model {model}  tensor_parallel_size=2  exchange {os.environ.get('VLLM_XPU_EXCHANGE', 'gloo relay')}"]

def log(s):
    print(s, flush=True); lines.append(s)

if __name__ == "__main__":
    import torch
    log(f"torch {torch.__version__}, cards: " + ", ".join(torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())))
    from vllm import LLM, SamplingParams
    extra = {}
    if (backend := os.environ.get("SMOKE_ATTN_BACKEND")):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        extra["attention_backend"] = AttentionBackendEnum[backend]
    t0 = time.time()
    # 0.70, not 0.85: each card holds half the weights (7.5 GB), so 70% of 23.3 GB leaves ~8 GB of KV cache and still
    # starts when the router's 3.7 GB model is idling on a card (03:11 the 0.85 ask refused with 19.2 GiB free).
    # SMOKE_EAGER=0 lets vLLM compile the model (torch.compile / Inductor on XPU): the two-card decode is now bounded by the
    # Python thread's kernel launch rate, and fewer launches is the only lever left (2026-10-11 11:20).
    eager = os.environ.get("SMOKE_EAGER", "1") != "0"
    if not eager and os.environ.get("SMOKE_INDUCTOR_DEBUG") == "1":
        # a device synchronize after every generated kernel: the first faulting kernel then raises at its own launch,
        # with its name, instead of poisoning the queue for whoever synchronizes next (the first all_reduce, 10:58)
        extra["compilation_config"] = {"inductor_compile_config": {"triton.debug_sync_kernel": True}}
    # SMOKE_DTYPE: bfloat16 by default; AWQ/GPTQ checkpoints are fp16 (2026-10-11, the 32B-AWQ on the oneDNN int4 GEMMs)
    dtype, util = os.environ.get("SMOKE_DTYPE", "bfloat16"), float(os.environ.get("SMOKE_GPU_UTIL", "0.70"))
    llm = LLM(model=model, dtype=dtype, max_model_len=MAX_LEN,
              gpu_memory_utilization=util, enforce_eager=eager,
              tensor_parallel_size=2, distributed_executor_backend="mp", **extra)
    log(f"engine up on two cards (tensor split) in {time.time()-t0:.1f} s ({os.path.basename(os.path.normpath(model))}, {dtype}, "
        f"max_model_len {MAX_LEN}, util {util}, attention {os.environ.get('SMOKE_ATTN_BACKEND', 'default')}, eager {eager})")
    params = SamplingParams(temperature=0.0, max_tokens=256)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=16))   # warm-up
    rates = []
    for i in range(3):
        t1 = time.time()
        o = llm.generate([prompt], params)[0]
        dt = time.time() - t1
        n_in, n_out = len(o.prompt_token_ids), len(o.outputs[0].token_ids)
        rates.append(n_out / dt)
        log(f"run {i+1}: {n_in} in, {n_out} out, {dt:.2f} s -> {n_out/dt:.1f} tok/s end to end")
    log(f"median {sorted(rates)[1]:.1f} tok/s; first 200 chars: {o.outputs[0].text.strip()[:200]!r}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
