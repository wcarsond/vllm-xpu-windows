# smoke-pp2.input.py - two cards: Qwen2.5-Coder-7B-Instruct split pipeline-style across both B60s on Windows.
# Usage:  .venv312\Scripts\python.exe server\smoke-pp2.input.py      (writes server\smoke-pp2.output.txt)
# Why (2026-10-10, rung 6): the first two-card run. pipeline_parallel_size=2 puts half the layers on each card; the
# hand-off between them goes through the Windows RAM relay (xpu_ram_relay.py, measured 0.13 ms per token). Same prompt
# and budget as the one-card race so the number is comparable: the honest expectation is roughly one-card speed with
# twice the room, since each token still walks every layer. Both cards must be visible (no ZE_AFFINITY_MASK); the two
# worker processes pick xpu:0 and xpu:1 by local rank. Run only after smoke-offline.input.py has passed on one card.
import datetime, os, pathlib, sys, time

os.environ.pop("ZE_AFFINITY_MASK", None)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
model = r"C:\Users\claude\models-hf\Qwen2.5-Coder-7B-Instruct"
prompt = "Write a Python function that parses an INI file into a nested dict, with comments explaining each step."
out = pathlib.Path(__file__).with_name("smoke-pp2.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  model {model}  pipeline_parallel_size=2"]

def log(s):
    print(s, flush=True); lines.append(s)

if __name__ == "__main__":
    import torch
    log(f"torch {torch.__version__}, cards: " + ", ".join(torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())))
    from vllm import LLM, SamplingParams
    # SMOKE_ATTN_BACKEND=TRITON_ATTN|FLASH_ATTN: explicit attention backend for A/B runs (see smoke-offline.input.py).
    extra = {}
    if (backend := os.environ.get("SMOKE_ATTN_BACKEND")):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        extra["attention_backend"] = AttentionBackendEnum[backend]
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, enforce_eager=True,
              pipeline_parallel_size=2, distributed_executor_backend="mp", **extra)
    log(f"engine up on two cards in {time.time()-t0:.1f} s (attention {os.environ.get('SMOKE_ATTN_BACKEND', 'default')})")
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
