# race-7b-vllm.input.py - vLLM side of the 7B race: Qwen2.5-Coder-7B-Instruct bf16 on one B60, decode tokens/s.
# Usage:  .venv312\Scripts\python.exe server\race-7b-vllm.input.py      (writes server\race-7b-vllm.output.txt)
# Why (2026-10-10): the same weights the router serves as qwen2.5-coder-7b (Q4_K_M GGUF, llama.cpp Vulkan). Same prompt,
# same 256-token budget, greedy, three runs after one warm-up; the router side is race-7b-router.input.ps1. Run this
# with the router's model unloaded (it idles out after 10 minutes) so the card is not shared. One card: ZE_AFFINITY_MASK=0.
# Honest labels: vLLM here is bf16 (15 GB), the router is 4-bit (4.4 GB); bytes per token differ by ~3.5x, which
# favours the router on decode. The number that matters is tok/s as each is actually run.
import datetime, os, pathlib, sys, time

os.environ.setdefault("ZE_AFFINITY_MASK", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# SMOKE_MODEL overrides the model (added 2026-10-11 13:12, after two "one-card 14B" runs had quietly raced the 7B instead)
model = os.environ.get("SMOKE_MODEL", r"C:\Users\claude\models-hf\Qwen2.5-Coder-7B-Instruct")
prompt = "Write a Python function that parses an INI file into a nested dict, with comments explaining each step."
out = pathlib.Path(__file__).with_name("race-7b-vllm.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  model {model}  prompt {prompt!r}"]

def log(s):
    print(s, flush=True); lines.append(s)

if __name__ == "__main__":
    import torch
    log(f"torch {torch.__version__}, card: {torch.xpu.get_device_name(0)}")
    from vllm import LLM, SamplingParams
    # SMOKE_ATTN_BACKEND=TRITON_ATTN|FLASH_ATTN: explicit attention backend for A/B runs (see smoke-offline.input.py).
    extra = {}
    if (backend := os.environ.get("SMOKE_ATTN_BACKEND")):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        extra["attention_backend"] = AttentionBackendEnum[backend]
    t0 = time.time()
    # SMOKE_EAGER=0 lets vLLM compile the model (torch.compile / Inductor on XPU); the two-card run with it hung at the
    # first all_reduce inside the compiled graph (2026-10-11), and one card is the control that has no exchange at all.
    eager = os.environ.get("SMOKE_EAGER", "1") != "0"
    # SMOKE_OFFLOAD_GB spills that many GB of weights into pinned host RAM (vLLM's cpu_offload_gb): the one-card way to run a
    # model that does not fit one card, to set against the two-card tensor split (2026-10-11, the 14B).
    offload = float(os.environ.get("SMOKE_OFFLOAD_GB", "0"))
    if offload > 0:
        extra["cpu_offload_gb"] = offload
    # SMOKE_DTYPE / SMOKE_GPU_UTIL as in smoke-tp2 (the one-card comparison for the 32B-AWQ needs float16 and more of the card)
    llm = LLM(model=model, dtype=os.environ.get("SMOKE_DTYPE", "bfloat16"), max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("SMOKE_GPU_UTIL", "0.85")), enforce_eager=eager, **extra)
    log(f"engine up in {time.time()-t0:.1f} s ({os.path.basename(model)}, {os.environ.get('SMOKE_DTYPE', 'bfloat16')}, eager {eager}, max_model_len 4096, "
        f"attention {os.environ.get('SMOKE_ATTN_BACKEND', 'default')}, cpu_offload_gb {offload})")
    params = SamplingParams(temperature=0.0, max_tokens=256)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=16))   # warm-up
    rates = []
    for i in range(3):
        t1 = time.time()
        o = llm.generate([prompt], params)[0]
        dt = time.time() - t1
        n_in, n_out = len(o.prompt_token_ids), len(o.outputs[0].token_ids)
        rates.append(n_out / dt)
        log(f"run {i+1}: {n_in} in, {n_out} out, {dt:.2f} s -> {n_out/dt:.1f} tok/s end to end (prompt+decode)")
    log(f"median {sorted(rates)[1]:.1f} tok/s; first 200 chars: {o.outputs[0].text.strip()[:200]!r}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
