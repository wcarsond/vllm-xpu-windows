# smoke-offline.input.py - first light: vLLM on Windows generates tokens on one B60, no HTTP server in the way.
# Usage:  .venv312\Scripts\python.exe server\smoke-offline.input.py      (writes server\smoke-offline.output.txt)
# Why (2026-10-10, rung 5): the offline LLM() path has the fewest moving parts (no uvicorn/winloop, no API process), so
# the first run isolates the engine + kernels on xpu:0. Measured: load time, prompt and generation tokens/s, and the text,
# which must be sensible (counting to 40) or the kernels are wrong even if nothing crashes. One card by ZE_AFFINITY_MASK.
import datetime, os, pathlib, sys, time

os.environ.setdefault("ZE_AFFINITY_MASK", "0")          # one card, the one torch calls xpu:0
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
model = r"C:\Users\claude\models-hf\Qwen2.5-0.5B-Instruct"
out = pathlib.Path(__file__).with_name("smoke-offline.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  python {sys.version.split()[0]}  model {model}"]

def log(s):
    print(s, flush=True); lines.append(s)

if __name__ == "__main__":
    import torch
    log(f"torch {torch.__version__}, xpu devices visible: {torch.xpu.device_count()}, name: {torch.xpu.get_device_name(0)}")
    from vllm import LLM, SamplingParams
    # SMOKE_ATTN_BACKEND=TRITON_ATTN picks vLLM's Triton attention instead of the cutlass flash-attention kernels.
    # The old VLLM_ATTENTION_BACKEND environment variable is ignored by this vLLM (22:33); the engine argument is the way.
    extra = {}
    if (backend := os.environ.get("SMOKE_ATTN_BACKEND")):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        extra["attention_backend"] = AttentionBackendEnum[backend]
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.5, enforce_eager=True, **extra)
    log(f"engine up in {time.time()-t0:.1f} s (enforce_eager, max_model_len 2048, 50% of the card, attention {os.environ.get('SMOKE_ATTN_BACKEND', 'default')})")
    prompts = ["Count from 1 to 40, separated by spaces."]
    params = SamplingParams(temperature=0.0, max_tokens=96)
    t1 = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t1
    o = outs[0]
    n_prompt = len(o.prompt_token_ids); n_gen = len(o.outputs[0].token_ids)
    log(f"generate: {n_prompt} prompt tokens, {n_gen} generated in {dt:.2f} s = {n_gen/dt:.1f} tok/s end to end")
    log("text: " + o.outputs[0].text.strip().replace("\n", " ")[:400])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
