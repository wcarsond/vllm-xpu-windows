# hf-forward.input.py - the same Qwen2.5-0.5B model through plain transformers on xpu:0, no vLLM: every real GEMM shape, real weights.
# Usage:  .venv312\Scripts\python.exe probe\hf-forward.input.py      (writes probe\hf-forward.output.txt)
# Why (2026-10-10): vLLM's first real step dies at an oneDNN GEMM launch (OUT_OF_DEVICE_MEMORY) that passes in every
# isolated probe, but the probes never used two of this model's shapes (down_proj K=4864 -> N=896, qkv K=896 -> N=1152).
# transformers runs the identical layers with the identical weights through torch alone. Pass: the fault is vLLM-specific.
# Fail: it is torch/oneDNN on this Windows driver for a real shape, and the failing layer is printed.
import datetime, pathlib, time, torch

out = pathlib.Path(__file__).with_name("hf-forward.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}"]
model_dir = r"C:\Users\claude\models-hf\Qwen2.5-0.5B-Instruct"
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("xpu:0").eval()
lines.append(f"loaded; {sum(p.numel() for p in model.parameters())/1e6:.0f} M params on xpu:0")
msgs = [{"role": "user", "content": "Count from 1 to 40, separated by spaces."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt")["input_ids"].to("xpu:0")
lines.append(f"prompt tokens: {ids.shape[1]}")
try:
    with torch.inference_mode():
        t0 = time.perf_counter(); logits = model(ids).logits; torch.xpu.synchronize()
        lines.append(f"forward (prefill, M={ids.shape[1]}): ok in {(time.perf_counter()-t0)*1000:.1f} ms, logits {tuple(logits.shape)}")
        t1 = time.perf_counter(); gen = model.generate(ids, max_new_tokens=96, do_sample=False); torch.xpu.synchronize()
        dt = time.perf_counter() - t1; n = gen.shape[1] - ids.shape[1]
        lines.append(f"generate: {n} tokens in {dt:.2f} s = {n/dt:.1f} tok/s (transformers, eager, no KV paging)")
        lines.append("text: " + tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")[:400])
except Exception as e:
    lines.append(f"FAIL: {type(e).__name__}: {str(e)[:300]}")
text = "\n".join(lines); print(text); out.write_text(text + "\n", encoding="utf-8")
