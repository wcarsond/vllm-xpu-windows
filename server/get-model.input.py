# get-model.input.py - fetch the smoke-test model into its own folder on C: (F: is the slow spinning disk).
# Usage:  .venv312\Scripts\python.exe server\get-model.input.py      (writes server\get-model.output.txt)
# Why (2026-10-10, Carson's OK 17:38): first light needs a dense Llama/Qwen-family model in HF safetensors (the default
# kernel preset covers head size 128 only; GGUF is not the path here). Qwen2.5-0.5B-Instruct is ~1 GB and is a test
# bulb, not a model choice. Downloaded files are data from the internet: they get a folder of their own.
# Second model (Carson, 17:45 "do what you need to do"): Qwen2.5-Coder-7B-Instruct, bf16 ~15 GB. Same weights as his
# router's qwen2.5-coder-7b (Q4/Q8 GGUF), so it races vLLM against llama.cpp on the same card. Already-complete
# folders are skipped by snapshot_download (it checks sizes), so re-running fetches only what is missing.
import datetime, pathlib, time
from huggingface_hub import snapshot_download

repos = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"]
root = pathlib.Path(r"C:\Users\claude\models-hf")
out = pathlib.Path(__file__).with_name("get-model.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"]
for repo in repos:
    dest = root / repo.split("/")[1]
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = snapshot_download(repo, local_dir=str(dest), allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"])
    secs = time.time() - t0
    files = sorted(p for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts)
    total = sum(p.stat().st_size for p in files)
    lines.append(f"{repo} -> {path}: {len(files)} files, {total/1e9:.2f} GB, {secs:.0f} s ({total/1e6/max(secs,1):.0f} MB/s)")
    lines += [f"  {p.relative_to(dest)}  {p.stat().st_size:,}" for p in files]
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
