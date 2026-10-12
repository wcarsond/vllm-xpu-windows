# pull-model.input.py - download a Hugging Face model into C:\Users\claude\models-hf\<folder> (safetensors + configs only).
# Usage:  .venv312\Scripts\python.exe server\pull-model.input.py Qwen/Qwen2.5-14B-Instruct [folder]
#         (folder defaults to the repo's last path part; progress and the final size go to server\pull-model.output.txt)
# Why (2026-10-11, Carson: "pull what you need, we should have space"): the two-card path has only run the 7B; the point of
# two cards is a model that does not fit one. Qwen2.5-14B-Instruct in bf16 is ~29.5 GB: too big for one 24 GB B60, right
# for two. Models live on C: (F: is the slow spinning disk). Resumable: rerunning skips finished files.
import datetime, pathlib, sys, time

from huggingface_hub import snapshot_download

repo = sys.argv[1]
folder = sys.argv[2] if len(sys.argv) > 2 else repo.split("/")[-1]
dest = pathlib.Path(r"C:\Users\claude\models-hf") / folder
out = pathlib.Path(__file__).with_name("pull-model.output.txt")
t0 = time.time()
print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} pulling {repo} -> {dest}", flush=True)
snapshot_download(repo, local_dir=str(dest),
                  allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt", "vocab.json", "tokenizer*", "*.md"],
                  max_workers=4)
size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} done: {repo} -> {dest}, {size/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min"
print(line, flush=True)
with out.open("a", encoding="utf-8") as f:
    f.write(line + "\n")
