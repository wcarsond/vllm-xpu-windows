# pull-gemma4-big.input.py - the two big Gemma 4 GGUFs with their vision projectors (see pull-gemma4-big.input.cmd).
import datetime, pathlib, time
from huggingface_hub import hf_hub_download

root = pathlib.Path(r"C:\Users\claude\models-b60")
pulls = [
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf", "gemma4-26b-a4b"),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "mmproj-F16.gguf",                   "gemma4-26b-a4b"),
    ("unsloth/gemma-4-31B-it-GGUF",     "gemma-4-31B-it-UD-Q4_K_XL.gguf",     "gemma4-31b"),
    ("unsloth/gemma-4-31B-it-GGUF",     "mmproj-F16.gguf",                   "gemma4-31b"),
]
out = pathlib.Path(__file__).with_name("pull-model.output.txt")
for repo, fname, folder in pulls:
    t0 = time.time()
    print(f"{datetime.datetime.now():%H:%M:%S} pulling {repo}/{fname} -> {root / folder}", flush=True)
    p = pathlib.Path(hf_hub_download(repo, fname, local_dir=str(root / folder)))
    line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} done: {repo}/{fname} -> {p}, "
            f"{p.stat().st_size/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min")
    print(line, flush=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
