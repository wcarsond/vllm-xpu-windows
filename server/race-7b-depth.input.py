# race-7b-depth.input.py - vLLM side of the depth leg: Coder-7B bf16 on one B60 at 0/4K/8K/16K/32K tokens of context.
# Usage:  .venv312\Scripts\python.exe server\race-7b-depth.input.py      (writes server\race-7b-depth.output.txt; ~10 min)
#         SMOKE_ATTN_BACKEND=TRITON_ATTN|FLASH_ATTN picks the attention kernels; run once each after the cutlass rebuild.
# Why (2026-10-10, Carson: "vulkan stays fast at depth, or faster"): the one-card race was a 21-token prompt; attention cost
# grows with context, and that is where the kernels separate. Per depth: exact prompt tokens, prefill tok/s (a 1-token
# generate times the prefill alone), decode tok/s at that depth (128 tokens, prefill time subtracted). Prefix caching OFF so
# the second run cannot reuse the first; the first run at each depth is a warm-up (Triton compiles per new shape).
import datetime, os, pathlib, sys, time

# RACE_TP=2 runs the same leg on both cards (tensor split, GPU-timeline exchange). RACE_CHUNK is max_num_batched_tokens:
# 32768 on one card (one prefill pass); on two cards it defaults to 2048 so every all_reduce of a prefill chunk
# (tokens x 3584 x 2 B) fits the exchange's 16 MB slot instead of falling back to the gloo relay (2026-10-11 10:30).
TP = int(os.environ.get("RACE_TP", "1"))
# RACE_PP=2: the pipeline split instead (half the layers on each card, one activation hop per stage boundary, no
# all_reduce); its prefill should sit near one card's at every depth. Files carry a -pp2 suffix.
PP = int(os.environ.get("RACE_PP", "1"))
CHUNK = int(os.environ.get("RACE_CHUNK", "32768" if TP == 1 else "2048"))
if TP == 1 and PP == 1:
    os.environ.setdefault("ZE_AFFINITY_MASK", "0")
else:
    os.environ.pop("ZE_AFFINITY_MASK", None)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# SMOKE_MODEL / SMOKE_DTYPE / SMOKE_GPU_UTIL (2026-10-11 13:40): another model than the 7B, e.g. the 32B-AWQ in float16 on two
# cards; files then carry the model folder name after the split suffix, so no table overwrites another.
model = os.environ.get("SMOKE_MODEL", r"C:\Users\claude\models-hf\Qwen2.5-Coder-7B-Instruct")
DTYPE = os.environ.get("SMOKE_DTYPE", "bfloat16")
MODEL_TAG = "-" + os.path.basename(os.path.normpath(model)) if "SMOKE_MODEL" in os.environ else ""
task = "Write a Python function that parses an INI file into a nested dict, with comments explaining each step."
para = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report on regional "
        "logistics, inventory turnover, and the maintenance schedule for the conveyor system in building four. ")
# The model's max_position_embeddings is 32768 (vLLM refuses a longer max_model_len), so the deepest leg is 32000 tokens
# of prompt plus 128 generated inside that window.
DEPTHS = [0, 4096, 8192, 16384, 32000]
# RACE_DEPTHS="4096,16384,32000" runs only those depths; RACE_TAG=<text> adds a suffix to the output file name, so a sweep
# keeps one table per setting (2026-10-13, as on the Ubuntu side's copy of this script)
if os.environ.get("RACE_DEPTHS"):
    DEPTHS = [int(d) for d in os.environ["RACE_DEPTHS"].split(",")]
SUFFIX = ("-tp2" if TP == 2 else "-pp2" if PP == 2 else "") + MODEL_TAG + (("-" + os.environ["RACE_TAG"]) if os.environ.get("RACE_TAG") else "")
SHAPE = "two cards, tensor split" if TP == 2 else "two cards, pipeline split" if PP == 2 else "one card"
out = pathlib.Path(__file__).with_name(f"race-7b-depth{SUFFIX}.output.txt")
backend = os.environ.get("SMOKE_ATTN_BACKEND", "default")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  model {model}  {SHAPE}, "
         f"{DTYPE}, attention {backend}, prefix caching off, max_num_batched_tokens {CHUNK}"]

def log(s):
    print(s, flush=True); lines.append(s)

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(model)
    extra = {}
    if backend != "default":
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        extra["attention_backend"] = AttentionBackendEnum[backend]
    t0 = time.time()
    if TP > 1 or PP > 1:
        extra["tensor_parallel_size"] = TP
        extra["pipeline_parallel_size"] = PP
        extra["distributed_executor_backend"] = "mp"
    util = float(os.environ.get("SMOKE_GPU_UTIL", "0.9" if TP == 1 and PP == 1 else "0.85"))
    # SMOKE_MAX_LEN (2026-10-12): a smaller window for a model whose KV cannot hold 32K on one card; deeper depths are skipped
    MAXLEN = int(os.environ.get("SMOKE_MAX_LEN", "32768"))
    llm = LLM(model=model, dtype=DTYPE, max_model_len=MAXLEN, gpu_memory_utilization=util,
              enforce_eager=os.environ.get("SMOKE_EAGER", "1") != "0",   # SMOKE_EAGER=0: torch.compile (+ graphs where allowed)
              enable_prefix_caching=False, max_num_batched_tokens=CHUNK, **extra)
    log(f"engine up in {time.time()-t0:.1f} s ({os.path.basename(os.path.normpath(model))}, {DTYPE}, max_model_len {MAXLEN}, "
        f"{SHAPE}, chunk {CHUNK}, util {util})")
    for depth in DEPTHS:
        if depth + 200 > MAXLEN:
            log(f"depth {depth:>5}: skipped, window {MAXLEN}"); continue
        if depth == 0:
            prompt = task
        else:
            # grow the filler until the whole prompt tokenizes to >= depth, then trim to the exact count
            filler = para
            while len(tok(f"Here is a document:\n{filler}\nNow, {task[0].lower()}{task[1:]}")["input_ids"]) < depth:
                filler += para * 8
            ids = tok(f"Here is a document:\n{filler}")["input_ids"][:depth - 40]
            prompt = tok.decode(ids) + f"\nNow, {task[0].lower()}{task[1:]}"
        n_prompt = len(tok(prompt)["input_ids"])
        one = SamplingParams(temperature=0.0, max_tokens=1)
        many = SamplingParams(temperature=0.0, max_tokens=128)
        llm.generate([prompt], one); llm.generate([prompt], many)          # warm-up at this depth
        t1 = time.time(); llm.generate([prompt], one); t_prefill = time.time() - t1
        t2 = time.time(); o = llm.generate([prompt], many)[0]; t_total = time.time() - t2
        n_out = len(o.outputs[0].token_ids)
        decode = (n_out - 1) / max(t_total - t_prefill, 1e-6)
        log(f"depth {depth:>5}: prompt_n {n_prompt:>5}, prefill {n_prompt/t_prefill:8.1f} tok/s ({t_prefill:.2f} s), "
            f"decode {decode:6.1f} tok/s over {n_out} tokens, wall {t_total:.2f} s")
    log("last answer starts: " + o.outputs[0].text.strip().replace("\n", " ")[:120])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
