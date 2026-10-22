# concurrency-sweep-candidates.input.py - read-only check of Hugging Face 4-bit builds of a newer Qwen 27B before pulling one
# for concurrency-sweep.input.py: architecture, quantization config and size against the working Qwen3.5-27B-GPTQ-Int4.
# Usage:  .venv312\Scripts\python.exe server\concurrency-sweep-candidates.input.py repo1 [repo2 ...]
#         QUANT_FULL=1 also prints each full quantization_config (which layers stay unquantized)
#         (prints; appends to server\concurrency-sweep-candidates.output.txt)
# Why (2026-10-21, Carson: "yeah let's use newer models if we can"): Qwen publishes Qwen3.6-27B only as bf16 and FP8, so a
# community 4-bit build has to be picked, and it must be the same family and quant format as the 27B that already runs.
import datetime, json, pathlib, sys

from huggingface_hub import HfApi, hf_hub_download

OUT = pathlib.Path(__file__).with_name("concurrency-sweep-candidates.output.txt")
BASE = pathlib.Path(r"F:\models-hf\Qwen3.5-27B-GPTQ-Int4\config.json")


def summary(cfg):
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    tc = cfg.get("text_config", {})
    return (f"arch {cfg.get('architectures')} model_type {cfg.get('model_type')} layers {tc.get('num_hidden_layers', cfg.get('num_hidden_layers'))} "
            f"layer_types {sorted(set(tc.get('layer_types', [])))} max_pos {tc.get('max_position_embeddings', cfg.get('max_position_embeddings'))} | quant "
            + json.dumps({k: q.get(k) for k in ('quant_method', 'bits', 'group_size', 'sym', 'desc_act', 'checkpoint_format', 'packing_format', 'format')}))


_b = json.loads(BASE.read_text(encoding="utf-8"))
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "BASE Qwen3.5-27B-GPTQ-Int4: " + summary(json.loads(BASE.read_text(encoding="utf-8")))]
if "QUANT_FULL" in __import__("os").environ:
    lines.append("BASE quantization_config: " + json.dumps(_b.get("quantization_config")))
api = HfApi()
for repo in sys.argv[1:]:
    try:
        info = api.model_info(repo, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
        cfg = json.loads(pathlib.Path(hf_hub_download(repo, "config.json")).read_text(encoding="utf-8"))
        files = [s.rfilename for s in info.siblings]
        extra = [f for f in files if f in ("quantize_config.json", "quantization_config.json", "chat_template.jinja", "preprocessor_config.json")]
        if "QUANT_FULL" in __import__("os").environ:
            lines.append(f"{repo} quantization_config: " + json.dumps(cfg.get("quantization_config")))
        lines.append(f"{repo}: {size/1e9:.1f} GB safetensors, modified {info.last_modified:%Y-%m-%d}, downloads {info.downloads}, files {extra} | " + summary(cfg))
    except Exception as e:  # noqa: BLE001
        lines.append(f"{repo}: FAILED {type(e).__name__}: {e}"[:300])
for s in lines:
    print(s, flush=True)
with OUT.open("a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
