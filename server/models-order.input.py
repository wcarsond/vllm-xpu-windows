# models-order.input.py - set the `order` and `about` keys of every model in models.input.ini (where it sits in VS Code's
# model list and the one line shown after its id), from the table below. Leaves every other key and comment alone.
# Usage:  .venv312\Scripts\python.exe server\models-order.input.py     then  scripts\vscode-models\vscode-models.input.ps1
# Why (2026-10-14, Carson: "can you sort the models json"): the list was alphabetical, the 80Bs mixed in with the 4Bs.
# Order: coders, big general models, Granite, vision, small; older duplicate copies last. Speeds are decode tok/s
# measured through the service on 2026-10-13 (~ = estimated from a short answer that includes prompt time).
import pathlib, re

INI = pathlib.Path(__file__).with_name("models.input.ini")
TABLE = [
    ("qwen3-coder-next",       "80B coder (MoE), the pick for code · 53 tok/s · 64K"),
    ("qwen3-next-80b",         "80B generalist (MoE) · 70 tok/s · 64K"),
    ("qwen3-coder-30b-q4",     "30B coder (MoE), fastest big model · 98 tok/s · 96K"),
    ("qwen3.6-35b-a3b",        "35B newest MoE, thinks · 71 tok/s · 128K"),
    ("gemma4-26b-a4b",         "Gemma 4 26B MoE (4B active), vision, thinks · ~60 tok/s · 64K"),   # 2026-10-16
    ("qwen3.6-27b",            "27B newest dense, vision, thinks · 18 tok/s · 64K"),               # projector added 2026-10-16
    ("gemma4-31b",             "Gemma 4 31B dense, vision, thinks · ~16 tok/s · 64K"),            # 2026-10-16
    ("qwen3.5-27b",            "27B dense, vision · 29 tok/s · 64K"),
    ("qwen3.5-27b-thinking",   "27B dense, vision, thinks · 29 tok/s · 64K"),
    ("granite-4.2-30b",        "Granite 30B, 4-bit · 44 tok/s · 32K"),
    ("granite-4.2-30b-q8",     "Granite 30B, 8-bit (best quality) · ~13 tok/s · 32K"),
    ("granite4-32b-a9b",       "Granite 32B MoE · ~25 tok/s · 128K"),
    ("qwen2.5-72b",            "72B · 20 tok/s · 2K context, chat only"),
    ("qwen2.5-32b",            "32B · 42 tok/s · 32K (slow first load from F:)"),
    ("qwen3-vl-8b",            "8B vision specialist · 37 tok/s · 32K"),
    ("qwen3.5-9b",             "9B vision, thinks · 35 tok/s · 64K"),
    ("gemma4-e4b",             "4B vision, thinks · ~70 tok/s · 32K"),
    ("qwen3.5-4b",             "4B small and fast · 87 tok/s · 64K"),
    ("qwen2.5-coder-7b",       "7B coder · 57 tok/s · 32K"),
    ("omnicoder-9b",           "9B coder, thinks · 52 tok/s · 64K"),
    ("mistral-small-3.2-24b",  "24B, creative writing · 14 tok/s · 64K"),
    ("qwen36-35b-a3b",         "older 4-bit copy of qwen3.6-35b-a3b · 62 tok/s · 128K"),
    ("qwen36-35b-a3b-nothink", "older copy, thinking off · ~48 tok/s · 128K"),
    ("qwen-coder-30b",         "older 3-bit copy of qwen3-coder-30b · 96K"),
    ("granite42-30b",          "older llama.cpp copy of granite-4.2-30b · 17 tok/s · 32K"),
    ("qwen2.5-coder-14b",      "14B coder, no tool calls · 20 tok/s · 32K"),
    ("qwen2.5-coder-7b-q8",    "7B coder 8-bit, no tool calls · 33 tok/s · 32K"),
]

text = INI.read_text(encoding="utf-8")
# drop any order/about lines from an earlier run, then add fresh ones right after each section header
text = re.sub(r"^(order|about) = .*\n", "", text, flags=re.M)
missing = []
for i, (mid, about) in enumerate(TABLE, 1):
    pat = re.compile(r"^\[" + re.escape(mid) + r"\]\n", re.M)
    if not pat.search(text):
        missing.append(mid)
        continue
    text = pat.sub(lambda m: f"{m.group(0)}order = {i}\nabout = {about}\n", text, count=1)
INI.write_text(text, encoding="utf-8")
listed = set(re.findall(r"^\[([^\]]+)\]", text, re.M)) - {"router"}
print(f"ordered {len(TABLE) - len(missing)} models; not in the ini: {missing}; in the ini but not ordered: "
      f"{sorted(listed - {m for m, _ in TABLE})}")
