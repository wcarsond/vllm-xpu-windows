# vision-check.input.py - send one picture through the service (port 11434) to a model and print what it says.
# Usage:  .venv312\Scripts\python.exe server\vision-check.input.py [model-id] [image-path]
#         (defaults: qwen3.6-27b and the Muse Spark Open screenshot from 2026-10-15; writes vision-check.output.txt)
# Why (2026-10-16): the 3.6 27B got its vision projector today (mmproj in models.input.ini); a picture it reads correctly
# is the only proof that counts. The right answer for the default image names Muse Spark Open, 816B total / 42B active.
import base64, datetime, json, pathlib, sys, time, urllib.request

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6-27b"
image = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                     r"C:\Users\wcars\.claude\uploads\5172d671-0095-4184-a252-14695d8d93eb\3f6f029a-image.png")
out = pathlib.Path(__file__).with_name("vision-check.output.txt")
b64 = base64.b64encode(image.read_bytes()).decode()
body = {
    "model": model, "max_tokens": 600, "temperature": 0,     # 600: thinking models spend tokens before the answer
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What model does this page announce, and what are its total and active parameter counts? One line."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
}
req = urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    think = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    usage = d.get("usage", {})
    line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {model} on {image.name} in {time.time()-t0:.0f} s "
            f"(finish {d['choices'][0].get('finish_reason')}, {usage.get('completion_tokens')} tokens out): {text!r}"
            + (f"  [thinking {len(think)} chars: {think[:160]!r}]" if think else ""))
except Exception as e:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {model} on {image.name} FAILED after {time.time()-t0:.0f} s: {e}"
print(line)
with out.open("a", encoding="utf-8") as f:
    f.write(line + "\n")
