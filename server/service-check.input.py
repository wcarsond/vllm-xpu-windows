# service-check.input.py - the live service on 11434, as VS Code uses it: for each model id, a chat request offering one
# tool (the answer should be a parsed tool call), then a plain chat request; prints load + answer time, decode rate and
# what came back. Usage:  .venv312\Scripts\python.exe server\service-check.input.py granite-4.2-30b+qwen3.5-27b
# (appends to server\service-check.output.txt)
# Why (2026-10-13): after switching the service to the one-process executor with decode graphs, and giving Granite the
# chat template its folder was missing (every Granite chat request had failed with HTTP 400), check both end to end.
import json, pathlib, sys, time, urllib.error, urllib.request

OUT = pathlib.Path(__file__).with_name("service-check.output.txt")
BASE = "http://127.0.0.1:11434"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def say(s):
    print(s, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def chat(model, messages, tools=None, max_tokens=200):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return time.time() - t0, None, f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    return time.time() - t0, d, None


def test_image_url():
    """A drawn 'conveyor' picture - a grey belt with a brown box labelled LOT 4417 - as a data URL (2026-10-13: the
    vision models are for conveyor-line image datasets; this checks they read what is on the box, not just that they load)."""
    import base64, io
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (640, 360), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 230, 640, 300], fill=(60, 60, 60))                  # the belt
    d.rectangle([230, 110, 430, 235], fill=(160, 110, 60), outline=(90, 60, 30), width=4)   # the box
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    d.text((252, 150), "LOT 4417", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def vision_check(model):
    msg = [{"role": "user", "content": [
        {"type": "text", "text": "What is on the conveyor, and what does its label say? One sentence."},
        {"type": "image_url", "image_url": {"url": test_image_url()}}]}]
    # 400 tokens: thinking models spend the budget reasoning first (gemma4 returned nothing at 80, 2026-10-13)
    dt, d, err = chat(model, msg, max_tokens=400)
    if err:
        say(f"{model} vision request: FAILED - {err}")
        return
    m = d["choices"][0]["message"]
    text = (m.get("content") or "").strip()
    think = (m.get("reasoning_content") or m.get("reasoning") or "").strip()
    seen = "4417" in text or "4417" in think
    say(f"{model} vision request ({dt:.1f} s): reads the label: {seen}; answer {text[:110]!r}"
        + (f"; reasoning {len(think)} chars" if think else ""))


if __name__ == "__main__":
    say(f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  {BASE}")
    # a second argument "vision" adds the image check for every model listed
    want_vision = len(sys.argv) > 2 and sys.argv[2] == "vision"
    for model in sys.argv[1].split("+"):
        if want_vision:
            vision_check(model)
        dt, d, err = chat(model, [{"role": "user", "content": "What's the weather in Toronto right now?"}], TOOLS)
        if err:
            say(f"{model} tool request: FAILED in {dt:.1f} s - {err}")
            continue
        m = d["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])]
        say(f"{model} tool request ({dt:.1f} s incl. any load): tool_calls {calls or 'NONE'}; text "
            f"{(m.get('content') or '').strip()[:80]!r}")
        dt, d, err = chat(model, [{"role": "user", "content": "Explain in three sentences why the sky is blue."}],
                          max_tokens=256)
        if err:
            say(f"{model} plain request: FAILED - {err}")
            continue
        n = d["usage"]["completion_tokens"]
        say(f"{model} plain request: {n} tokens in {dt:.2f} s ({n / dt:.1f} tok/s incl. prefill): "
            f"{(d['choices'][0]['message'].get('content') or '').strip()[:100]!r}")
