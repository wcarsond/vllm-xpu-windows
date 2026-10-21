# service-bench.input.py - each service model exactly as the router would start it (same `vllm serve` command line, from
# vllm-router.input.py's serve_cmd and models.input.ini), new setup against old: prefill time of a short prompt and
# decode tok/s over 256 tokens through the OpenAI API, plus whether the answer text is the same.
#   new = the ini as it is (executor oneproc, full decode graphs, max_seqs 16 unless a section says otherwise)
#   old = executor mp, eager, max_seqs 256 (the service before 2026-10-13)
# Usage:  .venv312\Scripts\python.exe server\service-bench.input.py qwen3.5-27b+granite-4.2-30b [new+old]
#         (appends to server\service-bench.output.txt; each run's vLLM output in service-bench-<id>-<config>.console.log)
# Runs on port 11445, beside the service (11434/11435); run it while the service has nothing loaded.
# Why (2026-10-13, Carson: "if its faster set it up"): measure at the service's own settings (65K context for the 27B,
# util 0.90-0.97, tool and reasoning parsers) before switching the service over.
import importlib.util, json, pathlib, subprocess, sys, time, urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "service-bench.output.txt"
PORT = 11445
spec = importlib.util.spec_from_file_location("router", HERE / "vllm-router.input.py")
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)


def say(s):
    print(s, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


NO_TEMPLATE = set()   # models whose folder has no chat template: timed through /v1/completions instead


def post(model_id, max_tokens):
    q = "Write a Python function that parses an INI file into a nested dict, with comments explaining each step."
    body = {"model": model_id, "temperature": 0, "max_tokens": max_tokens, "ignore_eos": True}
    if model_id in NO_TEMPLATE:
        body["prompt"] = q
        url = f"http://127.0.0.1:{PORT}/v1/completions"
    else:
        body["messages"] = [{"role": "user", "content": q}]
        url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # vLLM puts the reason in the body, not its log (2026-10-13 19:09: Granite answered 400 with nothing logged)
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from None
    c = d["choices"][0]
    return time.time() - t0, d["usage"]["completion_tokens"], (c["message"].get("content") if "message" in c else c.get("text")) or ""


def bench(model_id, cfg, config):
    cfg = dict(cfg)
    if config == "old":
        cfg.update(executor="mp", graphs="eager", max_seqs="256")
    cmd, env, folder, executor = router.serve_cmd(model_id, cfg, PORT)
    con = (HERE / f"service-bench-{model_id}-{config}.console.log").open("w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=con, stderr=subprocess.STDOUT, cwd=str(HERE))
    try:
        while True:
            if proc.poll() is not None:
                return f"{model_id} {config}: vLLM exited during load (code {proc.returncode}); see {con.name}", None
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > 1200:
                return f"{model_id} {config}: not ready after 20 min", None
            time.sleep(2)
        load = time.time() - t0
        try:
            post(model_id, 8)                                 # warm-up
        except RuntimeError as e:
            # Granite 4.2's folder has no chat template (2026-10-13 19:19): time it through plain completions
            if "chat template" not in str(e):
                raise
            NO_TEMPLATE.add(model_id)
            post(model_id, 8)
        t1, _, _ = post(model_id, 1)
        t2, n, text = post(model_id, 256)
        line = (f"{model_id} {config}{' [no chat template: /v1/completions]' if model_id in NO_TEMPLATE else ''} ({'oneproc' if executor == router.ONEPROC else 'mp'}, {cfg['graphs']}, max_len "
                f"{cfg['max_len']}, util {cfg['util']}): ready in {load:.0f} s; first token {t1:.2f} s; decode "
                f"{(n - 1) / max(t2 - t1, 1e-6):.1f} tok/s over {n} tokens")
        return line, text
    finally:
        subprocess.run(["taskkill", "/pid", str(proc.pid), "/t", "/f"], capture_output=True)
        time.sleep(10)


if __name__ == "__main__":
    # lists separated by + (cmd splits arguments at commas; 2026-10-13 18:54 a comma list ran one model)
    ids = sys.argv[1].replace(",", "+").split("+")
    configs = (sys.argv[2] if len(sys.argv) > 2 else "new+old").replace(",", "+").split("+")
    _, models = router.read_ini()
    say(f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  service settings from models.input.ini, port {PORT}")
    for mid in ids:
        texts = {}
        for c in configs:
            try:
                line, text = bench(mid, models[mid], c)
            except Exception as e:  # noqa: BLE001
                line, text = f"{mid} {c}: FAILED - {e}", None
            say(line)
            texts[c] = text
        if len(texts) == 2 and all(t is not None for t in texts.values()):
            a, b = texts.values()
            say(f"{mid}: answer text identical across configs: {a == b}")
