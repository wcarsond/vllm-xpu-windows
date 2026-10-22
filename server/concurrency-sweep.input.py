r"""concurrency-sweep.input.py - decode throughput by context length and number of concurrent requests, one block per model.

Usage:  .venv312\Scripts\python.exe server\concurrency-sweep.input.py <preset>[+<preset>] [contexts=0,8192,32768,65536]
            [requests=1,2,4,8,16] [gen=128] [budget=<minutes>] [cellcap=<seconds, default 600>]
        Appends to server\concurrency-sweep.output.md (tables) and server\concurrency-sweep.output.json (every cell's raw
        numbers); each model's vLLM output goes to server\concurrency-sweep-<preset>-<stamp>.console.log.

Why (2026-10-21, Carson saw another person's benchmark page: "see how this guy does it"): one block per model, header
"<model> <dtype> — 2x B60, vLLM <version> · <date>", rows = context length, columns = concurrent requests, cells = aggregate
decode tokens/s. "yeah let's use newer models if we can": presets come from models.input.ini plus models-sweep.input.ini
(the newer Qwen3.6 27B GPTQ build lives only there).

How a preset is run: exactly as the router starts it (vllm-router.input.py's serve_cmd and the ini, executor oneproc, full
decode graphs, max_num_seqs 16, the router's own util and max_len), on port 11445 beside the service, inside the MSVC
environment (as service-bench.input.cmd sets it). The command line is NOT changed; prefix caching stays as the service has
it (on), and every request carries its own numbered first line so no request can reuse another's cache (the depth races
turned prefix caching off for the same reason; the prefix-cache hit counter is recorded per cell to prove it stayed 0).
A folder that model-store.ps1 moved to another disk (<folder>.moved-to-X.txt beside where it was) is followed there.

Per context (row): a prompt of exactly that many tokens (race-7b-depth.input.py's filler-and-task method, built as token ids
with the model's own tokenizer and sent to /v1/completions as ids, so the count is exact; 0 = the task line alone), one
warm-up request, then prefill timed with a 1-token request. Context + gen must fit max_len; the context at max_len is
trimmed to max_len - gen and the row label says so. Per request count N (column): N identical-length requests streamed at
once from N threads, temperature 0, ignore_eos, max_tokens = gen. Every token's arrival time is kept.
  decode (the table) = tokens generated while all N requests were decoding together (after the last request's first token,
                       before the first request's last token) / that window's seconds. Prefill is entirely outside it.
                       At N=1 this is race-7b-depth's number: (gen - 1) / (time from first to last token).
  raw (second table) = all generated tokens / wall time from the first send to the last completion, prefill included.
Marks: "—" the batch was not run because N x (context + gen) exceeds the KV cache vLLM reported at start, or the server
answered an error for it; "*" some request finished early (fewer than gen tokens), was preempted (vllm:num_preemptions
rose), or the requests never all decoded together (later prefills outlasted earlier requests' decode); the starred cell
then shows the rate after the last first token to the end, still prefill-free, but with fewer than N requests decoding.
"n/r" not run for time: N prefills at the measured 1-request prefill rate would take longer than cellcap (600 s), so
the sweep fits in its evening; it says nothing about whether the batch fits.

Gates before any card work: no python process with train.py, chantrain.py or predict.py in its command line (polled every
60 s, up to 40 min, then the run stops), and the service's router at :11434 reports "loaded": null (waits for its idle
unload). Only the vLLM process this script started is ever stopped.
2026-10-21.
"""
import concurrent.futures, configparser, datetime, importlib.util, json, math, os, pathlib, re, subprocess, sys, threading, time
import urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT_MD = HERE / "concurrency-sweep.output.md"
OUT_JSON = HERE / "concurrency-sweep.output.json"
SWEEP_INI = HERE / "models-sweep.input.ini"
PORT = 11445
BASE = f"http://127.0.0.1:{PORT}"
ROUTER_STATUS = "http://127.0.0.1:11434/router/status"
VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
spec = importlib.util.spec_from_file_location("router", HERE / "vllm-router.input.py")
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)

TASK = "Write a Python function that parses an INI file into a nested dict, with comments explaining each step."
PARA = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report on regional "
        "logistics, inventory turnover, and the maintenance schedule for the conveyor system in building four. ")
UID = [0]
UID_LOCK = threading.Lock()
T0_RUN = time.time()


def say(s):
    print(f"{datetime.datetime.now():%H:%M:%S} {s}", flush=True)


# ---------------------------------------------------------------- gates
def busy_trainers():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          "ForEach-Object { [string]$_.ProcessId + ' ' + $_.CommandLine }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    pat = re.compile(r"(^|[\\/\s\"'])(train|chantrain|predict)\.py\b")
    return [ln.strip()[:160] for ln in out.splitlines() if pat.search(ln)]


def router_loaded():
    """(reachable, loaded id or None)"""
    try:
        with urllib.request.urlopen(ROUTER_STATUS, timeout=5) as r:
            return True, json.loads(r.read()).get("loaded")
    except Exception:  # noqa: BLE001
        return False, None


def gates():
    t0 = time.time()
    while True:
        b = busy_trainers()
        if not b:
            break
        if time.time() - t0 > 40 * 60:
            say(f"GATE: training still running after 40 min, stopping: {b}")
            return False
        say(f"GATE: waiting for training to finish ({len(b)} process(es)): {b[0]}")
        time.sleep(60)
    t1 = time.time()
    while True:
        ok, loaded = router_loaded()
        if loaded is None:
            say(f"GATE: router {'reports loaded: null' if ok else 'not reachable (service off?)'}; cards free")
            return True
        if time.time() - t1 > 30 * 60:
            say(f"GATE: router still has {loaded} loaded after 30 min, stopping")
            return False
        say(f"GATE: router has {loaded} loaded; waiting for its idle unload")
        time.sleep(30)


# ---------------------------------------------------------------- presets
def presets():
    _, models = router.read_ini()
    cp = configparser.ConfigParser(interpolation=None)
    if SWEEP_INI.exists():
        cp.read(SWEEP_INI, encoding="utf-8")
        for sec in cp.sections():
            d = dict(router.DEFAULTS)
            d.update(cp[sec])
            d["_from"] = SWEEP_INI.name
            models[sec] = d
    return models


def resolve_folder(cfg):
    """The model folder, following a model-store.ps1 '<folder>.moved-to-X.txt' note when the folder itself is gone."""
    folder = pathlib.Path(cfg["folder"])
    if not folder.is_absolute():
        folder = router.MODELS_DIR / folder
    if folder.exists():
        return folder, None
    for note in folder.parent.glob(folder.name + ".moved-to-*.txt"):
        m = re.search(r"moved to (\S+)", note.read_text(encoding="utf-8-sig"))
        if m and pathlib.Path(m.group(1)).exists():
            return pathlib.Path(m.group(1)), f"folder moved: {folder} -> {m.group(1)} ({note.name})"
    return None, f"folder missing: {folder}"


def msvc_env():
    """vcvars64 + what service-bench.input.cmd sets, merged into os.environ (serve_cmd copies os.environ)."""
    out = subprocess.run(f'cmd /c ""{VCVARS}" >nul 2>&1 && set"', capture_output=True, text=True, shell=True).stdout
    for ln in out.splitlines():
        if "=" in ln and not ln.startswith("="):
            k, v = ln.split("=", 1)
            os.environ[k] = v
    os.environ["CC"] = "cl"
    os.environ["PATH"] = str(HERE.parent / ".venv312" / "Library" / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONUNBUFFERED"] = "1"


def dtype_label(folder, cfg):
    try:
        c = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        c = {}
    q = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config")
    if q:
        meth = str(q.get("quant_method", "?")).upper()
        return f"{meth} int{q.get('bits', '?')} ({cfg['dtype']} activations)"
    return cfg["dtype"]


# ---------------------------------------------------------------- HTTP
def get_json(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def metrics():
    """sums of the counters we watch, from /metrics"""
    want = {"vllm:num_preemptions_total": 0.0, "vllm:prefix_cache_hits_total": 0.0, "vllm:prefix_cache_queries_total": 0.0}
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
            for ln in r.read().decode(errors="replace").splitlines():
                if ln.startswith("#"):
                    continue
                name = ln.split("{")[0].split(" ")[0]
                if name in want:
                    want[name] += float(ln.rsplit(" ", 1)[1])
    except Exception:  # noqa: BLE001
        return None
    return want


def post_stream(model_id, prompt_ids, max_tokens, timeout=1800):
    """One streamed /v1/completions request. Returns dict(t_send, times[], tokens per chunk[], usage tokens, error)."""
    body = {"model": model_id, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    res = {"t_send": time.perf_counter(), "times": [], "counts": [], "usage": None, "finish": None, "error": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                ln = raw.strip()
                if not ln.startswith(b"data:"):
                    continue
                data = ln[5:].strip()
                if data == b"[DONE]":
                    break
                now = time.perf_counter()
                d = json.loads(data)
                if d.get("usage"):
                    res["usage"] = d["usage"].get("completion_tokens")
                for ch in d.get("choices") or []:
                    n = len(ch.get("token_ids") or []) if "token_ids" in ch else (1 if ch.get("text") else 0)
                    if n:
                        res["times"].append(now)
                        res["counts"].append(n)
                    if ch.get("finish_reason"):
                        res["finish"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        res["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    res["t_end"] = time.perf_counter()
    return res


def post_once(model_id, prompt_ids, max_tokens, timeout=1800):
    body = {"model": model_id, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return time.perf_counter() - t, d["usage"]["prompt_tokens"]


# ---------------------------------------------------------------- prompts
class Prompts:
    """Exact-length token-id prompts, each with its own numbered first line (no shared prefix between requests)."""

    def __init__(self, tok):
        self.tok = tok
        self.enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        self.task0 = self.enc(TASK)
        self.tail = self.enc(f"\nNow, {TASK[0].lower()}{TASK[1:]}")
        filler = PARA * 40
        self.filler = self.enc("Here is a document:\n" + filler)
        self.chunk = self.enc(PARA * 8)

    def head(self, uid):
        return self.enc(f"Request {uid:07d}.\n")

    def make(self, depth):
        with UID_LOCK:
            UID[0] += 1
            uid = UID[0]
        h = self.head(uid)
        if depth == 0:
            return h + self.task0
        need = depth - len(h) - len(self.tail)
        while len(self.filler) < need:
            self.filler += self.chunk
        return h + self.filler[:need] + self.tail


# ---------------------------------------------------------------- one cell
def run_cell(model_id, prompts, depth, n, gen):
    ps = [prompts.make(depth) for _ in range(n)]
    m0 = metrics()
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        rs = list(ex.map(lambda p: post_stream(model_id, p, gen), ps))
    m1 = metrics()
    errors = [r["error"] for r in rs if r["error"]]
    cell = {"n": n, "prompt_tokens": len(ps[0])}
    if errors:
        cell.update(mark="—", reason="server error: " + errors[0], errors=len(errors))
        return cell
    if not all(r["times"] for r in rs):
        cell.update(mark="—", reason="a request streamed no tokens", errors=sum(1 for r in rs if not r["times"]))
        return cell
    toks = [sum(r["counts"]) for r in rs]
    usage = [r["usage"] for r in rs]
    gen_tokens = [u if u is not None else t for u, t in zip(usage, toks)]
    t_end = max(r["t_end"] for r in rs)
    wall = t_end - t_start
    firsts = [r["times"][0] for r in rs if r["times"]]
    lasts = [r["times"][-1] for r in rs if r["times"]]
    tf, te = max(firsts), min(lasts)
    events = [(t, c) for r in rs for t, c in zip(r["times"], r["counts"])]
    staggered = te <= tf
    in_win = sum(c for t, c in events if tf < t <= te)
    if not staggered and in_win >= 2 * n:
        decode = in_win / (te - tf)
        window = (te - tf, in_win, "all N decoding")
    else:
        t_last = max(lasts)
        after = sum(c for t, c in events if t > tf)
        decode = after / max(t_last - tf, 1e-6)
        window = (t_last - tf, after, "last first token to end")
        staggered = True
    pre = (m1 or {}).get("vllm:num_preemptions_total", 0) - (m0 or {}).get("vllm:num_preemptions_total", 0)
    hits = (m1 or {}).get("vllm:prefix_cache_hits_total", 0) - (m0 or {}).get("vllm:prefix_cache_hits_total", 0)
    short = sum(1 for g in gen_tokens if g < gen)
    why = []
    if short:
        why.append(f"{short} of {n} requests returned fewer than {gen} tokens")
    if pre > 0:
        why.append(f"{int(pre)} preemption(s)")
    if staggered:
        why.append("requests never all decoded together")
    cell.update(mark="*" if why else "", reason="; ".join(why), decode_tps=round(decode, 2),
                raw_tps=round(sum(gen_tokens) / wall, 2), wall_s=round(wall, 3), tokens=sum(gen_tokens),
                tokens_per_request=gen_tokens, chunk_token_counts=toks, finish=[r["finish"] for r in rs],
                first_token_s=[round(f - t_start, 3) for f in firsts], last_token_s=[round(t - t_start, 3) for t in lasts],
                decode_window_s=round(window[0], 3), decode_window_tokens=window[1], decode_window=window[2],
                preemptions=pre, prefix_cache_hit_tokens=hits)
    return cell


# ---------------------------------------------------------------- one preset
def save_json(run):
    runs = []
    if OUT_JSON.exists():
        try:
            runs = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            # never lose an earlier file: keep it beside, start a list
            OUT_JSON.rename(OUT_JSON.with_suffix(f".unreadable-{int(time.time())}.json"))
            runs = []
    runs = [r for r in runs if r.get("run_id") != run["run_id"]] + [run]
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    os.replace(tmp, OUT_JSON)


def ctx_label(c):
    return "0" if c == 0 else (f"{c // 1024}K" if c % 1024 == 0 else f"{c:,}")


def write_block(b, requests):
    def table(key):
        rows = [f"| context \\ requests | " + " | ".join(str(n) for n in requests) + " | prefill tok/s (1 req) |",
                "|---|" + "---:|" * (len(requests) + 1)]
        for row in b["rows"]:
            cells = []
            for n in requests:
                c = row["cells"].get(str(n))
                if c is None:
                    cells.append("")
                elif c["mark"] in ("—", "n/r"):
                    cells.append(c["mark"])
                else:
                    cells.append(f"{c[key]:.1f}{c['mark']}")
            lab = ctx_label(row["context"]) + ("" if row["prompt_tokens"] == row["context"] or row["context"] == 0
                                               else f" ({row['prompt_tokens']:,})")
            if row["context"] == 0:
                lab = f"0 ({row['prompt_tokens']})"
            pf = f"{row['prefill_tps']:,.0f}" if row.get("prefill_tps") else ""
            rows.append(f"| {lab} | " + " | ".join(cells) + f" | {pf} |")
        return "\n".join(rows)

    notes = []
    for row in b["rows"]:
        for n in requests:
            c = row["cells"].get(str(n))
            if c and c["mark"]:
                notes.append(f"{ctx_label(row['context'])} x {n}: {c['mark']} {c['reason']}")
    lines = [f"### {b['model']} {b['dtype']} — 2x B60, vLLM {b['version']} · {b['date']}", "",
             f"Preset `{b['preset']}` ({b['ini']}), run {b['started']}: {b['executor']}, {b['graphs']} graphs, max_len "
             f"{b['max_len']}, util {b['util']}, max_num_seqs {b['max_seqs']}, KV cache {b['kv_tokens'] or '?'} tokens, "
             f"{b['gen']} tokens generated per request (greedy, ignore_eos), ready in {b['ready_s']:.0f} s.", "",
             "Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; "
             "prefill excluded):", "", table("decode_tps"), "",
             "Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):",
             "", table("raw_tps"), "",
             "— = not run: N x (context + gen) is more than the KV cache vLLM reported, or the server returned an error. "
             "* = some request finished short or was preempted, or the requests never all decoded together (later "
             "prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, "
             "still prefill-free but with fewer than N decoding. n/r = not run for time (N prefills at the measured "
             "1-request prefill rate would pass the per-cell cap); says nothing about fit." + (" This block: " + "; ".join(notes) + "." if notes else
                                                                      " No cell in this block was marked."), ""]
    if b.get("problems"):
        lines += ["Problems: " + "; ".join(b["problems"]), ""]
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sweep(preset, cfg, contexts, requests, gen, run, deadline, cellcap):
    from transformers import AutoTokenizer
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    folder, note = resolve_folder(cfg)
    if folder is None:
        say(f"{preset}: SKIPPED, {note}")
        run["skipped"].append({"preset": preset, "why": note})
        return
    cfg = dict(cfg)
    cfg["folder"] = str(folder)
    max_len = int(cfg["max_len"])
    b = {"preset": preset, "ini": cfg.get("_from", "models.input.ini"), "model": folder.name, "dtype": dtype_label(folder, cfg),
         "version": "?", "date": datetime.date.today().isoformat(), "started": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
         "executor": cfg["executor"], "graphs": cfg["graphs"], "max_len": max_len, "util": cfg["util"],
         "max_seqs": cfg["max_seqs"], "gen": gen, "kv_tokens": None, "ready_s": 0, "rows": [], "problems": []}
    if note:
        b["problems"].append(note + " (the router's own start of this preset would fail)")
    run["blocks"].append(b)
    ok, loaded = router_loaded()
    if loaded is not None:
        b["problems"].append(f"router had {loaded} loaded at start; preset not run")
        say(f"{preset}: router has {loaded} loaded, not starting")
        return
    cmd, env, _, executor = router.serve_cmd(preset, cfg, PORT)
    b["command"] = " ".join(cmd)
    con_path = HERE / f"concurrency-sweep-{preset}-{stamp}.console.log"
    con = con_path.open("w", encoding="utf-8")
    t0 = time.time()
    say(f"{preset}: starting {folder} on :{PORT} -> {con_path.name}")
    proc = subprocess.Popen(cmd, env=env, stdout=con, stderr=subprocess.STDOUT, cwd=str(HERE))
    b["pid"] = proc.pid
    try:
        while True:
            if proc.poll() is not None:
                b["problems"].append(f"vLLM exited during load (code {proc.returncode}); see {con_path.name}")
                say(f"{preset}: vLLM exited during load ({proc.returncode})")
                return
            try:
                get_json("/v1/models", 3)
                break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > 1800:
                b["problems"].append("not ready after 30 min")
                return
            time.sleep(3)
        b["ready_s"] = time.time() - t0
        try:
            b["version"] = get_json("/version")["version"]
        except Exception:  # noqa: BLE001
            pass
        if b["version"] in ("?", "dev"):
            # this editable build's vllm/_version.py is missing, so the server says "dev"; the installed package knows
            import importlib.metadata
            b["version"] = importlib.metadata.version("vllm")
        txt = con_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"KV cache size: ([\d,]+) tokens", txt)
        b["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
        if b["kv_tokens"] is None:
            # 2026-10-21: serve mode on this build does not print the capacity line; vllm:cache_config_info carries
            # num_gpu_blocks and block_size (for the hybrids this counts the attention layers' blocks only)
            try:
                with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
                    mt = r.read().decode(errors="replace")
                ci = re.search(r"^vllm:cache_config_info\{([^}]*)\}", mt, re.M)
                lab = dict(re.findall(r'(\w+)="([^"]*)"', ci.group(1))) if ci else {}
                b["cache_config_info"] = lab
                if lab.get("kv_cache_size_tokens") not in (None, "", "None"):
                    # vLLM's own group-aware capacity (the number its "KV cache size" log line would print)
                    b["kv_tokens"] = int(float(lab["kv_cache_size_tokens"]))
                    b["kv_source"] = "kv_cache_size_tokens from /metrics"
                elif lab.get("num_gpu_blocks") not in (None, "", "None") and lab.get("block_size") not in (None, ""):
                    b["kv_tokens"] = int(lab["num_gpu_blocks"]) * int(lab["block_size"])
                    b["kv_source"] = "num_gpu_blocks x block_size from /metrics"
            except Exception as e:  # noqa: BLE001
                b["problems"].append(f"KV capacity unknown ({e})")
        m = re.search(r"max_num_batched_tokens[=:\s]+(\d+)", txt)
        b["max_num_batched_tokens"] = int(m.group(1)) if m else None
        say(f"{preset}: ready in {b['ready_s']:.0f} s, vLLM {b['version']}, KV {b['kv_tokens']} tokens")
        tok = AutoTokenizer.from_pretrained(str(folder))
        prompts = Prompts(tok)
        for ctx in contexts:
            if ctx > max_len:
                say(f"{preset}: context {ctx} > max_len {max_len}, row left out")
                continue
            depth = ctx if ctx + gen <= max_len else max_len - gen
            row = {"context": ctx, "prompt_tokens": depth if ctx else len(prompts.make(0)), "cells": {}}
            b["rows"].append(row)
            if time.time() > deadline:
                b["problems"].append(f"time budget reached before context {ctx}")
                break
            try:
                r = post_stream(preset, prompts.make(depth), gen)             # warm-up at this context
                if r["error"]:
                    raise RuntimeError(r["error"])
                t_pf, n_prompt = post_once(preset, prompts.make(depth), 1)    # prefill alone
                row["prefill_s"], row["prefill_tps"], row["prompt_tokens"] = round(t_pf, 3), round(n_prompt / t_pf, 1), n_prompt
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)[:300]
                say(f"{preset}: context {ctx}: warm-up failed: {row['error']}")
                for n in requests:
                    row["cells"][str(n)] = {"n": n, "mark": "—", "reason": "warm-up failed: " + row["error"]}
                save_json(run)
                if proc.poll() is not None:
                    b["problems"].append(f"vLLM died (code {proc.returncode})")
                    return
                continue
            say(f"{preset}: context {ctx} ({row['prompt_tokens']} tokens): prefill {row['prefill_tps']} tok/s")
            for n in requests:
                if time.time() > deadline:
                    b["problems"].append(f"time budget reached at context {ctx} x {n}")
                    break
                if b["kv_tokens"] and n * (row["prompt_tokens"] + gen) > b["kv_tokens"]:
                    c = {"n": n, "mark": "—", "reason": f"needs {n * (row['prompt_tokens'] + gen):,} KV tokens, cache "
                                                         f"holds {b['kv_tokens']:,}"}
                elif n * row["prompt_tokens"] / max(row["prefill_tps"], 1) > cellcap:
                    c = {"n": n, "mark": "n/r", "reason": f"not run: {n} prefills of {row['prompt_tokens']:,} tokens at "
                         f"{row['prefill_tps']:,.0f} tok/s would take {n * row['prompt_tokens'] / row['prefill_tps']:.0f} s "
                         f"(cap {cellcap:.0f} s)"}
                else:
                    c = run_cell(preset, prompts, depth, n, gen)
                    ok, loaded = router_loaded()
                    if loaded is not None:
                        c["router_loaded_during"] = loaded
                        b["problems"].append(f"the service loaded {loaded} during {ctx_label(ctx)} x {n}")
                row["cells"][str(n)] = c
                say(f"{preset}: ctx {ctx} x {n}: " + (f"decode {c.get('decode_tps')} raw {c.get('raw_tps')} {c['mark']} "
                                                     f"{c.get('reason', '')}" if c["mark"] not in ("—", "n/r") else c["mark"] + " " + c["reason"]))
                save_json(run)
                if proc.poll() is not None:
                    b["problems"].append(f"vLLM died (code {proc.returncode}) at {ctx_label(ctx)} x {n}")
                    return
            else:
                continue
            break
    finally:
        subprocess.run(["taskkill", "/pid", str(proc.pid), "/t", "/f"], capture_output=True)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        con.close()
        say(f"{preset}: stopped pid {proc.pid}")
        b["finished"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        save_json(run)
        write_block(b, requests)
        time.sleep(10)


def main():
    if len(sys.argv) < 2:
        print(__doc__.split("\n\n")[1])
        return 2
    ids = sys.argv[1].replace(",", "+").split("+")
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    contexts = [int(x) for x in kv.get("contexts", "0,8192,32768,65536").replace("+", ",").split(",")]
    requests = [int(x) for x in kv.get("requests", "1,2,4,8,16").replace("+", ",").split(",")]
    gen = int(kv.get("gen", "128"))
    deadline = time.time() + float(kv.get("budget", "1e9")) * 60
    cellcap = float(kv.get("cellcap", "600"))
    models = presets()
    run = {"run_id": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:], "contexts": contexts,
           "requests": requests, "gen": gen, "blocks": [], "skipped": []}
    todo = []
    for p in ids:
        if p not in models:
            run["skipped"].append({"preset": p, "why": "no such preset"})
        elif models[p].get("engine", "vllm") != "vllm":
            run["skipped"].append({"preset": p, "why": f"engine = {models[p]['engine']}, not vLLM"})
        else:
            todo.append(p)
    for s in run["skipped"]:
        say(f"skipping {s['preset']}: {s['why']}")
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=3):
            say(f"something already answers on :{PORT}; not starting (it is not ours)")
            return 1
    except Exception:  # noqa: BLE001
        pass
    if todo and not gates():
        run["skipped"] += [{"preset": p, "why": "gates did not open"} for p in todo]
        save_json(run)
        return 1
    msvc_env()
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n## Run {run['run_id']} — `{' '.join(sys.argv[1:])}`\n\n"
                + "".join(f"- skipped {s['preset']}: {s['why']}\n" for s in run["skipped"]) + "\n")
    save_json(run)
    for p in todo:
        try:
            sweep(p, models[p], contexts, requests, gen, run, deadline, cellcap)
        except Exception as e:  # noqa: BLE001
            say(f"{p}: FAILED - {type(e).__name__}: {e}")
            run["skipped"].append({"preset": p, "why": f"failed: {type(e).__name__}: {e}"[:300]})
            save_json(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
