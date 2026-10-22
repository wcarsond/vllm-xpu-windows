r"""concurrency-sweep-llamacpp.input.py - the concurrency sweep's grid (context x concurrent requests) on llama.cpp, so the
page can show vLLM against llama.cpp under load, not only single stream.

Usage:  .venv312\Scripts\python.exe server\concurrency-sweep-llamacpp.input.py <preset>[+<preset>] [contexts=0,8192,32768,65536] [requests=1,2,4,8,16] [gen=128]
        also: budget=<minutes for the whole run, gate waits included, default 180> cellcap=<seconds, default 400>
              mmproj=0 (start WITHOUT the preset's vision projector: a text-only benchmark, said in the block)
        Appends to server\concurrency-sweep-llamacpp.output.md (tables) and server\concurrency-sweep-llamacpp.output.json
        (every cell's raw numbers); each llama-server's own output goes to
        server\concurrency-sweep-llamacpp-<preset>-<stamp>-np<N>-c<C>.console.log.

Why (2026-10-22): Carson, going to bed after the vLLM concurrency sweep (concurrency-sweep.input.py) was started: "you can
keep running tests man". The vLLM sweep gives one block per model, rows = context, columns = concurrent requests; this
gives the same block for the llama.cpp copy of the same model (qwen3.6-27b GGUF against the Qwen3.6-27B-GPTQ-Int4 vLLM
ran; qwen2.5-coder-7b-q8 against the 7B bf16, as in the README's race; qwen3.5-27b has no GGUF preset) plus
qwen3.6-35b-a3b, the fastest big model on these cards.

Shared with the vLLM sweep, imported from concurrency-sweep.input.py (not copied): the prompt construction (Prompts:
exact-length token-id prompts, each request with its own numbered first line so no request reuses another's cache), the
cell (run_cell: N identical-length requests streamed at once from N threads to /v1/completions, temperature 0,
ignore_eos, max_tokens = gen, every token's arrival kept; decode = tokens while all N were decoding / that window, raw =
all tokens / wall time from first send to last completion), the 1-token prefill timing (post_once), the cell marks and
the n/r time cap. Swapped in for llama.cpp: the tokenizer is llama-server's own /tokenize (add_special false), so counts
are exact for the GGUF; post_stream is the same request body, keeping llama.cpp's token ids and final timings (cache_n =
prompt tokens llama.cpp reused from a slot's cache, recorded per cell; it should stay ~0, like vLLM's prefix-cache
hits); vLLM's /metrics counters do not exist here (llama.cpp never preempts: a slot holds its whole context from load).
COPIED from concurrency-sweep.input.py, because they write the vLLM sweep's own files: save_json, and write_block (its
header here names llama.cpp, the GGUF and the -c / --parallel used per row).

How a preset is run: the router's own llama-server command (vllm-router.input.py's serve_cmd for engine = llamacpp:
vk-allreduce-fa build, -dev the two B60s, -sm layer, -ngl 99, -ncmoe from the ini, -fa on, --jinja, -lm none, the mmproj
when the ini has one; the router sets no KV env, it only drops ZE_AFFINITY_MASK / ONEAPI_DEVICE_SELECTOR) on port 11446,
with only -c and -np changed. llama.cpp splits -c evenly across --parallel slots (kv_unified is off when -np is given),
so per row: slot = context + gen + 64 rounded up to 256, -c = np x slot, trying np = 16, 8, 4, 2, 1 (the request counts)
until one comes up (/health ok, /slots shows n_ctx >= slot) WITHOUT spilling: Windows lets an over-full card spill into
system RAM silently and run at a quarter speed (llama.cpp-b60\serve-fit-test.output.txt, 2026-10-13), so the shared GPU
memory of all adapters (GPU Adapter Memory perf counters) is read before start, after load and after the row's warm-up,
and a rise of more than 1.5 GiB counts as not fitting. Columns above the np that fit are "—". The context-0 row runs on
the next row's server when that fits (one load fewer). A GGUF that model-store.ps1 moved (<file>.moved-to-X.txt) is
followed. Rows above the preset's max_len are left out and the row at max_len is trimmed to max_len - gen, as in the
vLLM sweep, so the two tables line up row for row.

mmproj=0 (2026-10-22 01:50): with --mmproj loaded, this build (0.5.0-dev build 55) answers every /v1/completions whose
prompt is a token-id array with HTTP 400 "[json.exception.type_error.302] type must be string, but is array" (the
multimodal path reads prompt as a string), so qwen3.6-27b's whole block came out "—". Sending text instead would
re-tokenize and lose the exact counts, so the preset is run without its mmproj; the block's text says so.

Gates before any card work (polled every 60 s until 20 min before the budget ends; then the run heading says so and it
stops): (1) no vllm.exe process exists and nothing listens on 11445 (the vLLM sweep is done), (2) no python process with
train.py / chantrain.py / predict.py in its command line, (3) http://127.0.0.1:11434/router/status reports
"loaded": null. Only the llama-server this script started is ever stopped. 2026-10-22.
"""
import datetime, importlib.util, json, math, os, pathlib, re, struct, subprocess, sys, threading, time, traceback
import urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT_MD = HERE / "concurrency-sweep-llamacpp.output.md"
OUT_JSON = HERE / "concurrency-sweep-llamacpp.output.json"
PORT = 11446
BASE = f"http://127.0.0.1:{PORT}"
VLLM_SWEEP_PORT = 11445
SPILL_MIB = 1536
NO_MMPROJ = False               # mmproj=0 on the command line
T0_RUN = time.time()

# the vLLM sweep, imported whole (it imports the router in turn); only its pure functions are used, never its
# save_json / write_block / sweep / main, which write the vLLM sweep's own files
_spec = importlib.util.spec_from_file_location("csweep", HERE / "concurrency-sweep.input.py")
CS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CS)
router = CS.router
CS.BASE = BASE                  # post_once and run_cell's requests go to OUR llama-server
CS.metrics = lambda: None       # vLLM counters; llama.cpp has none
say = CS.say
ctx_label = CS.ctx_label
router_loaded = CS.router_loaded
busy_trainers = CS.busy_trainers

LOCK = threading.Lock()
CELL_RESULTS = []


# ---------------------------------------------------------------- llama.cpp request (same body as the vLLM sweep's)
def post_stream(model_id, prompt_ids, max_tokens, timeout=1800):
    """concurrency-sweep's post_stream, plus llama.cpp's per-chunk token ids and final timings (cache_n)."""
    body = {"model": model_id, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True, "return_tokens": True}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    res = {"t_send": time.perf_counter(), "times": [], "counts": [], "usage": None, "finish": None, "error": None,
           "timings": None, "prompt_usage": None, "count_source": None}
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
                    res["prompt_usage"] = d["usage"].get("prompt_tokens")
                if d.get("timings"):
                    res["timings"] = d["timings"]
                for ch in d.get("choices") or []:
                    if "token_ids" in ch:
                        n, src = len(ch.get("token_ids") or []), "token_ids"
                    elif isinstance(ch.get("tokens"), list):
                        n, src = len(ch["tokens"]), "tokens"
                    elif isinstance(d.get("tokens"), list):
                        n, src = len(d["tokens"]), "tokens"
                    else:
                        n, src = (1 if ch.get("text") else 0), "text chunks"
                    res["count_source"] = src
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
    with LOCK:
        CELL_RESULTS.append(res)
    return res


CS.post_stream = post_stream    # CS.run_cell calls it through its module globals


def run_cell(model_id, prompts, depth, n, gen):
    with LOCK:
        CELL_RESULTS.clear()
    c = CS.run_cell(model_id, prompts, depth, n, gen)
    with LOCK:
        rs = list(CELL_RESULTS)
    c.pop("preemptions", None)                  # vLLM-only counters, always 0 here
    c.pop("prefix_cache_hit_tokens", None)
    c["prompt_cache_reused_tokens"] = [(r.get("timings") or {}).get("cache_n") for r in rs]
    c["server_prompt_n"] = [(r.get("timings") or {}).get("prompt_n") for r in rs]
    c["server_predicted_per_second"] = [(r.get("timings") or {}).get("predicted_per_second") for r in rs]
    c["count_source"] = sorted({str(r.get("count_source")) for r in rs})
    return c


class Tok:
    """HF-tokenizer-shaped wrapper round llama-server's /tokenize, so concurrency-sweep's Prompts works unchanged."""

    def __init__(self):
        self.cache = {}

    def __call__(self, s, add_special_tokens=False):
        if s not in self.cache:
            body = json.dumps({"content": s, "add_special": bool(add_special_tokens)}).encode()
            req = urllib.request.Request(BASE + "/tokenize", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.cache[s] = json.loads(r.read())["tokens"]
        return {"input_ids": list(self.cache[s])}


def get_json(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- gates
def vllm_sweep_running():
    """why the vLLM sweep still counts as running, or None"""
    ps = ("$p = @(Get-Process vllm -ErrorAction SilentlyContinue).Count; "
          f"$l = @(Get-NetTCPConnection -State Listen -LocalPort {VLLM_SWEEP_PORT} -ErrorAction SilentlyContinue).Count; "
          "\"$p $l\"")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=60).stdout.split()
        p, l = (int(out[0]), int(out[1])) if len(out) == 2 else (1, 1)
    except Exception:  # noqa: BLE001
        p, l = 1, 1
    why = []
    if p:
        why.append(f"{p} vllm.exe process(es)")
    if l:
        why.append(f"something listens on {VLLM_SWEEP_PORT}")
    return ", ".join(why) or None


def gates(deadline):
    last = None
    while True:
        why = vllm_sweep_running()
        if not why:
            b = busy_trainers()
            if b:
                why = f"training running: {b[0]}"
        if not why:
            ok, loaded = router_loaded()
            if loaded is not None:
                why = f"router has {loaded} loaded"
        if not why:
            say("GATE: vLLM sweep done, no training, router loaded: null; cards free")
            return True
        if time.time() > deadline:
            say(f"GATE: still closed at the time limit ({why}); stopping")
            return False
        if why != last:
            say(f"GATE: waiting ({why})")
            last = why
        time.sleep(60)


# ---------------------------------------------------------------- GGUF, build, GPU memory
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
              19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
              26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"}


def gguf_meta(path, want=("general.file_type", "general.name", "general.architecture")):
    """A few header keys of a GGUF file (no gguf package in .venv312; the format is simple)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return out
        struct.unpack("<I", f.read(4))
        _, n_kv = struct.unpack("<QQ", f.read(16))

        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                it, n = struct.unpack("<IQ", f.read(12))
                if it in sizes:
                    f.seek(sizes[it] * n, 1)
                else:
                    for _ in range(n):
                        rval(it)
                return None
            return struct.unpack(fmts[t], f.read(sizes[t]))[0]

        for _ in range(n_kv):
            k = rstr()
            t, = struct.unpack("<I", f.read(4))
            v = rval(t)
            if k in want:
                out[k] = v
                if len(out) == len(want):
                    break
    return out


def resolve_gguf(cfg):
    g = pathlib.Path(cfg["gguf"])
    if not g.is_absolute():
        g = router.GGUF_DIR / g
    if g.exists():
        return g, None
    for note in g.parent.glob(g.name + ".moved-to-*.txt"):
        m = re.search(r"moved to (\S+)", note.read_text(encoding="utf-8-sig"))
        if m and pathlib.Path(m.group(1)).exists():
            return pathlib.Path(m.group(1)), f"GGUF moved: {g} -> {m.group(1)} ({note.name})"
    return None, f"GGUF missing: {g}"


def llama_version():
    try:
        p = subprocess.run([str(router.LLAMA_SERVER), "--version"], capture_output=True, text=True, timeout=120)
        lines = [ln.strip() for ln in (p.stdout + "\n" + p.stderr).splitlines() if ln.strip()]
        v = [ln for ln in lines if ln.startswith("version")]
        b = [ln for ln in lines if ln.startswith("built with")]
        return (v[0] if v else (lines[0] if lines else "?")), (b[0] if b else "")
    except Exception as e:  # noqa: BLE001
        return f"? ({e})", ""


def gpu_mem():
    """{luid: {"dedicated": MiB, "shared": MiB}} for every GPU adapter, from the Windows perf counters"""
    ps = (r"(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage')"
          r".CounterSamples | ForEach-Object { $_.Path + '=' + [int64]$_.CookedValue }")
    out = {}
    try:
        txt = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=90).stdout
        for ln in txt.splitlines():
            m = re.search(r"\((luid_[^)]+)\)\\(dedicated|shared) usage=(\d+)", ln, re.I)
            if m:
                out.setdefault(m.group(1), {})[m.group(2).lower()] = int(m.group(3)) // (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return out


def shared_total(g):
    return sum(v.get("shared", 0) for v in g.values())


# ---------------------------------------------------------------- one llama-server
class Server:
    def __init__(self, preset, cfg, np_, c, stamp, need):
        self.preset, self.np, self.c, self.slot, self.need = preset, np_, c, c // np_, need
        self.warmed = set()
        cmd, env, _, _ = router.serve_cmd(preset, cfg, PORT)
        cmd[cmd.index("-c") + 1] = str(c)
        cmd[cmd.index("-np") + 1] = str(np_)
        self.con_path = HERE / f"concurrency-sweep-llamacpp-{preset}-{stamp}-np{np_}-c{c}.console.log"
        before = gpu_mem()
        self.info = {"np": np_, "c": c, "slot_tokens": c // np_, "command": " ".join(cmd), "console": self.con_path.name,
                     "shared_mib_before": shared_total(before), "gpu_before": before}
        say(f"{preset}: starting llama-server -np {np_} -c {c:,} on :{PORT} -> {self.con_path.name}")
        self.con = self.con_path.open("w", encoding="utf-8")
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.con, stderr=subprocess.STDOUT, cwd=str(HERE))
        self.info["pid"] = self.proc.pid

    def wait_ready(self, limit=900):
        """None when ready, else why not"""
        while True:
            if self.proc.poll() is not None:
                tail = self.con_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
                return f"llama-server exited during load (code {self.proc.returncode}): " + " / ".join(tail)[:300]
            try:
                with urllib.request.urlopen(BASE + "/health", timeout=3) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - self.t0 > limit:
                return f"not ready after {limit} s"
            time.sleep(1)
        self.info["ready_s"] = round(time.time() - self.t0, 1)
        try:
            slots = get_json("/slots", 10)
            self.info["n_ctx_slot"] = slots[0].get("n_ctx") if slots else None
            self.info["slots"] = len(slots)
        except Exception as e:  # noqa: BLE001
            self.info["n_ctx_slot"] = None
            self.info["slots_error"] = str(e)[:200]
        txt = self.con_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"n_ctx_slot = (\d+)", txt)
        if m and not self.info.get("n_ctx_slot"):
            self.info["n_ctx_slot"] = int(m.group(1))
        if self.info.get("slots") and self.info["slots"] < self.np:
            return f"came up with {self.info['slots']} slots, not {self.np}"
        # 2026-10-22 00:48: llama.cpp caps a slot at the model's training context (7B: 33024 -> 32768); that is fine
        # as long as context + gen still fits (32640 + 128 = 32768, as vLLM's max_len 32768 holds it)
        if self.info.get("n_ctx_slot") and self.info["n_ctx_slot"] < self.need:
            return f"slots came up with n_ctx {self.info['n_ctx_slot']}, less than the {self.need} needed"
        if self.info.get("n_ctx_slot") and self.info["n_ctx_slot"] < self.slot:
            self.info["capped"] = f"slot capped by llama.cpp from {self.slot} to {self.info['n_ctx_slot']} (training context)"
            self.slot = self.info["n_ctx_slot"]
            self.info["slot_tokens"] = self.slot
        return None

    def spilled(self, when):
        """None, or why the cards count as spilled into system RAM"""
        g = gpu_mem()
        self.info[f"gpu_{when}"] = g
        # only the two adapters this server loaded onto (the most dedicated memory now): the integrated GPU's shared
        # memory moves with the desktop and is not ours
        b60 = sorted(g, key=lambda k: g[k].get("dedicated", 0), reverse=True)[:2]
        before = self.info["gpu_before"]
        rise = sum(g[k].get("shared", 0) - before.get(k, {}).get("shared", 0) for k in b60)
        self.info[f"shared_rise_mib_{when}"] = rise
        self.info[f"dedicated_mib_{when}"] = {k: g[k].get("dedicated", 0) for k in b60}
        say(f"{self.preset}: -np {self.np} -c {self.c:,}: B60s dedicated "
            f"{', '.join(str(g[k].get('dedicated', 0)) for k in b60)} MiB, shared {rise:+,} MiB after {when}")
        if rise > SPILL_MIB:
            return f"shared GPU memory rose {rise:,} MiB after {when} (spilled into system RAM)"
        return None

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        if self.proc.poll() is None:              # never taskkill a pid that has already exited (it could be reused)
            subprocess.run(["taskkill", "/pid", str(self.proc.pid), "/t", "/f"], capture_output=True)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self.con.close()
        self.info["stopped"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        say(f"{self.preset}: stopped llama-server pid {self.proc.pid}")
        time.sleep(5)


# ---------------------------------------------------------------- output (copied from concurrency-sweep.input.py)
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
    per_row = "; ".join(f"{ctx_label(r['context'])}: -c {r['server']['c']:,} --parallel {r['server']['np']}"
                        f" ({r['server']['slot_tokens']:,} per slot)" for r in b["rows"] if r.get("server"))
    lines = [f"### {b['model']} {b['quant']} — 2x B60, llama.cpp {b['version']} · {b['date']}", "",
             f"Preset `{b['preset']}` ({b['ini']}), run {b['started']}: `{b['gguf']}` ({b['gguf_gb']:.1f} GB, GGUF file "
             f"type {b['file_type']}); the router's llama-server command (Vulkan, both B60s, -sm layer, -ngl 99, "
             f"-ncmoe {b['ncmoe']}, -fa on{', with its mmproj' if b.get('mmproj') else ''}) on :{PORT} with only -c and "
             f"--parallel changed, per row: {per_row or 'none came up'}. Build: `{b['version_line']}`"
             f"{' (' + b['built_with'] + ')' if b.get('built_with') else ''}. {b['gen']} tokens generated per request "
             f"(greedy, ignore_eos)." +
             (f" TEXT-ONLY: started without the preset's mmproj (`{b['mmproj_dropped']}`): with it loaded this build "
              f"rejects /v1/completions token-id prompts (HTTP 400 type_error.302, \"type must be string, but is "
              f"array\"), which made the {b['date']} 01:29 block all —." if b.get("mmproj_dropped") else ""), "",
             "Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; "
             "prefill excluded):", "", table("decode_tps"), "",
             "Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):",
             "", table("raw_tps"), "",
             "— = not run: N is more than the --parallel that fit at this context (N slots of context + gen did not come "
             "up on the two cards: llama-server failed to load, came up short, or spilled into system RAM), or the server "
             "returned an error. * = some request finished short (fewer than gen tokens), or the requests never all "
             "decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the "
             "last first token to the end, still prefill-free but with fewer than N decoding. n/r = not run for time (N "
             "prefills at the measured 1-request prefill rate would pass the per-cell cap); says nothing about fit. "
             "llama.cpp has no preemption (each slot holds its whole context from load)." +
             (" This block: " + "; ".join(notes) + "." if notes else " No cell in this block was marked."), ""]
    if b.get("problems"):
        lines += ["Problems: " + "; ".join(b["problems"]), ""]
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- one preset
def slot_for(tokens, gen):
    return int(math.ceil((tokens + gen + 64) / 256.0) * 256)


def warm(srv, preset, prompts, depth, gen):
    """the row's warm-up request (as in the vLLM sweep), then the spill check; None or why it failed"""
    r = post_stream(preset, prompts.make(depth), gen)
    if r["error"]:
        return f"warm-up failed: {r['error']}"
    srv.info.setdefault("warmups", []).append({"depth": depth, "tokens": r["usage"], "chunk_tokens": sum(r["counts"]),
                                               "count_source": r["count_source"], "timings": r["timings"]})
    srv.warmed.add(depth)
    return srv.spilled(f"warm-up at {depth}")


def sweep(preset, cfg, contexts, requests, gen, run, deadline, cellcap, version):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    gguf, note = resolve_gguf(cfg)
    if gguf is None:
        say(f"{preset}: SKIPPED, {note}")
        run["skipped"].append({"preset": preset, "why": note})
        return
    cfg = dict(cfg)
    cfg["gguf"] = str(gguf)
    mmproj_dropped = cfg.pop("mmproj", None) if NO_MMPROJ else None
    max_len = int(cfg["max_len"])
    try:
        meta = gguf_meta(gguf)
    except Exception as e:  # noqa: BLE001
        meta = {"error": str(e)}
    ft = FILE_TYPES.get(meta.get("general.file_type"), str(meta.get("general.file_type", "?")))
    m = re.search(r"(UD-)?(I?Q\d+_[0-9A-Z]+(?:_[A-Z]+)?|BF16|F16)", gguf.stem, re.I)
    quant = m.group(0) if m else ft
    b = {"preset": preset, "ini": "models.input.ini", "model": gguf.stem, "quant": quant, "file_type": ft,
         "gguf": str(gguf), "gguf_gb": gguf.stat().st_size / 1e9, "gguf_name": meta.get("general.name"),
         "arch": meta.get("general.architecture"), "version": re.sub(r"^version:\s*(\d+)\s", r"b\1 ", version[0]).replace("version: ", ""),
         "version_line": version[0], "built_with": version[1], "date": datetime.date.today().isoformat(),
         "started": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "max_len": max_len, "gen": gen,
         "ncmoe": cfg.get("ncmoe", "0"), "mmproj": cfg.get("mmproj"), "mmproj_dropped": mmproj_dropped, "rows": [], "problems": [], "servers": []}
    if note:
        b["problems"].append(note + " (the router's own start of this preset would fail)")
    run["blocks"].append(b)
    ok, loaded = router_loaded()
    if loaded is not None:
        b["problems"].append(f"router had {loaded} loaded at start; preset not run")
        say(f"{preset}: router has {loaded} loaded, not starting")
        return
    plan = []
    for ctx in contexts:
        if ctx > max_len:
            say(f"{preset}: context {ctx} > max_len {max_len}, row left out")
            continue
        plan.append((ctx, ctx if ctx + gen <= max_len else max_len - gen))
    srv = None
    prompts = None
    fail_total = None          # smallest -c (total tokens) that did not fit
    nmax = max(requests)
    try:
        for i, (ctx, depth) in enumerate(plan):
            row = {"context": ctx, "prompt_tokens": depth, "cells": {}, "tried": []}
            b["rows"].append(row)
            if time.time() > deadline:
                b["problems"].append(f"time budget reached before context {ctx}")
                break
            need = (depth if ctx else 64) + gen
            slot = slot_for(depth if ctx else 64, gen)
            cands = []
            if ctx == 0 and i + 1 < len(plan):
                cands.append((nmax, slot_for(plan[i + 1][1], gen), plan[i + 1][1] + gen))
            cands += [(n, slot, need) for n in sorted(set(requests), reverse=True)]
            for np_, sl, nd in cands:
                if srv and srv.np >= np_ and srv.slot >= need:
                    break                                   # the running server already holds this row
                if fail_total and np_ * sl >= fail_total:
                    row["tried"].append({"np": np_, "c": np_ * sl, "why": f"-c {np_ * sl:,} >= {fail_total:,}, which did not fit"})
                    continue
                if srv:
                    srv.stop()
                    srv = None
                s = Server(preset, cfg, np_, np_ * sl, stamp, nd)
                b["servers"].append(s.info)
                why = s.wait_ready() or s.spilled("load")
                if not why:
                    if prompts is None:
                        prompts = CS.Prompts(Tok())
                    why = warm(s, preset, prompts, depth, gen)
                if why:
                    say(f"{preset}: -np {np_} -c {np_ * sl:,}: {why}")
                    s.info["failed"] = why
                    row["tried"].append({"np": np_, "c": np_ * sl, "why": why})
                    fail_total = min(fail_total or 10 ** 12, np_ * sl)
                    s.stop()
                    continue
                srv = s
                break
            if srv is None or srv.slot < need:
                row["error"] = "no --parallel came up for this context"
                for n in requests:
                    row["cells"][str(n)] = {"n": n, "mark": "—", "reason": "no llama-server came up with even 1 slot of "
                                            f"{slot:,} tokens: " + "; ".join(t["why"] for t in row["tried"])[:300]}
                save_json(run)
                if srv:
                    srv.stop()
                    srv = None
                continue
            row["server"] = {"np": srv.np, "c": srv.c, "slot_tokens": srv.slot, "n_ctx_slot": srv.info.get("n_ctx_slot"),
                             "console": srv.info["console"]}
            try:
                if depth not in srv.warmed:
                    why = warm(srv, preset, prompts, depth, gen)          # reused server: this row's own warm-up
                    if why and why.startswith("warm-up failed"):
                        raise RuntimeError(why)
                    if why:
                        b["problems"].append(f"{ctx_label(ctx)}: {why} on the reused -np {srv.np} server")
                if ctx == 0:
                    row["prompt_tokens"] = len(prompts.make(0))
                t_pf, n_prompt = CS.post_once(preset, prompts.make(depth), 1)              # prefill alone
                row["prefill_s"], row["prefill_tps"], row["prompt_tokens"] = round(t_pf, 3), round(n_prompt / t_pf, 1), n_prompt
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)[:300]
                say(f"{preset}: context {ctx}: warm-up failed: {row['error']}")
                for n in requests:
                    row["cells"][str(n)] = {"n": n, "mark": "—", "reason": "warm-up failed: " + row["error"]}
                save_json(run)
                if not srv.alive():
                    b["problems"].append(f"llama-server died (code {srv.proc.returncode}) at {ctx_label(ctx)} warm-up")
                    srv.stop()
                    srv = None
                continue
            say(f"{preset}: context {ctx} ({row['prompt_tokens']} tokens) on -np {srv.np} -c {srv.c:,}: "
                f"prefill {row['prefill_tps']} tok/s")
            out_of_time = False
            for n in requests:
                if time.time() > deadline:
                    b["problems"].append(f"time budget reached at context {ctx} x {n}")
                    out_of_time = True
                    break
                if n > srv.np:
                    c = {"n": n, "mark": "—", "reason": f"needs --parallel {n} (-c {n * slot:,}); the most that came up "
                                                         f"at this context was --parallel {srv.np} (-c {srv.np * srv.slot:,})"}
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
                                                     f"{c.get('reason', '')} cache_n {c.get('prompt_cache_reused_tokens')}"
                                                     if c["mark"] not in ("—", "n/r") else c["mark"] + " " + c["reason"]))
                save_json(run)
                if not srv.alive():
                    b["problems"].append(f"llama-server died (code {srv.proc.returncode}) at {ctx_label(ctx)} x {n}")
                    srv.stop()
                    srv = None
                    break
            if out_of_time:
                break
    finally:
        if srv:
            srv.stop()
        b["finished"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        save_json(run)
        write_block(b, requests)


def main():
    if len(sys.argv) < 2:
        print(__doc__.split("\n\n")[1])
        return 2
    ids = sys.argv[1].replace(",", "+").split("+")
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    contexts = [int(x) for x in kv.get("contexts", "0,8192,32768,65536").replace("+", ",").split(",")]
    requests = [int(x) for x in kv.get("requests", "1,2,4,8,16").replace("+", ",").split(",")]
    gen = int(kv.get("gen", "128"))
    global NO_MMPROJ
    NO_MMPROJ = kv.get("mmproj", "1").lower() in ("0", "no", "off", "false")
    deadline = T0_RUN + float(kv.get("budget", "180")) * 60
    cellcap = float(kv.get("cellcap", "400"))
    _, models = router.read_ini()
    run = {"run_id": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:], "contexts": contexts,
           "requests": requests, "gen": gen, "port": PORT, "cellcap": cellcap, "no_mmproj": NO_MMPROJ, "blocks": [],
           "skipped": []}
    todo = []
    for p in ids:
        if p not in models:
            run["skipped"].append({"preset": p, "why": "no such preset in models.input.ini"})
        elif models[p].get("engine", "vllm") != "llamacpp":
            run["skipped"].append({"preset": p, "why": f"engine = {models[p].get('engine', 'vllm')}: no llama.cpp (GGUF) "
                                                      f"preset of this model in models.input.ini"})
        else:
            todo.append(p)
    for s in run["skipped"]:
        say(f"skipping {s['preset']}: {s['why']}")
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3):
            say(f"something already answers on :{PORT}; not starting (it is not ours)")
            return 1
    except urllib.error.HTTPError:
        say(f"something already answers on :{PORT}; not starting (it is not ours)")
        return 1
    except Exception:  # noqa: BLE001
        pass
    gates_ok = bool(todo) and gates(deadline - 20 * 60)
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n## Run {run['run_id']} — `{' '.join(sys.argv[1:])}`\n\n"
                + "".join(f"- skipped {s['preset']}: {s['why']}\n" for s in run["skipped"])
                + ("" if gates_ok or not todo else f"- gates never opened before the time limit; nothing run "
                   f"({', '.join(todo)})\n") + "\n")
    if not gates_ok:
        run["skipped"] += [{"preset": p, "why": "gates did not open"} for p in todo]
        save_json(run)
        return 1
    version = llama_version()
    run["llama_server_version"] = version
    say(f"llama-server: {version[0]} {version[1]}")
    save_json(run)
    for i, p in enumerate(todo):
        share = (deadline - time.time()) / (len(todo) - i)
        try:
            sweep(p, models[p], contexts, requests, gen, run, time.time() + share, cellcap, version)
        except Exception as e:  # noqa: BLE001
            say(f"{p}: FAILED - {type(e).__name__}: {e}\n{traceback.format_exc()}")
            run["skipped"].append({"preset": p, "why": f"failed: {type(e).__name__}: {e}"[:300]})
            save_json(run)
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
