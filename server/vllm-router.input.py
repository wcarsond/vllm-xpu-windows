# vllm-router.input.py - the front door on port 11434: offers every model id in models.input.ini, runs ONE vLLM behind it,
# loads the model a request names and swaps when a request names another. OpenAI-compatible, streaming passed through.
# Usage:  started by run-router.input.cmd (which sets the MSVC environment vLLM's Triton needs); that is what the
#         vllm-b60 service runs. Direct:  .venv312\Scripts\python.exe server\vllm-router.input.py [port]
# Behaviour:
#   GET  /v1/models            every id from the ini (re-read each time), "loaded": true on the one on the cards
#   POST /v1/*                 body's "model" picks the model; if it is not the loaded one, the loaded one is stopped and the
#                              named one started (2.5 min for a 27B). A streaming request gets the SSE headers at once and a
#                              ": loading <id>" comment every 5 s until the model answers, so editors keep the connection.
#   GET  /router/status        what is loaded, since when, and the last switch time
# Logs: router.console.log (this program) and vllm-<id>.console.log (each model's vLLM output), beside this file.
# Why (2026-10-12, Carson: "make it so I can switch models just like we currently do ... switch models via VS Code"): the
# llama.cpp router it replaces offered several ids and loaded on demand; vLLM is one process per model, so the swap has to
# live in front of it. VS Code's languageModels.json lists the ids below under http://192.168.0.21:11434/v1.
import configparser, datetime, http.client, http.server, json, os, pathlib, subprocess, sys, threading, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
INI = HERE / "models.input.ini"
LOG = HERE / "router.console.log"
VLLM = HERE.parent / ".venv312" / "Scripts" / "vllm.exe"
MODELS_DIR = pathlib.Path(r"C:\Users\claude\models-hf")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 11434   # importable (service-bench)
DEFAULTS = {"tp": "2", "max_len": "32768", "util": "0.90", "tool_parser": "hermes", "reasoning_parser": "-",
            "think": "0", "dtype": "bfloat16", "executor": "oneproc", "graphs": "full", "max_seqs": "16"}
# executor oneproc (2026-10-13): both cards in one process (vllm/v1/executor/oneproc_executor.py), 15-22% faster decode
# than the two-process mp executor on every model measured, same tokens; only for tp 2, anything else runs mp.
# graphs full: whole-step decode graphs without torch.compile; eager: no graphs (the setting before 2026-10-13).
# max_seqs 16: graph capture needs one Mamba cache block per sequence on the hybrids (27B: 143 blocks at util 0.85, the
# default 256 refused); one user's editor never has 16 requests in flight.
ONEPROC = "vllm.v1.executor.oneproc_executor.OneProcExecutor"


def log(s):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {s}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_ini():
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(INI, encoding="utf-8")
    router = dict(cp["router"]) if cp.has_section("router") else {}
    models = {}
    for sec in cp.sections():
        if sec == "router":
            continue
        d = dict(DEFAULTS)
        d.update(cp[sec])
        models[sec] = d
    return router, models


LLAMA_SERVER = pathlib.Path(r"C:\Users\claude\llama.cpp-b60\vk-allreduce-fa\llama.cpp\build\bin\Release\llama-server.exe")
GGUF_DIR = pathlib.Path(r"C:\Users\claude\models-b60")


def b60_vulkan_devices():
    """The llama.cpp Vulkan device names of the two B60s, asked of llama-server itself in THIS session. Never hard-code
    them: in an interactive session Vulkan0 is the integrated GPU and Vulkan1/2 the B60s, but in session 0 (the service,
    or SSH) the B60s are Vulkan0/1 and Vulkan2 is the integrated GPU - "Vulkan1,Vulkan2" there put half of Qwen3-Next on
    the iGPU and in system RAM, 10 tok/s instead of 70 (2026-10-13 21:22)."""
    import re
    out = subprocess.run([str(LLAMA_SERVER), "--list-devices"], capture_output=True, text=True, timeout=60)
    names = re.findall(r"^\s*(Vulkan\d+):.*B60", out.stdout + out.stderr, re.M)
    if len(names) != 2:
        raise RuntimeError(f"expected two B60s in llama-server --list-devices, found {names}")
    return names


def llamacpp_cmd(model_id, cfg, port):
    """engine = llamacpp (2026-10-13): llama-server on both B60s for a GGUF model, on the same upstream port as vLLM, so
    switching, idle unload and VS Code's view are unchanged. Keys: gguf (under C:\\Users\\claude\\models-b60 or a full
    path), max_len (context), ncmoe (expert layers kept in system RAM, default 0). First user: Qwen3-Next-80B-A3B
    UD-Q4_K_XL - 70 tok/s decode and 1,720 tok/s prompt reading at 64K context, ready in ~17 s; 128K context spilled
    into system RAM and ran at a quarter of that (llama.cpp-b60\\serve-fit-test.output.txt)."""
    gguf = pathlib.Path(cfg["gguf"])
    if not gguf.is_absolute():
        gguf = GGUF_DIR / gguf
    cmd = [str(LLAMA_SERVER), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "--alias", model_id,
           "-dev", ",".join(b60_vulkan_devices()), "-sm", "layer", "-ngl", "99", "-ncmoe", cfg.get("ncmoe", "0"), "-fa", "on",
           "-c", cfg["max_len"], "-np", "1", "--jinja", "-lm", "none"]
    # mmproj: a vision projector GGUF (gemma4-e4b); chat_template_kwargs: e.g. {"enable_thinking":false} for a "nothink"
    # id of a thinking model, for clients that cannot send it (as in the old llama.cpp presets.ini, 2026-10-02)
    if cfg.get("mmproj"):
        mm = pathlib.Path(cfg["mmproj"])
        cmd += ["--mmproj", str(mm if mm.is_absolute() else GGUF_DIR / mm)]
    if cfg.get("chat_template_kwargs"):
        cmd += ["--chat-template-kwargs", cfg["chat_template_kwargs"]]
    env = dict(os.environ)
    env.pop("ZE_AFFINITY_MASK", None)
    env.pop("ONEAPI_DEVICE_SELECTOR", None)
    return cmd, env, gguf, "llama.cpp"


def serve_cmd(model_id, cfg, port):
    """(command line, environment, model folder, executor) for one model's `vllm serve` - the router's, and
    service-bench.input.py's so a benchmark runs exactly what the service runs."""
    if cfg.get("engine", "vllm") == "llamacpp":
        return llamacpp_cmd(model_id, cfg, port)
    folder = pathlib.Path(cfg["folder"])
    if not folder.is_absolute():
        folder = MODELS_DIR / folder
    executor = ONEPROC if cfg["executor"] == "oneproc" and cfg["tp"] == "2" else "mp"
    if cfg["graphs"] == "full":
        graph_args = ["--compilation-config", json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})]
    else:
        graph_args = ["--enforce-eager"]
    cmd = [str(VLLM), "serve", str(folder), "--host", "127.0.0.1", "--port", str(port),
           "--served-model-name", model_id, "--tensor-parallel-size", cfg["tp"], "--dtype", cfg["dtype"],
           "--distributed-executor-backend", executor, *graph_args, "--max-num-seqs", cfg["max_seqs"],
           "--max-model-len", cfg["max_len"],
           "--gpu-memory-utilization", cfg["util"], "--limit-mm-per-prompt", json.dumps({"image": 4}),
           "--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_parser"]]
    if cfg["reasoning_parser"] != "-":
        cmd += ["--reasoning-parser", cfg["reasoning_parser"]]
    if cfg["think"] == "0":
        cmd += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": False})]
    env = dict(os.environ)
    env.pop("ZE_AFFINITY_MASK", None)
    env.pop("ONEAPI_DEVICE_SELECTOR", None)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["PYTHONFAULTHANDLER"] = "1"
    return cmd, env, folder, executor


class Engine:
    """The one vLLM process behind the router."""

    def __init__(self):
        self.lock = threading.Lock()         # held for the whole of a switch
        self.proc = None
        self.model_id = None
        self.ready = False
        self.since = None
        self.last_switch_s = None
        self.last_used = time.time()         # a request arrived, or a model became ready
        self.upstream_port = int(read_ini()[0].get("upstream_port", "11435"))

    def idle_watch(self):
        """Unload the model after [router] idle_unload_s seconds with no request (0 or missing = never)."""
        while True:
            time.sleep(15)
            try:
                limit = float(read_ini()[0].get("idle_unload_s", "0") or 0)
            except ValueError:
                limit = 0
            if limit > 0 and self.model_id and self.ready and time.time() - self.last_used > limit:
                with self.lock:
                    if self.model_id and time.time() - self.last_used > limit:
                        log(f"idle {int(time.time() - self.last_used)} s > {int(limit)} s: unloading {self.model_id}")
                        self.stop()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None:
            log(f"stopping {self.model_id} (pid {self.proc.pid})")
            subprocess.run(["taskkill", "/pid", str(self.proc.pid), "/t", "/f"], capture_output=True)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        self.proc, self.model_id, self.ready, self.since = None, None, False, None

    def start(self, model_id, cfg):
        cmd, env, folder, executor = serve_cmd(model_id, cfg, self.upstream_port)
        con = (HERE / f"vllm-{model_id}.console.log").open("a", encoding="utf-8")
        con.write(f"\n{datetime.datetime.now():%Y-%m-%d %H:%M:%S} start: {' '.join(cmd)}\n")
        con.flush()
        self.proc = subprocess.Popen(cmd, env=env, stdout=con, stderr=subprocess.STDOUT, cwd=str(HERE))
        self.model_id, self.ready, self.since = model_id, False, time.time()
        log(f"starting {model_id} (pid {self.proc.pid}): {folder.name}, tp {cfg['tp']}, max_len {cfg['max_len']}, util {cfg['util']}, "
            f"executor {'oneproc' if executor == ONEPROC else executor}, graphs {cfg['graphs'] if executor != 'llama.cpp' else '-'}")

    def wait_ready(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.alive():
                code = self.proc.returncode if self.proc else None
                log(f"{self.model_id} exited during load, code {code}")
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.upstream_port}/v1/models", timeout=3) as r:
                    if r.status == 200:
                        self.ready = True
                        self.last_used = time.time()
                        self.last_switch_s = round(time.time() - t0, 1)
                        log(f"{self.model_id} ready in {self.last_switch_s} s")
                        return True
            except Exception:
                pass
            time.sleep(2)
        log(f"{self.model_id} not ready after {timeout} s")
        return False

    def ensure(self, model_id, cfg, progress=None):
        """Make model_id the loaded, ready model. Serialised; progress() is called every 5 s while waiting."""
        with self.lock:
            if self.model_id == model_id and self.ready and self.alive():
                return True
            self.stop()
            self.start(model_id, cfg)
            done = {"ok": None}
            t = threading.Thread(target=lambda: done.__setitem__("ok", self.wait_ready()), daemon=True)
            t.start()
            while t.is_alive():
                t.join(5)
                if t.is_alive() and progress:
                    progress()
            if not done["ok"]:
                self.stop()
            return bool(done["ok"])


ENGINE = Engine()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):            # quiet; the router log has what matters
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, data: bytes):
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self):
        router, models = read_ini()
        if self.path.startswith("/v1/models"):
            # context_length / vision / engine (2026-10-13): what scripts\vscode-models needs to write VS Code's list
            data = [{"id": mid, "object": "model", "created": 0, "owned_by": "local",
                     "loaded": (mid == ENGINE.model_id and ENGINE.ready),
                     "context_length": int(cfg["max_len"]),
                     "vision": cfg.get("vision", "0") == "1" or bool(cfg.get("mmproj")),
                     "engine": cfg.get("engine", "vllm"),
                     # order / about (2026-10-14): where the model sits in VS Code's list and the one line shown with it
                     "order": int(cfg.get("order", "99")), "about": cfg.get("about", "")}
                    for mid, cfg in models.items()]
            return self._json(200, {"object": "list", "data": data})
        if self.path.startswith("/router/status"):
            return self._json(200, {"loaded": ENGINE.model_id, "ready": ENGINE.ready,
                                    "since": ENGINE.since and datetime.datetime.fromtimestamp(ENGINE.since).isoformat(timespec="seconds"),
                                    "last_switch_s": ENGINE.last_switch_s, "idle_s": int(time.time() - ENGINE.last_used),
                                    "idle_unload_s": router.get("idle_unload_s", "0"), "models": list(models)})
        if self.path in ("/", "/health", "/v1/health"):
            return self._json(200, {"status": "ok", "loaded": ENGINE.model_id, "ready": ENGINE.ready})
        return self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": {"message": "body is not JSON"}})
        router, models = read_ini()
        ENGINE.last_used = time.time()
        model_id = body.get("model") or ENGINE.model_id or router.get("fallback") or router.get("default")
        if model_id not in models:
            return self._json(404, {"error": {"message": f"unknown model '{model_id}'; known: {', '.join(models)}",
                                              "type": "invalid_request_error"}})
        streaming = bool(body.get("stream"))
        headers_sent = False
        needs_switch = not (ENGINE.model_id == model_id and ENGINE.ready and ENGINE.alive())

        if needs_switch and streaming:
            # commit to the SSE response now and keep the client fed while the model loads
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            headers_sent = True
            self._chunk(f": loading {model_id}\n\n".encode())

        def progress():
            if headers_sent:
                try:
                    self._chunk(f": loading {model_id} {int(time.time() - (ENGINE.since or time.time()))}s\n\n".encode())
                except OSError:
                    pass

        if needs_switch:
            log(f"request for {model_id} while {ENGINE.model_id or 'nothing'} loaded: switching")
            if not ENGINE.ensure(model_id, models[model_id], progress):
                msg = {"error": {"message": f"model '{model_id}' failed to start; see vllm-{model_id}.console.log",
                                 "type": "server_error"}}
                if headers_sent:
                    self._chunk(f"data: {json.dumps(msg)}\n\n".encode())
                    self._chunk(b"data: [DONE]\n\n")
                    self._chunk(b"")
                    self.wfile.write(b"\r\n")
                else:
                    self._json(503, msg)
                return

        # forward to vLLM
        try:
            conn = http.client.HTTPConnection("127.0.0.1", ENGINE.upstream_port, timeout=900)
            fwd = {k: v for k, v in self.headers.items()
                   if k.lower() in ("content-type", "accept", "authorization", "user-agent")}
            fwd["Content-Length"] = str(len(raw))
            conn.request("POST", self.path, body=raw, headers=fwd)
            resp = conn.getresponse()
        except Exception as e:
            msg = {"error": {"message": f"upstream vLLM unreachable: {e}", "type": "server_error"}}
            if headers_sent:
                self._chunk(f"data: {json.dumps(msg)}\n\n".encode())
                self._chunk(b"")
                self.wfile.write(b"\r\n")
            else:
                self._json(502, msg)
            return
        if not headers_sent:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("content-type", "cache-control", "x-request-id"):
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        try:
            while True:
                piece = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not piece:
                    break
                self._chunk(piece)
            self._chunk(b"")
            self.wfile.write(b"\r\n")
            self.wfile.flush()
        except (OSError, ConnectionError):
            pass           # client went away mid-stream; vLLM finishes or aborts on its own
        finally:
            conn.close()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    router, models = read_ini()
    log(f"router starting on 0.0.0.0:{PORT}; models: {', '.join(models)}; upstream {ENGINE.upstream_port}")
    default = router.get("default")
    if default in models:
        threading.Thread(target=lambda: ENGINE.ensure(default, models[default]), daemon=True).start()
    threading.Thread(target=ENGINE.idle_watch, daemon=True).start()
    srv = Server(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    finally:
        ENGINE.stop()
