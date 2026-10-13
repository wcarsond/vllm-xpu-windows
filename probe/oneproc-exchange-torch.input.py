# oneproc-exchange-torch.input.py - the one-process exchange under the baton, at torch level, before any vLLM:
# two rank threads, one card each, a 28-layer fake decode step (norm, two matmuls of a half-width MLP, all_reduce,
# residual), hidden 3584 bf16 (the 7B's decode all_reduce: 7 KB).
#   eager   each rank runs the step eagerly, handing the baton over after every all_reduce
#   graph   each rank records its step as an XPU graph (handing over during capture too), then replays it every step and
#           hands over after each replay
# Checks: both cards end every step with identical bytes, and they equal a reference that adds the two cards' partial
# sums with torch on one card (bit-exact). Times: us per step against one card alone (no exchange), and the baton's
# hand-overs per step.
# Usage:  .venv312\Scripts\python.exe probe\oneproc-exchange-torch.input.py <eager|graph|both>
#         (appends to probe\oneproc-exchange-torch.output.txt; oneproc-exchange-torch.input.cmd runs it under a timeout)
# Why (2026-10-13): stage 2 of the one-process executor - the exchange and the baton have to work and be fast with
# torch's own streams, allocator and graphs before vLLM is put on top.
import faulthandler, os, pathlib, sys, threading, time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "vllm"))
from vllm.distributed.device_communicators import xpu_oneproc as op  # noqa: E402

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("oneproc-exchange-torch.output.txt")
LAYERS, STEPS, H, F = 28, 200, 3584, 2304
MODE = sys.argv[1] if len(sys.argv) > 1 else "both"
faulthandler.dump_traceback_later(150, exit=True)


def say(s):
    print(s, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def weights(rank):
    g = torch.Generator().manual_seed(1234 + rank)
    d = torch.device("xpu", rank)
    w1 = (torch.randn(H, F, generator=g) * 0.02).to(torch.bfloat16).to(d)
    w2 = (torch.randn(F, H, generator=g) * 0.02).to(torch.bfloat16).to(d)
    return w1, w2


def x0(rank):
    g = torch.Generator().manual_seed(99)
    return torch.randn(1, H, generator=g).to(torch.bfloat16).to(torch.device("xpu", rank))


def layer_partial(x, w1, w2):
    y = x * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
    return torch.nn.functional.silu(y @ w1) @ w2


def step(x, w1, w2, ex, record=None, sums=None):
    for _ in range(LAYERS):
        p = layer_partial(x, w1, w2)
        if record is not None:
            record.append(p.clone())
        p = ex.all_reduce(p)
        op.yield_baton()
        if sums is not None:
            sums.append(p.clone())
        x = x + p
    return x


class Rank(threading.Thread):
    def __init__(self, rank, fn, res):
        super().__init__(name=f"rank{rank}")
        self.rank, self.fn, self.res = rank, fn, res
        self.err = None

    def run(self):
        op.set_thread_rank(self.rank)
        torch.xpu.set_device(self.rank)
        b = op.baton()
        b.acquire(self.rank)
        try:
            self.res[self.rank] = self.fn(self.rank)
        except BaseException as e:  # noqa: BLE001
            self.err = e
        finally:
            b.finish(self.rank)


def both(fn):
    res = {}
    b = op.baton()
    b.begin([0, 1])
    ts = [Rank(r, fn, res) for r in (0, 1)]
    for t in ts: t.start()
    for t in ts: t.join()
    for t in ts:
        if t.err is not None:
            raise t.err
    return res


def main():
    ex = op.pair()
    W = {}

    def setup(r):
        W[r] = weights(r)
        torch.xpu.synchronize(r)
        return True
    both(setup)

    # one card alone, no exchange: the floor
    def alone(r):
        if r != 0:
            return None
        w1, w2 = W[0]
        x = x0(0)
        for _ in range(5):
            for _ in range(LAYERS):
                x = x + layer_partial(x, w1, w2)
        torch.xpu.synchronize(0)
        t0 = time.perf_counter()
        x = x0(0)
        for _ in range(STEPS):
            for _ in range(LAYERS):
                x = x + layer_partial(x, w1, w2)
        torch.xpu.synchronize(0)
        return 1e6 * (time.perf_counter() - t0) / STEPS
    t_alone = both(alone)[0]

    # correctness: one eager step, recording each layer's partials; the reference sums them with torch on card 0
    rec = {0: [], 1: []}

    def check(r):
        w1, w2 = W[r]
        x = step(x0(r), w1, w2, ex, record=rec[r])
        torch.xpu.synchronize(r)
        return x
    out = both(check)
    same = torch.equal(out[0].cpu(), out[1].cpu())
    xr = x0(0)
    for p0, p1 in zip(rec[0], rec[1]):
        xr = xr + (p0 + p1.to(0))
    exact = torch.equal(xr.cpu(), out[0].cpu())
    say(f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  {LAYERS} layers, hidden {H} bf16, {STEPS} steps")
    say(f"correctness: cards identical {same}, equal to the torch reference {exact} "
        f"(max diff {(xr.float().cpu() - out[0].float().cpu()).abs().max().item():.3g}); status {ex.status()}")
    say(f"one card alone, no exchange: {t_alone:.0f} us per step")

    if MODE in ("eager", "both"):
        def eager(r):
            w1, w2 = W[r]
            x = x0(r)
            for _ in range(5):
                x = step(x, w1, w2, ex)
            torch.xpu.synchronize(r)
            op.yield_baton()
            h0 = op.baton().handoffs
            t0 = time.perf_counter()
            x = x0(r)
            for _ in range(STEPS):
                x = step(x, w1, w2, ex)
            torch.xpu.synchronize(r)
            return 1e6 * (time.perf_counter() - t0) / STEPS, op.baton().handoffs - h0
        e = both(eager)
        say(f"eager, two rank threads under the baton: {e[0][0]:.0f} / {e[1][0]:.0f} us per step "
            f"({(e[0][1] + 0) / STEPS:.0f} hand-overs per step); status {ex.status()}")

    if MODE in ("graph", "both"):
        G, GP = {}, {}

        def capture(r):
            w1, w2 = W[r]
            xin = x0(r)
            s = torch.xpu.Stream(device=torch.device("xpu", r))
            s.wait_stream(torch.xpu.current_stream(r))
            with torch.xpu.stream(s):
                for _ in range(3):
                    step(xin, w1, w2, ex)
            torch.xpu.current_stream(r).wait_stream(s)
            torch.xpu.synchronize(r)
            g = torch.xpu.XPUGraph()
            # each card its own capture stream: torch.xpu.graph's default is ONE class-wide stream made on whichever card
            # used it first, so card 1 recorded on card 0's stream and got an empty graph (2026-10-13 16:55)
            parts, sums = [], []
            # OPX_REC = both (default) | parts | sums | none: what the graph clones per layer (2026-10-13 17:00: with no
            # clones the replayed step was 0.5 off the eager one; with both clones exact)
            rec_kind = os.environ.get("OPX_REC", "both")
            with torch.xpu.graph(g, stream=s):
                y = step(xin, w1, w2, ex, record=parts if rec_kind in ("both", "parts") else None,
                         sums=sums if rec_kind in ("both", "sums") else None)
            torch.xpu.synchronize(r)
            G[r] = (g, xin, y)
            GP[r] = (parts, sums)
            return ex.counts(r)
        cnt = both(capture)

        def replay_check(r):
            g, xin, y = G[r]
            xin.copy_(x0(r))
            g.replay()
            op.yield_baton()
            torch.xpu.synchronize(r)
            return y.clone()
        gy = both(replay_check)
        reps = [both(replay_check)[0] for _ in range(4)]
        say(f"graph clones {os.environ.get('OPX_REC', 'both')}: five replays from the same input, max diff to the eager "
            f"step: {[round((v.float().cpu() - out[0].float().cpu()).abs().max().item(), 4) for v in [gy[0]] + reps]}")
        # per layer: is the graph's sum the torch sum of the graph's own partials (both cards)?
        bad = []
        for i in range(LAYERS if GP[0][0] and GP[0][1] else 0):
            ref = GP[0][0][i] + GP[1][0][i].to(0)
            for r in (0, 1):
                got = GP[r][1][i].to(0)
                if not torch.equal(ref, got):
                    bad.append((i, r, (ref.float() - got.float()).abs().max().item(),
                                int((ref != got).sum().item())))
        say(f"graph per-layer sums vs torch sum of the graph's own partials: "
            f"{'all exact' if not bad else 'WRONG at (layer, card, max diff, words) ' + str(bad[:8])}")
        gsame = torch.equal(gy[0].cpu(), gy[1].cpu())
        gexact = torch.equal(gy[0].cpu(), out[0].cpu())
        gdiff = (gy[0].float().cpu() - out[0].float().cpu()).abs().max().item()
        # the same graph-vs-eager comparison on ONE card with no exchange at all, to tell the exchange from the kernels
        def solo(r):
            if r != 0:
                return None
            w1, w2 = W[0]
            xi = x0(0)
            def plain(x):
                for _ in range(LAYERS):
                    x = x + layer_partial(x, w1, w2)
                return x
            e = plain(xi)
            s0 = torch.xpu.Stream(device=torch.device("xpu", 0))
            s0.wait_stream(torch.xpu.current_stream(0))
            with torch.xpu.stream(s0):
                plain(xi)
            torch.xpu.current_stream(0).wait_stream(s0)
            torch.xpu.synchronize(0)
            g0 = torch.xpu.XPUGraph()
            with torch.xpu.graph(g0, stream=s0):
                yy = plain(xi)
            g0.replay()
            torch.xpu.synchronize(0)
            return (yy.float() - e.float()).abs().max().item()
        sd = both(solo)[0]

        def replay(r):
            g, xin, y = G[r]
            for _ in range(5):
                g.replay(); op.yield_baton()
            torch.xpu.synchronize(r)
            op.yield_baton()
            t0 = time.perf_counter()
            for _ in range(STEPS):
                g.replay()
                op.yield_baton()
            torch.xpu.synchronize(r)
            return 1e6 * (time.perf_counter() - t0) / STEPS
        gt = both(replay)
        say(f"graph: counts after capture card0 {cnt[0]}, card1 {cnt[1]} (small, recorded, big); replay: cards identical "
            f"{gsame}, equal to the eager step {gexact} (max diff {gdiff:.3g}; one card, no exchange, graph vs eager max diff {sd:.3g}); {gt[0]:.0f} / {gt[1]:.0f} us per step; status {ex.status()}")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
