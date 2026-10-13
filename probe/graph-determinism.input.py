# graph-determinism.input.py - does a replayed XPU graph of plain torch ops (norm, oneDNN matmuls, silu, residual) give
# the eager answer every time, on ONE card with no exchange? Replays the 28-layer fake decode step N times from the same
# input and prints each replay's max difference to the eager step.
#   plain    the step as is
#   sync     the same, with torch.xpu.synchronize() after every replay (no back-to-back replays)
#   nomm     matmuls replaced by elementwise ops (is it oneDNN?)
#   presync  synchronize between the input copy and the replay (is the copy racing the replay?)
#   samestr  the input copy made on the capture stream itself
#   eager    no graph: the eager step N times against the first eager step (is the matmul itself order-dependent?)
# Usage:  .venv312\Scripts\python.exe probe\graph-determinism.input.py [plain|sync|nomm] [replays]
#         (appends one line per case to probe\graph-determinism.output.txt)
# Why (2026-10-13 17:00): in the one-process exchange test, replays of a graph from the same input gave answers 0 to 1.0
# off the eager step (both cards always agreeing), while every layer's sum matched the graph's own partials exactly: the
# partials themselves came out different. The two-process graph test (server/graph-test) had no matmuls in its graph, so
# this was never covered.
import pathlib, sys, time

import torch

OUT = pathlib.Path(__file__).with_name("graph-determinism.output.txt")
LAYERS, H, F = 28, 3584, 2304
CASE = sys.argv[1] if len(sys.argv) > 1 else "plain"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

d = torch.device("xpu", 0)
g = torch.Generator().manual_seed(1234)
w1 = (torch.randn(H, F, generator=g) * 0.02).to(torch.bfloat16).to(d)
w2 = (torch.randn(F, H, generator=g) * 0.02).to(torch.bfloat16).to(d)
x0 = torch.randn(1, H, generator=torch.Generator().manual_seed(99)).to(torch.bfloat16).to(d)


def step(x):
    for _ in range(LAYERS):
        y = x * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
        if CASE == "nomm":
            y = torch.nn.functional.silu(y * 1.01) * 0.5
        else:
            y = torch.nn.functional.silu(y @ w1) @ w2
        x = x + y
    return x


eager = step(x0.clone())
if CASE == "eager":
    torch.xpu.synchronize(d)
    diffs = [round((step(x0.clone()).float() - eager.float()).abs().max().item(), 4) for _ in range(N)]
    line = f"{time.strftime('%H:%M:%S')} eager {N} eager steps, max diff to the first each: {diffs}"
    print(line, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    sys.exit(0)
eager2 = step(x0.clone())
torch.xpu.synchronize(d)
xin = x0.clone()
s = torch.xpu.Stream(device=d)
s.wait_stream(torch.xpu.current_stream(d))
with torch.xpu.stream(s):
    for _ in range(3):
        step(xin)
torch.xpu.current_stream(d).wait_stream(s)
torch.xpu.synchronize(d)
gr = torch.xpu.XPUGraph()
with torch.xpu.graph(gr, stream=s):
    y = step(xin)
torch.xpu.synchronize(d)
diffs = []
for _ in range(N):
    if CASE == "samestr":
        with torch.xpu.stream(s):
            xin.copy_(x0)
    else:
        xin.copy_(x0)
    if CASE == "presync":
        torch.xpu.synchronize(d)
    gr.replay()
    if CASE == "sync":
        torch.xpu.synchronize(d)
    diffs.append(round((y.float() - eager.float()).abs().max().item(), 4))
line = (f"{time.strftime('%H:%M:%S')} {CASE:5} eager twice equal {torch.equal(eager, eager2)}; {N} replays, max diff "
        f"to eager each: {diffs}")
print(line, flush=True)
with OUT.open("a", encoding="utf-8") as f:
    f.write(line + "\n")
