# attn-direct.input.py - call the chunk-prefill attention kernel directly (no pytest) with controlled lengths and flags, one process per case.
# Usage:  .venv312\Scripts\python.exe probe\attn-direct.input.py      (writes probe\attn-direct.output.txt; ~2 min)
# Why (2026-10-10): every attention test case kills the device context on its first launch except non-paged+non-causal+windowed.
# The windowed path touches one or two KV tiles per query tile; the full and causal paths loop over many. Varying the sequence
# length tells "the loop faults past some tile count" from "the code path faults at once". Each child prints ok/FAIL and the
# error against a CPU reference; the parent tabulates. Shapes follow tests/flash_attn (8 query heads, 2 KV heads when paged).
import datetime, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve()
OUT = HERE.with_name("attn-direct.output.txt")
HQ, HKV = 8, 2
# name: (seq_q, seq_k, causal, window, paged, block_size, head_size)
CASES = {
    "np_full_8":        (8, 8, False, (-1, -1), False, 0, 64),
    "np_full_64":       (64, 64, False, (-1, -1), False, 0, 64),
    "np_full_128":      (128, 128, False, (-1, -1), False, 0, 64),
    "np_full_512":      (512, 512, False, (-1, -1), False, 0, 64),
    "np_full_2048":     (2048, 2048, False, (-1, -1), False, 0, 64),
    "np_win64_8":       (8, 8, False, (64, 64), False, 0, 64),
    "np_win64_512":     (512, 512, False, (64, 64), False, 0, 64),
    "np_win64_2048":    (2048, 2048, False, (64, 64), False, 0, 64),
    "np_causal_8":      (8, 8, True, (-1, -1), False, 0, 64),
    "np_causal_512":    (512, 512, True, (-1, -1), False, 0, 64),
    "np_causal_win_512": (512, 512, True, (64, 64), False, 0, 64),
    "np_full_512_h128": (512, 512, False, (-1, -1), False, 0, 128),
    "np_win64_512_h128": (512, 512, False, (64, 64), False, 0, 128),
    "pg16_causal_13":   (13, 13, True, (-1, -1), True, 16, 64),     # the smoke run's prompt: 13 tokens, Qwen2.5-0.5B head 64
    "pg64_causal_13":   (13, 13, True, (-1, -1), True, 64, 64),     # vLLM chose block 64 for FLASH_ATTN
    "pg64_causal_13_h128": (13, 13, True, (-1, -1), True, 64, 128),
    "pg64_win_13":      (13, 13, False, (64, 64), True, 64, 64),
    "pg64_full_13":     (13, 13, False, (-1, -1), True, 64, 64),
}


def reference(q, k, v, causal, window, scale):
    # q [S, HQ, D], k/v [S, HKV_or_HQ, D] on CPU, fp32; bottom-right alignment is moot because seq_q == seq_k here.
    import torch
    S = q.shape[0]
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, dim=1) if rep > 1 else k
    v = v.repeat_interleave(rep, dim=1) if rep > 1 else v
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    i = torch.arange(S)[:, None]; j = torch.arange(S)[None, :]
    allow = torch.ones(S, S, dtype=torch.bool)
    if causal:
        allow &= j <= i
    left, right = window
    if left >= 0:
        allow &= (i - j) <= left
    if right >= 0:
        allow &= (j - i) <= right
    s = s.masked_fill(~allow, float("-inf"))
    p = torch.softmax(s, dim=-1).nan_to_num(0.0)
    return torch.einsum("hqk,khd->qhd", p, v)


def run_case(name):
    import torch
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
    seq_q, seq_k, causal, window, paged, block, hd = CASES[name]
    torch.manual_seed(4242)
    dev = "xpu:0"
    scale = hd ** -0.5
    q = torch.randn(seq_q, HQ, hd, dtype=torch.bfloat16, device=dev)
    cu_q = torch.tensor([0, seq_q], dtype=torch.int32, device=dev)
    if paged:
        nblocks = 256
        kc = torch.randn(nblocks, block, HKV, hd, dtype=torch.bfloat16, device=dev)
        vc = torch.randn_like(kc)
        nb = (seq_k + block - 1) // block
        table = torch.randperm(nblocks, device=dev)[:nb].to(torch.int32)[None, :]
        seqused = torch.tensor([seq_k], dtype=torch.int32, device=dev)
        out = flash_attn_varlen_func(q, kc, vc, seq_q, cu_q, seq_k, seqused_k=seqused, softmax_scale=scale,
                                     causal=causal, window_size=list(window), block_table=table, fa_version=2)
        torch.xpu.synchronize()
        # gather the logical K/V for the reference
        idx = table[0].long()
        k_log = kc[idx].reshape(nb * block, HKV, hd)[:seq_k]
        v_log = vc[idx].reshape(nb * block, HKV, hd)[:seq_k]
    else:
        k_log = torch.randn(seq_k, HQ, hd, dtype=torch.bfloat16, device=dev)
        v_log = torch.randn_like(k_log)
        cu_k = torch.tensor([0, seq_k], dtype=torch.int32, device=dev)
        out = flash_attn_varlen_func(q, k_log, v_log, seq_q, cu_q, seq_k, cu_seqlens_k=cu_k, softmax_scale=scale,
                                     causal=causal, window_size=list(window), fa_version=2)
        torch.xpu.synchronize()
    out = out[0] if isinstance(out, tuple) else out
    ref = reference(q.float().cpu(), k_log.float().cpu(), v_log.float().cpu(), causal, window, scale)
    err = (out.float().cpu() - ref).abs().max().item()
    return f"ok   max abs err {err:.4f}"


if len(sys.argv) > 1:
    try:
        print(run_case(sys.argv[1]))
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:140]}")
    sys.exit(0)

lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", f"{'case':<22} {'q':>5} {'k':>5} causal window    paged blk head  result"]
for name, (sq, sk, causal, window, paged, block, hd) in CASES.items():
    r = subprocess.run([sys.executable, str(HERE), name], capture_output=True, text=True, timeout=300)
    err = r.stderr.strip().splitlines()
    body = r.stdout.strip() or (err[-1] if err else "(no output)")
    # The wrapper swallows "kernel tuple not compiled" and runs a PyTorch reference instead; that is not the kernel passing.
    if "falling back to PyTorch reference" in (r.stderr + r.stdout):
        cfg = next((l.strip() for l in (r.stderr + r.stdout).splitlines() if l.strip()[:1].isdigit() and l.count(",") == 5), "?")
        body = f"FALLBACK (kernel not compiled: {cfg}); reference {body}"
    lines.append(f"{name:<22} {sq:>5} {sk:>5} {str(causal):<6} {str(window):<9} {str(paged):<5} {block:>3} {hd:>4}  {body}")
text = "\n".join(lines); print(text); OUT.write_text(text + "\n", encoding="utf-8")
