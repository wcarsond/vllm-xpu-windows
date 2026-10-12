# compiled-embedding.input.py - does torch.compile (Inductor, Triton on XPU) survive vLLM's tensor-parallel embedding on one B60?
# Usage:  .venv312\Scripts\python.exe probe\compiled-embedding.input.py      (writes probe\compiled-embedding.output.txt)
#         run through probe\compiled-embedding.input.cmd so Inductor finds cl.exe (its launcher stubs are compiled with MSVC).
# Why (2026-10-11 10:58): the two-card 7B with torch.compile died at its first all_reduce - the embedding's - with the card's
# queue already poisoned (UR_RESULT_ERROR_UNKNOWN on the copy out, or a hang with the GPU exchange), while one card compiled
# runs fine. The tensor-parallel embedding is the one piece one card never runs: each rank holds a vocabulary slice and masks
# the ids outside it (vocab_parallel_embedding.py). This reproduces that math alone, eager and compiled, with the 7B's split
# (151936 / 2 = 75968 rows x 3584), and dumps the generated Triton code when asked (TORCH_LOGS=output_code).
import datetime, os, pathlib, sys, time

import torch

out = pathlib.Path(__file__).with_name("compiled-embedding.output.txt")
lines = [f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  torch {torch.__version__}"]

def log(s):
    print(s, flush=True); lines.append(s)

# the masked-gather exactly as vllm/model_executor/layers/vocab_parallel_embedding.py does it for rank r of 2
def get_masked_input_and_mask(input_, start, end):
    vocab_mask = (input_ >= start) & (input_ < end)
    valid_offset = start * vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask

def tp_embedding(input_, weight, start, end):
    masked_input, input_mask = get_masked_input_and_mask(input_, start, end)
    output_parallel = torch.nn.functional.embedding(masked_input, weight)
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    return output_parallel

if __name__ == "__main__":
    dev = torch.device("xpu", 0)
    vocab, hidden, rank = 151936, 3584, int(os.environ.get("EMB_RANK", "1"))
    per = vocab // 2
    start, end = rank * per, (rank + 1) * per
    torch.manual_seed(0)
    weight = torch.randn(per, hidden, dtype=torch.bfloat16, device=dev)
    ids = torch.randint(0, vocab, (21,), dtype=torch.int64, device=dev)
    log(f"rank {rank}: rows {start}..{end}, ids {ids.tolist()[:8]}... ({int(((ids >= start) & (ids < end)).sum())} of 21 in my slice)")
    ref = tp_embedding(ids, weight, start, end)
    torch.xpu.synchronize()
    log(f"eager ok: {tuple(ref.shape)}, zero rows {int((ref == 0).all(dim=-1).sum())}")
    for mode in ("default", "reduce-overhead", "max-autotune-no-cudagraphs"):
        try:
            t0 = time.time()
            fn = torch.compile(tp_embedding, mode=mode, dynamic=False)
            got = fn(ids, weight, start, end)
            torch.xpu.synchronize()
            same = torch.equal(got.cpu(), ref.cpu())
            log(f"compiled ({mode}): ran in {time.time()-t0:.1f} s, matches eager: {same}")
            if not same:
                log(f"   first differing row: {int((got != ref).any(dim=-1).nonzero()[0])}")
        except Exception as e:  # noqa: BLE001
            log(f"compiled ({mode}): FAILED: {type(e).__name__}: {str(e)[:300]}")
    # the whole thing again with the ids as a decode step would see them (batch of 1) and as a prefill chunk (2048)
    for n in (1, 2048):
        ids_n = torch.randint(0, vocab, (n,), dtype=torch.int64, device=dev)
        try:
            fn = torch.compile(tp_embedding, dynamic=True)
            got = fn(ids_n, weight, start, end); torch.xpu.synchronize()
            same = torch.equal(got.cpu(), tp_embedding(ids_n, weight, start, end).cpu())
            log(f"compiled dynamic, {n} ids: matches eager: {same}")
        except Exception as e:  # noqa: BLE001
            log(f"compiled dynamic, {n} ids: FAILED: {type(e).__name__}: {str(e)[:300]}")
    # Round 2 (11:50): vLLM's real function and the real dtypes. In the engine the ids are int32, the padded vocabulary is
    # 152064 (rows per rank 76032), and the generated kernel carries a tl.device_assert on the gather index. Rank 1's
    # generated math is right (verified by hand), so if this faults the fault is in how the card runs that kernel.
    from vllm.model_executor.layers.vocab_parallel_embedding import get_masked_input_and_mask
    import torch._inductor.config as icfg

    def vllm_tp_embedding(input_, weight, start, end, pad, added_start, added_end):
        masked_input, input_mask = get_masked_input_and_mask(input_, start, end, pad, added_start, added_end)
        output_parallel = torch.nn.functional.embedding(masked_input, weight)
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        return output_parallel

    padded_vocab, per_pad = 152064, 76032
    weight_pad = torch.randn(per_pad, hidden, dtype=torch.bfloat16, device=dev)
    for r in (0, 1):
        s, e = r * per_pad, (r + 1) * per_pad
        for n in (21, 512, 8192):
            ids32 = torch.randint(0, vocab, (n,), dtype=torch.int32, device=dev)
            ref = vllm_tp_embedding(ids32, weight_pad, s, e, 0, padded_vocab, padded_vocab)
            for assert_idx in (True, False):
                icfg.assert_indirect_indexing = assert_idx
                try:
                    torch._dynamo.reset()
                    fn = torch.compile(vllm_tp_embedding, dynamic=True)
                    got = fn(ids32, weight_pad, s, e, 0, padded_vocab, padded_vocab)
                    torch.xpu.synchronize()
                    same = torch.equal(got.cpu(), ref.cpu())
                    log(f"vllm math, rank {r}, {n} int32 ids, index assert {assert_idx}: matches eager: {same}")
                except Exception as ex:  # noqa: BLE001
                    log(f"vllm math, rank {r}, {n} int32 ids, index assert {assert_idx}: FAILED: {type(ex).__name__}: {str(ex)[:200]}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
