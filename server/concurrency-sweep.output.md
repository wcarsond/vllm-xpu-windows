
## Run 2026-10-21 23:25:26 — `qwen2.5-coder-7b contexts=0,8192 requests=1,4 gen=64`


### Qwen2.5-Coder-7B-Instruct bfloat16 — 2x B60, vLLM dev · 2026-10-21

Preset `qwen2.5-coder-7b` (models.input.ini), run 2026-10-21 23:25: oneproc, full graphs, max_len 32768, util 0.90, max_num_seqs 16, KV cache ? tokens, 64 tokens generated per request (greedy, ignore_eos), ready in 610 s.

Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; prefill excluded):

| context \ requests | 1 | 4 | prefill tok/s (1 req) |
|---|---:|---:|---:|
| 0 (31) | 56.7 | 217.0 | 452 |
| 8K | 54.0 | 189.0 | 7,235 |

Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):

| context \ requests | 1 | 4 | prefill tok/s (1 req) |
|---|---:|---:|---:|
| 0 (31) | 54.4 | 199.6 | 452 |
| 8K | 27.8 | 43.5 | 7,235 |

— = not run: N x (context + gen) is more than the KV cache vLLM reported, or the server returned an error. * = some request finished short or was preempted, or the requests never all decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, still prefill-free but with fewer than N decoding. No cell in this block was marked.


## Run 2026-10-21 23:37:18 — `qwen3.6-27b-gptq+qwen3.5-27b+qwen2.5-coder-7b budget=115 cellcap=400`


### Qwen3.6-27B-GPTQ-Int4 GPTQ int4 (bfloat16 activations) — 2x B60, vLLM 0.1.dev1+g276fbcff2.d20261010.xpu · 2026-10-21

Preset `qwen3.6-27b-gptq` (models-sweep.input.ini), run 2026-10-21 23:37: oneproc, full graphs, max_len 65536, util 0.90, max_num_seqs 16, KV cache 164736 tokens, 128 tokens generated per request (greedy, ignore_eos), ready in 102 s.

Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; prefill excluded):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (32) | 29.1 | 55.9 | 104.1 | 79.0 | 160.1 | 238 |
| 8K | 28.3 | 53.6 | 96.8 | 81.0 | 73.8* | 1,906 |
| 32K | 26.8 | 48.1 | 82.1 | — | — | 1,712 |
| 64K (65,408) | 24.6 | 43.3 | — | — | — | 1,502 |

Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (32) | 28.5 | 52.4 | 98.6 | 75.5 | 156.7 | 238 |
| 8K | 14.6 | 19.2 | 22.9 | 22.4 | 23.9* | 1,906 |
| 32K | 5.4 | 5.8 | 6.1 | — | — | 1,712 |
| 64K (65,408) | 2.6 | 2.7 | — | — | — | 1,502 |

— = not run: N x (context + gen) is more than the KV cache vLLM reported, or the server returned an error. * = some request finished short or was preempted, or the requests never all decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, still prefill-free but with fewer than N decoding. n/r = not run for time (N prefills at the measured 1-request prefill rate would pass the per-cell cap); says nothing about fit. This block: 8K x 16: * 2 preemption(s); requests never all decoded together; 32K x 8: — needs 263,168 KV tokens, cache holds 164,736; 32K x 16: — needs 526,336 KV tokens, cache holds 164,736; 64K x 4: — needs 262,144 KV tokens, cache holds 164,736; 64K x 8: — needs 524,288 KV tokens, cache holds 164,736; 64K x 16: — needs 1,048,576 KV tokens, cache holds 164,736.

### Qwen3.5-27B-GPTQ-Int4 GPTQ int4 (bfloat16 activations) — 2x B60, vLLM 0.1.dev1+g276fbcff2.d20261010.xpu · 2026-10-21

Preset `qwen3.5-27b` (models.input.ini), run 2026-10-21 23:50: oneproc, full graphs, max_len 65536, util 0.90, max_num_seqs 16, KV cache 164736 tokens, 128 tokens generated per request (greedy, ignore_eos), ready in 1659 s.

Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; prefill excluded):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (32) | 29.1 | 55.9 | 104.1 | 76.6 | 145.5 | 232 |
| 8K | 28.3 | 53.4 | 97.3 | 78.8 | 72.6* | 1,924 |
| 32K | 26.7 | 48.1 | 81.5 | — | — | 1,717 |
| 64K (65,408) | 24.6 | 43.3 | — | — | — | 1,506 |

Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (32) | 28.4 | 52.7 | 98.7 | 73.5 | 142.4 | 232 |
| 8K | 14.6 | 19.3 | 23.0 | 22.3 | 24.0* | 1,924 |
| 32K | 5.4 | 5.8 | 6.1 | — | — | 1,717 |
| 64K (65,408) | 2.6 | 2.7 | — | — | — | 1,506 |

— = not run: N x (context + gen) is more than the KV cache vLLM reported, or the server returned an error. * = some request finished short or was preempted, or the requests never all decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, still prefill-free but with fewer than N decoding. n/r = not run for time (N prefills at the measured 1-request prefill rate would pass the per-cell cap); says nothing about fit. This block: 8K x 16: * 2 preemption(s); requests never all decoded together; 32K x 8: — needs 263,168 KV tokens, cache holds 164,736; 32K x 16: — needs 526,336 KV tokens, cache holds 164,736; 64K x 4: — needs 262,144 KV tokens, cache holds 164,736; 64K x 8: — needs 524,288 KV tokens, cache holds 164,736; 64K x 16: — needs 1,048,576 KV tokens, cache holds 164,736.

Problems: folder moved: C:\Users\claude\models-hf\Qwen3.5-27B-GPTQ-Int4 -> F:\models-hf\Qwen3.5-27B-GPTQ-Int4 (Qwen3.5-27B-GPTQ-Int4.moved-to-F.txt) (the router's own start of this preset would fail)

### Qwen2.5-Coder-7B-Instruct bfloat16 — 2x B60, vLLM 0.1.dev1+g276fbcff2.d20261010.xpu · 2026-10-22

Preset `qwen2.5-coder-7b` (models.input.ini), run 2026-10-22 00:29: oneproc, full graphs, max_len 32768, util 0.90, max_num_seqs 16, KV cache 490880 tokens, 128 tokens generated per request (greedy, ignore_eos), ready in 596 s.

Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; prefill excluded):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (31) | 56.6 | 111.8 | 215.3 | 412.1 | 466.4 | 499 |
| 8K | 53.9 | 103.2 | 188.4 | 330.0 | 458.6 | 7,210 |
| 32K (32,640) | 49.2 | 87.4 | 143.6 | 205.1 | — | 5,417 |

Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (31) | 55.6 | 108.1 | 207.6 | 379.2 | 451.6 | 499 |
| 8K | 36.7 | 53.8 | 70.8 | 84.5 | 91.8 | 7,210 |
| 32K (32,640) | 14.9 | 16.9 | 18.2 | 19.1 | — | 5,417 |

— = not run: N x (context + gen) is more than the KV cache vLLM reported, or the server returned an error. * = some request finished short or was preempted, or the requests never all decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, still prefill-free but with fewer than N decoding. n/r = not run for time (N prefills at the measured 1-request prefill rate would pass the per-cell cap); says nothing about fit. This block: 32K x 16: — needs 524,288 KV tokens, cache holds 490,880.

