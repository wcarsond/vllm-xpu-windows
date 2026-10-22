
## Run 2026-10-22 00:32:12 — `qwen2.5-coder-7b-q8+qwen3.6-35b-a3b+qwen3.6-27b+qwen3.5-27b contexts=0,8192,32768,65536 requests=1,2,4,8,16 gen=128 budget=157 cellcap=400`

- skipped qwen3.5-27b: engine = vllm: no llama.cpp (GGUF) preset of this model in models.input.ini

- 2026-10-22 00:48, stopped by the builder: llama.cpp capped the 7B's 32K-row slot at its training context (33,024 -> 32,768) and the script counted that as not fitting, so every --parallel would have shown "—". Fixed (a slot only has to hold context + gen) and rerun below; this run's 0 and 8K cells are kept in the .output.json.


## Run 2026-10-22 00:48:58 — `qwen2.5-coder-7b-q8+qwen3.6-35b-a3b+qwen3.6-27b+qwen3.5-27b contexts=0,8192,32768,65536 requests=1,2,4,8,16 gen=128 budget=148 cellcap=400`

- skipped qwen3.5-27b: engine = vllm: no llama.cpp (GGUF) preset of this model in models.input.ini

### qwen2.5-coder-7b-q8 Q8_0 — 2x B60, llama.cpp 0.5.0-dev (build 55, commit 4fcdeee) · 2026-10-22

Preset `qwen2.5-coder-7b-q8` (models.input.ini), run 2026-10-22 00:49: `D:\models-b60\qwen2.5-coder-7b-q8.gguf` (8.1 GB, GGUF file type Q8_0); the router's llama-server command (Vulkan, both B60s, -sm layer, -ngl 99, -ncmoe 0, -fa on) on :11446 with only -c and --parallel changed, per row: 0: -c 135,168 --parallel 16 (8,448 per slot); 8K: -c 135,168 --parallel 16 (8,448 per slot); 32K: -c 528,384 --parallel 16 (32,768 per slot). Build: `version: 0.5.0-dev (build 55, commit 4fcdeee)` (built with MSVC 19.29.30146.0 for x64). 128 tokens generated per request (greedy, ignore_eos).

Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; prefill excluded):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (31) | 42.9 | 81.8 | 159.3 | 263.2 | 174.9 | 314 |
| 8K | 39.5 | 73.1 | 129.3 | 67.2 | 102.9 | 2,526 |
| 32K (32,640) | 32.6 | 56.1 | 82.1 | 24.9 | n/r | 1,175 |

Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (31) | 41.7 | 79.3 | 151.4 | 250.2 | 171.7 | 314 |
| 8K | 19.9 | 24.7 | 27.9 | 23.3 | 25.0 | 2,526 |
| 32K (32,640) | 4.0 | 3.9 | 3.7 | 3.1 | n/r | 1,175 |

— = not run: N is more than the --parallel that fit at this context (N slots of context + gen did not come up on the two cards: llama-server failed to load, came up short, or spilled into system RAM), or the server returned an error. * = some request finished short (fewer than gen tokens), or the requests never all decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, still prefill-free but with fewer than N decoding. n/r = not run for time (N prefills at the measured 1-request prefill rate would pass the per-cell cap); says nothing about fit. llama.cpp has no preemption (each slot holds its whole context from load). This block: 32K x 16: n/r not run: 16 prefills of 32,640 tokens at 1,175 tok/s would take 444 s (cap 400 s).

Problems: GGUF moved: C:\Users\claude\models-b60\qwen2.5-coder-7b-q8.gguf -> D:\models-b60\qwen2.5-coder-7b-q8.gguf (qwen2.5-coder-7b-q8.gguf.moved-to-D.txt) (the router's own start of this preset would fail)

### Qwen3.6-35B-A3B-UD-Q4_K_XL UD-Q4_K_XL — 2x B60, llama.cpp 0.5.0-dev (build 55, commit 4fcdeee) · 2026-10-22

Preset `qwen3.6-35b-a3b` (models.input.ini), run 2026-10-22 01:03: `C:\Users\claude\models-b60\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.4 GB, GGUF file type Q4_K_M); the router's llama-server command (Vulkan, both B60s, -sm layer, -ngl 99, -ncmoe 0, -fa on) on :11446 with only -c and --parallel changed, per row: 0: -c 135,168 --parallel 16 (8,448 per slot); 8K: -c 135,168 --parallel 16 (8,448 per slot); 32K: -c 528,384 --parallel 16 (33,024 per slot); 64K: -c 1,052,672 --parallel 16 (65,792 per slot). Build: `version: 0.5.0-dev (build 55, commit 4fcdeee)` (built with MSVC 19.29.30146.0 for x64). 128 tokens generated per request (greedy, ignore_eos).

Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; prefill excluded):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (32) | 72.1 | 113.4 | 159.5 | 198.6 | 139.1 | 111 |
| 8K | 68.7 | 101.0 | 138.3 | 122.3 | 58.8 | 2,254 |
| 32K | 59.6 | 86.8 | 112.7 | 96.7 | 114.3* | 1,312 |
| 64K | — | — | — | — | — |  |

Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (32) | 62.8 | 60.6 | 139.4 | 179.2 | 130.9 | 111 |
| 8K | 23.6 | 24.0 | 25.6 | 24.7 | 20.6 | 2,254 |
| 32K | 4.7 | 4.3 | 4.2 | 4.1 | 3.0* | 1,312 |
| 64K | — | — | — | — | — |  |

— = not run: N is more than the --parallel that fit at this context (N slots of context + gen did not come up on the two cards: llama-server failed to load, came up short, or spilled into system RAM), or the server returned an error. * = some request finished short (fewer than gen tokens), or the requests never all decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, still prefill-free but with fewer than N decoding. n/r = not run for time (N prefills at the measured 1-request prefill rate would pass the per-cell cap); says nothing about fit. llama.cpp has no preemption (each slot holds its whole context from load). This block: 32K x 16: * requests never all decoded together; 64K x 1: — warm-up failed: [WinError 10054] An existing connection was forcibly closed by the remote host; 64K x 2: — warm-up failed: [WinError 10054] An existing connection was forcibly closed by the remote host; 64K x 4: — warm-up failed: [WinError 10054] An existing connection was forcibly closed by the remote host; 64K x 8: — warm-up failed: [WinError 10054] An existing connection was forcibly closed by the remote host; 64K x 16: — warm-up failed: [WinError 10054] An existing connection was forcibly closed by the remote host.

### Qwen3.6-27B-UD-Q4_K_XL UD-Q4_K_XL — 2x B60, llama.cpp 0.5.0-dev (build 55, commit 4fcdeee) · 2026-10-22

Preset `qwen3.6-27b` (models.input.ini), run 2026-10-22 01:29: `C:\Users\claude\models-b60\qwen3.6-27b\Qwen3.6-27B-UD-Q4_K_XL.gguf` (17.6 GB, GGUF file type Q4_K_M); the router's llama-server command (Vulkan, both B60s, -sm layer, -ngl 99, -ncmoe 0, -fa on, with its mmproj) on :11446 with only -c and --parallel changed, per row: none came up. Build: `version: 0.5.0-dev (build 55, commit 4fcdeee)` (built with MSVC 19.29.30146.0 for x64). 128 tokens generated per request (greedy, ignore_eos).

Aggregate decode tok/s (all requests' tokens while all were decoding together, divided by that time; prefill excluded):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (0) | — | — | — | — | — |  |
| 8K | — | — | — | — | — |  |
| 32K | — | — | — | — | — |  |
| 64K (65,408) | — | — | — | — | — |  |

Raw aggregate tok/s (all generated tokens / wall time from first send to last completion, prefill included):

| context \ requests | 1 | 2 | 4 | 8 | 16 | prefill tok/s (1 req) |
|---|---:|---:|---:|---:|---:|---:|
| 0 (0) | — | — | — | — | — |  |
| 8K | — | — | — | — | — |  |
| 32K | — | — | — | — | — |  |
| 64K (65,408) | — | — | — | — | — |  |

— = not run: N is more than the --parallel that fit at this context (N slots of context + gen did not come up on the two cards: llama-server failed to load, came up short, or spilled into system RAM), or the server returned an error. * = some request finished short (fewer than gen tokens), or the requests never all decoded together (later prefills outlasted earlier decodes); a starred decode cell is the rate from the last first token to the end, still prefill-free but with fewer than N decoding. n/r = not run for time (N prefills at the measured 1-request prefill rate would pass the per-cell cap); says nothing about fit. llama.cpp has no preemption (each slot holds its whole context from load). This block: 0 x 1: — no llama-server came up with even 1 slot of 256 tokens: warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_request_error"}}; warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_; 0 x 2: — no llama-server came up with even 1 slot of 256 tokens: warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_request_error"}}; warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_; 0 x 4: — no llama-server came up with even 1 slot of 256 tokens: warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_request_error"}}; warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_; 0 x 8: — no llama-server came up with even 1 slot of 256 tokens: warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_request_error"}}; warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_; 0 x 16: — no llama-server came up with even 1 slot of 256 tokens: warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_request_error"}}; warm-up failed: HTTP 400: {"error":{"code":400,"message":"[json.exception.type_error.302] type must be string, but is array","type":"invalid_; 8K x 1: — no llama-server came up with even 1 slot of 8,448 tokens: -c 135,168 >= 256, which did not fit; -c 67,584 >= 256, which did not fit; -c 33,792 >= 256, which did not fit; -c 16,896 >= 256, which did not fit; -c 8,448 >= 256, which did not fit; 8K x 2: — no llama-server came up with even 1 slot of 8,448 tokens: -c 135,168 >= 256, which did not fit; -c 67,584 >= 256, which did not fit; -c 33,792 >= 256, which did not fit; -c 16,896 >= 256, which did not fit; -c 8,448 >= 256, which did not fit; 8K x 4: — no llama-server came up with even 1 slot of 8,448 tokens: -c 135,168 >= 256, which did not fit; -c 67,584 >= 256, which did not fit; -c 33,792 >= 256, which did not fit; -c 16,896 >= 256, which did not fit; -c 8,448 >= 256, which did not fit; 8K x 8: — no llama-server came up with even 1 slot of 8,448 tokens: -c 135,168 >= 256, which did not fit; -c 67,584 >= 256, which did not fit; -c 33,792 >= 256, which did not fit; -c 16,896 >= 256, which did not fit; -c 8,448 >= 256, which did not fit; 8K x 16: — no llama-server came up with even 1 slot of 8,448 tokens: -c 135,168 >= 256, which did not fit; -c 67,584 >= 256, which did not fit; -c 33,792 >= 256, which did not fit; -c 16,896 >= 256, which did not fit; -c 8,448 >= 256, which did not fit; 32K x 1: — no llama-server came up with even 1 slot of 33,024 tokens: -c 528,384 >= 256, which did not fit; -c 264,192 >= 256, which did not fit; -c 132,096 >= 256, which did not fit; -c 66,048 >= 256, which did not fit; -c 33,024 >= 256, which did not fit; 32K x 2: — no llama-server came up with even 1 slot of 33,024 tokens: -c 528,384 >= 256, which did not fit; -c 264,192 >= 256, which did not fit; -c 132,096 >= 256, which did not fit; -c 66,048 >= 256, which did not fit; -c 33,024 >= 256, which did not fit; 32K x 4: — no llama-server came up with even 1 slot of 33,024 tokens: -c 528,384 >= 256, which did not fit; -c 264,192 >= 256, which did not fit; -c 132,096 >= 256, which did not fit; -c 66,048 >= 256, which did not fit; -c 33,024 >= 256, which did not fit; 32K x 8: — no llama-server came up with even 1 slot of 33,024 tokens: -c 528,384 >= 256, which did not fit; -c 264,192 >= 256, which did not fit; -c 132,096 >= 256, which did not fit; -c 66,048 >= 256, which did not fit; -c 33,024 >= 256, which did not fit; 32K x 16: — no llama-server came up with even 1 slot of 33,024 tokens: -c 528,384 >= 256, which did not fit; -c 264,192 >= 256, which did not fit; -c 132,096 >= 256, which did not fit; -c 66,048 >= 256, which did not fit; -c 33,024 >= 256, which did not fit; 64K x 1: — no llama-server came up with even 1 slot of 65,792 tokens: -c 1,052,672 >= 256, which did not fit; -c 526,336 >= 256, which did not fit; -c 263,168 >= 256, which did not fit; -c 131,584 >= 256, which did not fit; -c 65,792 >= 256, which did not fit; 64K x 2: — no llama-server came up with even 1 slot of 65,792 tokens: -c 1,052,672 >= 256, which did not fit; -c 526,336 >= 256, which did not fit; -c 263,168 >= 256, which did not fit; -c 131,584 >= 256, which did not fit; -c 65,792 >= 256, which did not fit; 64K x 4: — no llama-server came up with even 1 slot of 65,792 tokens: -c 1,052,672 >= 256, which did not fit; -c 526,336 >= 256, which did not fit; -c 263,168 >= 256, which did not fit; -c 131,584 >= 256, which did not fit; -c 65,792 >= 256, which did not fit; 64K x 8: — no llama-server came up with even 1 slot of 65,792 tokens: -c 1,052,672 >= 256, which did not fit; -c 526,336 >= 256, which did not fit; -c 263,168 >= 256, which did not fit; -c 131,584 >= 256, which did not fit; -c 65,792 >= 256, which did not fit; 64K x 16: — no llama-server came up with even 1 slot of 65,792 tokens: -c 1,052,672 >= 256, which did not fit; -c 526,336 >= 256, which did not fit; -c 263,168 >= 256, which did not fit; -c 131,584 >= 256, which did not fit; -c 65,792 >= 256, which did not fit.

