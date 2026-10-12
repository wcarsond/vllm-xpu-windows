# Backend race: every backend on two Intel Arc Pro B60s

Same three models on every backend, one card and two, prompts and context from short to 32K tokens. One machine:
Core Ultra 5 250K, 32 GB RAM, two Arc Pro B60 24 GB on PCIe Gen5 x8, Windows 11, Intel driver 32.0.101.9033.
llama.cpp builds are all at b11200 (`2145525`) so stock and patched differ by the patch alone.

- **Vulkan stock / SYCL stock**: the official b11200 release binaries.
- **Vulkan patched**: b11200 plus the coopmat1 flash-attention fix for Intel's 8x16x16 tile (https://github.com/wcarsond/b60-flash-attention).
- **Vulkan exchange**: the two-card all-reduce for the Vulkan backend (tensor split), with and without that fix (same repository).
- **vLLM**: this port, `server/race-7b-depth.input.py`.

Flash attention on, every layer on the card(s), greedy, one request at a time. Decode is tokens/s with that much context
already cached (higher is better). First token is seconds to process a whole prompt of that length (lower is better).

| model | backend | weights | cards | decode short | decode 16K | decode 32K | first token, 16K prompt (s) | first token, 32K prompt (s) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B | Vulkan stock | Q8_0 | 1 | 44.4 | 38.5 | 34.0 | 26.4 | 90.8 |
| Qwen2.5-Coder-7B | Vulkan patched | Q8_0 | 1 | 44.5 | 38.4 | 34.0 | 15.1 | 47.3 |
| Qwen2.5-Coder-7B | SYCL stock | Q8_0 | 1 | 38.6 | 20.5 | 13.5 | 22.5 | 47.4 |
| Qwen2.5-Coder-7B | vLLM | bf16 | 1 | 30.0 | 27.9 | 26.3 | 3.5 | 8.5 |
| Qwen2.5-Coder-7B | Vulkan stock | Q8_0 | 2 | 43.4 | 37.4 | 33.1 | 17.6 | 60.5 |
| Qwen2.5-Coder-7B | Vulkan patched | Q8_0 | 2 | 43.4 | 37.4 | 33.1 | 10.2 | 31.7 |
| Qwen2.5-Coder-7B | SYCL stock | Q8_0 | 2 | 37.4 | 20.2 | 13.4 | 22.6 | 47.6 |
| Qwen2.5-Coder-7B | Vulkan exchange | Q8_0 | 2 | 66.9 | 55.7 | 48.3 | 14.6 | 48.2 |
| Qwen2.5-Coder-7B | Vulkan exchange + fix | Q8_0 | 2 | 66.0 | 56.5 | 48.6 | 9.9 | 30.2 |
| Qwen2.5-Coder-7B | vLLM | bf16 | 2 | 46.4 | 43.8 | 40.1 | 4.0 | 8.8 |
| Qwen2.5-14B | Vulkan stock | Q4_K_M | 1 | 38.8 | 28.6 | 22.9 | 67.1 | 236.0 |
| Qwen2.5-14B | Vulkan patched | Q4_K_M | 1 | 38.8 | 28.6 | 22.9 | 33.7 | 108.4 |
| Qwen2.5-14B | SYCL stock | Q4_K_M | 1 | 34.8 | 10.1 | 6.0 | 46.2 | 98.1 |
| Qwen2.5-14B | vLLM | bf16 | 1 | does not fit one card | | | | |
| Qwen2.5-14B | Vulkan stock | Q4_K_M | 2 | 37.9 | 27.9 | 22.4 | 43.9 | 154.0 |
| Qwen2.5-14B | Vulkan patched | Q4_K_M | 2 | 37.9 | 28.0 | 22.3 | 22.1 | 71.0 |
| Qwen2.5-14B | SYCL stock | Q4_K_M | 2 | 33.8 | 10.0 | 5.9 | 46.3 | 98.3 |
| Qwen2.5-14B | Vulkan exchange | Q4_K_M | 2 | 50.2 | 38.3 | 31.3 | 38.5 | 130.9 |
| Qwen2.5-14B | Vulkan exchange + fix | Q4_K_M | 2 | 50.2 | 38.7 | 31.5 | 22.0 | 67.2 |
| Qwen2.5-14B | vLLM | bf16 | 2 | 25.2 | 22.9 | 20.9 | 8.0 | 17.7 |
| Granite 4.2 30B | Vulkan stock | Q4_K_M | 1 | 20.1 | 16.2 | 7.3 * | 82.0 | 579.7 * |
| Granite 4.2 30B | Vulkan patched | Q4_K_M | 1 | 20.1 | 16.2 | 7.3 * | 49.0 | 309.8 * |
| Granite 4.2 30B | SYCL stock | Q4_K_M | 1 | 17.2 | 14.0 | timed out | 94.1 | timed out |
| Granite 4.2 30B | vLLM | GPTQ int4 | 1 | does not fit one card | | | | |
| Granite 4.2 30B | Vulkan stock | Q4_K_M | 2 | 19.8 | 16.0 | 13.6 | 53.2 | 176.1 |
| Granite 4.2 30B | Vulkan patched | Q4_K_M | 2 | 19.8 | 16.0 | 13.6 | 31.8 | 93.7 |
| Granite 4.2 30B | SYCL stock | Q4_K_M | 2 | 17.0 | 13.9 | 11.9 | 94.2 | 193.2 |
| Granite 4.2 30B | Vulkan exchange | Q4_K_M | 2 | 29.5 | 24.2 | 20.2 | 47.2 | 152.2 |
| Granite 4.2 30B | Vulkan exchange + fix | Q4_K_M | 2 | 29.3 | 23.9 | 20.3 | 28.4 | 80.2 |
| Granite 4.2 30B | vLLM | GPTQ int4 | 2 | 31.9 | 28.7 | 25.3 | 12.8 | 27.8 |

\* At 32K the card is full and about 1 GB of the cache spills to host memory (driver counters: 23.5 of 24 GB
dedicated, ~1 GB shared, GPU mostly idle). A one-card capacity limit, gone on two cards.

**Reading it honestly.**
- Weights only match across engines on Granite (4-bit both sides). On the 7B and 14B vLLM reads bf16, two to four
  times the bytes per token of the GGUF, so its decode there is handicapped; its time-to-first-token lead is not.
- **Long prompts:** vLLM is 3-10x faster to the first token and loses only ~20% of its prefill speed from 4K to 32K.
- **Decode:** the two-card Vulkan exchange is the fastest llama.cpp setup on every model; on Granite (fair match) it is
  within 8% of vLLM at short context and 20% behind at 32K.
- **The flash-attention fix** roughly halves time to first token at 16-32K on every model and card count, and leaves
  decode unchanged. On top of the exchange it gives one binary with both.
- **SYCL** keeps a flat prefill curve (its attention is not the bottleneck) but its matrix math is ~3.5x slower than
  Vulkan's, and its decode falls steeply with context.
- A second card never slowed layer-split decode much and never sped it up; the speedups for two cards come from
  tensor split (the exchange, or vLLM).

Raw results and both full tables: `testing/backend-race/` in https://github.com/wcarsond/b60-flash-attention.
