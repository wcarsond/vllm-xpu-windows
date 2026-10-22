"""readme-results.input.py - the call: a dated "Results at a glance" section at the top of README.md (Carson, 2026-10-21: the
Reddit post links the repo, "not a html", so the repo's front page must carry the numbers). Inserted once, after the intro
table (the first blank line after the first pipe table); rerunning does nothing if the heading is already there. The five
tables are the ones in docs/index.html and REDDIT-POST-2026-10-21-numbers.md.

    .venv312\\Scripts\\python.exe readme-results.input.py"""
import os, re
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'); s = open(p, encoding='utf-8').read()
H = '## Results at a glance (2026-10-21)'
if H in s: print('already there'); raise SystemExit
SECTION = H + """

Two Arc Pro B60 24 GB on PCIe Gen5 x8, Core Ultra 5 250K, 32 GB RAM, Windows 11, driver 32.0.101.9033. "Linux" is the same
board booted into Ubuntu with Intel's llm-scaler vLLM 0.26.0-b2 and oneCCL. Decode = tok/s over 128-256 generated tokens,
greedy; prefill = tok/s over the prompt; context = tokens already in the KV cache. Measured 2026-10-10 to 10-13; the dated
sections below hold the details and the raw logs are under `server/`. Page version: https://wcarsond.github.io/vllm-xpu-windows/

**Two-card all_reduce** (Windows port vs Intel's oneCCL on Linux, same board)

| size | Windows, first port | Windows, now | Linux, oneCCL |
|---|---:|---:|---:|
| 14.7 MB (prefill chunk) | 4,204 us | **1,289 us** | ~2,720 us |
| 3.7 MB | 887 us | **319 us** | 681 us |
| 7B decode size | 28 us | **7 us** | 13 us |

**Qwen2.5-Coder-7B bf16, two cards** (tensor parallel, greedy tokens identical)

| context | decode, Windows | decode, Linux | prefill, Windows | prefill, Linux |
|---|---:|---:|---:|---:|
| 0 | **56.6** | 51.5 | | |
| 4K | **54.9** | 50.2 | **7,589** | 6,019 |
| 16K | **52.3** | 50.2 | **6,561** | 5,472 |
| 32K | **49.4** | 44.7 | **5,456** | 4,729 |

**Bigger models, two cards, decode tok/s** (two worker processes vs both ranks in one process)

| model | two processes | one process |
|---|---:|---:|
| Qwen3.5-27B GPTQ-Int4 | 24.6 | **29.0** |
| Granite 4.2 30B GPTQ-Int4 | 36.5 | **44.6** |
| Qwen2.5-72B-Instruct AWQ | 17.5 | **20.1** |
| Qwen3.5-4B | 64.8 | **87.1** |
| gpt-oss-20b (MoE) | **97.7** | 70.3 |

**vLLM vs llama.cpp, same two cards** (decode short / at 32K, then seconds to first token on a 32K prompt)

| model | llama.cpp stock Vulkan | llama.cpp, exchange + FA fix | vLLM (this port) |
|---|---:|---:|---:|
| Qwen2.5-Coder-7B (Q8_0 / bf16) | 43.4 / 33.1, 60.5 s | **66.0 / 48.6**, 30.2 s | 46.4 / 40.1, **8.8 s** |
| Qwen2.5-14B (Q4_K_M / bf16) | 37.9 / 22.4, 154.0 s | **50.2 / 31.5**, 67.2 s | 25.2 / 20.9, **17.7 s** |
| Granite 4.2 30B (Q4_K_M / GPTQ-Int4) | 19.8 / 13.6, 176.1 s | 29.3 / 20.3, 80.2 s | **31.9 / 25.3, 27.8 s** |

**llama.cpp Vulkan flash attention on Xe, one card** (Qwen3-Coder-30B-A3B Q3_K_XL, pp512 tok/s at depth; generation 98 / 70 / 57 / 40 both ways)

| build | 0 | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| stock llama.cpp b11191 | 1,707 | 319 | 173 | 89 |
| 16x16x16 tile fixed to 8x16x16 on Xe ([b60-flash-attention](https://github.com/wcarsond/b60-flash-attention)) | **1,847** | **586** | **341** | **187** |

"""
# insert after the intro table: the first blank line that follows a line starting with '|'
lines = s.split('\n'); at = None
for i in range(1, len(lines)):
    if lines[i - 1].startswith('|') and lines[i].strip() == '': at = i + 1; break
assert at, 'intro table not found'
lines[at:at] = SECTION.split('\n')
open(p, 'w', encoding='utf-8').write('\n'.join(lines)); print('inserted at line', at)
