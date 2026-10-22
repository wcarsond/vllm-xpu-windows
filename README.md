# vLLM on native Windows, on two Intel Arc Pro B60s

No WSL, no Linux, no Docker. vLLM serving on two Intel Arc Pro B60 cards on Windows 11, with the
tensor split actually faster than one card.

| model | 1 card | 2 cards |
|---|---|---|
| Qwen2.5-Coder-7B bf16 | 29.9 tok/s | **47.2 tok/s** |
| Qwen2.5-32B-Instruct-AWQ int4 | 23.2 (4K context max) | **33.2** (32K context) |
| Granite 4.2 30B int4 | not measured | **33.1** |
| Qwen3.5-27B int4 (hybrid Gated DeltaNet, vision) | will not fit | **23.7** |
| Qwen2.5-72B-Instruct-AWQ int4 | will not fit | **17.3** (2048-token window; it only just fits) |
| Qwen2.5-14B bf16 | 1.8 (spilled to host RAM) | **25.1** |

## Results at a glance (2026-10-21)

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


Decode tokens/second, same prompt and 256-token budget, greedy, median of three runs. Decode at
depth on two cards holds 1.5-1.7x one card out to 32K context.

Everything here was measured on one machine: a Core Ultra 5 250K, 32 GB of RAM, two Arc Pro B60s
on PCIe Gen5 x8 each, Windows 11, Intel driver 32.0.101.9033.

**Under load: concurrent requests by context** (vLLM on two B60s, aggregate decode tok/s for all streams together, prefill excluded; 128 tokens per request)

*Qwen3.6-27B GPTQ-Int4*, pulled 2026-10-21 (raydelossantos, same recipe as Qwen's own 3.5 GPTQ), 30.3 GB

| context | 1 | 2 | 4 | 8 | 16 | prefill tok/s, 1 req |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 29.1 | 55.9 | 104.1 | 79.0 | 160.1 | 238 |
| 8K | 28.3 | 53.6 | 96.8 | 81.0 | 73.8* | 1,906 |
| 32K | 26.8 | 48.1 | 82.1 | — | — | 1,712 |
| 64K | 24.6 | 43.3 | — | — | — | 1,502 |

*Qwen3.5-27B GPTQ-Int4*, the router's preset

| context | 1 | 2 | 4 | 8 | 16 | prefill tok/s, 1 req |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 29.1 | 55.9 | 104.1 | 76.6 | 145.5 | 232 |
| 8K | 28.3 | 53.4 | 97.3 | 78.8 | 72.6* | 1,924 |
| 32K | 26.7 | 48.1 | 81.5 | — | — | 1,717 |
| 64K | 24.6 | 43.3 | — | — | — | 1,506 |

*Qwen2.5-Coder-7B bf16*

| context | 1 | 2 | 4 | 8 | 16 | prefill tok/s, 1 req |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 56.6 | 111.8 | 215.3 | 412.1 | 466.4 | 499 |
| 8K | 53.9 | 103.2 | 188.4 | 330.0 | 458.6 | 7,210 |
| 32K | 49.2 | 87.4 | 143.6 | 205.1 | — | 5,417 |

Aggregate decode tok/s: all requests' generated tokens while all were decoding together, divided by that time, prefill excluded; 128 tokens generated per request, temperature 0, prefix cache hits measured 0. Prefill is one request. A dash means N x (context + 128) is more than the KV cache vLLM reported, so the cell was not run. The star at 8K x 16 on the 27Bs: two preemptions, the requests never all decoded together. The dip at 8 requests on both 27Bs is real: on two Windows cards vLLM captures decode graphs for batch 1, 2 and 4 only, so 8 and 16 run without graphs; the 7B shows no dip. vLLM 0.1.dev1+g276fbcff2.d20261010.xpu, 2026-10-21/22; raw numbers in `server/`concurrency-sweep.output.json.

*Qwen2.5-Coder-7B Q8_0 on llama.cpp*, --parallel 16, -c 135,168 (8,448 tokens per slot); the single-stream race in the table above ran one slot

| context | 1 | 2 | 4 | 8 | 16 | prefill tok/s, 1 req |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 42.9 | 81.8 | 159.3 | 263.2 | 174.9 | 314 |
| 8K | 39.5 | 73.1 | 129.3 | 67.2 | 102.9 | 2,526 |
| 32K | 32.6 | 56.1 | 82.1 | 24.9 | — | 1,175 |

*Qwen3.6-35B-A3B UD-Q4_K_XL on llama.cpp*, the fastest big model on these cards; 64K did not come up on two cards

| context | 1 | 2 | 4 | 8 | 16 | prefill tok/s, 1 req |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 72.1 | 113.4 | 159.5 | 198.6 | 139.1 | 111 |
| 8K | 68.7 | 101.0 | 138.3 | 122.3 | 58.8 | 2,254 |
| 32K | 59.6 | 86.8 | 112.7 | 96.7 | 114.3* | 1,312 |
| 64K | — | — | — | — | — |  |

llama.cpp, same two cards, same measure: aggregate decode tok/s while all requests were decoding together, prefill excluded, 128 tokens per request. llama-server prefills prompts one slot at a time, so its raw throughput with prefill included is far below vLLM's at 8K and 32K (the 7B at 8K x 16: 25 tok/s raw against vLLM's 92). A dash means that many slots of that context did not come up on the two cards. The dips at 8 and 16 streams are real and repeatable on this build. llama-server 0.5.0-dev (build 55, vk-allreduce-fa), 2026-10-22; raw numbers in `server/`concurrency-sweep-llamacpp.output.json.

## 2026-10-13 (evening): both cards in one process - ahead of Linux on decode too

On Windows a kernel cannot touch another process's memory and IPC handles are refused for both card and host memory,
so between two worker processes every decode-size all_reduce was relayed by a CPU helper thread (28 us). With both
tensor-parallel ranks inside one process, both cards' kernels read and write the same pinned host buffer directly:
**7 us** per 7B decode all_reduce (`probe/one-process-allreduce.cpp`; spreading the tagged words over 16-32 work-groups
is what makes it fast). The new executor, `vllm.v1.executor.oneproc_executor.OneProcExecutor`, runs the two ranks as
threads. Two threads running oneDNN matmuls on two cards at the same time hang this driver (`probe/threads-hang`), so
the threads take turns under a baton that changes hands at every collective and after every graph replay; per-rank
module state is swapped with it. Run with full decode graphs (`compilation_config={"mode": 0, "cudagraph_mode":
"FULL_DECODE_ONLY"}`); eager is slower in one process because the two ranks' host work no longer overlaps.

Decode tok/s, two B60s, full decode graphs, greedy tokens identical between the two executors:

| model | two processes | **one process** | Linux, Intel's stack |
|---|---|---|---|
| Qwen2.5-Coder-7B bf16, short / 4K / 16K / 32K context | 48.3 / 46.7 / 44.5 / 42.4 | **56.6 / 54.9 / 52.3 / 49.4** | 51.5 / 50.2 / 50.2 / 44.7 |
| Qwen3.5-27B int4 | 24.6 | **29.0** | |
| Granite 4.2 30B int4 | 36.5 | **44.6** | |
| Qwen2.5-72B-Instruct-AWQ | 17.5 | **20.1** | |

One exception so far: **gpt-oss-20b** (MoE, ~3.6B active) decodes 97.7 tok/s on two processes against 70.3 in one
(tokens identical; 88.5 on one card). Its card work is only ~6.5 ms per token per card, so the two ranks' serialized
host work and the 400 KB logits gather show; the 4B dense Qwen3.5 still wins in one process (87.1 against 64.8). Pick
the executor per model (`server/oneproc-profile.input.py` has the per-kernel breakdown).

Prefill is unchanged (7B: 7,589 / 6,561 / 5,456 tok/s at 4K / 16K / 32K, 27-36% ahead of Linux). The 7B decode step
(17.8 ms) is within ~4% of what the cards' memory bandwidth allows. Scripts: `server/oneproc-smoke`, `oneproc-race`,
`oneproc-models`, `oneproc-ab` (async scheduling is worth +8% of it; aligning the two cards' replays nothing).

## 2026-10-13: the two-card exchange without a CPU hop

The large-message path of the two-card exchange (prefill chunks) used to stage every byte through a CPU thread. The
shared Windows section can be imported into Level Zero (`zexDriverImportExternalPointer`); kernels still cannot touch
imported memory, but the copy engine can. Each card now copies its half straight into its slot of the section and the
peer card copies it straight out. Probes: `probe/copy-bw.input.py` (raw card/host copies: 28 GB/s, same as Linux) and
`probe/section-copy.cpp` (two processes, one imported section, bit-exact).

| two B60s | before | now | Linux, Intel's stock oneCCL, same board |
|---|---|---|---|
| all_reduce, 14.7 MB | 4,204 us | **1,289 us** | ~2,720 us |
| all_reduce, 3.7 MB | 887 us | **319 us** | 681 us |
| Qwen2.5-Coder-7B bf16 prefill, 4K / 16K / 32K tokens | 4,504 / 4,087 / 3,629 | **7,617 / 6,518 / 5,420** | 6,019 / 5,472 / 4,729 |

Two-card prefill on Windows is now 15-27% ahead of Linux running Intel's own stack (llm-scaler vLLM 0.26.0-b2,
oneCCL with its all_reduce size threshold raised). Decode is unchanged at 47.2 tok/s short against Linux's 51.5: the
small, decode-size exchange (28 us against oneCCL's 13) is the next target. The tables in BACKEND-RACE.md predate this.

## Against llama.cpp on the same cards

Same models through stock and patched llama.cpp Vulkan, SYCL, a two-card Vulkan exchange, and this port, one card
and two, from short prompts to 32K tokens: [BACKEND-RACE.md](BACKEND-RACE.md). In short, on two cards:

| model | best llama.cpp: decode short / 32K, first token for a 32K prompt | vLLM: same |
|---|---|---|
| Granite 4.2 30B, 4-bit both sides | 29.3 / 20.3 tok/s, 80.2 s | **31.9 / 25.3 tok/s, 27.8 s** |
| Qwen2.5-Coder-7B (llama.cpp Q8_0, vLLM bf16) | **66.0 / 48.6 tok/s**, 30.2 s | 46.4 / 40.1 tok/s, **8.8 s** |
| Qwen2.5-14B (llama.cpp Q4_K_M, vLLM bf16) | **50.2 / 31.5 tok/s**, 67.2 s | 25.2 / 20.9 tok/s, **17.7 s** |

vLLM gets to the first token 3-4x sooner on long prompts. Decode is close where the weights match (Granite); on the
other two vLLM reads two to four times the bytes per token and the comparison favours llama.cpp.

## Why this exists

Intel's vLLM support is Linux-only, and the two-card path there depends on oneCCL, which has no
Windows build. Plenty of people run Windows servers. The hardware does not change between operating
systems, so this is a port, not a research project.

## What was actually wrong, and what fixed it

Five real bugs, each found by measurement rather than guessing. The full investigation, including
every wrong turn, is in [PORT-NOTES.md](PORT-NOTES.md).

**1. `long` is 32 bits in SYCL device code on Windows.** Every cutlass-sycl flash-attention kernel
killed the GPU context on its first launch. The rearchitected Xe copy atoms declare IGC's
address-payload builtins taking `long`, which is 64-bit on Linux and 32-bit under the MSVC data
model, so every tensor base address reached the card with its top half cut off. Changing four
headers to `int64_t` took it from "every kernel kills the card" to 58 TFlop/s.
Patch: [`patches/cutlass-sycl-windows-64bit-addresses.patch`](patches/).

**2. No card-to-card peer access on Windows, so host memory is the channel.** Intel's own open
source shows why: `query_peer_access_wddm.cpp` returns `false` unconditionally, while the Linux
file genuinely probes. The replacement is a GPU-timeline all-reduce that never synchronises the
host, modelled on NCCL's LL protocol: every 8-byte unit carries 4 bytes of data plus a 4-byte
sequence tag, so a stale cache line fails its own tag and is simply read again. 11 microseconds per
all-reduce, against 258 microseconds for the host-synchronous version it replaced.

**3. Windows renames are not atomic, so Triton's cache races between ranks.** Two tensor-parallel
ranks compiling the same kernel at the same moment kill one another with `PermissionError`. Each
rank now gets its own cache directory. This one hits anyone running tensor parallel on Windows,
not just Intel cards.

**4. Unkillable "zombie" workers after every clean run.** A worker exiting normally runs DLL unload
handlers; the attention kernel library destroys a static SYCL device there, and the OpenCL runtime
spins forever releasing its context. The process is already marked exited, so nothing can kill it
and it burns a core until reboot. Diagnosed by walking another process's stacks, fixed by ending
Windows XPU workers with `TerminateProcess` the way the executor already ends workers it kills.

**5. vLLM was capturing a GPU graph it should not have.** Two-card `torch.compile` runs hung or
corrupted the card with "requires reset". Under graph capture nothing executes, so a host-memory
collective records instead of running. Disabled for Windows with more than one card.

**6. Two tensor-parallel workers loading one checkpoint kill each other.** An access violation
inside the driver, exit code 0xC0000005, no Python traceback, at a different shard every run.
One worker is fine, a pipeline split at the same memory per card is fine, dummy weights are
fine. Memory pressure was ruled out on the cards (the driver's own counters show nothing
spilled) and on the host (14.5 GB free at the moment of the fault). What tensor split alone
does is feed the same physical page-cache pages to two adapters at once. The fix is one host
memcpy per tensor: each worker copies a weight into private memory before any card sees it, so
the CPUs share the file pages and the cards never do. Four parallel loads of the 72B in a row
after that, none before it, and a cold start 40 seconds faster than the serialized workaround.
The notes record the seven wrong explanations that preceded it, each killed by a measurement,
and the method that finally found it: cut the failing case down instead of building a probe up.

**7. Two processes each masked to its own card fault the driver.** `ZE_AFFINITY_MASK=0` in one process and `=1` in
another, then any concurrent host-to-device copy, and one process dies with 0xC0000005: 15 KB or 1 GB, heap or
pinned, with or without the private-copy loader fix. Two unmasked processes that each pick a card are clean at
20 GiB apiece. Reproducer: two 0.5B vLLM engines, seconds. Workaround: choose the card above the driver with
`ONEAPI_DEVICE_SELECTOR=level_zero:N`.

## Some things that are true about this driver

Measured with the probes in [`probe/`](probe/), because none of it is documented:

- A card's writes into host memory **do not** invalidate the CPU's cached copies of those lines.
  The CPU must flush before reading, or it reads stale data roughly 1 time in 300.
- A kernel can only see a CPU write to host memory with a cache-bypassing load plus an acquire
  fence on every poll. Plain loads: 0 of 20. System-scope atomics: 4 of 20. The recipe: 20 of 20.
- Imported external host pointers are copy-only. Kernels page-fault on them.
- 32-byte vector *stores* with an uncached hint are about 80x slower than scalar stores. Loads are
  fine.
- The driver reports `has_subgroup_matrix_multiply_accumulate = False` and
  `has_subgroup_2d_block_io = False` on hardware that does both. It is a reporting gap, not missing
  silicon.
- Per-card PCIe is 25-28 GB/s each direction, so the two cards have independent paths to host memory.

## Layout

Every script is named `<thing>.input.<ext>` and writes its results to `<thing>.output.*` beside it.
Run things by name; nothing needs arguments typed into a shell.

- [`PORT-NOTES.md`](PORT-NOTES.md) — the whole investigation, in order, with the numbers and the
  hypotheses that turned out to be wrong. This is the real content of the repository.
- [`kernels/`](kernels/) — building `vllm-xpu-kernels` on Windows, including the oneDNN quantized
  GEMMs and the MoE kernels.
- [`server/`](server/) — the runs: one card, two cards, tensor split, pipeline split, at depth.
- [`probe/`](probe/) — small standalone programs that answer one driver question each.
- [`patches/`](patches/) — the cutlass-sycl fix as a patch.
- [`toolchain/`](toolchain/) — installing the compiler and SDK.

## The source trees

Not included here; they are forks with their own branches.

- vLLM runtime port: branch `windows-xpu`
- `vllm-xpu-kernels` Windows build: branch `windows`
- cutlass-sycl 64-bit address fix: branch `windows-64bit-addresses`

## Honesty notes

- AI assistance was used heavily throughout this work, and the notes record it as it happened.
- The numbers are from one machine with two cards. Nothing here has been tested on four cards, on
  other Arc models, or against a Linux build of the same vLLM commit. The Linux comparison is
  written up as a protocol in the notes and has not been run yet.
- The 72B AWQ model took an afternoon and is bug 6 above. It runs at 17.3 tok/s on the pair in a
  2048-token window; a 39 GB model on 48 GB of cards is genuinely tight and no fix changes that.
  No standalone probe reaches its trigger at vLLM's memory level, so the reproducer is the
  lock-off tensor-split load itself (`VLLM_XPU_PRIVATE_COPY=0`); two neighbouring triggers that do
  reproduce without vLLM, masked processes and overflow past the card, are in `probe/`.