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

Decode tokens/second, same prompt and 256-token budget, greedy, median of three runs. Decode at
depth on two cards holds 1.5-1.7x one card out to 32K context.

Everything here was measured on one machine: a Core Ultra 5 250K, 32 GB of RAM, two Arc Pro B60s
on PCIe Gen5 x8 each, Windows 11, Intel driver 32.0.101.9033.

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