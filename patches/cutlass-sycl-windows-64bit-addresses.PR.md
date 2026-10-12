# Draft PR for intel/sycl-tla (cutlass-sycl) — for Carson to review, edit and push

Patch: `cutlass-sycl-windows-64bit-addresses.patch` (against 87f68506, "FIX BMG GEMM Performance regression (#846)").
Four files, 64 insertions, 61 deletions, all type changes.

---

**Title:** Use 64-bit integer types for block-2D addresses (Windows: `long` is 32 bits)

**Summary**

The Xe block-2D copy paths pass GPU addresses to IGC builtins as `long`:

- `include/cute/atom/copy_traits_xe_2d.hpp`: `__builtin_IB_subgroup_createBlock2DAddressPayload(long base, ...)` and
  `__builtin_IB_subgroup_setBlock2DAddressPayloadBase(int*, long base)`, called with the `uint64_t base_ptr` member.
- `include/cute/arch/copy_xe_legacy_builtin.hpp` and `copy_xe_legacy_spirv.hpp`: `long baseoffset` parameters and
  `reinterpret_cast<long>(pointer)` at the call sites (a few of these were already `intptr_t`).
- `include/cute/util/sycl_vec.hpp`: `using ulong = unsigned long;`, used as the 64-bit element type of the
  `*_transpose_u64_*` block reads.

On Linux `long` is 64 bits and this is correct. On Windows the MSVC data model applies and the SYCL device pass
follows it: `long` is 32 bits inside kernels (verified with a `static_assert(sizeof(long) == 8)` in device code, which
fails with `4 == 8` under oneAPI 2026.1 on Windows). Every address passed through these declarations therefore loses
its upper 32 bits, and the first block-2D load page-faults: the device context is lost on the first launch of any kernel
built on the rearchitected copy atoms (`MainloopXeL1Staged`, `XE_LOAD_2D*`). The legacy SPIR-V path is unaffected only
because it passes `const void*`.

**Change**

- `int64_t` for the two address-payload base parameters.
- `intptr_t` for the remaining `long baseoffset` parameters and casts in the legacy builtin/SPIR-V headers (matching
  the ones already converted).
- `unsigned long long` for `cute::intel::ulong` (OpenCL's `ulong` is 64 bits).

No behaviour change on Linux (`long`, `intptr_t` and `int64_t` are all 64 bits there).

**Verification** (Intel Arc Pro B60, Windows 11, driver 32.0.101.9033, oneAPI 2026.1, AOT `-device bmg`)

`examples/00_bmg_gemm/00_bmg_gemm.cpp`, 1024x1024x1024, bf16 → f32:

| build | before | after |
|---|---|---|
| default (128 GRF) | exit 0xC0000409, `UR_RESULT_ERROR_DEVICE_LOST` on the first launch | Passed, 7.9 TFlop/s |
| `-cl-intel-256-GRF-per-thread` | same | Passed, 58.0 TFlop/s |
| `examples/00_bmg_gemm/legacy` (old atoms) | Passed, 56.9 TFlop/s | Passed, 56.9 TFlop/s |

Also fixes the vllm-xpu-kernels flash-attention and mhc kernels on the same machine, which showed the same first-launch
device loss.

---

Notes for the reviewer thread (not for the PR body): the one-line reproducer of the type difference is
`static_assert(sizeof(long) == 8)` inside a kernel compiled with `icx -fsycl` on Windows. If the project prefers, a
`cute::intel::address_t` alias (`int64_t`) in `sycl_vec.hpp` would make the intent explicit at every call site.
