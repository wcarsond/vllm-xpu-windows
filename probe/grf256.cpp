// grf256.cpp - the smallest SYCL kernel there is, built by grf256.input.cmd three ways to ask one question of the driver:
// does a kernel compiled for 256 GRF per thread run on the B60 under Windows at all?
// Why (2026-10-10): every kernel in the Xe2 libraries (attention, mhc) dies on its first launch; every kernel in the base
// library runs. The Xe2 libraries are the ones linked with -cl-intel-256-GRF-per-thread. This removes cutlass, torch and
// vLLM from the picture: one vector add, same source, with and without the option.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

// The Xe2 2D block load every cutlass-sycl copy atom issues (cute/arch/copy_xe_legacy_spirv.hpp). Device-only: the
// coordinate is a clang ext-vector int2 there, and the translator maps this exact signature to OpSubgroup2DBlockLoadINTEL.
#ifdef __SYCL_DEVICE_ONLY__
using coord2 = int __attribute__((ext_vector_type(2)));
using ushort16v = unsigned short __attribute__((ext_vector_type(16)));
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockLoadINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount, const void* src_base_pointer, int memory_width,
    int memory_height, int memory_pitch, coord2 coordinate, void* dst_pointer);
// The rest of cutlass-sycl's instruction set (cute/arch/copy_xe_legacy_spirv.hpp, mma_xe_legacy_spirv.hpp,
// cute/util/xe_split_barrier.hpp), declared exactly as cute declares them.
using short8v = short __attribute__((ext_vector_type(8)));
using int8v = int __attribute__((ext_vector_type(8)));
using float8v = float __attribute__((ext_vector_type(8)));
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockLoadTransformINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount, const void* src_base_pointer, int memory_width,
    int memory_height, int memory_pitch, coord2 coordinate, void* dst_pointer);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockStoreINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount, void* src_pointer, const void* dst_base_pointer,
    int memory_width, int memory_height, int memory_pitch, coord2 coordinate);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockPrefetchINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount, const void* src_base_pointer, int memory_width,
    int memory_height, int memory_pitch, coord2 coordinate);
SYCL_EXTERNAL float8v __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, short8v, int8v, float8v, int32_t);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_ControlBarrierWaitINTEL(int execution_scope, int memory_scope, int memory_semantics);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_ControlBarrierArriveINTEL(int execution_scope, int memory_scope, int memory_semantics);
#endif

int main(int argc, char** argv) {
  const char* tag = argc > 1 ? argv[1] : "?";
  const std::string want = argc > 2 ? argv[2] : "B60";  // gpu_selector_v picks the Arrow Lake iGPU here; choose by name
  try {
    std::optional<sycl::device> pick;
    for (auto& p : sycl::platform::get_platforms())
      for (auto& d : p.get_devices())
        if (d.is_gpu() && !pick && d.get_info<sycl::info::device::name>().find(want) != std::string::npos &&
            p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
          pick = d;
    if (!pick) { std::printf("[%s] FAIL no Level Zero GPU named *%s*\n", tag, want.c_str()); return 3; }
    sycl::queue q{*pick, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("[%s] device: %s  driver %s  platform %s\n", tag, dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::driver_version>().c_str(),
                dev.get_platform().get_info<sycl::info::platform::name>().c_str());
    const int n = 1 << 20;
    float* a = sycl::malloc_device<float>(n, q);
    float* b = sycl::malloc_device<float>(n, q);
    float* c = sycl::malloc_device<float>(n, q);
    q.fill(a, 1.0f, n);
    q.fill(b, 2.0f, n);
    q.wait();
    // Work-group 256 with sub-group 16: in 256-GRF mode an EU holds half the threads, so this also exercises the
    // dispatch bookkeeping the driver must get right for large-GRF kernels.
    q.parallel_for(sycl::nd_range<1>{n, 256}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
      int i = it.get_global_id(0);
      float x = a[i], y = b[i];
      // a little register pressure so the compiler has something to allocate
      float acc[16];
      for (int k = 0; k < 16; ++k) acc[k] = x * (k + 1) + y;
      float s = 0.f;
      for (int k = 0; k < 16; ++k) s += acc[k];
      c[i] = s;  // = 16*2 + x*(1+..+16) = 32 + 136 = 168
    });
    q.wait();
    float h[4] = {0, 0, 0, 0};
    q.memcpy(h, c, sizeof h).wait();
    bool ok = h[0] == 168.f && h[1] == 168.f && h[2] == 168.f && h[3] == 168.f;
    std::printf("[%s] A plain:               c[0..3] = %g %g %g %g -> %s\n", tag, h[0], h[1], h[2], h[3], ok ? "OK" : "WRONG");
    // B: local memory + work-group barrier + sub-group reduction. Every Xe2 library kernel has this shape; the base
    // library has it only at 128 GRF, and the base library is the one that runs (PORT-NOTES 2026-10-10 22:25).
    float* d = sycl::malloc_device<float>(n / 256, q);
    q.submit([&](sycl::handler& hnd) {
      sycl::local_accessor<float, 1> sm(256, hnd);
      hnd.parallel_for(sycl::nd_range<1>{n, 256}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        int l = it.get_local_id(0);
        int g = it.get_group(0);
        sm[l] = a[it.get_global_id(0)] + b[it.get_global_id(0)];                      // 3
        sycl::group_barrier(it.get_group());
        float v = sm[(l + 1) % 256];                                                   // the neighbour's 3
        float sg = sycl::reduce_over_group(it.get_sub_group(), v, sycl::plus<>());    // 16 * 3 = 48
        if (it.get_sub_group().get_local_id()[0] == 0) sm[l] = sg;                   // one partial per sub-group
        sycl::group_barrier(it.get_group());
        if (l == 0) { float s = 0.f; for (int k = 0; k < 256; k += 16) s += sm[k]; d[g] = s; }  // 16 * 48 = 768
      });
    }).wait();
    float hd[4] = {0, 0, 0, 0};
    q.memcpy(hd, d, sizeof hd).wait();
    bool okB = hd[0] == 768.f && hd[1] == 768.f && hd[2] == 768.f && hd[3] == 768.f;
    std::printf("[%s] B slm+barrier+subgroup: d[0..3] = %g %g %g %g -> %s\n", tag, hd[0], hd[1], hd[2], hd[3], okB ? "OK" : "WRONG");
    // C: one DPAS tile through joint_matrix (bf16 8x16 * 16x16 -> f32 8x16), one sub-group per work-group, 1024 groups.
    // All-ones inputs make the packed (VNNI) B layout a non-issue: every element of C must be K = 16.
    // Every attention kernel is a DPAS kernel and the one mhc kernel that dies is the one with has_dpas (22:25).
    {
      namespace jm = sycl::ext::oneapi::experimental::matrix;
      using bf16 = sycl::ext::oneapi::bfloat16;
      const int groups = 1024;
      bf16* A = sycl::malloc_device<bf16>(8 * 16, q);
      bf16* Bm = sycl::malloc_device<bf16>(16 * 16, q);
      float* C = sycl::malloc_device<float>(groups * 8 * 16, q);
      q.fill(A, bf16(1.0f), 8 * 16);
      q.fill(Bm, bf16(1.0f), 16 * 16);
      q.wait();
      q.parallel_for(sycl::nd_range<1>{groups * 16, 16}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major> ma;
        jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::ext_intel_packed> mb;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> mc;
        jm::joint_matrix_fill(sg, mc, 0.0f);
        jm::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(A), 16);
        jm::joint_matrix_load(sg, mb, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(Bm), 32);
        jm::joint_matrix_mad(sg, mc, ma, mb, mc);
        float* out = C + it.get_group(0) * 8 * 16;
        jm::joint_matrix_store(sg, mc, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(out), 16, jm::layout::row_major);
      });
      q.wait();
      float hc[4] = {0, 0, 0, 0};
      q.memcpy(hc, C + (groups - 1) * 8 * 16, sizeof hc).wait();  // the last group's tile
      bool okC = hc[0] == 16.f && hc[1] == 16.f && hc[2] == 16.f && hc[3] == 16.f;
      std::printf("[%s] C dpas joint_matrix:   C[0..3] = %g %g %g %g -> %s\n", tag, hc[0], hc[1], hc[2], hc[3], okC ? "OK" : "WRONG");
      sycl::free(A, q); sycl::free(Bm, q); sycl::free(C, q);
      ok = ok && okC;
    }
    // D: one 2D block load per sub-group: a 16x16 tile of u16 out of a 64x64 matrix (pitch 128 bytes), every element 7,
    // so each work-item's 16 values sum to 112. The driver does not advertise this feature (has_subgroup_2d_block_io
    // False) while every cutlass tile load depends on it.
    {
      const int rows = 64, cols = 64, groups = 16;
      unsigned short* S = sycl::malloc_device<unsigned short>(rows * cols, q);
      int* o = sycl::malloc_device<int>(groups * 16, q);
      q.fill(S, (unsigned short)7, rows * cols);
      q.wait();
      q.parallel_for(sycl::nd_range<1>{groups * 16, 16}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        int g = it.get_group(0);
        int lane = it.get_local_id(0);
        // captures must be identical in the host and device passes, so everything the builtin needs is named here
        const void* base = S;
        int width_bytes = cols * 2, height = rows, pitch_bytes = cols * 2;
        int x = (g % 4) * 16, y = (g / 4) * 16;
        int sum = -1;
#ifdef __SYCL_DEVICE_ONLY__
        ushort16v frag;
        coord2 xy = {x, y};
        __spirv_Subgroup2DBlockLoadINTEL(2, 16, 16, 1, base, width_bytes, height, pitch_bytes, xy, &frag);
        sum = 0;
        for (int k = 0; k < 16; ++k) sum += frag[k];
#else
        sum = (base && width_bytes && height && pitch_bytes && x >= 0 && y >= 0) ? -1 : -2;
#endif
        o[g * 16 + lane] = sum;
      });
      q.wait();
      int ho[4] = {0, 0, 0, 0};
      q.memcpy(ho, o + (groups - 1) * 16, sizeof ho).wait();
      bool okD = ho[0] == 112 && ho[1] == 112 && ho[2] == 112 && ho[3] == 112;
      std::printf("[%s] D 2D block load:       o[0..3] = %d %d %d %d -> %s\n", tag, ho[0], ho[1], ho[2], ho[3], okD ? "OK" : "WRONG");
      sycl::free(S, q); sycl::free(o, q);
      ok = ok && okD;
    }
    // E: one cutlass mainloop step with cutlass's own builtins: prefetch B, 2D block load A (8x16 bf16), transform
    // (VNNI) load B (16x16 bf16), split-barrier arrive/wait across two sub-groups, SubgroupMatrixMultiplyAccumulate
    // (bf16 -> f32, K=16), 2D block STORE of the 8x16 f32 tile. All ones in, so every C element must be 16.
    {
      const int rows = 64, cols = 64, groups = 4;  // 4 work-groups x 2 sub-groups = 8 tiles of 8x16 = the 64x16 C strip
      unsigned short* A = sycl::malloc_device<unsigned short>(rows * cols, q);
      unsigned short* Bm = sycl::malloc_device<unsigned short>(rows * cols, q);
      float* C = sycl::malloc_device<float>(rows * cols, q);
      q.fill(A, (unsigned short)0x3F80, rows * cols);   // bf16 1.0
      q.fill(Bm, (unsigned short)0x3F80, rows * cols);
      q.fill(C, -1.0f, rows * cols);
      q.wait();
      q.parallel_for(sycl::nd_range<1>{groups * 32, 32}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        int g = it.get_group(0);
        int s = it.get_sub_group().get_group_id()[0];
        int tile = g * 2 + s;                  // 0..7
        const void* abase = A; const void* bbase = Bm; void* cbase = C;
        int wb16 = cols * 2, h = rows, pb16 = cols * 2;   // u16 matrices: width/pitch in bytes
        int wb32 = cols * 4, pb32 = cols * 4;             // f32 matrix
        int ay = tile * 8;                     // A rows [tile*8, +8), cols [0,16)
        int bx = (tile % 4) * 16;              // B cols [bx, +16), rows [0,16)
        int cy = tile * 8;                     // C rows [tile*8, +8), cols [0,16)
        int marker = (abase && bbase && cbase && wb16 && h && pb16 && wb32 && pb32 && ay >= 0 && bx >= 0 && cy >= 0) ? 1 : 0;
#ifdef __SYCL_DEVICE_ONLY__
        coord2 axy = {0, ay}, bxy = {bx, 0}, cxy = {0, cy};
        __spirv_Subgroup2DBlockPrefetchINTEL(2, 16, 16, 1, bbase, wb16, h, pb16, bxy);
        short8v afrag;
        __spirv_Subgroup2DBlockLoadINTEL(2, 16, 8, 1, abase, wb16, h, pb16, axy, &afrag);
        int8v bfrag;
        __spirv_Subgroup2DBlockLoadTransformINTEL(2, 16, 16, 1, bbase, wb16, h, pb16, bxy, &bfrag);
        __spirv_ControlBarrierArriveINTEL(2, 2, 0);
        __spirv_ControlBarrierWaitINTEL(2, 2, 0);
        float8v acc = {0, 0, 0, 0, 0, 0, 0, 0};
        acc = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, afrag, bfrag, acc, 0x1000 | 0x2000);
        __spirv_Subgroup2DBlockStoreINTEL(4, 16, 8, 1, &acc, cbase, wb32, h, pb32, cxy);
#else
        (void)marker;
#endif
      });
      q.wait();
      float he[4] = {0, 0, 0, 0};
      q.memcpy(he, C + 7 * 8 * cols + 0, sizeof he).wait();   // first row of the last tile
      float hl[2] = {0, 0};
      q.memcpy(hl, C + (7 * 8 + 7) * cols + 14, sizeof hl).wait();  // last two elements of the last tile
      bool okE = he[0] == 16.f && he[1] == 16.f && he[2] == 16.f && he[3] == 16.f && hl[0] == 16.f && hl[1] == 16.f;
      std::printf("[%s] E cutlass step:        C = %g %g %g %g .. %g %g -> %s\n", tag, he[0], he[1], he[2], he[3], hl[0], hl[1], okE ? "OK" : "WRONG");
      sycl::free(A, q); sycl::free(Bm, q); sycl::free(C, q);
      ok = ok && okE;
    }
    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q); sycl::free(d, q);
    return (ok && okB) ? 0 : 2;
  } catch (const sycl::exception& e) {
    std::printf("[%s] FAIL sycl::exception: %s\n", tag, e.what());
    return 1;
  } catch (const std::exception& e) {
    std::printf("[%s] FAIL exception: %s\n", tag, e.what());
    return 1;
  }
}
