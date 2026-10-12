// compat-devices.cpp - what cutlass's compat::dev_mgr sees on this box: the default device (its index 0, the one every
// query_device_multiprocessor_count(0) in the attention and MoE kernels reads), then every device of every backend in
// enumeration order, each with backend, name, and the Xe-core count cutlass uses as sm_count (gpu_slices x gpu_subslices).
// Usage:  cmd /c probe\compat-devices.input.cmd      (writes probe\compat-devices.output.txt)
// Why (2026-10-11 03:58): the zombie stacks showed attn_kernels_xe_2.dll holding an OpenCL context at exit. dev_mgr builds a
// context and a queue for *every* device of every backend (cute/util/compat/device.hpp), and the kernels size their tile
// schedulers from device 0 of that list, whatever card torch is actually using. If device 0 is the Arrow Lake iGPU (4 Xe
// cores) instead of a B60 (20), every cutlass kernel on this box is launched for the wrong machine.
#define _CRT_SECURE_NO_WARNINGS
#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <cstdlib>

static void show(const sycl::device& d, const std::string& tag) {
    std::string be = "other";
    switch (d.get_backend()) {
        case sycl::backend::ext_oneapi_level_zero: be = "level_zero"; break;
        case sycl::backend::opencl: be = "opencl"; break;
        default: break;
    }
    std::cout << tag << "  " << be << "  " << (d.is_gpu() ? "gpu" : d.is_cpu() ? "cpu" : "acc") << "  "
              << d.get_info<sycl::info::device::name>();
    if (d.is_gpu()) {
        try {
            auto s = d.get_info<sycl::ext::intel::info::device::gpu_slices>();
            auto ss = d.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
            auto eu = d.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
            std::cout << "  slices " << s << " x subslices " << ss << " = sm_count " << s * ss << ", EUs " << eu;
        } catch (std::exception& e) {
            std::cout << "  (no Intel GPU info: " << e.what() << ")";
        }
    }
    std::cout << "  usm_device=" << (d.has(sycl::aspect::usm_device_allocations) ? "yes" : "no") << "\n";
}

int main() {
    const char* sel = std::getenv("ONEAPI_DEVICE_SELECTOR");
    std::cout << "ONEAPI_DEVICE_SELECTOR=" << (sel ? sel : "(unset)") << "\n";
    try {
        sycl::device def(sycl::default_selector_v);
        show(def, "default_selector (compat device 0):");
    } catch (std::exception& e) {
        std::cout << "default_selector threw: " << e.what() << "\n";
    }
    int i = 0;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::all))
        show(d, "all[" + std::to_string(i++) + "]:");
    return 0;
}
