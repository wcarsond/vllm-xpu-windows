# probe-import.input.py - can the ported vLLM be imported on Windows, and how far does platform detection get?
# Usage:  .venv\Scripts\python.exe server\probe-import.input.py      (run by install-vllm.input.ps1; prints, no file)
# Why (2026-10-10, rung 5): three steps, each reported on its own line so the first failure names itself.
#   1. import vllm                      - the runtime port's import-time changes (winloop, envs defaults)
#   2. vllm.platforms.current_platform  - detects the Intel cards and imports vllm_xpu_kernels; expected to fail until
#                                         rung 4's build is installed, and the error text is what we want to see
#   3. the xpu platform's device name    - only if 2 passed
import platform, sys, traceback

print(f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
try:
    import vllm
    print(f"1. import vllm: ok, version {vllm.__version__}")
except Exception as e:
    print(f"1. import vllm: FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from vllm.platforms import current_platform
    print(f"2. current_platform: {current_platform.__class__.__name__} device_type={current_platform.device_type}")
except Exception as e:
    print(f"2. current_platform: FAILED: {type(e).__name__}: {e}")
    sys.exit(2)

try:
    import torch
    n = torch.xpu.device_count()
    print(f"3. {n} xpu device(s): " + ", ".join(current_platform.get_device_name(i) for i in range(n)))
except Exception as e:
    print(f"3. device names: FAILED: {type(e).__name__}: {e}")
    sys.exit(3)
