@echo off
rem run-linear-mask.input.cmd - probe/linear-shapes2.input.py again, but under ZE_AFFINITY_MASK=0 like the vLLM runs.
rem Usage:  cmd /c probe\run-linear-mask.input.cmd      (writes probe\linear-shapes2.output.txt, same file as the unmasked run)
rem Why (2026-10-10): every passing F.linear probe ran with both cards visible; smoke-offline.input.py sets ZE_AFFINITY_MASK=0
rem so the engine sees one card. A device mask changes enumeration in the Level Zero adapter; if oneDNN's GEMM fails to
rem launch only under the mask, that is the whole difference and the fix is to pick the card with torch.xpu.set_device.
set HERE=%~dp0
set ZE_AFFINITY_MASK=0
"%HERE%..\.venv312\Scripts\python.exe" "%HERE%linear-shapes2.input.py"
