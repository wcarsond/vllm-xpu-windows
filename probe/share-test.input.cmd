@echo off
rem share-test.input.cmd - is SHARING the checkpoint's pages between two cards the trigger? Same two readers, same files,
rem copy into device tensors retained up to a 20 GiB cap (no overflow, vLLM's shape), no affinity mask:
rem   same     both readers read every tensor      (tensor split: the same physical page-cache pages go to both cards)
rem   disjoint reader r reads tensors with index%2==r (pipeline split: no page is read by both)
rem Three repeats each; the fault is a race at vLLM's size, so counts matter. Verdicts in probe\share-test.output.txt.
rem Usage:  cmd /c probe\share-test.input.cmd      (~4 min)
rem Why (2026-10-11 18:45): at the moment of the vLLM fault the host had 14.5 GB free and the workers 2.8 GB resident, so
rem memory pressure is out. Tensor split faults, pipeline split does not, at the same memory per card; the private-heap
rem storms never fault. What tensor split alone does is feed the SAME file pages to two adapters at once.
setlocal enabledelayedexpansion
set HERE=%~dp0
set PY=%HERE%..\.venv312\Scripts\python.exe
set LOG=%HERE%share-test.output.txt
set MMAP_DEVICE_PER_READER=1
set MMAP_NO_MASK=1
set MMAP_COPY_INTO=1
set MMAP_RETAIN_CAP_GB=20
echo run %DATE% %TIME%  two readers, one card each, cap 20 GiB, 3 repeats each > "%LOG%"
rem same+clone = both read every tensor, but each DMA runs from a PRIVATE heap copy (MMAP_CLONE=1): Carson's "two
rem cursors" - CPUs share the file pages, the cards only ever see private memory. If same faults and same+clone
rem does not, the fix is a staging copy per tensor in the loader, no second file, no lock, no stagger.
for %%M in (same disjoint same+clone) do (
    for /l %%I in (1,1,3) do (
        if "%%M"=="disjoint" (set MMAP_DISJOINT=1) else (set MMAP_DISJOINT=)
        if "%%M"=="same+clone" (set MMAP_CLONE=1) else (set MMAP_CLONE=)
        echo. >> "%LOG%"
        echo == %%M  repeat %%I  !TIME! >> "%LOG%"
        "%PY%" "%HERE%mmap-readers.input.py" Qwen2.5-72B-Instruct-AWQ 2 1 2>&1 | findstr /c:"reader" /c:"wall" >> "%LOG%"
    )
)
type "%LOG%"
endlocal
