# oneproc-ab.input.ps1 - what each one-process decode change is worth: the smoke test (7B, two cards, full decode graphs)
# with async scheduling and the pre-replay hand-over each switched off in turn.
# Usage:  powershell -File server\oneproc-ab.input.ps1        (writes server\oneproc-ab.output.txt)
# Why (2026-10-13): one process went 49.9 -> 56.5 tok/s with both switched on together; which one carries it decides what
# is worth keeping and explaining.
$root = "C:\Users\claude\scripts\vllm-xpu-windows"
$out = "$root\server\oneproc-ab.output.txt"
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  7B, two cards, one process, full decode graphs; decode tok/s over 256 tokens")
foreach ($c in @(@("1", "1"), @("0", "1"), @("1", "0"), @("0", "0"))) {
    $env:VLLM_XPU_ONEPROC_ASYNC = $c[0]; $env:VLLM_XPU_ONEPROC_ALIGN = $c[1]
    cmd /c "$root\server\oneproc-smoke.input.cmd" graphs oneproc | Out-Null
    $d = (Select-String -Path "$root\server\oneproc-smoke-oneproc-graphs.output.txt" -Pattern '^decode ').Line
    $b = (Select-String -Path "$root\server\oneproc-smoke-oneproc-graphs.output.txt" -Pattern '^baton ').Line
    $rows += "async $($c[0]) align $($c[1]): $d"
    $rows += "    $b"
    $rows | Out-File $out -Encoding utf8
}
Remove-Item Env:VLLM_XPU_ONEPROC_ASYNC, Env:VLLM_XPU_ONEPROC_ALIGN -ErrorAction SilentlyContinue
$rows
