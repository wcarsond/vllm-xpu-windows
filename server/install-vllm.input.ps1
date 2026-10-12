# install-vllm.input.ps1 - put the ported vLLM (src\vllm, branch windows-xpu) and its Python dependencies into .venv.
# Usage:  powershell -NoProfile -File install-vllm.input.ps1     (writes install-vllm.output.log; pip detail in install-vllm.pip.log)
#
# Why (2026-10-10, rung 5): upstream vLLM main plus the Windows runtime port lives in src\vllm. For VLLM_TARGET_DEVICE=xpu
# setup.py builds no C++ (the kernels are the separate vllm_xpu_kernels package), so this is a pure-Python install.
#   --no-build-isolation : pyproject's build requirements pin torch==2.13.0; the venv holds torch 2.14.1+xpu and keeps it.
#   --no-deps            : requirements/xpu.txt pins torch==2.14.0 and vllm_xpu_kernels==0.1.15.4 (Linux wheel); both are ours
#                          to supply. Runtime deps come from requirements/common.txt, which has no torch or kernels in it.
#   setuptools-rust      : setup.py imports it at the top; the Rust extensions themselves are optional and skipped without cargo.
#   winloop              : the Windows stand-in for uvloop that the port imports.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = "$here\install-vllm.output.log"
$plog = "$here\install-vllm.pip.log"
# .venv312, not .venv: upstream main requires Python >= 3.11 (found 17:17 on the 3.10 venv); venv312.input.ps1 makes it.
$py   = "$here\..\.venv312\Scripts\python.exe"
$src  = "$here\..\src\vllm"
"run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  branch $(git -C $src branch --show-current) at $(git -C $src rev-parse --short HEAD)" | Out-File $log -Encoding utf8
"" | Out-File $plog -Encoding utf8

# The parameter must not be called $args: that name is PowerShell's automatic unbound-argument array, and a
# parameter named $args made "@args" splat an empty list, so the first run (17:16) executed bare "pip" three times.
function Step($name, $pipArgs) {
    $t0 = Get-Date
    "== $name" | Out-File $plog -Append -Encoding utf8
    & $py -m pip @pipArgs 2>&1 | ForEach-Object { "$_" } | Out-File $plog -Append -Encoding utf8
    $rc = $LASTEXITCODE
    "$name : exit $rc in $([math]::Round(((Get-Date)-$t0).TotalSeconds)) s" | Out-File $log -Append -Encoding utf8
    return $rc
}

$rc1 = Step 'build tools + winloop' @('install', '--quiet', 'setuptools-rust', 'setuptools-scm', 'setuptools>=77,<81', 'wheel', 'cmake', 'ninja', 'packaging', 'jinja2', 'regex', 'winloop')
$rc2 = Step 'runtime deps (requirements/common.txt)' @('install', '--quiet', '-r', "$src\requirements\common.txt")
if ($rc2 -ne 0) {
    "runtime deps failed; last pip lines:" | Out-File $log -Append -Encoding utf8
    Get-Content $plog -Tail 15 | ForEach-Object { "  $_" } | Out-File $log -Append -Encoding utf8
}
$env:VLLM_TARGET_DEVICE = 'xpu'
$rc3 = Step 'vllm editable, no deps' @('install', '--no-deps', '--no-build-isolation', '-e', $src, '-v')
if ($rc3 -ne 0) {
    "vllm install failed; last pip lines:" | Out-File $log -Append -Encoding utf8
    Get-Content $plog -Tail 25 | ForEach-Object { "  $_" } | Out-File $log -Append -Encoding utf8
}

# Proof 1: the package imports. Proof 2: platform detection, which reaches for vllm_xpu_kernels and is
# expected to fail until rung 4 delivers them; its exact error is the record.
"-- torch untouched? " + (& $py -m pip list 2>$null | Select-String '^torch ') | Out-File $log -Append -Encoding utf8
"-- import vllm:" | Out-File $log -Append -Encoding utf8
& $py "$here\probe-import.input.py" 2>&1 | ForEach-Object { "  $_" } | Out-File $log -Append -Encoding utf8
Get-Content $log
