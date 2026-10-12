# trace-kernels.input.ps1 - name every kernel launched in the failing step of a SYCL_UR_TRACE log, in order, with results.
# Usage:  powershell -NoProfile -File trace-kernels.input.ps1     (writes probe\trace-kernels.output.txt)
# Why (2026-10-10): the trace shows urEnqueueKernelLaunch calls by handle only; names come from the urKernelCreate
# (pKernelName) and urKernelGetInfo(FUNCTION_NAME) lines that produced each handle, often thousands of lines earlier.
# Joining them gives the real sequence of kernels before the launch that fails, including our SYCL kernels.
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$trace = Join-Path $here '..\server\smoke-offline.trace.log'
$out   = Join-Path $here 'trace-kernels.output.txt'
$lines = Get-Content $trace
$names = @{}
foreach ($l in $lines) {
    # SYCL kernels: pKernelName = <addr> (<mangled typeid name>); Triton/oneDNN: FUNCTION_NAME queries below.
    if ($l -match 'urKernelCreate\(.*pKernelName = [0-9A-F]+ \(([^)]+)\).*phKernel = [0-9A-F]+ \(([0-9A-F]+)\)') { $names[$Matches[2]] = $Matches[1] }
    elseif ($l -match 'urKernelCreate\(.*pKernelName = "([^"]+)".*phKernel = [0-9A-F]+ \(([0-9A-F]+)\)') { $names[$Matches[2]] = $Matches[1] }
    elseif ($l -match 'urKernelGetInfo\(\.hKernel = ([0-9A-F]+), \.propName = UR_KERNEL_INFO_FUNCTION_NAME, \.propSize = \d+, \.pPropValue = [0-9A-F]+ \(([^)]+)\)') { $names[$Matches[1]] = $Matches[2] }
}
$start = ($lines | Select-String -Pattern 'engine up in' | Select-Object -First 1).LineNumber
$end   = ($lines | Select-String -Pattern '-> UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY' | Select-Object -First 1).LineNumber
$rows = @("run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  trace $trace  step window lines $start..$end  ($($names.Count) named kernels)")
$n = 0
for ($i = $start - 1; $i -lt $end; $i++) {
    $l = $lines[$i]
    if ($l -match '<--- urEnqueueKernelLaunch(WithArgsExp)?\(.*\.hKernel = ([0-9A-F]+).*pGlobalWorkSize = [0-9A-F]+ \(([0-9]+)\).*pLocalWorkSize = [0-9A-F]+ \(([0-9]+)\).*-> (UR_RESULT_[A-Z_]+)') {
        $n++; $h = $Matches[2]; $nm = if ($names.ContainsKey($h)) { $names[$h] } else { "? ($h)" }
        $rows += ('{0,3}  {1,-60} global0 {2,-8} local0 {3,-5} {4}' -f $n, $nm.Substring(0, [math]::Min(60, $nm.Length)), $Matches[3], $Matches[4], $Matches[5])
    }
}
$rows | Out-File $out -Encoding utf8
Get-Content $out
