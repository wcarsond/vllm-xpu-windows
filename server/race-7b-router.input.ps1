# race-7b-router.input.ps1 - router side of the 7B race: llama-server (Vulkan, Carson's patched build) on qwen2.5-coder-7b.
# Usage:  powershell -NoProfile -File race-7b-router.input.ps1     (writes race-7b-router.output.txt)
# Why (2026-10-10): the other half of race-7b-vllm.input.py: same prompt, same 256-token budget, greedy, three runs
# after a warm-up, timings read from llama-server's own response (predicted_per_second is pure decode, so the end-to-end
# figure is computed too, to match the vLLM side). The router runs as it is configured today (presets.ini [*]:
# device = Vulkan1,Vulkan2, layer split), so this is "the router as Carson uses it", not a one-card tuning. Run it after
# the vLLM script has exited so the two never share a card. The first request loads the model (--models-max 1).
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$out   = "$here\race-7b-router.output.txt"
$model = 'qwen2.5-coder-7b'
$prompt = "Write a Python function that parses an INI file into a nested dict, with comments explaining each step."
"run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  model $model (router, Q4_K_M GGUF, Vulkan1+Vulkan2 layer split)  prompt '$prompt'" | Out-File $out -Encoding utf8

function Ask($maxTokens) {
    $body = @{ model = $model; max_tokens = $maxTokens; temperature = 0; stream = $false
               messages = @(@{ role = 'user'; content = $prompt }) } | ConvertTo-Json -Depth 4
    $t0 = Get-Date
    $r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11434/v1/chat/completions -ContentType 'application/json' -Body $body -TimeoutSec 600
    $dt = ((Get-Date) - $t0).TotalSeconds
    return @{ r = $r; dt = $dt }
}

$w = Ask 16
"warm-up (loads the model if idle): $([math]::Round($w.dt,1)) s" | Out-File $out -Append -Encoding utf8
$rates = @()
for ($i = 1; $i -le 3; $i++) {
    $a = Ask 256
    $t = $a.r.timings
    $e2e = $t.predicted_n / $a.dt
    $rates += $e2e
    "run $i : $($t.prompt_n) in, $($t.predicted_n) out, $([math]::Round($a.dt,2)) s -> $([math]::Round($e2e,1)) tok/s end to end; server says decode $([math]::Round($t.predicted_per_second,1)) tok/s, prompt $([math]::Round($t.prompt_per_second,1)) tok/s" | Out-File $out -Append -Encoding utf8
}
$sorted = $rates | Sort-Object
"median $([math]::Round($sorted[1],1)) tok/s end to end; first 200 chars: $($a.r.choices[0].message.content.Trim().Substring(0, [math]::Min(200, $a.r.choices[0].message.content.Trim().Length)))" | Out-File $out -Append -Encoding utf8
Get-Content $out
