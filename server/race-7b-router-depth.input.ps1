# race-7b-router-depth.input.ps1 - router side of the depth leg: llama-server (Carson's Vulkan build) on qwen2.5-coder-7b at 0/4K/8K/16K/32K tokens of context.
# Usage:  powershell -NoProfile -File server\race-7b-router-depth.input.ps1     (writes server\race-7b-router-depth.output.txt)
# Why (2026-10-10, Carson: "something i need to be sure you are testing is vulkan stays fast at depth, or faster"): the first
# race used a 21-token prompt. His coopmat fix is about long context (pp512 at 8K/16K/32K). Same model, same server as he
# runs it; per depth: prompt tokens actually seen, prefill tok/s and decode tok/s from the server's own timings, 128 tokens
# greedy, second of two runs reported (the first warms the KV layout). Depths above the server's context size are skipped.
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$out   = "$here\race-7b-router-depth.output.txt"
$model = 'qwen2.5-coder-7b'
$base  = 'http://127.0.0.1:11434'
$para  = "The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report on regional logistics, inventory turnover, and the maintenance schedule for the conveyor system in building four. "
"run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  model $model (router, Q4_K_M GGUF, Vulkan1+Vulkan2 layer split)  depth leg" | Out-File $out -Encoding utf8

function Ask($text, $maxTokens) {
    # cache_prompt false: llama-server reuses the KV cache for a repeated prefix by default, so the second run of the same
    # prompt reported prompt_n 1 and a meaningless prefill figure (23:53 run). Every request here evaluates its whole prompt.
    $body = @{ model = $model; max_tokens = $maxTokens; temperature = 0; stream = $false; cache_prompt = $false
               messages = @(@{ role = 'user'; content = $text }) } | ConvertTo-Json -Depth 4 -Compress
    $t0 = Get-Date
    $r = Invoke-RestMethod -Method Post -Uri "$base/v1/chat/completions" -ContentType 'application/json' -Body $body -TimeoutSec 1200
    return @{ r = $r; dt = ((Get-Date) - $t0).TotalSeconds }
}

$w = Ask "Say OK." 4
"warm-up (loads the model if idle): $([math]::Round($w.dt,1)) s" | Out-File $out -Append -Encoding utf8
$nctx = $null
try { $props = Invoke-RestMethod -Uri "$base/props" -TimeoutSec 30; $nctx = $props.default_generation_settings.n_ctx } catch {}
"server context size (n_ctx): $nctx" | Out-File $out -Append -Encoding utf8

# ~1.3 tokens per word on this tokenizer; the server reports the exact prompt_n, which is what gets written down
foreach ($depth in 0, 4096, 8192, 16384, 32768) {
    if ($nctx -and ($depth + 200) -gt $nctx) { "depth $depth : skipped, exceeds n_ctx $nctx" | Out-File $out -Append -Encoding utf8; continue }
    $words = [math]::Floor($depth / 1.3)
    $filler = ""
    while (($filler.Split(' ').Count) -lt $words) { $filler += $para }
    $text = if ($depth -eq 0) { "Write a Python function that parses an INI file into a nested dict, with comments explaining each step." } else { "Here is a document:`n$filler`nNow, write a Python function that parses an INI file into a nested dict, with comments explaining each step." }
    $null = Ask $text 128      # warm run at this depth
    $a = Ask $text 128
    $t = $a.r.timings
    "depth $depth : prompt_n $($t.prompt_n), prefill $([math]::Round($t.prompt_per_second,1)) tok/s ($([math]::Round($t.prompt_ms/1000,2)) s), decode $([math]::Round($t.predicted_per_second,1)) tok/s over $($t.predicted_n) tokens, wall $([math]::Round($a.dt,2)) s" | Out-File $out -Append -Encoding utf8
}
Get-Content $out
