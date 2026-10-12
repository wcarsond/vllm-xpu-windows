# model-speed.input.ps1 - ask the model llama-server already has loaded for 64 tokens and read its own timing.
# Usage:  powershell -NoProfile -File model-speed.input.ps1 [-Model qwen-coder-30b]
# Why (2026-10-02): Windows' GPU counters read 0 on both B60s and the cards' own query does not see other
# processes, so neither says whether the model is on a card. Speed does: a 30B-A3B model at Q3 runs several
# times faster on a B60 than on six CPU threads.
param([string]$Model = 'qwen-coder-30b')
$body = @{ model = $Model; max_tokens = 64; stream = $false
           messages = @(@{ role = 'user'; content = 'Count from 1 to 40, separated by spaces.' }) } | ConvertTo-Json -Depth 4
$r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11434/v1/chat/completions -ContentType 'application/json' -Body $body -TimeoutSec 120
"{0}: {1} tokens written at {2:N1} tokens/s; prompt read at {3:N1} tokens/s" -f $Model, $r.timings.predicted_n, $r.timings.predicted_per_second, $r.timings.prompt_per_second
