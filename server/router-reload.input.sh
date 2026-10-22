# router-reload.input.sh - the call: restart the vllm-b60 service (the router on 11434) so an edit to vllm-router.input.py takes
# effect. The service needs an admin token; this shell on the big PC has none, but the Pi's SSH session into the big PC does
# (service-control.input.ps1's note, proven 2026-10-12), so the restart goes big PC -> AIPI -> big PC. Home side only.
# 2026-10-21: after the idle-unloader fix (an answer in flight is never idle). Run only when nothing is being answered:
# check http://127.0.0.1:11434/router/status first; "loaded": null means nothing is lost.
#   bash C:/Users/claude/scripts/vllm-xpu-windows/server/router-reload.input.sh
curl -s -m 5 http://127.0.0.1:11434/router/status | cut -c1-100; echo
ssh -o BatchMode=yes aipi 'ssh -o BatchMode=yes bigpc "powershell -NoProfile -ExecutionPolicy Bypass -File C:/Users/claude/scripts/vllm-xpu-windows/server/service-control.input.ps1 restart"'
sleep 8
curl -s -m 5 http://127.0.0.1:11434/router/status | cut -c1-100; echo
