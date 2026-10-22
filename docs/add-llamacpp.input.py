"""add-llamacpp.input.py - the call: the llama.cpp concurrency blocks (server/concurrency-sweep-llamacpp.output.md, 2026-10-22)
added under the vLLM ones on the results page (docs/results-2026-10-21.body.html, docs/index.html rebuilt) and in README.md's
"Results at a glance". One block per model; a model whose heading is already on the page is skipped, so the 27B can be added
by a later run once its block exists. Same build of llama-server as the router runs (0.5.0-dev build 55, vk-allreduce-fa).

    .venv312\\Scripts\\python.exe docs\\add-llamacpp.input.py"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = [
    ('Qwen2.5-Coder-7B Q8_0 on llama.cpp', '--parallel 16, -c 135,168 (8,448 tokens per slot); the single-stream race in the table above ran one slot',
     [('0', ['42.9', '81.8', '159.3', '263.2', '174.9'], '314'), ('8K', ['39.5', '73.1', '129.3', '67.2', '102.9'], '2,526'),
      ('32K', ['32.6', '56.1', '82.1', '24.9', '—'], '1,175')]),
    ('Qwen3.6-35B-A3B UD-Q4_K_XL on llama.cpp', 'the fastest big model on these cards; 64K did not come up on two cards',
     [('0', ['72.1', '113.4', '159.5', '198.6', '139.1'], '111'), ('8K', ['68.7', '101.0', '138.3', '122.3', '58.8'], '2,254'),
      ('32K', ['59.6', '86.8', '112.7', '96.7', '114.3*'], '1,312'), ('64K', ['—', '—', '—', '—', '—'], '')]),
    ('Qwen3.6-27B UD-Q4_K_XL on llama.cpp', 'text-only (the vision projector makes this build refuse token-id prompts); 32K ran 8 slots, 64K 4 slots; n/r = skipped for time (prefill alone over 400 s)',
     [('0', ['18.0', '26.1', '41.8', '58.8', '45.6'], '60'), ('8K', ['17.5', '24.9', '39.0', '23.8', '31.9'], '823'),
      ('32K', ['16.1', '22.2', '23.0', 'n/r', '—'], '535'), ('64K', ['14.6', '13.9', 'n/r', '—', '—'], '350')]),
]
NOTE = ('llama.cpp, same two cards, same measure: aggregate decode tok/s while all requests were decoding together, prefill excluded, '
        '128 tokens per request. llama-server prefills prompts one slot at a time, so its raw throughput with prefill included is far '
        'below vLLM\'s at 8K and 32K (the 7B at 8K x 16: 25 tok/s raw against vLLM\'s 92). A dash means that many slots of that context '
        'did not come up on the two cards. The dips at 8 and 16 streams are real and repeatable on this build. '
        'llama-server 0.5.0-dev (build 55, vk-allreduce-fa), 2026-10-22; raw numbers in server/concurrency-sweep-llamacpp.output.json.')
HEADS = ['Context', '1', '2', '4', '8', '16', 'Prefill, 1 req']

def table_html(rows):
    out = ['    <div class="box"><table>',
           '      <thead><tr>' + ''.join('<th%s>%s</th>' % (' class="n"' if i else '', h) for i, h in enumerate(HEADS)) + '</tr></thead>', '      <tbody>']
    for ctx, cells, pre in rows:
        num = lambda c: c not in ('—', 'n/r')
        vals = [float(c.rstrip('*')) for c in cells if num(c)]; best = max(vals) if vals else None
        tds = ''.join('<td class="n%s">%s</td>' % (' win' if num(c) and float(c.rstrip('*')) == best else '', c) for c in cells)
        out.append('        <tr><td>%s</td>%s<td class="n">%s</td></tr>' % (ctx, tds, pre))
    out += ['      </tbody>', '    </table></div>']; return '\n'.join(out)

body = os.path.join(ROOT, 'docs', 'results-2026-10-21.body.html'); s = open(body, encoding='utf-8').read()
r = os.path.join(ROOT, 'README.md'); t = open(r, encoding='utf-8').read()
anchor = '    <p class="note">' + 'Aggregate decode tok/s: all requests'
assert anchor in s, 'vLLM note anchor missing'
added = []
for name, sub, rows in BLOCKS:
    if name in s: continue
    block = ('    <h3 style="font-size:17px;margin:14px 0 6px;color:var(--head)">%s</h3>\n    <p class="sub">%s</p>\n%s\n' % (name, sub, table_html(rows)))
    s = s.replace(anchor, block + anchor); added.append(name)
    md = ['*%s*, %s' % (name, sub), '', '| context | 1 | 2 | 4 | 8 | 16 | prefill tok/s, 1 req |', '|---|---:|---:|---:|---:|---:|---:|']
    md += ['| %s | %s | %s |' % (ctx, ' | '.join(cells), pre) for ctx, cells, pre in rows] + ['']
    i = t.index('\n## ', t.index('## Results at a glance (2026-10-21)') + 10); t = t[:i] + '\n' + '\n'.join(md) + t[i:]
if added and 'llama.cpp, same two cards, same measure' not in s:
    s = s.replace(anchor, '    <p class="note">%s</p>\n' % NOTE + anchor)
    i = t.index('\n## ', t.index('## Results at a glance (2026-10-21)') + 10); t = t[:i] + '\n' + NOTE.replace('server/', '`server/`') + '\n' + t[i:]
open(body, 'w', encoding='utf-8').write(s); open(r, 'w', encoding='utf-8').write(t)
head, main = s.split('<main>', 1)
idx = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       '<style>html{color-scheme:dark}body{margin:0}img{max-width:100%}</style>\n' + head + '</head>\n<body>\n<main>' + main + '</body>\n</html>\n')
open(os.path.join(ROOT, 'docs', 'index.html'), 'w', encoding='utf-8').write(idx)
print('added:', added or 'nothing new')
