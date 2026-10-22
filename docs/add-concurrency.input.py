"""add-concurrency.input.py - the call: the concurrency sweep's three decode blocks (server/concurrency-sweep.output.md,
2026-10-21/22) added to the results page (docs/results-2026-10-21.body.html, the published page; docs/index.html rebuilt
from it for GitHub Pages) and to README.md's "Results at a glance". Inserted once; rerunning finds the heading and stops.

    .venv312\\Scripts\\python.exe docs\\add-concurrency.input.py"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = [
    ('Qwen3.6-27B GPTQ-Int4', 'pulled 2026-10-21 (raydelossantos, same recipe as Qwen\'s own 3.5 GPTQ), 30.3 GB',
     [('0', ['29.1', '55.9', '104.1', '79.0', '160.1'], '238'), ('8K', ['28.3', '53.6', '96.8', '81.0', '73.8*'], '1,906'),
      ('32K', ['26.8', '48.1', '82.1', '—', '—'], '1,712'), ('64K', ['24.6', '43.3', '—', '—', '—'], '1,502')]),
    ('Qwen3.5-27B GPTQ-Int4', 'the router\'s preset',
     [('0', ['29.1', '55.9', '104.1', '76.6', '145.5'], '232'), ('8K', ['28.3', '53.4', '97.3', '78.8', '72.6*'], '1,924'),
      ('32K', ['26.7', '48.1', '81.5', '—', '—'], '1,717'), ('64K', ['24.6', '43.3', '—', '—', '—'], '1,506')]),
    ('Qwen2.5-Coder-7B bf16', '',
     [('0', ['56.6', '111.8', '215.3', '412.1', '466.4'], '499'), ('8K', ['53.9', '103.2', '188.4', '330.0', '458.6'], '7,210'),
      ('32K', ['49.2', '87.4', '143.6', '205.1', '—'], '5,417')]),
]
NOTE = ('Aggregate decode tok/s: all requests\' generated tokens while all were decoding together, divided by that time, prefill '
        'excluded; 128 tokens generated per request, temperature 0, prefix cache hits measured 0. Prefill is one request. '
        'A dash means N x (context + 128) is more than the KV cache vLLM reported, so the cell was not run. The star at 8K x 16 '
        'on the 27Bs: two preemptions, the requests never all decoded together. The dip at 8 requests on both 27Bs is real: '
        'on two Windows cards vLLM captures decode graphs for batch 1, 2 and 4 only, so 8 and 16 run without graphs; the 7B shows no dip. '
        'vLLM 0.1.dev1+g276fbcff2.d20261010.xpu, 2026-10-21/22; raw numbers in server/concurrency-sweep.output.json.')
HEADS = ['Context', '1', '2', '4', '8', '16', 'Prefill, 1 req']

def html():
    out = ['  <section>', '    <h2>Under load: concurrent requests by context</h2>',
           '    <p class="sub">vLLM on two B60s, decode tok/s for all streams together. Columns are concurrent requests</p>']
    for name, sub, rows in BLOCKS:
        out.append('    <h3 style="font-size:17px;margin:14px 0 6px;color:var(--head)">%s</h3>' % name)
        if sub: out.append('    <p class="sub">%s</p>' % sub)
        out.append('    <div class="box"><table>')
        out.append('      <thead><tr>' + ''.join('<th%s>%s</th>' % (' class="n"' if i else '', h) for i, h in enumerate(HEADS)) + '</tr></thead>')
        out.append('      <tbody>')
        for ctx, cells, pre in rows:
            best = max((float(c.rstrip('*')) for c in cells if c != '—'), default=0)
            tds = ''.join('<td class="n%s">%s</td>' % (' win' if c != '—' and float(c.rstrip('*')) == best else '', c) for c in cells)
            out.append('        <tr><td>%s</td>%s<td class="n">%s</td></tr>' % (ctx, tds, pre))
        out.append('      </tbody>'); out.append('    </table></div>')
    out.append('    <p class="note">%s</p>' % NOTE); out.append('  </section>'); out.append('')
    return '\n'.join(out)

def md():
    out = ['**Under load: concurrent requests by context** (vLLM on two B60s, aggregate decode tok/s for all streams together, prefill excluded; 128 tokens per request)', '']
    for name, sub, rows in BLOCKS:
        out.append('*%s*%s' % (name, (', ' + sub) if sub else '')); out.append('')
        out.append('| context | 1 | 2 | 4 | 8 | 16 | prefill tok/s, 1 req |'); out.append('|---|---:|---:|---:|---:|---:|---:|')
        for ctx, cells, pre in rows: out.append('| %s | %s | %s |' % (ctx, ' | '.join(cells), pre))
        out.append('')
    out.append(NOTE.replace('server/', '`server/`')); out.append('')
    return '\n'.join(out)

body = os.path.join(ROOT, 'docs', 'results-2026-10-21.body.html'); s = open(body, encoding='utf-8').read()
if 'Under load' in s: print('already there'); raise SystemExit
anchor = '  <section>\n    <h2>What was broken on Windows</h2>'
assert anchor in s; s = s.replace(anchor, html() + anchor); open(body, 'w', encoding='utf-8').write(s)
head, main = s.split('<main>', 1)
idx = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       '<style>html{color-scheme:dark}body{margin:0}img{max-width:100%}</style>\n' + head + '</head>\n<body>\n<main>' + main + '</body>\n</html>\n')
open(os.path.join(ROOT, 'docs', 'index.html'), 'w', encoding='utf-8').write(idx)
r = os.path.join(ROOT, 'README.md'); t = open(r, encoding='utf-8').read()
a = t.index('## Results at a glance (2026-10-21)'); b = t.index('\n## ', a + 10)
t = t[:b] + '\n' + md() + t[b:]; open(r, 'w', encoding='utf-8').write(t)
print('page, index and README updated')
