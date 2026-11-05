"""Build voice_samples/expressions.html (grid: expression x speaker, one player each) from qwen3_expressions/meta.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f'{HERE}/../voice_samples'
meta = json.load(open(f'{ROOT}/qwen3_expressions/meta.json'))

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Qwen3-TTS expressions (French)</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:20px;color:#222}
 .sub{color:#666;margin-bottom:12px}
 table{border-collapse:collapse}
 th,td{border:1px solid #e3e3e3;padding:6px;text-align:center;vertical-align:top}
 th{background:#fafafa;position:sticky;top:0}
 th.row{text-align:left;max-width:260px;position:sticky;left:0;background:#fafafa}
 th.row small{display:block;font-weight:400;color:#666}
 audio{width:170px;height:30px;display:block}
 select{margin-top:3px;font-size:.8em}
 button{padding:6px 12px;margin-right:8px}
</style></head><body>
<h2>Qwen3-TTS 1.7B CustomVoice: French expressions</h2>
<div class="sub">Each cell is one render (speaker x expression). Expressions are approximated with onomatopoeia in the text plus an English style instruction; Qwen3-TTS has no official non-verbal tags, so results vary. Rate 1-5 (saved in this browser).</div>
<p><button id="export">Export ratings (JSON)</button><button id="reset">Clear</button></p>
<table id="t"></table>
<script>
const META = __META__;
const KEY = 'qwen-expr-ratings-v1';
let store = JSON.parse(localStorage.getItem(KEY) || '{}');
const speakers = [...new Set(Object.keys(META).map(k => k.split('/')[0]))];
const exprs = [...new Set(Object.keys(META).map(k => k.split('/')[1]))];
const t = document.getElementById('t');
function render() {
  let h = '<tr><th></th>' + speakers.map(s => `<th>${s}</th>`).join('') + '<th>avg</th></tr>';
  for (const e of exprs) {
    const any = META[speakers[0] + '/' + e] || Object.values(META).find((_, i) => Object.keys(META)[i].endsWith('/' + e));
    let sum = 0, n = 0;
    h += `<tr><th class="row">${e}<small>${any.text}</small><small><i>${any.instruct}</i></small></th>`;
    for (const s of speakers) {
      const id = s + '/' + e, m = META[id];
      if (!m) { h += '<td></td>'; continue; }
      const r = +store[id] || 0; if (r) { sum += r; n++; }
      h += `<td><audio controls preload="none" src="qwen3_expressions/${id}.wav"></audio>` +
           `<select data-id="${id}"><option value="">-</option>` +
           [1,2,3,4,5].map(x => `<option ${r === x ? 'selected' : ''}>${x}</option>`).join('') +
           `</select> <small>${m.dur}s</small></td>`;
    }
    h += `<td>${n ? (sum / n).toFixed(1) : ''}</td></tr>`;
  }
  h += '<tr><th class="row">avg per voice</th>' + speakers.map(s => {
    const v = exprs.map(e => +store[s + '/' + e]).filter(x => x > 0);
    return `<td>${v.length ? (v.reduce((a, b) => a + b, 0) / v.length).toFixed(1) : ''}</td>`;
  }).join('') + '<td></td></tr>';
  t.innerHTML = h;
}
t.addEventListener('change', e => { if (e.target.dataset.id) { store[e.target.dataset.id] = e.target.value; localStorage.setItem(KEY, JSON.stringify(store)); render(); } });
document.addEventListener('play', e => document.querySelectorAll('audio').forEach(a => { if (a !== e.target) a.pause(); }), true);
document.getElementById('export').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(store, null, 2)], {type: 'application/json'}));
  a.download = 'qwen_expression_ratings.json'; a.click();
};
document.getElementById('reset').onclick = () => { if (confirm('Clear ratings?')) { store = {}; localStorage.setItem(KEY, '{}'); render(); } };
render();
</script></body></html>
"""
open(f'{ROOT}/expressions.html', 'w').write(TEMPLATE.replace('__META__', json.dumps(meta, ensure_ascii=False)))
print('wrote expressions.html with', len(meta), 'renders')
