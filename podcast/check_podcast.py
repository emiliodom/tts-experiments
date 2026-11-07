"""Checks for the podcast episodes.

Critical (exit 1 on failure): script structure, speakers match the requested trio,
audio files exist, per-turn durations plausible, total length, mp3 decodes.
Non-critical (warnings only): ASR round-trip (Whisper) similarity per turn.

    python check_podcast.py [--asr]      (--asr needs whisper: use cosy_repo/.venv/bin/python)
"""
import os, re, sys, json, difflib, subprocess, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
TRIOS = {'ep1': {'ryan', 'dylan', 'sohee'}, 'ep2': {'aiden', 'eric', 'serena'},
         'ep3': {'vivian', 'ono_anna', 'ryan'}}
fails, warns = [], []


def norm(s):
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]+', ' ', s).split()


for ep, trio in TRIOS.items():
    sc = json.load(open(f'{HERE}/{ep}.json', encoding='utf-8'))
    words = sum(len(t.split()) for _, t in sc['turns'])
    if set(sc['cast'].values()) != trio: fails.append(f'{ep}: cast {set(sc["cast"].values())} != {trio}')
    if not 350 <= words <= 900: fails.append(f'{ep}: {words} words outside 350-900')
    if any(n not in sc['cast'] for n, _ in sc['turns']): fails.append(f'{ep}: unknown speaker in turns')
    if {sc['cast'][n] for n, _ in sc['turns']} != trio: fails.append(f'{ep}: a trio member never speaks')
    long = [i for i, (_, t) in enumerate(sc['turns']) if len(t.split()) > 45]
    if long: fails.append(f'{ep}: turns too long for stable TTS: {long}')

    mp3, meta_p = f'{HERE}/audio/{ep}.mp3', f'{HERE}/audio/{ep}.meta.json'
    if not (os.path.exists(mp3) and os.path.exists(meta_p)):
        fails.append(f'{ep}: audio missing'); continue
    meta = json.load(open(meta_p, encoding='utf-8'))
    if len(meta['turns']) != len(sc['turns']): fails.append(f'{ep}: turn count mismatch')
    for t in meta['turns']:
        if len(t['text'].split()) < 6: continue          # sign-offs like 'Au revoir !' are too short for a rate test
        wps = len(t['text'].split()) / t['duration']       # speech rate, words/s (normal 2-3.5)
        if not 1.2 <= wps <= 4.5: fails.append(f'{ep} turn {t["i"]}: {wps:.2f} words/s ({t["duration"]}s)')
    if not 120 <= meta['total_s'] <= 600: fails.append(f'{ep}: total {meta["total_s"]}s outside 120-600')
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', mp3, '-f', 'null', '-'], capture_output=True, text=True)
    if r.returncode or r.stderr.strip(): fails.append(f'{ep}: mp3 decode error {r.stderr[:100]}')
    print(f'{ep}: {words} words, {meta["total_s"]}s, {len(meta["turns"])} turns')

if '--asr' in sys.argv:
    import whisper
    m = whisper.load_model('base')
    for ep in TRIOS:
        meta_p = f'{HERE}/audio/{ep}.meta.json'
        if not os.path.exists(meta_p): continue
        meta = json.load(open(meta_p, encoding='utf-8'))
        res = m.transcribe(f'{HERE}/audio/{ep}.wav', language='fr', fp16=False)
        heard = norm(' '.join(s['text'] for s in res['segments']))
        ref = norm(' '.join(t['text'] for t in meta['turns']))
        ratio = difflib.SequenceMatcher(None, ref, heard, autojunk=False).ratio()
        print(f'{ep}: ASR whole-episode word similarity {ratio:.2f}')
        if ratio < 0.5: fails.append(f'{ep}: ASR similarity {ratio:.2f} < 0.5 (audio likely not matching script)')
        elif ratio < 0.85: warns.append(f'{ep}: ASR similarity {ratio:.2f} < 0.85 (non-critical)')
        json.dump(dict(similarity=round(ratio, 3), transcript=res['text']),
                  open(f'{HERE}/audio/{ep}.asr.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

for w in warns: print('WARN', w)
for f in fails: print('FAIL', f)
print('RESULT', 'FAIL' if fails else 'PASS')
sys.exit(1 if fails else 0)
