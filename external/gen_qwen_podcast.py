"""Render a multi-voice podcast episode with Qwen3-TTS CustomVoice.

Run inside external/Qwen3-TTS/.venv:
    python ../gen_qwen_podcast.py ../podcast/ep1.json [--device mps|cpu] [--out DIR]

Episode file: {"id", "title", "language", "cast": {Name: speaker_id}, "turns": [[Name, text], ...]}
Output (in --out, default podcast/audio): <id>.wav, <id>.mp3, <id>.meta.json (per-turn timings).
Per-turn clips are cached in <out>/.clips/<id>/ so an interrupted run resumes.
"""
import os, sys, json, time, argparse, subprocess
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))
PAUSE_S = 0.45

p = argparse.ArgumentParser()
p.add_argument('episode')
p.add_argument('--model', default='1.7B')
p.add_argument('--device', default='mps')
p.add_argument('--out', default=f'{HERE}/../podcast/audio')
args = p.parse_args()

ep = json.load(open(args.episode, encoding='utf-8'))
os.makedirs(args.out, exist_ok=True)
clip_dir = f'{args.out}/.clips/{ep["id"]}'
os.makedirs(clip_dir, exist_ok=True)

dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
model = Qwen3TTSModel.from_pretrained(
    f'{HERE}/Qwen3-TTS/checkpoints/Qwen3-TTS-12Hz-{args.model}-CustomVoice',
    device_map=args.device, dtype=dtype, attn_implementation='sdpa')
supported = set(model.get_supported_speakers())
for name, spk in ep['cast'].items():
    assert spk in supported, f'{spk} not a supported speaker: {sorted(supported)}'

meta, chunks, sr_out, t_cursor = [], [], None, 0.0
for i, (name, text) in enumerate(ep['turns']):
    path = f'{clip_dir}/{i:03d}.wav'
    gen = 0.0
    if not os.path.exists(path):
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(text=text, language=ep['language'], speaker=ep['cast'][name])
        gen = time.time() - t0
        sf.write(path, wavs[0], sr)
    wav, sr = sf.read(path, dtype='float32')
    sr_out = sr
    dur = len(wav) / sr
    meta.append(dict(i=i, speaker=name, voice=ep['cast'][name], text=text,
                     start=round(t_cursor, 2), duration=round(dur, 2), gen_s=round(gen, 1)))
    chunks += [wav, np.zeros(int(PAUSE_S * sr), dtype='float32')]
    t_cursor += dur + PAUSE_S
    print(f'[{ep["id"]}] {i + 1}/{len(ep["turns"])} {name:<7} {dur:5.1f}s (gen {gen:5.1f}s)', flush=True)

full = np.concatenate(chunks)
wav_path, mp3_path = f'{args.out}/{ep["id"]}.wav', f'{args.out}/{ep["id"]}.mp3'
sf.write(wav_path, full, sr_out)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav_path, '-codec:a', 'libmp3lame',
                '-b:a', '96k', '-ac', '1', mp3_path], check=True)
json.dump(dict(id=ep['id'], title=ep['title'], sample_rate=sr_out, total_s=round(len(full) / sr_out, 1),
               turns=meta), open(f'{args.out}/{ep["id"]}.meta.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('done', mp3_path, f'{len(full) / sr_out:.0f}s')
