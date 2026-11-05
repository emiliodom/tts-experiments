"""Generate one French sample per Qwen3-TTS CustomVoice speaker.

Run inside external/Qwen3-TTS/.venv:
    python ../gen_qwen_french.py [--model 1.7B] [--device mps|cpu]
"""
import os
import time
import argparse

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = ("Bonjour, je suis un ami des enfants. Les enfants ont acheté des petits oiseaux, "
        "et nous allons en Amérique cet été. Un grand homme est arrivé tout à coup.")

p = argparse.ArgumentParser()
p.add_argument('--model', default='1.7B')
p.add_argument('--device', default='mps')
p.add_argument('--out', default=f'{HERE}/../voice_samples/qwen3')
p.add_argument('--speakers', default='', help='comma list; default = all supported')
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
model = Qwen3TTSModel.from_pretrained(
    f'{HERE}/Qwen3-TTS/checkpoints/Qwen3-TTS-12Hz-{args.model}-CustomVoice',
    device_map=args.device, dtype=dtype, attn_implementation='sdpa')

speakers = args.speakers.split(',') if args.speakers else model.get_supported_speakers()
print('speakers:', speakers)
print(f'{"speaker":<12}{"audio_s":>8}{"gen_s":>8}{"RTF":>7}')
for spk in speakers:
    t0 = time.time()
    wavs, sr = model.generate_custom_voice(text=TEXT, language='French', speaker=spk)
    gen = time.time() - t0
    dur = len(wavs[0]) / sr
    sf.write(f'{args.out}/{spk}.wav', wavs[0], sr)
    print(f'{spk:<12}{dur:>8.2f}{gen:>8.2f}{gen / dur:>7.2f}', flush=True)
