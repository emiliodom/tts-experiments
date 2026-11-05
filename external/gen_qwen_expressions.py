"""Render non-verbal / expressive French samples (laugh, yawn, sigh, ...) for every Qwen3-TTS CustomVoice speaker.

Qwen3-TTS has no official non-verbal tag set: expressions are approximated with onomatopoeia in the text
plus a natural-language `instruct` (1.7B model only).

Run inside external/Qwen3-TTS/.venv:
    python ../gen_qwen_expressions.py [--speakers aiden,vivian] [--expressions laugh,yawn]
"""
import os
import time
import json
import argparse

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (French text, English style instruction)
EXPRESSIONS = {
    'laugh':   ("Ha ha ha ! Haha ! Oh là là, c'est trop drôle, hahaha ! Je n'en peux plus, hihihi !",
                "Laughing out loud, giggling between words, very amused."),
    'yawn':    ("Aaaah... hmmm... je suis tellement fatigué... aaaah... il est déjà si tard...",
                "Sleepy, yawning heavily, slow and drowsy voice."),
    'sigh':    ("Haaaah... pfff... Encore une longue journée qui se termine... haaah...",
                "Heavy tired sigh, exhausted and resigned, slow."),
    'groan':   ("Ohhh... aïe, aïe, aïe... mmmh... ça fait mal... ohhh... ahhh...",
                "Groaning and moaning in pain, weak and breathy voice."),
    'cry':     ("Snif... je... je n'y arrive plus... snif... pourquoi... pourquoi moi... snif...",
                "Crying and sobbing, voice breaking, trembling."),
    'gasp':    ("Ah ! Oh mon dieu ! Quoi ?! Non ! Je ne m'attendais vraiment pas à ça !",
                "Shocked, gasping in surprise, fast and high-pitched."),
    'whisper': ("Chut... doucement... ne fais pas de bruit... ils dorment encore...",
                "Whispering very softly and close to the microphone."),
    'cough':   ("Hum hum... euh-hum... Excusez-moi, j'ai la gorge un peu sèche... hum.",
                "Clearing throat and coughing between words."),
    'angry':   ("Ça suffit ! J'en ai assez ! Je vous ai déjà dit mille fois de ne pas faire ça !",
                "Very angry, shouting, harsh and loud."),
    'thinking': ("Hmmm... euh... voyons voir... attendez... ah oui ! Hmmm, je crois que j'ai trouvé.",
                 "Hesitant, thinking out loud, with long pauses and filler sounds."),
}

p = argparse.ArgumentParser()
p.add_argument('--model', default='1.7B')
p.add_argument('--device', default='mps')
p.add_argument('--out', default=f'{HERE}/../voice_samples/qwen3_expressions')
p.add_argument('--speakers', default='', help='comma list; default = all supported')
p.add_argument('--expressions', default='', help='comma list; default = all')
args = p.parse_args()

dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
model = Qwen3TTSModel.from_pretrained(
    f'{HERE}/Qwen3-TTS/checkpoints/Qwen3-TTS-12Hz-{args.model}-CustomVoice',
    device_map=args.device, dtype=dtype, attn_implementation='sdpa')

speakers = args.speakers.split(',') if args.speakers else model.get_supported_speakers()
exprs = args.expressions.split(',') if args.expressions else list(EXPRESSIONS)
os.makedirs(args.out, exist_ok=True)

meta_path = f'{args.out}/meta.json'
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

print(f'{"speaker":<10}{"expr":<10}{"audio_s":>8}{"gen_s":>8}{"RTF":>7}', flush=True)
for spk in speakers:
    os.makedirs(f'{args.out}/{spk}', exist_ok=True)
    for name in exprs:
        text, instruct = EXPRESSIONS[name]
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(text=text, language='French', speaker=spk, instruct=instruct)
        gen = time.time() - t0
        dur = len(wavs[0]) / sr
        sf.write(f'{args.out}/{spk}/{name}.wav', wavs[0], sr)
        meta[f'{spk}/{name}'] = {'dur': round(dur, 2), 'gen': round(gen, 2), 'text': text, 'instruct': instruct}
        json.dump(meta, open(meta_path, 'w'), ensure_ascii=False, indent=1)
        print(f'{spk:<10}{name:<10}{dur:>8.2f}{gen:>8.2f}{gen / dur:>7.2f}', flush=True)
