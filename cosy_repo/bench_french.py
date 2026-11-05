"""Time French synthesis (liaisons / native pronunciation phrases) for each component setting.

Usage (from cosy_repo, venv active):
    python bench_french.py --prompt prompt_audio/macron.wav
    python bench_french.py --settings original,llm,flow,hifigan,llm_flow_hifigan
"""
import os
import sys
import time
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

PHRASES = {
    'shared': "Bonjour, je suis un ami des enfants. Les enfants ont acheté des petits oiseaux, et nous allons en Amérique cet été. Un grand homme est arrivé tout à coup.",
    'liaisons': "Les enfants ont acheté des petits oiseaux et ils en ont pris un autre à Paris. Nous allons en Amérique, vous avez eu un ami en été.",
    'enchainements': "Un grand homme est arrivé tout à coup, et il est allé chez elle avec ses amis italiens.",
    'native': "Je voudrais un croissant, une baguette et un verre de vin rouge, s'il vous plaît. Quel beau pays que la France !",
    'tongue_twister': "Les chaussettes de l'archiduchesse sont-elles sèches, archi-sèches ? Un chasseur sachant chasser sait chasser sans son chien.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', default='pretrained_models/CosyVoice2-0.5B-EU')
    p.add_argument('--prompt', default=None, help='>=16 kHz wav; defaults to the sample prompt in the model dir')
    p.add_argument('--settings', default='final', help="comma list: 'final' (llm.pt/flow.pt/hift.pt) or original|llm|flow|hifigan|llm_flow_hifigan (needs *-original.pt / *-<run_id>.pt)")
    p.add_argument('--run-id', default='latest')
    p.add_argument('--out-dir', default='audio_output/bench')
    p.add_argument('--phrases', default=None, help='comma list of phrase names (default: all)')
    args = p.parse_args()

    prompt = args.prompt or os.path.join(args.model_dir, 'sample_audio_prompt.wav')
    prompt_16k = load_wav(prompt, 16000)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'{"setting":<18}{"phrase":<16}{"audio_s":>8}{"gen_s":>8}{"RTF":>7}')
    for setting in args.settings.split(','):
        final = setting == 'final'
        model = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False,
                           setting='original' if final else setting,
                           llm_run_id=args.run_id, flow_run_id=args.run_id, hifigan_run_id=args.run_id,
                           final=final)
        for name, text in PHRASES.items():
            if args.phrases and name not in args.phrases.split(','):
                continue
            t0 = time.time()
            with torch.no_grad():
                wavs = [o['tts_speech'] for o in model.inference_cross_lingual(text, prompt_16k, stream=False, text_frontend=False)]
            gen = time.time() - t0
            audio = torch.cat(wavs, dim=1)
            dur = audio.shape[1] / model.sample_rate
            torchaudio.save(f'{args.out_dir}/{setting}_{name}.wav', audio, model.sample_rate)
            print(f'{setting:<18}{name:<16}{dur:>8.2f}{gen:>8.2f}{gen / dur:>7.2f}')
        del model


if __name__ == '__main__':
    main()
