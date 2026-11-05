import argparse, os, time
import soundfile as sf
from melo.api import TTS

TEXT = ("Bonjour, je suis un ami des enfants. Les enfants ont acheté des petits oiseaux, "
        "et nous allons en Amérique cet été. Un grand homme est arrivé tout à coup.")

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")  # cpu | mps | auto
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "voice_samples", "melotts"))
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
model = TTS(language="FR", device=a.device)
spk = model.hps.data.spk2id["FR"]
path = os.path.join(a.out, "fr.wav")

t0 = time.time()
model.tts_to_file(TEXT, spk, path, speed=a.speed)
gen = time.time() - t0
info = sf.info(path)
print(f"fr  audio_s={info.duration:.2f}  gen_s={gen:.2f}  RTF={gen / info.duration:.2f}  -> {path}")
