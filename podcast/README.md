# « Sous tension » – French B2 podcast about electricity

Three episodes, ~3-4 min each, rendered with Qwen3-TTS 1.7B CustomVoice (Apache-2.0) on Apple MPS.

| Episode | Voices | Topics |
|---|---|---|
| ep1 | ryan, dylan, sohee | Life with/without electricity, 700M people without access |
| ep2 | aiden, eric, serena | France (nuclear, Ecowatt, Tempo, Flamanville), prices worldwide |
| ep3 | vivian, ono_anna, ryan | China (Three Gorges, UHV lines, solar), Dubai / Abu Dhabi |

Scripts: `ep*.json`. Audio: `audio/ep*.mp3`, timings in `audio/ep*.meta.json`, ASR in `audio/ep*.asr.json`.
Voices are Qwen preset voices speaking French, so some foreign accent is expected. Figures in the scripts are approximate, rounded values.

```bash
cd external/Qwen3-TTS && source .venv/bin/activate
python ../gen_qwen_podcast.py ../../podcast/ep1.json        # -> podcast/audio/ep1.{wav,mp3,meta.json}
cd ../.. && cosy_repo/.venv/bin/python podcast/check_podcast.py --asr   # checks (ASR is non-critical unless similarity < 0.5)
```
