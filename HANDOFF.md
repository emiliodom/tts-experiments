# Handoff: French TTS voice evaluation (CosyVoice2-EU, Qwen3-TTS, MeloTTS, Fish-Speech)

Date: 2026-11-04. Machine used: MacBook Pro, Apple Silicon (arm64), macOS 15.7, 16 GB RAM, PyTorch MPS available, no NVIDIA GPU.

## 1. What exists

| Component | Location | Venv | Status |
|---|---|---|---|
| CosyVoice2-EU (French fine-tune) | `cosy_repo/` + `cosy_repo/pretrained_models/CosyVoice2-0.5B-EU` (4.8 GB) | `cosy_repo/.venv` (Python 3.10) | Working, CPU only |
| Qwen3-TTS 1.7B CustomVoice | `external/Qwen3-TTS/` + `checkpoints/` (4.2 GB model + 657 MB tokenizer) | `external/Qwen3-TTS/.venv` (Python 3.12) | Working on MPS |
| MeloTTS (French) | `external/MeloTTS/` | `external/MeloTTS/.venv` (Python 3.10) | Working on CPU |
| Fish-Speech S2-Pro | `external/fish-speech/` + `checkpoints/s2-pro` (10 GB) | `external/fish-speech/.venv` (Python 3.12) | Installed, **does not run on 16 GB** (swap exhausted, process killed) |

Each model has its own venv because their dependency pins conflict (transformers 4.40 vs 4.57, torch 2.3 vs 2.8+, etc.). Never install them into one environment.

Skipped by decision: fine-tuning of any kind (CosyVoice2 LLM / Flow / HiFi-GAN fine-tuning is CUDA/NCCL/DeepSpeed only; Qwen fine-tuning; Fish "Drama 3" needs an NVIDIA 48 GB GPU).

## 2. Rendering the Qwen3-TTS voices (how-to)

### 2.1 Setup from scratch

```bash
cd external
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS.git && cd Qwen3-TTS
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e . huggingface_hub
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir checkpoints/Qwen3-TTS-12Hz-1.7B-CustomVoice
hf download Qwen/Qwen3-TTS-Tokenizer-12Hz        --local-dir checkpoints/Qwen3-TTS-Tokenizer-12Hz
```

Installed versions: `qwen-tts` 0.1.1 (editable), `transformers` 4.57.3 (pinned by the package), `accelerate` 1.12.0, torch 2.x with MPS.
Licence: Apache-2.0 (code and model card).

### 2.2 Render one French sample per voice

```bash
cd external/Qwen3-TTS && source .venv/bin/activate
python ../gen_qwen_french.py                       # all 9 voices -> voice_samples/qwen3/<speaker>.wav
python ../gen_qwen_french.py --speakers vivian,ryan
python ../gen_qwen_french.py --device cpu          # slower, float32
```

Options: `--model` (default `1.7B`), `--device` (`mps` default, or `cpu`), `--out`, `--speakers`.
The sentence used (same for every model so results are comparable):

> Bonjour, je suis un ami des enfants. Les enfants ont acheté des petits oiseaux, et nous allons en Amérique cet été. Un grand homme est arrivé tout à coup.

### 2.3 Render the expressions (laugh, yawn, sigh, ...)

```bash
cd external/Qwen3-TTS && source .venv/bin/activate
python ../gen_qwen_expressions.py                                  # 9 voices x 10 expressions = 90 clips (~31 min of generation)
python ../gen_qwen_expressions.py --speakers vivian --expressions laugh,yawn
python ../build_expressions_page.py                                # rebuilds voice_samples/expressions.html (run with any python3)
```

Output: `voice_samples/qwen3_expressions/<speaker>/<expression>.wav` and `meta.json` (text, instruction, duration, generation time per clip; the script merges into an existing `meta.json`, so partial re-renders are safe).

Expressions defined in `EXPRESSIONS` in [external/gen_qwen_expressions.py](external/gen_qwen_expressions.py): `laugh`, `yawn`, `sigh`, `groan`, `cry`, `gasp`, `whisper`, `cough`, `angry`, `thinking`. To add one, add an entry `name: (french_text, english_style_instruction)` and re-run with `--expressions name`.

### 2.4 Minimal API usage

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "external/Qwen3-TTS/checkpoints/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="mps", dtype=torch.bfloat16, attn_implementation="sdpa")   # use float32 on cpu
print(model.get_supported_speakers())
wavs, sr = model.generate_custom_voice(
    text="Bonjour tout le monde.", language="French", speaker="vivian",
    instruct="Whispering very softly.")        # instruct is optional, 1.7B only
sf.write("out.wav", wavs[0], sr)
```

### 2.5 Technical notes on Qwen3-TTS

- Speakers (lowercase ids): `aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian`. They are preset voices described by Qwen as Chinese/English/Japanese/Korean speakers rendered in French through the multilingual model, so some foreign accent in French is likely (not verified by listening).
- Languages supported by the model: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian. Pass `language="French"`.
- `instruct` is a free-text style instruction (English or Chinese). It is **not supported on the 0.6B model**.
- **There are no official non-verbal tags** (no `[laugh]`). Expressions are approximated with onomatopoeia in the text plus an `instruct`. Quality varies per voice and expression and was not audited by listening; use the rating grid.
- Model family also includes Voice Design (describe a voice in text) and Voice Clone (clone from a reference clip). These were **not downloaded or tested**.
- Generation is autoregressive over 12 Hz codec tokens; `pad_token_id` warnings in the log are harmless.
- Output sample rate comes back from `generate_custom_voice` (`sr`); the assessment page plays the files as they are.

## 3. Results (same sentence, 16 GB MacBook)

`Gen (s)` = wall-clock generation time of that clip, model load excluded. RTF = Gen / audio duration (< 1 is faster than real time).

| Model | Voices | Device | RTF | Notes |
|---|---|---|---|---|
| MeloTTS FR | 1 (`FR`) | CPU | 0.62 (warm) | first run 7.3 because it downloads a French BERT |
| Qwen3-TTS 1.7B | 9 | MPS (bf16) | ~2.6 to 3.2 (expressions: avg ~2.5) | |
| CosyVoice2-EU | 1 (sample prompt voice) | CPU | ~8.9 to 11.6 | `torch 2.3.1`, no GPU backend used |
| Fish-Speech S2-Pro | n/a | n/a | n/a | OOM / swap thrash at ~10 GB weights |

Each number comes from a single run (no repeats).

Per-clip numbers for all Qwen renders are in `voice_samples/qwen3_expressions/meta.json`; the main sentence timings are in `voice_samples/index.html`.

## 4. Assessment pages

- `voice_samples/index.html`: one row per voice for the shared sentence (MeloTTS, CosyVoice2-EU, 9 Qwen voices), players, duration / gen time / RTF, ratings (naturalness, liaisons, accent), notes, JSON export. A second table has the 4 extra CosyVoice2-EU phrases.
- `voice_samples/expressions.html`: grid of 10 expressions x 9 Qwen voices with a player and 1-5 rating per cell, averages, JSON export.

Open them directly in a browser (`open voice_samples/index.html`). Ratings are stored in the browser's `localStorage`, not in the repo. The timing numbers in `index.html` are hard-coded in its `VOICES` array; update them if you re-render.

## 5. Other models: how to run

### CosyVoice2-EU (CPU)

```bash
cd cosy_repo && source .venv/bin/activate
python bench_french.py                              # 5 phrases, prints RTF table, writes audio_output/bench/final_<phrase>.wav
python bench_french.py --phrases shared             # only the shared sentence
python bench_french.py --prompt prompt_audio/common_voice_fr_40952142.wav
python webui_french.py                              # web UI (edited but never launched or tested)
```

- Loader semantics: `final=True` loads `llm.pt`, `flow.pt`, `hift.pt`; otherwise `{llm,flow,hift}-{run_id|original}.pt` per `setting`. `setting='all'` is invalid and raises in the loader.
- `bench_french.py --prompt` defaults to `sample_audio_prompt.wav` in the model dir. `prompt_audio/macron.wav` referenced by `run_inference.py` does not exist.
- Fixes that were required (keep them): `ruamel.yaml<0.18` (HyperPyYAML `max_depth` error); `pynini` 2.1.7 needs `brew install openfst` plus `CPPFLAGS/LDFLAGS` and `--no-build-isolation`; `WeTextProcessing` installed with `--no-deps` plus `importlib_resources`; `openai-whisper` needs `uv pip install "setuptools<81" wheel` and `--no-build-isolation`. `ttsfrd` is not available, WeTextProcessing is used instead (log line "failed to import ttsfrd" is expected).

### MeloTTS (CPU)

```bash
cd external/MeloTTS && source .venv/bin/activate
python ../gen_melo_french.py --device cpu           # -> voice_samples/melotts/fr.wav
```

Setup: `git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git`, `uv venv --python 3.10 .venv`, `uv pip install -e . "setuptools<81" wheel`, `python -m unidic download` (526 MB). MIT licence. Only one French speaker (`FR`).

### Fish-Speech S2-Pro

Installed in `external/fish-speech` with weights in `checkpoints/s2-pro` (documented CLI: `python fish_speech/models/text2semantic/inference.py --checkpoint-path checkpoints/s2-pro --text ... [--prompt-audio ref.wav --prompt-text "..."] --output out.wav`). It needs 24 GB or more RAM or an NVIDIA GPU; on this 16 GB machine `--device mps` made swap hit 18.9 GB and the process was killed. Inline tags such as `[whisper]` and `[excited]` are documented but were never rendered. Licence: Fish Audio Research License, **non-commercial only**.
Reference transcript obtained for cloning (Whisper `base`): "Ces décisions sont prises en assemblée générale réunie sur convocation du syndic de copropriété." (`cosy_repo/prompt_audio/common_voice_fr_40952142.wav`, 10.2 s, 32 kHz).

## 6. Licences summary

| Component | Licence | Commercial use |
|---|---|---|
| Qwen3-TTS | Apache-2.0 | yes |
| MeloTTS | MIT | yes |
| CosyVoice2 code | Apache-2.0 | yes (check the EU model card for weights) |
| Fish-Speech S2-Pro | Fish Audio Research License | **no** |

## 7. Repo / upload notes

- Git state at handoff: `cosy_repo/webui_french.py` modified; untracked: `cosy_repo/bench_french.py`, `external/`, `voice_samples/`.
- Sizes on disk: `external/fish-speech` 11 GB, `external/Qwen3-TTS` 6.1 GB, `external/MeloTTS` 2.7 GB, `cosy_repo/pretrained_models` 4.8 GB, `voice_samples` 42 MB. **Do not commit** `external/*/checkpoints/`, `external/*/.venv/`, or cloned third-party repos (they are separate git repos; clone them via the commands above or add them as submodules). `.gitignore` currently ignores `.venv/` and `cosy_repo/pretrained_models/` but not `external/**/checkpoints/` or the cloned repos.
- Suggested to commit: `external/gen_*.py`, `external/build_expressions_page.py`, `cosy_repo/bench_french.py`, `cosy_repo/webui_french.py`, `voice_samples/index.html`, `voice_samples/expressions.html`, this file, and optionally the WAVs (42 MB total).
- Free disk after all installs was about 8 GB.

## 8. Open items

1. Listen to and rate the renders (the author of this handoff could not listen). Use the two assessment pages and export the JSON.
2. Fish-Speech S2-Pro: needs a machine with 24 GB or more RAM or an NVIDIA GPU.
3. `webui_french.py` was edited (uses `final=True`) but not launched.
4. Qwen Voice Design and Voice Clone models were not tested; they would allow French-native voices instead of the accented presets.
5. Fine-tuning (CosyVoice2 LLM / Flow / HiFi-GAN) requires CUDA hardware and was intentionally skipped.
6. CosyVoice2-EU timings are CPU-only; a CUDA machine would be much faster.
