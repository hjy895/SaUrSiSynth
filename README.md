# SaUrSiSynth

Synthetic tri-parallel speech corpus for **Saraiki–Urdu–Sindhi** spoken language translation.

## Links
- Dataset (Hugging Face): https://huggingface.co/datasets/saursisynth/
- This repo: scripts + experiment results (no bulk WAVs)

## Quickstart

```bash
pip install -r requirements.txt
pip install omnivoice sentence-transformers

# Full pipeline (bitext -> OmniVoice TTS -> Whisper/NLLB eval -> package HF+GitHub)
python scripts/run_all.py --n 2687

# Faster pilot
python scripts/run_all.py --n 150 --tts-limit 150 --eval-max-per-pair 30
```

Working corpus is written under `../experiments_saursisynth/corpus/`.
Hugging Face audio layout is copied to `../HF_saursisynth/`.
Results JSON lands in `results/`.

## Scripts
| Script | Role |
|--------|------|
| `scripts/build_bitext.py` | Tatoeba EN–UR + NLLB Sindhi + Saraiki mining |
| `scripts/synthesize.py` | OmniVoice TTS @ 16 kHz |
| `scripts/evaluate.py` | Whisper-small → NLLB-600M (BLEU/COMET/ChrF/WER) |
| `scripts/run_all.py` | End-to-end + separate HF/GitHub packaging |

## License
Apache-2.0 (code). Dataset license on the Hugging Face card.
