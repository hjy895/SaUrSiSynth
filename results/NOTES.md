# Experiment notes

- Pipeline: `scripts/run_all.py --n 2687 --skip-mine`
- Bitext: OPUS Tatoeba + OpenSubtitles EN–UR; NLLB-200-distilled-600M EN→Sindhi
- Saraiki text carrier: Urdu orthography (NLLB has no Saraiki code); OmniVoice TTS uses `language=saraiki`
- Audio: OmniVoice `k2-fsa/OmniVoice`, instruct `male, young adult, indian accent`, released at 16 kHz
- Eval: Whisper-small → NLLB-600M; metrics BLEU / COMET / ChrF / WER

## Separate release trees
| Tree | Path | Contents |
|------|------|----------|
| Hugging Face | `../HF_saursisynth/` | WAVs + manifest + dataset card |
| GitHub | `./` | Scripts + `data/triplets.tsv` + `results/*.json` |
| Working corpus | `../experiments_saursisynth/corpus/` | Full build workspace |
