# All tables filled from THIS experiment run

Structure matches `main (3).tex`. **Bold left = our measured data.**
FLEURS ST columns in Table 4 are **paper draft** (FLEURS cascade not re-run here).

- Triplets: 2687
- Audio files: 8044 (17 TTS fails vs target 8061)
- Total hours: 6.90

## Table 1 — Public corpora compared with SaUrSiSynth

| Dataset | Domain | Nature | #L | Hours | Trio | #Uttr |
|---|---|---|---:|---:|---|---:|
| Gupta et al. | TED | Synth. | 2 | 116 | × | 121K |
| Mondal et al. | Book | Synth. | 2 | 2 | × | — |
| FLEURS | Wiki | Read | 102 | 1.4K | part. | 2.7K |
| SpeechMatrix | EP | Spont. | 17 | 418K | × | — |
| CVSS | Open | Synth. | 21 | 719 | × | — |
| VoxPopuli | EP | Spont. | 15 | 17.3K | × | — |
| Common Voice | Open | Read | 100+ | >10K | × | — |
| **SaUrSiSynth (this run)** | Gen/Cul/HLT | Synth. | **3** | **6.90** | **✓** | **8.0K** |
| SaUrSiSynth (paper draft) | Gen/Cul/HLT | Synth. | 3 | 19.27 | ✓ | 8.1K |

## Table 2 — SaUrSiSynth vs FLEURS: hours and sentence counts

| Languages | SaUrSiSynth Hours | SaUrSiSynth Sentences | FLEURS Hours | FLEURS Sentences |
|---|---:|---:|---:|---:|
| **Saraiki** | **2.31** | **2683** | 1.20 | 420 |
| **Urdu** | **2.30** | **2686** | 4.40 | 1520 |
| **Sindhi** | **2.30** | **2675** | 3.85 | 1380 |

Cross-check (paper draft hours): Saraiki 6.10 · Urdu 6.89 · Sindhi 6.28

## Table 3 — Speech hours by domain for each language

| Domain | #Trip. | Sa h | Ur h | Si h | Total h |
|---|---:|---:|---:|---:|---:|
| General | 897 | 0.71 | 0.70 | 0.71 | 2.11 |
| Culture | 895 | 0.72 | 0.72 | 0.72 | 2.17 |
| Healthcare | 895 | 0.88 | 0.88 | 0.87 | 2.62 |
| **Total** | **2687** | **2.31** | **2.30** | **2.30** | **6.90** |

Paper draft totals: 2687 trips · 6.10/6.89/6.28 · 19.27 h

## Table 4 — Spoken translation: SaUrSiSynth vs FLEURS

| Language Pair | Ours BLEU | Ours COMET | Ours ChrF | Ours WER | FLEURS BLEU | FLEURS COMET | FLEURS ChrF | FLEURS WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sa--Ur | **36.31** | **0.826** | **58.56** | **0.304** | 16.88† | 0.712† | 43.05† | 0.719† |
| Sa--Si | **14.77** | **0.728** | **37.42** | **0.304** | 15.94† | 0.698† | 41.8† | 0.735† |
| Ur--Si | **15.44** | **0.748** | **36.13** | **0.271** | 15.21† | 0.711† | 42.1† | 0.699† |

† FLEURS columns = paper draft (not re-run). Ours = this cascade on SaUrSiSynth test (n≈60/pair, domain-balanced).

### Cross-check vs paper draft (Ours column)

| Pair | Paper BLEU/COMET/ChrF/WER | This run |
|---|---|---|
| Sa-Ur | 18.42 / 0.751 / 46.18 / 0.682 | 36.31 / 0.826 / 58.56 / 0.304 |
| Sa-Si | 17.65 / 0.737 / 45.02 / 0.702 | 14.77 / 0.728 / 37.42 / 0.304 |
| Ur-Si | 16.9 / 0.706 / 44.35 / 0.724 | 15.44 / 0.748 / 36.13 / 0.271 |

## Table 5 — Spoken translation by domain on SaUrSiSynth

| Language Pair | Dom | BLEU | COMET | ChrF | WER | n |
|---|---|---:|---:|---:|---:|---:|
| Sa--Ur | GENERAL | 29.93 | 0.822 | 57.91 | 0.244 | 20 |
| Sa--Ur | CULTURE | 39.01 | 0.829 | 57.02 | 0.303 | 20 |
| Sa--Ur | HEALTHCARE | 38.15 | 0.826 | 60.17 | 0.364 | 20 |
| Sa--Si | GENERAL | 25.93 | 0.781 | 47.31 | 0.244 | 20 |
| Sa--Si | CULTURE | 6.09 | 0.718 | 32.17 | 0.303 | 20 |
| Sa--Si | HEALTHCARE | 10.69 | 0.686 | 33.97 | 0.364 | 20 |
| Ur--Si | GENERAL | 28.92 | 0.798 | 45.94 | 0.212 | 20 |
| Ur--Si | CULTURE | 6.68 | 0.749 | 32.77 | 0.269 | 20 |
| Ur--Si | HEALTHCARE | 9.24 | 0.697 | 31.23 | 0.332 | 20 |

## Table 6 — Language and domain statistics

| Lang. | Dom. | Files | Hours | Dur Max | Dur Min | Dur Avg | Tok. | Tok Max | Tok Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Saraiki | Gen | 897 | 0.71 | 12.17 | 1.17 | 2.84 | 8,199 | 32 | 9.1 |
| Saraiki | Cul | 894 | 0.72 | 11.77 | 1.22 | 2.91 | 8,308 | 35 | 9.3 |
| Saraiki | HLT | 892 | 0.88 | 17.28 | 1.07 | 3.53 | 9,994 | 47 | 11.2 |
| Urdu | Gen | 897 | 0.70 | 12.67 | 1.22 | 2.81 | 8,199 | 32 | 9.1 |
| Urdu | Cul | 894 | 0.72 | 12.62 | 1.21 | 2.89 | 8,308 | 35 | 9.3 |
| Urdu | HLT | 895 | 0.88 | 17.28 | 0.31 | 3.53 | 10,009 | 47 | 11.2 |
| Sindhi | Gen | 895 | 0.71 | 10.41 | 0.44 | 2.84 | 7,924 | 35 | 8.9 |
| Sindhi | Cul | 894 | 0.72 | 12.12 | 0.39 | 2.92 | 8,207 | 127 | 9.2 |
| Sindhi | HLT | 886 | 0.87 | 22.98 | 0.45 | 3.52 | 9,636 | 64 | 10.9 |
| **Total** | — | **8,044** | **6.90** | **22.98** | **0.31** | **3.09** | **78,784** | **127** | **9.8** |

Paper draft: 8,061 files · 19.27 h · mean ~8.6 s

## Table 7 — Train, development, and test splits

| Language | Gen Train (h/#) | Gen Dev/Test (h/#) | Cul Train (h/#) | Cul Dev/Test (h/#) | HLT Train (h/#) | HLT Dev/Test (h/#) |
|---|---|---|---|---|---|---|
| **Saraiki** | 0.56/714 | 0.15/183 | 0.58/710 | 0.14/184 | 0.70/721 | 0.17/171 |
| **Urdu** | 0.55/714 | 0.15/183 | 0.58/710 | 0.14/184 | 0.70/724 | 0.18/171 |
| **Sindhi** | 0.56/712 | 0.15/183 | 0.58/710 | 0.14/184 | 0.69/717 | 0.17/169 |

Paper draft used fixed 50 Dev + 50 Test per lang×domain; this run used ~10%/10% global split (~80–94 Dev/Test per cell).

## What was missing / notes

1. **COMET** — now installed (`unbabel-comet`) and included in Tables 4–5.
2. **FLEURS cascade re-run** — not done (needs FLEURS audio + matched fine-tune). Draft FLEURS scores kept for cross-check only.
3. **Hours gap** — paper draft 19.27 h vs this run ~6.90 h (OmniVoice clips shorter, ~3 s mean).
4. **Saraiki text** — Urdu orthography carrier + OmniVoice `language=saraiki` (no NLLB Saraiki code).
5. **Fine-tuning** — scores are cascade ASR→MT on synthetic test audio (not Whisper/NLLB fine-tuned as in paper narrative).