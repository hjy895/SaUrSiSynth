# SaUrSiSynth — check tables (from real run)

- Manifest rows: **8044**
- Triplets: **2687**
- Domain counts: `{'Gen': 897, 'Cul': 895, 'HLT': 895}`
- Split counts: `{'Train': 2149, 'Test': 269, 'Dev': 269}`

## Table A — Hours by language

| Language | # Files | Hours | Mean (s) |
|---|---:|---:|---:|
| Saraiki | 2683 | 2.31 | 3.10 |
| Urdu | 2686 | 2.30 | 3.08 |
| Sindhi | 2675 | 2.30 | 3.09 |
| **Total** | **8044** | **6.90** | **3.09** |

## Table B — Language × domain

| Lang | Dom | Files | Hours | Dur Max | Dur Min | Dur Avg | Tokens | Tok Max | Tok Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Saraiki | Gen | 897 | 0.71 | 12.17 | 1.17 | 2.84 | 8199 | 32 | 9.1 |
| Saraiki | Cul | 894 | 0.72 | 11.77 | 1.22 | 2.91 | 8308 | 35 | 9.3 |
| Saraiki | HLT | 892 | 0.88 | 17.28 | 1.07 | 3.53 | 9994 | 47 | 11.2 |
| Urdu | Gen | 897 | 0.70 | 12.67 | 1.22 | 2.81 | 8199 | 32 | 9.1 |
| Urdu | Cul | 894 | 0.72 | 12.62 | 1.21 | 2.89 | 8308 | 35 | 9.3 |
| Urdu | HLT | 895 | 0.88 | 17.28 | 0.31 | 3.53 | 10009 | 47 | 11.2 |
| Sindhi | Gen | 895 | 0.71 | 10.41 | 0.44 | 2.84 | 7924 | 35 | 8.9 |
| Sindhi | Cul | 894 | 0.72 | 12.12 | 0.39 | 2.92 | 8207 | 127 | 9.2 |
| Sindhi | HLT | 886 | 0.87 | 22.98 | 0.45 | 3.52 | 9636 | 64 | 10.9 |

## Table C — Train / Dev / Test (hours / #files)

| Language | Gen Train | Gen Dev | Gen Test | Cul Train | Cul Dev | Cul Test | HLT Train | HLT Dev | HLT Test |
|---|---|---|---|---|---|---|---|---|---|
| Saraiki | 0.56/714 | 0.08/94 | 0.07/89 | 0.58/710 | 0.07/93 | 0.07/91 | 0.70/721 | 0.08/82 | 0.10/89 |
| Urdu | 0.55/714 | 0.08/94 | 0.07/89 | 0.58/710 | 0.07/93 | 0.07/91 | 0.70/724 | 0.08/82 | 0.10/89 |
| Sindhi | 0.56/712 | 0.08/94 | 0.07/89 | 0.58/710 | 0.07/93 | 0.07/91 | 0.69/717 | 0.08/80 | 0.10/89 |

## Table D — Cascade ST (Whisper-small → NLLB-600M)

| Pair | BLEU | COMET | ChrF | WER | n |
|---|---:|---:|---:|---:|---:|
| Sa-Ur | 32.27 | None | 59.69 | 0.248 | 50 |
| Sa-Si | 33.15 | None | 52.89 | 0.248 | 50 |
| Ur-Si | 32.61 | None | 52.08 | 0.223 | 50 |
