#!/usr/bin/env python3
"""Synthesize Saraiki/Urdu/Sindhi speech with OmniVoice (16 kHz release)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

LANG_MAP = {
    "Sa": ("sa_text", "saraiki"),
    "Ur": ("ur_text", "urdu"),
    "Si": ("si_text", "sindhi"),
}
# OmniVoice voice-design tokens only (comma + space).
INSTRUCT = "male, young adult, indian accent"


def resample_to_16k(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == 16000:
        return wav.astype(np.float32), 16000
    import librosa

    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    return y, 16000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    args = p.parse_args()

    out = Path(args.out_dir)
    for split in ("Train_data", "Dev_data", "Test_data"):
        (out / split).mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(Path(args.triplets).open(encoding="utf-8"), delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Loading OmniVoice on {device} ...")
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype)

    # resolve language name spelling from model list
    raw_names = model.supported_language_names
    name_list = list(raw_names() if callable(raw_names) else raw_names)
    names = {str(n).lower(): str(n) for n in name_list}

    def resolve(lang_key: str) -> str:
        if lang_key.lower() in names:
            return names[lang_key.lower()]
        for k, v in names.items():
            if lang_key.lower() in k:
                return v
        return lang_key

    lang_resolved = {code: resolve(name) for code, (_, name) in LANG_MAP.items()}
    print("language map:", lang_resolved)

    manifest = []
    stats = {"ok": 0, "fail": 0}

    for row in tqdm(rows, desc="TTS"):
        uid = row["utt_id"]
        domain = row["domain"]
        split = row["split"]
        split_dir = {"Train": "Train_data", "Dev": "Dev_data", "Test": "Test_data"}[split]
        for code, (col, _) in LANG_MAP.items():
            text = (row.get(col) or "").strip()
            if not text:
                stats["fail"] += 1
                continue
            wav_name = f"{split}_{code}_{domain}_M_{uid}.wav"
            wav_path = out / split_dir / wav_name
            if wav_path.exists() and wav_path.stat().st_size > 1000:
                dur = len(sf.read(wav_path)[0]) / 16000.0
                manifest.append(
                    {
                        "utt_id": wav_name.replace(".wav", ""),
                        "triplet_id": uid,
                        "split": split,
                        "lang": {"Sa": "Saraiki", "Ur": "Urdu", "Si": "Sindhi"}[code],
                        "lang_code": code,
                        "domain": domain,
                        "gender": "M",
                        "text": text,
                        "wav_path": f"{split_dir}/{wav_name}",
                        "duration_sec": round(dur, 3),
                    }
                )
                stats["ok"] += 1
                continue
            try:
                audio = model.generate(
                    text=text,
                    language=lang_resolved[code],
                    instruct=INSTRUCT,
                    normalize_text=True,
                )
                wav = np.asarray(audio[0], dtype=np.float32)
                # OmniVoice default 24 kHz
                wav, sr = resample_to_16k(wav, 24000)
                peak = float(np.max(np.abs(wav)) + 1e-9)
                wav = wav / peak * 0.95
                sf.write(str(wav_path), wav, sr)
                manifest.append(
                    {
                        "utt_id": wav_name.replace(".wav", ""),
                        "triplet_id": uid,
                        "split": split,
                        "lang": {"Sa": "Saraiki", "Ur": "Urdu", "Si": "Sindhi"}[code],
                        "lang_code": code,
                        "domain": domain,
                        "gender": "M",
                        "text": text,
                        "wav_path": f"{split_dir}/{wav_name}",
                        "duration_sec": round(len(wav) / sr, 3),
                    }
                )
                stats["ok"] += 1
            except Exception as e:  # noqa: BLE001
                stats["fail"] += 1
                print(f"FAIL {wav_name}: {e}")

    man_path = out / "manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    (out / "synthesis_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("wrote", man_path, stats)


if __name__ == "__main__":
    main()
