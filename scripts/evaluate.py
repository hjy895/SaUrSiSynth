#!/usr/bin/env python3
"""Evaluate Whisper-small -> NLLB-600M cascade (BLEU, COMET, ChrF, WER)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PAIR_LANG = {
    "Sa-Ur": ("Saraiki", "Urdu", "urd_Arab"),
    "Sa-Si": ("Saraiki", "Sindhi", "snd_Arab"),
    "Ur-Si": ("Urdu", "Sindhi", "snd_Arab"),
}

# Whisper language hints (best-effort; Saraiki falls back to Urdu)
WHISPER_LANG = {"Saraiki": "ur", "Urdu": "ur", "Sindhi": "ur"}


def load_manifest(path: Path, split: str = "Test"):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split:
                rows.append(r)
    return rows


def wer(ref: str, hyp: str) -> float:
    import jiwer

    return float(jiwer.wer(ref, hyp))


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


def whisper_transcribe(model, processor, audio: np.ndarray, language: str, device: str) -> str:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=model.dtype)
    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.no_grad():
        pred_ids = model.generate(input_features, forced_decoder_ids=forced, max_new_tokens=128)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Corpus root with manifest.jsonl")
    p.add_argument("--out", required=True, help="JSON results path")
    p.add_argument("--whisper", default="openai/whisper-small")
    p.add_argument("--nllb", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--max-per-pair", type=int, default=50)
    p.add_argument("--skip-comet", action="store_true")
    args = p.parse_args()

    root = Path(args.data_root)
    manifest = load_manifest(root / "manifest.jsonl", "Test")
    by_trip = defaultdict(dict)
    for r in manifest:
        by_trip[r["triplet_id"]][r["lang"]] = r

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Whisper ...")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.whisper)
    dtype = torch.float16 if device == "cuda" else torch.float32
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.whisper, dtype=dtype).to(device)
    asr_model.eval()

    print("Loading NLLB ...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.nllb)
    mt = AutoModelForSeq2SeqLM.from_pretrained(args.nllb).to(device)

    import sacrebleu

    comet_model = None
    if not args.skip_comet:
        try:
            from comet import download_model, load_from_checkpoint

            comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        except Exception as e:  # noqa: BLE001
            print("COMET unavailable:", e)

    results = {}
    domain_results = defaultdict(lambda: defaultdict(list))

    for pair, (src_lang, tgt_lang, tgt_code) in PAIR_LANG.items():
        hyps, refs, wers = [], [], []
        comet_data = []
        used = 0
        for tid, langs in by_trip.items():
            if src_lang not in langs or tgt_lang not in langs:
                continue
            if used >= args.max_per_pair:
                break
            src = langs[src_lang]
            tgt = langs[tgt_lang]
            wav = root / src["wav_path"]
            if not wav.exists():
                continue
            try:
                audio, _sr = load_wav(wav)
                hyp_asr = whisper_transcribe(
                    asr_model, processor, audio, WHISPER_LANG[src_lang], device
                )
            except Exception as e:  # noqa: BLE001
                print(f"ASR fail {wav.name}: {e}")
                hyp_asr = ""
            if not hyp_asr:
                continue
            tok.src_lang = "urd_Arab" if src_lang in ("Saraiki", "Urdu") else "snd_Arab"
            enc = tok(hyp_asr, return_tensors="pt", truncation=True, max_length=128).to(device)
            out_ids = mt.generate(
                **enc,
                forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code),
                max_new_tokens=128,
                num_beams=4,
            )
            hyp = tok.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            ref = tgt["text"]
            hyps.append(hyp)
            refs.append(ref)
            try:
                wers.append(wer(src["text"], hyp_asr))
            except Exception:
                wers.append(1.0)
            comet_data.append({"src": hyp_asr, "mt": hyp, "ref": ref})
            domain_results[pair][src["domain"]].append((hyp, ref, wers[-1]))
            used += 1
            if used % 10 == 0:
                print(f"  {pair} {used}/{args.max_per_pair}", flush=True)

        if not hyps:
            results[pair] = {"BLEU": 0, "COMET": None, "ChrF": 0, "WER": 1.0, "n": 0}
            continue
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        comet_score = None
        if comet_model is not None:
            try:
                comet_score = float(
                    comet_model.predict(comet_data, batch_size=8, gpus=1 if device == "cuda" else 0)["system_score"]
                )
            except Exception as e:  # noqa: BLE001
                print("COMET failed:", e)
        results[pair] = {
            "BLEU": round(bleu, 2),
            "COMET": round(comet_score, 3) if comet_score is not None else None,
            "ChrF": round(chrf, 2),
            "WER": round(float(np.mean(wers)), 3),
            "n": len(hyps),
        }
        print(pair, results[pair], flush=True)

    dom_out = {}
    for pair, doms in domain_results.items():
        dom_out[pair] = {}
        for dom, items in doms.items():
            if not items:
                continue
            hyps = [h for h, _, _ in items]
            refs = [r for _, r, _ in items]
            wers = [w for _, _, w in items]
            dom_out[pair][dom] = {
                "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
                "ChrF": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
                "WER": round(float(np.mean(wers)), 3),
                "n": len(items),
            }

    payload = {"pairs": results, "domains": dom_out, "whisper": args.whisper, "nllb": args.nllb}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # also mirror to GitHub/HF
    for dest in (
        Path(r"D:\ICASSP MT 2027\GitHub_SaUrSiSynth\results\cascade_eval_results.json"),
        Path(r"D:\ICASSP MT 2027\HF_saursisynth\cascade_eval_results.json"),
    ):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
