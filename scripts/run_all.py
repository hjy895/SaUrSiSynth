#!/usr/bin/env python3
"""Run full SaUrSiSynth pipeline and package HF + GitHub releases separately."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SCRIPTS = ROOT / "scripts"
EXP = WORKSPACE / "experiments_saursisynth"
HF_DST = WORKSPACE / "HF_saursisynth"
GH_DST = ROOT


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def package_hf(data_root: Path, results: Path | None) -> None:
    """Hugging Face dataset layout only (audio + manifests + card)."""
    HF_DST.mkdir(parents=True, exist_ok=True)
    for split in ("Train_data", "Dev_data", "Test_data"):
        src = data_root / split
        dst = HF_DST / split
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)
    # parallel text
    pt = HF_DST / "parallel_text"
    pt.mkdir(exist_ok=True)
    trip = data_root / "triplets.tsv"
    if trip.exists():
        shutil.copy2(trip, pt / "triplets.tsv")
    man = data_root / "manifest.jsonl"
    if man.exists():
        shutil.copy2(man, HF_DST / "manifest.jsonl")
        # also sample
        lines = man.read_text(encoding="utf-8").splitlines()[:20]
        (pt / "sample_manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if results and results.exists():
        shutil.copy2(results, HF_DST / "cascade_eval_results.json")
    # update card note
    readme = HF_DST / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "Culture" not in text:
            text = text.replace("Gen · Gov · HLT", "Gen · Cul · HLT")
            readme.write_text(text, encoding="utf-8")
    meta = {
        "dataset": "SaUrSiSynth",
        "layout": ["Train_data", "Dev_data", "Test_data", "parallel_text", "manifest.jsonl"],
        "sample_rate_hz": 16000,
        "languages": ["Saraiki", "Urdu", "Sindhi"],
        "domains": ["Gen", "Cul", "HLT"],
    }
    (HF_DST / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("HF package ->", HF_DST)


def package_github(data_root: Path, results: Path | None) -> None:
    """GitHub code + scripts + results (no bulk wavs by default)."""
    res_dir = GH_DST / "results"
    res_dir.mkdir(exist_ok=True)
    if results and results.exists():
        shutil.copy2(results, res_dir / "cascade_eval_results.json")
    # copy triplets for reproducibility (text only)
    pt = GH_DST / "data"
    pt.mkdir(exist_ok=True)
    trip = data_root / "triplets.tsv"
    if trip.exists():
        shutil.copy2(trip, pt / "triplets.tsv")
    stats = data_root / "synthesis_stats.json"
    if stats.exists():
        shutil.copy2(stats, res_dir / "synthesis_stats.json")
    # corpus stats
    man = data_root / "manifest.jsonl"
    if man.exists():
        n = sum(1 for _ in man.open(encoding="utf-8"))
        (res_dir / "corpus_file_count.json").write_text(
            json.dumps({"manifest_rows": n}, indent=2), encoding="utf-8"
        )
    print("GitHub package ->", GH_DST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2687)
    ap.add_argument("--tts-limit", type=int, default=0, help="Limit triplets for TTS (0=all)")
    ap.add_argument("--eval-max-per-pair", type=int, default=50)
    ap.add_argument("--skip-bitext", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-mine", action="store_true")
    args = ap.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    data_root = EXP / "corpus"
    data_root.mkdir(exist_ok=True)
    triplets = data_root / "triplets.tsv"
    results = EXP / "cascade_eval_results.json"

    py = sys.executable
    if not args.skip_bitext:
        cmd = [py, str(SCRIPTS / "build_bitext.py"), "--out", str(triplets), "--n", str(args.n)]
        if args.skip_mine:
            cmd.append("--skip-mine")
        run(cmd)

    if not args.skip_tts:
        cmd = [
            py,
            str(SCRIPTS / "synthesize.py"),
            "--triplets",
            str(triplets),
            "--out-dir",
            str(data_root),
        ]
        if args.tts_limit:
            cmd += ["--limit", str(args.tts_limit)]
        run(cmd)

    if not args.skip_eval:
        run(
            [
                py,
                str(SCRIPTS / "evaluate.py"),
                "--data-root",
                str(data_root),
                "--out",
                str(results),
                "--max-per-pair",
                str(args.eval_max_per_pair),
            ]
        )

    package_hf(data_root, results if results.exists() else None)
    package_github(data_root, results if results.exists() else None)
    print("ALL DONE")


if __name__ == "__main__":
    main()
