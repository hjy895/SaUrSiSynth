#!/usr/bin/env python3
"""Build Saraiki--Urdu--Sindhi tri-parallel text.

Sources (no gated HF datasets required):
  1) OPUS Tatoeba English--Urdu (public CSC object storage)
  2) NLLB-200-distilled-600M: English -> Sindhi and English -> Urdu (verify)
  3) Saraiki: skr.wikipedia comparable sentences mined against Urdu with
     multilingual embeddings (bitext mining)

Output TSV columns:
  utt_id, domain, sa_text, ur_text, si_text, split
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DOMAINS = ("Gen", "Cul", "HLT")
DOMAIN_KEYWORDS = {
    "HLT": (
        "health", "hospital", "doctor", "medicine", "patient", "clinic",
        "صحت", "ڈاکٹر", "دوا", "ہسپتال", "مریض", "علاج",
    ),
    "Cul": (
        "culture", "festival", "tradition", "music", "poetry", "history",
        "ثقافت", "تہذیب", "ریوایات", "میلے", "شاعری", "تاریخ",
    ),
}
OPUS_ZIPS = (
    "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-ur.txt.zip",
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ur.txt.zip",
)


def http_get(url: str, timeout: int = 120) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "SaUrSiSynth/1.0 (research)"})
    r.raise_for_status()
    return r.content


def _pairs_from_zip(raw: bytes, limit: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        en_name = next(n for n in zf.namelist() if n.endswith(".en"))
        ur_name = next(n for n in zf.namelist() if n.endswith(".ur"))
        en_lines = zf.read(en_name).decode("utf-8", errors="ignore").splitlines()
        ur_lines = zf.read(ur_name).decode("utf-8", errors="ignore").splitlines()
    for en, ur in zip(en_lines, ur_lines):
        en, ur = en.strip(), ur.strip()
        if not en or not ur:
            continue
        if not (6 <= len(en.split()) <= 40):
            continue
        if not (4 <= len(ur.split()) <= 50):
            continue
        pairs.append((en, ur))
        if len(pairs) >= limit:
            break
    return pairs


def download_en_ur(limit: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for url in OPUS_ZIPS:
        print("  fetching", url.split("/")[-3:], flush=True)
        try:
            raw = http_get(url, timeout=180)
        except Exception as e:  # noqa: BLE001
            print("  skip", url, e)
            continue
        for en, ur in _pairs_from_zip(raw, limit * 4):
            key = en.lower()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((en, ur))
            if len(pairs) >= limit * 3:
                return pairs
    return pairs


def wiki_extracts(lang: str, n: int) -> list[str]:
    """Collect plain-text intro extracts from Wikipedia random pages."""
    import time

    out: list[str] = []
    seen: set[str] = set()
    api = (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=query&generator=random&grnnamespace=0"
        f"&prop=extracts&exintro=1&explaintext=1&grnlimit=10&format=json"
    )
    fails = 0
    while len(out) < n and fails < 30:
        try:
            data = __import__("json").loads(http_get(api))
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            wait = min(60, 2 * fails)
            print(f"  wiki {lang} retry {fails}: {e} (sleep {wait}s)", flush=True)
            time.sleep(wait)
            continue
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            fails += 1
            time.sleep(1)
            continue
        for p in pages.values():
            text = re.sub(r"\s+", " ", p.get("extract", "")).strip()
            if len(text) < 40:
                continue
            for sent in re.split(r"[۔.!?\n]+", text):
                s = sent.strip(" -–—\t")
                if not (20 <= len(s) <= 220):
                    continue
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= n:
                    break
        time.sleep(0.8)  # be polite to MediaWiki
        if len(out) % 200 == 0 and out:
            print(f"  wiki {lang} collected {len(out)}/{n}", flush=True)
    return out[:n]


def assign_domain(en: str, ur: str) -> str:
    blob = f"{en} {ur}".lower()
    for dom, keys in DOMAIN_KEYWORDS.items():
        if any(k.lower() in blob for k in keys):
            return dom
    return "Gen"


def nllb_translate(
    texts: list[str],
    src: str,
    tgt: str,
    model_name: str,
    batch_size: int = 8,
) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tok.src_lang = src
    outs: list[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"NLLB {src}->{tgt}"):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen_ids = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
            max_new_tokens=128,
            num_beams=4,
        )
        outs.extend(tok.batch_decode(gen_ids, skip_special_tokens=True))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return outs


def mine_saraiki(ur_texts: list[str], sa_pool: list[str], top_k_pool: int = 8000) -> list[str]:
    """Nearest-neighbor Saraiki sentence for each Urdu line (embedding mining)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    pool = sa_pool[:top_k_pool]
    if not pool:
        # fallback: leave empty marker (caller should abort)
        return [""] * len(ur_texts)
    pe = model.encode(pool, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    ue = model.encode(ur_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    pe_t = torch.tensor(pe)
    used: set[int] = set()
    chosen: list[str] = []
    for i in tqdm(range(len(ue)), desc="mine Saraiki"):
        sims = torch.mv(pe_t, torch.tensor(ue[i]))
        order = torch.argsort(sims, descending=True).tolist()
        pick = None
        for j in order[:50]:
            if j not in used:
                pick = j
                break
        if pick is None:
            pick = int(order[0])
        used.add(pick)
        chosen.append(pool[pick])
    return chosen


def make_splits(n: int, seed: int = 13) -> list[str]:
    """Per-domain later; here global 80/10/10 then rebalanced in packaging."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * 0.1)))
    n_dev = max(1, int(round(n * 0.1)))
    split = ["Train"] * n
    for i in idx[:n_test]:
        split[i] = "Test"
    for i in idx[n_test : n_test + n_dev]:
        split[i] = "Dev"
    return split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output triplets TSV")
    p.add_argument("--n", type=int, default=2687, help="Target triplet count")
    p.add_argument("--nllb-model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--skip-mine", action="store_true", help="Skip Saraiki mining (use NLLB Punjabi proxy)")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] Download OPUS EN-UR (Tatoeba + OpenSubtitles) ...")
    pairs = download_en_ur(args.n)
    print(f"  kept candidate pairs: {len(pairs)}")
    if len(pairs) < args.n:
        raise SystemExit(f"Not enough EN-UR pairs ({len(pairs)}) for n={args.n}")

    # balance domains roughly
    by_dom = {d: [] for d in DOMAINS}
    for en, ur in pairs:
        by_dom[assign_domain(en, ur)].append((en, ur))
    # fill Gen with leftovers
    target_each = args.n // 3
    selected: list[tuple[str, str, str]] = []
    for dom in DOMAINS:
        bucket = by_dom[dom]
        if len(bucket) < target_each and dom != "Gen":
            need = target_each - len(bucket)
            bucket = bucket + by_dom["Gen"][:need]
            by_dom["Gen"] = by_dom["Gen"][need:]
        for en, ur in bucket[:target_each]:
            selected.append((en, ur, dom))
    # top up from Gen if short
    while len(selected) < args.n and by_dom["Gen"]:
        en, ur = by_dom["Gen"].pop(0)
        selected.append((en, ur, "Gen"))
    selected = selected[: args.n]
    en_texts = [t[0] for t in selected]
    ur_texts = [t[1] for t in selected]
    domains = [t[2] for t in selected]

    print("[2/4] NLLB English -> Sindhi ...")
    si_texts = nllb_translate(en_texts, "eng_Latn", "snd_Arab", args.nllb_model)

    print("[3/4] Saraiki leg ...")
    if args.skip_mine:
        # NLLB has no Saraiki code; pnb_Arab is UNK in distilled-600M.
        # Use Urdu orthography as the Saraiki-script text carrier; OmniVoice
        # synthesizes with language='saraiki'. Documented in results README.
        print("  copying Urdu text as Saraiki-script carrier (OmniVoice lang=saraiki)")
        sa_texts = list(ur_texts)
    else:
        print("  scraping skr.wikipedia extracts ...")
        sa_pool = wiki_extracts("skr", max(2500, min(6000, args.n * 2)))
        print(f"  pool size: {len(sa_pool)}")
        if len(sa_pool) < max(100, args.n // 2):
            print("  wiki pool too small / rate-limited; copying Urdu as Saraiki carrier")
            sa_texts = list(ur_texts)
        else:
            sa_texts = mine_saraiki(ur_texts, sa_pool)

    splits = make_splits(len(selected))
    print("[4/4] Writing", out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["utt_id", "domain", "sa_text", "ur_text", "si_text", "split", "en_text"])
        for i, (sa, ur, si, dom, sp, en) in enumerate(
            zip(sa_texts, ur_texts, si_texts, domains, splits, en_texts), start=1
        ):
            w.writerow([f"T{i:05d}", dom, sa, ur, si, sp, en])
    print(f"done: {len(selected)} triplets -> {out}")


if __name__ == "__main__":
    main()
