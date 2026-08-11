#!/usr/bin/env python3
"""Turn transcripts into repetition-counting outcomes.

The central measurement of the paper is: *given text asking for k repetitions,
how many did the model actually produce?* That is harder than it sounds, because
the judge (Whisper) is itself an autoregressive model that collapses repeated
material. We therefore never trust a single estimate:

  Count A  transcript-based — occurrences of the target unit in the normalised
           transcript. Accurate when the ASR renders every repetition, which it
           does reliably at low k and unreliably at high k.
  Count B  duration-based — per (model, template) we fit  duration = a + b*k
           on items where Count A is corroborated (low k), then invert the fit.
           Immune to ASR de-duplication, blind to the model babbling at the
           right length.

They fail in different directions, so agreement is evidence and disagreement is
an explicit `audit` bucket rather than a silent wrong number.

Usage:
  python src/common/score_counts.py --models llasa1b llasa3b --out data/results/behavioural.csv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"

# Digit forms Whisper prefers over words, mapped back so number stimuli can be
# counted in the same normalised space as everything else.
DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
TENS = {
    "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty",
    "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "10": "ten",
}


def expand_number_token(tok: str) -> list[str]:
    """Expand a bare numeral into English number words.

    Whisper writes "666,666" where the stimulus said "six hundred sixty six
    thousand six hundred sixty six"; without this the number family would score
    zero occurrences of every target.
    """
    t = tok.replace(",", "")
    if not t.isdigit():
        return [tok]
    out: list[str] = []

    def under_thousand(n: int) -> list[str]:
        r: list[str] = []
        if n >= 100:
            r += [DIGIT_WORDS[str(n // 100)], "hundred"]
            n %= 100
        if n >= 20:
            r.append(TENS[str((n // 10) * 10)])
            n %= 10
        elif 10 <= n <= 19:
            r.append(TENS[str(n)])
            n = 0
        if n > 0:
            r.append(DIGIT_WORDS[str(n)])
        return r

    n = int(t)
    if n == 0:
        return ["zero"]
    for div, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")):
        if n >= div:
            out += under_thousand(n // div) + [name]
            n %= div
    out += under_thousand(n)
    return out


def normalise(text: str) -> list[str]:
    """Lowercase, split, and expand numerals.

    Commas survive the character filter only so that thousands separators reach
    `expand_number_token` intact; every non-numeric token has them stripped,
    otherwise `"very,"` would fail to match the target unit `"very"`.
    """
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9,\s]", " ", text)
    toks: list[str] = []
    for t in text.split():
        if re.fullmatch(r"[\d,]+", t):
            toks.extend(expand_number_token(t))
        else:
            t = t.replace(",", "")
            if t:
                toks.append(t)
    return [t for t in toks if t]


def count_occurrences(tokens: list[str], unit: str) -> int:
    """Exact whole-token match for single-word units; greedy non-overlapping
    sequence match for multi-word units."""
    parts = normalise(unit)
    if len(parts) <= 1:
        u = parts[0] if parts else unit.lower()
        return sum(1 for t in tokens if t == u)
    n, i, c = len(parts), 0, 0
    while i + n <= len(tokens):
        if tokens[i:i + n] == parts:
            c += 1
            i += n
        else:
            i += 1
    return c


def load(model: str) -> tuple[dict, dict]:
    asr = {}
    p = Path(DATA_ROOT) / "asr" / f"{model}.jsonl"
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                asr[r["stem"]] = r
            except Exception:  # noqa: BLE001
                pass
    meta = {}
    p = Path(DATA_ROOT) / "tokens" / f"{model}_meta.jsonl"
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                meta[(r["item_id"], r["seed"])] = r
            except Exception:  # noqa: BLE001
                pass
    return asr, meta


def fit_duration_models(rows: list[dict]) -> dict:
    """duration = a + b*k, fitted per (model, family, template) on the trusted
    low-k regime. Falls back to (model, family) then (model) when a cell is thin.

    The trusted regime is defined *before* looking at high-k behaviour: k<=4 and
    Count A equal to the requested k. That keeps the calibration independent of
    the effect being measured.
    """
    buckets: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r["k"] <= 4 and r["count_a"] == r["expected_count"] and r["duration_s"] > 0.2:
            for key in ((r["model"], r["family"], r["template"]),
                        (r["model"], r["family"]),
                        (r["model"],)):
                buckets[key].append((r["k"], r["duration_s"]))
    fits = {}
    for key, pts in buckets.items():
        ks = np.array([p[0] for p in pts], float)
        ds = np.array([p[1] for p in pts], float)
        if len(set(ks.tolist())) >= 2:
            b, a = np.polyfit(ks, ds, 1)
        elif len(ks):
            b, a = ds.mean() / max(ks.mean(), 1e-6), 0.0
        else:
            continue
        if b > 1e-3:
            fits[key] = (float(a), float(b), len(pts))
    return fits


def classify(r: dict) -> str:
    """Outcome label. Order matters: audio-level degeneracy dominates, because a
    model emitting noise at the right length would otherwise score as correct."""
    if r["duration_s"] < 0.25 or r.get("rms", 1.0) < 1e-3:
        return "empty"
    if r.get("spectral_flatness", 0.0) > 0.35 or not r["transcript"].strip():
        return "degenerate"
    exp, got = r["expected_count"], r["count_final"]
    if got is None:
        return "audit"
    if got == exp:
        return "correct"
    if got > exp:
        return "loop" if (r.get("hit_cap") or r.get("duration_ratio", 1.0) > 1.6) else "overcount"
    return "truncation" if r.get("duration_ratio", 1.0) < 0.7 else "undercount"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--stimuli", default="data/stimuli/stimuli.jsonl")
    ap.add_argument("--out", default="data/results/behavioural.csv")
    args = ap.parse_args()

    stim = {}
    for line in open(args.stimuli):
        it = json.loads(line)
        stim[it["item_id"]] = it

    rows: list[dict] = []
    for model in args.models:
        asr, meta = load(model)
        for stem, a in asr.items():
            if "_s" not in stem:
                continue
            item_id, seed_s = stem.rsplit("_s", 1)
            it = stim.get(item_id)
            if it is None:
                continue
            seed = int(seed_s)
            m = meta.get((item_id, seed), {})
            toks = normalise(a.get("text", ""))
            rows.append(dict(
                model=model, item_id=item_id, seed=seed, family=it["family"],
                template=it["template"], k=it["k"],
                expected_count=it["expected_count"], expected_words=it["expected_words"],
                target_unit=it["target_unit"], transcript=a.get("text", ""),
                n_words=len(toks), duration_s=a.get("duration_s", 0.0),
                count_a=count_occurrences(toks, it["target_unit"]),
                rms=a.get("rms", float("nan")),
                spectral_flatness=a.get("spectral_flatness", float("nan")),
                trailing_silence_s=a.get("trailing_silence_s", float("nan")),
                loop_ac_peak=a.get("loop_ac_peak", float("nan")),
                n_speech_tokens=m.get("n_speech_tokens", -1),
                hit_cap=bool(m.get("hit_cap", False)),
            ))

    fits = fit_duration_models(rows)
    n_audit = 0
    for r in rows:
        fit = (fits.get((r["model"], r["family"], r["template"]))
               or fits.get((r["model"], r["family"]))
               or fits.get((r["model"],)))
        if fit:
            a, b, _ = fit
            r["count_b"] = max(0, int(round((r["duration_s"] - a) / b)))
            r["expected_duration_s"] = a + b * r["expected_count"] if r["expected_count"] else a + b
            r["duration_ratio"] = (r["duration_s"] / r["expected_duration_s"]
                                   if r["expected_duration_s"] > 1e-6 else float("nan"))
        else:
            r["count_b"] = None
            r["expected_duration_s"] = float("nan")
            r["duration_ratio"] = float("nan")
        # reconcile
        ca, cb = r["count_a"], r["count_b"]
        tol = max(1.0, 0.15 * max(r["expected_count"], 1))
        if cb is None or abs(ca - cb) <= tol:
            r["count_final"] = ca
            r["agree"] = True
        else:
            # ASR de-duplication is the dominant failure at high k, so when the
            # two disagree we take the larger — but flag the item for audit.
            r["count_final"] = max(ca, cb)
            r["agree"] = False
            n_audit += 1
        r["outcome"] = classify(r)
        r["correct"] = int(r["outcome"] == "correct")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    cols = [c for c in rows[0] if c != "transcript"] + ["transcript"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows -> {out}")
    print(f"audit bucket: {n_audit} ({100*n_audit/max(len(rows),1):.1f}%)")
    from collections import Counter
    for model in args.models:
        sub = [r for r in rows if r["model"] == model and r["family"] == "word_rep"]
        if not sub:
            continue
        print(f"\n[{model}] word_rep accuracy by k")
        byk = defaultdict(list)
        for r in sub:
            byk[r["k"]].append(r["correct"])
        for k in sorted(byk):
            v = byk[k]
            print(f"  k={k:2d}  acc={np.mean(v):.2f}  n={len(v)}")
        print(f"  outcomes: {dict(Counter(r['outcome'] for r in sub))}")


if __name__ == "__main__":
    main()
