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
    # A looping model produces numerals of absurd length -- Whisper has rendered
    # a runaway "six six six ..." as a 700-digit integer. Place-value expansion
    # is undefined past `billion` anyway, and for a loop the digit-wise reading
    # is the semantically correct one: each digit was one spoken word.
    if len(t) > 12:
        return [DIGIT_WORDS[c] for c in t]
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


def count_units(tokens: list[str], units: list[str]) -> int:
    """How many of an ordered unit list the transcript delivers, in order.

    One rule serves both item types, which is what makes them comparable. For a
    repeated item the list is k copies of one word, so this counts occurrences of
    that word. For its length-matched control the list is k *distinct* fillers,
    so this counts how many of them were rendered. In both cases the answer to
    "how many of the k requested units came out?" is computed the same way, and
    matching never rewinds, so a model that repeats one filler cannot score for
    the others.
    """
    if not units:
        return 0
    i, c = 0, 0
    for u in units:
        parts = normalise(u)
        if not parts:
            continue
        n = len(parts)
        while i + n <= len(tokens):
            if tokens[i:i + n] == parts:
                c += 1
                i += n
                break
            i += 1
        else:
            break
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
        if r["k"] <= 3 and r["count_a"] == r["expected_count"] and r["duration_s"] > 0.2:
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


DUR_LO, DUR_HI = 0.70, 1.45


def classify(r: dict) -> str:
    """Outcome label from two independent evidence streams.

    Correctness is *conjunctive*: the transcript must show exactly the requested
    number of occurrences AND the audio must be the length that many occurrences
    take. Requiring both is what makes the label robust to the two known failure
    directions --- Whisper de-duplicating a genuine loop (transcript too short,
    duration long) and a model babbling for the right duration (duration right,
    transcript wrong). We never have to *estimate* a count in the failure cases,
    only certify it in the successful ones.

    Order matters: audio-level degeneracy is checked first, because noise of the
    correct length would otherwise pass the duration test.
    """
    if r["duration_s"] < 0.25 or r.get("rms", 1.0) < 1e-3:
        return "empty"
    if r.get("spectral_flatness", 0.0) > 0.35 or not r["transcript"].strip():
        return "degenerate"
    exp, ca = r["expected_count"], r["count_a"]
    dr = r.get("duration_ratio", float("nan"))
    dr_ok = (dr == dr) and (DUR_LO <= dr <= DUR_HI)   # NaN-safe
    if ca == exp and dr_ok:
        return "correct"
    # long: the model kept going. short: it stopped early.
    if r.get("hit_cap") or ((dr == dr) and dr > 1.6):
        return "loop"
    if ca > exp:
        return "overcount"
    if (dr == dr) and dr < DUR_LO:
        return "truncation"
    if ca < exp:
        return "undercount"
    return "miscount"


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
            # Repeated and control items are scored by the same question: how
            # many of the k requested units were rendered? `boundary_units`
            # carries the k copies for a repeated item and the k distinct
            # fillers for its control, so one call covers both. Families without
            # a unit list (numbers, twisters) keep the occurrence count.
            units = it.get("boundary_units")
            if units and len(set(units)) > 1:
                # distinct units (a control): how many of the k were rendered
                cnt, exp = count_units(toks, units), len(units)
            elif units:
                # identical units (a repeated item): the *total* occurrence
                # count, deliberately unbounded so that a model producing more
                # than k is visible as an overcount rather than capped at k
                cnt, exp = count_occurrences(toks, units[0]), len(units)
            else:
                cnt, exp = count_occurrences(toks, it["target_unit"]), it["expected_count"]
            rows.append(dict(
                model=model, item_id=item_id, seed=seed, family=it["family"],
                template=it["template"], k=it["k"],
                expected_count=exp, expected_words=it["expected_words"],
                target_unit=it["target_unit"], transcript=a.get("text", ""),
                n_words=len(toks), duration_s=a.get("duration_s", 0.0),
                count_a=cnt,
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
            # The fit regresses duration on k, so the expectation must use k too
            # -- using expected_count here silently gave every control the
            # duration of a k=1 item and labelled all of them runaway loops.
            r["expected_duration_s"] = a + b * max(r["k"], 1)
            r["duration_ratio"] = (r["duration_s"] / r["expected_duration_s"]
                                   if r["expected_duration_s"] > 1e-6 else float("nan"))
        else:
            r["count_b"] = None
            r["expected_duration_s"] = float("nan")
            r["duration_ratio"] = float("nan")
        # The two estimators are reported side by side rather than merged. Their
        # agreement rate is a measurement-quality statistic; correctness itself
        # is decided conjunctively in `classify`, so a disagreement can never
        # silently become a count.
        ca, cb = r["count_a"], r["count_b"]
        tol = max(1.0, 0.15 * max(r["expected_count"], 1))
        r["agree"] = cb is None or abs(ca - cb) <= tol
        r["count_final"] = ca if r["agree"] else max(ca, cb)
        if not r["agree"]:
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
    # Overall disagreement is not a defect: at high k the model's output genuinely
    # does not correspond to any clean count, so the two estimators have nothing
    # to agree about. The estimator-validation statistic is agreement in the
    # low-k regime, where a correct rendering exists to be measured.
    low = [r for r in rows if r["k"] <= 4]
    agree_low = 100 * np.mean([r["agree"] for r in low]) if low else float("nan")
    print(f"estimator agreement: {agree_low:.1f}% at k<=4 "
          f"({100*(1-n_audit/max(len(rows),1)):.1f}% overall)")
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
