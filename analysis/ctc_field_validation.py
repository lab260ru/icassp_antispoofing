#!/usr/bin/env python3
"""Validate the CTC judge on genuinely model-generated repetition.

`analysis/ctc_validation.py` validated the CTC recogniser
(facebook/wav2vec2-large-960h-lv60-self) against Whisper large-v3 on
*concatenative* audio: one verified k=1 rendering, spliced N times, so the true
count is known by construction. A reviewer's objection is correct: that audio
is *easier* than the phenomenon under study. Splices give the recogniser clean,
unambiguous silences to key word/segment boundaries on; real model-generated
repetition instead has coarticulation across repeats, prosodic drift, and
degrading audio quality as a model loops, all of which are exactly the
conditions that could make a CTC judge's count untrustworthy in a way the
concatenative test cannot see. This script does not have ground-truth
transcripts for real generations (no human-transcription budget), so it
assembles convergent evidence from four checks that do not require one:

  1. Agreement in the regime where both judges should be right. At low k
     (k<=3) the panel's own duration-based cross-check (`score_counts.py`)
     already shows repeated items are rendered correctly, and control items at
     any k never contain periodic material for either judge to mis-handle. If
     CTC and Whisper agree there but diverge as k grows, the divergence is
     localised to periodicity -- exactly what the paper's mechanism predicts --
     rather than being generic CTC noise that would show up everywhere.

  2. A word-level content check on the carrier. Getting the *count* of a
     repeated word right is a weak signal if the recogniser is hallucinating
     the rest of the sentence. This measures word error rate on the words
     *outside* the repeated/control block (the fixed carrier text), which is
     the part of the recognition problem that has nothing to do with counting.
     Low carrier WER at the same k where the repeated-block count is
     compromised means the recogniser is doing its job and the count is
     informative rather than noise.

  3. Internal consistency under perturbation. A judge whose count changes
     under acoustic perturbations that do not add or remove any spoken word
     (mild noise, a 16k->15.5k->16k resample round trip) is measuring an
     artefact of the recording, not the acoustics of repetition. Real
     generations are not studio-clean, so this is the closest thing to a
     robustness bound we can put on the judge without ground truth.

  4. A second, architecturally independent non-autoregressive recogniser.
     Two CTC/non-AR judges of different lineage (wav2vec2 self-supervised
     pretraining + CTC fine-tune, vs HuBERT masked-prediction pretraining +
     CTC fine-tune) agreeing on real generations is much stronger evidence
     than either one validated alone on synthetic audio, because their
     specific errors are not expected to be correlated.

None of these four checks requires knowing the true repetition count of a real
generation -- each instead asks whether the judge behaves the way a working
instrument should, from a different angle. Agreement across all four is
evidence the CTC counts the paper reports are trustworthy; a failure in any one
of them is a real problem with the instrument and is reported as such, not
smoothed over.

Usage:
  python analysis/ctc_field_validation.py --gpu 1
  python analysis/ctc_field_validation.py --skip-gpu   # checks 1-2 only, no model loads
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402

# Reuse the paper's own counting logic verbatim so every number here is
# comparable to `data/results/behavioural_ctc.csv` -- this script validates
# the instrument the paper used, not a reimplementation of it.
from common.score_counts import normalise, count_occurrences, count_units  # noqa: E402

DATA_ROOT = "/home/kirill/mnt/hdd_6tb_1/icassp_tts"
AUDIO_ROOT = Path(DATA_ROOT) / "audio"
CTC_SR = 16000
PANEL_MODELS = ["llasa1b", "llasa3b", "llasa8b", "xtts2", "qwen06b", "qwen17b"]


# ---------------------------------------------------------------------------
# Shared I/O and counting (mirrors score_counts.py / asr_ctc.py exactly)
# ---------------------------------------------------------------------------

def load_stimuli(path: str) -> dict:
    stim = {}
    for line in open(path):
        it = json.loads(line)
        stim[it["item_id"]] = it
    return stim


def load_asr(model: str, judge: str) -> dict:
    """judge in {'ctc', 'whisper'}."""
    sub = "asr" if judge == "whisper" else "asr_ctc"
    p = Path(DATA_ROOT) / sub / f"{model}.jsonl"
    out = {}
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                out[r["stem"]] = r
            except Exception:  # noqa: BLE001
                pass
    return out


def score_item(it: dict, text: str) -> tuple[int, int]:
    """(counted, expected), computed by the exact branch score_counts.main()
    uses per row: distinct-unit control items via count_units, identical-unit
    repeated items (and their target-word count) via count_occurrences."""
    toks = normalise(text)
    units = it.get("boundary_units")
    if units and len(set(units)) > 1:
        return count_units(toks, units), len(units)
    elif units:
        return count_occurrences(toks, units[0]), len(units)
    else:
        return count_occurrences(toks, it["target_unit"]), it["expected_count"]


def kbucket(k: int) -> str:
    if k <= 3:
        return "low_k<=3"
    if k <= 8:
        return "mid_4-8"
    return "high_k>=12"


def family_group(family: str) -> str:
    return "control" if family == "control_word" else "repeated"


def _agg(vals_exact: list[int], vals_absdiff: list[float]) -> dict:
    n = len(vals_exact)
    if n == 0:
        return dict(n=0, exact_match_rate=None, mean_abs_diff=None)
    return dict(n=n,
                exact_match_rate=float(np.mean(vals_exact)),
                mean_abs_diff=float(np.mean(vals_absdiff)))


# ---------------------------------------------------------------------------
# Check 1: CTC vs Whisper agreement, easy regime vs full k range
# ---------------------------------------------------------------------------

def run_check1(stim: dict, models: list[str]) -> dict:
    rows = []
    coverage = {}
    for model in models:
        ctc = load_asr(model, "ctc")
        wsp = load_asr(model, "whisper")
        common = sorted(set(ctc) & set(wsp))
        coverage[model] = dict(ctc_n=len(ctc), whisper_n=len(wsp), common_n=len(common))
        for stem in common:
            if "_s" not in stem:
                continue
            item_id, seed_s = stem.rsplit("_s", 1)
            it = stim.get(item_id)
            if it is None:
                continue
            ctc_text, wsp_text = ctc[stem].get("text", ""), wsp[stem].get("text", "")
            c_ctc, exp = score_item(it, ctc_text)
            c_wsp, _ = score_item(it, wsp_text)
            # Duration comes off the CTC record (both judges transcribe the same
            # wav, so it is identical either way); used below for the output-rate
            # plausibility check, which needs no ground truth at all.
            dur = ctc[stem].get("duration_s") or wsp[stem].get("duration_s") or 0.0
            n_words_ctc, n_words_wsp = len(normalise(ctc_text)), len(normalise(wsp_text))
            rows.append(dict(
                model=model, item_id=item_id, seed=int(seed_s), family=it["family"],
                k=it["k"], expected=exp, count_ctc=c_ctc, count_whisper=c_wsp,
                abs_diff=abs(c_ctc - c_wsp), exact=int(c_ctc == c_wsp),
                easy_regime=bool(it["family"] == "control_word" or it["k"] <= 3),
                duration_s=dur,
                words_per_sec_ctc=(n_words_ctc / dur if dur > 0.05 else None),
                words_per_sec_whisper=(n_words_wsp / dur if dur > 0.05 else None),
            ))

    easy = [r for r in rows if r["easy_regime"]]
    per_model = {
        m: _agg([r["exact"] for r in easy if r["model"] == m],
                [r["abs_diff"] for r in easy if r["model"] == m])
        for m in models
    }
    per_model_k = defaultdict(list)
    for r in easy:
        per_model_k[(r["model"], r["k"])].append(r)
    per_model_k_table = [
        dict(model=m, k=k, **_agg([r["exact"] for r in v], [r["abs_diff"] for r in v]))
        for (m, k), v in sorted(per_model_k.items())
    ]

    # Localisation: pooled agreement by (repeated vs control, k), over the
    # FULL k range (not just the easy regime) -- this is what shows whether
    # any divergence is specific to periodic material and grows with k, or is
    # just generic CTC/Whisper disagreement present everywhere.
    loc = defaultdict(list)
    for r in rows:
        loc[(family_group(r["family"]), r["k"])].append(r)
    localization_by_k = [
        dict(group=g, k=k, **_agg([r["exact"] for r in v], [r["abs_diff"] for r in v]))
        for (g, k), v in sorted(loc.items())
    ]

    # Output-rate plausibility, on repeated items only. This does not need any
    # ground truth: it asks whether each judge's TOTAL transcript length (not
    # just the target word's count) implies a spoken-word rate that is
    # physically possible. English speech tops out around 4-5 words/sec even
    # at a fast clip; a judge whose implied rate is far above that at high k is
    # not transcribing acoustics, it is generating text on its own momentum.
    # If one judge's rate stays flat and physically plausible across the whole
    # k range while the other's climbs into implausible territory exactly
    # where the two disagree, that tells us *which* judge to believe without
    # needing to know the true count.
    rate_rows = [r for r in rows if family_group(r["family"]) == "repeated"
                and r["words_per_sec_ctc"] is not None]
    rate_by_k = defaultdict(list)
    for r in rate_rows:
        rate_by_k[r["k"]].append(r)
    output_rate_by_k = [
        dict(k=k, n=len(v),
            mean_wps_ctc=float(np.mean([r["words_per_sec_ctc"] for r in v])),
            median_wps_ctc=float(np.median([r["words_per_sec_ctc"] for r in v])),
            mean_wps_whisper=float(np.mean([r["words_per_sec_whisper"] for r in v])),
            median_wps_whisper=float(np.median([r["words_per_sec_whisper"] for r in v])))
        for k, v in sorted(rate_by_k.items())
    ]

    return dict(
        coverage=coverage,
        overall_easy_regime=_agg([r["exact"] for r in easy], [r["abs_diff"] for r in easy]),
        per_model_easy_regime=per_model,
        per_model_per_k_easy_regime=per_model_k_table,
        localization_by_k=localization_by_k,
        output_rate_by_k=output_rate_by_k,
        n_rows_total=len(rows),
        n_rows_easy=len(easy),
    )


# ---------------------------------------------------------------------------
# Check 2: carrier word error rate (content check, not just counts)
# ---------------------------------------------------------------------------

# sentence_rep is deliberately left out of the carrier-WER sample. For
# word_rep/control_word/numbers/twisters the carrier is a fixed set of words
# that occurs once regardless of k (only the target/filler slot repeats), so
# stripping the target leaves a carrier whose length does not depend on how
# many repeats the model actually produced. For sentence_rep the *entire*
# sentence is the repeated unit, so its "carrier" (the sentence minus the one
# marker word) also repeats k times -- if the model looped more or fewer times
# than requested (exactly the failure mode this paper studies), the carrier
# reference and hypothesis have different lengths for a reason that has
# nothing to do with recognition quality, and carrier WER would silently
# mislabel a correct transcription of an off-count loop as a content error.
CARRIER_WER_FAMILIES = ("word_rep", "control_word", "numbers", "twisters")


def excluded_words(it: dict) -> set:
    """Words to strip before scoring carrier WER. For a repeated item this is
    every occurrence of the target word (not just a contiguous run), so a
    count error can never leak into the carrier score. For a control item the
    analogous block is the k distinct substituted fillers -- target_unit
    itself usually does not even occur in a control's text."""
    if it["family"] == "control_word":
        words = it.get("control_units") or it.get("boundary_units") or []
    else:
        words = [it["target_unit"]]
    ex: set = set()
    for w in words:
        ex.update(normalise(w))
    return ex


def carrier_wer(ref_text: str, hyp_text: str, excluded: set) -> tuple[float, int]:
    ref = [t for t in normalise(ref_text) if t not in excluded]
    hyp = [t for t in normalise(hyp_text) if t not in excluded]
    if not ref:
        return float("nan"), 0
    return float(jiwer.wer(" ".join(ref), " ".join(hyp))), len(ref)


def stratified_sample_wer(stim: dict, asr_ctc_by_model: dict, flags_by_model: dict,
                          models: list[str], per_model: int = 7, seed: int = 0) -> list[tuple]:
    """~per_model items per model, evenly spread over (family, k), one random
    item drawn per (family, k) bucket per model.

    Two things this guards against, found while inspecting a first pass:
    (1) a deterministic same-index pick across models lands on the exact same
    item_id for every model whenever the per-(family,k) pools are structurally
    identical across models (they are here), which silently kills the
    cross-model diversity the sample is supposed to have; drawing at random
    per model fixes that. (2) a TTS model that produced empty or acoustically
    degenerate audio for the sampled item -- rms/spectral-flatness flags
    already computed alongside the Whisper transcript -- has no genuine
    carrier for the recogniser to get right; scoring carrier WER against a
    TTS failure would measure the TTS model, not the ASR judge, so those items
    are excluded here the same way `score_counts.classify` excludes them.
    """
    rng = np.random.default_rng(seed)
    sample = []
    for model in models:
        ctc = asr_ctc_by_model[model]
        flags = flags_by_model.get(model, {})
        buckets = defaultdict(list)
        for stem in ctc:
            if "_s" not in stem:
                continue
            item_id = stem.rsplit("_s", 1)[0]
            it = stim.get(item_id)
            if it is None or it["family"] not in CARRIER_WER_FAMILIES:
                continue
            fl = flags.get(stem, {})
            dur = fl.get("duration_s", ctc[stem].get("duration_s", 0.0)) or 0.0
            rms = fl.get("rms", 1.0)
            sflat = fl.get("spectral_flatness", 0.0)
            if dur < 0.25 or rms < 1e-3 or (sflat is not None and sflat > 0.35):
                continue  # degenerate/empty TTS output, not an ASR question
            buckets[(it["family"], it["k"])].append(item_id)
        keys = sorted(buckets)
        if not keys:
            continue
        idx = np.unique(np.round(np.linspace(0, len(keys) - 1, per_model)).astype(int))
        for i in idx:
            fam, k = keys[i]
            cands = buckets[(fam, k)]
            item_id = cands[int(rng.integers(len(cands)))]
            sample.append((model, item_id, fam, k))
    return sample


def run_check2(stim: dict, sample: list[tuple], asr_ctc_by_model: dict,
              flags_by_model: dict) -> dict:
    items = []
    for model, item_id, fam, k in sample:
        ctc = asr_ctc_by_model[model]
        stem = f"{item_id}_s0"
        rec = ctc.get(stem)
        if rec is None:
            continue
        it = stim[item_id]
        text = rec.get("text", "")
        ex = excluded_words(it)
        wer, n_carrier = carrier_wer(it["text"], text, ex)
        c_ctc, exp = score_item(it, text)
        # voiced_frac (fraction of the clip above a loudness floor) is kept
        # alongside the WER, not used to filter: a low value is a flag that a
        # high carrier-WER item may be genuinely degraded TTS output rather
        # than a CTC recognition failure, which the report checks by hand
        # rather than silently excluding -- excluding on a threshold invented
        # after seeing which items look bad would be exactly the kind of
        # after-the-fact tuning this validation is not supposed to do.
        voiced_frac = flags_by_model.get(model, {}).get(stem, {}).get("voiced_frac")
        items.append(dict(
            model=model, item_id=item_id, family=fam, k=k, expected=exp,
            count_ctc=c_ctc, count_rel_error=((c_ctc - exp) / exp if exp else None),
            carrier_wer=wer, n_carrier_words=n_carrier, voiced_frac=voiced_frac,
        ))

    def summarize(rows):
        w = [r["carrier_wer"] for r in rows if r["carrier_wer"] == r["carrier_wer"]]
        ce = [abs(r["count_rel_error"]) for r in rows if r["count_rel_error"] is not None]
        return dict(n=len(rows),
                    mean_carrier_wer=float(np.mean(w)) if w else None,
                    median_carrier_wer=float(np.median(w)) if w else None,
                    mean_abs_count_rel_error=float(np.mean(ce)) if ce else None)

    by_bucket = {b: summarize([r for r in items if kbucket(r["k"]) == b])
                for b in ("low_k<=3", "mid_4-8", "high_k>=12")}
    # Diagnostic, not a filter: does high carrier WER track low voiced_frac
    # (garbled/degenerate TTS audio) rather than clean speech the recogniser
    # simply got wrong? Reported alongside the headline number so a reader can
    # see whether "CTC's carrier WER is 0.27" mostly reflects TTS failures it
    # correctly rendered as garbage, or clean speech it misread.
    by_voiced = {
        "voiced_frac>=0.5": summarize([r for r in items
                                       if (r["voiced_frac"] or 0) >= 0.5]),
        "voiced_frac<0.5": summarize([r for r in items
                                      if r["voiced_frac"] is not None and r["voiced_frac"] < 0.5]),
    }

    return dict(sample_n=len(items), overall=summarize(items),
                by_k_bucket=by_bucket, by_voiced_frac=by_voiced, items=items)


# ---------------------------------------------------------------------------
# Checks 3 & 4: perturbation stability and a second recogniser (need GPU)
# ---------------------------------------------------------------------------

def add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at a fixed SNR. ~30 dB is mild -- well
    above anything that would make a competent recogniser mishear a word --
    so this probes whether the judge's *count* is glued to acoustic detail
    that has nothing to do with how many words were spoken. Real generations
    were not recorded in a soundbooth-clean condition either, so a count that
    is fragile to this is a count that should not be trusted on them."""
    x = wav.astype(np.float64)
    sig_power = float(np.mean(x ** 2))
    if sig_power < 1e-12:
        return wav
    noise = rng.standard_normal(x.shape)
    noise_power = float(np.mean(noise ** 2))
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return (x + noise).astype(np.float32)


def resample_roundtrip(wav: np.ndarray, sr: int, mid_sr: int) -> np.ndarray:
    """sr -> mid_sr -> sr. Resampling-filter ringing and a slight time warp,
    with the same words at the same approximate rate -- a second, independent
    kind of perturbation that should not change how many words were spoken."""
    down = librosa.resample(wav, orig_sr=sr, target_sr=mid_sr)
    return librosa.resample(down, orig_sr=mid_sr, target_sr=sr)


def make_transcriber(model, proc, device: str, chunk_s: float = 25.0):
    """Chunked greedy CTC decode, structurally identical to
    `src/common/asr_ctc.py`'s transcriber (chunk + short overlap so a word is
    not cut at a boundary), generalised over model/processor so the primary
    and secondary recognisers are driven by the exact same code path -- any
    difference we find between them cannot be a chunking artefact."""
    step = int(chunk_s * CTC_SR)
    overlap = int(0.25 * CTC_SR)

    @torch.no_grad()
    def transcribe(wav: np.ndarray, sr: int) -> str:
        if sr != CTC_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=CTC_SR)
        pieces = []
        i = 0
        while i < wav.size:
            seg = wav[max(0, i - (overlap if i else 0)): i + step]
            if seg.size < CTC_SR // 20:
                break
            inp = proc(seg, sampling_rate=CTC_SR, return_tensors="pt")
            logits = model(inp.input_values.to(device)).logits
            ids = torch.argmax(logits, dim=-1)
            pieces.append(proc.batch_decode(ids)[0].strip())
            i += step
        return " ".join(p for p in pieces if p)

    return transcribe


def _not_degenerate(flags: dict, stem: str, fallback_dur: float) -> bool:
    """Same gate as `stratified_sample_wer`: exclude items where the TTS
    model itself produced empty/degenerate audio, so checks 3-4 measure the
    ASR judge's behaviour, not a TTS failure that neither judge could read."""
    fl = flags.get(stem, {})
    dur = fl.get("duration_s", fallback_dur) or 0.0
    rms = fl.get("rms", 1.0)
    sflat = fl.get("spectral_flatness", 0.0)
    return dur >= 0.25 and rms >= 1e-3 and (sflat is None or sflat <= 0.35)


def sample_word_rep_full_k(stim: dict, asr_ctc_by_model: dict, flags_by_model: dict,
                           models: list[str], seed: int = 0) -> list[tuple]:
    """One word_rep item per (model, k), covering the family's full k grid
    {1,2,3,4,6,8,12,16,24,32} -- the family and range the paper's central
    counting claim is about. Requires a CTC transcript AND the wav on disk
    (checks 3-4 re-run inference on the audio), and excludes degenerate/empty
    TTS output for the reason given in `_not_degenerate`."""
    rng = np.random.default_rng(seed)
    by_k = defaultdict(list)
    for iid, it in stim.items():
        if it["family"] == "word_rep":
            by_k[it["k"]].append(iid)
    sample = []
    for model in models:
        ctc = asr_ctc_by_model[model]
        flags = flags_by_model.get(model, {})
        for k in sorted(by_k):
            cands = [iid for iid in by_k[k]
                     if f"{iid}_s0" in ctc and (AUDIO_ROOT / model / f"{iid}_s0.wav").exists()
                     and _not_degenerate(flags, f"{iid}_s0", ctc[f"{iid}_s0"].get("duration_s", 0.0))]
            if not cands:
                continue
            sample.append((model, cands[int(rng.integers(len(cands)))], "word_rep", k))
    return sample


def sample_control_companion(stim: dict, asr_ctc_by_model: dict, flags_by_model: dict,
                             models: list[str], ks: tuple, seed: int = 1) -> list[tuple]:
    """A small control_word companion sample (same k's, both ends of the
    range) so check 3 also reports whether perturbation stability holds on
    non-periodic audio -- if a control's (true-zero) count started moving
    under perturbation, that would be evidence the judge fabricates counts
    from noise rather than acoustics."""
    rng = np.random.default_rng(seed)
    by_k = defaultdict(list)
    for iid, it in stim.items():
        if it["family"] == "control_word":
            by_k[it["k"]].append(iid)
    sample = []
    for model in models:
        ctc = asr_ctc_by_model[model]
        flags = flags_by_model.get(model, {})
        for k in ks:
            cands = [iid for iid in by_k.get(k, [])
                     if f"{iid}_s0" in ctc and (AUDIO_ROOT / model / f"{iid}_s0.wav").exists()
                     and _not_degenerate(flags, f"{iid}_s0", ctc[f"{iid}_s0"].get("duration_s", 0.0))]
            if not cands:
                continue
            sample.append((model, cands[int(rng.integers(len(cands)))], "control_word", k))
    return sample


def run_gpu_checks(stim: dict, asr_ctc_by_model: dict, sample_core: list[tuple],
                   sample_control: list[tuple], args) -> tuple[dict, dict]:
    device = f"cuda:{args.gpu}"
    from transformers import AutoModelForCTC, AutoProcessor

    print(f"[gpu] loading primary CTC {args.ctc} on {device}", flush=True)
    proc1 = AutoProcessor.from_pretrained(args.ctc)
    model1 = AutoModelForCTC.from_pretrained(args.ctc).to(device).eval()
    tx1 = make_transcriber(model1, proc1, device)

    print(f"[gpu] loading secondary CTC {args.second_ctc} on {device}", flush=True)
    proc2 = AutoProcessor.from_pretrained(args.second_ctc)
    model2 = AutoModelForCTC.from_pretrained(args.second_ctc).to(device).eval()
    tx2 = make_transcriber(model2, proc2, device)

    rng = np.random.default_rng(args.seed)
    core_set = set(sample_core)
    all_items = sample_core + sample_control

    rows3, rows4 = [], []
    for j, (model, item_id, fam, k) in enumerate(all_items):
        stem = f"{item_id}_s0"
        wav_path = AUDIO_ROOT / model / f"{stem}.wav"
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        it = stim[item_id]
        stored_text = asr_ctc_by_model[model].get(stem, {}).get("text", "")
        count_stored, exp = score_item(it, stored_text)

        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=CTC_SR) if sr != CTC_SR else wav

        # Fresh clean re-transcription with THIS script's own transcriber --
        # doubles as a sanity check that our chunking reproduces the stored
        # production transcript, and gives an apples-to-apples baseline (same
        # code path as the perturbed variants) for measuring perturbation
        # sensitivity.
        clean_text = tx1(wav16, CTC_SR)
        count_clean, _ = score_item(it, clean_text)

        noisy = add_noise(wav16, args.snr_db, rng)
        noisy_text = tx1(noisy, CTC_SR)
        count_noise, _ = score_item(it, noisy_text)

        rt = resample_roundtrip(wav16, CTC_SR, args.resample_mid)
        rt_text = tx1(rt, CTC_SR)
        count_rt, _ = score_item(it, rt_text)

        rows3.append(dict(
            model=model, item_id=item_id, family=fam, k=k, expected=exp,
            count_stored=count_stored, count_clean=count_clean,
            count_noise=count_noise, count_resample=count_rt,
            abs_diff_noise=abs(count_noise - count_clean),
            abs_diff_resample=abs(count_rt - count_clean),
            stored_vs_clean_match=int(count_stored == count_clean),
        ))

        if (model, item_id, fam, k) in core_set:
            sec_text = tx2(wav16, CTC_SR)
            count_sec, _ = score_item(it, sec_text)
            rows4.append(dict(
                model=model, item_id=item_id, family=fam, k=k, expected=exp,
                count_primary=count_clean, count_secondary=count_sec,
                abs_diff=abs(count_sec - count_clean),
                exact=int(count_sec == count_clean),
            ))

        if (j + 1) % 10 == 0:
            print(f"  [gpu] {j + 1}/{len(all_items)}", flush=True)

    del model1, model2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    check3 = summarize_check3(rows3)
    check4 = summarize_check4(rows4, args.second_ctc)
    return check3, check4


def summarize_check3(rows: list[dict]) -> dict:
    sanity = dict(
        n=len(rows),
        agree_rate=float(np.mean([r["stored_vs_clean_match"] for r in rows])) if rows else None,
    )

    def summarize(sub):
        n = len(sub)
        if n == 0:
            return dict(n=0, mean_abs_diff_noise=None, mean_abs_diff_resample=None,
                        frac_unchanged_noise=None, frac_unchanged_resample=None)
        return dict(
            n=n,
            mean_abs_diff_noise=float(np.mean([r["abs_diff_noise"] for r in sub])),
            mean_abs_diff_resample=float(np.mean([r["abs_diff_resample"] for r in sub])),
            frac_unchanged_noise=float(np.mean([r["abs_diff_noise"] == 0 for r in sub])),
            frac_unchanged_resample=float(np.mean([r["abs_diff_resample"] == 0 for r in sub])),
        )

    by_group = {}
    for g in ("word_rep", "control_word"):
        sub = [r for r in rows if r["family"] == g]
        by_group[g] = dict(
            overall=summarize(sub),
            by_k_bucket={b: summarize([r for r in sub if kbucket(r["k"]) == b])
                        for b in ("low_k<=3", "mid_4-8", "high_k>=12")},
        )

    return dict(sanity_stored_vs_clean=sanity, by_group=by_group, items=rows)


def summarize_check4(rows: list[dict], second_model: str) -> dict:
    def summarize(sub):
        n = len(sub)
        if n == 0:
            return dict(n=0, exact_match_rate=None, mean_abs_diff=None)
        return dict(n=n,
                    exact_match_rate=float(np.mean([r["exact"] for r in sub])),
                    mean_abs_diff=float(np.mean([r["abs_diff"] for r in sub])))

    return dict(
        second_model=second_model,
        rationale=("HuBERT-large-ls960-ft: a non-AR CTC recogniser of comparable capacity and "
                  "accuracy to the primary judge (wav2vec2-large-960h-lv60-self), but from a "
                  "different self-supervised pretraining objective (masked cluster-label "
                  "prediction vs contrastive), so its specific errors are not expected to be "
                  "correlated with the primary judge's. wav2vec2-conformer-rope-large-960h-ft "
                  "was passed over as still being wav2vec2-lineage (same pretraining objective, "
                  "just a conformer/RoPE encoder); wav2vec2-base-960h was passed over as "
                  "substantially weaker (WER roughly double the primary judge's), which would "
                  "confound 'independent recogniser disagrees' with 'weak recogniser is noisy'."),
        overall=summarize(rows),
        by_k_bucket={b: summarize([r for r in rows if kbucket(r["k"]) == b])
                    for b in ("low_k<=3", "mid_4-8", "high_k>=12")},
        items=rows,
    )


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/ctc_field_validation.json"))
    ap.add_argument("--models", nargs="+", default=PANEL_MODELS)
    ap.add_argument("--ctc", default="facebook/wav2vec2-large-960h-lv60-self")
    ap.add_argument("--second-ctc", default="facebook/hubert-large-ls960-ft")
    ap.add_argument("--wer-sample-per-model", type=int, default=7)
    ap.add_argument("--snr-db", type=float, default=30.0)
    ap.add_argument("--resample-mid", type=int, default=15500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-gpu", action="store_true",
                    help="run checks 1-2 only (no model loads, no GPU needed)")
    args = ap.parse_args()

    stim = load_stimuli(args.stimuli)
    print(f"loaded {len(stim)} stimuli", flush=True)

    print("\n=== Check 1: CTC vs Whisper agreement ===", flush=True)
    check1 = run_check1(stim, args.models)
    print(f"  easy-regime (k<=3 repeated + all-k control): "
          f"n={check1['overall_easy_regime']['n']} "
          f"exact_match={check1['overall_easy_regime']['exact_match_rate']:.3f} "
          f"mean_abs_diff={check1['overall_easy_regime']['mean_abs_diff']:.3f}", flush=True)
    for m, s in check1["per_model_easy_regime"].items():
        if s["n"]:
            print(f"    {m:10s} n={s['n']:4d} exact={s['exact_match_rate']:.3f} "
                  f"mean|diff|={s['mean_abs_diff']:.3f}", flush=True)
    print("  localization_by_k (repeated vs control, full k range):", flush=True)
    for r in check1["localization_by_k"]:
        print(f"    {r['group']:9s} k={r['k']:<3d} n={r['n']:<4d} "
              f"exact={r['exact_match_rate']:.3f} mean|diff|={r['mean_abs_diff']:.2f}", flush=True)
    print("  output-rate plausibility (words/sec implied by the WHOLE transcript, "
          "repeated items; physically plausible ceiling ~4-5 wps):", flush=True)
    for r in check1["output_rate_by_k"]:
        print(f"    k={r['k']:<3d} n={r['n']:<4d} "
              f"median_wps_ctc={r['median_wps_ctc']:6.2f} "
              f"median_wps_whisper={r['median_wps_whisper']:6.2f}", flush=True)

    print("\n=== Check 2: carrier WER (content check) ===", flush=True)
    asr_ctc_by_model = {m: load_asr(m, "ctc") for m in args.models}
    # Whisper's records double as the degeneracy-flag source (rms, spectral
    # flatness) -- computed once alongside the Whisper pass and judge-
    # independent, exactly as `score_counts.load` borrows them for the CTC-
    # judged table.
    flags_by_model = {m: load_asr(m, "whisper") for m in args.models}
    wer_sample = stratified_sample_wer(stim, asr_ctc_by_model, flags_by_model, args.models,
                                       per_model=args.wer_sample_per_model, seed=args.seed)
    check2 = run_check2(stim, wer_sample, asr_ctc_by_model, flags_by_model)
    print(f"  n={check2['overall']['n']} mean_carrier_wer={check2['overall']['mean_carrier_wer']:.3f} "
          f"mean_abs_count_rel_error={check2['overall']['mean_abs_count_rel_error']:.3f}", flush=True)
    for b, s in check2["by_k_bucket"].items():
        if s["n"]:
            print(f"    {b:12s} n={s['n']:3d} carrier_wer={s['mean_carrier_wer']:.3f} "
                  f"count_rel_err={s['mean_abs_count_rel_error']:.3f}", flush=True)
    print("  by voiced_frac (does high WER track degraded TTS audio rather than "
          "clean speech CTC misread?):", flush=True)
    for b, s in check2["by_voiced_frac"].items():
        if s["n"]:
            print(f"    {b:18s} n={s['n']:3d} carrier_wer={s['mean_carrier_wer']:.3f}", flush=True)

    out = dict(
        config=dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            models=args.models, ctc=args.ctc, second_ctc=args.second_ctc,
            snr_db=args.snr_db, resample_mid=args.resample_mid, seed=args.seed,
            gpu=args.gpu, skip_gpu=args.skip_gpu,
        ),
        check1_agreement_easy_regime=check1,
        check2_carrier_wer=check2,
    )

    if not args.skip_gpu:
        print("\n=== Checks 3-4: perturbation stability + second recogniser (GPU) ===", flush=True)
        sample_core = sample_word_rep_full_k(stim, asr_ctc_by_model, flags_by_model,
                                             args.models, seed=args.seed)
        ks = (min(it["k"] for it in stim.values() if it["family"] == "control_word"),
              max(it["k"] for it in stim.values() if it["family"] == "control_word"))
        sample_ctrl = sample_control_companion(stim, asr_ctc_by_model, flags_by_model,
                                               args.models, ks, seed=args.seed + 1)
        print(f"  core sample (word_rep, full k grid): {len(sample_core)} items", flush=True)
        print(f"  control companion sample (k in {ks}): {len(sample_ctrl)} items", flush=True)
        check3, check4 = run_gpu_checks(stim, asr_ctc_by_model, sample_core, sample_ctrl, args)

        print(f"\n  [check3] stored-vs-fresh-clean sanity agreement: "
              f"{check3['sanity_stored_vs_clean']['agree_rate']:.3f} "
              f"(n={check3['sanity_stored_vs_clean']['n']})", flush=True)
        for g, d in check3["by_group"].items():
            o = d["overall"]
            if o["n"]:
                print(f"  [check3] {g:14s} n={o['n']:3d} "
                      f"mean|dcount| noise={o['mean_abs_diff_noise']:.3f} "
                      f"resample={o['mean_abs_diff_resample']:.3f} "
                      f"frac_unchanged noise={o['frac_unchanged_noise']:.3f} "
                      f"resample={o['frac_unchanged_resample']:.3f}", flush=True)

        print(f"\n  [check4] second recogniser ({args.second_ctc}) vs primary, n="
              f"{check4['overall']['n']}: exact_match={check4['overall']['exact_match_rate']:.3f} "
              f"mean|diff|={check4['overall']['mean_abs_diff']:.3f}", flush=True)

        out["check3_perturbation_stability"] = check3
        out["check4_second_recogniser"] = check4
    else:
        print("\n--skip-gpu set: checks 3-4 not run", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
