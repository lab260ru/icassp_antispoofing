#!/usr/bin/env python3
"""A stratified audio sample small enough to ship with the paper.

Three review rounds made the same point: every behavioural number in this work
runs through an ASR judge's transcript of generated audio, and the paper spends a
paragraph arguing that judge is trustworthy where Whisper is not --- but the
audio is 3.4 GB on a private host, "available on request". A reader who wants to
check a single "exact" against an "undercount" cannot, without rebuilding four
conda environments and regenerating everything. That is replication, not
verification, and it is a much higher bar than the claim needs.

This picks a small stratified sample and encodes it as 16 kHz mono Opus, which is
what the CTC judge consumes anyway, so what ships is what was scored. The strata
span what a sceptical reader would want to hear: each model, low and high `k`,
and each outcome class including the degenerate ones we exclude. Every clip is
paired in `manifest.csv` with its stimulus text, its transcript, the counts, and
the outcome label, so a listener can check the scoring decision directly rather
than take it on trust.

Deliberately small. The point is that a reader can hear a handful of the
judgements the paper rests on without downloading anything; the full set stays
regenerable from the released code and seeds.

Usage:  python scripts/make_audio_sample.py [--per-cell 1] [--max-mb 8]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
AUDIO = Path("/home/kirill/mnt/hdd_6tb_1/icassp_tts/audio")
sys.path.insert(0, str(REPO))
from src.common.population import ABLATIONS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioural", default="data/results/behavioural_ctc.csv")
    ap.add_argument("--out", default="data/audio_sample")
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--max-mb", type=float, default=8.0)
    ap.add_argument("--bitrate", default="24k")
    args = ap.parse_args()

    d = pd.read_csv(args.behavioural)
    d = d[~d.model.isin(ABLATIONS)]
    d = d[d.family.isin(["word_rep", "control_word"])]
    d = d.assign(err=(d.count_a - d.k) / d.k,
                 k_band=lambda x: pd.cut(x.k, [0, 4, 12, 64],
                                         labels=["low", "mid", "high"]))

    # One clip per (model, family, k-band, outcome): the cells a sceptic would
    # sample by hand, including the degenerate ones the analysis discards.
    picked = (d.sort_values(["model", "item_id", "seed"])
                .groupby(["model", "family", "k_band", "outcome"], observed=True)
                .head(args.per_cell))

    out = REPO / args.out
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows, total = [], 0.0
    for r in picked.itertuples():
        src = AUDIO / r.model / f"{r.item_id}_s{r.seed}.wav"
        if not src.exists():
            continue
        dst = clips / f"{r.model}__{r.item_id}__s{r.seed}.opus"
        if not dst.exists():
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                   "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                   "-b:a", args.bitrate, str(dst)]
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                continue
        mb = dst.stat().st_size / 1e6
        if total + mb > args.max_mb:
            dst.unlink(missing_ok=True)
            break
        total += mb
        rows.append(dict(
            clip=f"clips/{dst.name}", model=r.model, item_id=r.item_id,
            seed=r.seed, family=r.family, k=r.k, outcome=r.outcome,
            counted=r.count_a, requested=r.k, rel_error=round(r.err, 4),
            duration_s=round(r.duration_s, 2),
            text=r.text if hasattr(r, "text") else "",
            transcript=r.transcript))

    if not rows:
        raise SystemExit("no clips written; is the audio directory mounted?")

    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    (out / "README.md").write_text(
        "# Audio sample\n\n"
        f"{len(rows)} clips ({total:.1f} MB), stratified over model, item family, "
        "k band and outcome class, including the degenerate outcomes the analysis "
        "excludes.\n\n"
        "16 kHz mono Opus --- the same sample rate and channel count the CTC judge "
        "consumes, so what is shipped is what was scored.\n\n"
        "`manifest.csv` pairs each clip with its stimulus text, the judge's "
        "transcript, the requested and counted repetitions, the relative error and "
        "the outcome label. Listening to a clip and reading its row is enough to "
        "check a scoring decision without regenerating anything.\n\n"
        "This is a sample for verification, not the dataset: the full 3.4 GB is "
        "regenerable from the released code and seeds.\n\n"
        "Regenerate with `python scripts/make_audio_sample.py`.\n")

    print(f"wrote {len(rows)} clips, {total:.1f} MB -> {out}")
    by_outcome = pd.DataFrame(rows).outcome.value_counts().to_dict()
    print(f"outcomes covered: {by_outcome}")


if __name__ == "__main__":
    main()
