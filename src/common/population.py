#!/usr/bin/env python3
"""One definition of which rows the paper is allowed to report.

Every analysis had its own copy of the exclusion rules and they had already
drifted apart: `analysis/count_error.py` listed one ablation where
`analysis/capacity.py` listed four, so re-scoring the full model list would have
folded the three repetition-penalty arms into the panel without anything
complaining. Exclusions belong in one place where they can be read and audited
together, so they live here and every analysis calls `panel()`.

The five rules, and why each exists:

* **Ablations and baselines are not panel members.** The `xtts2*` and
  `qwen06b*` non-panel arms are panel checkpoints under altered decoding, run to
  sweep a mitigation; pooling them would count XTTS-v2 five times and
  Qwen3-TTS-0.6B eight (three penalties, greedy, three repetition-aware-sampling
  settings, and the checkpoint itself). `vits` is the non-autoregressive baseline, which the theorem makes no
  claim about and which exists precisely to contrast with the panel.
* **Degenerate and empty audio has no count.** Folding it in as a large negative
  error would let a failure to produce speech masquerade as a failure to count;
  it is reported as its own rate instead.
* **Judge-unmeasurable templates.** A word the CTC judge never emits (`okay`
  appears in 0 of 106 items, `hmm` in 0 of 89) cannot measure a decoder, so
  template t2 scores our recogniser's orthography. `analysis/judge_vocab_audit.py`
  derives the list from delivery rates; it is read from that file rather than
  hardcoded, and applies to both item families.
* **Items cut off by our own token budget.** `hit_cap` marks generations that ran
  into the harness limit rather than stopping on their own. Their median relative
  error is -0.44 against -0.08 for the rest: that is our budget truncating the
  audio, not the model failing to count.

* **Off-panel item arms.** The period ladder (`pd_*`) re-renders the published
  repeated and control arms under new ids on the *same* checkpoints, so nothing
  in the `model` column marks it out. See `NON_PANEL_ITEM_PREFIXES`.

`panel()` applies all five and returns the frame plus a record of what each rule
removed, so the counts can be reported instead of quietly vanishing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent

ABLATIONS = {"xtts2norp", "xtts2rp2", "xtts2rp3", "xtts2rp8",
             "qwen06brp10", "qwen06brp15", "qwen06brp30", "qwen06bgreedy",
             # The repetition-aware-sampling arms (VALL-E 2's RAS, tested
             # because the penalty sweep only covers penalty *magnitude*). Same
             # reason as the penalty arms, and one sharper one: `qwen06brasoff`
             # is the *stock* decoder rerun today under a no-op version of the
             # rule, so it duplicates `qwen06b` item for item. Folding it into
             # the panel would count Qwen3-TTS-0.6B twice with the same audio.
             "qwen06bras", "qwen06brasoff", "qwen06brasgr",
             "vits", "vitsdur", "vitsrate", "f5tts", "f5fix",
             # CosyVoice 2 is not an ablation; it is a fourth architecture,
             # added after the panel's numbers were fixed and reported
             # separately as an out-of-sample test of them. It is listed here
             # because `run_pipeline.sh` scores every model with transcripts
             # into the shared table, so without this line a routine rerun
             # would fold it into the panel and move every macro in the paper
             # -- silently, and in the direction that flatters us.
             "cosyvoice2",
             # The Spanish arm, for exactly the same reason as cosyvoice2 and
             # with a sharper edge: it is scored by a *different judge* against
             # a *different stimulus file*, so a routine rerun that folded it in
             # would not merely add rows, it would mix two recognisers' counts
             # inside one macro. It is reported on its own in
             # `analysis/crosslingual_es.py` and nowhere else.
             "xtts2es"}

# Arms that are not panel *items*, keyed by item-id prefix rather than by model.
# `cosyvoice2` and `xtts2es` could be excluded by model key because they are
# separate checkpoints; the period ladder is not -- it is the *panel's own*
# checkpoints rendering a different stimulus file, so nothing in the `model`
# column distinguishes it. What protects the panel today is that
# `score_counts.py` emits rows only for item ids present in the `--stimuli`
# file it was given, so a run over `stimuli.jsonl` cannot pick these up. That is
# a property of one script's control flow, not a stated rule, and it stops
# holding the moment someone scores a merged stimulus file -- which the period
# analysis does, deliberately, to recover the duration fits that need low-k
# anchors. So the rule is stated here too: `pd_*` is the period ladder
# (`data/stimuli/make_stimuli_period.py`), it re-renders the published repeated
# and control arms under new ids, and folding it into the English panel would
# count every cell it replicates twice.
NON_PANEL_ITEM_PREFIXES = ("pd_",)
DEGENERATE = {"empty", "degenerate"}
AUDIT = REPO / "data/results/judge_vocab_audit.json"
META_DIR = Path("/home/kirill/mnt/hdd_6tb_1/icassp_tts/tokens")


def excluded_templates(audit: Path | str | None = None) -> list[str]:
    """Templates whose scored vocabulary the judge cannot transcribe.

    `audit` selects the delivery audit to read. The default is the English
    judge's. It is a parameter because the rule is a statement about *one
    judge on one vocabulary*: the Spanish arm has a different recogniser and a
    different word list, so applying the English exclusion list to it would
    delete rows for a fact established about English orthography.
    """
    p = Path(audit) if audit is not None else AUDIT
    if not p.exists():
        return []
    return list(json.loads(p.read_text()).get("excluded_templates", []))


def cap_flags() -> dict[tuple[str, str, int], bool]:
    """(model, item_id, seed) -> did generation stop on the harness budget?"""
    out: dict[tuple[str, str, int], bool] = {}
    if not META_DIR.exists():
        return out
    for p in META_DIR.glob("*_meta.jsonl"):
        model = p.stem.replace("_meta", "")
        for line in p.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(model, r.get("item_id"), r.get("seed"))] = bool(r.get("hit_cap"))
    return out


def panel(d: pd.DataFrame, *, ablations: bool = False, degenerate: bool = False,
          bad_templates: bool = True, cap_hits: bool = True,
          period_arm: bool = False,
          audit: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Restrict `d` to the reportable population.

    Each flag *keeps* the corresponding rows when set True, so an analysis that
    genuinely wants the ablation arms (the mitigation sweep) or the degenerate
    rate (the outcome table) asks for them explicitly rather than reimplementing
    the filter.
    """
    n0 = len(d)
    drop: dict = {"n_input": n0}

    if not ablations:
        keep = ~d.model.isin(ABLATIONS)
        drop["ablations"] = int((~keep).sum())
        d = d[keep]
    if not period_arm and "item_id" in d.columns:
        keep = ~d.item_id.astype(str).str.startswith(NON_PANEL_ITEM_PREFIXES)
        drop["period_arm"] = int((~keep).sum())
        d = d[keep]
    if not degenerate and "outcome" in d.columns:
        keep = ~d.outcome.isin(DEGENERATE)
        drop["degenerate"] = int((~keep).sum())
        d = d[keep]
    if bad_templates and "template" in d.columns:
        bad = excluded_templates(audit)
        drop["bad_templates"] = int(d.template.isin(bad).sum())
        drop["excluded_templates"] = bad
        d = d[~d.template.isin(bad)]
    if cap_hits and {"model", "item_id", "seed"} <= set(d.columns):
        flags = cap_flags()
        if flags:
            hit = pd.Series(
                [flags.get((r.model, r.item_id, r.seed), False) for r in d.itertuples()],
                index=d.index)
            drop["cap_hits"] = int(hit.sum())
            d = d[~hit]

    drop["n_output"] = len(d)
    return d.copy(), drop


def describe(drop: dict) -> str:
    """One line naming what was removed, for analysis stdout."""
    bits = [f"{k}={v}" for k, v in drop.items()
            if k not in {"n_input", "n_output", "excluded_templates"} and v]
    tm = drop.get("excluded_templates")
    if tm:
        bits.append(f"templates={','.join(tm)}")
    return (f"population {drop['n_input']} -> {drop['n_output']}"
            + (f" (dropped: {', '.join(bits)})" if bits else ""))
