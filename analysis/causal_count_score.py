#!/usr/bin/env python3
"""Score the causal interventions of `analysis/causal_count.py` and return the
pre-committed verdict.

The interpretation this script applies was fixed in that file's docstring before
any audio existed; nothing here is free to choose a threshold after the fact. The
job of this file is arithmetic and honesty, in that order:

  1. run the **sanity gate**. Every no-op patch must have reproduced its resume
     reference bitwise and every $\\alpha=0$ steering run must have reproduced the
     free baseline bitwise. If not, the verdict is `pipeline inconclusive` and no
     effect in either arm is reported, however large.
  2. score the rendered count with the project's own CTC estimator, pair every
     intervened run against the reference that shares its prefix and its random
     stream, and summarise the paired shift on $\\log(1+\\text{count})$ --- the
     scale forced on us by a readout whose single-seed values at one $k$ span
     3 to 476.
  3. carry duration, speech-token count, cap-hit rate and degenerate-audio rate
     next to every cell, so a "count effect" that is really a "make it babble
     longer" effect is visible in the same table rather than in a later review.
  4. report at the least favourable reading: the strict Holm-corrected test, the
     control arms that could explain the effect away, and the transfer ratio,
     which asks not "is there a shift" but "did the receiver render the donor's
     count".

Usage:
  python analysis/causal_count_score.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import stats  # noqa: E402

from src.common.asr_transcribe import audio_flags  # noqa: E402
from src.common.score_counts import (  # noqa: E402
    count_occurrences, count_units, normalise,
)

DATA_ROOT = Path("/home/kirill/mnt/hdd_6tb_1/icassp_tts")
KEYS = {"A": "patch1b", "B": "steer1b"}
DEGEN_TOL = 15.0        # percentage points, pre-committed
SPECIFICITY = 1.5       # rep/ctl effect ratio below which the direction is generic
RHO_MIN = 0.5


# ---------------------------------------------------------------- loading


def audio_flag_cache(key: str) -> dict:
    """Degeneracy flags per stem, computed once and cached beside the audio."""
    cache_p = REPO / "data/results" / f"causal_{key}_flags.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    aud = DATA_ROOT / "audio" / key
    todo = [f for f in sorted(aud.glob("*.wav")) if f.stem not in cache]
    for f in todo:
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        d = audio_flags(wav, sr)
        d["duration_s"] = float(wav.size / sr)
        cache[f.stem] = d
    if todo:
        cache_p.write_text(json.dumps(cache))
    return cache


def load_arm(arm: str, stim: dict) -> list[dict]:
    key = KEYS[arm]
    man = REPO / "data/results" / f"causal_{key}_manifest.jsonl"
    if not man.exists():
        return []
    asr = {}
    p = DATA_ROOT / "asr_ctc" / f"{key}.jsonl"
    if p.exists():
        for line in p.open():
            try:
                r = json.loads(line)
                asr[r["stem"]] = r
            except Exception:  # noqa: BLE001
                pass
    flags = audio_flag_cache(key)

    rows = []
    for line in man.open():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        a = asr.get(r["stem"])
        if a is None:
            continue
        it = stim[r["recv_item"]]
        toks = normalise(a.get("text", ""))
        units = it.get("boundary_units")
        if units and len(set(units)) > 1:
            cnt = count_units(toks, units)
        elif units:
            cnt = count_occurrences(toks, units[0])
        else:
            cnt = count_occurrences(toks, it["target_unit"])
        fl = flags.get(r["stem"], {})
        dur = float(a.get("duration_s", fl.get("duration_s", 0.0)))
        degen = bool(
            dur < 0.25 or fl.get("rms", 1.0) < 1e-3
            or fl.get("spectral_flatness", 0.0) > 0.35
            or not a.get("text", "").strip())
        r.update(count=int(cnt), y=float(math.log1p(cnt)), duration_s=dur,
                 degenerate=degen, family=it["family"],
                 spectral_flatness=fl.get("spectral_flatness", float("nan")))
        rows.append(r)
    return rows


# ---------------------------------------------------------------- stats


def boot_median_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, size=(n, x.size), replace=True), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sign_test(x: np.ndarray) -> tuple[int, int, float]:
    """Two-sided sign test on a paired difference; ties are dropped."""
    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, float("nan")
    return pos, neg, float(stats.binomtest(pos, n, 0.5).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, keyed as the input."""
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda t: t[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    for k, v in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def cell_summary(deltas: np.ndarray) -> dict:
    lo, hi = boot_median_ci(deltas)
    pos, neg, p = sign_test(deltas)
    return dict(n=int(deltas.size), median=float(np.median(deltas)) if deltas.size else float("nan"),
                ci_lo=lo, ci_hi=hi, n_pos=pos, n_neg=neg, p=p)


# ---------------------------------------------------------------- arm A


def analyse_a(rows: list[dict]) -> dict:
    res: dict = {}
    noop = [r for r in rows if r.get("donor_kind") == "self"]
    res["n_noop"] = len(noop)
    res["noop_all_identical"] = bool(noop) and all(r.get("noop_identical") is True
                                                   for r in noop)
    res["noop_failures"] = [r["cond_id"] for r in noop if r.get("noop_identical") is not True]

    ref = {(r["recv_item"], r["seed"], r["patch_pos"]): r
           for r in rows if r["kind"] == "resume"}
    base = defaultdict(list)
    for r in rows:
        if r["kind"] == "baseline":
            base[(r["template"], r["recv_k"])].append(r["y"])

    # health of the arm as a whole
    for name, sel in (("reference", [r for r in rows if r["kind"] == "resume"]),
                      ("patched", [r for r in rows if r["kind"] == "patch"
                                   and r["donor_kind"] != "self"])):
        if sel:
            res[f"{name}_degenerate_pct"] = 100 * float(np.mean([r["degenerate"] for r in sel]))
            res[f"{name}_caphit_pct"] = 100 * float(np.mean([bool(r["hit_cap"]) for r in sel]))
            res[f"{name}_median_count"] = float(np.median([r["count"] for r in sel]))
            res[f"{name}_median_duration_s"] = float(np.median([r["duration_s"] for r in sel]))
            res[f"{name}_stop_pct"] = 100 * float(np.mean([not bool(r["hit_cap"]) for r in sel]))
            res[f"{name}_n"] = len(sel)

    cells: dict[str, dict] = {}
    pairs_by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["kind"] != "patch" or r["donor_kind"] == "self":
            continue
        ref_r = ref.get((r["recv_item"], r["seed"], r["patch_pos"]))
        if ref_r is None:
            continue
        # Only the cross-k donors carry a direction. The control, same-seed and
        # unrelated donors are matched or meaningless in k, so signing their
        # shift by a k difference would invent a prediction they do not make.
        direction = (int(np.sign((r["donor_k"] or 0) - r["recv_k"]))
                     if r["donor_kind"] == "crossk" else 0)
        delta = r["y"] - ref_r["y"]
        g_hi = base.get((r["template"], r["donor_k"]), [])
        g_lo = base.get((r["template"], r["recv_k"]), [])
        gap = (float(np.median(g_hi)) - float(np.median(g_lo))) if g_hi and g_lo else float("nan")
        pairs_by_cell[f"{r['donor_kind']}|L{r['layer']}|P{r['patch_pos']}"].append(
            dict(delta=delta, direction=direction, gap=gap,
                 signed=delta * direction if direction else delta,
                 clean=not (r["degenerate"] or ref_r["degenerate"]
                            or r["hit_cap"] or ref_r["hit_cap"]),
                 d_dur=r["duration_s"] - ref_r["duration_s"],
                 degen=r["degenerate"], cap=bool(r["hit_cap"]),
                 # the one readout a token budget cannot censor: did the decoder
                 # decide to stop, and did the patch change that decision?
                 stopped=not bool(r["hit_cap"]),
                 ref_stopped=not bool(ref_r["hit_cap"])))

    for cell, ps in sorted(pairs_by_cell.items()):
        signed = np.array([p["signed"] for p in ps])
        clean = np.array([p["signed"] for p in ps if p["clean"]])
        raw = np.array([p["delta"] for p in ps])
        gaps = np.array([abs(p["gap"]) for p in ps if p["gap"] == p["gap"]])
        tr = np.array([p["signed"] / abs(p["gap"]) for p in ps
                       if p["gap"] == p["gap"] and abs(p["gap"]) > 0.1])
        e = dict(cell=cell, **cell_summary(signed))
        e["clean"] = cell_summary(clean)
        e["raw_median_delta"] = float(np.median(raw)) if raw.size else float("nan")
        e["median_gap"] = float(np.median(gaps)) if gaps.size else float("nan")
        e["transfer_ratio_median"] = float(np.median(tr)) if tr.size else float("nan")
        e["median_delta_duration_s"] = float(np.median([p["d_dur"] for p in ps]))
        e["degenerate_pct"] = 100 * float(np.mean([p["degen"] for p in ps]))
        e["caphit_pct"] = 100 * float(np.mean([p["cap"] for p in ps]))
        e["stop_pct"] = 100 * float(np.mean([p["stopped"] for p in ps]))
        e["ref_stop_pct"] = 100 * float(np.mean([p["ref_stopped"] for p in ps]))
        e["stop_pct_shift"] = e["stop_pct"] - e["ref_stop_pct"]
        # McNemar on the paired stop decision: of the pairs that disagree, how
        # often did the patch turn a runaway into a stop rather than the reverse?
        b = sum(1 for p in ps if p["stopped"] and not p["ref_stopped"])
        c = sum(1 for p in ps if p["ref_stopped"] and not p["stopped"])
        e["stop_gained"], e["stop_lost"] = int(b), int(c)
        e["stop_p"] = (float(stats.binomtest(b, b + c, 0.5).pvalue)
                       if b + c else float("nan"))
        e["median_abs_delta"] = float(np.median(np.abs(raw))) if raw.size else float("nan")
        # split by donor direction, because "up" and "down" donors are separate
        # predictions and a cell that only moves one way is not a count effect
        for lab, sgn in (("up", 1), ("down", -1)):
            sub = np.array([p["delta"] for p in ps if p["direction"] == sgn])
            if sub.size:
                e[lab] = cell_summary(sub)
        cells[cell] = e

    crossk = {c: e for c, e in cells.items() if c.startswith("crossk|")}
    adj = holm({c: e["p"] for c, e in crossk.items()})
    for c, a in adj.items():
        cells[c]["p_holm"] = a
    res["cells"] = cells

    # ---- the pre-committed decision
    hits = []
    for c, e in crossk.items():
        ctl_cell = cells.get("diffseed|L7|P128")
        # Opposite raw directions for higher-k and lower-k donors. The donor
        # pool is unbalanced 2:1 toward lower-k, so "signed shift toward the
        # donor" is satisfiable by a patch that merely shortens the utterance;
        # requiring the two donor directions to push the count opposite ways is
        # not.
        up, dn = e.get("up", {}).get("median"), e.get("down", {}).get("median")
        opp = (up is not None and dn is not None and up > 0 > dn)
        e["A1_opposite_directions"] = bool(opp)
        a1 = (e["p_holm"] < 0.05) and (e["median"] > 0) and opp
        a2 = (ctl_cell is None
              or abs(ctl_cell["median"]) < 0.5 * abs(e["median"]))
        a3 = (e["clean"]["median"] == e["clean"]["median"]
              and e["clean"]["median"] > 0
              and e["degenerate_pct"] - res.get("reference_degenerate_pct", 0) <= DEGEN_TOL)
        if a1 and a2 and a3:
            hits.append(c)
        cells[c]["A1_signed_shift"] = bool(a1)
        cells[c]["A2_not_generic_state"] = bool(a2)
        cells[c]["A3_survives_cleaning"] = bool(a3)
    res["hits"] = sorted(hits)
    res["causal_hit"] = bool(hits)

    # ---- the disruption check, which outranks both other verdicts
    #
    # A null and a broken instrument look identical in a table of medians. The
    # unrelated donor is states that *cannot* encode the receiver's count; if
    # splicing those in moves the rendered count about as much as an informative
    # donor does, the protocol is perturbing generation, and no comparison
    # between donors means anything.
    idx = {}
    for L, P in {(int(c.split("|L")[1].split("|")[0]),
                  int(c.split("|P")[1])) for c in crossk}:
        u = cells.get(f"unrelated|L{L}|P{P}")
        x = cells.get(f"crossk|L{L}|P{P}")
        if u and x and x["median_abs_delta"] == x["median_abs_delta"] \
                and x["median_abs_delta"] > 1e-9:
            idx[f"L{L}|P{P}"] = u["median_abs_delta"] / x["median_abs_delta"]
    res["disruption_index"] = idx
    d1 = bool(idx) and max(idx.values()) >= 0.5
    d2 = (res.get("patched_degenerate_pct", 0.0)
          - res.get("reference_degenerate_pct", 0.0)) > DEGEN_TOL
    res["D1_unrelated_donor_as_disruptive"] = bool(d1)
    res["D2_degeneracy_inflated"] = bool(d2)
    res["too_disruptive"] = bool(d1 or d2)
    return res


# ---------------------------------------------------------------- arm B


def analyse_b(rows: list[dict]) -> dict:
    res: dict = {}
    zero = [r for r in rows if r.get("alpha") == 0.0]
    res["n_alpha0"] = len(zero)
    res["alpha0_all_identical"] = bool(zero) and all(r.get("noop_identical") is True
                                                     for r in zero)
    res["alpha0_failures"] = [r["cond_id"] for r in zero
                              if r.get("noop_identical") is not True]

    def arm(fam: str, direction: str, layer: int) -> list[dict]:
        return [r for r in rows if r["family"] == fam and r["direction"] == direction
                and r["layer"] == layer]

    out: dict = {}
    for direction, layer in sorted({(r["direction"], r["layer"]) for r in rows}):
        tag = f"{direction}|L{layer}"
        entry: dict = {}
        for fam in ("word_rep", "control_word"):
            sel = arm(fam, direction, layer)
            if not sel:
                continue
            by_item: dict[str, list[tuple[float, dict]]] = defaultdict(list)
            for r in sel:
                by_item[r["recv_item"]].append((r["alpha"], r))
            rhos, effects, dur_eff = [], [], []
            za, zy = [], []
            for iid, pts in by_item.items():
                pts.sort()
                al = np.array([a for a, _ in pts])
                yy = np.array([r["y"] for _, r in pts])
                dd = np.array([r["duration_s"] for _, r in pts])
                if len(set(al.tolist())) >= 3:
                    rhos.append(float(stats.spearmanr(al, yy).statistic))
                    s = yy.std()
                    za += al.tolist()
                    zy += ((yy - yy.mean()) / (s if s > 1e-9 else 1.0)).tolist()
                amax, amin = al.max(), al.min()
                effects.append(float(yy[al == amax].mean() - yy[al == amin].mean()))
                dur_eff.append(float(dd[al == amax].mean() - dd[al == amin].mean()))
            rhos_a = np.array([r for r in rhos if r == r])
            pooled = (stats.spearmanr(za, zy) if len(za) > 5 else None)
            entry[fam] = dict(
                n_items=len(by_item), alphas=sorted({r["alpha"] for r in sel}),
                per_item_rho_median=float(np.median(rhos_a)) if rhos_a.size else float("nan"),
                sign_consistency=(float(np.mean(np.sign(rhos_a) == np.sign(np.median(rhos_a))))
                                  if rhos_a.size else float("nan")),
                pooled_rho=float(pooled.statistic) if pooled is not None else float("nan"),
                pooled_p=float(pooled.pvalue) if pooled is not None else float("nan"),
                effect_median=float(np.median(effects)) if effects else float("nan"),
                effect_ci=boot_median_ci(np.array(effects)),
                duration_effect_median=float(np.median(dur_eff)) if dur_eff else float("nan"),
                by_alpha={
                    f"{a:+.1f}": dict(
                        n=int(sum(1 for r in sel if r["alpha"] == a)),
                        median_count=float(np.median([r["count"] for r in sel if r["alpha"] == a])),
                        median_y=float(np.median([r["y"] for r in sel if r["alpha"] == a])),
                        median_duration_s=float(np.median([r["duration_s"] for r in sel if r["alpha"] == a])),
                        degenerate_pct=100 * float(np.mean([r["degenerate"] for r in sel if r["alpha"] == a])),
                        caphit_pct=100 * float(np.mean([bool(r["hit_cap"]) for r in sel if r["alpha"] == a])),
                        stop_pct=100 * float(np.mean([not bool(r["hit_cap"]) for r in sel if r["alpha"] == a])),
                        median_speech_tokens=float(np.median([r["n_speech_tokens"] for r in sel if r["alpha"] == a])),
                    ) for a in sorted({r["alpha"] for r in sel})},
            )
        out[tag] = entry

    res["arms"] = out

    # ---- the pre-committed decision, on the primary direction only
    prim = out.get("dim|L13", {})
    rep, ctl = prim.get("word_rep"), prim.get("control_word")
    if rep and ctl:
        b1 = (abs(rep["pooled_rho"]) > RHO_MIN and rep["pooled_p"] < 0.05
              and rep["sign_consistency"] >= 2 / 3)
        big = [v for a, v in rep["by_alpha"].items() if abs(float(a)) >= 2.0]
        b0 = rep["by_alpha"].get("+0.0", {}).get("degenerate_pct", 0.0)
        b2 = all(v["degenerate_pct"] - b0 <= DEGEN_TOL for v in big) if big else False
        er, ec = abs(rep["effect_median"]), abs(ctl["effect_median"])
        b3 = er > ec and (ec < 1e-9 or er / ec >= SPECIFICITY)
        res["B1_dose_response"] = bool(b1)
        res["B2_not_degenerate"] = bool(b2)
        res["B3_specific"] = bool(b3)
        res["effect_ratio_rep_over_ctl"] = float(er / ec) if ec > 1e-9 else float("inf")
        res["causal_hit"] = bool(b1 and b2 and b3)
    else:
        res["causal_hit"] = False
    return res


# ---------------------------------------------------------------- report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/results/causal_count.json"))
    args = ap.parse_args()

    stim = {json.loads(l)["item_id"]: json.loads(l)
            for l in (REPO / "data/stimuli/stimuli.jsonl").open()}
    A = load_arm("A", stim)
    B = load_arm("B", stim)
    res: dict = {"n_rows_A": len(A), "n_rows_B": len(B)}
    if A:
        res["armA"] = analyse_a(A)
    if B:
        res["armB"] = analyse_b(B)

    gate_a = res.get("armA", {}).get("noop_all_identical")
    gate_b = res.get("armB", {}).get("alpha0_all_identical")
    gate = (gate_a is not False) and (gate_b is not False) and (gate_a or gate_b)
    res["sanity_gate_pass"] = bool(gate)

    print("=" * 78)
    print("SANITY GATE")
    if A:
        print(f"  no-op patch bitwise-identical to its resume reference: "
              f"{res['armA']['noop_all_identical']} "
              f"({res['armA']['n_noop']} runs)")
        for c in res["armA"]["noop_failures"][:5]:
            print(f"    FAILED: {c}")
    if B:
        print(f"  alpha=0 steering bitwise-identical to baseline:        "
              f"{res['armB']['alpha0_all_identical']} "
              f"({res['armB']['n_alpha0']} runs)")
        for c in res["armB"]["alpha0_failures"][:5]:
            print(f"    FAILED: {c}")

    if A:
        a = res["armA"]
        print("\n" + "=" * 78)
        print(f"ARM A -- activation patching  (n={a.get('patched_n')} patched runs, "
              f"{a.get('reference_n')} references)")
        for nm in ("reference", "patched"):
            print(f"  {nm:9s} arm: n={a.get(nm + '_n')}, "
                  f"median count {a.get(nm + '_median_count')}, "
                  f"median duration {a.get(nm + '_median_duration_s', float('nan')):.1f}s, "
                  f"degenerate {a.get(nm + '_degenerate_pct', float('nan')):.1f}%, "
                  f"cap-hit {a.get(nm + '_caphit_pct', float('nan')):.1f}%, "
                  f"stopped {a.get(nm + '_stop_pct', float('nan')):.1f}%")
        hdr = (f"{'cell':>23s} {'n':>4s} {'med d':>7s} {'95% CI':>17s} "
               f"{'p':>6s} {'holm':>6s} {'trans':>6s} {'|d|':>5s} "
               f"{'up':>6s} {'down':>6s} {'ddur':>6s} {'cap%':>5s} "
               f"{'stop%':>6s} {'dstop':>6s}")
        print("\n" + hdr)
        for c, e in sorted(res["armA"]["cells"].items()):
            up = e.get("up", {}).get("median", float("nan"))
            dn = e.get("down", {}).get("median", float("nan"))
            print(f"{c:>23s} {e['n']:4d} {e['median']:+7.3f} "
                  + f"[{e['ci_lo']:+.2f},{e['ci_hi']:+.2f}]".rjust(17)
                  + f" {e['p']:6.3f} {e.get('p_holm', float('nan')):6.3f} "
                  f"{e['transfer_ratio_median']:+6.2f} "
                  f"{e['median_abs_delta']:5.2f} {up:+6.2f} {dn:+6.2f} "
                  f"{e['median_delta_duration_s']:+6.1f} {e['caphit_pct']:5.1f} "
                  f"{e['stop_pct']:6.1f} {e['stop_pct_shift']:+6.1f}")
        print(f"\n  disruption index (unrelated / cross-k median |shift|): "
              f"{ {k: round(v, 2) for k, v in res['armA'].get('disruption_index', {}).items()} }")
        print("\n  d = paired shift in log(1+count) against the resume reference, "
              "signed toward the\n  donor's k for cross-k donors and unsigned for "
              "the three control donors, whose\n  k carries no prediction. trans = "
              "that shift over the baseline gap between\n  donor-k and receiver-k "
              "items: 1.0 would mean the receiver rendered the donor's\n  count. "
              "|d| = median absolute shift, the quantity the unrelated-donor "
              "disruption\n  index is built from. up/down are the raw shifts for "
              "higher-k and lower-k\n  donors: a real count effect is positive on "
              "one and negative on the other.\n  ddur, cap%, stop% and dstop are "
              "the artifact watch -- an intervention that only\n  lengthens the "
              "audio, or only breaks it, moves the count too. stop% is the\n  "
              "fraction that emitted end-of-speech on their own, the readout the "
              "token\n  budget cannot censor; dstop is that minus the reference "
              "arm's.")
        print(f"\n  cells clearing all three pre-committed conditions: "
              f"{res['armA']['hits'] or 'none'}")

    if B:
        b = res["armB"]
        print("\n" + "=" * 78)
        print("ARM B -- steering")
        for tag, entry in sorted(b["arms"].items()):
            print(f"\n  direction {tag}")
            for fam, e in sorted(entry.items()):
                print(f"    {fam:13s} n_items={e['n_items']:2d}  "
                      f"pooled rho={e['pooled_rho']:+.3f} (p={e['pooled_p']:.3f})  "
                      f"per-item rho med={e['per_item_rho_median']:+.3f}  "
                      f"effect={e['effect_median']:+.3f} log-count")
                for al, v in sorted(e["by_alpha"].items(), key=lambda t: float(t[0])):
                    print(f"      a={al:>5s}  count={v['median_count']:7.1f}  "
                          f"dur={v['median_duration_s']:6.2f}s  "
                          f"tok={v['median_speech_tokens']:7.0f}  "
                          f"degen={v['degenerate_pct']:5.1f}%  "
                          f"stop={v['stop_pct']:5.1f}%  n={v['n']}")
        if "B1_dose_response" in b:
            print(f"\n  B1 dose-response {b['B1_dose_response']}   "
                  f"B2 not-degenerate {b['B2_not_degenerate']}   "
                  f"B3 specific {b['B3_specific']} "
                  f"(rep/ctl effect ratio {b['effect_ratio_rep_over_ctl']:.2f})")

    # ---- verdict
    if not gate:
        verdict = ("pipeline inconclusive: the sanity gate failed, so no "
                   "intervention result in this file is reportable")
    elif res.get("armA", {}).get("too_disruptive"):
        a = res["armA"]
        why = []
        if a["D1_unrelated_donor_as_disruptive"]:
            why.append("states from an unrelated item move the rendered count "
                       f"{100*max(a['disruption_index'].values()):.0f}% as much "
                       "as a donor that actually differs in k")
        if a["D2_degeneracy_inflated"]:
            why.append("patching inflates the degenerate-audio rate beyond the "
                       "pre-committed tolerance")
        verdict = ("intervention too disruptive to interpret: " + "; ".join(why)
                   + ". The patching protocol perturbs generation rather than "
                     "transplanting a count, so neither a causal hit nor a null "
                     "can be read off it")
    else:
        ha = res.get("armA", {}).get("causal_hit", False)
        hb = res.get("armB", {}).get("causal_hit", False)
        if ha or hb:
            arms = ", ".join(x for x, h in (("patching", ha), ("steering", hb)) if h)
            verdict = (f"causal locus found: the rendered count follows the "
                       f"intervention in the {arms} arm")
        else:
            verdict = ("count decodable but not causally used: the sanity "
                       "controls pass, so the intervention machinery works, and "
                       "neither patching donor states nor steering along the "
                       "count direction moves the rendered count toward the "
                       "donor's k at any locus tested")
    res["verdict"] = verdict
    print("\n" + "=" * 78)
    print("VERDICT: " + verdict)

    Path(args.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
