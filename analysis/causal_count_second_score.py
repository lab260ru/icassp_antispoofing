#!/usr/bin/env python3
"""Score the second-checkpoint causal arms, under the verdict pre-committed in
`analysis/causal_count_second.py`.

This file computes nothing new about what counts as a hit: every threshold below
is read off that file's pre-committed docstring, which was written before any
generation was run. What it adds is the arithmetic --- the pairing, the cell
summaries, the disruption index, the dose-response, and the **equivalence
bound**, which is the deliverable for a null. A confidence interval containing
zero says only that the experiment failed to reject; the bound says how large an
effect the experiment could have seen, and it is the bound that licenses the
paper's "not at all".

Reused rather than reimplemented, so that a second checkpoint is scored by the
same instrument as the first: `cell_summary`, `boot_median_ci`, `sign_test`,
`holm`, `partial_spearman` and `audio_flag_cache` come from
`analysis/causal_count_score.py`, and `boot_dist`, `boot_dist_cluster`,
`equivalence` and `reps_at` come from `analysis/equivalence.py`. Only the
manifest loader is duplicated, because the original's is keyed by a hard-coded
`KEYS` dict with no model parameter.

Usage:
  python analysis/causal_count_second_score.py --models qwen06b llasa1b
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
from scipy import stats  # noqa: E402

from analysis.causal_count_score import (  # noqa: E402
    audio_flag_cache, boot_median_ci, cell_summary, holm, partial_spearman,
    sign_test,
)
from analysis.equivalence import (  # noqa: E402
    boot_dist, boot_dist_cluster, equivalence, reps_at,
)
from analysis.causal_count_second import (  # noqa: E402
    CONFIG, DATA_ROOT, PUBLISHED_LLASA1B, load_stimuli,
)
from src.common.score_counts import count_occurrences, count_units, normalise  # noqa: E402

OUT = REPO / "data/results/causal_count_second_checkpoint.json"

P1_DISRUPTION_MAX = 0.5      # unrelated must move it less than half as much
NOISE_FLOOR_FRAC = 0.25      # the post-hoc escape hatch, declared in advance
DEGEN_MAX_POINTS = 15.0      # P3
STOP_MAX_POINTS = 15.0       # P3
BAND_DEGEN_MAX = 25.0        # interpretable band for the sweep
FULL_TRANSFER_FRACTION = 0.5  # the equivalence margin
R_RHO_MIN, R_PARTIAL_MIN, R_SPEC_MIN, R_SIGN_FRAC = 0.4, 0.25, 1.5, 2 / 3


# ---------------------------------------------------------------- loading


def load_rows(key: str, stim: dict) -> list[dict]:
    """The original `load_arm`, with the arm key as a parameter."""
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
        degen = bool(dur < 0.25 or fl.get("rms", 1.0) < 1e-3
                     or fl.get("spectral_flatness", 0.0) > 0.35
                     or not a.get("text", "").strip())
        r.update(count=int(cnt), y=float(math.log1p(cnt)), duration_s=dur,
                 degenerate=degen, family=it["family"],
                 spectral_flatness=fl.get("spectral_flatness", float("nan")))
        rows.append(r)
    return rows


def pct(xs: list[bool]) -> float:
    return 100.0 * float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------- arm A


def analyse_patch(model: str, rows: list[dict]) -> dict:
    cfg = CONFIG[model]
    P, central = cfg["patch_pos"], cfg["central_layer"]
    central_cell = f"crossk|L{central}|P{P}"
    out: dict = dict(model=model, n_rows=len(rows), central_cell=central_cell)

    ref = {(r["recv_item"], r["seed"]): r for r in rows if r["kind"] == "baseline"}
    base = defaultdict(list)
    for r in rows:
        if r["kind"] == "baseline":
            base[(r["template"], r["recv_k"])].append(r["y"])

    # ---- the sanity gate, as recorded in the manifest
    noop = [r for r in rows if r.get("donor_kind") == "self"]
    out["n_noop"] = len(noop)
    out["noop_all_identical"] = bool(noop) and all(
        r.get("noop_identical") is True for r in noop)
    out["noop_failures"] = [r["cond_id"] for r in noop
                            if r.get("noop_identical") is not True]
    deltas_noop = [float(r.get("max_abs_delta") or 0.0) for r in noop]
    out["noop_max_abs_delta"] = max(deltas_noop) if deltas_noop else float("nan")
    out["noop_delta_exactly_zero"] = bool(noop) and all(d == 0.0 for d in deltas_noop)

    refs = list(ref.values())
    rec_refs = [r for r in refs if r["family"] == "word_rep"]
    out["reference_n"] = len(rec_refs)
    out["reference_degenerate_pct"] = pct([r["degenerate"] for r in rec_refs])
    out["reference_caphit_pct"] = pct([bool(r["hit_cap"]) for r in rec_refs])
    out["reference_stop_pct"] = pct([bool(r.get("stopped")) for r in rec_refs])
    out["reference_median_count"] = (float(np.median([r["count"] for r in rec_refs]))
                                     if rec_refs else float("nan"))

    patched = [r for r in rows if r["kind"] == "patch" and r["donor_kind"] != "self"]
    out["patched_n"] = len(patched)
    out["patched_degenerate_pct"] = pct([r["degenerate"] for r in patched])
    out["patched_caphit_pct"] = pct([bool(r["hit_cap"]) for r in patched])
    out["patched_stop_pct"] = pct([bool(r.get("stopped")) for r in patched])

    # ---- cells
    cells: dict[str, dict] = {}
    per_cell: dict[str, list[dict]] = defaultdict(list)
    for r in patched:
        rr = ref.get((r["recv_item"], r["seed"]))
        if rr is None:
            continue
        cell = f"{r['donor_kind']}|L{r['layer']}|P{r['patch_pos']}"
        d = int(np.sign((r["donor_k"] or 0) - r["recv_k"]))
        per_cell[cell].append(dict(
            raw=r["y"] - rr["y"], signed=(r["y"] - rr["y"]) * d, direction=d,
            recv_item=r["recv_item"], template=r["template"],
            recv_k=r["recv_k"], donor_k=r["donor_k"],
            degenerate=r["degenerate"], hit_cap=bool(r["hit_cap"]),
            stopped=bool(r.get("stopped")), ref_stopped=bool(rr.get("stopped")),
            d_dur=r["duration_s"] - rr["duration_s"], ref_count=rr["count"],
            gap=(abs(float(np.median(base[(r["template"], r["donor_k"])]))
                     - float(np.median(base[(r["template"], r["recv_k"])])))
                 if base.get((r["template"], r["donor_k"]))
                 and base.get((r["template"], r["recv_k"])) else float("nan"))))

    pvals = {}
    for cell, items in sorted(per_cell.items()):
        raw = np.array([i["raw"] for i in items])
        signed = np.array([i["signed"] for i in items])
        use = signed if cell.startswith("crossk") else raw
        s = cell_summary(use)
        s.update(cell=cell,
                 raw_median=float(np.median(raw)),
                 median_abs_shift=float(np.median(np.abs(raw))),
                 median_gap=float(np.nanmedian([i["gap"] for i in items])),
                 median_delta_duration_s=float(np.median([i["d_dur"] for i in items])),
                 degenerate_pct=pct([i["degenerate"] for i in items]),
                 caphit_pct=pct([i["hit_cap"] for i in items]),
                 stop_pct=pct([i["stopped"] for i in items]),
                 ref_stop_pct=pct([i["ref_stopped"] for i in items]))
        s["stop_pct_shift"] = s["stop_pct"] - s["ref_stop_pct"]
        if s["median_gap"] == s["median_gap"] and s["median_gap"] > 0:
            s["transfer_ratio_median"] = s["median"] / s["median_gap"]
        if cell.startswith("crossk"):
            up = np.array([i["raw"] for i in items if i["direction"] > 0])
            dn = np.array([i["raw"] for i in items if i["direction"] < 0])
            s["up"] = cell_summary(up) if up.size else None
            s["down"] = cell_summary(dn) if dn.size else None
            s["opposite_raw_directions"] = bool(
                up.size and dn.size and np.median(up) * np.median(dn) < 0)
            pvals[cell] = s["p"]
        cells[cell] = s
    for cell, adj in holm(pvals).items():
        cells[cell]["p_holm"] = adj
    out["cells"] = cells

    # ---- P1: is the arm readable at all?
    cx = cells.get(central_cell)
    un = cells.get(f"unrelated|L{central}|P{P}")
    di = (un["median_abs_shift"] / cx["median_abs_shift"]
          if cx and un and cx["median_abs_shift"] > 0 else float("nan"))
    out["disruption_index"] = di
    out["P1_readable"] = bool(di == di and di < P1_DISRUPTION_MAX)

    full = cx["median_gap"] if cx else float("nan")
    out["full_transfer_logcount"] = full
    worst = max((c["median_abs_shift"] for c in cells.values()), default=float("nan"))
    out["largest_median_abs_shift"] = worst
    out["posthoc_all_cells_at_noise_floor"] = bool(
        full == full and full > 0 and worst < NOISE_FLOOR_FRAC * full)
    out["arm_readable"] = bool(out["P1_readable"]
                               or out["posthoc_all_cells_at_noise_floor"])

    # ---- P2 / P3
    out["P2_signed_shift"] = bool(
        cx and cx["median"] > 0 and cx.get("p_holm", 1.0) < 0.05
        and cx.get("opposite_raw_directions"))
    out["P3_intact"] = bool(
        out["patched_degenerate_pct"] - out["reference_degenerate_pct"]
        <= DEGEN_MAX_POINTS
        and out["reference_stop_pct"] - out["patched_stop_pct"] <= STOP_MAX_POINTS)

    # ---- the equivalence bound: the deliverable for a null
    items = per_cell.get(central_cell, [])
    if items and full == full and full > 0:
        signed = np.array([i["signed"] for i in items])
        clusters = np.array([i["recv_item"] for i in items])
        margin = FULL_TRANSFER_FRACTION * full
        eq = equivalence(boot_dist(signed, "median"), float(np.median(signed)),
                         margin, direction=+1)
        eqc = equivalence(boot_dist_cluster(signed, clusters, "median"),
                          float(np.median(signed)), margin, direction=+1)
        eqc["n_clusters"] = int(len(set(clusters.tolist())))
        med_count = float(np.median([i["ref_count"] for i in items]))
        eq.update(cell=central_cell, n=int(signed.size),
                  unit="log(1+count), signed toward donor",
                  full_transfer_logcount=full,
                  bound_as_fraction_of_transfer=eq["bound"] / full,
                  one_sided_bound_as_fraction_of_transfer=eq["one_sided_bound"] / full,
                  median_reference_count=med_count,
                  bound_repetitions_at_k={str(k): reps_at(eq["bound"], float(k))
                                          for k in (16, 24, 32)},
                  sensitivity_cluster_by_receiver=eqc)
        eq["entitled_to_no"] = bool(eq["equivalent_at_margin"]
                                    and eqc["equivalent_at_margin"])
        out["equivalence"] = eq
    else:
        out["equivalence"] = None

    # ---- verdict, exactly the three-way one Llasa-1B's run used
    gate_ok = out["noop_all_identical"] and out["noop_delta_exactly_zero"]
    junk = [r for r in rows if r["kind"] in ("patch", "baseline")]
    unusable = pct([bool(r["degenerate"] or r["hit_cap"]) for r in junk])
    out["unusable_pct"] = unusable
    eq = out["equivalence"]
    if not gate_ok:
        out["verdict"] = "pipeline inconclusive (no-op gate failed)"
    elif unusable > 50:
        out["verdict"] = ("pipeline inconclusive (more than half the arm is "
                          "degenerate, empty or cap-hit)")
    elif not out["arm_readable"]:
        out["verdict"] = (f"too disruptive to interpret (disruption index "
                          f"{di:.2f} >= {P1_DISRUPTION_MAX}, and cells are not "
                          f"all at the noise floor)")
    elif out["P2_signed_shift"] and out["P3_intact"]:
        out["verdict"] = "causal locus"
    elif eq and eq["entitled_to_no"]:
        out["verdict"] = "readable null"
    else:
        out["verdict"] = "underpowered"
    out["causal_hit"] = out["verdict"] == "causal locus"

    if model == "llasa1b":
        pub = PUBLISHED_LLASA1B
        inside = bool(cx and pub["ci90_lo"] <= cx["median"] <= pub["ci90_hi"])
        out["bridge_U1"] = dict(
            published_cell=pub["cell"], published_median=pub["median"],
            published_ci90=[pub["ci90_lo"], pub["ci90_hi"]],
            bridge_median=(cx["median"] if cx else float("nan")),
            median_inside_published_ci90=inside,
            published_bound=pub["bound"],
            bridge_bound=(eq["bound"] if eq else float("nan")),
            reproduces=bool(inside and out["verdict"] in
                            ("readable null", "causal locus")))
    return out


# ---------------------------------------------------------------- arm R


def analyse_sweep(model: str, rows: list[dict]) -> dict:
    out: dict = dict(model=model, n_rows=len(rows))
    a0 = [r for r in rows if r.get("alpha") == 0.0]
    out["n_alpha0"] = len(a0)
    out["alpha0_all_identical"] = bool(a0) and all(
        r.get("noop_identical") is True for r in a0)
    out["alpha0_failures"] = [r["cond_id"] for r in a0
                              if r.get("noop_identical") is not True]

    fams: dict[str, dict] = {}
    for fam in ("word_rep", "control_word"):
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        by_a = defaultdict(list)
        for r in sub:
            by_a[float(r["alpha"])].append(r)
        stats_a, band = {}, []
        for a in sorted(by_a):
            rs = by_a[a]
            dg = pct([r["degenerate"] for r in rs])
            stats_a[f"{a:+.2f}"] = dict(
                n=len(rs), median_count=float(np.median([r["count"] for r in rs])),
                median_y=float(np.median([r["y"] for r in rs])),
                median_duration_s=float(np.median([r["duration_s"] for r in rs])),
                degenerate_pct=dg,
                stop_pct=pct([bool(r.get("stopped")) for r in rs]),
                caphit_pct=pct([bool(r["hit_cap"]) for r in rs]))
            if dg <= BAND_DEGEN_MAX:
                band.append(a)
        inband = [r for r in sub if float(r["alpha"]) in band]

        by_item = defaultdict(list)
        for r in inband:
            by_item[r["recv_item"]].append(r)
        xs, zs, ds, per_sign = [], [], [], []
        for iid, rs in by_item.items():
            y = np.array([r["y"] for r in rs])
            if y.size < 3 or y.std() == 0:
                continue
            z = (y - y.mean()) / y.std()
            al = np.array([float(r["alpha"]) for r in rs])
            xs += al.tolist()
            zs += z.tolist()
            ds += [math.log(max(r["duration_s"], 1e-3)) for r in rs]
            # A control item whose rendered count never moves gives a constant
            # y and an undefined Spearman. Such an item carries no sign, so it
            # is dropped from the sign-consistency denominator rather than
            # counted as disagreeing -- counting it would make a *perfectly
            # inert* control arm look inconsistent, which is backwards.
            if len(set(al.tolist())) > 1 and len(set(y.tolist())) > 1:
                s = stats.spearmanr(al, y).statistic
                if s == s:
                    per_sign.append(np.sign(s))
        rho, p = (stats.spearmanr(xs, zs) if len(xs) > 3 else (float("nan"),) * 2)[:2]
        partial = partial_spearman(np.array(xs), np.array(zs), np.array(ds)) \
            if len(xs) > 3 else float("nan")
        eff = []
        if band:
            lo_a, hi_a = min(band), max(band)
            for iid, rs in by_item.items():
                lo = [r["y"] for r in rs if float(r["alpha"]) == lo_a]
                hi = [r["y"] for r in rs if float(r["alpha"]) == hi_a]
                if lo and hi:
                    eff.append(float(np.mean(hi)) - float(np.mean(lo)))
        eff = np.array(eff)
        sgn = np.array(per_sign)
        fams[fam] = dict(
            n_items=len(by_item), band=band, by_alpha=stats_a,
            pooled_rho=float(rho), pooled_p=float(p), partial_rho=float(partial),
            sign_consistency=(float(max((sgn > 0).mean(), (sgn < 0).mean()))
                              if sgn.size else float("nan")),
            effect_n=int(eff.size),
            effect_median=float(np.median(eff)) if eff.size else float("nan"),
            effect_abs_median=float(np.median(np.abs(eff))) if eff.size else float("nan"),
            effect_ci=list(boot_median_ci(eff)) if eff.size else [float("nan")] * 2,
            effect_p=sign_test(eff)[2] if eff.size else float("nan"))
    out["families"] = fams

    rep, ctl = fams.get("word_rep"), fams.get("control_word")
    ratio = (rep["effect_abs_median"] / ctl["effect_abs_median"]
             if rep and ctl and ctl["effect_abs_median"] > 0 else float("inf"))
    out["effect_ratio_rep_over_ctl"] = ratio
    out["R1_dose_response"] = bool(
        rep and abs(rep["pooled_rho"]) > R_RHO_MIN and rep["pooled_p"] < 0.05
        and rep["sign_consistency"] >= R_SIGN_FRAC)
    out["R2_not_duration"] = bool(
        rep and abs(rep["partial_rho"]) > R_PARTIAL_MIN
        and np.sign(rep["partial_rho"]) == np.sign(rep["pooled_rho"]))
    out["R3_specific"] = bool(ratio >= R_SPEC_MIN)
    out["band_too_narrow"] = bool(rep is None or len(rep["band"]) < 3)

    if not out["alpha0_all_identical"]:
        out["verdict"] = "pipeline inconclusive (alpha=0 was not bitwise exact)"
    elif out["band_too_narrow"]:
        out["verdict"] = "underpowered (no interpretable band of three alphas)"
    elif out["R1_dose_response"] and out["R2_not_duration"] and out["R3_specific"]:
        out["verdict"] = "causal locus (dose-response, specific, not duration)"
    else:
        out["verdict"] = "readable null (no dose-response along the ridge direction)"
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen06b", "llasa1b"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    stim = load_stimuli()
    res: dict = dict(checkpoints={})
    for m in args.models:
        entry: dict = {}
        pa = load_rows(f"pq_{m}", stim)
        if pa:
            entry["rank1_patch"] = analyse_patch(m, pa)
        rr = load_rows(f"rq_{m}", stim)
        if rr:
            entry["ridge_sweep"] = analyse_sweep(m, rr)
        if entry:
            res["checkpoints"][m] = entry

    for m, e in res["checkpoints"].items():
        p = e.get("rank1_patch")
        print(f"\n=== {m} ===")
        if p:
            print(f"  rank-1 patch: n_rows={p['n_rows']} "
                  f"no-op {p['n_noop']}/{p['n_noop']} identical="
                  f"{p['noop_all_identical']} max|delta|={p['noop_max_abs_delta']:.3g}")
            cx = p["cells"].get(p["central_cell"], {})
            print(f"  {p['central_cell']}: n={cx.get('n')} "
                  f"median {cx.get('median', float('nan')):+.3f} log-count "
                  f"95% CI [{cx.get('ci_lo', float('nan')):+.3f}, "
                  f"{cx.get('ci_hi', float('nan')):+.3f}]")
            print(f"  disruption index {p['disruption_index']:.2f} "
                  f"(P1 readable={p['P1_readable']}, "
                  f"noise floor={p['posthoc_all_cells_at_noise_floor']})")
            print(f"  degenerate patched {p['patched_degenerate_pct']:.1f}% vs "
                  f"reference {p['reference_degenerate_pct']:.1f}%; "
                  f"stop {p['patched_stop_pct']:.1f}% vs "
                  f"{p['reference_stop_pct']:.1f}%")
            eq = p.get("equivalence")
            if eq:
                print(f"  EQUIVALENCE: excluded any shift above {eq['bound']:.3f} "
                      f"log-count = {100 * eq['bound_as_fraction_of_transfer']:.0f}% "
                      f"of a full transfer ({eq['full_transfer_logcount']:.3f}) = "
                      f"{eq['bound_repetitions_at_k']['24']:+.1f} repetitions at "
                      f"k=24; TOST p={eq['tost_p']:.4f}; cluster bound "
                      f"{eq['sensitivity_cluster_by_receiver']['bound']:.3f}")
            print(f"  VERDICT: {p['verdict']}")
            if p.get("bridge_U1"):
                b = p["bridge_U1"]
                print(f"  U1 bridge: median {b['bridge_median']:+.3f} inside the "
                      f"published 90% CI {b['published_ci90']}: "
                      f"{b['median_inside_published_ci90']} -> reproduces="
                      f"{b['reproduces']}")
            if p["causal_hit"]:
                print("  ** CONTRADICTS-LLASA1B: the rank-1 patch moved the "
                      "rendered count on this checkpoint **")
        s = e.get("ridge_sweep")
        if s:
            rep = s["families"].get("word_rep", {})
            print(f"  ridge sweep: n_rows={s['n_rows']} alpha0 exact="
                  f"{s['alpha0_all_identical']} band={rep.get('band')}")
            print(f"    pooled rho {rep.get('pooled_rho', float('nan')):+.3f} "
                  f"(p={rep.get('pooled_p', float('nan')):.3f}), partial "
                  f"{rep.get('partial_rho', float('nan')):+.3f}, "
                  f"rep/ctl effect ratio "
                  f"{s['effect_ratio_rep_over_ctl']:.2f}")
            print(f"    VERDICT: {s['verdict']}")

    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
