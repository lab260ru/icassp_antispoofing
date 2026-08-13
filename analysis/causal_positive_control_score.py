#!/usr/bin/env python3
r"""Score the positive control, under the verdict pre-committed in
`analysis/causal_positive_control.py`.

Nothing here decides what counts as a hit. Every threshold below is read off
that file's docstring, which was written before a single generation of this
experiment existed:

  DETECTABLE           S1 >= 2 x S1_0  or  S3 >= S3_0 + 25 points, where the
                       subscript 0 is the diffseed cell measured in this run and
                       S1 is summarised by its mean (see the AMENDMENT in
                       `causal_positive_control.py`: the floor's *median* S1 is
                       identically zero, because a sampler re-roll moves the
                       timing and not the words, and a rule reading "at least
                       twice zero" would call every cell detectable).
  DIRECTION-APPROPRIATE  median S4 > 0, sign test surviving Holm across that
                       direction's alpha levels at p<0.05, 90% bootstrap
                       interval on the median excluding 0, the cell
                       interpretable (<=25% of pairs degenerate or with a
                       duration outside [0.5, 2]x the reference's -- repair R1),
                       and, for the count direction, the published P2 condition
                       that the higher-k and lower-k donors move the count in
                       opposite raw directions (repair R2). Both repairs are
                       documented in `causal_positive_control.py`; both can only
                       make a hit harder.
  VERDICT              PATCH IS LIVE / PATCH IS INERT / INTERMEDIATE (I-a/I-b/I-c).

Reused rather than reimplemented so the control is scored by the instrument
that scored the thing it controls: `audio_flag_cache` from
`analysis/causal_count_score.py`, `boot_dist`, `equivalence` and `holm` from
`analysis/equivalence.py` / `causal_count_score.py`, `normalise`,
`count_occurrences` and `count_units` from `src/common/score_counts.py`.

Usage:
  python analysis/causal_positive_control_score.py --models qwen17b llasa8b
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

from analysis.causal_count_score import audio_flag_cache, holm, sign_test  # noqa: E402
from analysis.equivalence import ALPHA, boot_dist  # noqa: E402
from analysis.causal_count_second import CONFIG, DATA_ROOT, load_stimuli  # noqa: E402
from analysis.causal_positive_control import ALPHAS, PROTOCOL  # noqa: E402
from src.common.score_counts import count_occurrences, count_units, normalise  # noqa: E402

OUT = REPO / "data/results/causal_positive_control.json"

DETECT_S1_FACTOR = 2.0        # S1 >= 2x the diffseed floor
DETECT_S3_POINTS = 25.0       # or degeneracy 25 points above the floor
BAND_DEGEN_MAX = 25.0         # interpretable band, as in the ridge sweep
LADDER_INERT_MIN_ALPHA = 4.0  # "PATCH IS INERT" needs the first hit at alpha>=4


# ---------------------------------------------------------------- text


def lev(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance. Small sequences; the simple DP is fine."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(hyp: list[str], ref: list[str]) -> float:
    return lev(hyp, ref) / max(1, len(ref))


def carrier_score(toks: list[str], recv_words: list[str],
                  donor_words: list[str]) -> float:
    """How much closer the transcript is to the donor's sentence than the
    receiver's, in normalised word-edit distance. Positive = toward the donor."""
    return wer(toks, recv_words) - wer(toks, donor_words)


# ---------------------------------------------------------------- loading


def load_rows(model: str, stim: dict) -> list[dict]:
    key = f"pc_{model}"
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
                 degenerate=degen, tokens=toks, family=it["family"])
        rows.append(r)
    return rows


def pct(xs: list[bool]) -> float:
    return 100.0 * float(np.mean(xs)) if xs else float("nan")


def med(xs) -> float:
    return float(np.median(xs)) if len(xs) else float("nan")


# ------------------------------------------------- the published arm vs a re-roll


PUBLISHED_ARM = {"qwen17b": ("pq_qwen17b", "baseline", 12, 32),
                 "llasa8b": ("rs_llasa8b", "resume", 16, 128),
                 "qwen06b": ("pq_qwen06b", "baseline", 12, 32),
                 "llasa1b": ("pq_llasa1b", "baseline", 7, 128)}


def published_arm_vs_reroll(model: str, stim: dict) -> dict | None:
    """Is the published patch's footprint distinguishable from resampling?

    This costs no GPU: it reads the arm the paper already reports and asks a
    question that arm was never asked. Its `diffseed` cell takes the donor
    states from the *same receiver item at a different seed* --- same text, same
    $k$, same carrier --- so that write carries no information about anything
    and whatever it does to the output is what a re-roll of the sampler does.
    Its `crossk` cell is the paper's informative write.

    If the two cells are indistinguishable on every readout --- rendered count,
    duration, and the rendered words --- then the informative write's entire
    measured effect on the output is a re-roll, which is the reviewer's
    objection stated as a measurement rather than a worry. Reported as
    Mann--Whitney U on the per-pair absolute shifts, two-sided, with the medians
    beside them.
    """
    if model not in PUBLISHED_ARM:
        return None
    key, refkind, L, P = PUBLISHED_ARM[model]
    man = REPO / "data/results" / f"causal_{key}_manifest.jsonl"
    if not man.exists():
        return None
    asr = {}
    ap = DATA_ROOT / "asr_ctc" / f"{key}.jsonl"
    if not ap.exists():
        return None
    for line in ap.open():
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
        r.update(count=int(cnt), y=float(math.log1p(cnt)), duration_s=dur,
                 tokens=toks)
        rows.append(r)
    ref = {(r["recv_item"], r["seed"]): r for r in rows if r["kind"] == refkind}
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["kind"] != "patch" or r.get("layer") != L or r["donor_kind"] == "self":
            continue
        rr = ref.get((r["recv_item"], r["seed"]))
        if rr is None:
            continue
        cells[r["donor_kind"]].append(dict(
            abs_count=abs(r["y"] - rr["y"]),
            abs_logdur=abs(math.log(max(r["duration_s"], 1e-3))
                           - math.log(max(rr["duration_s"], 1e-3))),
            wer=wer(r["tokens"], rr["tokens"])))
    if "crossk" not in cells or "diffseed" not in cells:
        return None
    out = dict(model=model, arm=key, layer=L, patch_pos=P,
               reference=refkind, metrics={})
    for m in ("abs_count", "abs_logdur", "wer"):
        x = np.array([i[m] for i in cells["crossk"]], dtype=float)
        z = np.array([i[m] for i in cells["diffseed"]], dtype=float)
        try:
            p = float(stats.mannwhitneyu(x, z, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        out["metrics"][m] = dict(
            crossk_n=int(x.size), crossk_median=float(np.median(x)),
            crossk_mean=float(x.mean()),
            diffseed_n=int(z.size), diffseed_median=float(np.median(z)),
            diffseed_mean=float(z.mean()),
            ratio_of_means=float(x.mean() / z.mean()) if z.mean() > 0 else float("nan"),
            mannwhitney_p=p)
    out["informative_write_indistinguishable_from_reroll"] = bool(
        all(v["mannwhitney_p"] != v["mannwhitney_p"] or v["mannwhitney_p"] >= 0.05
            for v in out["metrics"].values()))
    return out


# ---------------------------------------------------------------- scoring


V2_MIN_LOTO = 0.9             # a control direction has to be decodable


def analyse(model: str, rows: list[dict], stim: dict,
            dirs_meta: dict | None = None) -> dict:
    cfg = CONFIG[model]
    L, P = cfg["central_layer"], cfg["patch_pos"]
    out: dict = dict(model=model, protocol=PROTOCOL[model], layer=int(L),
                     patch_pos=int(P), n_rows=len(rows))

    ref = {(r["recv_item"], r["seed"]): r for r in rows if r["kind"] == "reference"}
    base = {(r["recv_item"], r["seed"]): r for r in rows if r["kind"] == "baseline"}

    # ---- the no-op gate as recorded during generation
    noop = [r for r in rows if r.get("donor_kind") == "self"]
    out["n_noop"] = len(noop)
    out["noop_all_identical"] = bool(noop) and all(
        r.get("noop_identical") is True for r in noop)
    out["noop_delta_exactly_zero"] = bool(noop) and all(
        float(r.get("max_abs_delta") or 0.0) == 0.0 for r in noop)

    # ---- pairs
    cells: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for r in rows:
        if r["kind"] != "patch" or r["donor_kind"] == "self":
            continue
        rr = ref.get((r["recv_item"], r["seed"]))
        if rr is None:
            continue
        d = dict(recv_item=r["recv_item"], template=r["template"],
                 recv_k=r["recv_k"], donor_k=r["donor_k"],
                 degenerate=bool(r["degenerate"]),
                 hit_cap=bool(r["hit_cap"]),
                 # R1: the `degenerate` flag does not fire on a collapse -- a
                 # 0.8 s stub in place of a 13 s utterance is a clean recording
                 # of almost nothing. Destruction is duration-relative.
                 destroyed=bool(r["degenerate"] or not (
                     0.5 <= (max(r["duration_s"], 1e-3)
                             / max(rr["duration_s"], 1e-3)) <= 2.0)),
                 duration_ratio=float(max(r["duration_s"], 1e-3)
                                      / max(rr["duration_s"], 1e-3)),
                 rel_magnitude=r.get("write_rel_magnitude"),
                 write_median_abs_delta=r.get("write_median_abs_delta"),
                 s1=wer(r["tokens"], rr["tokens"]),
                 s2=abs(math.log(max(r["duration_s"], 1e-3))
                        - math.log(max(rr["duration_s"], 1e-3))))
        if r["direction"] == "count":
            sgn = int(np.sign((r["donor_k"] or 0) - r["recv_k"]))
            d["s4"] = (r["y"] - rr["y"]) * sgn if sgn else float("nan")
            d["raw"] = r["y"] - rr["y"]
            d["donor_dir"] = sgn
            db = base.get((r["donor_item"], r["donor_seed"]))
            rb = base.get((r["recv_item"], r["seed"]))
            d["full_transfer"] = (abs(db["y"] - rb["y"])
                                  if db is not None and rb is not None
                                  else float("nan"))
        elif r["direction"] == "carrier":
            rw = normalise(stim[r["recv_item"]]["text"])
            dw = normalise(stim[r["donor_item"]]["text"])
            db = base.get((r["donor_item"], r["donor_seed"]))
            rb = base.get((r["recv_item"], r["seed"]))
            d["s4"] = (carrier_score(r["tokens"], rw, dw)
                       - carrier_score(rr["tokens"], rw, dw))
            d["full_transfer"] = (
                carrier_score(db["tokens"], rw, dw)
                - carrier_score(rb["tokens"], rw, dw)
                if db is not None and rb is not None else float("nan"))
        else:
            d["s4"] = float("nan")
            d["full_transfer"] = float("nan")
        cells[(r["direction"], r["donor_kind"], float(r["alpha"]))].append(d)

    # ---- the sampler re-roll floor
    floor = cells.get(("count", "diffseed", 1.0), [])
    floor_s1 = np.array([i["s1"] for i in floor], dtype=float)
    s1_0 = float(floor_s1.mean()) if floor_s1.size else float("nan")
    s2_0 = med([i["s2"] for i in floor])
    s3_0 = pct([i["destroyed"] for i in floor])
    out["floor"] = dict(cell="count|diffseed|a1", n=len(floor), S1_mean=s1_0,
                        S1_median=med(floor_s1),
                        S3_degenerate_pct_unrepaired=pct(
                            [i["degenerate"] for i in floor]),
                        S1_frac_any_word_change=(float((floor_s1 > 0).mean())
                                                 if floor_s1.size else float("nan")),
                        S2=s2_0, S3_destroyed_pct=s3_0,
                        V3_floor_usable=bool(s3_0 == s3_0 and s3_0 <= 25.0))

    # ---- cell summaries
    summ: dict[str, dict] = {}
    pvals: dict[str, dict[str, float]] = defaultdict(dict)
    for (dname, kind, a), items in sorted(cells.items()):
        name = f"{dname}|{kind}|a{a:g}"
        s1 = np.array([i["s1"] for i in items])
        s2 = np.array([i["s2"] for i in items])
        dg = pct([i["degenerate"] for i in items])
        dest = pct([i["destroyed"] for i in items])
        s4 = np.array([i["s4"] for i in items], dtype=float)
        s4 = s4[~np.isnan(s4)]
        ft = np.array([i["full_transfer"] for i in items], dtype=float)
        mw = float("nan")
        if floor_s1.size and s1.size:
            try:
                mw = float(stats.mannwhitneyu(s1, floor_s1,
                                              alternative="greater").pvalue)
            except ValueError:
                mw = float("nan")
        entry = dict(direction=dname, donor_kind=kind, alpha=float(a),
                     n=len(items), S1=float(s1.mean()),
                     S1_median=float(np.median(s1)),
                     S1_frac_any_word_change=float((s1 > 0).mean()),
                     S1_mannwhitney_p_vs_floor=mw,
                     S2=float(np.median(s2)), S3_degenerate_pct=dg,
                     caphit_pct=pct([i["hit_cap"] for i in items]),
                     rel_magnitude=med([i["rel_magnitude"] for i in items]),
                     write_median_abs_delta=med(
                         [i["write_median_abs_delta"] for i in items]),
                     full_transfer=(float(np.nanmedian(ft))
                                    if ft.size and not np.all(np.isnan(ft))
                                    else float("nan")),
                     destroyed_pct=dest,
                     median_duration_ratio=med([i["duration_ratio"]
                                                for i in items]),
                     interpretable=bool(dest <= BAND_DEGEN_MAX))
        entry["S1_over_floor"] = ((entry["S1"] / s1_0) if s1_0 > 0
                                  else float("nan"))
        # R1 applies to both uses of "destroyed", not only to interpretability:
        # a run that collapsed to a stub has obviously changed, and calling that
        # undetectable would be as wrong as calling it a steer.
        entry["S3_over_floor_points"] = entry["destroyed_pct"] - s3_0
        if s1_0 > 0:
            s1_route = entry["S1_over_floor"] >= DETECT_S1_FACTOR
        else:
            # The declared fallback: with a floor of exactly zero the ratio is
            # undefined, so detection is decided by a one-sided Mann-Whitney U
            # of the cell's S1 against the floor's.
            s1_route = bool(mw == mw and mw < 0.05)
        entry["S1_route_used"] = "ratio" if s1_0 > 0 else "mannwhitney"
        entry["detectable"] = bool(
            s1_route or entry["S3_over_floor_points"] >= DETECT_S3_POINTS)
        if s4.size:
            b = boot_dist(s4, "median")
            lo, hi = (float(np.percentile(b, 100 * ALPHA)),
                      float(np.percentile(b, 100 * (1 - ALPHA))))
            n_pos, n_neg, p = sign_test(s4)
            # Can the pre-committed bar fire AT ALL at this cell's effective n?
            # `sign_test` drops ties, and Holm runs across the direction's five
            # alpha levels, so below n_eff = 8 no agreement level whatsoever --
            # not even unanimity -- reaches p < 0.05 after correction. A cell
            # where this is False did not fail the test; it could not take it,
            # and reporting the two as though they were the same thing would be
            # dishonest. The bootstrap interval below is the statement that
            # survives, and is the one to quote.
            n_eff = int(n_pos + n_neg)
            min_agree = None
            for kk in range(n_eff // 2 + 1, n_eff + 1):
                if stats.binomtest(kk, n_eff, 0.5).pvalue * len(ALPHAS) < 0.05:
                    min_agree = kk
                    break
            entry.update(S4_n_pos=n_pos, S4_n_neg=n_neg,
                         S4_n_ties=int(s4.size - n_pos - n_neg),
                         S4_n_effective=n_eff,
                         S4_min_agreement_that_could_reject=min_agree,
                         S4_sign_test_can_reject=bool(min_agree is not None))
            entry.update(S4_median=float(np.median(s4)), S4_n=int(s4.size),
                         S4_ci90=[lo, hi], S4_p=float(p),
                         S4_ci90_excludes_zero=bool(lo > 0 or hi < 0),
                         S4_as_fraction_of_transfer=(
                             float(np.median(s4)) / entry["full_transfer"]
                             if entry["full_transfer"] == entry["full_transfer"]
                             and entry["full_transfer"] > 0 else float("nan")),
                         # The statement that does not depend on the sign test:
                         # how much of a full donor-to-receiver transfer the
                         # write is confined to at 90%. This is the number to
                         # quote for an underpowered cell.
                         S4_ci90_as_fraction_of_transfer=(
                             [lo / entry["full_transfer"],
                              hi / entry["full_transfer"]]
                             if entry["full_transfer"] == entry["full_transfer"]
                             and entry["full_transfer"] > 0
                             else [float("nan")] * 2),
                         S4_bound_as_fraction_of_transfer=(
                             max(abs(lo), abs(hi)) / entry["full_transfer"]
                             if entry["full_transfer"] == entry["full_transfer"]
                             and entry["full_transfer"] > 0 else float("nan")))
            if kind != "diffseed":
                pvals[dname][name] = float(p)
        if dname == "count" and kind == "crossk":
            up = np.array([i["raw"] for i in items if i.get("donor_dir", 0) > 0])
            dn = np.array([i["raw"] for i in items if i.get("donor_dir", 0) < 0])
            entry["opposite_raw_directions"] = bool(
                up.size and dn.size
                and float(np.median(up)) * float(np.median(dn)) < 0)
        summ[name] = entry

    for dname, pv in pvals.items():
        for name, adj in holm(pv).items():
            summ[name]["S4_p_holm"] = adj
    for name, e in summ.items():
        if "S4_median" not in e:
            e["direction_appropriate"] = False
            continue
        ok_p2 = (e.get("opposite_raw_directions", False)
                 if e["direction"] == "count" else True)
        e["direction_appropriate"] = bool(
            e["S4_median"] > 0 and e.get("S4_p_holm", 1.0) < 0.05
            and e["S4_ci90_excludes_zero"] and e["interpretable"] and ok_p2)
    out["cells"] = summ

    # ---- the verdict, exactly as pre-committed
    def cell(dname: str, a: float) -> dict | None:
        for k, e in summ.items():
            if (e["direction"] == dname and e["alpha"] == a
                    and e["donor_kind"] != "diffseed"):
                return e
        return None

    dirs = sorted({e["direction"] for e in summ.values()
                   if e["donor_kind"] != "diffseed"})
    first_detect = {}
    first_appropriate = {}
    for dn in dirs:
        det = [a for a in ALPHAS if (cell(dn, a) or {}).get("detectable")]
        app = [a for a in ALPHAS if (cell(dn, a) or {}).get("direction_appropriate")]
        first_detect[dn] = min(det) if det else None
        first_appropriate[dn] = min(app) if app else None
    out["first_detectable_alpha"] = first_detect
    out["first_direction_appropriate_alpha"] = first_appropriate

    any_detect_a1 = any(first_detect[d] == 1.0 for d in dirs)
    any_detect_a2 = any(first_detect[d] is not None and first_detect[d] <= 2.0
                        for d in dirs)
    carrier_a1 = bool((cell("carrier", 1.0) or {}).get("direction_appropriate"))
    count_a1 = bool((cell("count", 1.0) or {}).get("direction_appropriate"))
    app_any = {d: first_appropriate[d] for d in dirs
               if first_appropriate[d] is not None}

    # ---- V2: is the carrier direction decodable enough to be a control?
    # Pre-committed as a disqualifier, not as something to tune afterwards. A
    # carrier probe below the bar still has its cells reported -- they are
    # evidence about the write -- but it cannot carry the PATCH IS LIVE verdict,
    # because "the decoder ignored a direction the state barely encodes" is not
    # the control the argument needs.
    carrier_acc = {k: v["loto_sign_accuracy"]
                   for k, v in ((dirs_meta or {}).get("carrier") or {}).items()}
    out["V2_carrier_loto"] = carrier_acc
    out["V2_carrier_qualified"] = bool(
        carrier_acc and min(carrier_acc.values()) >= V2_MIN_LOTO)
    if not out["V2_carrier_qualified"]:
        carrier_a1 = False

    gate_ok = out["noop_all_identical"] and out["noop_delta_exactly_zero"]
    if not gate_ok:
        out["verdict"] = "pipeline inconclusive (no-op gate failed)"
    elif not out["floor"]["V3_floor_usable"]:
        out["verdict"] = ("pipeline inconclusive (V3: the diffseed floor is "
                          "itself degenerate)")
    elif carrier_a1 and not count_a1:
        out["verdict"] = "PATCH IS LIVE (null survives)"
    elif (not any_detect_a1
          and all(first_detect[d] is None or first_detect[d] >= LADDER_INERT_MIN_ALPHA
                  for d in dirs)):
        out["verdict"] = "PATCH IS INERT (null is uninformative)"
    elif any_detect_a1 and not app_any:
        out["verdict"] = ("INTERMEDIATE I-a (write is potent at alpha=1 but no "
                          "direction was shown to steer)")
    elif not any_detect_a1 and any_detect_a2:
        out["verdict"] = ("INTERMEDIATE I-b (alpha=1 sits just below the "
                          "decoder's sensitivity threshold)")
    elif app_any and min(app_any.values()) >= LADDER_INERT_MIN_ALPHA:
        out["verdict"] = (f"INTERMEDIATE I-c (a probe direction steers this "
                          f"decoder, but only at alpha>="
                          f"{min(app_any.values()):g})")
    else:
        out["verdict"] = "INTERMEDIATE (unclassified; see cells)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen17b", "llasa8b"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    stim = load_stimuli()
    res: dict = dict(
        precommitment="analysis/causal_positive_control.py docstring",
        thresholds=dict(detect_S1_factor=DETECT_S1_FACTOR,
                        detect_S3_points=DETECT_S3_POINTS,
                        band_degenerate_max_pct=BAND_DEGEN_MAX,
                        inert_min_alpha=LADDER_INERT_MIN_ALPHA,
                        alphas=ALPHAS),
        checkpoints={})
    for m in args.models:
        rows = load_rows(m, stim)
        if not rows:
            print(f"[{m}] no scored rows yet")
            continue
        d = REPO / "data/results" / f"causal_pc_{m}_directions.json"
        dirs_meta = json.loads(d.read_text()) if d.exists() else None
        e = analyse(m, rows, stim, dirs_meta)
        if dirs_meta:
            e["directions"] = dirs_meta
        e["published_arm_vs_reroll"] = published_arm_vs_reroll(m, stim)
        res["checkpoints"][m] = e

        print(f"\n=== {m} ({e['protocol']} protocol, L{e['layer']}, "
              f"P{e['patch_pos']}) ===")
        f = e["floor"]
        print(f"  re-roll floor (diffseed, n={f['n']}): "
              f"S1 mean={f['S1_mean']:.4f} median={f['S1_median']:.4f} "
              f"any-word-change={100 * f['S1_frac_any_word_change']:.0f}%  "
              f"S2={f['S2']:.3f}  destroyed={f['S3_destroyed_pct']:.1f}%")
        print(f"  {'cell':28s} {'n':>3s} {'S1':>6s} {'xfloor':>6s} "
              f"{'|dlogT|':>7s} {'dstr%':>5s} {'relmag':>6s} {'S4':>8s} "
              f"{'p_holm':>7s}  det appr")
        for name, c in sorted(e["cells"].items(),
                              key=lambda kv: (kv[1]["direction"], kv[1]["alpha"])):
            print(f"  {name:28s} {c['n']:3d} {c['S1']:6.3f} "
                  f"{c['S1_over_floor']:6.2f} {c['S2']:7.3f} "
                  f"{c['destroyed_pct']:5.1f} {c['rel_magnitude']:6.3f} "
                  f"{c.get('S4_median', float('nan')):+8.3f} "
                  f"{c.get('S4_p_holm', float('nan')):7.3f}  "
                  f"{'Y' if c['detectable'] else '.'}   "
                  f"{'Y' if c['direction_appropriate'] else '.'}")
        under = [n for n, c in e["cells"].items()
                 if c.get("S4_sign_test_can_reject") is False]
        if under:
            print(f"  UNDERPOWERED (the sign test could not have rejected at "
                  f"this cell's effective n, whatever the data showed): "
                  f"{sorted(under)}")
            print(f"    -> for these, quote the bootstrap bound as a fraction "
                  f"of a full transfer, not the sign test")
        print(f"  first DETECTABLE alpha:            "
              f"{e['first_detectable_alpha']}")
        print(f"  first DIRECTION-APPROPRIATE alpha: "
              f"{e['first_direction_appropriate_alpha']}")
        if not e["V2_carrier_qualified"]:
            print(f"  V2: the carrier control DID NOT QUALIFY on this "
                  f"checkpoint (LOTO sign accuracy "
                  f"{ {k: round(v, 2) for k, v in e['V2_carrier_loto'].items()} }, "
                  f"bar {V2_MIN_LOTO}); its cells are reported but cannot carry "
                  f"a PATCH IS LIVE verdict")
        pr = e.get("published_arm_vs_reroll")
        if pr:
            print(f"  -- the PUBLISHED arm ({pr['arm']}, L{pr['layer']}): is the "
                  f"informative write distinguishable from a sampler re-roll?")
            for k, v in pr["metrics"].items():
                print(f"     {k:11s} crossk median {v['crossk_median']:.3f} "
                      f"(n={v['crossk_n']})  diffseed median "
                      f"{v['diffseed_median']:.3f} (n={v['diffseed_n']})  "
                      f"mean ratio {v['ratio_of_means']:.2f}  "
                      f"Mann-Whitney p={v['mannwhitney_p']:.3f}")
            print(f"     indistinguishable from a re-roll on every readout: "
                  f"{pr['informative_write_indistinguishable_from_reroll']}")
        print(f"  VERDICT: {e['verdict']}")

    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
