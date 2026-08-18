"""
Measure the adjustment layer, term by term.

    python tools/validate_adjustments.py --seasons 2015-2026

`backtest.py` reports one blended number. This reports the CONTRIBUTION of each
adjustment separately, and the multiplier that would be optimal for it — which a
single headline figure cannot show, however honest that figure is.

The two now share engine/replay.py, so what is measured here is exactly what the
backtest scores. That was not always true: the backtest omitted this layer
entirely, and head-to-head once turned a single 1-0 meeting into a +47 Elo swing
— larger than most rating gaps — while backtesting *identically* to the fixed
version, because the backtest could not see the term.

This script rebuilds each adjustment from the same frozen, leak-free tables the
engine uses and asks two questions per adjustment:

    1. Does it reduce out-of-sample log loss at its current magnitude?
    2. What multiplier on it would be optimal — is the hand-tuned size right,
       and is the SIGN even right?

A multiplier near 0 means the adjustment is doing nothing. Below 0 means it is
actively backwards.

Leak-free by construction
-------------------------
Guaranteed by engine/replay.py rather than re-argued here: conditions.parquet is
pre-match by construction, and the head-to-head walk consults only meetings
already played. `tests/test_no_leakage.py` section 5 asserts it directly.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import markov  # noqa: E402
from engine import matchups as mu  # noqa: E402
from engine import replay  # noqa: E402
from engine.predict import W_ELO, blend_logit  # noqa: E402
from engine.schema import ELO_SCALE, PROCESSED, RAW  # noqa: E402
from engine.serve_return import point_probabilities  # noqa: E402


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load(tours, lo, hi, min_matches):
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    sr = pd.read_parquet(PROCESSED / "serve_return.parquet")
    drop = ["tour", "tourney_date", "season", "surface", "winner_id", "loser_id"]
    df = r.merge(sr.drop(columns=drop), on="match_id", how="inner")
    df = df[df["tour"].isin(tours) & df["season"].between(lo, hi)]

    ctx = pd.concat([
        pd.read_parquet(RAW / f"matches_{t}.parquet",
                        columns=["match_id", "best_of", "completed", "surface",
                                 "tourney_name", "winner_name", "loser_name"])
        for t in tours if (RAW / f"matches_{t}.parquet").exists()
    ], ignore_index=True)
    df = df.merge(ctx.drop(columns=["surface"]), on="match_id", how="left")
    df = df[df["completed"].fillna(False).astype(bool)]
    df = df[(df["w_matches"] >= min_matches) & (df["l_matches"] >= min_matches)]
    return df.sort_values("tourney_date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seasons", default="2015-2026")
    ap.add_argument("--min-matches", type=int, default=20)
    args = ap.parse_args()
    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]

    df = load(tuple(args.tours), int(lo), int(hi), args.min_matches)
    print(f"Evaluating the adjustment layer on {len(df):,} matches "
          f"({lo}-{hi})\n")

    # Orientation independent of the result.
    a_is_w = df["winner_id"].to_numpy() < df["loser_id"].to_numpy()
    y = a_is_w.astype(int)
    pick = lambda w, l: np.where(a_is_w, df[w].to_numpy(), df[l].to_numpy())

    gap = pick("w_elo_blend", "l_elo_blend") - pick("l_elo_blend", "w_elo_blend")
    tour = df["tour"].to_numpy()
    surf = df["surface"].to_numpy()
    bo = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int).to_numpy()

    a_s = pick("w_serve_excess", "l_serve_excess")
    b_s = pick("l_serve_excess", "w_serve_excess")
    a_r = pick("w_return_excess", "l_return_excess")
    b_r = pick("l_return_excess", "w_return_excess")
    p_sim = np.array([
        markov.match_win_prob(*point_probabilities(a_s[i], a_r[i], b_s[i], b_r[i],
                                                   tour[i], surf[i]), int(bo[i]))
        for i in range(len(df))
    ])

    # ── each adjustment, in Elo points, oriented onto player A ────────────────
    adj = {}

    # handedness
    attrs = pd.concat([
        pd.read_parquet(RAW / f"players_{t}.parquet", columns=["player_id", "hand", "height"])
        .assign(tour=t) for t in args.tours if (RAW / f"players_{t}.parquet").exists()
    ], ignore_index=True)
    hmap = {(r.tour, int(r.player_id)): r.hand for r in attrs.itertuples(index=False)}
    a_id = pick("winner_id", "loser_id").astype(int)
    b_id = pick("loser_id", "winner_id").astype(int)
    adj["handedness"] = np.array([
        mu.handedness_elo_delta(hmap.get((tour[i], a_id[i])), hmap.get((tour[i], b_id[i])))
        for i in range(len(df))
    ])

    # conditions and head-to-head — both reconstructed by engine/replay.py, the
    # same code backtest.py now uses. This file used to carry its own copy of
    # each; two implementations of a leak-sensitive rebuild is one too many, and
    # the whole point of measuring the layer is that the measurement and the
    # thing being measured agree.
    adj["conditions"] = replay.conditions_elo(
        df["match_id"].to_numpy(), a_id, b_id)
    adj["head_to_head"] = replay.h2h_elo(
        pd.to_datetime(df["tourney_date"]).to_list(),
        a_id, b_id, surf, gap, y)

    # ── evaluate ──────────────────────────────────────────────────────────────
    def probs(extra_elo):
        p_elo = 1.0 / (1.0 + np.exp(-(gap + extra_elo) * math.log(10.0) / ELO_SCALE))
        return np.array([blend_logit([e, s], [W_ELO, 1 - W_ELO])
                         for e, s in zip(p_elo, p_sim)])

    base_ll = log_loss(probs(np.zeros(len(df))), y)
    print(f"  baseline (elo + serve/return, no adjustments)   logloss {base_ll:.5f}\n")
    print(f"  {'adjustment':<16}{'applied':>10}{'best mult':>11}{'best':>10}{'verdict':>22}")
    print("  " + "─" * 69)

    results = {}
    for name, delta in adj.items():
        if not np.any(delta):
            print(f"  {name:<16}{'—':>10}{'—':>11}{'—':>10}{'never fires':>22}")
            continue
        applied = log_loss(probs(delta), y)
        best_m, best_ll = 0.0, base_ll
        for m in np.arange(-1.0, 3.01, 0.1):
            ll = log_loss(probs(delta * m), y)
            if ll < best_ll:
                best_m, best_ll = float(m), ll
        if best_m <= 0.05:
            verdict = "NO VALUE - remove"
        elif abs(best_m - 1.0) <= 0.35:
            verdict = "about right"
        elif best_m < 1.0:
            verdict = f"too strong (x{best_m:.1f})"
        else:
            verdict = f"too weak (x{best_m:.1f})"
        print(f"  {name:<16}{applied:>10.5f}{best_m:>11.1f}{best_ll:>10.5f}{verdict:>22}")
        results[name] = (applied, best_m, best_ll)

    total = sum(adj.values())
    print(f"\n  all together                                  {log_loss(probs(total), y):.5f}"
          f"   vs baseline {base_ll:.5f}")
    tuned = sum(d * results.get(n, (0, 0, 0))[1] for n, d in adj.items())
    print(f"  all at their best multipliers                 {log_loss(probs(tuned), y):.5f}")
    print("\n  A best multiplier at or below 0 means the term is noise or backwards.")

    # ── height / style ────────────────────────────────────────────────────────
    # The one adjustment the loop above cannot reach. Everything else moves the
    # ELO gap, so it can be added as a delta and re-scored cheaply. Height does
    # not: it shifts the serve and return EXCESSES, which means the Markov chain
    # has to be re-solved from scratch for every candidate multiplier. That cost
    # is why it stayed unmeasured while the rest of the layer was audited — and
    # "expensive to measure" is not evidence that a constant is right.
    #
    # Height is also missing for ~15% of players; those rows get no shift, so
    # this measures the term where it actually fires.
    print("\n  HEIGHT / STYLE  (shifts serve/return, so the chain is re-solved)")
    ht = {(r.tour, int(r.player_id)): r.height for r in attrs.itertuples(index=False)}
    shifts = np.array([
        [*mu.height_style_delta(ht.get((tour[i], a_id[i])), tour[i], surf[i]),
         *mu.height_style_delta(ht.get((tour[i], b_id[i])), tour[i], surf[i])]
        for i in range(len(df))
    ])
    fires = np.any(shifts != 0.0, axis=1)
    print(f"  fires on {fires.sum():,} of {len(df):,} matches "
          f"({fires.mean()*100:.0f}% — the rest have no recorded height)")

    print(f"  {'multiplier':<14}{'logloss':>10}{'vs baseline':>14}")
    print("  " + "─" * 38)
    best_m, best_ll = None, None
    for m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        ps = np.array([
            markov.match_win_prob(
                *point_probabilities(a_s[i] + m * shifts[i, 0],
                                     a_r[i] + m * shifts[i, 1],
                                     b_s[i] + m * shifts[i, 2],
                                     b_r[i] + m * shifts[i, 3],
                                     tour[i], surf[i]), int(bo[i]))
            for i in range(len(df))
        ])
        p_elo = 1.0 / (1.0 + np.exp(-gap * math.log(10.0) / ELO_SCALE))
        ll = log_loss(np.array([blend_logit([e, s], [W_ELO, 1 - W_ELO])
                                for e, s in zip(p_elo, ps)]), y)
        if best_ll is None or ll < best_ll:
            best_m, best_ll = m, ll
        tag = "  <- shipped" if m == 1.0 else ""
        print(f"  x{m:<13.1f}{ll:>10.5f}{'':>14}{tag}")
    print(f"\n  best multiplier x{best_m:.1f} at {best_ll:.5f}. "
          f"x0.0 is the term switched OFF —")
    print("  if that wins, the height shift is noise and should be retired like "
          "handedness was.")


if __name__ == "__main__":
    main()
