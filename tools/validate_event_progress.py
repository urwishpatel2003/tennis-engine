"""
How should in-event progress be shaped — linear, capped, or off?

    python tools/validate_event_progress.py --seasons 2015-2026

The term gives a player Elo credit for matches already won at the current event.
It fires on roughly a quarter of matches, and its dominant case is a seed with a
first-round bye meeting somebody who had to play that round. That is by design,
not an accident: the claim is that a player already match-tight beats one
arriving cold, which is a real and well-known effect.

What has never been checked is the SHAPE. The credit is linear and unbounded, so
a two-match difference is worth 160 Elo and a three-match difference 240 — enough
to invert a matchup on its own. A Cincinnati 2026 round of 32 priced Alexandrova
at 51% instead of 58.6% entirely on this term, which is what prompted the
question.

So the variants are measured rather than argued:

    off        the term does nothing
    cap 1      at most one match of credit (a bye is worth 80, no more)
    cap 2      at most two
    linear     what ships today, unbounded

Fitted on odd-indexed matches and scored on even, because a term this cheap to
tune is exactly the kind that fits noise — head-to-head's optimal multiplier
moved from 0.6 to 2.2 between two archive sizes earlier today.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import conditions as cond  # noqa: E402
from engine import markov  # noqa: E402
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
                        columns=["match_id", "best_of", "completed"])
        for t in tours if (RAW / f"matches_{t}.parquet").exists()
    ], ignore_index=True)
    df = df.merge(ctx, on="match_id", how="left")
    df = df[df["completed"].fillna(False).astype(bool)]
    df = df[(df["w_matches"] >= min_matches) & (df["l_matches"] >= min_matches)]

    c = pd.read_parquet(PROCESSED / "conditions.parquet",
                        columns=["match_id", "player_id", "matches_this_event"])
    for side, idcol in (("w", "winner_id"), ("l", "loser_id")):
        part = c.rename(columns={"player_id": idcol,
                                 "matches_this_event": f"{side}_played_here"})
        df = df.merge(part, on=["match_id", idcol], how="left")
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seasons", default="2015-2026")
    ap.add_argument("--min-matches", type=int, default=20)
    args = ap.parse_args()
    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]

    df = load(tuple(args.tours), int(lo), int(hi), args.min_matches)
    print(f"Evaluating in-event progress on {len(df):,} matches ({lo}-{hi})\n")

    a_is_w = df["winner_id"].to_numpy() < df["loser_id"].to_numpy()
    y = a_is_w.astype(int)
    pick = lambda w, l: np.where(a_is_w, df[w].to_numpy(), df[l].to_numpy())
    gap = pick("w_elo_blend", "l_elo_blend") - pick("l_elo_blend", "w_elo_blend")
    tour, surf = df["tour"].to_numpy(), df["surface"].to_numpy()
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

    played_a = pd.to_numeric(pd.Series(pick("w_played_here", "l_played_here")),
                             errors="coerce").fillna(0).to_numpy()
    played_b = pd.to_numeric(pd.Series(pick("l_played_here", "w_played_here")),
                             errors="coerce").fillna(0).to_numpy()
    fires = played_a != played_b
    print(f"  the term fires on {fires.mean()*100:.0f}% of matches "
          f"({int(fires.sum()):,}), i.e. where the two arrived by different paths")
    diff = played_a - played_b
    for d in sorted(set(np.abs(diff[fires]).astype(int)))[:5]:
        n = int((np.abs(diff) == d).sum())
        print(f"     |difference| of {d} match(es): {n:,}")

    def probs(extra):
        p_elo = 1.0 / (1.0 + np.exp(-(gap + extra) * math.log(10.0) / ELO_SCALE))
        return np.array([blend_logit([e, s], [W_ELO, 1 - W_ELO])
                         for e, s in zip(p_elo, p_sim)])

    tr = np.arange(len(df)) % 2 == 1
    te = ~tr

    print(f"\n  {'variant':<12}{'train':>10}{'HOLDOUT':>10}{'vs off':>10}"
          f"{'acc':>9}   verdict")
    print("  " + "-" * 62)

    def credit(cap):
        return cond.EVENT_PROGRESS_ELO * (np.minimum(played_a, cap)
                                          - np.minimum(played_b, cap))

    results = {}
    off_te = None
    for label, cap in (("off", 0.0), ("cap 1", 1.0), ("cap 2", 2.0),
                       ("cap 3", 3.0), ("linear", 99.0)):
        extra = credit(cap)
        p = probs(extra)
        ll_tr, ll_te = log_loss(p[tr], y[tr]), log_loss(p[te], y[te])
        acc = float(((p[te] > 0.5) == (y[te] == 1)).mean())
        if off_te is None:
            off_te = ll_te
        results[label] = ll_te
        mark = "  <- ships today" if label == "linear" else ""
        print(f"  {label:<12}{ll_tr:>10.5f}{ll_te:>10.5f}"
              f"{off_te - ll_te:>+10.5f}{acc:>9.3f}{mark}")

    best = min(results, key=results.get)
    print(f"\n  best on holdout: {best} ({results[best]:.5f})")
    print(f"  linear - best  : {results['linear'] - results[best]:+.5f}")
    print("\n  A difference under ~0.0002 is inside what a change of sample moves;")
    print("  head-to-head's optimum went from x0.6 to x2.2 between two archives.")
    print("  Prefer the SIMPLER shape unless the gain clearly survives that.")


if __name__ == "__main__":
    main()
