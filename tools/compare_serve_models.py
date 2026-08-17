"""
A/B the aggregate serve model against the first/second-serve split.

    python tools/compare_serve_models.py --seasons 2018-2026

Both models are scored on IDENTICAL matches from the same frozen pre-match
tables, so the only thing that differs is how service points are modelled:

    aggregate  p = base_spw + serve_excess_server - return_excess_returner
    split      p = first_in x (base_1st + first_won_s - ret_first_r)
                 + (1 - first_in) x (base_2nd + second_won_s - ret_second_r)

The point of this script is to be able to say "the split helps by X" or "it does
not help" with a number rather than an intuition. A richer model that does not
improve out-of-sample log loss is not an improvement, it is just more code.
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
from engine.predict import W_ELO, blend_logit  # noqa: E402
from engine.schema import ELO_SCALE, PROCESSED, RAW  # noqa: E402
from engine.serve_return import point_probabilities  # noqa: E402
from engine.serve_split import METRICS, point_prob_split  # noqa: E402


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load(tours, seasons=None, min_matches=20):
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    sr = pd.read_parquet(PROCESSED / "serve_return.parquet")
    sp = pd.read_parquet(PROCESSED / "serve_split.parquet")

    drop = ["tour", "tourney_date", "season", "surface", "winner_id", "loser_id"]
    df = r.merge(sr.drop(columns=drop), on="match_id", how="inner", suffixes=("", "_sr"))
    df = df.merge(sp.drop(columns=drop), on="match_id", how="inner", suffixes=("", "_sp"))
    df = df[df["tour"].isin(tours)]
    if seasons:
        lo, hi = seasons
        df = df[df["season"].between(lo, hi)]

    ctx = []
    for t in tours:
        p = RAW / f"matches_{t}.parquet"
        if p.exists():
            m = pd.read_parquet(p, columns=["match_id", "best_of", "completed"])
            ctx.append(m)
    df = df.merge(pd.concat(ctx, ignore_index=True), on="match_id", how="left")
    df = df[df["completed"].fillna(False).astype(bool)]
    df = df[(df["w_matches"] >= min_matches) & (df["l_matches"] >= min_matches)]
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seasons", default="2018-2026")
    ap.add_argument("--min-matches", type=int, default=20)
    args = ap.parse_args()
    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]

    df = load(tuple(args.tours), (int(lo), int(hi)), args.min_matches)
    print(f"Comparing on {len(df):,} matches ({lo}-{hi}, both players "
          f">={args.min_matches} prior matches)\n")

    # Orient A = lower player id, independent of the result.
    a_is_w = df["winner_id"].to_numpy() < df["loser_id"].to_numpy()
    y = a_is_w.astype(int)
    pick = lambda wc, lc: np.where(a_is_w, df[wc].to_numpy(), df[lc].to_numpy())

    gap = pick("w_elo_blend", "l_elo_blend") - pick("l_elo_blend", "w_elo_blend")
    p_elo = 1.0 / (1.0 + np.exp(-gap * math.log(10.0) / ELO_SCALE))

    tour = df["tour"].to_numpy()
    surf = df["surface"].to_numpy()
    bo = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int).to_numpy()

    # ── aggregate model ───────────────────────────────────────────────────────
    a_serve = pick("w_serve_excess", "l_serve_excess")
    b_serve = pick("l_serve_excess", "w_serve_excess")
    a_ret = pick("w_return_excess", "l_return_excess")
    b_ret = pick("l_return_excess", "w_return_excess")
    p_agg = np.empty(len(df))
    for i in range(len(df)):
        pa, pb = point_probabilities(a_serve[i], a_ret[i], b_serve[i], b_ret[i],
                                     tour[i], surf[i])
        p_agg[i] = markov.match_win_prob(pa, pb, int(bo[i]))

    # ── split model ───────────────────────────────────────────────────────────
    A = {m: pick(f"w_{m}", f"l_{m}") for m in METRICS}
    B = {m: pick(f"l_{m}", f"w_{m}") for m in METRICS}
    p_split = np.empty(len(df))
    for i in range(len(df)):
        sa = {m: A[m][i] for m in METRICS}
        sb = {m: B[m][i] for m in METRICS}
        pa = point_prob_split(sa, sb, tour[i], surf[i])
        pb = point_prob_split(sb, sa, tour[i], surf[i])
        p_split[i] = markov.match_win_prob(pa, pb, int(bo[i]))

    blend = lambda p: np.array([blend_logit([e, s], [W_ELO, 1 - W_ELO])
                                for e, s in zip(p_elo, p)])
    rows = [
        ("Elo only", p_elo),
        ("serve/return AGGREGATE", p_agg),
        ("serve/return SPLIT", p_split),
        ("blend: elo + aggregate", blend(p_agg)),
        ("blend: elo + SPLIT", blend(p_split)),
    ]
    print(f"  {'model':<26}{'logloss':>10}{'brier':>10}{'accuracy':>11}")
    for name, p in rows:
        print(f"  {name:<26}{log_loss(p, y):>10.4f}"
              f"{np.mean((p - y) ** 2):>10.4f}{np.mean((p > .5) == (y == 1)):>11.4f}")

    ll_a, ll_s = log_loss(blend(p_agg), y), log_loss(blend(p_split), y)
    d = ll_a - ll_s
    print(f"\n  blended log-loss change from the split: {d:+.5f}"
          f"  ({'SPLIT IS BETTER' if d > 0 else 'no improvement — do not ship'})")

    print("\n  by tour:")
    for t in sorted(set(tour)):
        m = tour == t
        if m.sum() < 500:
            continue
        print(f"    {t.upper():<5} n={m.sum():>6}  aggregate {log_loss(blend(p_agg)[m], y[m]):.4f}"
              f"   split {log_loss(blend(p_split)[m], y[m]):.4f}")
    print("\n  by surface:")
    for s in sorted(set(surf)):
        m = surf == s
        if m.sum() < 500:
            continue
        print(f"    {s:<7} n={m.sum():>6}  aggregate {log_loss(blend(p_agg)[m], y[m]):.4f}"
              f"   split {log_loss(blend(p_split)[m], y[m]):.4f}")


if __name__ == "__main__":
    main()
