"""
Validate the score markets — the handicap and totals lines the dashboard shows.

    python tools/validate_score_markets.py --sample 12000

The win probability is well tested. The SCORE outputs are not: `fair_game_handicap`
and `fair_total_games_line` appear on the Today tab and on every matchup page, and
until now nothing has checked whether they are what they claim to be.

A "fair line" makes a specific, falsifiable promise: at that number the two sides
should be a coin flip. So that is what gets tested here.

    1. At the fair handicap, does player A actually cover 50% of the time?
    2. At the fair total, does the match actually go over 50% of the time?
    3. Are the stated probabilities calibrated across their whole range, not just
       at the midpoint?
    4. Is the game margin unbiased?

Every prediction is rebuilt from the frozen pre-match tables, so this inherits the
same no-leakage guarantee as backtest.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import markov, score_calib  # noqa: E402
from engine.predict import (  # noqa: E402
    W_ELO,
    blend_logit,
    calibrate_total_games,
)
from engine.schema import ELO_SCALE, PROCESSED, RAW  # noqa: E402
from engine.serve_return import point_probabilities  # noqa: E402


def load(tours, lo, hi, min_matches, sample, seed):
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    sr = pd.read_parquet(PROCESSED / "serve_return.parquet")
    drop = ["tour", "tourney_date", "season", "surface", "winner_id", "loser_id"]
    df = r.merge(sr.drop(columns=drop), on="match_id", how="inner")
    df = df[df["tour"].isin(tours) & df["season"].between(lo, hi)]

    ctx = pd.concat([
        pd.read_parquet(RAW / f"matches_{t}.parquet",
                        columns=["match_id", "best_of", "completed", "retirement",
                                 "games_w", "games_l", "total_games"])
        for t in tours if (RAW / f"matches_{t}.parquet").exists()
    ], ignore_index=True)
    df = df.merge(ctx, on="match_id", how="left")
    # Retirements have no meaningful scoreline; they would poison every totals test.
    df = df[df["completed"].fillna(False).astype(bool)]
    df = df[~df["retirement"].fillna(False).astype(bool)]
    df = df[(df["w_matches"] >= min_matches) & (df["l_matches"] >= min_matches)]
    df = df[df["total_games"].between(12, 70)]
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seasons", default="2018-2026")
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--sample", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]

    df = load(tuple(args.tours), int(lo), int(hi), args.min_matches,
              args.sample, args.seed)
    print(f"Validating score markets on {len(df):,} completed matches "
          f"({lo}-{hi})\n")

    a_is_w = df["winner_id"].to_numpy() < df["loser_id"].to_numpy()
    pick = lambda w, l: np.where(a_is_w, df[w].to_numpy(), df[l].to_numpy())
    gap = pick("w_elo_blend", "l_elo_blend") - pick("l_elo_blend", "w_elo_blend")
    p_elo = 1.0 / (1.0 + np.exp(-gap * math.log(10.0) / ELO_SCALE))
    tour, surf = df["tour"].to_numpy(), df["surface"].to_numpy()
    bo = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int).to_numpy()

    a_s = pick("w_serve_excess", "l_serve_excess")
    b_s = pick("l_serve_excess", "w_serve_excess")
    a_r = pick("w_return_excess", "l_return_excess")
    b_r = pick("l_return_excess", "w_return_excess")

    games_a = pick("games_w", "games_l").astype(float)
    games_b = pick("games_l", "games_w").astype(float)
    actual_margin = games_a - games_b
    actual_total = df["total_games"].to_numpy().astype(float)

    # Both line rules are graded in the SAME pass. Rebuilding 12,000 Markov
    # distributions is the expensive part and it is identical for both, so
    # comparing them across two runs would cost twice as much and invite an
    # apples-to-oranges slip if anything else changed in between.
    RULES = ("centre", "best")
    fair_hcap = {r: np.zeros(len(df)) for r in RULES}
    fair_total = {r: np.zeros(len(df)) for r in RULES}
    p_cover = {r: np.zeros(len(df)) for r in RULES}
    p_over = {r: np.zeros(len(df)) for r in RULES}
    exp_margin = np.zeros(len(df))
    exp_total = np.zeros(len(df))

    for i in range(len(df)):
        pa, pb = point_probabilities(a_s[i], a_r[i], b_s[i], b_r[i], tour[i], surf[i])
        p_sim = markov.match_win_prob(pa, pb, int(bo[i]))
        target = blend_logit([p_elo[i], p_sim], [W_ELO, 1 - W_ELO])
        xa, xb = markov.invert_to_target(pa, pb, target, int(bo[i]))
        d = markov.match_distribution(xa, xb, int(bo[i]))
        g = d["games"]

        exp_margin[i] = d["game_margin"]
        exp_total[i] = calibrate_total_games(d["exp_total_games"], int(bo[i]))

        # Call the SAME code predict.py ships. An earlier version of this file
        # re-implemented the line choice and was grading a function the engine
        # does not have; anything measured here has to come from engine/.
        margins = score_calib.margin_distribution(g)
        totals = markov.total_games_distribution(g)
        for r in RULES:
            h = -score_calib.fair_line(margins, "margin", tour[i], int(bo[i]), rule=r)
            fair_hcap[r][i] = h
            raw_thresh = score_calib.uncalibrate(
                -h, margins, "margin", tour[i], int(bo[i]))
            p_cover[r][i] = markov.prob_cover_handicap(g, -raw_thresh)

            t = score_calib.fair_line(totals, "total", tour[i], int(bo[i]), rule=r)
            fair_total[r][i] = t
            p_over[r][i] = markov.prob_over_games(
                g, score_calib.uncalibrate(t, totals, "total", tour[i], int(bo[i])))

    # The line is what the user acts on, so it is graded against what ACTUALLY
    # happened. The stated probability sits beside it because the two
    # disagreeing was the original defect — a rule can centre one and miss the
    # other, and only showing both makes that visible.
    print("  1+2. DO THE FAIR LINES SPLIT 50/50?   (* = shipped rule)")
    print(f"     {'rule':<10}{'handicap':>10}{'stated':>9}"
          f"{'total':>10}{'stated':>9}{'pushes':>8}")
    graded = {}
    for r in RULES:
        cov = (actual_margin + fair_hcap[r]) > 0
        ov = actual_total > fair_total[r]
        graded[r] = (cov, ov)
        mark = "*" if r == score_calib.LINE_RULE else " "
        print(f"   {mark} {r:<8}{cov.mean()*100:>9.1f}%{p_cover[r].mean()*100:>8.1f}%"
              f"{ov.mean()*100:>9.1f}%{p_over[r].mean()*100:>8.1f}%"
              f"{int((actual_total == fair_total[r]).sum()):>8}")

    covered, over = graded[score_calib.LINE_RULE]
    fair_hcap, fair_total = (fair_hcap[score_calib.LINE_RULE],
                             fair_total[score_calib.LINE_RULE])
    p_cover, p_over = (p_cover[score_calib.LINE_RULE],
                       p_over[score_calib.LINE_RULE])

    print("\n  3. GAME MARGIN")
    print(f"     MAE {np.mean(np.abs(exp_margin-actual_margin)):.3f}   "
          f"bias {np.mean(exp_margin-actual_margin):+.3f}")
    print("     TOTAL GAMES")
    print(f"     MAE {np.mean(np.abs(exp_total-actual_total)):.3f}   "
          f"bias {np.mean(exp_total-actual_total):+.3f}")

    print("\n  4. CALIBRATION of the stated over/under probability")
    print(f"     {'stated':>12}{'n':>8}{'actual':>10}{'gap':>9}")
    edges = [0, .35, .45, .5, .55, .65, 1.01]
    for j in range(len(edges) - 1):
        m = (p_over >= edges[j]) & (p_over < edges[j + 1])
        if m.sum() < 100:
            continue
        print(f"     {edges[j]:.2f}-{edges[j+1]:.2f}{m.sum():>8}"
              f"{over[m].mean():>10.3f}{over[m].mean()-p_over[m].mean():>+9.3f}")

    # Split by TOUR as well as format. A pooled number hides a real problem: with
    # one shared best-of-3 totals constant the ATP went over 52.8% and the WTA
    # 48.0%, averaging to a healthy-looking 50.4% that was correct for neither.
    # Any aggregate here should be read alongside this table, not instead of it.
    print("\n  by tour and format:")
    for t in ("atp", "wta"):
        for b in (3, 5):
            m = (tour == t) & (bo == b)
            if m.sum() < 200:
                continue
            print(f"     {t.upper()} best-of-{b}  n={m.sum():>6}"
                  f"  handicap cover {covered[m].mean()*100:5.1f}%"
                  f"   over {over[m].mean()*100:5.1f}%"
                  f"   total bias {np.mean(exp_total[m]-actual_total[m]):+.2f}")


if __name__ == "__main__":
    main()
