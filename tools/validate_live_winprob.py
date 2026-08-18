"""
Is the in-match win probability actually any good?

    python tools/validate_live_winprob.py --sample 8000

`engine/live_state.py` is exact arithmetic given the score and the two serve
probabilities, so it cannot be "wrong" in the way a fitted model can. That is
not the same as being USEFUL, and the distinction is the whole point of this
script. A live number is only worth showing if:

    1. it is CALIBRATED — when it says 75%, that player wins about 75% of the
       time. An uncalibrated live figure is worse than none, because a viewer
       reads it as a fact rather than an estimate.
    2. it BEATS the pre-match price once the score is known. If knowing a set
       has been won does not improve the forecast, the feature is decoration.

Method
------
Every completed match is replayed at its SET boundaries. Set boundaries, not
points, because the archive stores a final scoreline ('6-4 3-6 7-5') and not a
point-by-point tape — the per-point path is genuinely not knowable from this
data, so it is not guessed at. Point-level replay needs the Live Tennis API's
point tape, which is a paid tier.

At each boundary the state is (sets_a, sets_b) with games 0-0, which is exactly
recoverable, and the outcome is who actually won. Player A is fixed as the LOWER
player id, matching backtest.py, so row orientation carries no information.

Serve probabilities come from the same frozen pre-match tables the backtest uses
and are reconciled to the blended Elo+serve/return price, so this measures the
live layer rather than re-measuring the base model.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import markov  # noqa: E402
from engine.live_state import win_prob_from_state  # noqa: E402
from engine.predict import W_ELO, blend_logit  # noqa: E402
from engine.schema import ELO_SCALE, PROCESSED, RAW  # noqa: E402
from engine.serve_return import point_probabilities  # noqa: E402

SET_RE = re.compile(r"^(\d+)-(\d+)")


def set_scores(score: object) -> list[tuple[int, int]]:
    """['6-4', '7-6(3)'] -> [(6,4), (7,6)], from the WINNER's perspective."""
    out: list[tuple[int, int]] = []
    for tok in str(score or "").split():
        m = SET_RE.match(tok.strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 30 or b > 30:          # guards against junk in the archive
            continue
        out.append((a, b))
    return out


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load(tours, lo, hi, min_matches, sample, seed):
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    sr = pd.read_parquet(PROCESSED / "serve_return.parquet")
    drop = ["tour", "tourney_date", "season", "surface", "winner_id", "loser_id"]
    df = r.merge(sr.drop(columns=drop), on="match_id", how="inner")
    df = df[df["tour"].isin(tours) & df["season"].between(lo, hi)]
    ctx = pd.concat([
        pd.read_parquet(RAW / f"matches_{t}.parquet",
                        columns=["match_id", "best_of", "completed", "retirement", "score"])
        for t in tours if (RAW / f"matches_{t}.parquet").exists()
    ], ignore_index=True)
    df = df.merge(ctx, on="match_id", how="left")
    df = df[df["completed"].fillna(False).astype(bool)]
    df = df[~df["retirement"].fillna(False).astype(bool)]
    df = df[(df["w_matches"] >= min_matches) & (df["l_matches"] >= min_matches)]
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def calibration(p: np.ndarray, y: np.ndarray, label: str) -> None:
    print(f"\n  calibration — {label}")
    print(f"     {'band':>12}{'n':>8}{'stated':>9}{'actual':>9}{'gap':>8}")
    edges = [0, .1, .3, .5, .7, .9, 1.01]
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() < 40:
            continue
        print(f"     {edges[i]:.1f}-{edges[i+1]:.1f}{m.sum():>10}"
              f"{p[m].mean():>9.3f}{y[m].mean():>9.3f}{y[m].mean()-p[m].mean():>+8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seasons", default="2018-2026")
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]

    df = load(tuple(args.tours), int(lo), int(hi), args.min_matches,
              args.sample, args.seed)
    print(f"Replaying {len(df):,} completed matches at set boundaries "
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
    y_all = a_is_w.astype(int)
    scores = df["score"].tolist()

    # stage -> (predictions, outcomes). Stage 0 is the pre-match price, so the
    # later stages have something honest to be compared against.
    stages: dict[int, list[tuple[float, int]]] = {0: [], 1: [], 2: [], 3: []}

    for i in range(len(df)):
        pa, pb = point_probabilities(a_s[i], a_r[i], b_s[i], b_r[i], tour[i], surf[i])
        p_sim = markov.match_win_prob(pa, pb, int(bo[i]))
        target = blend_logit([p_elo[i], p_sim], [W_ELO, 1 - W_ELO])
        xa, xb = markov.invert_to_target(pa, pb, target, int(bo[i]))

        sets = set_scores(scores[i])
        if not sets:
            continue
        # Scorelines are winner-first; re-orient onto player A.
        if not a_is_w[i]:
            sets = [(b, a) for a, b in sets]

        y = int(y_all[i])
        stages[0].append((target, y))

        sa = sb = 0
        for k, (ga, gb) in enumerate(sets[:-1]):   # skip the match-ending set
            sa += 1 if ga > gb else 0
            sb += 0 if ga > gb else 1
            sets_to_win = 3 if int(bo[i]) == 5 else 2
            if sa >= sets_to_win or sb >= sets_to_win:
                break
            p = win_prob_from_state(xa, xb, sets_a=sa, sets_b=sb,
                                    best_of=int(bo[i]))
            stages[min(k + 1, 3)].append((p, y))

    print(f"  {'stage':<22}{'n':>8}{'logloss':>10}{'accuracy':>10}{'vs pre-match':>14}")
    print("  " + "-" * 64)
    base_ll = None
    for k in sorted(stages):
        rows = stages[k]
        if len(rows) < 100:
            continue
        p = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        ll = log_loss(p, y)
        acc = float(((p > 0.5) == (y == 1)).mean())
        if k == 0:
            base_ll = ll
            label = "pre-match (baseline)"
            delta = ""
        else:
            label = f"after {k} set{'s' if k > 1 else ''}"
            delta = f"{base_ll - ll:+.4f}"
        print(f"  {label:<22}{len(rows):>8,}{ll:>10.4f}{acc:>10.3f}{delta:>14}")

    for k in sorted(stages):
        rows = stages[k]
        if len(rows) < 400:
            continue
        p = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        calibration(p, y, "pre-match" if k == 0 else f"after {k} set(s)")

    print("\n  A live figure is only worth showing if the gap column stays small.")
    print("  Beating the pre-match baseline is expected - a set result is real")
    print("  information. Being CALIBRATED while doing so is the actual test.")


if __name__ == "__main__":
    main()
