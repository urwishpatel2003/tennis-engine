"""
Measure the adjustment layer — the part backtest.py has never tested.

    python tools/validate_adjustments.py --seasons 2015-2026

`backtest.py` scores Elo + serve/return. It does NOT exercise conditions
(fatigue, rest, home crowd), head-to-head, or the height/style shift, because
those are applied per-query inside predict.py rather than stored per match. So
every constant in that layer has been held to reasoning rather than evidence.

That gap is not hypothetical. Head-to-head originally turned a single 1-0 meeting
into a +47 Elo swing — larger than most rating gaps — and it backtested
*identically* to the fixed version, because the backtest never saw the term.

This script rebuilds each adjustment from the same frozen, leak-free tables the
engine uses and asks two questions per adjustment:

    1. Does it reduce out-of-sample log loss at its current magnitude?
    2. What multiplier on it would be optimal — is the hand-tuned size right,
       and is the SIGN even right?

A multiplier near 0 means the adjustment is doing nothing. Below 0 means it is
actively backwards.

Leak-free by construction
-------------------------
conditions.parquet is written pre-match by engine/conditions.py, so it is joined
directly. Head-to-head is rebuilt by walking the match log chronologically and
consulting only meetings already played — the same rule as
matchups.h2h_record(before=...), but O(n) instead of a full-frame scan per row.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import conditions as cond  # noqa: E402
from engine import markov  # noqa: E402
from engine import matchups as mu  # noqa: E402
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


def attach_conditions(df):
    """Join the pre-match fatigue/rest/home state for both players."""
    p = PROCESSED / "conditions.parquet"
    if not p.exists():
        return df
    c = pd.read_parquet(p, columns=["match_id", "player_id", "days_rest",
                                    "fatigue_index", "is_home"])
    for side, idcol in (("w", "winner_id"), ("l", "loser_id")):
        part = c.rename(columns={
            "player_id": idcol,
            "days_rest": f"{side}_days_rest",
            "fatigue_index": f"{side}_fatigue",
            "is_home": f"{side}_home",
        })
        df = df.merge(part, on=["match_id", idcol], how="left")
    return df


def h2h_deltas(df):
    """
    Head-to-head Elo adjustment per match, walking forward.

    Mirrors matchups.h2h_record: recency half-life of 3 years and same-surface
    meetings weighted more heavily, then h2h_elo_delta against what the ratings
    alone expected.
    """
    hist = defaultdict(list)          # (a,b) -> [(date, surface, a_won)]
    out = np.zeros(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        w, l = int(r.winner_id), int(r.loser_id)
        key = (min(w, l), max(w, l))
        prior = hist[key]
        if prior:
            wins = losses = 0.0
            for d, s, a_won in prior:
                yrs = max((r.tourney_date - d).days, 0) / 365.25
                wt = 0.5 ** (yrs / mu.H2H_RECENCY_HALFLIFE_YEARS)
                if s == r.surface:
                    wt *= mu.H2H_SURFACE_WEIGHT
                # `a_won` is stored for the lower id; re-orient onto the winner.
                lower_won = a_won
                if (w == key[0]) == lower_won:
                    wins += wt
                else:
                    losses += wt
            rec = {"wins": wins, "losses": losses, "n": len(prior),
                   "raw_wins": 0, "raw_losses": 0}
            gap = r.w_elo_blend - r.l_elo_blend
            exp = 1.0 / (1.0 + 10.0 ** (-gap / ELO_SCALE))
            out[i] = mu.h2h_elo_delta(rec, exp)
        hist[key].append((r.tourney_date, r.surface, (w == key[0])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seasons", default="2015-2026")
    ap.add_argument("--min-matches", type=int, default=20)
    args = ap.parse_args()
    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]

    df = load(tuple(args.tours), int(lo), int(hi), args.min_matches)
    df = attach_conditions(df)
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

    # conditions: fatigue + short rest / long layoff + home crowd
    if "w_fatigue" in df.columns:
        def cond_delta(side):
            f = pick(f"{side}_fatigue", f"{'l' if side=='w' else 'w'}_fatigue")
            rest = pick(f"{side}_days_rest", f"{'l' if side=='w' else 'w'}_days_rest")
            home = pick(f"{side}_home", f"{'l' if side=='w' else 'w'}_home")
            out = np.zeros(len(df))
            fv = pd.to_numeric(pd.Series(f), errors="coerce").fillna(0).to_numpy()
            out += cond.FATIGUE_ELO_PER_INDEX * fv
            rv = pd.to_numeric(pd.Series(rest), errors="coerce").to_numpy()
            out += np.where(rv <= 1, cond.SHORT_REST_PENALTY, 0.0)
            out += np.where(rv >= 60,
                            cond.LONG_LAYOFF_PENALTY
                            * np.clip((rv - 60) / 120.0 + 0.5, 0, 1), 0.0)
            out += np.where(pd.Series(home).fillna(False).to_numpy().astype(bool),
                            cond.HOME_ELO_BONUS, 0.0)
            return np.nan_to_num(out)
        adj["conditions"] = cond_delta("w") - cond_delta("l")

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

    # head-to-head
    h = h2h_deltas(df)
    adj["head_to_head"] = np.where(a_is_w, h, -h)

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


if __name__ == "__main__":
    main()
