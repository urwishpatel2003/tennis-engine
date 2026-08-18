"""
Rebuild the per-query adjustment layer for a whole frame of past matches.

Inputs : data/processed/conditions.parquet (pre-match by construction)
         arrays describing each match, oriented onto player A
Outputs: Elo deltas and serve/return shifts, one row per match

Why this exists
---------------
`engine/predict.py` applies conditions, head-to-head and height/style per QUERY,
one match at a time, from live lookups. `backtest.py` reads frozen per-match
columns. Those two paths had no code in common, so for a long time the backtest
scored Elo + serve/return only and silently omitted everything in this module —
the reported log loss described a model that was never the one shipping.

That gap was not harmless. Head-to-head once turned a single 1-0 meeting into a
+47 Elo swing, larger than most rating gaps, and it backtested *identically* to
the fixed version because the backtest could not see the term.

This module is the shared reconstruction, so `backtest.py` and
`tools/validate_adjustments.py` compute the adjustment layer the same way or not
at all.

No future leakage
-----------------
Every function here is given only what was knowable before the match:

  * conditions.parquet is written pre-match by engine/conditions.py, so it is
    joined rather than recomputed.
  * head-to-head walks the log in date order and consults only meetings already
    played — the same rule as matchups.h2h_record(before=...), but O(n) instead
    of a full-frame scan per row.
  * height is a static player attribute and cannot leak.

`tests/test_no_leakage.py` guards the first; the h2h walk is guarded by its own
assertion that a match never sees itself.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from engine import conditions as cond
from engine import matchups as mu
from engine.schema import ELO_SCALE, PROCESSED, RAW


def _conditions_lookup() -> dict:
    """(match_id, player_id) -> (days_rest, fatigue_index, is_home)."""
    p = PROCESSED / "conditions.parquet"
    if not p.exists():
        return {}
    c = pd.read_parquet(
        p, columns=["match_id", "player_id", "days_rest", "fatigue_index", "is_home",
                    "matches_this_event"],
    )
    return {
        (r.match_id, int(r.player_id)): (r.days_rest, r.fatigue_index, r.is_home,
                                         r.matches_this_event)
        for r in c.itertuples(index=False)
    }


def conditions_elo(match_ids, a_ids, b_ids) -> np.ndarray:
    """
    Conditions Elo delta for player A (A's bonus minus B's).

    Mirrors the arithmetic in predict.py exactly. Note that two of the four
    terms are currently ZERO by measurement — fatigue and short rest both
    scored backwards in tools/validate_adjustments.py — so this is dominated by
    the long-layoff penalty and the home-crowd bonus.
    """
    look = _conditions_lookup()
    if not look:
        return np.zeros(len(match_ids))

    def side(ids):
        out = np.zeros(len(ids))
        for i, (m, pid) in enumerate(zip(match_ids, ids)):
            rec = look.get((m, int(pid)))
            if rec is None:
                continue
            rest, fatigue, home, played = rec
            f = float(fatigue) if pd.notna(fatigue) else 0.0
            out[i] += cond.FATIGUE_ELO_PER_INDEX * f
            if pd.notna(rest):
                r = float(rest)
                if r <= 1:
                    out[i] += cond.SHORT_REST_PENALTY
                if r >= 60:
                    out[i] += cond.LONG_LAYOFF_PENALTY * float(
                        np.clip((r - 60) / 120.0 + 0.5, 0, 1)
                    )
            if bool(home) if pd.notna(home) else False:
                out[i] += cond.HOME_ELO_BONUS
            # Matches already won at THIS event. Previously omitted here, which
            # meant the term predict.py applies was invisible to both the
            # backtest and the adjustment validator - the same blind spot that
            # let a +47 Elo head-to-head swing go unnoticed.
            if pd.notna(played) and float(played) > 0:
                out[i] += cond.event_progress_elo(float(played))
        return out

    return np.nan_to_num(side(a_ids) - side(b_ids))


def h2h_elo(dates, a_ids, b_ids, surfaces, elo_gap, a_won) -> np.ndarray:
    """
    Head-to-head Elo delta for player A, walking forward in time.

    `elo_gap` is A's rating edge BEFORE any adjustment; the head-to-head term
    measures the record against what those ratings expected, so a 4-2 record
    built as a 70% favourite every time is underperformance and moves the line
    the other way.

    `a_won` is required. Past RESULTS are the whole input to a head-to-head
    record, so there is no meaningful resultless variant — an earlier draft had
    one and it silently scored every prior meeting as a loss. Live predictions
    do not come through here at all; they read h2h.parquet via
    matchups.h2h_record(before=...).

    Rows are processed in date order regardless of the order they arrive in, so
    callers need not pre-sort. The returned array is in the CALLER's order.
    """
    n = len(a_ids)
    order = np.argsort(np.asarray(dates, dtype="datetime64[ns]"), kind="stable")
    hist: dict[tuple[int, int], list] = defaultdict(list)
    out = np.zeros(n)

    for i in order:
        a, b = int(a_ids[i]), int(b_ids[i])
        key = (min(a, b), max(a, b))
        prior = hist[key]
        if prior:
            wins = losses = 0.0
            for d, s, lower_won in prior:
                yrs = max((dates[i] - d).days, 0) / 365.25
                wt = 0.5 ** (yrs / mu.H2H_RECENCY_HALFLIFE_YEARS)
                if s == surfaces[i]:
                    wt *= mu.H2H_SURFACE_WEIGHT
                if (a == key[0]) == lower_won:
                    wins += wt
                else:
                    losses += wt
            rec = {"wins": wins, "losses": losses, "n": len(prior),
                   "raw_wins": 0, "raw_losses": 0}
            exp = 1.0 / (1.0 + 10.0 ** (-elo_gap[i] / ELO_SCALE))
            out[i] = mu.h2h_elo_delta(rec, exp)
        lower_won = bool(a_won[i]) if a == key[0] else not bool(a_won[i])
        hist[key].append((dates[i], surfaces[i], lower_won))
    return out


def height_shifts(a_heights, b_heights, tours, surfaces) -> np.ndarray:
    """
    (n, 4) array of [a_serve, a_return, b_serve, b_return] excess shifts.

    Zero where height is unknown — abstaining beats guessing, and the term is
    measured only where it actually fires.
    """
    out = np.zeros((len(tours), 4))
    for i in range(len(tours)):
        out[i, 0], out[i, 1] = mu.height_style_delta(
            _f(a_heights[i]), tours[i], surfaces[i])
        out[i, 2], out[i, 3] = mu.height_style_delta(
            _f(b_heights[i]), tours[i], surfaces[i])
    return out


def _f(v) -> float | None:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def player_attributes(tours=("atp", "wta")) -> pd.DataFrame:
    """player_id / tour / hand / height, for callers that need to join it on."""
    frames = []
    for t in tours:
        p = RAW / f"players_{t}.parquet"
        if p.exists():
            frames.append(
                pd.read_parquet(p, columns=["player_id", "hand", "height"]).assign(tour=t)
            )
    if not frames:
        return pd.DataFrame(columns=["player_id", "hand", "height", "tour"])
    return pd.concat(frames, ignore_index=True)
