"""
Head-to-head and stylistic matchup effects.

Inputs : data/raw/matches_{tour}.parquet
Outputs: data/processed/h2h.parquet — every completed meeting, long form, so the
         predictor and dashboard can both read a pair's history cheaply.

Two distinct things live here:

1. **Head-to-head.** Tennis fans over-weight H2H enormously; the model must not.
   Most of a 6-2 head-to-head is just "the better player kept winning", which the
   ratings already know. What is left over — genuine stylistic mismatch — is small
   and needs heavy shrinkage, so `h2h_elo_delta` measures the record against what
   the ratings *expected* and keeps only a shrunk fraction of the surprise.

2. **Style interactions.** Height still shifts the serve/return excesses. The
   handedness term is now ZERO: the left-hander advantage is real in the
   literature but does not survive measurement here — see LEFTY_VS_RIGHTY_ELO.
   Both were previously asserted from reasoning; both have now been measured by
   tools/validate_adjustments.py, and only one survived.
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow both `python engine/matchups.py` and `python -m engine.matchups`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import ELO_SCALE, PROCESSED, RAW, TOURS

# ──────────────────────────────────────────────────────────────────────────────
# Tuned constants
# ──────────────────────────────────────────────────────────────────────────────
# Head-to-head is damped by three separate things, and each does a distinct job.
#
# The problem being solved: a probability gap converts to Elo at ~695 points per
# unit near even money, and a SMALL head-to-head always shows maximal surprise —
# a 1-0 record is a 100% win rate against a ~50% expectation regardless of who
# played. Left alone that produced a +47 Elo swing off a single meeting, swamping
# a +10 rating gap. That is the "he owns him" trap this module exists to avoid.
#
#   H2H_PRIOR         shrinks by sample size: n_eff / (n_eff + prior)
#   H2H_ELO_PER_PROB  how much of the shrunk surprise we actually believe. The
#                     full-credit conversion is ~695 Elo per unit of probability;
#                     200 says roughly 30% of what survives shrinkage is real
#                     signal and the rest is noise in a small sample.
#   H2H_MAX_ELO       a backstop for the extreme tail (10-0 and similar)
#
# Keeping the strength as its own constant matters: cranking the prior alone to
# get the magnitude right made the cap bind for every record from 2-0 upward, so
# a 2-0 and a 10-0 scored identically and the signal went flat. The resulting
# curve is ~1.6% for 1-0, 3.9% for 3-0, 5.5% for 5-0, capping near 6.5%.
# Measured at a best multiplier of 0.6 over 34,563 matches, so the strength came
# down from 200 to 120. The gain was real but tiny (0.61567 -> 0.61566): once the
# ratings are in, a pair's history says very little that is not already priced.
#
# TREAT THAT 0.6 WITH SUSPICION. Re-measured on a refreshed archive of 47,563
# matches, the same tool now says the best multiplier is 2.2 — which would put
# the strength at ~264, ABOVE the 200 it was cut from. Two runs, opposite advice.
#
# Neither is wrong; the objective is simply flat. The whole spread between them
# is ~0.00005 log loss, far inside what a change of sample moves. So the honest
# reading of any "best multiplier" in this layer is directional only: does the
# term help at all, and is the sign right. It does NOT locate a magnitude, and
# re-tuning this constant every time the archive grows would be fitting noise.
#
# Left at 120 deliberately, and it should stay there unless a measurement shows
# a gain that survives a change of sample.
H2H_PRIOR = 8.0
H2H_ELO_PER_PROB = 120.0
H2H_MAX_ELO = 45.0        # hard cap; no pairing is worth more than ~6.5% win prob
H2H_SURFACE_WEIGHT = 1.6  # meetings on the same surface count for more
H2H_RECENCY_HALFLIFE_YEARS = 3.0  # a 2014 meeting says little about 2026

# The left-hander advantage is well documented in the tennis literature — the
# serve swings into a right-hander's backhand in the deuce court, and right-handers
# see it far less often than the reverse. It does NOT show up in this archive.
#
# Measured over 34,563 matches (tools/validate_adjustments.py), applying it at 12
# Elo made log loss slightly WORSE (0.61579 vs 0.61569 baseline) and the best
# multiplier came out at -0.8. Whatever edge exists is either already inside the
# ratings — a lefty's results already reflect it — or is too small to survive the
# noise at this sample size.
#
# Set to 0 on that evidence rather than removed, so it is one edit to revisit if a
# larger or differently-sliced sample ever says otherwise.
LEFTY_VS_RIGHTY_ELO = 0.0

# Height: tall players serve bigger but return worse. Expressed as a serve/return
# excess shift per cm away from tour-average height, applied only where it matters
# (fast surfaces amplify it, clay damps it).
#
# MEASURED, and it survives — unlike handedness. This term shifts the serve/return
# excesses rather than the Elo gap, so the Markov chain has to be re-solved for
# every candidate magnitude, which is why it went unaudited long after the rest of
# the layer had been. Over 47,563 matches (2015-2026), height present for 98%:
#
#     off (x0)   0.61267        x1.5   0.61243
#     x0.5       0.61256        x2.0   0.61241   <- best
#     x1.0       0.61248        x3.0   0.61248
#
# Switching it OFF is the worst outcome and the curve turns back up past x2, so
# this is a genuine interior optimum, not noise fitting.
#
# KEPT AT x1.0 rather than doubled. Off-versus-on is worth 0.00019; x1.0-versus-x2.0
# is worth 0.00007, and an objective this flat does not locate a magnitude
# reliably — the head-to-head constant below is the proof, having flipped from
# "best x0.6" to "best x2.2" between two data refreshes. Read these numbers as
# "is the term real and is the sign right", never as a target to tune to.
TOUR_AVG_HEIGHT = {"atp": 185.0, "wta": 173.0}
HEIGHT_SERVE_PER_CM = 0.00090   # +0.09pt of service points won per cm above average
HEIGHT_RETURN_PER_CM = -0.00055  # …paid back on return
SURFACE_HEIGHT_MULT = {"Grass": 1.35, "Carpet": 1.25, "Hard": 1.0, "Clay": 0.65}


# ──────────────────────────────────────────────────────────────────────────────
# Head-to-head table
# ──────────────────────────────────────────────────────────────────────────────
def build_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form meeting log: two rows per completed match (one per player).

    Kept as raw meetings rather than pre-aggregated totals so that a prediction
    can filter to "before date D" — pre-aggregating would leak the future into
    every backtest.
    """
    m = matches[matches["completed"] | matches["retirement"]].copy()
    frames = []
    for me, opp, won in (("winner", "loser", True), ("loser", "winner", False)):
        f = pd.DataFrame(
            {
                "tour": m["tour"],
                "tourney_date": m["tourney_date"],
                "season": m["season"],
                "surface": m["surface"],
                "match_id": m["match_id"],
                "player_id": m[f"{me}_id"].astype("int64"),
                "opp_id": m[f"{opp}_id"].astype("int64"),
                "won": won,
            }
        )
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)


def h2h_record(
    h2h: pd.DataFrame,
    player_id: int,
    opp_id: int,
    before: pd.Timestamp | None = None,
    surface: str | None = None,
) -> dict:
    """
    Weighted head-to-head record for `player_id` vs `opp_id`.

    `before` MUST be supplied in any backtest context — without it this reads the
    full history including matches after the one being predicted.
    """
    sel = h2h[(h2h["player_id"] == player_id) & (h2h["opp_id"] == opp_id)]
    if before is not None:
        sel = sel[sel["tourney_date"] < before]
    if sel.empty:
        return {"wins": 0.0, "losses": 0.0, "n": 0, "raw_wins": 0, "raw_losses": 0}

    ref = before if before is not None else sel["tourney_date"].max()
    years = (ref - sel["tourney_date"]).dt.days / 365.25
    w = 0.5 ** (years / H2H_RECENCY_HALFLIFE_YEARS)
    if surface is not None:
        w = w * np.where(sel["surface"] == surface, H2H_SURFACE_WEIGHT, 1.0)

    won = sel["won"].to_numpy()
    return {
        "wins": float(w[won].sum()),
        "losses": float(w[~won].sum()),
        "n": int(len(sel)),
        "raw_wins": int(won.sum()),
        "raw_losses": int((~won).sum()),
    }


def h2h_elo_delta(record: dict, expected_win_prob: float) -> float:
    """
    Elo adjustment from a head-to-head record, measured against expectation.

    The question is never "who won more meetings" but "did they win more than
    their ratings said they would". A 4-2 record when you were a 70% favourite
    every time is *under*performance and should move the line the other way.
    """
    n_eff = record["wins"] + record["losses"]
    if n_eff <= 0:
        return 0.0

    actual = record["wins"] / n_eff
    surprise = actual - float(expected_win_prob)

    # Shrink by sample size, using the RECENCY-WEIGHTED count rather than the raw
    # meeting count: five meetings from a decade ago should not carry the weight
    # of five from last season, and raw n would give them exactly that.
    shrink = n_eff / (n_eff + H2H_PRIOR)
    elo = surprise * H2H_ELO_PER_PROB * shrink
    return float(np.clip(elo, -H2H_MAX_ELO, H2H_MAX_ELO))


# ──────────────────────────────────────────────────────────────────────────────
# Style interactions
# ──────────────────────────────────────────────────────────────────────────────
def handedness_elo_delta(hand_a: object, hand_b: object) -> float:
    """Elo adjustment for A from the handedness matchup (0 if same or unknown)."""
    a = hand_a if isinstance(hand_a, str) else ""
    b = hand_b if isinstance(hand_b, str) else ""
    a, b = a.strip().upper()[:1], b.strip().upper()[:1]
    if a not in ("L", "R") or b not in ("L", "R") or a == b:
        return 0.0
    return LEFTY_VS_RIGHTY_ELO if a == "L" else -LEFTY_VS_RIGHTY_ELO


def height_style_delta(
    height_cm: object, tour: str, surface: str
) -> tuple[float, float]:
    """
    (serve_excess_shift, return_excess_shift) from a player's height.

    Returns (0, 0) when height is unknown — roughly 15% of tour rows, and guessing
    is worse than abstaining.
    """
    if height_cm is None or not np.isfinite(height_cm) or height_cm <= 0:
        return 0.0, 0.0
    avg = TOUR_AVG_HEIGHT.get(tour, 180.0)
    dev = float(height_cm) - avg
    mult = SURFACE_HEIGHT_MULT.get(surface, 1.0)
    return HEIGHT_SERVE_PER_CM * dev * mult, HEIGHT_RETURN_PER_CM * dev * mult


def build_all(tours: tuple[str, ...] = TOURS) -> pd.DataFrame:
    out = []
    for tour in tours:
        path = RAW / f"matches_{tour}.parquet"
        if not path.exists():
            print(f"  [matchups] no {path.name}, skipping {tour}")
            continue
        m = pd.read_parquet(path)
        print(f"  [matchups] {tour}: {len(m):,} matches")
        out.append(build_h2h(m))
    if not out:
        raise FileNotFoundError(
            f"No match parquets found in {RAW}. Run `python fetch_data.py` first."
        )
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build head-to-head meeting log.")
    ap.add_argument("--tours", nargs="+", default=list(TOURS))
    args = ap.parse_args()

    df = build_all(tuple(args.tours))
    df.to_parquet(PROCESSED / "h2h.parquet", index=False)
    print(f"  [matchups] wrote {len(df):,} meeting rows")


if __name__ == "__main__":
    main()
