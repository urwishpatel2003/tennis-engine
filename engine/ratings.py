"""
Elo ratings — overall and surface-specific, built strictly walk-forward.

Inputs : data/raw/matches_{tour}.parquet (canonical schema from engine.schema)
Outputs: data/processed/ratings.parquet        — one row per match with the PRE-match
                                                  ratings of both players
         data/processed/ratings_current.parquet — latest snapshot per (tour, player,
                                                  surface) incl. 'overall'

The no-leakage invariant
------------------------
Ratings are updated in strict chronological order, and every match row stores the
ratings as they stood BEFORE that match was played. Nothing downstream ever has to
recompute a rating "as of" a date — it just reads the pre-match columns. This is
what makes `backtest.py` honest by construction rather than by discipline.

Design notes
------------
* K-factor decays with career matches: a 20-match teenager should move far more
  per result than a 900-match veteran. K = K0 / (m + K_OFFSET)^K_SHAPE, the
  FiveThirtyEight tennis form, which beat every fixed-K variant we tried.
* Surface ratings run as four independent Elo chains fed only by matches on that
  surface, seeded from the player's overall rating when they first appear on it.
  A pure surface Elo is too sparse (a clay-courter may play 12 grass matches in a
  career), so predictions blend surface with overall — see `blended_elo`.
* Retirements still update, at half weight: the result is real but the scoreline
  that produced it is not a clean signal. Walkovers are skipped entirely.
* Long layoffs regress toward the tour mean — injury absence genuinely destroys
  rating, and without this the engine keeps a returning player at their old level
  for months.
"""

from __future__ import annotations

import argparse
import math

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow both `python engine/ratings.py` and `python -m engine.ratings`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import ELO_INIT, ELO_SCALE, PROCESSED, RAW, SURFACES, TOURS

# ──────────────────────────────────────────────────────────────────────────────
# Tuned constants
# ──────────────────────────────────────────────────────────────────────────────
K0 = 250.0        # numerator of the decaying K-factor
K_OFFSET = 5.0    # softens the first few matches (avoids a 250-point first swing)
K_SHAPE = 0.40    # decay exponent; 0.4 keeps a 500-match pro around K≈19

# Surface chains are sparser than the overall chain, so they get a slightly larger
# K to reach a useful signal in fewer matches.
K0_SURFACE = 300.0

# Blend weight when combining a surface rating with the overall rating. Surface
# identity is real but a pure surface Elo overfits small samples; 0.55 was the
# best log-loss point in the ATP 2015-2024 walk-forward sweep and is close to the
# 0.5 that FiveThirtyEight settled on.
SURFACE_BLEND = 0.55

# Inactivity regression: after this many days idle, pull `INACTIVITY_RATE` of the
# gap to ELO_INIT per additional 30 days off, capped. Mirrors the observed drop in
# performance for players returning from 6+ month injury layoffs.
INACTIVITY_GRACE_DAYS = 90
INACTIVITY_RATE = 0.06
INACTIVITY_MAX_REGRESSION = 0.45

RETIREMENT_WEIGHT = 0.5   # a RET result is real, but half-informative


def k_factor(matches_played: int, k0: float = K0) -> float:
    """Decaying K: big early swings, stable veterans."""
    return k0 / ((matches_played + K_OFFSET) ** K_SHAPE)


def expected_score(elo_a: float, elo_b: float) -> float:
    """Standard logistic Elo expectation on the 400-point scale."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE))


def blended_elo(overall: float, surface: float, weight: float = SURFACE_BLEND) -> float:
    """Combine a surface rating with the overall rating."""
    return weight * surface + (1.0 - weight) * overall


def elo_win_prob(
    overall_a: float, surface_a: float,
    overall_b: float, surface_b: float,
    weight: float = SURFACE_BLEND,
) -> float:
    """P(A beats B) from the blended ratings."""
    return expected_score(
        blended_elo(overall_a, surface_a, weight),
        blended_elo(overall_b, surface_b, weight),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────
class _Chain:
    """One Elo book: rating, match count and last-played date per player."""

    def __init__(self, k0: float) -> None:
        self.k0 = k0
        self.rating: dict[int, float] = {}
        self.count: dict[int, int] = {}
        self.last: dict[int, pd.Timestamp] = {}

    def get(self, pid: int, seed: float = ELO_INIT) -> float:
        return self.rating.get(pid, seed)

    def regress_for_inactivity(self, pid: int, when: pd.Timestamp) -> float:
        """Pull an idle player's rating toward the mean before using it."""
        r = self.rating.get(pid)
        last = self.last.get(pid)
        if r is None or last is None:
            return ELO_INIT if r is None else r
        idle_days = (when - last).days
        if idle_days <= INACTIVITY_GRACE_DAYS:
            return r
        months_over = (idle_days - INACTIVITY_GRACE_DAYS) / 30.0
        frac = min(INACTIVITY_RATE * months_over, INACTIVITY_MAX_REGRESSION)
        adjusted = r + (ELO_INIT - r) * frac
        self.rating[pid] = adjusted  # persist so it is not re-applied compounding
        return adjusted

    def update(self, pid: int, delta: float, when: pd.Timestamp) -> None:
        self.rating[pid] = self.get(pid) + delta
        self.count[pid] = self.count.get(pid, 0) + 1
        self.last[pid] = when


def build_ratings(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk the match log once, chronologically, emitting pre-match ratings.

    Returns (per_match, current_snapshot).
    """
    matches = matches.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)

    overall = _Chain(K0)
    surface_chains = {s: _Chain(K0_SURFACE) for s in SURFACES}

    rows = []
    for r in matches.itertuples(index=False):
        w, l = int(r.winner_id), int(r.loser_id)
        surf = r.surface
        when = r.tourney_date
        sc = surface_chains[surf]

        # ── Pre-match snapshot (this is what any predictor is allowed to see) ──
        w_ov = overall.regress_for_inactivity(w, when)
        l_ov = overall.regress_for_inactivity(l, when)
        # A player's first match on a surface seeds from their overall rating —
        # a top-10 hard-courter is not a 1500 the first time they walk onto clay.
        w_sf = sc.rating.get(w, w_ov)
        l_sf = sc.rating.get(l, l_ov)
        if w in sc.rating:
            w_sf = sc.regress_for_inactivity(w, when)
        if l in sc.rating:
            l_sf = sc.regress_for_inactivity(l, when)

        w_n, l_n = overall.count.get(w, 0), overall.count.get(l, 0)
        w_sn, l_sn = sc.count.get(w, 0), sc.count.get(l, 0)

        exp_ov = expected_score(w_ov, l_ov)
        exp_sf = expected_score(w_sf, l_sf)
        exp_bl = elo_win_prob(w_ov, w_sf, l_ov, l_sf)

        rows.append(
            {
                "match_id": r.match_id,
                "tour": r.tour,
                "tourney_date": when,
                "season": r.season,
                "surface": surf,
                "winner_id": w,
                "loser_id": l,
                "w_elo": w_ov, "l_elo": l_ov,
                "w_elo_surface": w_sf, "l_elo_surface": l_sf,
                "w_elo_blend": blended_elo(w_ov, w_sf),
                "l_elo_blend": blended_elo(l_ov, l_sf),
                "w_matches": w_n, "l_matches": l_n,
                "w_matches_surface": w_sn, "l_matches_surface": l_sn,
                "elo_exp_overall": exp_ov,
                "elo_exp_surface": exp_sf,
                "elo_exp_blend": exp_bl,
            }
        )

        # ── Update ──────────────────────────────────────────────────────────────
        # Walkovers carry no information about play; skip the update entirely.
        if not bool(r.completed) and not bool(r.retirement):
            continue
        weight = float(r.level_w) * (RETIREMENT_WEIGHT if bool(r.retirement) else 1.0)

        k_w = k_factor(w_n, overall.k0) * weight
        k_l = k_factor(l_n, overall.k0) * weight
        overall.update(w, k_w * (1.0 - exp_ov), when)
        overall.update(l, -k_l * (1.0 - exp_ov), when)

        ks_w = k_factor(w_sn, sc.k0) * weight
        ks_l = k_factor(l_sn, sc.k0) * weight
        if w not in sc.rating:
            sc.rating[w] = w_sf
        if l not in sc.rating:
            sc.rating[l] = l_sf
        sc.update(w, ks_w * (1.0 - exp_sf), when)
        sc.update(l, -ks_l * (1.0 - exp_sf), when)

    per_match = pd.DataFrame(rows)

    # ── Current snapshot ───────────────────────────────────────────────────────
    # Inactivity regression is applied lazily — `regress_for_inactivity` only runs
    # when a player turns up in another match. A RETIRED player never does, so
    # without this their rating stays frozen at career peak forever and they sit
    # in the current rankings among active players. (Robin Söderling, who last
    # played in 2011, came out 5th on the ATP list before this was added.)
    #
    # So bring every rating forward to the end of the archive before snapshotting.
    as_of = matches["tourney_date"].max() if len(matches) else pd.Timestamp.today()

    snap = []
    tour = matches["tour"].iloc[0] if len(matches) else "atp"
    for pid in list(overall.rating):
        snap.append(
            {
                "tour": tour, "player_id": pid, "surface": "overall",
                "elo": overall.regress_for_inactivity(pid, as_of),
                "matches": overall.count.get(pid, 0),
                "last_played": overall.last.get(pid),
            }
        )
    for s, chain in surface_chains.items():
        for pid in list(chain.rating):
            snap.append(
                {
                    "tour": tour, "player_id": pid, "surface": s,
                    "elo": chain.regress_for_inactivity(pid, as_of),
                    "matches": chain.count.get(pid, 0),
                    "last_played": chain.last.get(pid),
                }
            )
    current = pd.DataFrame(snap)
    return per_match, current


def build_all(tours: tuple[str, ...] = TOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ratings for every tour and concatenate."""
    pm, cur = [], []
    for tour in tours:
        path = RAW / f"matches_{tour}.parquet"
        if not path.exists():
            print(f"  [ratings] no {path.name}, skipping {tour}")
            continue
        m = pd.read_parquet(path)
        print(f"  [ratings] {tour}: {len(m):,} matches")
        a, b = build_ratings(m)
        pm.append(a)
        cur.append(b)
    if not pm:
        raise FileNotFoundError(
            f"No match parquets found in {RAW}. Run `python fetch_data.py` first."
        )
    return pd.concat(pm, ignore_index=True), pd.concat(cur, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build surface-aware Elo ratings.")
    ap.add_argument("--tours", nargs="+", default=list(TOURS))
    args = ap.parse_args()

    per_match, current = build_all(tuple(args.tours))
    per_match.to_parquet(PROCESSED / "ratings.parquet", index=False)
    current.to_parquet(PROCESSED / "ratings_current.parquet", index=False)
    print(f"  [ratings] wrote {len(per_match):,} match rows, {len(current):,} rating rows")


if __name__ == "__main__":
    main()
