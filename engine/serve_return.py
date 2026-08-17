"""
Serve and return quality — the point-level engine behind the score simulation.

Inputs : data/raw/matches_{tour}.parquet
Outputs: data/processed/serve_return.parquet         — pre-match rolling serve/return
                                                        strength for both players, per match
         data/processed/serve_return_current.parquet — latest snapshot per
                                                        (tour, player, surface|overall)

What is being measured
----------------------
Everything is expressed as an EXCESS over the (tour, surface) baseline, never as a
raw rate. Raw serve numbers are not comparable across tours or surfaces — 64% of
service points won is unremarkable for an ATP player on hard and outstanding for a
WTA player on clay. Working in excess space means one scale for everyone:

    serve_excess  = spw - base_spw(tour, surface)
    return_excess = rpw - (1 - base_spw(tour, surface))

Opponent adjustment (and why it is leak-free)
---------------------------------------------
A raw serve line is contaminated by who was returning. Because this module walks
the match log in strict chronological order, at the moment of match *i* we already
hold the opponent's return_excess as it stood BEFORE match *i* — so we can correct
the observation without ever looking forward:

    adjusted_serve_excess_i = (spw_i - base_i) + opp_return_excess_prior
    adjusted_return_excess_i = (rpw_i - (1 - base_i)) + opp_serve_excess_prior

A player who won 66% of service points against an elite returner did better than
the raw number says, so the opponent's (positive) return excess is added back.

Combining two players at prediction time
----------------------------------------
The additive Barnett/Sackmann form:

    P(A wins a point on A's serve) = base(tour, surface)
                                     + serve_excess_A
                                     - return_excess_B

which is what `point_probabilities()` returns, ready for engine.markov.
"""

from __future__ import annotations

import argparse
import math

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow both `python engine/serve_return.py` and `python -m engine.serve_return`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import PROCESSED, RAW, SURFACES, TOURS, base_spw

# ──────────────────────────────────────────────────────────────────────────────
# Tuned constants
# ──────────────────────────────────────────────────────────────────────────────
# EWMA half-life in matches. Serve quality is a stable skill (technique changes
# slowly), so it gets a long memory; 30 matches is roughly one season of a busy
# tour schedule. The surface book gets a shorter half-life only because it sees
# far fewer matches and would otherwise lag a genuine surface adaptation.
HALFLIFE_MATCHES = 30.0
HALFLIFE_MATCHES_SURFACE = 20.0

# Extra decay for time away, applied per year of gap between matches. Form is not
# preserved in amber across a 14-month injury absence.
TIME_DECAY_PER_YEAR = 0.35

# Shrinkage prior, in service points. A player with 300 career service points
# (~4 matches) should sit close to the tour baseline, not at whatever extreme
# their tiny sample suggests. 1500 ≈ 20 matches.
PRIOR_SVPT = 1500.0
PRIOR_RTPT = 1500.0

# Surface/overall blend, matching the philosophy in ratings.py: surface identity
# is real but sparse, so it is blended rather than used alone.
SURFACE_BLEND = 0.50

# Hard caps on the final excess values. Beyond ±0.12 of service points won you are
# outside anything observed on tour, and it is almost always a data artefact.
MAX_SERVE_EXCESS = 0.12
MAX_RETURN_EXCESS = 0.12


def _alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / halflife)


class _Book:
    """Rolling serve/return excess for one scope (overall, or one surface)."""

    def __init__(self, halflife: float) -> None:
        self.alpha = _alpha(halflife)
        self.serve: dict[int, float] = {}
        self.ret: dict[int, float] = {}
        self.svpt: dict[int, float] = {}
        self.rtpt: dict[int, float] = {}
        self.n: dict[int, int] = {}
        self.last: dict[int, pd.Timestamp] = {}

    def prior(self, pid: int, when: pd.Timestamp) -> tuple[float, float, float, float]:
        """
        Pre-match (serve_excess, return_excess, svpt_seen, rtpt_seen), shrunk toward
        the baseline by sample size and decayed for time away.
        """
        s = self.serve.get(pid, 0.0)
        r = self.ret.get(pid, 0.0)
        sv = self.svpt.get(pid, 0.0)
        rt = self.rtpt.get(pid, 0.0)

        last = self.last.get(pid)
        if last is not None and when is not None:
            years = max((when - last).days, 0) / 365.25
            if years > 0.25:  # a normal 2-3 week gap between events is not "time away"
                decay = math.exp(-TIME_DECAY_PER_YEAR * years)
                s *= decay
                r *= decay

        s_shrunk = s * (sv / (sv + PRIOR_SVPT)) if sv > 0 else 0.0
        r_shrunk = r * (rt / (rt + PRIOR_RTPT)) if rt > 0 else 0.0
        return (
            float(np.clip(s_shrunk, -MAX_SERVE_EXCESS, MAX_SERVE_EXCESS)),
            float(np.clip(r_shrunk, -MAX_RETURN_EXCESS, MAX_RETURN_EXCESS)),
            sv,
            rt,
        )

    def update(
        self, pid: int, serve_exc: float, ret_exc: float,
        svpt: float, rtpt: float, when: pd.Timestamp,
    ) -> None:
        a = self.alpha
        self.serve[pid] = (1 - a) * self.serve.get(pid, 0.0) + a * serve_exc
        self.ret[pid] = (1 - a) * self.ret.get(pid, 0.0) + a * ret_exc
        self.svpt[pid] = self.svpt.get(pid, 0.0) + svpt
        self.rtpt[pid] = self.rtpt.get(pid, 0.0) + rtpt
        self.n[pid] = self.n.get(pid, 0) + 1
        self.last[pid] = when


def build_serve_return(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single chronological pass producing pre-match serve/return strength."""
    matches = matches.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    tour = matches["tour"].iloc[0] if len(matches) else "atp"

    overall = _Book(HALFLIFE_MATCHES)
    per_surface = {s: _Book(HALFLIFE_MATCHES_SURFACE) for s in SURFACES}

    rows = []
    for r in matches.itertuples(index=False):
        w, l = int(r.winner_id), int(r.loser_id)
        surf, when = r.surface, r.tourney_date
        sb = per_surface[surf]
        base = base_spw(r.tour, surf)

        # ── Pre-match snapshot ────────────────────────────────────────────────
        w_s_ov, w_r_ov, w_sv_ov, w_rt_ov = overall.prior(w, when)
        l_s_ov, l_r_ov, l_sv_ov, l_rt_ov = overall.prior(l, when)
        w_s_sf, w_r_sf, w_sv_sf, w_rt_sf = sb.prior(w, when)
        l_s_sf, l_r_sf, l_sv_sf, l_rt_sf = sb.prior(l, when)

        # A player with no history on this surface falls back to their overall
        # book rather than to the flat baseline.
        if w_sv_sf <= 0:
            w_s_sf, w_r_sf = w_s_ov, w_r_ov
        if l_sv_sf <= 0:
            l_s_sf, l_r_sf = l_s_ov, l_r_ov

        w_serve = SURFACE_BLEND * w_s_sf + (1 - SURFACE_BLEND) * w_s_ov
        w_return = SURFACE_BLEND * w_r_sf + (1 - SURFACE_BLEND) * w_r_ov
        l_serve = SURFACE_BLEND * l_s_sf + (1 - SURFACE_BLEND) * l_s_ov
        l_return = SURFACE_BLEND * l_r_sf + (1 - SURFACE_BLEND) * l_r_ov

        rows.append(
            {
                "match_id": r.match_id, "tour": r.tour, "tourney_date": when,
                "season": r.season, "surface": surf,
                "winner_id": w, "loser_id": l,
                "w_serve_excess": w_serve, "w_return_excess": w_return,
                "l_serve_excess": l_serve, "l_return_excess": l_return,
                "w_serve_excess_overall": w_s_ov, "w_return_excess_overall": w_r_ov,
                "l_serve_excess_overall": l_s_ov, "l_return_excess_overall": l_r_ov,
                "w_svpt_seen": w_sv_ov, "l_svpt_seen": l_sv_ov,
                "base_spw": base,
            }
        )

        # ── Update from this match's observation ──────────────────────────────
        # Only completed matches with real stat lines teach us anything.
        w_spw, l_spw = r.w_spw, r.l_spw
        if not bool(r.completed) or not np.isfinite(w_spw) or not np.isfinite(l_spw):
            continue
        w_svpt = float(r.w_svpt) if np.isfinite(r.w_svpt) else 0.0
        l_svpt = float(r.l_svpt) if np.isfinite(r.l_svpt) else 0.0
        if w_svpt < 20 or l_svpt < 20:
            continue  # truncated stat line, not a usable observation

        w_rpw, l_rpw = 1.0 - l_spw, 1.0 - w_spw

        # Opponent-adjust using the opponent's PRIOR (pre-match) strength.
        w_serve_obs = (w_spw - base) + l_return
        l_serve_obs = (l_spw - base) + w_return
        w_ret_obs = (w_rpw - (1.0 - base)) + l_serve
        l_ret_obs = (l_rpw - (1.0 - base)) + w_serve

        overall.update(w, w_serve_obs, w_ret_obs, w_svpt, l_svpt, when)
        overall.update(l, l_serve_obs, l_ret_obs, l_svpt, w_svpt, when)
        sb.update(w, w_serve_obs, w_ret_obs, w_svpt, l_svpt, when)
        sb.update(l, l_serve_obs, l_ret_obs, l_svpt, w_svpt, when)

    per_match = pd.DataFrame(rows)

    # ── Current snapshot ──────────────────────────────────────────────────────
    snap = []
    now = matches["tourney_date"].max() if len(matches) else pd.Timestamp.today()
    for pid in overall.serve:
        s, rr, sv, rt = overall.prior(pid, now)
        snap.append(
            {
                "tour": tour, "player_id": pid, "surface": "overall",
                "serve_excess": s, "return_excess": rr,
                "svpt_seen": sv, "rtpt_seen": rt,
                "matches": overall.n.get(pid, 0), "last_played": overall.last.get(pid),
            }
        )
    for surf, book in per_surface.items():
        for pid in book.serve:
            s, rr, sv, rt = book.prior(pid, now)
            snap.append(
                {
                    "tour": tour, "player_id": pid, "surface": surf,
                    "serve_excess": s, "return_excess": rr,
                    "svpt_seen": sv, "rtpt_seen": rt,
                    "matches": book.n.get(pid, 0), "last_played": book.last.get(pid),
                }
            )
    return per_match, pd.DataFrame(snap)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction-time helper
# ──────────────────────────────────────────────────────────────────────────────
def point_probabilities(
    serve_a: float, return_a: float,
    serve_b: float, return_b: float,
    tour: str, surface: str,
) -> tuple[float, float]:
    """
    Additive serve/return combination → (P(A wins point on A's serve),
    P(B wins point on B's serve)).

    Clamped to [0.35, 0.85]: outside that band the Markov chain produces hold
    probabilities that never occur in professional tennis, and it is always a
    symptom of a bad input rather than a real matchup.
    """
    base = base_spw(tour, surface)
    pa = base + serve_a - return_b
    pb = base + serve_b - return_a
    return (
        float(np.clip(pa, 0.35, 0.85)),
        float(np.clip(pb, 0.35, 0.85)),
    )


def build_all(tours: tuple[str, ...] = TOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm, cur = [], []
    for tour in tours:
        path = RAW / f"matches_{tour}.parquet"
        if not path.exists():
            print(f"  [serve_return] no {path.name}, skipping {tour}")
            continue
        m = pd.read_parquet(path)
        print(f"  [serve_return] {tour}: {len(m):,} matches")
        a, b = build_serve_return(m)
        pm.append(a)
        cur.append(b)
    if not pm:
        raise FileNotFoundError(
            f"No match parquets found in {RAW}. Run `python fetch_data.py` first."
        )
    return pd.concat(pm, ignore_index=True), pd.concat(cur, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build serve/return strength books.")
    ap.add_argument("--tours", nargs="+", default=list(TOURS))
    args = ap.parse_args()

    per_match, current = build_all(tuple(args.tours))
    per_match.to_parquet(PROCESSED / "serve_return.parquet", index=False)
    current.to_parquet(PROCESSED / "serve_return_current.parquet", index=False)
    print(f"  [serve_return] wrote {len(per_match):,} match rows, {len(current):,} snapshot rows")


if __name__ == "__main__":
    main()
