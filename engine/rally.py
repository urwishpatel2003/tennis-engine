"""
Rally quality — winners, unforced errors and net play, as a pre-match profile.

Inputs : data/raw/matches_{tour}.parquet   (needs the RALLY_STAT_COLS columns —
         see tools/backfill_rally_stats.py)
Outputs: data/processed/rally.parquet          — pre-match profile per match
         data/processed/rally_current.parquet  — latest snapshot per player

Two warnings that govern everything here
----------------------------------------
1. **These statistics are mostly an OUTCOME, not a predictor.** Measured over the
   8,112 ATP matches that carry them, the match WINNER averages a 1.39
   winner/unforced ratio and the LOSER 0.89. Averaging a player's raw numbers
   therefore mostly measures how often they have been winning, which the ratings
   already know far more reliably. The only defensible use is a strictly
   leak-free rolling profile, opponent-adjusted, which is what this builds — the
   same machinery as serve_return.py, for the same reason.

2. **Coverage is thin and recent.** Nothing exists before 2021, and even in 2026
   only 58% of ATP matches carry it; the WTA feed carries none. Every consumer
   gets `matches_seen` alongside the numbers so a profile built on four matches is
   never displayed as though it were built on forty.

Metrics, all as rates over total points played:
    winner_rate   winners hit
    ue_rate       unforced errors made
    net_freq      how often the player comes to the net
    net_win       share of those net points won
    aggression    winner_rate + ue_rate — how much of the match this player
                  decides outright, in either direction. A high-aggression player
                  shortens points; that is a style fact independent of quality.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import PROCESSED, RAW, SURFACES, TOURS

# Measured over the 8,112 matches that carry these stats (2021-2026, ATP).
BASE = {"winner_rate": 0.1469, "ue_rate": 0.1347,
        "net_freq": 0.0971, "net_win": 0.6734}

# Surface shifts, also measured. Clay produces the most unforced errors and grass
# the fewest — longer rallies, more chances to miss — while grass yields the most
# winners. This is real tennis, not noise.
SURFACE_SHIFT = {
    "Clay":   {"winner_rate": +0.001, "ue_rate": +0.021},
    "Grass":  {"winner_rate": +0.010, "ue_rate": -0.015},
    "Hard":   {"winner_rate": -0.002, "ue_rate": -0.007},
    "Carpet": {"winner_rate": +0.008, "ue_rate": -0.012},
}

METRICS = ("winner_rate", "ue_rate", "net_freq", "net_win")

HALFLIFE_MATCHES = 20.0      # shorter than serve: far fewer observations exist
TIME_DECAY_PER_YEAR = 0.35
# Shrinkage prior in POINTS. A player with 400 points of rally data (~3 matches)
# should sit near the tour baseline, not at whatever their tiny sample says.
PRIOR_POINTS = 1200.0
CAP = 0.09


def base_for(surface: str) -> dict:
    d = SURFACE_SHIFT.get(surface, {})
    return {k: BASE[k] + d.get(k, 0.0) for k in METRICS}


class _Book:
    def __init__(self, halflife: float = HALFLIFE_MATCHES) -> None:
        self.alpha = 1.0 - 0.5 ** (1.0 / halflife)
        self.val: dict[int, dict] = {}
        self.pts: dict[int, float] = {}
        self.n: dict[int, int] = {}
        self.last: dict[int, pd.Timestamp] = {}

    def prior(self, pid: int, when) -> dict:
        v = self.val.get(pid)
        if not v:
            return {m: 0.0 for m in METRICS}
        seen = self.pts.get(pid, 0.0)
        decay = 1.0
        last = self.last.get(pid)
        if last is not None and when is not None:
            years = max((when - last).days, 0) / 365.25
            if years > 0.25:
                decay = math.exp(-TIME_DECAY_PER_YEAR * years)
        shrink = seen / (seen + PRIOR_POINTS) if seen > 0 else 0.0
        return {m: float(np.clip(v.get(m, 0.0) * decay * shrink, -CAP, CAP))
                for m in METRICS}

    def seen(self, pid: int) -> float:
        return self.pts.get(pid, 0.0)

    def matches(self, pid: int) -> int:
        return self.n.get(pid, 0)

    def update(self, pid: int, obs: dict, points: float, when) -> None:
        a = self.alpha
        v = self.val.setdefault(pid, {m: 0.0 for m in METRICS})
        for m in METRICS:
            if m in obs and np.isfinite(obs[m]):
                v[m] = (1 - a) * v[m] + a * obs[m]
        self.pts[pid] = self.pts.get(pid, 0.0) + points
        self.n[pid] = self.n.get(pid, 0) + 1
        self.last[pid] = when


def _observe(winners, ue, net_w, net_t, tp, base) -> dict | None:
    if not np.isfinite(tp) or tp <= 20 or not np.isfinite(winners) or not np.isfinite(ue):
        return None
    o = {
        "winner_rate": winners / tp - base["winner_rate"],
        "ue_rate": ue / tp - base["ue_rate"],
    }
    if np.isfinite(net_t) and net_t > 0:
        o["net_freq"] = net_t / tp - base["net_freq"]
        if np.isfinite(net_w):
            o["net_win"] = net_w / net_t - base["net_win"]
    return o


def build(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "w_winners" not in matches.columns:
        return pd.DataFrame(), pd.DataFrame()

    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    tour = matches["tour"].iloc[0] if len(matches) else "atp"

    overall = _Book()
    per_surface = {s: _Book() for s in SURFACES}

    rows = []
    for r in matches.itertuples(index=False):
        w, l = int(r.winner_id), int(r.loser_id)
        surf, when = r.surface, r.tourney_date
        sb = per_surface[surf]
        base = base_for(surf)

        snap = {}
        for pid, tag in ((w, "w"), (l, "l")):
            ov = overall.prior(pid, when)
            sf = sb.prior(pid, when) if sb.seen(pid) > 0 else ov
            blended = {m: 0.5 * sf[m] + 0.5 * ov[m] for m in METRICS}
            snap[tag] = blended
            for m in METRICS:
                snap.setdefault("_flat", {})[f"{tag}_{m}"] = blended[m]
            snap["_flat"][f"{tag}_rally_matches"] = overall.matches(pid)

        rows.append({
            "match_id": r.match_id, "tour": r.tour, "tourney_date": when,
            "season": r.season, "surface": surf,
            "winner_id": w, "loser_id": l, **snap["_flat"],
        })

        # Only completed matches with a real stat line teach anything.
        if not bool(r.completed):
            continue
        ow = _observe(r.w_winners, r.w_unforced, r.w_netWon, r.w_netTotal, r.w_tp, base)
        ol = _observe(r.l_winners, r.l_unforced, r.l_netWon, r.l_netTotal, r.l_tp, base)
        if ow is None or ol is None:
            continue
        for book in (overall, sb):
            book.update(w, ow, float(r.w_tp), when)
            book.update(l, ol, float(r.l_tp), when)

    per_match = pd.DataFrame(rows)

    snap_rows = []
    now = matches["tourney_date"].max() if len(matches) else pd.Timestamp.today()
    for scope, book in [("overall", overall)] + list(per_surface.items()):
        for pid in book.val:
            v = book.prior(pid, now)
            snap_rows.append({
                "tour": tour, "player_id": pid, "surface": scope, **v,
                "points_seen": book.seen(pid),
                "rally_matches": book.matches(pid),
                "last_played": book.last.get(pid),
            })
    return per_match, pd.DataFrame(snap_rows)


def describe(profile: dict, surface: str = "Hard") -> dict:
    """Turn excesses into displayable rates plus a plain-language style label."""
    base = base_for(surface)
    out = {m: round(base[m] + float(profile.get(m, 0.0) or 0.0), 4) for m in METRICS}
    out["aggression"] = round(out["winner_rate"] + out["ue_rate"], 4)
    wr, ue = profile.get("winner_rate", 0.0) or 0.0, profile.get("ue_rate", 0.0) or 0.0
    # Style, not quality: a big hitter and a grinder can be equally good.
    if wr > 0.012 and ue > 0.008:
        out["style"] = "high-risk aggressor"
    elif wr > 0.012:
        out["style"] = "clean ball-striker"
    elif ue < -0.008 and wr < 0.004:
        out["style"] = "counterpuncher"
    elif ue < -0.008:
        out["style"] = "consistent"
    else:
        out["style"] = "balanced"
    return out


def build_all(tours: tuple[str, ...] = TOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm, cur = [], []
    for tour in tours:
        p = RAW / f"matches_{tour}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        if "w_winners" not in m.columns or not m["w_winners"].notna().any():
            print(f"  [rally] {tour}: no rally statistics in this archive, skipping")
            continue
        a, b = build(m)
        if not a.empty:
            n = int(m["w_winners"].notna().sum())
            print(f"  [rally] {tour}: {n:,} matches with rally stats "
                  f"({n/len(m)*100:.1f}% of archive)")
            pm.append(a)
            cur.append(b)
    if not pm:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(pm, ignore_index=True), pd.concat(cur, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build rally-quality profiles.")
    ap.add_argument("--tours", nargs="+", default=list(TOURS))
    args = ap.parse_args()
    pm, cur = build_all(tuple(args.tours))
    if pm.empty:
        print("  [rally] nothing to build")
        return
    pm.to_parquet(PROCESSED / "rally.parquet", index=False)
    cur.to_parquet(PROCESSED / "rally_current.parquet", index=False)
    print(f"  [rally] wrote {len(pm):,} match rows, {len(cur):,} snapshot rows")


if __name__ == "__main__":
    main()
