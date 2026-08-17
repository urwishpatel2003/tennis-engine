"""
Serve/return quality, decomposed into first and second serve.

Inputs : data/raw/matches_{tour}.parquet
Outputs: data/processed/serve_split.parquet         — pre-match values per match
         data/processed/serve_split_current.parquet — latest snapshot per player

What this adds over engine/serve_return.py
------------------------------------------
`serve_return.py` tracks one number per side: share of service points won, and
share of return points won. That aggregate hides the SHAPE of a serve. Two
players can both win 64% of service points while being completely different
propositions:

    A: lands 55% of first serves, wins 78% of them, wins 47% behind the second
    B: lands 70% of first serves, wins 70% of them, wins 50% behind the second

Against a returner who is ordinary on first serves but brutal on seconds, A is far
more exposed — A plays 45% of points on a second serve, B only 30%. The aggregate
model prices those two identically. This one does not.

Six quantities are tracked per player, each as an EWMA excess over the (tour,
surface) baseline, opponent-adjusted and shrunk by sample size exactly as in
serve_return.py:

    serving   first_in      how often the first serve lands
              first_won     points won when it does
              second_won    points won behind the second serve
    returning ret_first      points won against an opponent's first serve
              ret_second     points won against an opponent's second serve
              (plus ace_rate and df_rate, carried for display and for the
               variance they imply, not yet used in the point model)

Combination at prediction time:

    p(server wins point) = first_in x (base_first_won + first_won_s - ret_first_r)
                         + (1 - first_in) x (base_second_won + second_won_s - ret_second_r)

The returner barely affects whether a serve LANDS, so first_in carries no opponent
term — only the two conditional win rates do.
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

from engine.schema import PROCESSED, RAW, SURFACES, TOURS, base_split

# Same memory characteristics as serve_return.py — serve mechanics are stable.
HALFLIFE_MATCHES = 30.0
HALFLIFE_MATCHES_SURFACE = 20.0
TIME_DECAY_PER_YEAR = 0.35
SURFACE_BLEND = 0.50

# Shrinkage priors, in the natural denominator for each metric. first_won is
# measured over first serves landed, second_won over second serves — roughly 60/40
# of service points — so their priors differ accordingly.
PRIOR = {
    "first_in": 1500.0,     # service points
    "first_won": 950.0,     # first serves landed
    "second_won": 600.0,    # second serves
    "ret_first": 950.0,
    "ret_second": 600.0,
    "ace_rate": 1500.0,
    "df_rate": 1500.0,
}

# Hard caps. Beyond these you are outside anything seen on tour and it is a data
# artefact rather than a player.
CAP = {
    "first_in": 0.12, "first_won": 0.13, "second_won": 0.13,
    "ret_first": 0.13, "ret_second": 0.13, "ace_rate": 0.12, "df_rate": 0.10,
}

METRICS = ("first_in", "first_won", "second_won", "ret_first", "ret_second",
           "ace_rate", "df_rate")


def _alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / halflife)


class _Book:
    """Rolling excesses for one scope (overall, or one surface)."""

    def __init__(self, halflife: float) -> None:
        self.alpha = _alpha(halflife)
        self.val: dict[int, dict[str, float]] = {}
        self.den: dict[int, dict[str, float]] = {}
        self.n: dict[int, int] = {}
        self.last: dict[int, pd.Timestamp] = {}

    def prior(self, pid: int, when: pd.Timestamp) -> dict[str, float]:
        v = self.val.get(pid)
        if not v:
            return {m: 0.0 for m in METRICS}
        d = self.den.get(pid, {})
        decay = 1.0
        last = self.last.get(pid)
        if last is not None and when is not None:
            years = max((when - last).days, 0) / 365.25
            if years > 0.25:
                decay = math.exp(-TIME_DECAY_PER_YEAR * years)
        out = {}
        for m in METRICS:
            raw = v.get(m, 0.0) * decay
            seen = d.get(m, 0.0)
            shrunk = raw * (seen / (seen + PRIOR[m])) if seen > 0 else 0.0
            out[m] = float(np.clip(shrunk, -CAP[m], CAP[m]))
        return out

    def seen(self, pid: int, metric: str) -> float:
        return self.den.get(pid, {}).get(metric, 0.0)

    def update(self, pid: int, obs: dict[str, float], den: dict[str, float],
               when: pd.Timestamp) -> None:
        a = self.alpha
        v = self.val.setdefault(pid, {m: 0.0 for m in METRICS})
        d = self.den.setdefault(pid, {m: 0.0 for m in METRICS})
        for m in METRICS:
            if m in obs and np.isfinite(obs[m]) and den.get(m, 0) > 0:
                v[m] = (1 - a) * v[m] + a * obs[m]
                d[m] += den[m]
        self.n[pid] = self.n.get(pid, 0) + 1
        self.last[pid] = when


def _observe(svpt, fin, fwon, swon, ace, df) -> tuple[dict, dict] | None:
    """Raw rates and their denominators from one player's service line."""
    if not all(np.isfinite(x) for x in (svpt, fin, fwon, swon)):
        return None
    second = svpt - fin
    if svpt < 20 or fin <= 0 or second <= 0:
        return None
    obs = {
        "first_in": fin / svpt,
        "first_won": fwon / fin,
        "second_won": swon / second,
        "ace_rate": (ace / svpt) if np.isfinite(ace) else np.nan,
        "df_rate": (df / svpt) if np.isfinite(df) else np.nan,
    }
    den = {"first_in": svpt, "first_won": fin, "second_won": second,
           "ace_rate": svpt, "df_rate": svpt}
    return obs, den


def build(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single chronological pass — same no-leakage guarantee as serve_return.py."""
    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    tour = matches["tour"].iloc[0] if len(matches) else "atp"

    overall = _Book(HALFLIFE_MATCHES)
    per_surface = {s: _Book(HALFLIFE_MATCHES_SURFACE) for s in SURFACES}

    rows = []
    for r in matches.itertuples(index=False):
        w, l = int(r.winner_id), int(r.loser_id)
        surf, when = r.surface, r.tourney_date
        sb = per_surface[surf]
        base = base_split(r.tour, surf)

        # ── pre-match snapshot, surface blended onto overall ──────────────────
        snap = {}
        for pid, tag in ((w, "w"), (l, "l")):
            ov = overall.prior(pid, when)
            sf = sb.prior(pid, when)
            has_surface = sb.seen(pid, "first_in") > 0
            blended = {
                m: (SURFACE_BLEND * sf[m] + (1 - SURFACE_BLEND) * ov[m])
                if has_surface else ov[m]
                for m in METRICS
            }
            snap[tag] = blended
            for m in METRICS:
                rows_key = f"{tag}_{m}"
                snap.setdefault("_flat", {})[rows_key] = blended[m]

        rows.append({
            "match_id": r.match_id, "tour": r.tour, "tourney_date": when,
            "season": r.season, "surface": surf,
            "winner_id": w, "loser_id": l,
            **snap["_flat"],
            "w_svpt_seen": overall.seen(w, "first_in"),
            "l_svpt_seen": overall.seen(l, "first_in"),
        })

        # ── update from this match ────────────────────────────────────────────
        if not bool(r.completed):
            continue
        ow = _observe(r.w_svpt, r.w_1stIn, r.w_1stWon, r.w_2ndWon, r.w_ace, r.w_df)
        ol = _observe(r.l_svpt, r.l_1stIn, r.l_1stWon, r.l_2ndWon, r.l_ace, r.l_df)
        if ow is None or ol is None:
            continue
        (obs_w, den_w), (obs_l, den_l) = ow, ol

        # Opponent-adjust with the opponent's PRIOR return strength, and vice
        # versa — the same leak-free trick serve_return.py uses.
        pw, pl = snap["w"], snap["l"]

        serve_w = {
            "first_in": obs_w["first_in"] - base["first_in"],
            "first_won": (obs_w["first_won"] - base["first_won"]) + pl["ret_first"],
            "second_won": (obs_w["second_won"] - base["second_won"]) + pl["ret_second"],
            "ace_rate": obs_w["ace_rate"] - 0.0,
            "df_rate": obs_w["df_rate"] - 0.0,
        }
        serve_l = {
            "first_in": obs_l["first_in"] - base["first_in"],
            "first_won": (obs_l["first_won"] - base["first_won"]) + pw["ret_first"],
            "second_won": (obs_l["second_won"] - base["second_won"]) + pw["ret_second"],
            "ace_rate": obs_l["ace_rate"] - 0.0,
            "df_rate": obs_l["df_rate"] - 0.0,
        }
        # Return excess is the mirror of what the opponent achieved on serve.
        ret_w = {
            "ret_first": ((1 - obs_l["first_won"]) - (1 - base["first_won"]))
                         + pl["first_won"],
            "ret_second": ((1 - obs_l["second_won"]) - (1 - base["second_won"]))
                          + pl["second_won"],
        }
        ret_l = {
            "ret_first": ((1 - obs_w["first_won"]) - (1 - base["first_won"]))
                         + pw["first_won"],
            "ret_second": ((1 - obs_w["second_won"]) - (1 - base["second_won"]))
                          + pw["second_won"],
        }

        den_ret_w = {"ret_first": den_l["first_won"], "ret_second": den_l["second_won"]}
        den_ret_l = {"ret_first": den_w["first_won"], "ret_second": den_w["second_won"]}

        for book in (overall, sb):
            book.update(w, {**serve_w, **ret_w}, {**den_w, **den_ret_w}, when)
            book.update(l, {**serve_l, **ret_l}, {**den_l, **den_ret_l}, when)

    per_match = pd.DataFrame(rows)

    snap_rows = []
    now = matches["tourney_date"].max() if len(matches) else pd.Timestamp.today()
    for scope, book in [("overall", overall)] + list(per_surface.items()):
        for pid in book.val:
            v = book.prior(pid, now)
            snap_rows.append({
                "tour": tour, "player_id": pid, "surface": scope,
                **v,
                "svpt_seen": book.seen(pid, "first_in"),
                "matches": book.n.get(pid, 0),
                "last_played": book.last.get(pid),
            })
    return per_match, pd.DataFrame(snap_rows)


# ──────────────────────────────────────────────────────────────────────────────
def point_prob_split(server: dict, returner: dict, tour: str, surface: str) -> float:
    """
    P(server wins a point), from the decomposed model.

    `server` and `returner` are excess dicts (the six METRICS). The returner has
    no term in first_in — how often a serve lands is the server's business.
    """
    base = base_split(tour, surface)
    fi = float(np.clip(base["first_in"] + server.get("first_in", 0.0), 0.30, 0.90))
    fw = base["first_won"] + server.get("first_won", 0.0) - returner.get("ret_first", 0.0)
    sw = base["second_won"] + server.get("second_won", 0.0) - returner.get("ret_second", 0.0)
    fw = float(np.clip(fw, 0.25, 0.95))
    sw = float(np.clip(sw, 0.15, 0.90))
    return float(np.clip(fi * fw + (1.0 - fi) * sw, 0.35, 0.85))


def build_all(tours: tuple[str, ...] = TOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm, cur = [], []
    for tour in tours:
        path = RAW / f"matches_{tour}.parquet"
        if not path.exists():
            print(f"  [serve_split] no {path.name}, skipping {tour}")
            continue
        m = pd.read_parquet(path)
        print(f"  [serve_split] {tour}: {len(m):,} matches")
        a, b = build(m)
        pm.append(a)
        cur.append(b)
    if not pm:
        raise FileNotFoundError(f"No match parquets in {RAW}.")
    return pd.concat(pm, ignore_index=True), pd.concat(cur, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the split serve/return books.")
    ap.add_argument("--tours", nargs="+", default=list(TOURS))
    args = ap.parse_args()
    pm, cur = build_all(tuple(args.tours))
    pm.to_parquet(PROCESSED / "serve_split.parquet", index=False)
    cur.to_parquet(PROCESSED / "serve_split_current.parquet", index=False)
    print(f"  [serve_split] wrote {len(pm):,} match rows, {len(cur):,} snapshot rows")


if __name__ == "__main__":
    main()
