"""
The prediction engine — turns two player names into a full match forecast.

Inputs : data/processed/{ratings_current, serve_return_current, h2h, conditions}.parquet
         data/raw/players_{tour}.parquet
Outputs: a prediction dict (and data/processed/predictions_*.parquet from run_engine.py)

How a prediction is assembled
-----------------------------
Two independent views of the match are formed and then reconciled:

  A. **Rating view.** Blended surface Elo, adjusted in Elo points for conditions
     (fatigue, rest, home crowd), handedness and head-to-head surprise. Gives
     `p_elo`.

  B. **Point view.** Each player's serve and return excess (surface-blended,
     opponent-adjusted) is shifted for height/style and venue physics, combined
     additively into two point-on-serve probabilities, and run through the exact
     Barnett-Clarke chain in engine.markov. Gives `p_sim` plus a full score
     distribution.

These are blended in LOG-ODDS space (averaging raw probabilities systematically
drags confident predictions toward the middle). When a market price is supplied,
it is blended in too.

  C. **Reconciliation.** The blended probability is not what the raw point rates
     produce, so `markov.invert_to_target` solves for the symmetric shift in the
     two serve probabilities that reproduces the headline number exactly. Only
     then are the set-score, game-handicap and total-games outputs read off. This
     is what stops the engine from saying "Alcaraz 78%" and simultaneously
     offering a games line that implies 64%.

Fallbacks
---------
Every lookup degrades rather than raises: unknown surface → overall book, unknown
player → tour-average newcomer, missing height → no style shift. A prediction with
sparse data is still returned, with `data_quality` flagging how thin it was.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Allow both `python engine/predict.py` and `python -m engine.predict`. Run as a
# script the package root is not on sys.path, so put it there before importing.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import conditions as cond
from engine import matchups as mu
from engine import markov
from engine.ratings import SURFACE_BLEND as ELO_SURFACE_BLEND
from engine.ratings import blended_elo, expected_score
from engine.schema import (
    ELO_INIT,
    ELO_SCALE,
    PROCESSED,
    RAW,
    SURFACES,
    base_spw,
    normalise_surface,
)
from engine.serve_return import point_probabilities

# ──────────────────────────────────────────────────────────────────────────────
# Blend weights
# ──────────────────────────────────────────────────────────────────────────────
# Elo carries more weight than the point model: it sees every match ever played,
# while serve/return stats only exist for matches with a recorded stat line and
# are noisier match to match. The point model earns its keep by producing the
# score distribution and by encoding surface/style texture Elo cannot express.
W_ELO = 0.62

# When a market price is available, weight it heavily. Tennis match markets are
# efficient — this mirrors the finding already recorded for the NFL engine, and
# the honest use of the blend is a better-calibrated forecast, not an edge.
W_MARKET = 0.55

# Global multiplier on the Elo gap before it becomes a probability. 1.0 means the
# rating spread is taken at face value; below 1.0 damps an over-confident book.
#
# LEAVE THIS AT 1.0 UNTIL IT HAS BEEN FIT ON REAL DATA. `backtest.py` reports the
# optimal value on whatever archive is loaded ("optimal Elo spread multiplier").
#
# Read that number PER SEASON, never in aggregate. On the synthetic archive the
# pooled figure is 0.86, which reads as "the ratings are over-confident" — but the
# season breakdown runs 0.67 → 0.80 → 0.84 → 0.86 → 0.98 → 0.98, so the shortfall
# is entirely rating warm-up in the years right after the archive starts, and the
# spread is already correct once Elo has converged. Tuning to the pooled number
# would permanently damp mature ratings to compensate for a transient.
ELO_SPREAD_MULT = 1.0


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blend_logit(probs: list[float], weights: list[float]) -> float:
    """Weighted average in log-odds space, renormalising the weights."""
    tw = sum(weights)
    if tw <= 0:
        return 0.5
    return _sigmoid(sum(w * _logit(p) for p, w in zip(probs, weights)) / tw)


# ──────────────────────────────────────────────────────────────────────────────
# Player state
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PlayerState:
    player_id: int
    name: str
    tour: str
    hand: str | None = None
    height: float | None = None
    ioc: str | None = None
    elo: float = ELO_INIT
    elo_surface: float = ELO_INIT
    matches: int = 0
    matches_surface: int = 0
    serve_excess: float = 0.0
    return_excess: float = 0.0
    svpt_seen: float = 0.0
    last_played: pd.Timestamp | None = None
    condition: dict = field(default_factory=dict)

    @property
    def elo_blend(self) -> float:
        return blended_elo(self.elo, self.elo_surface, ELO_SURFACE_BLEND)


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────
class Engine:
    """Loads the processed tables once and answers prediction queries."""

    def __init__(self, tours: tuple[str, ...] = ("atp", "wta")) -> None:
        self.tours = tours
        self.ratings = self._load(PROCESSED / "ratings_current.parquet")
        self.sr = self._load(PROCESSED / "serve_return_current.parquet")
        self.h2h = self._load(PROCESSED / "h2h.parquet")
        self.conditions = self._load(PROCESSED / "conditions.parquet")
        self.players = self._load_players()

        # Fast lookups
        self._elo_idx = self._index(self.ratings, ["tour", "player_id", "surface"])
        self._sr_idx = self._index(self.sr, ["tour", "player_id", "surface"])
        self._latest_cond = self._latest_conditions()

    # ── loading ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load(path) -> pd.DataFrame:
        if not path.exists():
            print(f"  [predict] WARNING: missing {path.name} — running degraded")
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _load_players(self) -> pd.DataFrame:
        frames = []
        for tour in self.tours:
            p = RAW / f"players_{tour}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["tour"] = tour
                frames.append(df)
        if not frames:
            return pd.DataFrame(
                columns=["player_id", "name", "hand", "height", "ioc", "tour"]
            )
        out = pd.concat(frames, ignore_index=True)
        out["name_lower"] = out["name"].astype(str).str.lower().str.strip()
        return out

    @staticmethod
    def _index(df: pd.DataFrame, keys: list[str]) -> dict:
        if df.empty:
            return {}
        return {
            tuple(r[k] for k in keys): r
            for r in df.to_dict("records")
        }

    def _latest_conditions(self) -> dict:
        """Most recent condition row per (tour, player) — the state they carry in."""
        if self.conditions.empty:
            return {}
        c = self.conditions.sort_values("tourney_date")
        last = c.groupby(["tour", "player_id"], as_index=False).tail(1)
        return {(r["tour"], int(r["player_id"])): r for r in last.to_dict("records")}

    # ── resolution ────────────────────────────────────────────────────────────
    def _match_counts(self) -> dict:
        """player_id -> career matches on record, for disambiguation."""
        if self.ratings.empty:
            return {}
        return (
            self.ratings[self.ratings["surface"] == "overall"]
            .set_index("player_id")["matches"].to_dict()
        )

    def resolve_player(self, who: str | int, tour: str) -> dict | None:
        """
        Resolve a player id, exact name, or name fragment.

        Ambiguity is resolved by career match count, and this applies to EXACT
        name matches too — which is not a nicety, it is a correctness fix.
        Upstream assigns the same person more than one player_id: Martin
        Landaluce exists as 211776 (1 match, 2022) and 212021 (28 matches,
        current), and roughly 1,800 player rows share a name with another row.
        Returning the first exact hit silently picked the orphaned id, so a
        perfectly well-known player came back with a near-default rating and a
        "low data quality" warning. Whichever id carries the real career wins.
        """
        if self.players.empty:
            return None
        pool = self.players[self.players["tour"] == tour]

        if isinstance(who, (int, np.integer)) or (isinstance(who, str) and who.isdigit()):
            hit = pool[pool["player_id"] == int(who)]
            return hit.iloc[0].to_dict() if len(hit) else None

        q = str(who).lower().strip()
        # Exact matches are strictly preferred over substring matches; only when
        # there are none do we fall back to a fragment search.
        candidates = pool[pool["name_lower"] == q]
        if candidates.empty:
            candidates = pool[pool["name_lower"].str.contains(q, regex=False, na=False)]
        if candidates.empty:
            return None
        if len(candidates) == 1:
            return candidates.iloc[0].to_dict()

        counts = self._match_counts()
        candidates = candidates.assign(
            _n=candidates["player_id"].map(counts).fillna(0)
        ).sort_values("_n", ascending=False)
        return candidates.iloc[0].to_dict()

    # ── state assembly ────────────────────────────────────────────────────────
    def player_state(
        self, who: str | int, tour: str, surface: str,
        as_of: pd.Timestamp | None = None,
    ) -> PlayerState:
        surface = normalise_surface(surface)
        rec = self.resolve_player(who, tour)
        if rec is None:
            raise KeyError(
                f"Could not resolve player {who!r} on the {tour.upper()} tour. "
                f"Try the full name as it appears in players_{tour}.parquet."
            )

        pid = int(rec["player_id"])
        st = PlayerState(
            player_id=pid,
            name=str(rec.get("name", who)),
            tour=tour,
            hand=rec.get("hand"),
            height=rec.get("height") if pd.notna(rec.get("height")) else None,
            ioc=rec.get("ioc"),
        )

        ov = self._elo_idx.get((tour, pid, "overall"))
        if ov:
            st.elo = float(ov["elo"])
            st.matches = int(ov["matches"])
            st.last_played = ov.get("last_played")
        sf = self._elo_idx.get((tour, pid, surface))
        # No history on this surface → fall back to the overall rating, not to 1500.
        st.elo_surface = float(sf["elo"]) if sf else st.elo
        st.matches_surface = int(sf["matches"]) if sf else 0

        sr_sf = self._sr_idx.get((tour, pid, surface))
        sr_ov = self._sr_idx.get((tour, pid, "overall"))
        src = sr_sf if (sr_sf and float(sr_sf.get("svpt_seen", 0)) > 0) else sr_ov
        if src:
            st.serve_excess = float(src["serve_excess"])
            st.return_excess = float(src["return_excess"])
            st.svpt_seen = float(src.get("svpt_seen", 0.0))

        c = self._latest_cond.get((tour, pid))
        if c is not None:
            st.condition = dict(c)
            # days_rest must be recomputed against the match date we are predicting,
            # not left at whatever it was for their last completed match.
            if as_of is not None and pd.notna(c.get("tourney_date")):
                st.condition["days_rest"] = (as_of - pd.Timestamp(c["tourney_date"])).days
        return st

    # ── the prediction ────────────────────────────────────────────────────────
    def predict(
        self,
        player_a: str | int,
        player_b: str | int,
        tour: str = "atp",
        surface: str = "Hard",
        best_of: int = 3,
        match_date: str | pd.Timestamp | None = None,
        indoor: bool = False,
        altitude: float = 0.0,
        final_set_tb: bool = True,
        market_prob_a: float | None = None,
        tournament: str | None = None,
    ) -> dict:
        """
        Full forecast for A vs B. See module docstring for the assembly order.

        `market_prob_a` should already be vig-free (see weekly_picks.devig).
        """
        surface = normalise_surface(surface)
        as_of = pd.Timestamp(match_date) if match_date is not None else pd.Timestamp.today()

        # Venue context can be inferred from the tournament name when not given.
        if tournament:
            from engine.schema import altitude_m, is_indoor
            indoor = indoor or is_indoor(tournament)
            altitude = altitude or altitude_m(tournament)

        a = self.player_state(player_a, tour, surface, as_of)
        b = self.player_state(player_b, tour, surface, as_of)

        # ── A. Rating view ────────────────────────────────────────────────────
        elo_gap = a.elo_blend - b.elo_blend
        adjustments: dict[str, float] = {}

        ca, parts_a = cond.conditions_elo_delta(a.condition)
        cb, parts_b = cond.conditions_elo_delta(b.condition)
        if ca or cb:
            adjustments["conditions"] = ca - cb

        hand_d = mu.handedness_elo_delta(a.hand, b.hand)
        if hand_d:
            adjustments["handedness"] = hand_d

        # Head-to-head is measured against what the ratings alone expected.
        p_pre_h2h = expected_score(a.elo_blend + sum(adjustments.values()), b.elo_blend)
        rec = (
            mu.h2h_record(self.h2h, a.player_id, b.player_id, before=as_of, surface=surface)
            if not self.h2h.empty else {"wins": 0.0, "losses": 0.0, "n": 0,
                                        "raw_wins": 0, "raw_losses": 0}
        )
        h2h_d = mu.h2h_elo_delta(rec, p_pre_h2h)
        if h2h_d:
            adjustments["head_to_head"] = h2h_d

        elo_gap_adj = elo_gap + sum(adjustments.values())
        p_elo = _sigmoid(ELO_SPREAD_MULT * elo_gap_adj * math.log(10.0) / ELO_SCALE)

        # ── B. Point view ─────────────────────────────────────────────────────
        hs_a, hr_a = mu.height_style_delta(a.height, tour, surface)
        hs_b, hr_b = mu.height_style_delta(b.height, tour, surface)

        # Venue physics move BOTH players' serve the same way.
        venue_serve = cond.altitude_serve_delta(altitude) + (
            cond.INDOOR_SERVE_BONUS if indoor else 0.0
        )

        pa_raw, pb_raw = point_probabilities(
            a.serve_excess + hs_a + venue_serve, a.return_excess + hr_a,
            b.serve_excess + hs_b + venue_serve, b.return_excess + hr_b,
            tour, surface,
        )
        sim_raw = markov.match_distribution(pa_raw, pb_raw, best_of, final_set_tb)
        p_sim = sim_raw["win_prob"]

        # ── Blend ─────────────────────────────────────────────────────────────
        probs, weights, sources = [p_elo, p_sim], [W_ELO, 1.0 - W_ELO], ["elo", "serve_return"]
        if market_prob_a is not None and 0.0 < market_prob_a < 1.0:
            probs.append(float(market_prob_a))
            weights = [W_ELO * (1 - W_MARKET), (1 - W_ELO) * (1 - W_MARKET), W_MARKET]
            sources.append("market")
        p_final = blend_logit(probs, weights)

        # ── C. Reconcile the score model to the headline probability ──────────
        pa_adj, pb_adj = markov.invert_to_target(
            pa_raw, pb_raw, p_final, best_of, final_set_tb
        )
        sim = markov.match_distribution(
            pa_adj, pb_adj, best_of, final_set_tb, track_scorelines=True
        )

        # ── Derived market views ──────────────────────────────────────────────
        totals = markov.total_games_distribution(sim["games"])
        fair_total = _fair_total_line(totals)
        fair_handicap = _fair_handicap_line(sim["games"])

        top_scorelines = sorted(
            sim.get("scorelines", {}).items(), key=lambda kv: -kv[1]
        )[:6]

        quality = _data_quality(a, b, surface)

        return {
            "tour": tour,
            "surface": surface,
            "best_of": best_of,
            "match_date": as_of.date().isoformat(),
            "tournament": tournament,
            "indoor": bool(indoor),
            "altitude": float(altitude),
            "player_a": _player_out(a),
            "player_b": _player_out(b),
            # headline
            "win_prob_a": p_final,
            "win_prob_b": 1.0 - p_final,
            "fair_odds_a": _to_decimal(p_final),
            "fair_odds_b": _to_decimal(1.0 - p_final),
            # component views
            "components": {
                "elo": p_elo,
                "serve_return": p_sim,
                "market": market_prob_a,
                "weights": dict(zip(sources, [w / sum(weights) for w in weights])),
            },
            "elo_gap_raw": elo_gap,
            "elo_gap_adjusted": elo_gap_adj,
            "elo_adjustments": adjustments,
            "conditions_breakdown": {"a": parts_a, "b": parts_b},
            "h2h": rec,
            # point / score model
            "point_win_serve_a": pa_adj,
            "point_win_serve_b": pb_adj,
            "hold_prob_a": sim["hold_a"],
            "hold_prob_b": sim["hold_b"],
            "tiebreak_prob_a": sim["tb_a"],
            "set_score_probs": {f"{k[0]}-{k[1]}": v for k, v in sorted(sim["set_scores"].items())},
            "p_straight_sets_a": sim["p_straight"],
            "expected_games_a": sim["exp_games_a"],
            "expected_games_b": sim["exp_games_b"],
            "expected_total_games": sim["exp_total_games"],
            "game_margin_a": sim["game_margin"],
            "fair_game_handicap_a": fair_handicap,
            "fair_total_games_line": fair_total,
            "total_games_probs": totals,
            "likely_scorelines": [
                {"score": " ".join(f"{s[0]}-{s[1]}" for s in path), "prob": p}
                for path, p in top_scorelines
            ],
            "data_quality": quality,
            "_games_joint": sim["games"],  # kept for picks/backtest, not for display
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _player_out(s: PlayerState) -> dict:
    return {
        "id": s.player_id, "name": s.name, "hand": s.hand, "height": s.height,
        "ioc": s.ioc, "elo": s.elo, "elo_surface": s.elo_surface,
        "elo_blend": s.elo_blend, "matches": s.matches,
        "matches_surface": s.matches_surface,
        "serve_excess": s.serve_excess, "return_excess": s.return_excess,
        "days_rest": s.condition.get("days_rest"),
        "fatigue_index": s.condition.get("fatigue_index"),
        "is_home": bool(s.condition.get("is_home", False)),
    }


def _to_decimal(p: float) -> float:
    return round(1.0 / p, 3) if p > 0 else float("inf")


def _fair_total_line(totals: dict[int, float]) -> float:
    """The half-point total-games line closest to a 50/50 split."""
    if not totals:
        return float("nan")
    keys = sorted(totals)
    cum, best, best_gap = 0.0, keys[0] + 0.5, 1.0
    for k in keys:
        cum += totals[k]
        gap = abs(cum - 0.5)
        if gap < best_gap:
            best_gap, best = gap, k + 0.5
    return best


def _fair_handicap_line(games: dict) -> float:
    """The half-point game handicap for A closest to a 50/50 split."""
    if not games:
        return float("nan")
    margins: dict[int, float] = {}
    for (ga, gb), p in games.items():
        margins[ga - gb] = margins.get(ga - gb, 0.0) + p
    keys = sorted(margins)
    cum, best, best_gap = 0.0, -(keys[0] - 0.5), 1.0
    for k in keys:
        cum += margins[k]
        gap = abs(cum - 0.5)
        if gap < best_gap:
            best_gap, best = gap, -(k + 0.5)
    return best


# Career matches needed before a rating is trustworthy. 30 is roughly a full
# season of main-draw tennis, by which point the decaying K-factor has settled.
QUALITY_MATCHES_HIGH = 30
QUALITY_MATCHES_MED = 10
# Service points needed before the serve/return book means anything. ~70 service
# points per match, so 2000 is about 30 matches — the same bar, in the other unit.
QUALITY_SVPT_HIGH = 2000


def _player_quality(s: PlayerState, surface: str) -> dict:
    """
    How much evidence stands behind ONE player's rating.

    Career matches and service points are deliberately NOT counted as separate
    problems: they measure the same underlying thing (71% of rated players trip
    both), and treating them as independent made a single thin player produce
    three flags. Combined with a "3 flags = low" rule, that meant one unknown
    qualifier dragged an otherwise well-evidenced matchup to "low" — the label
    said the prediction was untrustworthy when only one side of it was.
    """
    if s.matches >= QUALITY_MATCHES_HIGH and s.svpt_seen >= QUALITY_SVPT_HIGH:
        level, reason = "high", f"{s.matches} matches on record"
    elif s.matches >= QUALITY_MATCHES_MED:
        level, reason = "medium", f"only {s.matches} matches on record"
    else:
        level, reason = "low", (
            f"only {s.matches} match{'' if s.matches == 1 else 'es'} on record"
        )

    # Surface coverage is a SEPARATE axis and never the player's fault when the
    # surface itself is defunct: the ATP retired carpet in 2009, so no active
    # player has carpet history and saying so tells you nothing about them.
    surface_note = None
    if surface == "Carpet":
        surface_note = "carpet was discontinued on tour in 2009 — surface rating falls back to overall"
    elif s.matches_surface < 5:
        surface_note = f"only {s.matches_surface} match{'' if s.matches_surface == 1 else 'es'} on {surface}"
        if level == "high":
            level = "medium"

    return {"level": level, "reason": reason, "surface_note": surface_note}


def _data_quality(a: PlayerState, b: PlayerState, surface: str) -> dict:
    """
    Overall confidence in the inputs — the WORSE of the two players.

    A forecast is only as good as the thinner side of it, so this deliberately
    does not average. It does, however, name which player is the weak link, so
    "low" is actionable instead of mysterious.
    """
    qa = _player_quality(a, surface)
    qb = _player_quality(b, surface)
    rank = {"low": 0, "medium": 1, "high": 2}
    level = min(qa["level"], qb["level"], key=lambda x: rank[x])

    # `flags` are written as plain sentences, not codes. The previous version
    # emitted things like "b:few_on_surface(1)", which the dashboard printed
    # verbatim and nobody could act on.
    flags: list[str] = []
    for s, q in ((a, qa), (b, qb)):
        if q["level"] != "high":
            flags.append(f"{s.name}: {q['reason']}")
        if q["surface_note"]:
            # The carpet note describes the surface, not the player — say it once.
            note = q["surface_note"] if surface == "Carpet" else f"{s.name}: {q['surface_note']}"
            if note not in flags:
                flags.append(note)

    return {"level": level, "flags": flags, "players": {"a": qa, "b": qb}}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _print_prediction(p: dict) -> None:
    a, b = p["player_a"], p["player_b"]
    print()
    print("=" * 74)
    print(f"  {a['name']}  vs  {b['name']}")
    ctx = f"{p['surface']} · best of {p['best_of']} · {p['tour'].upper()}"
    if p.get("tournament"):
        ctx = f"{p['tournament']} · " + ctx
    if p["indoor"]:
        ctx += " · indoor"
    if p["altitude"] > 500:
        ctx += f" · {p['altitude']:.0f}m"
    print(f"  {ctx}")
    print("=" * 74)
    print(f"  WIN PROBABILITY   {a['name']}: {p['win_prob_a']*100:5.1f}%   "
          f"({p['fair_odds_a']:.2f})")
    print(f"                    {b['name']}: {p['win_prob_b']*100:5.1f}%   "
          f"({p['fair_odds_b']:.2f})")
    print()
    c = p["components"]
    mk = f"{c['market']:.3f}" if c["market"] is not None else "  —  "
    print(f"  components        elo {c['elo']:.3f} | serve/return {c['serve_return']:.3f}"
          f" | market {mk}")
    print(f"  elo blend         {a['elo_blend']:7.1f} vs {b['elo_blend']:7.1f}"
          f"   (gap {p['elo_gap_raw']:+.1f} → {p['elo_gap_adjusted']:+.1f} adjusted)")
    if p["elo_adjustments"]:
        adj = "  ".join(f"{k} {v:+.1f}" for k, v in p["elo_adjustments"].items())
        print(f"  adjustments       {adj}")
    h = p["h2h"]
    if h["n"]:
        print(f"  head-to-head      {h['raw_wins']}-{h['raw_losses']} in {h['n']} meetings")
    print()
    print(f"  SERVE             hold {p['hold_prob_a']*100:.1f}% vs "
          f"{p['hold_prob_b']*100:.1f}%   "
          f"(pts on serve {p['point_win_serve_a']*100:.1f}% / "
          f"{p['point_win_serve_b']*100:.1f}%)")
    print(f"  SET SCORES        " + "  ".join(
        f"{k} {v*100:.0f}%" for k, v in p["set_score_probs"].items() if v > 0.02))
    print(f"  GAMES             {a['name']} {p['expected_games_a']:.1f} - "
          f"{p['expected_games_b']:.1f} {b['name']}   "
          f"(total {p['expected_total_games']:.1f})")
    print(f"  FAIR LINES        handicap {a['name']} {p['fair_game_handicap_a']:+.1f} games"
          f"   ·   total {p['fair_total_games_line']:.1f} games")
    print(f"  LIKELY SCORES     " + "   ".join(
        f"{s['score']} ({s['prob']*100:.0f}%)" for s in p["likely_scorelines"][:4]))
    q = p["data_quality"]
    print(f"  DATA QUALITY      {q['level']}"
          + (f"  [{', '.join(q['flags'])}]" if q["flags"] else ""))
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a tennis match.")
    ap.add_argument("--a", required=True, help="player A name or id")
    ap.add_argument("--b", required=True, help="player B name or id")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--surface", default="Hard", choices=list(SURFACES))
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--date", default=None, help="match date YYYY-MM-DD")
    ap.add_argument("--tournament", default=None)
    ap.add_argument("--indoor", action="store_true")
    ap.add_argument("--altitude", type=float, default=0.0)
    ap.add_argument("--no-final-tb", action="store_true",
                    help="deciding set played as an advantage set")
    ap.add_argument("--market-prob", type=float, default=None,
                    help="vig-free market probability for player A")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    eng = Engine()
    p = eng.predict(
        args.a, args.b, tour=args.tour, surface=args.surface, best_of=args.best_of,
        match_date=args.date, indoor=args.indoor, altitude=args.altitude,
        final_set_tb=not args.no_final_tb, market_prob_a=args.market_prob,
        tournament=args.tournament,
    )
    if args.json:
        import json
        p.pop("_games_joint", None)
        print(json.dumps(p, indent=2, default=str))
    else:
        _print_prediction(p)


if __name__ == "__main__":
    main()
