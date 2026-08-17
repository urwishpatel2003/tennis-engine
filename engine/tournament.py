"""
Tournament view — draws, per-match predictions, bracket odds and event accuracy.

    python engine/tournament.py --tour atp --season 2026 --list
    python engine/tournament.py --tour atp --tourney-id 2026-520

Inputs : data/raw/matches_{tour}.parquet          (the draw as played)
         data/processed/ratings.parquet           (PRE-match ratings, per match)
         data/processed/serve_return.parquet      (PRE-match serve/return)
         data/raw/players_{tour}.parquet
Outputs: dicts consumed by dashboard/server.py

The one rule that matters here
------------------------------
Every prediction shown against a historical match is computed from the PRE-MATCH
columns that ratings.py and serve_return.py froze before that match was played —
never from current ratings. Replaying a 2024 draw with 2026 ratings would let the
model "predict" results it has already been trained on, and the accuracy column
would be a fiction. This is the same guarantee `backtest.py` relies on, reusing
the same frozen tables and the same `predict_from_states` code path, so the
tournament view and the backtest can never disagree about what the model said.

Bracket reconstruction
----------------------
Sackmann gives a flat match list with a `round` label and a `match_num`, not a
tree. The bracket is recovered from the results themselves: a player in a
round-k match must have won some round-(k-1) match, which tells you exactly which
two earlier matches feed each later one. That is derived from observation rather
than assumed from `match_num` ordering, which is not a reliable draw order across
events and eras.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.predict import PlayerState, predict_from_states, quick_win_prob
from engine.ratings import SURFACE_BLEND
from engine.schema import PROCESSED, RAW, ROUND_ORDER, TOURS

# Rounds in the order they are played, so a draw renders top-down.
ROUND_LABEL = {
    "R128": "Round of 128", "R64": "Round of 64", "R32": "Round of 32",
    "R16": "Round of 16", "QF": "Quarter-final", "SF": "Semi-final",
    "F": "Final", "RR": "Round robin", "BR": "Bronze medal match",
}


class TournamentStore:
    """Loads the frames once; every query below is a filter over them."""

    def __init__(self, tours: tuple[str, ...] = TOURS) -> None:
        self.matches = self._concat([RAW / f"matches_{t}.parquet" for t in tours])
        self.players = self._players(tours)
        self.ratings = self._read(PROCESSED / "ratings.parquet")
        self.sr = self._read(PROCESSED / "serve_return.parquet")
        self.h2h = self._read(PROCESSED / "h2h.parquet")
        self._draw_cache: dict = {}
        self._bracket_cache: dict = {}

        # match_id -> frozen pre-match state for both players.
        #
        # Built with a single join and one to_dict, NOT by iterating. The first
        # version used `for mid, row in r.iterrows()` with a `s.loc[mid]` lookup
        # inside, which is a per-row indexed lookup across 90k rows and took 54
        # SECONDS to construct — an order of magnitude more than every prediction
        # the store then went on to make.
        self._pre: dict = {}
        if not self.ratings.empty:
            r = self.ratings.set_index("match_id")
            if not self.sr.empty:
                s = self.sr.set_index("match_id")
                s = s[[c for c in s.columns if c not in r.columns]]
                r = r.join(s, how="left")
            self._pre = r.to_dict("index")

    @staticmethod
    def _read(p: Path) -> pd.DataFrame:
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    @classmethod
    def _concat(cls, paths: list[Path]) -> pd.DataFrame:
        frames = [cls._read(p) for p in paths]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _players(self, tours) -> pd.DataFrame:
        frames = []
        for t in tours:
            p = RAW / f"players_{t}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["tour"] = t
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── listing ───────────────────────────────────────────────────────────────
    def list_tournaments(self, tour: str, season: int | None = None) -> list[dict]:
        m = self.matches
        if m.empty:
            return []
        m = m[m["tour"] == tour]
        if season is not None:
            m = m[m["season"] == season]
        if m.empty:
            return []

        g = m.groupby(["tourney_id", "tourney_name", "surface", "tourney_level",
                       "tourney_date", "draw_size"], as_index=False, dropna=False)
        out = g.agg(matches=("match_id", "size"))
        out = out.sort_values("tourney_date", ascending=False)
        return [
            {
                "tourney_id": r.tourney_id,
                "name": r.tourney_name,
                "surface": r.surface,
                "level": r.tourney_level,
                "date": r.tourney_date.date().isoformat() if pd.notna(r.tourney_date) else None,
                "draw_size": int(r.draw_size) if pd.notna(r.draw_size) else None,
                "matches": int(r.matches),
                "season": int(r.tourney_date.year) if pd.notna(r.tourney_date) else None,
            }
            for r in out.itertuples(index=False)
        ]

    def seasons(self, tour: str) -> list[int]:
        m = self.matches
        if m.empty:
            return []
        return sorted(m[m["tour"] == tour]["season"].dropna().unique().astype(int).tolist(),
                      reverse=True)

    # ── frozen pre-match state ────────────────────────────────────────────────
    def _state(self, pre: dict, side: str, pid: int, name: str, tour: str) -> PlayerState:
        """Build a PlayerState from the frozen pre-match columns ('w' or 'l')."""
        attr = {}
        if not self.players.empty:
            hit = self.players[(self.players["tour"] == tour)
                               & (self.players["player_id"] == pid)]
            if len(hit):
                attr = hit.iloc[0].to_dict()

        st = PlayerState(player_id=int(pid), name=name, tour=tour)
        st.elo = float(pre.get(f"{side}_elo", 1500.0))
        st.elo_surface = float(pre.get(f"{side}_elo_surface", st.elo))
        st.matches = int(pre.get(f"{side}_matches", 0) or 0)
        st.matches_surface = int(pre.get(f"{side}_matches_surface", 0) or 0)
        st.serve_excess = float(pre.get(f"{side}_serve_excess", 0.0) or 0.0)
        st.return_excess = float(pre.get(f"{side}_return_excess", 0.0) or 0.0)
        st.svpt_seen = float(pre.get(f"{side}_svpt_seen", 0.0) or 0.0)
        st.hand = attr.get("hand")
        st.height = attr.get("height") if pd.notna(attr.get("height")) else None
        st.ioc = attr.get("ioc")
        return st

    def predict_match(self, row, scorelines: bool = False) -> dict | None:
        """
        Prediction for one historical match, from its frozen pre-match state.

        Scoreline enumeration is OFF by default here. A slam draw is 127 matches
        of best-of-5, and enumerating set-by-set paths for every one of them is
        what made the first version of this take minutes per tournament.
        """
        pre = self._pre.get(row.match_id)
        if pre is None:
            return None
        a = self._state(pre, "w", int(row.winner_id), str(row.winner_name), row.tour)
        b = self._state(pre, "l", int(row.loser_id), str(row.loser_name), row.tour)
        return predict_from_states(
            a, b, tour=row.tour, surface=row.surface,
            best_of=int(row.best_of) if pd.notna(row.best_of) else 3,
            as_of=row.tourney_date, tournament=row.tourney_name, h2h=self.h2h,
            track_scorelines=scorelines,
        )

    # ── the draw ──────────────────────────────────────────────────────────────
    def draw(self, tour: str, tourney_id: str) -> dict:
        """
        Cached: a draw is ~127 predictions and never changes once built.

        The bracket simulation is deliberately NOT included — it evaluates every
        possible pairing in the tree (~8k for a 128 draw) and costs several times
        the draw itself. The dashboard loads it separately, on demand.
        """
        key = (tour, tourney_id)
        if key in self._draw_cache:
            return self._draw_cache[key]
        out = self._draw(tour, tourney_id)
        self._draw_cache[key] = out
        return out

    def bracket(self, tour: str, tourney_id: str) -> dict:
        """Cached title odds for one event."""
        key = (tour, tourney_id)
        if key in self._bracket_cache:
            return self._bracket_cache[key]
        out = self.bracket_odds(tour, tourney_id)
        self._bracket_cache[key] = out
        return out

    def _draw(self, tour: str, tourney_id: str) -> dict:
        m = self.matches
        m = m[(m["tour"] == tour) & (m["tourney_id"] == tourney_id)]
        if m.empty:
            return {"error": f"no tournament {tourney_id!r} on the {tour.upper()} tour"}

        m = m.sort_values(["round_ord", "match_num"])
        head = m.iloc[0]

        # Predict every match ONCE and hand the cache to the bracket simulation.
        # The first version recomputed each prediction there, and looked each
        # match up with a full scan of the 90k-row match table to do it.
        preds = {r.match_id: self.predict_match(r) for r in m.itertuples(index=False)}

        rounds: dict[str, list] = {}
        hits = misses = 0
        losses = []
        upsets_called, upsets_missed = [], []

        for r in m.itertuples(index=False):
            p = preds.get(r.match_id)
            entry = {
                "match_id": r.match_id,
                "round": r.round,
                "round_label": ROUND_LABEL.get(r.round, r.round),
                "winner": r.winner_name, "winner_id": int(r.winner_id),
                "loser": r.loser_name, "loser_id": int(r.loser_id),
                "score": r.score,
                "completed": bool(r.completed),
                "retirement": bool(r.retirement),
                "best_of": int(r.best_of) if pd.notna(r.best_of) else 3,
            }

            if p is not None:
                # `predict_match` puts the actual winner in slot A, so the model's
                # probability for slot A IS its probability for the true winner.
                pw = p["win_prob_a"]
                entry.update({
                    "model_prob_winner": pw,
                    "model_favourite": r.winner_name if pw >= 0.5 else r.loser_name,
                    "correct": bool(pw >= 0.5),
                    "fair_odds_winner": p["fair_odds_a"],
                    "expected_total_games": p["expected_total_games"],
                    "predicted_margin": p["game_margin_a"],
                    "likely_score": (p["likely_scorelines"][0]["score"]
                                     if p["likely_scorelines"] else None),
                    "data_quality": p["data_quality"]["level"],
                    "elo_winner": p["player_a"]["elo_blend"],
                    "elo_loser": p["player_b"]["elo_blend"],
                })
                if bool(r.completed):
                    hits += int(pw >= 0.5)
                    misses += int(pw < 0.5)
                    losses.append(-np.log(max(pw, 1e-9)))
                    # An "upset" is a result the model gave under 35%.
                    if pw < 0.35:
                        upsets_missed.append({"match": f"{r.winner_name} d. {r.loser_name}",
                                              "round": r.round, "score": r.score,
                                              "model_prob": pw})
                    elif pw > 0.65 and r.loser_rank is not None:
                        pass
                # Underdog by rating that the model still favoured.
                if p["player_a"]["elo_blend"] < p["player_b"]["elo_blend"] and pw >= 0.5:
                    upsets_called.append({"match": f"{r.winner_name} d. {r.loser_name}",
                                          "round": r.round, "score": r.score,
                                          "model_prob": pw})

            rounds.setdefault(r.round, []).append(entry)

        order = [x for x in ROUND_ORDER if x in rounds]
        played = hits + misses
        summary = {
            "matches": int(len(m)),
            "scored": played,
            "hits": hits,
            "accuracy": (hits / played) if played else None,
            "logloss": float(np.mean(losses)) if losses else None,
            "upsets_missed": sorted(upsets_missed, key=lambda x: x["model_prob"])[:5],
            "upsets_called": sorted(upsets_called, key=lambda x: -x["model_prob"])[:5],
        }

        return {
            "tourney_id": tourney_id,
            "tour": tour,
            "name": head.tourney_name,
            "surface": head.surface,
            "level": head.tourney_level,
            "date": head.tourney_date.date().isoformat(),
            "season": int(head.season),
            "draw_size": int(head.draw_size) if pd.notna(head.draw_size) else None,
            "best_of": int(head.best_of) if pd.notna(head.best_of) else 3,
            "round_order": order,
            "rounds": {k: rounds[k] for k in order},
            "summary": summary,
        }

    # ── bracket reconstruction + forward simulation ───────────────────────────
    def _pre_tournament_states(self, m: pd.DataFrame, tour: str) -> dict:
        """
        Every participant's rating AS OF THE START of the event.

        Taken from each player's FIRST match in the draw, whose frozen pre-match
        columns are by definition their state before the tournament began. This
        is what makes a forward simulation legitimate: hypothetical pairings that
        never happened are still priced with information available beforehand.
        """
        states: dict[int, PlayerState] = {}
        for r in m.sort_values(["round_ord", "match_num"]).itertuples(index=False):
            pre = self._pre.get(r.match_id)
            if pre is None:
                continue
            for side, pid, name in (("w", int(r.winner_id), r.winner_name),
                                    ("l", int(r.loser_id), r.loser_name)):
                if pid not in states:
                    states[pid] = self._state(pre, side, pid, str(name), tour)
        return states

    def bracket_odds(self, tour: str, tourney_id: str,
                     m: pd.DataFrame | None = None,
                     preds: dict | None = None) -> dict:
        """
        Simulate the draw forward from round one and report title odds.

        The bracket is rebuilt from results — whoever played a round-k match must
        have won a round-(k-1) match, which identifies exactly which two earlier
        matches feed it — and then every player is advanced through that fixed
        tree against EVERY opponent they could have met:

            P(x wins node) = Σ_y P(x reaches) · P(y reaches) · P(x beats y)

        An earlier version propagated a single pairing's probability to everyone
        in the bracket half, which is not a simulation at all — it just decayed
        each player by whatever happened in the one match that was played.

        These numbers are MORE confident than a bookmaker's, and legitimately so:
        compounding calibrated per-match probabilities over six or seven rounds is
        all this does. A market also prices withdrawal, injury mid-event and form
        collapse, none of which are modelled here, and all of which drag a heavy
        favourite down. Read a 70% title price as "70% if nothing goes wrong".
        """
        if m is None:
            m = self.matches[(self.matches["tour"] == tour)
                             & (self.matches["tourney_id"] == tourney_id)]
        if m.empty:
            return {"available": False, "reason": "no matches"}

        rounds_present = [r for r in ROUND_ORDER if r in set(m["round"])]
        knockout = [r for r in rounds_present if r not in ("RR", "BR")]
        if len(knockout) < 2:
            return {"available": False, "reason": "not a knockout draw"}

        states = self._pre_tournament_states(m, tour)
        head = m.iloc[0]
        surface = head.surface
        best_of = int(head.best_of) if pd.notna(head.best_of) else 3

        def pw(x: int, y: int) -> float:
            sx, sy = states.get(x), states.get(y)
            if sx is None or sy is None:
                return 0.5
            return quick_win_prob(sx, sy, tour, surface, best_of)

        by_round = {r: m[m["round"] == r] for r in knockout}

        # Layer 0 — opening-round nodes, each holding its two entrants and the
        # probability each of them emerges from that node.
        nodes = []
        for r in by_round[knockout[0]].itertuples(index=False):
            a, b = int(r.winner_id), int(r.loser_id)
            p = pw(a, b)
            nodes.append({"members": {a: p, b: 1.0 - p}})

        names = {pid: st.name for pid, st in states.items()}

        # Each later round merges two nodes; convolve over all possible arrivals.
        for rnd in knockout[1:]:
            merged, nxt = set(), []
            for r in by_round[rnd].itertuples(index=False):
                gi = _find_group(nodes, int(r.winner_id))
                gj = _find_group(nodes, int(r.loser_id))
                if gi is None or gj is None or gi == gj:
                    continue
                merged.update({gi, gj})
                left, right = nodes[gi]["members"], nodes[gj]["members"]
                out: dict[int, float] = {}
                for x, px in left.items():
                    if px <= 1e-9:
                        continue
                    for y, py in right.items():
                        if py <= 1e-9:
                            continue
                        joint = px * py
                        p = pw(x, y)
                        out[x] = out.get(x, 0.0) + joint * p
                        out[y] = out.get(y, 0.0) + joint * (1.0 - p)
                nxt.append({"members": out})
            for k, g in enumerate(nodes):
                if k not in merged:
                    nxt.append(g)
            nodes = nxt

        final = {}
        for n in nodes:
            for pid, p in n["members"].items():
                final[pid] = final.get(pid, 0.0) + p
        total = sum(final.values()) or 1.0

        table = sorted(
            ({"player_id": pid, "name": names.get(pid, str(pid)),
              "title_prob": v / total} for pid, v in final.items()),
            key=lambda x: -x["title_prob"],
        )[:16]

        champion = m[m["round"] == knockout[-1]]
        actual = champion.iloc[0]["winner_name"] if len(champion) else None
        actual_prob = None
        if len(champion):
            cid = int(champion.iloc[0]["winner_id"])
            actual_prob = final.get(cid, 0.0) / total
        return {
            "available": True,
            "rounds": knockout,
            "title_odds": table,
            "actual_champion": actual,
            "actual_champion_prob": actual_prob,
        }


def _pw(preds: dict, match_id: str) -> float | None:
    """Model probability that the ACTUAL winner of this match won it."""
    p = preds.get(match_id)
    return p["win_prob_a"] if p else None


def _find_group(groups: list[dict], pid: int) -> int | None:
    for i, g in enumerate(groups):
        if pid in g["members"]:
            return i
    return None


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Tournament draws and predictions.")
    ap.add_argument("--tour", default="atp", choices=list(TOURS))
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tourney-id", default=None)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    store = TournamentStore()

    if args.list or not args.tourney_id:
        rows = store.list_tournaments(args.tour, args.season)
        print(f"\n{args.tour.upper()} tournaments"
              + (f" — {args.season}" if args.season else "")
              + f"  ({len(rows)} events)\n")
        print(f"{'date':<12}{'id':<14}{'name':<34}{'surf':<8}{'lvl':<5}{'M':>5}")
        print("─" * 80)
        for r in rows[:args.top]:
            print(f"{str(r['date']):<12}{str(r['tourney_id']):<14}"
                  f"{str(r['name'])[:33]:<34}{str(r['surface']):<8}"
                  f"{str(r['level']):<5}{r['matches']:>5}")
        return

    d = store.draw(args.tour, args.tourney_id)
    if "error" in d:
        print(d["error"])
        sys.exit(1)

    print(f"\n{'='*84}")
    print(f"  {d['name']} {d['season']} — {d['surface']}, best of {d['best_of']}")
    s = d["summary"]
    if s["accuracy"] is not None:
        print(f"  model called {s['hits']}/{s['scored']} = {s['accuracy']*100:.1f}%"
              f"   log loss {s['logloss']:.4f}")
    print("=" * 84)
    for rnd in d["round_order"]:
        print(f"\n  {ROUND_LABEL.get(rnd, rnd)}")
        for e in d["rounds"][rnd]:
            mark = "✓" if e.get("correct") else ("✗" if "correct" in e else " ")
            prob = f"{e['model_prob_winner']*100:4.0f}%" if "model_prob_winner" in e else "   —"
            print(f"    {mark} {prob}  {e['winner'][:22]:<23} d. {e['loser'][:22]:<23} {e['score']}")

    b = store.bracket(args.tour, args.tourney_id)
    if b.get("available"):
        print(f"\n  PRE-TOURNAMENT TITLE ODDS (actual winner: {b['actual_champion']})")
        for t in b["title_odds"][:8]:
            print(f"    {t['title_prob']*100:5.1f}%  {t['name']}")
    print()


if __name__ == "__main__":
    main()
