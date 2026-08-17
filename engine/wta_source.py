"""
WTA results from the tour's own public API (api.wtatennis.com).

No key, no scraping, no third party: this is the same JSON the official WTA site
consumes, and completed main-draw results appear in it within hours.

I initially told this project's owner that no current WTA source existed. That was
wrong — it was one GitHub search and a premature conclusion. The API was public the
whole time.

What it gives and what it does not
----------------------------------
Gives : players, per-set scores with tiebreaks, seeds, rounds, surface, level,
        draw size, match state — everything Elo needs.
Lacks : serve statistics (aces, service points won/played). Those matches
        therefore move the RATINGS but cannot feed the serve/return books.
        `engine/serve_return.py` already skips rows without a stat line, so this
        degrades cleanly rather than corrupting the point model.

Endpoints used
--------------
GET /tennis/tournaments/?page=&pageSize=&from=YYYY-MM-DD   (the `from` filter is
    the one that works — `year=` is silently ignored and returns 1960s events)
GET /tennis/tournaments/{groupId}/{year}/matches
"""

from __future__ import annotations

import json
import re
import urllib.request

import numpy as np
import pandas as pd

API = "https://api.wtatennis.com/tennis"

# Their level strings → Sackmann tourney_level.
LEVEL_MAP = {"Grand Slam": "G", "WTA 1000": "P", "WTA 500": "P",
             "WTA 250": "I", "WTA 125": "I", "Finals": "F", "ITF": "S"}

# Matches in a round → round label. Derived from the round's own size rather than
# a round index, because draws with byes (the 96-player 1000s) do not start at a
# clean power of two and their RoundID=1 is not a 128-player round.
ROUND_BY_SIZE = {64: "R128", 32: "R64", 16: "R32", 8: "R16", 4: "QF", 2: "SF", 1: "F"}

STAT_COLS = ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms",
             "bpSaved", "bpFaced")


def _get(path: str, timeout: int = 60):
    req = urllib.request.Request(
        API + path,
        headers={"User-Agent": "tennis-engine/0.1", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def build_score(m: dict, a_won: bool) -> str:
    """
    Rebuild '6-3 7-6(4)' from the per-set columns, winner first.

    Scores are stored A-first regardless of who won, so when B won every set has
    to be flipped — otherwise the archive would contain scorelines the loser won.
    """
    parts = []
    for i in range(1, 6):
        a, b = m.get(f"ScoreSet{i}A"), m.get(f"ScoreSet{i}B")
        if a in (None, "") or b in (None, ""):
            continue
        try:
            ai, bi = int(a), int(b)
        except (TypeError, ValueError):
            continue
        tb = m.get(f"ScoreTbSet{i}") if i <= 4 else ""
        lo, hi = (ai, bi) if a_won else (bi, ai)
        parts.append(f"{lo}-{hi}" + (f"({tb})" if tb not in (None, "") else ""))
    return " ".join(parts)


def list_events(after: pd.Timestamp, max_pages: int = 12) -> list[dict]:
    """Non-ITF WTA events starting after `after`."""
    events, seen = [], set()
    for page in range(max_pages):
        try:
            j = _get(f"/tournaments/?page={page}&pageSize=200"
                     f"&from={after.date().isoformat()}")
        except Exception:
            break
        content = j.get("content") or []
        if not content:
            break
        for t in content:
            grp = t.get("tournamentGroup") or {}
            gid, yr, lvl = grp.get("id"), t.get("year"), grp.get("level")
            start = pd.to_datetime(t.get("startDate"), errors="coerce")
            # ITF is excluded so the archive keeps meaning "main tour" throughout.
            if lvl == "ITF" or gid is None or pd.isna(start) or start <= after:
                continue
            if (gid, yr) in seen:
                continue
            seen.add((gid, yr))
            events.append({"gid": gid, "year": yr, "name": str(grp.get("name", "")).title(),
                           "level": LEVEL_MAP.get(lvl, "I"),
                           "surface": t.get("surface") or "Hard",
                           "draw_size": t.get("singlesDrawSize"),
                           "start": start})
    return events


def fetch(after: pd.Timestamp, resolver, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Completed WTA main-draw singles after `after`, in canonical raw shape.

    `resolver` is a refresh.PlayerResolver — passed in so ATP and WTA share the
    same name-matching behaviour and id-minting policy.
    """
    events = list_events(after)
    if verbose:
        print(f"  [refresh] WTA: {len(events)} event(s) to check "
              f"(api.wtatennis.com, official)")

    rows, with_results = [], 0
    for ev in events:
        try:
            payload = _get(f"/tournaments/{ev['gid']}/{ev['year']}/matches")
        except Exception:
            continue
        ms = [x for x in (payload.get("matches") or [])
              if x.get("DrawMatchType") == "S"        # singles
              and x.get("DrawLevelType") != "Q"       # main draw
              and x.get("MatchState") == "F"]         # finished
        if not ms:
            continue
        with_results += 1

        by_round: dict[str, list] = {}
        for x in ms:
            by_round.setdefault(str(x.get("RoundID")), []).append(x)

        for rid, group in by_round.items():
            label = ROUND_BY_SIZE.get(len(group), "R128")
            for n, x in enumerate(group, 1):
                a_won = str(x.get("Winner")) == "2"
                na = f"{x.get('PlayerNameFirstA') or ''} {x.get('PlayerNameLastA') or ''}".strip()
                nb = f"{x.get('PlayerNameFirstB') or ''} {x.get('PlayerNameLastB') or ''}".strip()
                if not na or not nb:
                    continue
                w_name, l_name = (na, nb) if a_won else (nb, na)
                w_ioc = x.get("PlayerCountryA") if a_won else x.get("PlayerCountryB")
                l_ioc = x.get("PlayerCountryB") if a_won else x.get("PlayerCountryA")
                score = build_score(x, a_won)
                if not score:
                    continue

                rows.append({
                    "src_id": f"{ev['gid']}-{ev['year']}-{rid}-{n:03d}",
                    "tourney_id": f"{ev['year']}-W{ev['gid']}",
                    "tourney_name": ev["name"],
                    "surface": ev["surface"],
                    "draw_size": ev["draw_size"],
                    "tourney_level": ev["level"],
                    "tourney_date": int(ev["start"].strftime("%Y%m%d")),
                    "match_num": np.nan,
                    "winner_id": resolver.resolve(w_name, {"ioc": w_ioc}),
                    "winner_name": w_name,
                    "winner_seed": np.nan, "winner_entry": np.nan,
                    "winner_hand": np.nan, "winner_ht": np.nan,
                    "winner_ioc": w_ioc, "winner_age": np.nan,
                    "loser_id": resolver.resolve(l_name, {"ioc": l_ioc}),
                    "loser_name": l_name,
                    "loser_seed": np.nan, "loser_entry": np.nan,
                    "loser_hand": np.nan, "loser_ht": np.nan,
                    "loser_ioc": l_ioc, "loser_age": np.nan,
                    "score": score,
                    "best_of": 3,
                    "round": label,
                    "minutes": np.nan,
                    **{f"w_{c}": np.nan for c in STAT_COLS},
                    **{f"l_{c}": np.nan for c in STAT_COLS},
                    "winner_rank": np.nan, "winner_rank_points": np.nan,
                    "loser_rank": np.nan, "loser_rank_points": np.nan,
                })

    stats = {"events_checked": len(events), "events_with_results": with_results,
             "new": len(rows), "with_serve_stats": 0}
    if not rows:
        return pd.DataFrame(), stats
    return pd.DataFrame(rows), stats
