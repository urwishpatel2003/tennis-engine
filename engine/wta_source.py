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
        draw size, match state, AND per-match serve statistics — aces, double
        faults, first serves in and won, service points won, break points.
        Everything both the ratings and the serve/return books need.

Endpoints used
--------------
GET /tennis/tournaments/?page=&pageSize=&from=YYYY-MM-DD   (the `from` filter is
    the one that works — `year=` is silently ignored and returns 1960s events)
GET /tennis/tournaments/{groupId}/{year}/matches
GET /tennis/tournaments/{groupId}/{year}/matches/{matchId}/stats

The stats call is one request per match, so it is the slow part of a refresh. It
is worth it: without it the WTA archive would advance ELO while the serve/return
books silently stayed frozen at the last date Sackmann supplied stat lines, and
nothing on screen would say so.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from engine.schema import REFRESH_LOOKBACK_DAYS

API = "https://api.wtatennis.com/tennis"

# Their level strings → Sackmann tourney_level.
LEVEL_MAP = {"Grand Slam": "G", "WTA 1000": "P", "WTA 500": "P",
             "WTA 250": "I", "WTA 125": "I", "Finals": "F", "ITF": "S"}

# Matches in a round → round label. Derived from the round's own size rather than
# a round index, because draws with byes (the 96-player 1000s) do not start at a
# clean power of two and their RoundID=1 is not a 128-player round.
LOOKBACK_DAYS = REFRESH_LOOKBACK_DAYS  # see engine/schema.py


def _round_label(round_id: object, draw_size: object) -> str:
    """
    Map the API's RoundID onto a round name using the DRAW SIZE.

    The old code inferred the round from how many FINISHED matches shared a
    RoundID, which broke twice over on a live event: a partly-played R16 with two
    results so far was labelled "SF", and two rounds that legitimately hold the
    same number of matches (R128 and R64 both hold 32 in a bye draw) both came
    out as "R64". Cincinnati showed 64 matches labelled R64 and 2 labelled SF,
    with no QF in between — a shape no draw can have.

    RoundID is ordinal and stable whether or not a round has finished, so it is
    the right key. Named rounds map directly; numeric round i is the draw halved
    i-1 times.
    """
    rid = str(round_id).strip().upper()
    if rid in ("F", "S", "Q"):
        return {"F": "F", "S": "SF", "Q": "QF"}[rid]
    try:
        i = int(rid)
    except (TypeError, ValueError):
        return "R128"
    try:
        n = int(draw_size or 32)
    except (TypeError, ValueError):
        n = 32
    # Round the draw up to a power of two — a 96-draw is a 128 bracket with byes.
    bracket = 1 << max(n - 1, 1).bit_length()
    return f"R{max(bracket >> max(i - 1, 0), 16)}"

STAT_COLS = ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms",
             "bpSaved", "bpFaced")


class RateLimited(RuntimeError):
    """The API asked us to slow down. Distinct so callers cannot mistake it for
    'no data' — see `list_events`."""


# Politeness between calls. A full stats backfill is ~900 requests and firing
# them flat out earned an HTTP 429, after which every subsequent call failed and
# the refresh silently reported "already up to date".
_MIN_INTERVAL = 0.12
_last_call = 0.0


def _get(path: str, timeout: int = 60, retries: int = 3):
    global _last_call
    for attempt in range(retries + 1):
        wait = _MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            API + path,
            headers={"User-Agent": "tennis-engine/0.1", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                _last_call = time.time()
                return json.load(r)
        except urllib.error.HTTPError as e:
            _last_call = time.time()
            if e.code == 429 and attempt < retries:
                time.sleep(2.0 * (2 ** attempt))     # 2s, 4s, 8s
                continue
            if e.code == 429:
                raise RateLimited(f"429 after {retries + 1} attempts on {path}") from e
            raise
        except Exception:
            _last_call = time.time()
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise


def match_stats(gid: int, year: int, match_id: str) -> dict | None:
    """
    Per-match serve statistics, from /matches/{id}/stats.

    Without these the WTA half of the archive updates ELO but not the
    serve/return books — the point model would silently stay frozen at whatever
    date the historical Sackmann stat lines ran out, while the ratings moved on.
    That is a worse failure than missing data, because nothing on screen would
    say so.

    Returns the setnum==0 row (match totals) mapped onto our column names, from
    the WINNER's perspective is NOT possible here — the API is A/B, so the caller
    supplies which side won.
    """
    try:
        rows = _get(f"/tournaments/{gid}/{year}/matches/{match_id}/stats", timeout=30)
    except RateLimited:
        raise                       # let the caller stop cleanly rather than
                                    # silently write a stats-free archive
    except Exception:
        return None
    if not isinstance(rows, list):
        return None
    tot = next((r for r in rows if r.get("setnum") == 0), None)
    return tot


def stats_for_side(tot: dict, a_won: bool) -> dict:
    """
    Map one stats row onto our w_*/l_* columns.

    Break points need care: the API reports them from the RETURNER's side
    (`breakptsconva` = break points player A converted on B's serve), whereas our
    schema stores them from the SERVER's side (bpFaced/bpSaved). So the winner's
    break points faced are the loser's break points played, and saved is that
    minus what the loser converted.
    """
    def g(k, default=np.nan):
        v = tot.get(k)
        return float(v) if isinstance(v, (int, float)) else default

    w, l = ("a", "b") if a_won else ("b", "a")
    out = {}
    for tag, side in (("w", w), ("l", l)):
        other = "b" if side == "a" else "a"
        svpt = g(f"totservplayed{side}")
        first_in = g(f"ptsplayed1stserv{side}")
        first_won = g(f"ptswon1stserv{side}")
        serv_won = g(f"ptstotwonserv{side}")
        bp_played_vs = g(f"breakptsplayed{other}")   # BPs the OPPONENT had
        bp_conv_vs = g(f"breakptsconv{other}")       # ...and converted
        out.update({
            f"{tag}_ace": g(f"aces{side}"),
            f"{tag}_df": g(f"dblflt{side}"),
            f"{tag}_svpt": svpt,
            f"{tag}_1stIn": first_in,
            f"{tag}_1stWon": first_won,
            f"{tag}_2ndWon": (serv_won - first_won)
                             if np.isfinite(serv_won) and np.isfinite(first_won) else np.nan,
            f"{tag}_SvGms": g(f"servgamesplayed{side}"),
            f"{tag}_bpFaced": bp_played_vs,
            f"{tag}_bpSaved": (bp_played_vs - bp_conv_vs)
                              if np.isfinite(bp_played_vs) and np.isfinite(bp_conv_vs)
                              else np.nan,
        })
    return out


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
    """
    Non-ITF WTA events that could still have unrecorded results.

    That means everything starting after `after - LOOKBACK_DAYS`, not merely
    after `after` — see LOOKBACK_DAYS for why the stricter test silently froze
    in-progress tournaments.
    """
    since = after - pd.Timedelta(days=LOOKBACK_DAYS)
    events, seen = [], set()
    for page in range(max_pages):
        try:
            j = _get(f"/tournaments/?page={page}&pageSize=200"
                     f"&from={since.date().isoformat()}")
        except RateLimited:
            # Never degrade this into an empty list. An empty list means "nothing
            # new to fetch", the caller reports "already up to date", and the
            # archive quietly stops advancing. A rate limit is not up-to-date.
            raise
        except Exception as e:
            if page == 0:
                raise RuntimeError(f"WTA tournament list failed: {e}") from e
            break
        content = j.get("content") or []
        if not content:
            break
        for t in content:
            grp = t.get("tournamentGroup") or {}
            gid, yr, lvl = grp.get("id"), t.get("year"), grp.get("level")
            start = pd.to_datetime(t.get("startDate"), errors="coerce")
            # ITF is excluded so the archive keeps meaning "main tour" throughout.
            if lvl == "ITF" or gid is None or pd.isna(start) or start < since:
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


def fetch(after: pd.Timestamp, resolver, verbose: bool = True,
          with_stats: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Completed WTA main-draw singles after `after`, in canonical raw shape.

    `resolver` is a refresh.PlayerResolver — passed in so ATP and WTA share the
    same name-matching behaviour and id-minting policy.
    """
    events = list_events(after)
    if verbose:
        print(f"  [refresh] WTA: {len(events)} event(s) to check "
              f"(api.wtatennis.com, official)")

    rows, with_results, stats_ok = [], 0, 0
    for ev in events:
        try:
            payload = _get(f"/tournaments/{ev['gid']}/{ev['year']}/matches")
        except RateLimited:
            if verbose:
                print(f"  [refresh] WTA: rate limited after {with_results} event(s) — "
                      f"keeping what was fetched, re-run to continue")
            break
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
            label = _round_label(rid, ev.get("draw_size"))
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

                stat_cols = {f"w_{c}": np.nan for c in STAT_COLS}
                stat_cols.update({f"l_{c}": np.nan for c in STAT_COLS})
                if with_stats and x.get("MatchID"):
                    try:
                        tot = match_stats(ev["gid"], ev["year"], x["MatchID"])
                    except RateLimited:
                        with_stats = False      # finish this run without stats
                        tot = None
                    if tot:
                        mapped = stats_for_side(tot, a_won)
                        if np.isfinite(mapped.get("w_svpt", np.nan)):
                            stat_cols.update(mapped)
                            stats_ok += 1

                rows.append({
                    "src_id": f"{ev['gid']}-{ev['year']}-{x.get('MatchID') or f'{rid}-{n:03d}'}",
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
                    **stat_cols,
                    "winner_rank": np.nan, "winner_rank_points": np.nan,
                    "loser_rank": np.nan, "loser_rank_points": np.nan,
                })

    stats = {"events_checked": len(events), "events_with_results": with_results,
             "new": len(rows), "with_serve_stats": stats_ok}
    if not rows:
        return pd.DataFrame(), stats
    return pd.DataFrame(rows), stats


def stats_for_matches(gid: int, year: int, wanted: dict, verbose: bool = False) -> dict:
    """
    Fetch serve statistics for specific matches of one tournament.

    `wanted` maps frozenset({normalised winner, normalised loser}) -> our match_id.
    Rows ingested by a build carry no MatchID (builds skip the stats pass), so the
    only way back to the upstream match is the tournament's own match list plus a
    player-pair lookup. Names inside a single draw are unambiguous, which is what
    makes this safe.

    Returns {our_match_id: {w_ace: ..., l_svpt: ...}}. Raises RateLimited so the
    caller can stop cleanly and keep whatever it already has.
    """
    out: dict = {}
    if not wanted:
        return out
    try:
        payload = _get(f"/tournaments/{gid}/{year}/matches")
    except RateLimited:
        raise
    except Exception:
        return out

    for x in payload.get("matches") or []:
        if x.get("DrawMatchType") != "S" or x.get("MatchState") != "F":
            continue
        mid = x.get("MatchID")
        if not mid:
            continue
        na = _pname(x, "A")
        nb = _pname(x, "B")
        if not na or not nb:
            continue
        key = frozenset((na, nb))
        our_id = wanted.get(key)
        if our_id is None:
            continue
        tot = match_stats(gid, year, mid)          # may raise RateLimited
        if not tot:
            continue
        a_won = str(x.get("Winner")) == "2"
        mapped = stats_for_side(tot, a_won)
        if np.isfinite(mapped.get("w_svpt", np.nan)) and mapped["w_svpt"] > 0:
            out[our_id] = mapped
            if verbose:
                print(f"      + {na} / {nb}", flush=True)
    return out


def _pname(x: dict, side: str) -> str:
    from engine.refresh import _norm
    return _norm(f"{x.get(f'PlayerNameFirst{side}') or ''} "
                 f"{x.get(f'PlayerNameLast{side}') or ''}".strip())
