"""
Live match state from the Live Tennis API, mapped onto the engine's own terms.

Inputs : api.livetennisapi.com (LIVE_TENNIS_API_KEY), free tier is enough
Outputs: normalised in-progress matches, ready for live_state.win_prob_from_state

The free tier carries live and upcoming matches with a full score object —
server, point score, game score, set score, tiebreak flag — which is exactly
what an in-match win probability needs. Completed matches and the aggregate
statistics surface are paid tiers and are deliberately not used here, so this
module works on a free key.

Rate limits are the real design constraint. The free tier allows 100 requests a
day, so a panel that refreshed every 30 seconds would exhaust it before lunch.
Fetches are therefore cached and demand-driven: nobody looking means no request,
and one fetch serves every viewer. `LIVE_TTL_SECONDS` sets the floor.

    tier    requests/day    sustainable TTL if polled all day
    FREE           100          ~900s
    BASIC        1,000           ~90s
    PRO         10,000            ~9s

Nothing here raises on a bad response. A dashboard panel that 500s because an
upstream hiccuped is worse than one that says "no live matches".
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API = "https://api.livetennisapi.com/api/public/v1"
TIMEOUT = 20

# Demand-driven cache. Raise LIVE_TTL_SECONDS on a free key, lower it on BASIC+.
DEFAULT_TTL = int(os.environ.get("LIVE_TTL_SECONDS", "300"))

# A manual refresh bypasses the TTL, but not this. Without a floor, holding the
# button down would spend the free tier's whole daily allowance in under a
# minute and the panel would then fail for everyone until midnight. Twenty
# seconds is below anything a person perceives as unresponsive and far above
# what it takes to protect the quota.
FORCE_MIN_INTERVAL = int(os.environ.get("LIVE_FORCE_MIN_SECONDS", "20"))
_cache: dict[str, tuple[float, list[dict]]] = {}


class LiveFeedError(RuntimeError):
    """Upstream said no. Carries the HTTP status so callers can tell 429 from 403."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def configured() -> bool:
    return bool(os.environ.get("LIVE_TENNIS_API_KEY", "").strip())


def _get(path: str) -> dict:
    key = os.environ.get("LIVE_TENNIS_API_KEY", "").strip()
    if not key:
        raise LiveFeedError("LIVE_TENNIS_API_KEY is not set")
    req = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {key}",
                 "Accept": "application/json",
                 "User-Agent": "tennis-engine/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise LiveFeedError(f"HTTP {e.code}: {body}", e.code) from e
    except Exception as e:  # noqa: BLE001
        raise LiveFeedError(f"{type(e).__name__}: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# Score parsing
# ──────────────────────────────────────────────────────────────────────────────
def current_games(score: dict) -> tuple[int, int]:
    """
    Games in the SET CURRENTLY BEING PLAYED, as (player 1, player 2).

    `games` is indexed [player][set] — confirmed against a live payload rather
    than inferred, because the shape alone is ambiguous and the two readings
    disagree about the score:

        sets  [1, 0]          player 1 leads by a set
        games [[6, 5], [3, 3]]

    Read per player that is set one 6-3 and a current set of 5-3, which is
    coherent. Read per set it would be a set of 6-5 and another of 3-3 — two
    sets in progress at once, which cannot happen. That rules the alternative
    out completely, so there is no heuristic here any more.

    The index is clamped rather than trusted: a feed that reports a set score
    inconsistent with its own game arrays should degrade to the last set it does
    have, not raise inside a dashboard request.
    """
    games = score.get("games") or []
    sets = score.get("sets") or [0, 0]
    try:
        done = int(sets[0]) + int(sets[1])
    except (TypeError, ValueError, IndexError):
        done = 0
    if len(games) < 2:
        return 0, 0
    a_row, b_row = games[0] or [], games[1] or []
    if not isinstance(a_row, (list, tuple)) or not isinstance(b_row, (list, tuple)):
        return 0, 0
    idx = min(done, len(a_row) - 1, len(b_row) - 1)
    if idx < 0:
        return 0, 0
    try:
        return int(a_row[idx]), int(b_row[idx])
    except (TypeError, ValueError):
        return 0, 0


def parse_state(match: dict) -> dict | None:
    """
    Turn a live match into keyword arguments for live_state.win_prob_from_state.

    Player 1 in the payload is player A here. Returns None when the match is not
    a scoreable singles main-draw match in progress.
    """
    if match.get("is_doubles") or match.get("is_qualifying"):
        return None
    if str(match.get("status", "")).lower() != "live":
        return None
    score = match.get("score") or {}
    sets = score.get("sets") or [0, 0]
    points = score.get("points") or [0, 0]
    ga, gb = current_games(score)
    try:
        sa, sb = int(sets[0]), int(sets[1])
    except (TypeError, ValueError, IndexError):
        return None
    return {
        "sets_a": sa,
        "sets_b": sb,
        "games_a": ga,
        "games_b": gb,
        "points_a": points[0] if len(points) > 0 else 0,
        "points_b": points[1] if len(points) > 1 else 0,
        # `server` is 1 or 2; player 1 is A.
        "a_serving": int(score.get("server") or 1) == 1,
        "in_tiebreak": bool(score.get("is_tiebreak")),
        "best_of": 5 if str(match.get("format", "")).upper() == "BO5" else 3,
    }


def scoreline(match: dict) -> str:
    """A human scoreline, e.g. '6-4 3-2'. Display only."""
    score = match.get("score") or {}
    games = score.get("games") or []
    sets = score.get("sets") or [0, 0]
    try:
        done = int(sets[0]) + int(sets[1])
    except (TypeError, ValueError, IndexError):
        done = 0
    out = []
    for i in range(done + 1):
        if len(games) == 2:
            if i < len(games[0] or []) and i < len(games[1] or []):
                out.append(f"{games[0][i]}-{games[1][i]}")
        elif i < len(games):
            pair = games[i]
            if pair and len(pair) >= 2:
                out.append(f"{pair[0]}-{pair[1]}")
    return " ".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# Fetch
# ──────────────────────────────────────────────────────────────────────────────
def live_matches(tour: str | None = None, ttl: int | None = None,
                 force: bool = False) -> list[dict]:
    """
    In-progress singles matches, cached for `ttl` seconds.

    One upstream call serves every viewer; see the rate-limit table at the top
    for why that matters on a free key.

    `force` is what the panel's Refresh button sends. Without it the button did
    nothing observable: the cache returned the same payload, the page re-rendered
    identical content, and it read as broken. It still honours FORCE_MIN_INTERVAL,
    so the button is responsive without being a way to burn the daily quota.
    """
    ttl = DEFAULT_TTL if ttl is None else int(ttl)
    key = tour or "all"
    hit = _cache.get(key)
    now = time.time()
    if hit:
        age = now - hit[0]
        if force:
            if age < FORCE_MIN_INTERVAL:
                return hit[1]
        elif age < ttl:
            return hit[1]

    path = "/matches?status=live" + (f"&tour={tour}" if tour else "")
    payload = _get(path)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    out = [m for m in (rows or [])
           if not m.get("is_doubles") and not m.get("is_qualifying")]
    _cache[key] = (now, out)
    return out


def cache_age(tour: str | None = None) -> float | None:
    """Seconds since the cached fetch, so the UI can show its own staleness."""
    hit = _cache.get(tour or "all")
    return None if not hit else time.time() - hit[0]


def clear_cache() -> None:
    _cache.clear()
