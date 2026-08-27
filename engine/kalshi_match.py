"""
Find the Kalshi market for one of our fixtures — or refuse to guess.

Inputs : two player names and a tour
Outputs: the event, the two contracts, and which one backs which player

Getting this wrong spends real money on the wrong contract, so the whole module
is written to fail closed. Every refusal carries a reason.

Two traps, both real and both found in live data
------------------------------------------------
1. SERIES PREFIXES CATCH THE WRONG SPORT. Kalshi lists `KXWTABLETENNISMATCH`
   ("Women's Table Tennis Match"), which begins with `KXWTA`. Matching series by
   prefix would place tennis positions on table tennis. Only the exact tickers
   `KXATPMATCH` and `KXWTAMATCH` are used, and doubles, Challenger and
   set/game/spread series are deliberately excluded.

2. THE TICKER'S PLAYER CODE IS AMBIGUOUS. A ticker looks like
   `KXWTAMATCH-26AUG27TAUPAR-PAR`, where the suffix is a three-letter code. Live
   markets on one day contained:

       KXWTAMATCH-26AUG27TAUPAR-PAR   Diane Parry
       KXWTAMATCH-26AUG27PARANN-PAR   Alycia Parks

   The same code for different players. Matching on it would back Parks when the
   model meant Parry. Names come from `yes_sub_title` instead, which is
   unambiguous.

The safety property
-------------------
BOTH players must resolve inside the SAME event. Matching one player is not
enough — that is precisely how a fixture gets attached to the wrong opponent's
market. If only one side matches, this returns nothing and says so.
"""

from __future__ import annotations

import re
import unicodedata

from engine import kalshi

# Exact series only. Match winner, singles, main tour.
SERIES = {"atp": "KXATPMATCH", "wta": "KXWTAMATCH"}

# Series that a looser match would wrongly sweep in.
EXCLUDED_EXAMPLES = ("KXWTABLETENNISMATCH", "KXATPDOUBLES", "KXWTADOUBLES",
                     "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH",
                     "KXATPSETWINNER", "KXATPGAMESPREAD")


def norm(name: object) -> str:
    """Lowercase alphabetic key, accents stripped — 'Á. Müller' -> 'amuller'."""
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def surname(name: object) -> str:
    """Last whitespace-separated token, normalised. Used only as a fallback."""
    parts = [p for p in str(name or "").split() if p]
    return norm(parts[-1]) if parts else ""


def open_events(tour: str) -> dict:
    """
    Open match markets for a tour, grouped by event.

    Each event holds one contract per player; buying YES on a contract is
    backing that player to win.
    """
    series = SERIES.get(str(tour).lower())
    if not series:
        return {}
    events: dict[str, list] = {}
    for m in kalshi.markets(limit=1000, status="open", series=series):
        # Kalshi's status=open filter is not exact: a live pull returned 40
        # active markets and 10 already CLOSED ones. Trust the market's own
        # status over the filter, or tickets get built for matches that have
        # already finished.
        if str(m.get("status") or "active").lower() != "active":
            continue
        ev = m.get("event_ticker")
        if ev:
            events.setdefault(ev, []).append(m)
    # A match market has exactly two sides; anything else is not a fixture.
    return {k: v for k, v in events.items() if len(v) == 2}


def find_market(player_a: str, player_b: str, tour: str,
                events: dict | None = None) -> dict:
    """
    Locate the event for a fixture and say which contract backs which player.

    Returns {"ok": False, "reason": ...} rather than a guess. A fixture with no
    Kalshi market is the normal case, not an error.
    """
    evs = open_events(tour) if events is None else events
    if not evs:
        return {"ok": False, "reason": f"no open {tour.upper()} match markets"}

    a, b = norm(player_a), norm(player_b)
    if not a or not b:
        return {"ok": False, "reason": "both player names are required"}

    def sides(ms):
        return {norm(m.get("yes_sub_title") or m.get("title")): m for m in ms}

    # Exact full-name match on both sides, in the same event.
    for ev, ms in evs.items():
        by = sides(ms)
        if a in by and b in by:
            return {"ok": True, "event": ev, "match_type": "full name",
                    "a": by[a], "b": by[b],
                    "a_name": by[a].get("yes_sub_title"),
                    "b_name": by[b].get("yes_sub_title")}

    # Fallback: surnames, but ONLY when they identify exactly one event and are
    # unambiguous within it. Kalshi and our archive disagree on given names more
    # often than surnames ("Daniel Merida" vs "Daniel Merida Aguilar").
    sa, sb = surname(player_a), surname(player_b)
    hits = []
    for ev, ms in evs.items():
        names = {norm(m.get("yes_sub_title") or ""): m for m in ms}
        sur = {surname(m.get("yes_sub_title") or ""): m for m in ms}
        if len(sur) == 2 and sa in sur and sb in sur:
            hits.append((ev, sur[sa], sur[sb]))
    if len(hits) == 1:
        ev, ma, mb = hits[0]
        return {"ok": True, "event": ev, "match_type": "surname",
                "a": ma, "b": mb,
                "a_name": ma.get("yes_sub_title"), "b_name": mb.get("yes_sub_title")}
    if len(hits) > 1:
        return {"ok": False,
                "reason": f"surnames matched {len(hits)} events — too ambiguous to price"}

    return {"ok": False, "reason": "no Kalshi market for this fixture"}


def _num(value) -> float | None:
    """Kalshi V2 returns numbers as strings; treat both shapes as numbers."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ask_price(market: dict) -> float | None:
    """
    The price a taker pays to buy YES, in dollars.

    Reads `yes_ask_dollars` FIRST. Kalshi's V2 shape carries fixed-point dollar
    strings - "0.3700" - and leaves the legacy cent field `yes_ask` null, which
    is the same convention the order endpoint uses for `price`. Reading the cent
    field made every live market look like it had no offer resting, so every
    ticket was refused and the feature could never have produced a trade.

    The cent fallback stays for older payload shapes. No ask at all means there
    is nothing to buy, which is different from a price of zero.
    """
    p = _num(market.get("yes_ask_dollars"))
    if p is None:
        cents = _num(market.get("yes_ask"))
        p = cents / 100.0 if cents is not None else None
    if p is None:
        return None
    return p if 0.0 < p < 1.0 else None


def ask_size(market: dict) -> float | None:
    """
    How many contracts are actually offered at the ask.

    Sizing past this either partially fills or walks up the book, so a ticket is
    capped at what is really there rather than at what we would like.
    """
    return _num(market.get("yes_ask_size_fp"))
