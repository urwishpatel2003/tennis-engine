"""
Discover the Live Tennis API's real response shapes before writing a parser.

    LIVE_TENNIS_API_KEY=... python tools/probe_live_tennis.py

Why a probe first
-----------------
engine/live_tennis.py has to map this API onto the canonical raw schema, and
getting that wrong is expensive: a mis-parsed round label or a missing serve
statistic is invisible until it has already corrupted the archive. Exactly that
happened with the WTA source, where the round was inferred from a match count
and produced draws with an SF but no QF.

So nothing is written against a guessed schema. This script calls the endpoints,
prints what actually comes back, and the parser is written to THAT.

It also cannot run from the development machine: the corporate proxy blocks
api.livetennisapi.com outright (Zscaler returns its own 403 block page, with
`Server: Zscaler/6.2`, before the request ever leaves the network). Railway's
containers have unrestricted internet, so this is designed to run there — as a
temporary build step whose output is read back with `railway logs --build`.

The key is read from the environment and never printed. Endpoint paths are
probed from a candidate list rather than assumed, because the published docs are
behind bot protection and could not be read directly.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://api.livetennisapi.com/api/public/v1"

# Tried in order; the first that authenticates wins. The docs say only that the
# key goes "in the request header", and separate sources describe a bearer
# token, so both spellings are attempted rather than guessed at.
AUTH_STYLES = [
    ("Authorization: Bearer", lambda k: {"Authorization": f"Bearer {k}"}),
    ("X-API-Key", lambda k: {"X-API-Key": k}),
    ("x-api-key", lambda k: {"x-api-key": k}),
    ("apikey", lambda k: {"apikey": k}),
]

# Endpoint guesses, derived from the 12 published tools. Anything that 404s is
# simply reported — the point is to learn which exist.
CANDIDATES = [
    ("health", "/health"),
    ("matches (live)", "/matches?status=live"),
    ("matches (completed)", "/matches?status=completed"),
    ("matches (finished)", "/matches?status=finished"),
    ("matches by tour", "/matches?tour=atp"),
    ("schedule", "/schedule"),
    ("tournaments", "/tournaments"),
    ("rankings", "/rankings?type=atp"),
    ("player search", "/players/search?name=alcaraz"),
]


def call(path: str, headers: dict) -> tuple[int, object]:
    req = urllib.request.Request(BASE + path, headers={
        "User-Agent": "tennis-engine/0.1", "Accept": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw[:400]
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        # A Zscaler block page is not an API response; say so plainly rather
        # than letting it read as "the endpoint does not exist".
        if "zscaler" in body.lower() or e.headers.get("Server", "").startswith("Zscaler"):
            return -1, "BLOCKED BY CORPORATE PROXY (Zscaler) — run this on Railway"
        return e.code, body
    except Exception as e:  # noqa: BLE001
        return -2, f"{type(e).__name__}: {e}"


def shape(obj, depth: int = 0, max_depth: int = 3) -> str:
    """A compact description of a JSON value: keys and types, not the data."""
    pad = "  " * depth
    if isinstance(obj, dict):
        if depth >= max_depth:
            return "{...}"
        return "\n".join(
            f"{pad}{k}: {shape(v, depth + 1, max_depth)}" for k, v in list(obj.items())[:25]
        )
    if isinstance(obj, list):
        if not obj:
            return "[] (empty)"
        return f"[{len(obj)} items] first:\n{shape(obj[0], depth + 1, max_depth)}"
    if isinstance(obj, str):
        return f"str {obj[:60]!r}"
    return f"{type(obj).__name__} {obj!r}"[:80]


def main() -> None:
    key = os.environ.get("LIVE_TENNIS_API_KEY", "").strip()
    if not key:
        print("LIVE_TENNIS_API_KEY is not set.")
        print("Get a free key at https://livetennisapi.com/subscribe/free and set")
        print("it as a Railway service variable. It is read from the environment")
        print("and is never printed by this script.")
        sys.exit(2)
    print(f"key present (length {len(key)}), never displayed\n")

    # 1. Which auth header does it accept?
    working = None
    for name, build in AUTH_STYLES:
        status, body = call("/matches?status=live", build(key))
        print(f"  auth via {name:<22} -> {status}")
        if status == -1:
            print(f"     {body}")
            sys.exit(3)
        if status == 200:
            working = (name, build)
            break
    if working is None:
        print("\nNo auth style returned 200. Last response:")
        print(f"  {body}")
        sys.exit(4)
    name, build = working
    print(f"\nAUTH HEADER: {name}\n")

    # 2. Which endpoints exist, and what do they return?
    for label, path in CANDIDATES:
        status, body = call(path, build(key))
        print(f"\n{'='*70}\n{label}  ->  {path}\n  HTTP {status}")
        if status == 200:
            print(shape(body))
        else:
            print(f"  {str(body)[:220]}")

    # 3. The question that decides whether this feed is usable at all: are FINAL
    #    serve statistics available for a match that has already finished? The
    #    published tool is called "In-Play Statistics", which may mean live only.
    #    Without final stats the engine's serve/return model gains nothing.
    print(f"\n{'='*70}\nFINAL STATS FOR A COMPLETED MATCH (the decisive question)")
    status, body = call("/matches?status=completed", build(key))
    mid = None
    if status == 200:
        items = body.get("data") if isinstance(body, dict) else body
        if isinstance(items, list) and items:
            first = items[0]
            for k in ("id", "match_id", "matchId", "uuid"):
                if isinstance(first, dict) and first.get(k):
                    mid = first[k]
                    break
            print(f"  sample completed match keys: {list(first)[:20]}")
    if mid is None:
        print("  could not find a completed match id — inspect the dump above")
        return
    for path in (f"/matches/{mid}/statistics", f"/matches/{mid}/stats", f"/matches/{mid}"):
        status, body = call(path, build(key))
        print(f"\n  {path} -> HTTP {status}")
        if status == 200:
            print(shape(body, depth=1))


if __name__ == "__main__":
    main()
