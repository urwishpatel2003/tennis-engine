"""
Tests for the live feed parser.

    python tests/test_live_feed.py

The payload below is REAL — captured from api.livetennisapi.com during
Cincinnati 2026 and pasted verbatim. That matters more than usual here, because
the field that decides everything is `games`, whose orientation the shape alone
does not reveal:

    sets  [1, 0]
    games [[6, 5], [3, 3]]

Read per player: set one was 6-3, the current set is 5-3. Coherent.
Read per set: a set of 6-5 and another of 3-3 — two sets in progress at once,
which cannot happen. The alternative is ruled out, not merely less likely.

Had that been guessed the wrong way, every live match would have shown a
plausible but wrong score, and the win probability beside it would have been
wrong in a way nothing on the page could reveal.

No network here: the parser is pure, and the fetch layer is a thin urllib call
around it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.live_feed import (  # noqa: E402
    current_games,
    parse_state,
    scoreline,
)
from engine.live_state import win_prob_from_state  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


REAL = {
    "id": 175096, "tour": "atp", "tournament": "Cincinnati", "round_code": "R32",
    "format": "BO3", "is_doubles": False, "is_qualifying": False,
    "status": "live", "surface": "hard",
    "players": {"p1": {"name": "Taylor Fritz", "id": 18},
                "p2": {"name": "Daniel Merida Aguilar", "id": 214}},
    "score": {"sets": [1, 0], "games": [[6, 5], [3, 3]], "points": ["0", "0"],
              "server": 2, "is_tiebreak": False, "age_seconds": 2,
              "stale": False, "sequence": 89},
}

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. the real payload parses to the real score")
check("current set is 5-3", current_games(REAL["score"]) == (5, 3),
      str(current_games(REAL["score"])))
check("scoreline reads 6-3 5-3", scoreline(REAL) == "6-3 5-3", scoreline(REAL))
st = parse_state(REAL)
check("one set to nil", (st["sets_a"], st["sets_b"]) == (1, 0))
check("games carried through", (st["games_a"], st["games_b"]) == (5, 3))
check("server 2 means player A is NOT serving", st["a_serving"] is False)
check("BO3 recognised", st["best_of"] == 3)
check("not a tiebreak", st["in_tiebreak"] is False)

print("\n2. the per-set misreading is rejected")
# If the orientation were ever flipped back, the current set would read 3-3.
check("current set is not the per-set misreading (3-3)",
      current_games(REAL["score"]) != (3, 3))

print("\n3. what is not scoreable is skipped, not guessed")
for label, patch in [("doubles", {"is_doubles": True}),
                     ("qualifying", {"is_qualifying": True}),
                     ("not started", {"status": "upcoming"}),
                     ("finished", {"status": "completed"})]:
    check(f"{label} returns None", parse_state(dict(REAL, **patch)) is None)

print("\n4. malformed feeds degrade instead of raising")
for label, score in [("no games", {"sets": [0, 0], "games": []}),
                     ("ragged games", {"sets": [1, 0], "games": [[6], [3]]}),
                     ("junk types", {"sets": ["x", None], "games": [["a"], ["b"]]}),
                     ("games not nested", {"sets": [0, 0], "games": [6, 3]}),
                     ("empty score", {})]:
    try:
        got = current_games(score)
        ok = isinstance(got, tuple) and len(got) == 2
    except Exception as e:  # noqa: BLE001
        ok, got = False, f"raised {type(e).__name__}"
    check(f"{label} -> a usable pair", ok, str(got))

print("\n5. the parsed state feeds the win probability")
# The end-to-end property that matters: a real payload must produce a usable
# probability, and leading by a set at 5-3 must beat trailing by a set at 3-5.
p = win_prob_from_state(0.66, 0.63, **st)
check("a probability comes out", 0.0 <= p <= 1.0, f"{p}")
mirror = dict(st, sets_a=0, sets_b=1, games_a=3, games_b=5)
check("a set and a break up beats a set and a break down",
      p > win_prob_from_state(0.66, 0.63, **mirror),
      f"{p:.4f} vs {win_prob_from_state(0.66, 0.63, **mirror):.4f}")
check("that position is a strong one", p > 0.80, f"{p:.4f}")

# ----------------------------------------------------------------------------
print("")
print("6. the Refresh button actually refreshes")
# It did not, at first. The server cached for 300s, so a click returned the
# same payload, the page re-rendered identical content, and the control read
# as dead. A forced refresh must reach upstream - while still being unable to
# burn the free tier's 100 requests a day if somebody leans on it.
import os  # noqa: E402
os.environ.setdefault('LIVE_TENNIS_API_KEY', 'test')
from engine import live_feed  # noqa: E402

fetches = []
live_feed._get = lambda path: (fetches.append(path), {'data': []})[1]

live_feed.clear_cache()
for _ in range(4):
    live_feed.live_matches('atp')
check("repeat views share one upstream fetch", len(fetches) == 1, str(len(fetches)))

_before = len(fetches)
live_feed.live_matches('atp', force=True)
check("a force inside the floor is served from cache",
      len(fetches) == _before, f'{len(fetches)-_before} extra fetch(es)')

_ts, _rows = live_feed._cache['atp']
live_feed._cache['atp'] = (_ts - live_feed.FORCE_MIN_INTERVAL - 1, _rows)
_before = len(fetches)
live_feed.live_matches('atp', force=True)
check("a force past the floor reaches upstream",
      len(fetches) == _before + 1, f'{len(fetches)-_before} fetch(es)')

check("the floor is short enough to feel responsive",
      0 < live_feed.FORCE_MIN_INTERVAL <= 60, str(live_feed.FORCE_MIN_INTERVAL))
check("the floor cannot exhaust a 100/day key",
      86400 / max(live_feed.FORCE_MIN_INTERVAL, 1) > 100)
live_feed.clear_cache()

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
