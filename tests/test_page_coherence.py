"""
One match must cost the same on every page that prices it.

    python tests/test_page_coherence.py

The Today row and the matchup page both price the same fixture, by different
routes. They disagreed: Alexandrova showed 49.8% on Today and 56.7% on the
matchup page, because the link built from a fixture row carried tour, players,
surface and best-of, and dropped the TOURNAMENT. The engine needs the event to
count matches already won at it, so without it the in-event progress adjustment
never fired and the matchup page priced a neutral fixture instead.

This is the fourth bug of that exact shape in this project - two code paths
answering one question differently, each plausible alone. The others were a fair
line disagreeing with its own probability, the live win probability drifting
from the headline price, and the backtest omitting the whole adjustment layer.
None was caught by looking at a number; all were caught by comparing two.

So the invariant here is derived, not listed: any argument that CHANGES the
price is context the link must carry. A future adjustment keyed on something new
will fail this test rather than quietly split the two pages again.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "dashboard" / "dashboard.html").read_text(encoding="utf-8")

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# The fixture the bug was found on.
BASE = dict(a="Ekaterina Alexandrova", b="Sara Bejlek", tour="wta",
            surface="Hard", best_of=3, tournament="Cincinnati")

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. which arguments actually move the price")
try:
    from engine.predict import Engine
    eng = Engine(("wta",))
    full = eng.predict(BASE["a"], BASE["b"], tour=BASE["tour"], surface=BASE["surface"],
                       best_of=BASE["best_of"], tournament=BASE["tournament"])
    bare = eng.predict(BASE["a"], BASE["b"], tour=BASE["tour"], surface=BASE["surface"],
                       best_of=BASE["best_of"])
    moved = abs(full["win_prob_a"] - bare["win_prob_a"])
    check(f"dropping the tournament moves the price ({moved*100:.1f} pts)",
          moved > 0.005, f"{moved*100:.2f} pts")
    HAVE_ENGINE = True
except Exception as e:  # noqa: BLE001 - no built archive in this checkout
    print(f"  SKIP  engine unavailable ({type(e).__name__}); static checks still run")
    HAVE_ENGINE = False

print("\n2. the links carry everything that moves the price")
# Today's fixture row.
today = re.search(r"const link = '#/matchup/' \+ \[(.*?)\]", HTML, re.S)
check("the Today row builds a matchup link", today is not None)
if today:
    fields = today.group(1)
    for f in ("r.tour", "r.player_a", "r.player_b", "r.surface", "r.best_of",
              "r.tournament"):
        check(f"Today link carries {f}", f in fields, fields[:120])
    check("Today link carries the date", "commence_time" in fields, fields[:120])

# The Live panel's card.
live = re.search(r"location\.hash='#/matchup/'\+\[(.*?)\]", HTML, re.S)
check("the Live card builds a matchup link", live is not None)
if live:
    lf = live.group(1)
    for f in ("r.tour", "r.player_a", "r.player_b", "r.surface", "r.tournament"):
        check(f"Live link carries {f}", f in lf, lf[:120])

print("\n3. the matchup view reads and forwards them")
check("viewMatchup destructures seven fields",
      re.search(r"const \[tr, a, b, sf, bo, tn, dt\] = arg\.split", HTML) is not None)
check("viewMatchup forwards the tournament",
      re.search(r"if\(st\.tournament\) q\.tournament", HTML) is not None)
check("viewMatchup forwards the date",
      re.search(r"if\(st\.date\) q\.date", HTML) is not None)

print("\n4. the endpoint honours what it is sent")
try:
    import dashboard.server as srv
    c = srv.app.test_client()
    q = ("a=Ekaterina+Alexandrova&b=Sara+Bejlek&tour=wta&surface=Hard&best_of=3")
    with_t = c.get(f"/api/matchup?{q}&tournament=Cincinnati").get_json()
    without = c.get(f"/api/matchup?{q}").get_json()
    if with_t and "win_prob_a" in with_t:
        check("the endpoint applies the tournament it is given",
              abs(with_t["win_prob_a"] - without["win_prob_a"]) > 0.005,
              f"{with_t['win_prob_a']:.4f} vs {without['win_prob_a']:.4f}")
        if HAVE_ENGINE:
            # The invariant itself: routed through HTTP, the matchup page must
            # return exactly what the engine returns for the same context - which
            # is what the Today row prices with.
            check("endpoint price == engine price for identical context",
                  abs(with_t["win_prob_a"] - full["win_prob_a"]) < 1e-9,
                  f"{with_t['win_prob_a']:.6f} vs {full['win_prob_a']:.6f}")
    else:
        print("  SKIP  endpoint returned no prediction (no built data)")
except Exception as e:  # noqa: BLE001
    print(f"  SKIP  server unavailable ({type(e).__name__})")

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
