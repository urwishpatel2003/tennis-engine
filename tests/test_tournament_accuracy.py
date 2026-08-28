"""
The listing's accuracy must equal the draw's accuracy.

    python tests/test_tournament_accuracy.py

Two screens report "model accuracy" for the same event. The draw page computes
it from full predictions; the listing uses a probability-only path that skips
reconciling the score model, because that reconciliation is two thirds of the
cost and cannot change the headline probability.

"Cannot change it" is the claim worth testing. If it ever does, two screens
start disagreeing about the same number and neither is obviously wrong - which
is a far worse failure than the listing simply being slow.

Uses real data when it is present, and says so plainly when it is not, rather
than passing vacuously.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


from engine.tournament import TournamentStore  # noqa: E402

print("\n1. the shared scorer is the only definition of a hit")
S = TournamentStore._score
check("a favoured winner is a hit", S([(0.6, True)])["hits"] == 1)
check("exactly 0.5 counts as favoured", S([(0.5, True)])["hits"] == 1)
check("below 0.5 is a miss", S([(0.49, True)])["hits"] == 0)
check("an unplayed match is not scored", S([(0.9, False)])["scored"] == 0)
check("a missing prediction is not scored", S([(None, True)])["scored"] == 0)
check("no scored matches means no accuracy", S([(None, True)])["accuracy"] is None)
check("accuracy is hits over scored",
      S([(0.9, True), (0.4, True), (0.8, True), (0.1, False)])["accuracy"] == 2 / 3)

print("\n2. the fast path agrees with the full one")
try:
    st = TournamentStore()
    evs = [e for e in st.list_tournaments("atp", 2026) if e["matches"] >= 20][:2]
except Exception as e:
    evs = []
    print(f"  SKIP  no local archive ({type(e).__name__}) — this check needs data/")

if not evs:
    print("  SKIP  no ATP 2026 events available locally")
else:
    for e in evs:
        tid = e["tourney_id"]
        fast = st.accuracy("atp", tid)
        full = st.draw("atp", tid)["summary"]
        check(f"{e['name'][:24]}: accuracy matches",
              fast["accuracy"] == full["accuracy"],
              f"{fast['accuracy']} vs {full['accuracy']}")
        check(f"{e['name'][:24]}: hits match", fast["hits"] == full["hits"],
              f"{fast['hits']} vs {full['hits']}")
        check(f"{e['name'][:24]}: scored count matches",
              fast["scored"] == full["scored"],
              f"{fast['scored']} vs {full['scored']}")

print("\n3. the fast path really is cheaper")
if evs:
    import time
    st2 = TournamentStore()
    tid = evs[0]["tourney_id"]
    t0 = time.time(); st2.accuracy("atp", tid); fast_s = time.time() - t0
    t1 = time.time(); st2.draw("atp", tid); full_s = time.time() - t1
    check("skipping the score reconciliation saves real time",
          fast_s < full_s, f"{fast_s:.2f}s vs {full_s:.2f}s")
    print(f"        ({fast_s:.2f}s vs {full_s:.2f}s)")

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
