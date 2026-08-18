"""
Tests for the in-match win probability.

    python tests/test_live_state.py

This module is exact arithmetic, not a fit, so the assertions are identities
that must hold for any correct implementation rather than snapshots of what the
code currently emits. The first one is the important one: entered at 0-0 the
recursion must reproduce markov.match_win_prob. Getting that wrong is how a live
figure ends up quietly disagreeing with the headline price beside it, which is
the same defect class as a fair line disagreeing with its own probability.

It caught a real bug already. The base chain quantises probabilities to 1e-4 for
cache friendliness; the first version of this recursion did not, and drifted from
match_win_prob by ~1.5e-4 per set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import markov as mk  # noqa: E402
from engine.live_state import (  # noqa: E402
    game_prob_from,
    leverage,
    point_index,
    win_prob_from_state,
)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


CASES = [(0.64, 0.64, 3), (0.68, 0.60, 3), (0.62, 0.66, 5), (0.75, 0.55, 5)]

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. entering at 0-0 reproduces the base model")
for pa, pb, bo in CASES:
    base = mk.match_win_prob(pa, pb, bo)
    for srv in (True, False):
        got = win_prob_from_state(pa, pb, best_of=bo, a_serving=srv)
        check(f"pa={pa} pb={pb} bo={bo} a_serving={srv} matches match_win_prob",
              abs(got - base) < 2e-5, f"{got:.8f} vs {base:.8f}")

print("\n2. point scores inside a game")
p = 0.65
check("40-0 on serve beats 0-40 on serve",
      game_prob_from(p, 3, 0) > game_prob_from(p, 0, 3))
check("game already won is certain", game_prob_from(p, 4, 0) == 1.0)
check("game already lost is impossible", game_prob_from(p, 0, 4) == 0.0)
check("deuce sits between AD-40 and 40-AD",
      game_prob_from(p, 4, 3) > game_prob_from(p, 3, 3) > game_prob_from(p, 3, 4))
# Deuce is a fixed point: 40-40, 5-5, 6-6 are all the same state.
check("deuce is scale-free (40-40 == 5-5)",
      abs(game_prob_from(p, 3, 3) - game_prob_from(p, 5, 5)) < 1e-12)
check("a stronger server wins more games from deuce",
      game_prob_from(0.75, 3, 3) > game_prob_from(0.55, 3, 3))

print("\n3. the score moves the number in the right direction")
pa, pb = 0.66, 0.63
lead = win_prob_from_state(pa, pb, sets_a=1, sets_b=0, best_of=3)
level = win_prob_from_state(pa, pb, sets_a=0, sets_b=0, best_of=3)
trail = win_prob_from_state(pa, pb, sets_a=0, sets_b=1, best_of=3)
check("a set up beats level beats a set down", lead > level > trail,
      f"{lead:.4f} {level:.4f} {trail:.4f}")
up = win_prob_from_state(pa, pb, games_a=5, games_b=0, best_of=3)
dn = win_prob_from_state(pa, pb, games_a=0, games_b=5, best_of=3)
check("5-0 up beats 0-5 down", up > dn, f"{up:.4f} {dn:.4f}")
check("winning the match outright is certain",
      win_prob_from_state(pa, pb, sets_a=2, best_of=3) == 1.0)
check("losing the match outright is impossible",
      win_prob_from_state(pa, pb, sets_b=2, best_of=3) == 0.0)
check("best-of-5 recovers from a set down better than best-of-3",
      win_prob_from_state(pa, pb, sets_a=0, sets_b=1, best_of=5)
      > win_prob_from_state(pa, pb, sets_a=0, sets_b=1, best_of=3))

print("\n4. symmetry")
# Swapping both players and their serve rates must mirror the probability.
for pa_, pb_, bo in CASES:
    a = win_prob_from_state(pa_, pb_, sets_a=1, sets_b=0, games_a=3, games_b=2,
                            a_serving=True, best_of=bo)
    b = win_prob_from_state(pb_, pa_, sets_a=0, sets_b=1, games_a=2, games_b=3,
                            a_serving=False, best_of=bo)
    check(f"mirrored state mirrors the probability (pa={pa_}, bo={bo})",
          abs(a - (1.0 - b)) < 2e-5, f"{a:.6f} vs {1-b:.6f}")
# Identical players, symmetric score, no serve advantage in play -> a coin flip.
check("identical players at a symmetric score are 50/50",
      abs(win_prob_from_state(0.64, 0.64, sets_a=1, sets_b=1, games_a=3,
                              games_b=3, best_of=3) - 0.5) < 1e-3)

print("\n5. tiebreaks")
tb_lead = win_prob_from_state(0.65, 0.65, games_a=6, games_b=6, points_a=6,
                              points_b=0, in_tiebreak=True, best_of=3)
tb_trail = win_prob_from_state(0.65, 0.65, games_a=6, games_b=6, points_a=0,
                               points_b=6, in_tiebreak=True, best_of=3)
check("6-0 up in a tiebreak beats 0-6 down", tb_lead > tb_trail,
      f"{tb_lead:.4f} {tb_trail:.4f}")
# Bounded against the states either side rather than against a number picked by
# intuition. 6-0 in a FIRST-set tiebreak wins a SET, not the match, so it must
# sit just below "already a set up" — an earlier version of this test asserted
# > 0.75 and failed at 0.7471, because 0.98 x P(match | set up) is about 0.745.
# The model was right and the assertion was guesswork.
set_up = win_prob_from_state(0.65, 0.65, sets_a=1, sets_b=0, best_of=3)
set_down = win_prob_from_state(0.65, 0.65, sets_a=0, sets_b=1, best_of=3)
check("a 6-0 tiebreak lead is worth nearly, but not more than, the set itself",
      set_down < tb_lead < set_up and tb_lead > 0.9 * set_up,
      f"{set_down:.4f} < {tb_lead:.4f} < {set_up:.4f}")

print("\n6. leverage")
# A point at 5-1, 30-0 barely matters; break point in a decider is worth a lot.
dead = leverage(0.65, 0.65, sets_a=1, sets_b=0, games_a=5, games_b=1,
                points_a="30", points_b="0", a_serving=True, best_of=3)
live = leverage(0.65, 0.65, sets_a=1, sets_b=1, games_a=4, games_b=5,
                points_a="30", points_b="40", a_serving=True, best_of=3)
check("break point in a decider outweighs 30-0 at 5-1", live > dead,
      f"live={live:.4f} dead={dead:.4f}")
check("leverage is a probability difference", 0.0 <= live <= 1.0)

print("\n7. scoreboard values parse")
check("'40' is three points", point_index("40") == 3)
check("'AD' is four points", point_index("AD") == 4)
check("'0' is none", point_index("0") == 0)
check("an integer passes through", point_index(2) == 2)
check("nonsense degrades to zero rather than raising", point_index("???") == 0)
# The live feed sends strings; a mismatch here would silently mis-score a match.
check("string and integer forms agree",
      win_prob_from_state(0.65, 0.62, points_a="30", points_b="15")
      == win_prob_from_state(0.65, 0.62, points_a=2, points_b=1))

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
