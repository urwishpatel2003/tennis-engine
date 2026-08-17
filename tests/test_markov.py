"""
Correctness tests for the scoring model.

    python tests/test_markov.py

No pytest dependency — this runs standalone, like the NFL engine's tests. Every
assertion is an identity that must hold for ANY valid implementation, not a
snapshot of what the current code happens to emit. That is the point: these
catch a broken refactor, not a changed constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import markov as mk  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ──────────────────────────────────────────────────────────────────────────────
print("\ngame_prob")
# A coin-flip server wins exactly half their service games.
check("p=0.5 → 0.5", close(mk.game_prob(0.5), 0.5))
# The textbook value, verified by hand at p=0.6, q=0.4:
#   0.1296 + 0.20736 + 0.20736 + 0.27648·(0.36/0.52) = 0.7357292307…
# This pins the whole closed form, deuce term included.
check("p=0.60 → 0.7357292 (hand-checked)",
      close(mk.game_prob(0.60), 0.7357292307692308, 1e-12),
      f"got {mk.game_prob(0.60)}")
check("monotone increasing",
      all(mk.game_prob(p) < mk.game_prob(p + 0.01) for p in [0.3, 0.5, 0.7, 0.85]))
check("p=1 → 1", close(mk.game_prob(0.999999), 1.0, 1e-4))
check("game_prob(p) + game_prob(1-p) = 1 at p=0.5", close(
    mk.game_prob(0.5) + mk.game_prob(0.5), 1.0))

print("\ntiebreak_prob")
check("equal servers → 0.5", close(mk.tiebreak_prob(0.6, 0.6), 0.5, 1e-9))
check("symmetry: tb(a,b) = 1 - tb(b,a)", close(
    mk.tiebreak_prob(0.68, 0.58), 1.0 - mk.tiebreak_prob(0.58, 0.68), 1e-9))
check("better server favoured", mk.tiebreak_prob(0.70, 0.55) > 0.5)
check("monotone in own serve",
      mk.tiebreak_prob(0.60, 0.60) < mk.tiebreak_prob(0.65, 0.60))
check("monotone in opponent serve",
      mk.tiebreak_prob(0.60, 0.65) < mk.tiebreak_prob(0.60, 0.60))

print("\nset_distribution")
for ha, hb in [(0.8, 0.8), (0.85, 0.6), (0.5, 0.9)]:
    d = dict(mk.set_distribution(ha, hb, 0.5, True))
    check(f"sums to 1 (holds {ha}/{hb})", close(sum(d.values()), 1.0, 1e-9),
          f"got {sum(d.values())}")
    check(f"only legal set scores (holds {ha}/{hb})",
          all((max(a, b) == 6 and min(a, b) <= 4) or (max(a, b) == 7 and min(a, b) in (5, 6))
              for a, b in d))
check("equal holds → 50% set win", close(
    mk.set_prob(0.75, 0.75, 0.5, True), 0.5, 1e-9))

print("\nmatch_distribution")
for bo in (3, 5):
    m = mk.match_distribution(0.64, 0.64, best_of=bo)
    check(f"bo{bo}: equal serves → 0.5", close(m["win_prob"], 0.5, 1e-9))
    check(f"bo{bo}: set_scores sum to 1", close(sum(m["set_scores"].values()), 1.0, 1e-9))
    check(f"bo{bo}: games sum to 1", close(sum(m["games"].values()), 1.0, 1e-9))
    check(f"bo{bo}: zero margin when symmetric", close(m["game_margin"], 0.0, 1e-9))
    need = 3 if bo == 5 else 2
    check(f"bo{bo}: winner always has {need} sets",
          all(max(a, b) == need for a, b in m["set_scores"]))

m3 = mk.match_distribution(0.64, 0.64, best_of=3)
m5 = mk.match_distribution(0.64, 0.64, best_of=5)
check("best-of-5 has more games than best-of-3",
      m5["exp_total_games"] > m3["exp_total_games"])
check("symmetry: P(A) = 1 - P(B) with roles swapped", close(
    mk.match_distribution(0.68, 0.60, 3)["win_prob"],
    1.0 - mk.match_distribution(0.60, 0.68, 3)["win_prob"], 1e-9))
check("best-of-5 favours the better player more",
      mk.match_distribution(0.68, 0.60, 5)["win_prob"]
      > mk.match_distribution(0.68, 0.60, 3)["win_prob"])
check("expected games = sum over joint distribution", close(
    m3["exp_games_a"] + m3["exp_games_b"], m3["exp_total_games"], 1e-9))

print("\nmatch_win_prob (fast DP) agrees with match_distribution")
for pa, pb, bo in [(0.64, 0.64, 3), (0.68, 0.60, 3), (0.70, 0.58, 5),
                   (0.55, 0.60, 5), (0.50, 0.75, 3), (0.80, 0.45, 5)]:
    full = mk.match_distribution(pa, pb, bo)["win_prob"]
    fast = mk.match_win_prob(pa, pb, bo)
    check(f"pa={pa} pb={pb} bo{bo}", close(full, fast, 1e-12),
          f"{full} vs {fast}")

print("\nscoreline enumeration")
sl = mk.match_distribution(0.68, 0.60, 3, track_scorelines=True)["scorelines"]
check("scorelines are pruned but near-complete", 0.98 < sum(sl.values()) <= 1.0,
      f"sum={sum(sl.values())}")
check("every scoreline has 2 or 3 sets", all(len(k) in (2, 3) for k in sl))
check("scoreline set counts match the winner",
      all(sum(1 for a, b in k if a > b) == 2 or sum(1 for a, b in k if b > a) == 2
          for k in sl))

print("\ninvert_to_target")
for target in (0.25, 0.50, 0.65, 0.85, 0.95):
    for bo in (3, 5):
        pa, pb = mk.invert_to_target(0.64, 0.62, target, bo)
        got = mk.match_win_prob(pa, pb, bo)
        check(f"bo{bo} hits target {target}", abs(got - target) < 0.005,
              f"got {got:.4f}")
# The inversion must move the serve GAP, not the serve LEVEL — that is what keeps
# the total-games line stable while the win probability is re-aimed.
base_a, base_b = 0.64, 0.62
for target in (0.30, 0.70):
    pa, pb = mk.invert_to_target(base_a, base_b, target, 3)
    check(f"preserves serve level at target {target}",
          abs((pa + pb) - (base_a + base_b)) < 2e-4,
          f"level {pa+pb:.4f} vs {base_a+base_b:.4f}")

print("\nmarket helpers")
d = mk.match_distribution(0.66, 0.61, 3)
tot = mk.total_games_distribution(d["games"])
check("total distribution sums to 1", close(sum(tot.values()), 1.0, 1e-9))
check("P(over) is decreasing in the line",
      mk.prob_over_games(d["games"], 18.5) > mk.prob_over_games(d["games"], 24.5))
check("P(over 0.5) = 1", close(mk.prob_over_games(d["games"], 0.5), 1.0, 1e-9))
check("handicap at 0 ≈ win-by-games probability",
      0.0 < mk.prob_cover_handicap(d["games"], -0.5) < 1.0)
check("huge handicap is a certainty",
      close(mk.prob_cover_handicap(d["games"], 99.5), 1.0, 1e-9))

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
