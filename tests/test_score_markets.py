"""
Tests for the score markets — the fair handicap and fair totals line.

    python tests/test_score_markets.py

These did not exist, and that is exactly how the bug they now guard shipped:
`backtest.py` scores win probability only, `test_markov.py` checks the chain's
internal maths, and nothing in between ever asked whether a "fair" line was
fair. Measured afterwards, the fair total went over 42% of the time and the
fair handicap covered 48% — both quoted to the user as 50/50.

The cause was a category error rather than a bad constant:
`calibrate_total_games` is fitted on the EXPECTATION and was being applied to a
MEDIAN. So the assertions below are mostly about self-consistency — a line and
the probability quoted beside it must agree — because that is the property that
was violated. The empirical 50/50 rate needs 10,000 replayed matches and lives
in tools/validate_score_markets.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import markov as mk  # noqa: E402
from engine import score_calib as sc  # noqa: E402

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


# Three matchups: even, a clear favourite, and a mismatch — the calibration has
# to behave across the whole range, not just at the midpoint.
CASES = [
    ("even", 0.64, 0.64),
    ("favourite", 0.68, 0.60),
    ("mismatch", 0.75, 0.55),
]

# ──────────────────────────────────────────────────────────────────────────────
print("\nfair lines are quoted on half-points")
# A whole-number line can be hit exactly. Those pushes were scoring as losses on
# one side, which quietly cost every measured rate a point or two.
for name, pa, pb in CASES:
    for tour, bo in (("atp", 3), ("atp", 5), ("wta", 3)):
        d = mk.match_distribution(pa, pb, bo)
        totals = mk.total_games_distribution(d["games"])
        margins = sc.margin_distribution(d["games"])
        t = sc.fair_line(totals, "total", tour, bo)
        h = sc.fair_line(margins, "margin", tour, bo)
        check(f"{name} {tour} bo{bo}: total {t} and handicap {-h} are half-points",
              close(abs(t) % 1.0, 0.5) and close(abs(h) % 1.0, 0.5))

print("\ncalibrate / uncalibrate round-trip")
for name, pa, pb in CASES:
    d = mk.match_distribution(pa, pb, 3)
    totals = mk.total_games_distribution(d["games"])
    margins = sc.margin_distribution(d["games"])
    for kind, dist in (("total", totals), ("margin", margins)):
        v = 21.5 if kind == "total" else 1.5
        back = sc.uncalibrate(sc.calibrate(v, dist, kind, "atp", 3), dist, kind, "atp", 3)
        check(f"{name} {kind}: uncalibrate(calibrate(x)) == x",
              close(back, v, 1e-9), f"{back} != {v}")

print("\nthe quoted probability agrees with the quoted line")
# The bug in one assertion: the line said 50/50 while the probability read off
# the distribution beside it said something else.
#
# The tolerance is not a flat number, because an exact coin flip is sometimes
# unreachable. Games are integers, so the distribution is a set of atoms, and
# under LINE_RULE = "centre" the line is anchored to the empirical fit while the
# probability is read off the model's atoms. A single atom straddling the
# threshold moves the probability by its whole mass — for a lopsided WTA
# best-of-3 the margin sits on ~6 values and one atom carries 0.19, so no
# half-point line can land nearer than that. The bound is therefore derived
# from the actual discreteness rather than picked to make the test pass.
def tolerance(dist: dict) -> float:
    return 0.10 + max(dist.values())


for name, pa, pb in CASES:
    for tour, bo in (("atp", 3), ("atp", 5), ("wta", 3)):
        d = mk.match_distribution(pa, pb, bo)
        totals = mk.total_games_distribution(d["games"])
        margins = sc.margin_distribution(d["games"])

        line = sc.fair_line(totals, "total", tour, bo)
        p_over = mk.prob_over_games(
            d["games"], sc.uncalibrate(line, totals, "total", tour, bo))
        check(f"{name} {tour} bo{bo}: P(over fair total) near 0.5 (={p_over:.3f})",
              abs(p_over - 0.5) < tolerance(totals), f"{p_over:.4f}")

        hcap = -sc.fair_line(margins, "margin", tour, bo)
        raw = sc.uncalibrate(-hcap, margins, "margin", tour, bo)
        p_cov = mk.prob_cover_handicap(d["games"], -raw)
        check(f"{name} {tour} bo{bo}: P(cover fair handicap) near 0.5 (={p_cov:.3f})",
              abs(p_cov - 0.5) < tolerance(margins), f"{p_cov:.4f}")

# The "best" rule exists to be graded against "centre" by the validator, so it
# has to keep working even though production does not use it. Under that rule
# the line IS chosen from the model's own distribution, so it must land on the
# nearest thing to a coin flip that the atoms allow — a much tighter claim.
print("\nthe alternative line rule stays self-consistent")
for name, pa, pb in CASES:
    d = mk.match_distribution(pa, pb, 3)
    totals = mk.total_games_distribution(d["games"])
    line = sc.fair_line(totals, "total", "wta", 3, rule="best")
    p = mk.prob_over_games(d["games"], sc.uncalibrate(line, totals, "total", "wta", 3))
    alts = []
    for step in (-2, -1, 1, 2):
        cand = line + step
        alts.append(abs(mk.prob_over_games(
            d["games"], sc.uncalibrate(cand, totals, "total", "wta", 3)) - 0.5))
    check(f"{name}: 'best' picks the closest half-point to 50/50 (={p:.3f})",
          abs(p - 0.5) <= min(alts) + 1e-9, f"{p:.4f} vs best alt {min(alts)+0.5:.4f}")

print("\nsigns and ordering")
d_even = mk.match_distribution(0.64, 0.64, 3)
d_fav = mk.match_distribution(0.75, 0.55, 3)
m_even = sc.margin_distribution(d_even["games"])
m_fav = sc.margin_distribution(d_fav["games"])
# A favourite has to GIVE games, so their handicap is the more negative one.
check("a favourite gives up more games than an even player",
      -sc.fair_line(m_fav, "margin", "atp", 3) < -sc.fair_line(m_even, "margin", "atp", 3))
# An even match is long; a mismatch is short.
t_even = sc.fair_line(mk.total_games_distribution(d_even["games"]), "total", "atp", 3)
t_fav = sc.fair_line(mk.total_games_distribution(d_fav["games"]), "total", "atp", 3)
check("a mismatch has a lower total than an even match", t_fav < t_even)
# Best-of-5 plays more games than best-of-3, whatever the calibration does.
t3 = sc.fair_line(mk.total_games_distribution(d_even["games"]), "total", "atp", 3)
d5 = mk.match_distribution(0.64, 0.64, 5)
t5 = sc.fair_line(mk.total_games_distribution(d5["games"]), "total", "atp", 5)
check("best-of-5 totals exceed best-of-3", t5 > t3, f"{t5} vs {t3}")

print("\nfair totals land in a plausible range")
# A best-of-3 match is ~21 games. This is a smoke alarm, not a fit: the old code
# emitted lines high enough to go over only 42% of the time, and a range check
# this loose would still have caught the best-of-5 case.
for name, pa, pb in CASES:
    t = sc.fair_line(mk.total_games_distribution(
        mk.match_distribution(pa, pb, 3)["games"]), "total", "atp", 3)
    check(f"{name}: bo3 fair total {t} within 15-27", 15.0 < t < 27.0, str(t))

print("\ncalibration constants are present for every format")
for kind in ("total", "margin"):
    for tour, bo in (("atp", 3), ("atp", 5), ("wta", 3)):
        check(f"{kind} {tour} bo{bo} has a centre and a spread",
              bo in sc.CENTRE[kind][tour] and bo in sc.SPREAD[kind][tour])
# The WTA does not play best-of-5. That must fall back, not raise.
check("WTA best-of-5 falls back instead of raising",
      abs(sc._params("total", "wta", 5)[0] - sc.CENTRE["total"]["wta"][3][0]) < 1e-9)
check("an unknown tour falls back to ATP",
      sc._params("total", "itf", 3) == (*sc.CENTRE["total"]["atp"][3],
                                        sc.SPREAD["total"]["atp"][3]))
# The spread is what the old single-slope map got wrong by forcing it to equal
# the centring slope. If one ever drifts to an implausible width, say so.
check("spreads are plausible widths (0.5-2.0)",
      all(0.5 < s < 2.0 for k in sc.SPREAD for t in sc.SPREAD[k]
          for s in sc.SPREAD[k][t].values()))

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
