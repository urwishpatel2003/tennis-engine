"""
Calibration of the SCORE distributions — total games and game margin.

Inputs : the raw discrete distributions produced by engine/markov.py
Outputs: fair half-point lines, and a mapping that lets a real-world line be
         read back off the raw distribution as a probability.

Why this module exists
----------------------
The Markov chain assumes points are i.i.d. given the server. Real matches are
not — a player who breaks early often runs away with the set — so the raw chain
runs long. `predict.TOTAL_GAMES_CALIBRATION` corrects that for the EXPECTATION
and does it well (bias +1.80 → +0.27).

It was also being applied to the fair LINE, which is a median, not a mean. That
is a different statistic and the map does not transfer: matches went over the
supposedly-fair total only 42% of the time, and the handicap covered 48%.
Neither number was ever measured, because no test and no backtest looked at the
score markets at all — `backtest.py` scores win probability only.

The fix, fitted in tools/fit_score_calibration.py over 24,000 matches
(2018-2026, odd/even split, scored out of sample):

    calibrated = centre + spread · (raw − raw_median)
    centre     = a · raw_median + b        (quantile regression at tau = 0.5)

Two parameters do two separate jobs, which is the whole point. The earlier
single-slope affine map had to use one number for both: slope 0.73 was what it
took to centre bo3 totals, and it then shrank the distribution's WIDTH by 0.73
as a side effect, cutting the right tail short (PIT 0.31 in the top bin against
0.20 for a correct one). Fitting the width freely returns 1.00 — the raw chain's
dispersion was right all along, only its centre was wrong.

The constants are also keyed by TOUR. Pooling them looked fine in aggregate and
was not: one shared best-of-3 totals constant sent the ATP over 52.8% of the
time and the WTA 48.0%, averaging to a respectable-looking 50.4% that described
neither tour. Any aggregate score-market number here should be read alongside
the per-tour split, never instead of it.

A per-quantile fit (each tau level regressed separately) was tried and rejected:
it did not improve the 50/50 rate and made the PIT worse, so it did not earn its
table of constants.

The one thing this cannot fix
-----------------------------
Real total-games is more right-skewed than the model's — lots of straight-sets
matches, then a long thin tail. A location-scale family has no skew parameter,
so centring the median leaves the MEAN 0.77 games light (1.41 at best-of-5).
That is why totals keep two calibrations: `predict.calibrate_total_games` for
the reported expectation, and this module for the line and the probabilities.
The margin is near-symmetric and needs only one — its calibrated mean is within
0.10 games and its PIT is flat.
"""

from __future__ import annotations

import math

# ──────────────────────────────────────────────────────────────────────────────
# Fitted constants — tools/fit_score_calibration.py, 24,000 matches 2018-2026
# ──────────────────────────────────────────────────────────────────────────────
# Keyed by (kind, tour, best_of). Pooling the tours was the first thing tried
# and it does not hold: a shared best-of-3 totals constant sent ATP over 52.8%
# of the time and the WTA over 48.0%, averaging two tours into a number correct
# for neither. The fitted slopes are genuinely different — 0.57 for ATP against
# 0.75 for the WTA — because men's matches carry far more tiebreaks and hold
# streaks for the i.i.d. chain to over-extend.
#
# `centre` maps the raw median onto the observed median (quantile regression at
# tau = 0.5). Replayed over 12,000 completed matches the resulting lines split:
#
#     ATP best-of-3   handicap 50.3%   total 49.5%
#     ATP best-of-5   handicap 49.9%   total 50.0%
#     WTA best-of-3   handicap 51.0%   total 49.1%
#
# against 42.0% over and 48.1% cover before any of this, and against the 52.8 /
# 48.0 tour spread that the pooled constants left behind.
CENTRE = {
    "total": {
        "atp": {3: (0.5729, 8.105), 5: (0.9166, -2.249)},
        "wta": {3: (0.7503, 2.995)},
    },
    "margin": {
        "atp": {3: (1.1250, -0.500), 5: (1.3335, -0.666)},
        "wta": {3: (1.1848, -0.261)},
    },
}

# `spread` is the width multiplier, fitted independently by making the PIT
# histogram as flat as possible. Totals come out at 1.00 — the chain's
# dispersion was already right, and only the single-slope map made it look
# wrong. The margin wants ~1.15, i.e. real matches are slightly MORE lopsided
# than an i.i.d. model expects, which is the runaway-set effect seen from the
# other side.
#
# ATP best-of-5 rests on 2,366 matches, an order of magnitude fewer than the
# best-of-3 groups. Treat it as provisional and refit as the archive grows.
# There is no WTA best-of-5 row because the WTA does not play it; the fallback
# below covers the case rather than inventing a constant for it.
SPREAD = {
    "total":  {"atp": {3: 1.00, 5: 0.92}, "wta": {3: 1.00}},
    "margin": {"atp": {3: 1.14, 5: 1.24}, "wta": {3: 1.14}},
}


def _params(kind: str, tour: str, best_of: int) -> tuple[float, float, float]:
    """
    Look up (centre_slope, centre_intercept, spread), degrading gracefully.

    Falls back tour → best-of-3 → ATP rather than raising, matching the rest of
    the engine: a prediction with an unexpected tour or format still returns a
    line instead of erroring out.
    """
    def pick(table):
        by_tour = table[kind].get(str(tour).lower()) or table[kind]["atp"]
        return by_tour.get(int(best_of)) or by_tour.get(3) or table[kind]["atp"][3]

    a, b = pick(CENTRE)
    return a, b, pick(SPREAD)


def median_of(dist: dict) -> float:
    """Smallest value whose cumulative probability reaches one half."""
    if not dist:
        return float("nan")
    keys = sorted(dist)
    cum = 0.0
    for k in keys:
        cum += dist[k]
        if cum >= 0.5:
            return float(k)
    return float(keys[-1])


def margin_distribution(games: dict) -> dict[int, float]:
    """Collapse the joint (games_a, games_b) distribution onto A's margin."""
    out: dict[int, float] = {}
    for (ga, gb), p in games.items():
        out[ga - gb] = out.get(ga - gb, 0.0) + p
    return out


def calibrate(value: float, dist: dict, kind: str, tour: str, best_of: int) -> float:
    """Map a point on the RAW scale onto the observed scale."""
    a, b, s = _params(kind, tour, best_of)
    med = median_of(dist)
    return (a * med + b) + s * (float(value) - med)


def uncalibrate(value: float, dist: dict, kind: str, tour: str, best_of: int) -> float:
    """
    Inverse of `calibrate` — a real-world line back onto the raw scale.

    This is how probabilities stay honest without rescaling and re-binning the
    whole distribution: P(actual > L) is read as P(raw > uncalibrate(L)).
    """
    a, b, s = _params(kind, tour, best_of)
    med = median_of(dist)
    return med + (float(value) - (a * med + b)) / s


# Which anchor decides the line. Measured over 12,000 replayed matches, and the
# difference is not academic — the two rules disagree about where a coin flip is:
#
#   "centre"  floor the empirically fitted centre. Anchored to REALITY: the
#             centre comes from quantile regression against actual outcomes.
#   "best"    pick the half-point the model's own calibrated distribution splits
#             50/50. Anchored to the MODEL.
#
# "best" produces perfectly centred STATED probabilities (50.0% on both markets)
# but empirically worse lines, because it inherits whatever the chain's shape
# gets wrong. "centre" wins where it counts — the line is what the user acts on,
# so it is graded against what actually happened, not against the model's
# opinion of itself. See tools/validate_score_markets.py --rule.
LINE_RULE = "centre"


def fair_line(dist: dict, kind: str, tour: str, best_of: int,
              rule: str | None = None) -> float:
    """
    The fair half-point line — the value the outcome should beat half the time.

    Snapped to a STRICT half-point, never a whole number. Game counts are
    integers, so a whole-number line can be hit exactly, and those pushes score
    as losses on one side; real markets are quoted on half-points for the same
    reason.

    `rule` exists so the validator can grade both anchors in one pass rather
    than needing a code edit between runs. Production always uses LINE_RULE.
    """
    if not dist:
        return float("nan")
    centre = calibrate(median_of(dist), dist, kind, tour, best_of)
    if (rule or LINE_RULE) == "centre":
        return math.floor(centre) + 0.5

    # The "best" anchor. Flooring the centre ignores that the calibration
    # rescales the lattice: for a lopsided WTA best-of-3 the raw median margin
    # atom of 6 maps to 6.85 real games, so a 6.5 line sits BELOW its own median
    # atom and the probability quoted beside it reads 67%. This rule instead
    # scores the candidates on the calibrated distribution and takes the closest
    # to a true coin flip, breaking ties toward the fitted centre.
    best, best_key = None, None
    for step in (-2, -1, 0, 1, 2):
        line = math.floor(centre) + 0.5 + step
        thresh = uncalibrate(line, dist, kind, tour, best_of)
        cum = sum(p for k, p in dist.items() if k <= thresh)
        key = (round(abs((1.0 - cum) - 0.5), 6), abs(line - centre))
        if best_key is None or key < best_key:
            best, best_key = line, key
    return float(best)
