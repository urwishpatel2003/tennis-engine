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
0.20 for a correct one). Fitting the width freely returns 0.98 — the raw chain's
dispersion was right all along, only its centre was wrong.

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
# `centre` maps the raw median onto the observed median. Out-of-sample the
# resulting line split 50.4% / 49.6% for best-of-3 totals and 48.7% for the
# best-of-3 handicap, against 42.0% and 48.1% before.
CENTRE = {
    "total":  {3: (0.7267, 4.012), 5: (0.9166, -2.249)},
    "margin": {3: (1.1251, -0.375), 5: (1.3335, -0.666)},
}

# `spread` is the width multiplier, fitted independently by making the PIT
# histogram as flat as possible. Totals sit near 1.0 — the chain's dispersion
# was already right. The margin wants ~1.15, i.e. real matches are slightly
# MORE lopsided than an i.i.d. model expects, which is the same runaway-set
# effect seen from the other side.
#
# best-of-5 rests on 2,366 matches, an order of magnitude fewer than best-of-3.
# Treat those two constants as provisional and refit when the archive grows.
SPREAD = {
    "total":  {3: 0.98, 5: 0.92},
    "margin": {3: 1.14, 5: 1.24},
}


def _params(kind: str, best_of: int) -> tuple[float, float, float]:
    a, b = CENTRE[kind].get(int(best_of), CENTRE[kind][3])
    return a, b, SPREAD[kind].get(int(best_of), SPREAD[kind][3])


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


def calibrate(value: float, dist: dict, kind: str, best_of: int) -> float:
    """Map a point on the RAW scale onto the observed scale."""
    a, b, s = _params(kind, best_of)
    med = median_of(dist)
    return (a * med + b) + s * (float(value) - med)


def uncalibrate(value: float, dist: dict, kind: str, best_of: int) -> float:
    """
    Inverse of `calibrate` — a real-world line back onto the raw scale.

    This is how probabilities stay honest without rescaling and re-binning the
    whole distribution: P(actual > L) is read as P(raw > uncalibrate(L)).
    """
    a, b, s = _params(kind, best_of)
    med = median_of(dist)
    return med + (float(value) - (a * med + b)) / s


def fair_line(dist: dict, kind: str, best_of: int) -> float:
    """
    The fair half-point line — the value the outcome should beat half the time.

    Snapped to a STRICT half-point, never a whole number. Game counts are
    integers, so a whole-number line can be hit exactly, and those pushes score
    as losses on one side; real markets are quoted on half-points for the same
    reason.
    """
    if not dist:
        return float("nan")
    return math.floor(calibrate(median_of(dist), dist, kind, best_of)) + 0.5
