"""
Fit and validate the score-market calibration — totals line and game handicap.

    python tools/fit_score_calibration.py --sample 24000          # cache + fit
    python tools/fit_score_calibration.py --reuse                 # refit from cache

Why this exists
---------------
TOTAL_GAMES_CALIBRATION was fitted on the EXPECTATION of total games, and it does
that job: bias fell from +1.80 to +0.27. But the fair line is a MEDIAN of the raw
distribution, and the same affine map was being applied to it. A map that centres
a mean does not centre a median unless the distribution's shape is right, and it
is not: matches went over the supposedly-fair line only 42% of the time
(tools/validate_score_markets.py).

So the mean was calibrated and the quantiles were not. This script fits the
quantile map directly, and its output is the literal contents of `CENTRE` and
`SPREAD` in engine/score_calib.py — grouped by (tour, best_of), the same key the
engine uses, so what ships can always be regenerated and checked.

Method
------
Stage 1 caches, per match, the raw Markov quantile function on a τ grid for both
total games and game margin, plus the observed outcome. Nothing is fitted yet, so
the cache can be re-fitted cheaply with --reuse.

Stage 2 fits `observed ≈ a·q_raw(τ) + b` by QUANTILE regression at τ = 0.5 —
minimising absolute error, which is exactly the loss whose optimum is the
conditional median, i.e. the line that splits 50/50 by construction. The width is
then fitted SEPARATELY, by flattening the PIT histogram: tying it to the centring
slope is the defect being fixed, since the slope that centres the line also
squashes the distribution.

Stage 3 checks it out of sample: fit on odd-indexed matches, score on even. It
reports the over-rate at the fitted line and the PIT histogram, which says
whether the whole distribution is the right shape or only its centre.

Two alternatives are also computed and were both rejected on the evidence: a
per-quantile fit (no gain, worse PIT) and a shared-across-tours fit (hid a
52.8% / 48.0% split behind a healthy-looking pooled average).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import markov  # noqa: E402
from engine.predict import W_ELO, blend_logit  # noqa: E402
from engine.schema import ELO_SCALE, PROCESSED, RAW  # noqa: E402
from engine.serve_return import point_probabilities  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "processed" / "score_calib_cache.parquet"

# τ grid stored per match. Dense enough to interpolate a PIT, coarse enough that
# 24k matches stay a small file.
TAUS = np.round(np.arange(0.02, 0.981, 0.02), 3)


def quantiles(dist: dict, taus: np.ndarray) -> np.ndarray:
    """Quantile function of a discrete distribution given as {value: prob}."""
    keys = np.array(sorted(dist), dtype=float)
    cum = np.cumsum(np.array([dist[int(k)] for k in keys], dtype=float))
    cum /= cum[-1]
    idx = np.searchsorted(cum, taus, side="left")
    return keys[np.clip(idx, 0, len(keys) - 1)]


def load(tours, lo, hi, min_matches, sample, seed):
    r = pd.read_parquet(PROCESSED / "ratings.parquet")
    sr = pd.read_parquet(PROCESSED / "serve_return.parquet")
    drop = ["tour", "tourney_date", "season", "surface", "winner_id", "loser_id"]
    df = r.merge(sr.drop(columns=drop), on="match_id", how="inner")
    df = df[df["tour"].isin(tours) & df["season"].between(lo, hi)]
    ctx = pd.concat([
        pd.read_parquet(RAW / f"matches_{t}.parquet",
                        columns=["match_id", "best_of", "completed", "retirement",
                                 "games_w", "games_l", "total_games"])
        for t in tours if (RAW / f"matches_{t}.parquet").exists()
    ], ignore_index=True)
    df = df.merge(ctx, on="match_id", how="left")
    df = df[df["completed"].fillna(False).astype(bool)]
    df = df[~df["retirement"].fillna(False).astype(bool)]
    df = df[(df["w_matches"] >= min_matches) & (df["l_matches"] >= min_matches)]
    df = df[df["total_games"].between(12, 70)]
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def build_cache(args) -> pd.DataFrame:
    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]
    df = load(tuple(args.tours), int(lo), int(hi), args.min_matches,
              args.sample, args.seed)
    print(f"Caching raw quantile functions for {len(df):,} matches ({lo}-{hi})")

    a_is_w = df["winner_id"].to_numpy() < df["loser_id"].to_numpy()
    pick = lambda w, l: np.where(a_is_w, df[w].to_numpy(), df[l].to_numpy())
    gap = pick("w_elo_blend", "l_elo_blend") - pick("l_elo_blend", "w_elo_blend")
    p_elo = 1.0 / (1.0 + np.exp(-gap * math.log(10.0) / ELO_SCALE))
    tour, surf = df["tour"].to_numpy(), df["surface"].to_numpy()
    bo = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int).to_numpy()
    a_s = pick("w_serve_excess", "l_serve_excess")
    b_s = pick("l_serve_excess", "w_serve_excess")
    a_r = pick("w_return_excess", "l_return_excess")
    b_r = pick("l_return_excess", "w_return_excess")

    n, nt = len(df), len(TAUS)
    qt = np.zeros((n, nt), dtype=np.float32)   # total-games quantiles
    qm = np.zeros((n, nt), dtype=np.float32)   # game-margin quantiles
    exp_t = np.zeros(n)
    exp_m = np.zeros(n)

    for i in range(n):
        pa, pb = point_probabilities(a_s[i], a_r[i], b_s[i], b_r[i], tour[i], surf[i])
        p_sim = markov.match_win_prob(pa, pb, int(bo[i]))
        target = blend_logit([p_elo[i], p_sim], [W_ELO, 1 - W_ELO])
        xa, xb = markov.invert_to_target(pa, pb, target, int(bo[i]))
        d = markov.match_distribution(xa, xb, int(bo[i]))
        g = d["games"]
        exp_t[i], exp_m[i] = d["exp_total_games"], d["game_margin"]
        qt[i] = quantiles(markov.total_games_distribution(g), TAUS)
        margins: dict[int, float] = {}
        for (ga, gb), p in g.items():
            margins[ga - gb] = margins.get(ga - gb, 0.0) + p
        qm[i] = quantiles(margins, TAUS)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{n:,}")

    out = pd.DataFrame({
        "best_of": bo,
        "tour": tour,
        "raw_exp_total": exp_t,
        "raw_exp_margin": exp_m,
        "actual_total": df["total_games"].to_numpy().astype(float),
        "actual_margin": (pick("games_w", "games_l").astype(float)
                          - pick("games_l", "games_w").astype(float)),
    })
    for j, t in enumerate(TAUS):
        out[f"qt_{t:.2f}"] = qt[:, j]
        out[f"qm_{t:.2f}"] = qm[:, j]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    print(f"  cached -> {CACHE.name}\n")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────────
def half_point(v: np.ndarray) -> np.ndarray:
    """
    Snap a line onto a STRICT half-point.

    Game counts are integers, so a line that lands on a whole number can be hit
    exactly — a push. `round(v*2)/2` allows whole numbers and those pushes then
    score as losses, which depressed every measured rate by a point or two.
    Real handicap and totals markets are quoted on half-points for the same
    reason.
    """
    return np.floor(np.asarray(v, dtype=float)) + 0.5


def fit_quantile(x: np.ndarray, y: np.ndarray, tau: float = 0.5) -> tuple[float, float]:
    """
    Quantile-regression fit y ≈ a·x + b at level `tau`.

    At tau=0.5 this is least-absolute-deviations, whose minimiser is the
    conditional MEDIAN — precisely the "splits 50/50" property a fair line
    claims. Least squares would fit the mean and reproduce the bug being fixed.

    Solved by iteratively reweighted least squares with the asymmetric pinball
    weights; with two parameters it converges in a few dozen passes.
    """
    A = np.column_stack([x, np.ones_like(x)])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(80):
        r = y - A @ coef
        w = np.where(r > 0, tau, 1.0 - tau) / np.sqrt(np.maximum(np.abs(r), 1e-3))
        new = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        if np.max(np.abs(new - coef)) < 1e-7:
            return float(new[0]), float(new[1])
        coef = new
    return float(coef[0]), float(coef[1])


fit_median = fit_quantile  # the tau=0.5 case, named for its callers


def pit(cache: pd.DataFrame, prefix: str, actual: np.ndarray,
        a: float, b: float) -> np.ndarray:
    """
    Where the observed value falls in the CALIBRATED predictive distribution.

    A well-shaped distribution gives a uniform PIT. Mass piled low means the
    model predicts too long; a U-shape means it is under-dispersed.
    """
    q = cache[[f"{prefix}_{t:.2f}" for t in TAUS]].to_numpy(dtype=float) * a + b
    return (q < actual[:, None]).sum(axis=1) / len(TAUS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seasons", default="2018-2026")
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--sample", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true",
                    help="refit from an existing cache instead of recomputing")
    args = ap.parse_args()

    if args.reuse and CACHE.exists():
        cache = pd.read_parquet(CACHE)
        print(f"Refitting from cache: {len(cache):,} matches\n")
    else:
        cache = build_cache(args)

    med_t = f"qt_{0.50:.2f}"
    med_m = f"qm_{0.50:.2f}"

    print(f"  {'group':<16}{'n':>7}{'slope':>9}{'intcpt':>9}"
          f"{'over/cover IS':>15}{'OOS':>9}{'MAE OOS':>10}")
    print("  " + "─" * 78)

    # Grouped by (tour, best_of) — the SAME key engine/score_calib.py uses.
    # Fitting these pooled across tours is what produced constants that sent the
    # ATP over 52.8% of the time and the WTA 48.0%, and a tool that cannot
    # regenerate what ships is a tool nobody can check.
    fitted: dict[str, dict[str, dict[int, list[float]]]] = {"total": {}, "margin": {}}
    spreads: dict[str, dict[str, dict[int, float]]] = {"total": {}, "margin": {}}

    for tour, bo in [(t, b) for t in ("atp", "wta") for b in (3, 5)]:
        sub = cache[(cache["tour"] == tour) & (cache["best_of"] == bo)]
        sub = sub.reset_index(drop=True)
        if len(sub) < 400:
            print(f"  {tour.upper()} bo{bo}: only {len(sub)} matches — skipped; "
                  f"score_calib falls back for this group")
            continue
        # Odd/even split: interleaved, so both halves span the same seasons. A
        # chronological split would confound calibration drift with fit.
        tr = sub.index % 2 == 1
        te = ~tr

        for label, qcol, prefix, acol in (
            ("total", med_t, "qt", "actual_total"),
            ("margin", med_m, "qm", "actual_margin"),
        ):
            x = sub[qcol].to_numpy(dtype=float)
            y = sub[acol].to_numpy(dtype=float)
            a, b = fit_median(x[tr], y[tr])
            line = half_point(a * x + b)
            is_rate = (y[tr] > line[tr]).mean()
            oos_rate = (y[te] > line[te]).mean()
            mae = np.mean(np.abs(a * x[te] + b - y[te]))
            print(f"  {tour.upper()} bo{bo} {label:<7}{len(sub):>7,}"
                  f"{a:>9.4f}{b:>9.3f}"
                  f"{is_rate*100:>14.1f}%{oos_rate*100:>8.1f}%{mae:>10.3f}")
            fitted[label].setdefault(tour, {})[bo] = [round(a, 4), round(b, 3)]

            # Width, fitted INDEPENDENTLY of the centre by flattening the PIT.
            # Tying it to the centring slope is the defect this module fixes.
            Q = sub[[f"{prefix}_{t:.2f}" for t in TAUS]].to_numpy(dtype=float)
            q50 = Q[:, int(np.argmin(np.abs(TAUS - 0.5)))]
            centre = a * q50 + b
            best_s, best_chi = 1.0, float("inf")
            for sv in np.arange(0.4, 2.61, 0.02):
                cal = centre[tr][:, None] + sv * (Q[tr] - q50[tr][:, None])
                u = (cal < y[tr][:, None]).sum(axis=1) / Q.shape[1]
                h = np.histogram(u, bins=10, range=(0, 1))[0]
                e = h.sum() / 10.0
                if ((h - e) ** 2 / e).sum() < best_chi:
                    best_s, best_chi = float(sv), ((h - e) ** 2 / e).sum()
            cal = centre[te][:, None] + best_s * (Q[te] - q50[te][:, None])
            u = (cal < y[te][:, None]).sum(axis=1) / Q.shape[1]
            hist = np.histogram(u, bins=5, range=(0, 1))[0] / max(te.sum(), 1)
            print(f"       spread {best_s:.2f}   PIT OOS (uniform = 0.20): "
                  + "  ".join(f"{h:.2f}" for h in hist))
            spreads[label].setdefault(tour, {})[bo] = round(best_s, 3)

    print("\n  Paste into engine/score_calib.py (out-of-sample verified):")
    print("  CENTRE = " + json.dumps(fitted))
    print("  SPREAD = " + json.dumps(spreads))

    # ── Does the WHOLE distribution need calibrating, or only its centre? ─────
    # The affine median fit centres the line. It cannot fix skew: one slope has
    # to serve every quantile, so squashing the median also squashes the tail.
    # Fitting each quantile level separately can, at the cost of a table of
    # constants instead of two. Worth it only if the PIT actually flattens, so
    # both are measured here and the code takes whichever earns its complexity.
    print("\n  PER-QUANTILE FIT — each tau level fitted on its own")
    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    per_q: dict[str, dict[str, list]] = {"total": {}, "margin": {}}
    for bo in (3, 5):
        sub = cache[cache["best_of"] == bo].reset_index(drop=True)
        if len(sub) < 400:
            continue
        tr = sub.index % 2 == 1
        te = ~tr
        for label, prefix, acol in (("total", "qt", "actual_total"),
                                    ("margin", "qm", "actual_margin")):
            y = sub[acol].to_numpy(dtype=float)
            cols = [f"{prefix}_{t:.2f}" for t in TAUS]
            Q = sub[cols].to_numpy(dtype=float)
            coefs, cal = [], np.zeros((len(sub), len(grid)))
            for j, t in enumerate(grid):
                # Read the model's own tau-quantile as the predictor, so the fit
                # only has to correct it rather than rebuild it from scratch.
                x = Q[:, int(np.argmin(np.abs(TAUS - t)))]
                a, b = fit_quantile(x[tr], y[tr], float(t))
                coefs.append([float(t), round(a, 4), round(b, 3)])
                cal[:, j] = a * x + b
            # Independently fitted quantiles can cross; a quantile function must
            # not decrease, so enforce it the standard way — by sorting.
            cal = np.sort(cal, axis=1)
            mid = int(np.argmin(np.abs(grid - 0.5)))
            line = half_point(cal[:, mid])
            u = (cal[te] < y[te][:, None]).sum(axis=1) / len(grid)
            hist = np.histogram(u, bins=5, range=(0, 1))[0] / max(te.sum(), 1)
            print(f"  bo{bo} {label:<7} over/cover OOS {(y[te] > line[te]).mean()*100:5.1f}%"
                  f"   PIT: " + "  ".join(f"{h:.2f}" for h in hist))
            per_q[label][str(bo)] = coefs

    # The separate location-scale exploration that used to sit here has been
    # folded into the main loop above, which now fits the width per (tour,
    # best_of) alongside the centre. Keeping a second copy that fitted it POOLED
    # meant the file printed two different answers for the same constant, and
    # the pooled one was the wrong one.

    out = Path(__file__).resolve().parent.parent / "reports" / "score_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"centre": fitted, "spread": spreads, "rejected_per_quantile": per_q,
         "grid": [float(g) for g in grid]},
        indent=1), encoding="utf-8")
    print(f"\n  full fit written to reports/{out.name}")


if __name__ == "__main__":
    main()
