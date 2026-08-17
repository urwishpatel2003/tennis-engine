"""
Walk-forward, out-of-sample validation.

    python backtest.py --seasons 2020-2026
    python backtest.py --tours atp --surface Clay --min-matches 30

Why this is honest by construction
----------------------------------
It does NOT re-run the engine per season. It reads the PRE-MATCH columns that
engine/ratings.py and engine/serve_return.py already wrote — values computed from
matches strictly earlier in the chronological pass. There is no window in which a
future result can reach a prediction, because the number being read was frozen
before that match was played.

The one thing a backtest can still get wrong is label leakage through row order:
the raw tables are winner-first, so any model that peeked at column order would
score 100%. Player A is therefore fixed as the LOWER player_id, which is
independent of the result and deterministic across runs.

Reported
--------
* log loss / Brier / accuracy, against three baselines
* a calibration table (predicted vs realised, by decile)
* the optimal Elo-spread multiplier, which says whether the ratings are over- or
  under-confident and by how much
* score-market accuracy: total games and game handicap
* breakdowns by season, surface, tour and data density
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import markov  # noqa: E402
from engine.matchups import handedness_elo_delta  # noqa: E402
from engine.predict import W_ELO, blend_logit  # noqa: E402
from engine.schema import ELO_SCALE, PROCESSED, RAW, base_spw  # noqa: E402
from engine.serve_return import point_probabilities  # noqa: E402

EPS = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p > 0.5) == (y == 1)))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bucket": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": int(m.sum()),
                "predicted": float(p[m].mean()),
                "actual": float(y[m].mean()),
                "gap": float(y[m].mean() - p[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def optimal_elo_multiplier(elo_gap: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Find the scalar `k` minimising log loss for  p = sigmoid(k · gap · ln10/400).

    k = 1.0 means the rating spread is exactly right. k < 1 means the ratings are
    OVER-confident (gaps too wide for the outcomes they produce) and the K-factor
    or the surface blend should come down; k > 1 means under-confident.
    """
    best_k, best_ll = 1.0, float("inf")
    for k in np.arange(0.50, 1.81, 0.01):
        p = 1.0 / (1.0 + np.exp(-k * elo_gap * math.log(10.0) / ELO_SCALE))
        ll = log_loss(p, y)
        if ll < best_ll:
            best_k, best_ll = float(k), ll
    return best_k, best_ll


# ──────────────────────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────────────────────
def load_frame(tours: list[str]) -> pd.DataFrame:
    """Join the pre-match tables into one row per match, A = lower player_id."""
    need = {
        "ratings": PROCESSED / "ratings.parquet",
        "serve_return": PROCESSED / "serve_return.parquet",
    }
    for name, path in need.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python run_engine.py --build` first."
            )

    r = pd.read_parquet(need["ratings"])
    sr = pd.read_parquet(need["serve_return"])
    df = r.merge(
        sr.drop(columns=["tour", "tourney_date", "season", "surface",
                         "winner_id", "loser_id"]),
        on="match_id", how="inner",
    )
    df = df[df["tour"].isin(tours)]

    # Match context (best_of, completed) comes from the raw log.
    ctx = []
    for tour in tours:
        p = RAW / f"matches_{tour}.parquet"
        if p.exists():
            m = pd.read_parquet(p)
            ctx.append(m[["match_id", "best_of", "completed", "retirement",
                          "total_games", "games_w", "games_l", "tourney_level"]])
    if ctx:
        df = df.merge(pd.concat(ctx, ignore_index=True), on="match_id", how="left")

    # Player attributes for the handedness term.
    attrs = []
    for tour in tours:
        p = RAW / f"players_{tour}.parquet"
        if p.exists():
            a = pd.read_parquet(p)[["player_id", "hand", "height"]]
            a["tour"] = tour
            attrs.append(a)
    attrs = pd.concat(attrs, ignore_index=True) if attrs else pd.DataFrame(
        columns=["player_id", "hand", "height", "tour"]
    )

    # ── Orient every row so player A is the LOWER id (independent of the result) ─
    a_is_winner = df["winner_id"].to_numpy() < df["loser_id"].to_numpy()

    def pick(wcol: str, lcol: str) -> np.ndarray:
        return np.where(a_is_winner, df[wcol].to_numpy(), df[lcol].to_numpy())

    out = pd.DataFrame(
        {
            "match_id": df["match_id"].to_numpy(),
            "tour": df["tour"].to_numpy(),
            "season": df["season"].to_numpy(),
            "surface": df["surface"].to_numpy(),
            "tourney_level": df.get("tourney_level", pd.Series(index=df.index)).to_numpy(),
            "best_of": pd.to_numeric(df.get("best_of"), errors="coerce")
                        .fillna(3).astype(int).to_numpy(),
            "completed": df.get("completed", pd.Series(True, index=df.index))
                        .fillna(False).to_numpy(),
            "a_id": pick("winner_id", "loser_id").astype("int64"),
            "b_id": pick("loser_id", "winner_id").astype("int64"),
            "a_elo": pick("w_elo", "l_elo"),
            "b_elo": pick("l_elo", "w_elo"),
            "a_elo_surface": pick("w_elo_surface", "l_elo_surface"),
            "b_elo_surface": pick("l_elo_surface", "w_elo_surface"),
            "a_elo_blend": pick("w_elo_blend", "l_elo_blend"),
            "b_elo_blend": pick("l_elo_blend", "w_elo_blend"),
            "a_matches": pick("w_matches", "l_matches"),
            "b_matches": pick("l_matches", "w_matches"),
            "a_serve": pick("w_serve_excess", "l_serve_excess"),
            "b_serve": pick("l_serve_excess", "w_serve_excess"),
            "a_return": pick("w_return_excess", "l_return_excess"),
            "b_return": pick("l_return_excess", "w_return_excess"),
            "a_svpt_seen": pick("w_svpt_seen", "l_svpt_seen"),
            "b_svpt_seen": pick("l_svpt_seen", "w_svpt_seen"),
            "games_a": pick("games_w", "games_l"),
            "games_b": pick("games_l", "games_w"),
            "total_games": df.get("total_games", pd.Series(np.nan, index=df.index)).to_numpy(),
            "y": a_is_winner.astype(int),
        }
    )

    if not attrs.empty:
        for side in ("a", "b"):
            out = out.merge(
                attrs.rename(columns={"player_id": f"{side}_id", "hand": f"{side}_hand",
                                      "height": f"{side}_height"}),
                on=[f"{side}_id", "tour"], how="left",
            )
    else:
        out["a_hand"] = out["b_hand"] = None

    return out


def score_frame(df: pd.DataFrame, with_scores: bool = True) -> pd.DataFrame:
    """Attach every model's probability to each row."""
    # ── Elo view ──────────────────────────────────────────────────────────────
    gap = df["a_elo_blend"].to_numpy() - df["b_elo_blend"].to_numpy()
    hand = np.array(
        [handedness_elo_delta(a, b) for a, b in zip(df["a_hand"], df["b_hand"])]
    )
    gap_adj = gap + hand
    df["p_elo"] = 1.0 / (1.0 + np.exp(-gap_adj * math.log(10.0) / ELO_SCALE))
    df["p_elo_overall"] = 1.0 / (
        1.0 + np.exp(-(df["a_elo"] - df["b_elo"]).to_numpy() * math.log(10.0) / ELO_SCALE)
    )
    df["elo_gap_adj"] = gap_adj

    # ── Serve/return view ─────────────────────────────────────────────────────
    pa = np.empty(len(df))
    pb = np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        pa[i], pb[i] = point_probabilities(
            r.a_serve, r.a_return, r.b_serve, r.b_return, r.tour, r.surface
        )
    df["pa_point"] = pa
    df["pb_point"] = pb
    df["p_sim"] = [
        markov.match_win_prob(x, y, int(bo))
        for x, y, bo in zip(pa, pb, df["best_of"].to_numpy())
    ]

    # ── Blend ─────────────────────────────────────────────────────────────────
    df["p_model"] = [
        blend_logit([e, s], [W_ELO, 1.0 - W_ELO])
        for e, s in zip(df["p_elo"].to_numpy(), df["p_sim"].to_numpy())
    ]

    # ── Score markets (expensive; opt out for a quick run) ────────────────────
    if with_scores:
        exp_total = np.full(len(df), np.nan)
        exp_margin = np.full(len(df), np.nan)
        for i, (x, y, bo, target) in enumerate(
            zip(pa, pb, df["best_of"].to_numpy(), df["p_model"].to_numpy())
        ):
            xa, xb = markov.invert_to_target(x, y, target, int(bo))
            _, exp_total[i], exp_margin[i] = markov.match_summary(xa, xb, int(bo))
        df["exp_total_games"] = exp_total
        df["exp_game_margin"] = exp_margin

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def report(df: pd.DataFrame, label: str = "OVERALL") -> dict:
    y = df["y"].to_numpy()
    models = {
        "Elo (overall only)": df["p_elo_overall"].to_numpy(),
        "Elo (surface blend)": df["p_elo"].to_numpy(),
        "Serve/return sim": df["p_sim"].to_numpy(),
        "BLENDED MODEL": df["p_model"].to_numpy(),
    }

    print(f"\n{'─'*78}")
    print(f"  {label}   n = {len(df):,}")
    print(f"{'─'*78}")
    print(f"  {'model':<24}{'logloss':>10}{'brier':>10}{'accuracy':>11}")
    print(f"  {'coin flip':<24}{log_loss(np.full(len(y),0.5),y):>10.4f}"
          f"{brier(np.full(len(y),0.5),y):>10.4f}{0.5:>11.4f}")
    best = None
    for name, p in models.items():
        ll = log_loss(p, y)
        print(f"  {name:<24}{ll:>10.4f}{brier(p,y):>10.4f}{accuracy(p,y):>11.4f}")
        if best is None or ll < best[1]:
            best = (name, ll)

    p = df["p_model"].to_numpy()
    k, ll_k = optimal_elo_multiplier(df["elo_gap_adj"].to_numpy(), y)
    verdict = ("well scaled" if 0.95 <= k <= 1.05
               else ("OVER-confident — reduce K" if k < 0.95
                     else "UNDER-confident — raise K"))
    print(f"\n  optimal Elo spread multiplier: {k:.2f}  ({verdict})")

    print("\n  calibration")
    ct = calibration_table(p, y)
    for r in ct.itertuples(index=False):
        bar = "+" if r.gap > 0 else "-"
        print(f"    {r.bucket:<10} n={r.n:>6}  pred {r.predicted:.3f}"
              f"  actual {r.actual:.3f}   {bar}{abs(r.gap):.3f}")

    out = {
        "label": label, "n": len(df),
        "logloss": log_loss(p, y), "brier": brier(p, y), "accuracy": accuracy(p, y),
        "elo_multiplier": k, "best_model": best[0],
    }

    if "exp_total_games" in df.columns and df["exp_total_games"].notna().any():
        m = df["completed"].astype(bool) & df["total_games"].notna()
        if m.sum() > 50:
            err_t = df.loc[m, "exp_total_games"] - df.loc[m, "total_games"]
            margin_actual = df.loc[m, "games_a"] - df.loc[m, "games_b"]
            err_m = df.loc[m, "exp_game_margin"] - margin_actual
            print(f"\n  score markets (n={m.sum():,})")
            print(f"    total games   MAE {err_t.abs().mean():.2f}   "
                  f"bias {err_t.mean():+.2f}   (mean actual {df.loc[m,'total_games'].mean():.1f})")
            print(f"    game margin   MAE {err_m.abs().mean():.2f}   "
                  f"bias {err_m.mean():+.2f}")
            out["total_games_mae"] = float(err_t.abs().mean())
            out["total_games_bias"] = float(err_t.mean())
            out["game_margin_mae"] = float(err_m.abs().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward OOS backtest.")
    ap.add_argument("--seasons", default=None, help="e.g. 2020-2026")
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--min-matches", type=int, default=20,
                    help="require both players to have this many prior matches "
                         "(the burn-in period is not a fair test of the model)")
    ap.add_argument("--surface", default=None)
    ap.add_argument("--no-scores", action="store_true",
                    help="skip the games/handicap evaluation (much faster)")
    ap.add_argument("--by", nargs="*", default=["season", "surface", "tour"],
                    help="breakdown dimensions")
    ap.add_argument("--out", default=str(PROCESSED / "backtest_results.csv"))
    args = ap.parse_args()

    df = load_frame(args.tours)

    if args.seasons:
        lo, hi = (args.seasons.split("-") + [args.seasons])[:2]
        df = df[df["season"].between(int(lo), int(hi))]
    if args.surface:
        df = df[df["surface"] == args.surface]

    n_all = len(df)
    df = df[(df["a_matches"] >= args.min_matches) & (df["b_matches"] >= args.min_matches)]
    df = df[df["completed"].astype(bool)]
    print(f"Evaluating {len(df):,} of {n_all:,} matches "
          f"(both players ≥{args.min_matches} prior matches, completed only)")

    if df.empty:
        print("Nothing to evaluate.")
        return

    df = score_frame(df, with_scores=not args.no_scores)

    rows = [report(df, "OVERALL")]
    for dim in args.by:
        if dim not in df.columns:
            continue
        for val, sub in df.groupby(dim):
            if len(sub) < 200:
                continue
            rows.append(report(sub, f"{dim} = {val}"))

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
