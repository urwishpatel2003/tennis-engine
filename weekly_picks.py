"""
Market comparison — where the model disagrees with the price.

    python weekly_picks.py --slate slate.csv --top 10
    python weekly_picks.py --a "Carlos Alcaraz" --b "Jannik Sinner" \
        --surface Clay --odds-a 1.60 --odds-b 2.45

Read this before using it
-------------------------
Tennis match markets are efficient. The honest expectation for a model like this
one is that it produces a WELL-CALIBRATED price, not a beatable one. The same
finding is already on record for the NFL engine in this workspace: out of sample,
the model could not beat the market, and its value was as a standalone
power-ranking and prediction tool.

So `edge` here means "the model and the market disagree by this much". It does not
mean "this is a profitable bet". Before treating any of it as a signal, run
`backtest.py` with market prices attached and check whether the disagreements
actually predict outcomes — on almost every sport, most of them do not.

The vig removal below matters for that: a raw two-way book prices to ~105%, so
comparing a model probability against a raw implied probability manufactures a
phantom edge on the favourite every single time.

Slate CSV format (header required, one row per match):
    player_a,player_b,tour,surface,best_of,date,tournament,odds_a,odds_b
Odds are decimal. `odds_a`/`odds_b` may be blank for a pure model line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.markov import prob_cover_handicap, prob_over_games  # noqa: E402
from engine.predict import Engine  # noqa: E402
from engine.schema import PROCESSED, SURFACES  # noqa: E402


def devig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """
    Strip the bookmaker's margin from a two-way price.

    Proportional (multiplicative) method: divide each raw implied probability by
    the overround. It slightly over-corrects the favourite versus the
    'Shin'/power methods, but it needs no solver and the difference is well inside
    the model's own error.
    """
    if not (odds_a and odds_b) or odds_a <= 1.0 or odds_b <= 1.0:
        return float("nan"), float("nan")
    ia, ib = 1.0 / odds_a, 1.0 / odds_b
    total = ia + ib
    return ia / total, ib / total


def kelly_fraction(p: float, odds: float) -> float:
    """
    Full-Kelly stake as a fraction of bankroll. Negative means no bet.

    Reported for reference only. Full Kelly on a model whose edge is uncertain is
    a good way to lose the bankroll; if these are ever staked, quarter-Kelly is
    the usual discipline.
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p * b - (1.0 - p)) / b)


def evaluate_match(
    eng: Engine, row: dict
) -> dict:
    """Predict one match and, if prices are present, compare against them."""
    odds_a = _num(row.get("odds_a"))
    odds_b = _num(row.get("odds_b"))
    mkt_a, mkt_b = devig(odds_a, odds_b)

    p = eng.predict(
        row["player_a"], row["player_b"],
        tour=str(row.get("tour", "atp")).lower(),
        surface=str(row.get("surface", "Hard")),
        best_of=int(row.get("best_of") or 3),
        match_date=row.get("date"),
        tournament=row.get("tournament"),
        # The market is NOT blended in here. Blending it and then measuring the
        # gap to it would be circular — the model would agree with the price by
        # construction and the edge would shrink toward zero for the wrong reason.
        market_prob_a=None,
    )

    out = {
        "player_a": p["player_a"]["name"],
        "player_b": p["player_b"]["name"],
        "tour": p["tour"], "surface": p["surface"], "best_of": p["best_of"],
        "tournament": row.get("tournament"),
        "model_prob_a": p["win_prob_a"],
        "model_prob_b": p["win_prob_b"],
        "fair_odds_a": p["fair_odds_a"],
        "fair_odds_b": p["fair_odds_b"],
        "expected_total_games": p["expected_total_games"],
        "fair_total_line": p["fair_total_games_line"],
        "fair_handicap_a": p["fair_game_handicap_a"],
        "data_quality": p["data_quality"]["level"],
    }

    if np.isfinite(mkt_a):
        out.update(
            {
                "odds_a": odds_a, "odds_b": odds_b,
                "market_prob_a": mkt_a, "market_prob_b": mkt_b,
                "overround": (1 / odds_a + 1 / odds_b) - 1.0,
                "disagreement_a": p["win_prob_a"] - mkt_a,
                "ev_a": p["win_prob_a"] * odds_a - 1.0,
                "ev_b": p["win_prob_b"] * odds_b - 1.0,
                "kelly_a": kelly_fraction(p["win_prob_a"], odds_a),
                "kelly_b": kelly_fraction(p["win_prob_b"], odds_b),
            }
        )
        out["best_side"] = "A" if out["ev_a"] >= out["ev_b"] else "B"
        out["best_ev"] = max(out["ev_a"], out["ev_b"])
    else:
        out["best_ev"] = np.nan

    # Optional games markets, if lines were supplied.
    tl = _num(row.get("total_line"))
    if np.isfinite(tl):
        out["total_line"] = tl
        out["prob_over"] = prob_over_games(p["_games_joint"], tl)
    hl = _num(row.get("handicap_a"))
    if np.isfinite(hl):
        out["handicap_a_line"] = hl
        out["prob_cover_a"] = prob_cover_handicap(p["_games_joint"], hl)

    return out


def _num(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Model vs market for a slate.")
    ap.add_argument("--slate", type=Path, default=None, help="CSV of matches")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--surface", default="Hard", choices=list(SURFACES))
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--date", default=None)
    ap.add_argument("--tournament", default=None)
    ap.add_argument("--odds-a", type=float, default=None)
    ap.add_argument("--odds-b", type=float, default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-quality", default="low",
                    choices=["low", "medium", "high"],
                    help="drop matches whose inputs were thinner than this")
    ap.add_argument("--out", default=str(PROCESSED / "picks.csv"))
    args = ap.parse_args()

    if args.slate:
        slate = pd.read_csv(args.slate).to_dict("records")
    elif args.a and args.b:
        slate = [{
            "player_a": args.a, "player_b": args.b, "tour": args.tour,
            "surface": args.surface, "best_of": args.best_of, "date": args.date,
            "tournament": args.tournament,
            "odds_a": args.odds_a, "odds_b": args.odds_b,
        }]
    else:
        ap.error("supply either --slate or both --a and --b")

    eng = Engine()
    rows, failed = [], []
    for row in slate:
        try:
            rows.append(evaluate_match(eng, row))
        except KeyError as e:
            failed.append(f"{row.get('player_a')} vs {row.get('player_b')}: {e}")

    if failed:
        print("Could not price these matches:")
        for f in failed:
            print(f"  - {f}")

    if not rows:
        return

    df = pd.DataFrame(rows)
    order = {"low": 0, "medium": 1, "high": 2}
    df = df[df["data_quality"].map(order) >= order[args.min_quality]]

    has_market = "best_ev" in df.columns and df["best_ev"].notna().any()
    if has_market:
        df = df.sort_values("best_ev", ascending=False)

    # 'model', 'market' and 'diff' are all stated for player A. 'EV' is the best of
    # the two sides, so the 'side' column says which player it refers to —
    # without it a +6% EV sitting next to a 58% model number reads as if the model
    # liked A, when the value was on B all along.
    print(f"\n{'match':<44}{'model A':>9}{'market A':>9}{'diff':>7}"
          f"{'side':>6}{'EV':>8}{'qual':>7}")
    print("─" * 90)
    for r in df.head(args.top).itertuples(index=False):
        match = f"{str(r.player_a)[:20]} v {str(r.player_b)[:20]}"
        live = has_market and pd.notna(getattr(r, "market_prob_a", np.nan))
        mkt = f"{r.market_prob_a*100:.1f}%" if live else "   —"
        diff = f"{r.disagreement_a*100:+.1f}" if live else "   —"
        side = getattr(r, "best_side", None) if live else None
        side_name = "—"
        if side:
            side_name = str(r.player_a if side == "A" else r.player_b).split()[-1][:5]
        ev = f"{r.best_ev*100:+.1f}%" if live and pd.notna(r.best_ev) else "   —"
        print(f"{match:<44}{r.model_prob_a*100:>8.1f}%{mkt:>9}{diff:>7}"
              f"{side_name:>6}{ev:>8}{r.data_quality:>7}")

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} matches)")
    if has_market:
        print(
            "\nNOTE: 'diff' is disagreement with the price, not a proven edge.\n"
            "      Tennis match markets are efficient; validate against results\n"
            "      with backtest.py before treating any of this as a signal."
        )


if __name__ == "__main__":
    main()
