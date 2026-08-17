"""
Power rankings — surface-specific leaderboards, form and player profiles.

    python rankings.py --tour atp --surface Clay --top 30
    python rankings.py --tour wta --surface overall --min-matches 25 --csv

Outputs: data/processed/rankings_{tour}_{surface}.csv

These are MODEL rankings, not the official ATP/WTA lists. Official rankings are a
rolling 52-week points tally that rewards showing up and defends points by
calendar accident; these rank by estimated current strength. The two disagree most
for players returning from injury (protected ranking keeps them high, the model
does not) and for young players rising fast (the model sees it first).

Columns
-------
rank, player, elo_surface, elo_overall, elo_blend, serve_excess, return_excess,
hold_pct, break_pct, matches, form_90d, trend, last_played
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.markov import game_prob  # noqa: E402
from engine.ratings import SURFACE_BLEND, blended_elo  # noqa: E402
from engine.schema import PROCESSED, RAW, SURFACES, base_spw  # noqa: E402

FORM_WINDOW_DAYS = 90


def load_tables(tour: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_parquet(PROCESSED / "ratings_current.parquet")
    sr = pd.read_parquet(PROCESSED / "serve_return_current.parquet")
    players = pd.read_parquet(RAW / f"players_{tour}.parquet")
    per_match = pd.read_parquet(PROCESSED / "ratings.parquet")
    return (
        ratings[ratings["tour"] == tour],
        sr[sr["tour"] == tour],
        players,
        per_match[per_match["tour"] == tour],
    )


def recent_form(per_match: pd.DataFrame, days: int = FORM_WINDOW_DAYS) -> pd.DataFrame:
    """
    Win rate over the last `days`, and the Elo change across that window.

    `trend` is the honest "is this player rising" measure: raw recent win rate is
    dominated by draw luck, but the Elo delta is already opponent-adjusted.
    """
    if per_match.empty:
        return pd.DataFrame(columns=["player_id", "form_90d", "trend", "recent_matches"])

    cutoff = per_match["tourney_date"].max() - pd.Timedelta(days=days)
    recent = per_match[per_match["tourney_date"] >= cutoff]

    rows = []
    for pid_col, opp_col, won in (("winner_id", "loser_id", 1), ("loser_id", "winner_id", 0)):
        rows.append(
            pd.DataFrame(
                {
                    "player_id": recent[pid_col],
                    "won": won,
                    "elo_then": recent["w_elo" if won else "l_elo"],
                    "date": recent["tourney_date"],
                }
            )
        )
    long = pd.concat(rows, ignore_index=True).sort_values("date")

    agg = long.groupby("player_id").agg(
        form_90d=("won", "mean"),
        recent_matches=("won", "size"),
        elo_start=("elo_then", "first"),
    ).reset_index()
    return agg


# A power ranking is a claim about who is good NOW, so anyone who has not played
# recently is excluded rather than ranked on a stale rating. One tennis year is
# the natural window: it covers a full surface cycle and tolerates a long injury
# layoff without letting the retired stay on the list forever.
ACTIVE_WITHIN_DAYS = 365


def build_rankings(
    tour: str, surface: str = "overall", min_matches: int = 20,
    active_within_days: int | None = ACTIVE_WITHIN_DAYS,
) -> pd.DataFrame:
    ratings, sr, players, per_match = load_tables(tour)

    overall = ratings[ratings["surface"] == "overall"][
        ["player_id", "elo", "matches", "last_played"]
    ].rename(columns={"elo": "elo_overall", "matches": "matches_overall"})

    if active_within_days is not None and not overall.empty:
        # Measure recency against the end of the ARCHIVE, not today's date — the
        # data may be weeks old, and using wall-clock time would silently empty
        # the rankings as a stale build ages.
        as_of = pd.to_datetime(overall["last_played"]).max()
        cutoff = as_of - pd.Timedelta(days=active_within_days)
        overall = overall[pd.to_datetime(overall["last_played"]) >= cutoff]

    if surface == "overall":
        surf = overall.rename(columns={"elo_overall": "elo_surface",
                                       "matches_overall": "matches_surface"})[
            ["player_id", "elo_surface", "matches_surface"]
        ]
    else:
        surf = ratings[ratings["surface"] == surface][
            ["player_id", "elo", "matches"]
        ].rename(columns={"elo": "elo_surface", "matches": "matches_surface"})

    df = overall.merge(surf, on="player_id", how="left")
    # A player with no history on the surface is rated at their overall level.
    df["elo_surface"] = df["elo_surface"].fillna(df["elo_overall"])
    df["matches_surface"] = df["matches_surface"].fillna(0).astype(int)
    df["elo_blend"] = (
        blended_elo(df["elo_overall"], df["elo_surface"], SURFACE_BLEND)
        if surface != "overall" else df["elo_overall"]
    )

    sr_key = sr[sr["surface"] == surface]
    if sr_key.empty:
        sr_key = sr[sr["surface"] == "overall"]
    df = df.merge(
        sr_key[["player_id", "serve_excess", "return_excess", "svpt_seen"]],
        on="player_id", how="left",
    )
    df[["serve_excess", "return_excess"]] = df[["serve_excess", "return_excess"]].fillna(0.0)

    # Translate the excesses into the units a tennis fan reads: hold and break %.
    ref_surface = "Hard" if surface == "overall" else surface
    base = base_spw(tour, ref_surface)
    df["hold_pct"] = [game_prob(float(np.clip(base + s, 0.35, 0.85)))
                      for s in df["serve_excess"]]
    # Break % = 1 - opponent's hold, where the opponent serves at the baseline
    # minus this player's return strength.
    df["break_pct"] = [1.0 - game_prob(float(np.clip(base - r, 0.35, 0.85)))
                       for r in df["return_excess"]]

    form = recent_form(per_match)
    df = df.merge(form, on="player_id", how="left")
    df["trend"] = df["elo_overall"] - df["elo_start"]
    df["form_90d"] = df["form_90d"].fillna(np.nan)

    df = df.merge(
        players[["player_id", "name", "hand", "height", "ioc"]], on="player_id", how="left"
    )

    df = df[df["matches_overall"] >= min_matches]
    sort_col = "elo_blend" if surface != "overall" else "elo_overall"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["surface"] = surface
    df["tour"] = tour
    return df


def print_table(df: pd.DataFrame, top: int, surface: str) -> None:
    print(f"\n{'rank':>4}  {'player':<26}{'elo':>7}{'surf':>7}{'hold%':>7}"
          f"{'brk%':>7}{'form':>7}{'trend':>8}{'M':>6}")
    print("─" * 82)
    for r in df.head(top).itertuples(index=False):
        form = f"{r.form_90d*100:.0f}%" if pd.notna(r.form_90d) else "  —"
        trend = f"{r.trend:+.0f}" if pd.notna(r.trend) else "   —"
        name = (str(r.name)[:25]) if pd.notna(r.name) else str(r.player_id)
        print(f"{r.rank:>4}  {name:<26}{r.elo_overall:>7.0f}{r.elo_surface:>7.0f}"
              f"{r.hold_pct*100:>7.1f}{r.break_pct*100:>7.1f}{form:>7}{trend:>8}"
              f"{r.matches_overall:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Model power rankings.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--surface", default="overall",
                    choices=["overall", *SURFACES])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--active-within-days", type=int, default=ACTIVE_WITHIN_DAYS,
                    help="exclude players who have not played in this many "
                         "days; 0 disables the filter (shows all-time ratings)")
    ap.add_argument("--all-surfaces", action="store_true",
                    help="write a CSV for every surface")
    ap.add_argument("--csv", action="store_true", help="also write the CSV")
    args = ap.parse_args()

    surfaces = ["overall", *SURFACES] if args.all_surfaces else [args.surface]
    for s in surfaces:
        df = build_rankings(args.tour, s, args.min_matches,
                            args.active_within_days or None)
        if not args.all_surfaces:
            print(f"\n{args.tour.upper()} power rankings — {s}"
                  f"  (min {args.min_matches} matches, n={len(df)})")
            print_table(df, args.top, s)
        if args.csv or args.all_surfaces:
            out = PROCESSED / f"rankings_{args.tour}_{s.lower()}.csv"
            df.to_csv(out, index=False)
            print(f"  → {out.name} ({len(df)} players)")


if __name__ == "__main__":
    main()
