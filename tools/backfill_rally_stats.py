"""
Pull winners / unforced errors / net points onto the existing ATP archive.

    python tools/backfill_rally_stats.py --from 2021
    python tools/backfill_rally_stats.py --from 2021 --dry-run

Source: msolonskyi/ManTennisData, the same atptour.com scrape engine/refresh.py
already uses for results.

Read this before trusting the output
------------------------------------
These statistics are FAR patchier than serve lines, and the gap is structural,
not a fetching problem:

    year   winners/UE coverage
    ≤2020        0%
    2021         2%
    2024        17%
    2025        34%
    2026        58%

They are scored by human statisticians at the venue, so they exist only where the
ATP posted a full stats panel — mostly the bigger events. `forced_errors` is
present as a column upstream but is 0% populated in every year checked, so it is
not ingested. The WTA feed carries none of this at all.

Consequence: any player profile built on these covers a subset of recent ATP
matches, and that has to be stated wherever the numbers are shown rather than
letting a half-populated average look like a full one.

Matching
--------
Historical rows came from Sackmann and carry no upstream id, so rows are matched
on (season, normalised winner name, normalised loser name). A pair meeting twice
in one season is possible, so ties are broken by nearest tournament date, and any
remaining ambiguity is skipped rather than guessed.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.refresh import SOURCE, _norm  # noqa: E402
from engine.schema import RALLY_STAT_COLS, RAW  # noqa: E402

# upstream column -> our suffix, per side ("win"/"los" is their prefix convention)
FIELD_MAP = {
    "winners": "winners",
    "unforced_errors": "unforced",
    "net_points_won": "netWon",
    "net_points_total": "netTotal",
    "total_points_total": "tp",
}


def _num(v):
    if v in (None, "", "NA"):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def _get_csv(name: str, timeout: int = 240) -> list[dict]:
    req = urllib.request.Request(SOURCE + name, headers={"User-Agent": "tennis-engine/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8", "replace"))))


def build_index(years: list[int], verbose: bool = True) -> dict:
    """(season, norm winner, norm loser) -> list of upstream rows with rally stats."""
    idx: dict = {}
    for yr in years:
        try:
            rows = _get_csv(f"matches_{yr}.csv")
        except Exception as e:
            if verbose:
                print(f"  {yr}: unavailable ({str(e)[:40]})")
            continue
        kept = 0
        for r in rows:
            # Only rows that actually carry the statistics are worth indexing.
            if _num(r.get("win_winners")) != _num(r.get("win_winners")):   # NaN check
                continue
            key = (yr, _norm(r.get("winner_name")), _norm(r.get("loser_name")))
            idx.setdefault(key, []).append(r)
            kept += 1
        if verbose:
            print(f"  {yr}: {kept:,} of {len(rows):,} rows carry rally stats")
    return idx


def backfill(from_year: int = 2021, dry_run: bool = False, verbose: bool = True) -> dict:
    path = RAW / "matches_atp.parquet"
    m = pd.read_parquet(path)
    seasons = sorted(s for s in m["season"].dropna().unique().astype(int) if s >= from_year)
    if verbose:
        print(f"Indexing ManTennisData {seasons[0]}–{seasons[-1]} …")
    idx = build_index(seasons, verbose=verbose)
    if not idx:
        return {"filled": 0, "reason": "no upstream rows with rally stats"}

    for side in ("w", "l"):
        for col in RALLY_STAT_COLS:
            c = f"{side}_{col}"
            if c not in m.columns:
                m[c] = np.nan

    cand = m[m["season"].isin(seasons)]
    filled = ambiguous = 0
    updates: dict[int, dict] = {}

    for r in cand.itertuples():
        key = (int(r.season), _norm(r.winner_name), _norm(r.loser_name))
        hits = idx.get(key)
        if not hits:
            continue
        if len(hits) > 1:
            # Same pair, same season, more than once — pick the nearest tournament
            # date rather than assuming the first is right.
            def gap(h):
                d = pd.to_datetime(str(h.get("tournament_id", ""))[:4] + "-01-01",
                                   errors="coerce")
                return abs((r.tourney_date - d).days) if pd.notna(d) else 10**6
            hits = sorted(hits, key=gap)
            if len(hits) > 2:
                ambiguous += 1
                continue
        h = hits[0]
        vals = {}
        for up, ours in FIELD_MAP.items():
            vals[f"w_{ours}"] = _num(h.get(f"win_{up}"))
            vals[f"l_{ours}"] = _num(h.get(f"los_{up}"))
        if np.isfinite(vals.get("w_winners", np.nan)):
            updates[r.Index] = vals
            filled += 1

    if verbose:
        print(f"\nmatched {filled:,} archive rows"
              + (f"  ({ambiguous} skipped as ambiguous)" if ambiguous else ""))
    if dry_run:
        return {"filled": filled, "ambiguous": ambiguous, "written": False}

    for i, vals in updates.items():
        for c, v in vals.items():
            m.at[i, c] = v
    m.to_parquet(path, index=False)

    got = m[m["season"] >= from_year]
    cov = float(got["w_winners"].notna().mean()) if len(got) else 0.0
    if verbose:
        print(f"wrote {path.name}; rally coverage {from_year}+: {cov*100:.1f}%")
    return {"filled": filled, "ambiguous": ambiguous, "written": True,
            "coverage_from_year": round(cov, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill winners/UE/net points (ATP).")
    ap.add_argument("--from", dest="from_year", type=int, default=2021,
                    help="earliest season to attempt (nothing exists before 2021)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    r = backfill(args.from_year, dry_run=args.dry_run)
    print(r)


if __name__ == "__main__":
    main()
