"""
The only script that touches the network.

Downloads Jeff Sackmann's tennis_atp / tennis_wta CSV archives, normalises them
onto the engine's canonical schema, and writes parquet into data/raw/.

    python fetch_data.py --seasons 2010-2026
    python fetch_data.py --tours atp --seasons 2024 2025 2026 --force

Outputs (per tour):
    data/raw/matches_{tour}.parquet    canonical match log (see engine/schema.py)
    data/raw/players_{tour}.parquet    player_id, name, hand, dob, ioc, height
    data/raw/rankings_{tour}.parquet   ranking_date, rank, player_id, points

Mirrors
-------
Corporate networks frequently block raw.githubusercontent.com and api.github.com
while allowing github.com itself, so every file is tried against several mirrors
in turn before giving up. If all of them fail, the error message tells you how to
side-load a local clone with --from-clone.

Dependencies live in requirements-fetch.txt, NOT requirements.txt — nothing in the
engine or the dashboard imports this module, and the deployed server must never
need network access.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.schema import RAW, TOURS, normalise_matches  # noqa: E402

REPO = {"atp": "tennis_atp", "wta": "tennis_wta"}

# Where the data actually lives, tried in order. The first source that returns
# real CSV is pinned for the rest of the run (see _preferred_mirror).
#
# IMPORTANT — the upstream repos are GONE. `JeffSackmann/tennis_atp` and
# `JeffSackmann/tennis_wta` returned 404 as of 2026-08-17; the account's only
# remaining public repo is tennis_MatchChartingProject. This was originally
# misdiagnosed here as a corporate proxy blocking raw.githubusercontent.com,
# which cost a lot of time — the 404s were simply correct. Verify with
# `gh api repos/JeffSackmann/tennis_atp` before believing any network theory.
#
# The originals are kept first anyway, at no cost: a 404 is not retried, so if
# Jeff ever restores them they are used again automatically.
#
# Templates take {repo} (tennis_atp/tennis_wta), {tour} (atp/wta) and {f} (the
# bare filename, e.g. atp_matches_2024.csv). Note the archive mirror nests files
# under an atp/ or wta/ directory and uses `main`, not `master`.
MIRRORS = (
    "https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{f}",
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/{tour}/{f}",
    "https://cdn.jsdelivr.net/gh/Aneeshers/tennis-sackmann-archive@main/{tour}/{f}",
    "https://media.githubusercontent.com/media/Aneeshers/tennis-sackmann-archive/main/{tour}/{f}",
    "https://cdn.statically.io/gh/Aneeshers/tennis-sackmann-archive/main/{tour}/{f}",
)

HEADERS = {"User-Agent": "tennis-engine/0.1 (+local research)", "Accept": "*/*"}

# Sackmann's players/rankings files historically shipped without a header row.
# These are the positional column names for that case.
PLAYER_COLS = ["player_id", "name_first", "name_last", "hand", "dob", "ioc",
               "height", "wikidata_id"]
RANKING_COLS = ["ranking_date", "rank", "player_id", "points"]


# Per-request budget. These used to be 120s with 2 retries, which made the WORST
# case 120 × 3 attempts × 5 mirrors = 30 minutes for a single file — and a hosted
# build duly hung and was killed with no output to show for it. A Sackmann CSV is
# ~1 MB; anything that has not answered in 45s is not going to.
HTTP_TIMEOUT = 45
HTTP_RETRIES = 1

# Once one mirror has served a file, every later file tries it FIRST. Without this
# each of the ~50 downloads re-walks the mirror list from the top and pays the full
# failure cost of every dead mirror ahead of the working one, every single time.
_preferred_mirror: str | None = None


def _get(url: str, timeout: int = HTTP_TIMEOUT, retries: int = HTTP_RETRIES) -> bytes | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # A 404 is a definitive answer (that season/file does not exist) — do
            # not retry it, and do not try to be clever about it.
            if e.code == 429 and attempt < retries:
                time.sleep(4 * (attempt + 1))
                continue
            return None
        except Exception:
            if attempt < retries:
                time.sleep(2)
                continue
            return None
    return None


def _looks_like_csv(data: bytes | None) -> bool:
    """Reject proxy error pages and other HTML masquerading as a 200."""
    if not data or len(data) < 64:
        return False
    head = data.lstrip()[:200].lower()
    return not (head.startswith(b"<!doctype") or head.startswith(b"<html"))


def fetch_csv(repo: str, tour: str, filename: str,
              local_clone: Path | None = None) -> bytes | None:
    """Fetch one CSV: local clone first, then the preferred mirror, then the rest."""
    global _preferred_mirror

    if local_clone is not None:
        p = local_clone / filename
        if not p.exists():          # allow --from-clone to point at a parent dir
            p = local_clone / repo / filename
        if p.exists():
            return p.read_bytes()

    order = list(MIRRORS)
    if _preferred_mirror in order:
        order.remove(_preferred_mirror)
        order.insert(0, _preferred_mirror)

    for template in order:
        data = _get(template.format(repo=repo, tour=tour, f=filename))
        if _looks_like_csv(data):
            if _preferred_mirror != template:
                _preferred_mirror = template
                host = template.split("/")[2]
                print(f"    [using mirror: {host}]", flush=True)
            return data
    return None


def _read_csv(data: bytes, expected_first: str, fallback_cols: list[str]) -> pd.DataFrame:
    """Read a CSV that may or may not have a header row."""
    head = data[:200].decode("utf-8", "replace").split("\n")[0]
    has_header = expected_first in head.lower()
    return pd.read_csv(
        io.BytesIO(data),
        header=0 if has_header else None,
        names=None if has_header else fallback_cols,
        low_memory=False,
        encoding_errors="replace",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-artefact fetchers
# ──────────────────────────────────────────────────────────────────────────────
def fetch_matches(tour: str, seasons: list[int], clone: Path | None) -> pd.DataFrame:
    repo = REPO[tour]
    frames, missing = [], []
    for yr in seasons:
        data = fetch_csv(repo, tour, f"{tour}_matches_{yr}.csv", clone)
        if data is None:
            missing.append(yr)
            continue
        df = pd.read_csv(io.BytesIO(data), low_memory=False, encoding_errors="replace")
        frames.append(df)
        print(f"    {tour} {yr}: {len(df):,} matches")

    if missing:
        print(f"    (no file for: {', '.join(map(str, missing))})")
    if not frames:
        return pd.DataFrame()
    return normalise_matches(pd.concat(frames, ignore_index=True), tour)


def fetch_players(tour: str, clone: Path | None) -> pd.DataFrame:
    data = fetch_csv(REPO[tour], tour, f"{tour}_players.csv", clone)
    if data is None:
        return pd.DataFrame()
    df = _read_csv(data, "player_id", PLAYER_COLS)

    for c in PLAYER_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    out = pd.DataFrame(
        {
            "player_id": pd.to_numeric(df["player_id"], errors="coerce"),
            "name": (
                df["name_first"].fillna("").astype(str).str.strip()
                + " "
                + df["name_last"].fillna("").astype(str).str.strip()
            ).str.strip(),
            "name_first": df["name_first"],
            "name_last": df["name_last"],
            "hand": df["hand"].astype("string").str.strip().str.upper().str[:1],
            "dob": pd.to_datetime(
                df["dob"].astype("string").str.slice(0, 8), format="%Y%m%d", errors="coerce"
            ),
            "ioc": df["ioc"].astype("string").str.strip().str.upper(),
            "height": pd.to_numeric(df["height"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype("int64")
    # Heights below 140cm or above 220cm are data errors, not people.
    out.loc[~out["height"].between(140, 220), "height"] = pd.NA
    return out.drop_duplicates(subset=["player_id"]).reset_index(drop=True)


def fetch_rankings(tour: str, seasons: list[int], clone: Path | None) -> pd.DataFrame:
    """
    Rankings ship per decade, plus a 'current' file.

    These are BY FAR the heaviest artefacts in the archive — each decade file is
    millions of rows — and nothing in the engine currently reads
    `rankings_{tour}.parquet`: the model ranks players itself (see rankings.py),
    and official ranking points are a 52-week tally that would only duplicate what
    Elo already measures. They are fetched for completeness and for future use, so
    `--skip-rankings` exists to drop them from constrained environments like the
    Railway build, where the download and the concat are the main memory risk.
    """
    decades = sorted({(y // 10) * 10 for y in seasons})
    names = [f"{tour}_rankings_{str(d)[2:]}s.csv" for d in decades]
    names.append(f"{tour}_rankings_current.csv")

    frames = []
    for name in names:
        data = fetch_csv(REPO[tour], tour, name, clone)
        if data is None:
            continue
        df = _read_csv(data, "ranking_date", RANKING_COLS)
        frames.append(df)
        print(f"    {name}: {len(df):,} rows")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "player" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"player": "player_id"})
    df["ranking_date"] = pd.to_datetime(
        df["ranking_date"].astype("string").str.slice(0, 8), format="%Y%m%d", errors="coerce"
    )
    for c in ("rank", "player_id", "points"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["ranking_date", "player_id", "rank"])
    df["player_id"] = df["player_id"].astype("int64")
    df["rank"] = df["rank"].astype("int32")
    df["season"] = df["ranking_date"].dt.year
    df = df[df["season"].isin(seasons)]
    return df.drop_duplicates(subset=["ranking_date", "player_id"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
def parse_seasons(tokens: list[str]) -> list[int]:
    """Accept '2020 2021' and '2010-2026' interchangeably."""
    out: set[int] = set()
    for t in tokens:
        t = str(t).strip()
        if "-" in t:
            lo, hi = t.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(t))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Sackmann tennis data → data/raw/")
    ap.add_argument("--tours", nargs="+", default=list(TOURS), choices=list(TOURS))
    ap.add_argument("--seasons", nargs="+", default=["2000-2026"],
                    help="years or ranges, e.g. 2015 2016 or 2010-2026")
    ap.add_argument("--from-clone", default=None, type=Path,
                    help="path to a local clone of tennis_atp/tennis_wta "
                         "(skips the network entirely)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the parquet already exists")
    ap.add_argument("--skip-rankings", action="store_true",
                    help="skip the official ranking files — the largest download "
                         "in the archive, and nothing in the engine reads them yet")
    args = ap.parse_args()

    seasons = parse_seasons(args.seasons)
    clone = args.from_clone
    print(f"Seasons: {seasons[0]}–{seasons[-1]} ({len(seasons)} years)")
    if clone:
        print(f"Reading from local clone: {clone}")

    any_ok = False
    for tour in args.tours:
        print(f"\n── {tour.upper()} " + "─" * 60)
        mpath = RAW / f"matches_{tour}.parquet"
        if mpath.exists() and not args.force:
            print(f"  {mpath.name} exists — use --force to re-download")
            any_ok = True
            continue

        matches = fetch_matches(tour, seasons, clone)
        if matches.empty:
            print(f"  no match data retrieved for {tour}")
            continue
        matches.to_parquet(mpath, index=False)
        print(f"  → {mpath.name}: {len(matches):,} matches "
              f"({matches['season'].min()}–{matches['season'].max()})")
        any_ok = True

        players = fetch_players(tour, clone)
        if not players.empty:
            players.to_parquet(RAW / f"players_{tour}.parquet", index=False)
            print(f"  → players_{tour}.parquet: {len(players):,} players")

        if args.skip_rankings:
            print("  (skipping rankings — --skip-rankings)")
            continue
        rankings = fetch_rankings(tour, seasons, clone)
        if not rankings.empty:
            rankings.to_parquet(RAW / f"rankings_{tour}.parquet", index=False)
            print(f"  → rankings_{tour}.parquet: {len(rankings):,} rows")

    if not any_ok:
        print(
            "\nEVERY SOURCE FAILED — no match data was retrieved.\n"
            "\nCheck whether the SOURCE still exists before blaming the network.\n"
            "The original JeffSackmann/tennis_atp and tennis_wta repos were taken\n"
            "down, and a deleted repo 404s exactly like a blocked proxy:\n"
            "    gh api repos/JeffSackmann/tennis_atp\n"
            "\nIf the mirrors in MIRRORS have gone too, find a surviving one:\n"
            "    gh search repos tennis_atp\n"
            "or clone any mirror and side-load it:\n"
            "    python fetch_data.py --from-clone /path/to/parent_of_atp_and_wta\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nDone. Next: python run_engine.py --build")


if __name__ == "__main__":
    main()
