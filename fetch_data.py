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

# Tried in order. The first that returns a 200 wins.
MIRRORS = (
    "https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{f}",
    "https://media.githubusercontent.com/media/JeffSackmann/{repo}/master/{f}",
    "https://cdn.jsdelivr.net/gh/JeffSackmann/{repo}@master/{f}",
    "https://cdn.statically.io/gh/JeffSackmann/{repo}/master/{f}",
    "https://gitcdn.link/cdn/JeffSackmann/{repo}/master/{f}",
)

HEADERS = {"User-Agent": "tennis-engine/0.1 (+local research)", "Accept": "*/*"}

# Sackmann's players/rankings files historically shipped without a header row.
# These are the positional column names for that case.
PLAYER_COLS = ["player_id", "name_first", "name_last", "hand", "dob", "ioc",
               "height", "wikidata_id"]
RANKING_COLS = ["ranking_date", "rank", "player_id", "points"]


def _get(url: str, timeout: int = 120, retries: int = 2) -> bytes | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
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


def fetch_csv(repo: str, filename: str, local_clone: Path | None = None) -> bytes | None:
    """Fetch one CSV, trying a local clone first, then every mirror."""
    if local_clone is not None:
        p = local_clone / filename
        if not p.exists():          # allow --from-clone to point at a parent dir
            p = local_clone / repo / filename
        if p.exists():
            return p.read_bytes()

    for template in MIRRORS:
        data = _get(template.format(repo=repo, f=filename))
        if data and len(data) > 64 and not data.lstrip().startswith(b"<!DOCTYPE"):
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
        data = fetch_csv(repo, f"{tour}_matches_{yr}.csv", clone)
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
    data = fetch_csv(REPO[tour], f"{tour}_players.csv", clone)
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
    """Rankings ship per decade, plus a 'current' file."""
    decades = sorted({(y // 10) * 10 for y in seasons})
    names = [f"{tour}_rankings_{str(d)[2:]}s.csv" for d in decades]
    names.append(f"{tour}_rankings_current.csv")

    frames = []
    for name in names:
        data = fetch_csv(REPO[tour], name, clone)
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

        rankings = fetch_rankings(tour, seasons, clone)
        if not rankings.empty:
            rankings.to_parquet(RAW / f"rankings_{tour}.parquet", index=False)
            print(f"  → rankings_{tour}.parquet: {len(rankings):,} rows")

    if not any_ok:
        print(
            "\nEVERY MIRROR FAILED.\n"
            "This is usually a corporate proxy blocking raw.githubusercontent.com.\n"
            "Work around it by cloning the repos on any unrestricted machine:\n"
            "    git clone --depth 1 https://github.com/JeffSackmann/tennis_atp\n"
            "    git clone --depth 1 https://github.com/JeffSackmann/tennis_wta\n"
            "then point the fetcher at them:\n"
            "    python fetch_data.py --from-clone /path/to/parent_of_both_repos\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nDone. Next: python run_engine.py --build")


if __name__ == "__main__":
    main()
