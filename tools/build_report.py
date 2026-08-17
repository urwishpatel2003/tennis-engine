"""
Write a build manifest describing the data the image was built with.

    python tools/build_report.py

Runs as the last step of the Railway build (see nixpacks.toml) and writes
data/processed/build_info.json. The dashboard reads it for `/api/meta` and
`/api/health`, so a deployed instance can always answer "what data am I serving,
and when did it come from?" without anyone having to guess from the UI.

It also fails loudly if the build produced nothing usable — a silent empty deploy
is far worse than a failed one, because the dashboard would come up looking fine
and serve empty tables.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import PROCESSED, RAW, TOURS  # noqa: E402

REQUIRED = [
    PROCESSED / "ratings_current.parquet",
    PROCESSED / "serve_return_current.parquet",
    PROCESSED / "h2h.parquet",
    PROCESSED / "conditions.parquet",
]


def main() -> None:
    synthetic = (RAW / "SYNTHETIC.marker").exists()

    info = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": synthetic,
        "source": "synthetic generator" if synthetic
                  else "Jeff Sackmann tennis_atp / tennis_wta",
        "tours": {},
    }

    total_matches = 0
    for tour in TOURS:
        p = RAW / f"matches_{tour}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p, columns=["season", "tourney_date", "winner_id", "loser_id"])
        players = pd.concat([m["winner_id"], m["loser_id"]]).nunique()
        total_matches += len(m)
        info["tours"][tour] = {
            "matches": int(len(m)),
            "players": int(players),
            "seasons": [int(m["season"].min()), int(m["season"].max())],
            "latest_match": m["tourney_date"].max().date().isoformat(),
        }

    missing = [p.name for p in REQUIRED if not p.exists()]
    info["missing_tables"] = missing
    info["ok"] = total_matches > 0 and not missing

    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(json.dumps(info, indent=2))

    if not info["ok"]:
        print(
            "\nBUILD PRODUCED NO USABLE DATA.\n"
            f"  matches fetched: {total_matches:,}\n"
            f"  missing tables : {', '.join(missing) or 'none'}\n"
            "Failing the build rather than shipping an empty dashboard.\n"
            "If the mirrors are blocked, set FETCH_SEASONS to a narrower range or\n"
            "check https://github.com/JeffSackmann/tennis_atp is reachable.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nBuild OK — {total_matches:,} matches across "
          f"{len(info['tours'])} tour(s)"
          + ("  [SYNTHETIC DATA]" if synthetic else ""))


if __name__ == "__main__":
    main()
