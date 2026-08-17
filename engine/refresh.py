"""
Keep the archive current — append recent results and rebuild the engine.

    python -m engine.refresh --status
    python -m engine.refresh                 # append anything newer, then rebuild
    python -m engine.refresh --no-rebuild    # fetch only

Why a second source
-------------------
The Sackmann mirror this project is built on stopped updating (last match
2026-05-25). `msolonskyi/ManTennisData` scrapes atptour.com and is current to
within a few days, with FULLER statistics than Sackmann — aces, double faults,
first/second serve points won and attempted, break points saved and faced — which
is exactly what the serve/return model needs. So it is used to extend the archive
forward.

    ATP  → refreshed from ManTennisData
    WTA  → NO CURRENT SOURCE EXISTS. It stays at the Sackmann cutoff.

That asymmetry is deliberate and surfaced everywhere rather than hidden: a WTA
prediction made today is running on months-old ratings and the dashboard says so.

Design rules
------------
* **Append only, never overwrite.** Only matches strictly newer than the existing
  archive's last date are taken. The historical base stays exactly as Sackmann
  wrote it — the same data every backtest number in the README was measured on —
  so a refresh can never silently restate history.
* **Qualifying is excluded** (Q1/Q2/Q3), matching what the Sackmann main-tour
  files contain. Mixing quallies in would change what a "tour match" means
  mid-archive.
* **Players are matched by name**, since the two sources use different id spaces.
  Anyone genuinely new gets a fresh synthetic id so their rating chain still
  builds rather than their matches being dropped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import PROCESSED, RAW, normalise_matches

SOURCE = "https://raw.githubusercontent.com/msolonskyi/ManTennisData/master/atp/"

# Their round codes → ours. Qualifying rounds are dropped entirely.
ROUND_MAP = {"R128": "R128", "R64": "R64", "R32": "R32", "R16": "R16",
             "QF": "QF", "SF": "SF", "F": "F", "RR": "RR", "BR": "BR"}
QUALIFYING = {"Q1", "Q2", "Q3", "Q4", "QUAL"}

# Their series codes → Sackmann tourney_level.
LEVEL_MAP = {"gs": "G", "1000": "M", "atp": "A", "ch": "C", "fu": "S", "dc": "D"}

# Synthetic ids for players absent from the Sackmann player table start here.
# Sackmann ids are well below this, so there is no collision risk.
SYNTHETIC_ID_BASE = 900_000


def _norm(s: object) -> str:
    """Same normalisation the live-odds name matcher uses."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ").replace(".", " ").replace("'", "")
    return re.sub(r"[^a-z ]", "", s).strip()


def _get_csv(name: str, timeout: int = 180) -> list[dict]:
    req = urllib.request.Request(SOURCE + name, headers={"User-Agent": "tennis-engine/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


# ──────────────────────────────────────────────────────────────────────────────
# Score conversion
# ──────────────────────────────────────────────────────────────────────────────
def convert_score(raw: object) -> str:
    """
    '64 67(2) 76(4)' → '6-4 6-7(2) 7-6(4)'.

    Their scores are written without a separator, so a set is read as two digits
    unless the games ran past nine — '108' is 10-8, not 1-08. The digits are split
    by trying the longer left-hand reading first and keeping whichever produces a
    legal set.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    out = []
    for tok in raw.split():
        tb = ""
        m = re.match(r"^(\d+)\((\d+)\)$", tok)
        if m:
            tok, tb = m.group(1), f"({m.group(2)})"
        if not tok.isdigit():
            out.append(tok)          # RET, W/O, DEF and friends pass through
            continue
        if len(tok) == 2:
            a, b = tok[0], tok[1]
        elif len(tok) == 3:
            # 10-8 style. Prefer the two-digit winner reading when it is legal.
            a, b = (tok[:2], tok[2]) if int(tok[:2]) >= 10 else (tok[0], tok[1:])
        elif len(tok) == 4:
            a, b = tok[:2], tok[2:]
        else:
            continue
        out.append(f"{a}-{b}{tb}")
    return " ".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# Player identity
# ──────────────────────────────────────────────────────────────────────────────
class PlayerResolver:
    """Maps ManTennisData player codes onto our player_ids, minting new ones."""

    def __init__(self, players: pd.DataFrame) -> None:
        self.players = players.copy()
        self.by_norm: dict[str, int] = {}
        for r in self.players.itertuples(index=False):
            n = _norm(getattr(r, "name", ""))
            if n:
                self.by_norm.setdefault(n, int(r.player_id))
        self._next = max(
            SYNTHETIC_ID_BASE,
            int(self.players["player_id"].max()) + 1 if len(self.players) else SYNTHETIC_ID_BASE,
        )
        self.new_rows: list[dict] = []
        self.minted = 0
        self.matched = 0

    def resolve(self, name: str, meta: dict | None = None) -> int:
        n = _norm(name)
        if n in self.by_norm:
            self.matched += 1
            return self.by_norm[n]
        pid = self._next
        self._next += 1
        self.minted += 1
        self.by_norm[n] = pid
        meta = meta or {}
        self.new_rows.append({
            "player_id": pid,
            "name": name,
            "name_first": name.split(" ")[0] if name else "",
            "name_last": " ".join(name.split(" ")[1:]) if name else "",
            "hand": meta.get("hand"),
            "dob": meta.get("dob"),
            "ioc": meta.get("ioc"),
            "height": meta.get("height"),
        })
        return pid

    def merged_players(self) -> pd.DataFrame:
        if not self.new_rows:
            return self.players
        add = pd.DataFrame(self.new_rows)
        for c in self.players.columns:
            if c not in add.columns:
                add[c] = pd.NA
        return pd.concat([self.players, add[self.players.columns]], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Fetch + map
# ──────────────────────────────────────────────────────────────────────────────
def _f(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "NA"):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def fetch_new_atp(after: pd.Timestamp, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Every ATP main-draw match played after `after`, in canonical raw shape."""
    if verbose:
        print(f"  [refresh] source: ManTennisData (atptour.com scrape)")
        print(f"  [refresh] looking for matches after {after.date()}")

    tours = {r["id"]: r for r in _get_csv("tournaments.csv")}
    src_players = {r["code"]: r for r in _get_csv("players.csv")}

    years = sorted({after.year, after.year + 1, datetime.now(timezone.utc).year})
    rows, skipped_q, skipped_old = [], 0, 0
    for yr in years:
        try:
            matches = _get_csv(f"matches_{yr}.csv")
        except Exception as e:
            if verbose:
                print(f"  [refresh] {yr}: unavailable ({str(e)[:50]})")
            continue
        kept = 0
        for r in matches:
            t = tours.get(r.get("tournament_id"))
            if not t or not t.get("start_dtm"):
                continue
            start = pd.to_datetime(t["start_dtm"][:8], format="%Y%m%d", errors="coerce")
            if pd.isna(start) or start <= after:
                skipped_old += 1
                continue
            rnd = (r.get("stadie_id") or "").upper()
            if rnd in QUALIFYING or rnd not in ROUND_MAP:
                skipped_q += 1
                continue
            rows.append((r, t, start))
            kept += 1
        if verbose and kept:
            print(f"  [refresh] {yr}: {kept:,} new main-draw matches")

    if not rows:
        return pd.DataFrame(), {"new": 0, "skipped_qualifying": skipped_q}

    players = pd.read_parquet(RAW / "players_atp.parquet")
    resolver = PlayerResolver(players)

    def meta_for(code: str) -> dict:
        p = src_players.get(code, {})
        h = p.get("height")
        try:
            h = float(h) if h else None
        except ValueError:
            h = None
        return {"hand": (p.get("handedness") or "")[:1].upper() or None,
                "ioc": p.get("citizenship") or None,
                "height": h,
                "dob": pd.to_datetime(p.get("birth_date"), errors="coerce")}

    out = []
    for r, t, start in rows:
        level = LEVEL_MAP.get((t.get("series_id") or "").lower(), "A")
        best_of = 5 if level == "G" else 3
        wid = resolver.resolve(r["winner_name"], meta_for(r.get("winner_code")))
        lid = resolver.resolve(r["loser_name"], meta_for(r.get("loser_code")))
        score = convert_score(r.get("match_score"))
        if str(r.get("match_ret") or "").strip():
            score = (score + " RET").strip()

        out.append({
            "src_id": r.get("id"),
            "tourney_id": f"{t['id']}",
            "tourney_name": t.get("name"),
            "surface": t.get("surface") or "Hard",
            "draw_size": _f(t, "sgl_draw_qty"),
            "tourney_level": level,
            "tourney_date": int(t["start_dtm"][:8]),
            "match_num": _f(r, "match_order"),
            "winner_id": wid, "winner_name": r["winner_name"],
            "winner_seed": _f(r, "winner_seed"), "winner_entry": np.nan,
            "winner_hand": meta_for(r.get("winner_code"))["hand"],
            "winner_ht": meta_for(r.get("winner_code"))["height"],
            "winner_ioc": r.get("winner_citizenship"),
            "winner_age": _f(r, "winner_age"),
            "loser_id": lid, "loser_name": r["loser_name"],
            "loser_seed": _f(r, "loser_seed"), "loser_entry": np.nan,
            "loser_hand": meta_for(r.get("loser_code"))["hand"],
            "loser_ht": meta_for(r.get("loser_code"))["height"],
            "loser_ioc": r.get("loser_citizenship"),
            "loser_age": _f(r, "loser_age"),
            "score": score,
            "best_of": best_of,
            "round": ROUND_MAP[(r.get("stadie_id") or "").upper()],
            "minutes": _f(r, "match_duration"),
            "w_ace": _f(r, "win_aces"), "w_df": _f(r, "win_double_faults"),
            "w_svpt": _f(r, "win_service_points_total"),
            "w_1stIn": _f(r, "win_first_serves_in"),
            "w_1stWon": _f(r, "win_first_serve_points_won"),
            "w_2ndWon": _f(r, "win_second_serve_points_won"),
            "w_SvGms": _f(r, "win_service_games_played"),
            "w_bpSaved": _f(r, "win_break_points_saved"),
            "w_bpFaced": _f(r, "win_break_points_serve_total"),
            "l_ace": _f(r, "loss_aces"), "l_df": _f(r, "loss_double_faults"),
            "l_svpt": _f(r, "loss_service_points_total"),
            "l_1stIn": _f(r, "loss_first_serves_in"),
            "l_1stWon": _f(r, "loss_first_serve_points_won"),
            "l_2ndWon": _f(r, "loss_second_serve_points_won"),
            "l_SvGms": _f(r, "loss_service_games_played"),
            "l_bpSaved": _f(r, "loss_break_points_saved"),
            "l_bpFaced": _f(r, "loss_break_points_serve_total"),
            "winner_rank": np.nan, "winner_rank_points": np.nan,
            "loser_rank": np.nan, "loser_rank_points": np.nan,
        })

    raw = pd.DataFrame(out)

    # Their `match_order` column is empty for every row, and match_num is what
    # engine.schema builds match_id from. Leaving it null made every match_id NaN,
    # and the de-duplication step then collapsed 2,334 fetched matches into ONE.
    # So number the matches ourselves: stable, deterministic, unique per event.
    from engine.schema import ROUND_ORD
    raw["_r"] = raw["round"].map(ROUND_ORD).fillna(99)
    raw = raw.sort_values(["tourney_id", "_r", "src_id"]).reset_index(drop=True)
    raw["match_num"] = raw.groupby("tourney_id").cumcount() + 1
    raw = raw.drop(columns=["_r"])

    canon = normalise_matches(raw, "atp")
    canon["source"] = "mantennisdata"

    if resolver.new_rows:
        resolver.merged_players().to_parquet(RAW / "players_atp.parquet", index=False)

    stats = {
        "new": int(len(canon)),
        "players_matched": resolver.matched,
        "players_minted": resolver.minted,
        "skipped_qualifying": skipped_q,
        "with_serve_stats": int(canon["w_svpt"].notna().sum()),
        "date_range": [str(canon["tourney_date"].min().date()),
                       str(canon["tourney_date"].max().date())] if len(canon) else None,
    }
    return canon, stats


# ──────────────────────────────────────────────────────────────────────────────
def status() -> dict:
    out = {"tours": {}, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for tour in ("atp", "wta"):
        p = RAW / f"matches_{tour}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p, columns=["tourney_date", "season"])
        last = m["tourney_date"].max()
        out["tours"][tour] = {
            "matches": int(len(m)),
            "last_match": str(last.date()),
            "stale_days": int((pd.Timestamp.today().normalize() - last.normalize()).days),
            "refreshable": tour == "atp",
        }
    return out


def refresh(rebuild: bool = True, verbose: bool = True) -> dict:
    """Append new ATP results and rebuild. WTA has no current source."""
    path = RAW / f"matches_atp.parquet"
    if not path.exists():
        raise FileNotFoundError("No ATP archive to extend — run fetch_data.py first.")

    existing = pd.read_parquet(path)
    last = existing["tourney_date"].max()
    new, stats = fetch_new_atp(last, verbose=verbose)

    result = {"before": {"matches": int(len(existing)), "last_match": str(last.date())},
              "fetched": stats, "rebuilt": False}

    if new.empty:
        if verbose:
            print("  [refresh] already up to date")
        result["after"] = result["before"]
        return result

    for c in existing.columns:
        if c not in new.columns:
            new[c] = pd.NA
    if "source" not in existing.columns:
        existing["source"] = "sackmann"
    bad = int(new["match_id"].isna().sum())
    dupes = int(new["match_id"].duplicated().sum())
    if bad or dupes:
        raise ValueError(
            f"refusing to merge: {bad} null and {dupes} duplicate match_ids in the "
            f"{len(new)} fetched rows. That silently collapses matches on merge — "
            f"check match_num assignment before retrying."
        )

    merged = pd.concat([existing, new[existing.columns]], ignore_index=True)
    before_dedupe = len(merged)
    merged = merged.drop_duplicates(subset=["match_id"], keep="first")
    dropped = before_dedupe - len(merged)
    if dropped:
        print(f"  [refresh] {dropped} row(s) already present, skipped")
    merged = merged.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    merged.to_parquet(path, index=False)

    result["after"] = {"matches": int(len(merged)),
                       "last_match": str(merged["tourney_date"].max().date())}
    if verbose:
        print(f"  [refresh] {result['before']['matches']:,} → {result['after']['matches']:,} "
              f"matches; now current to {result['after']['last_match']}")

    if rebuild:
        from engine import conditions, matchups, ratings, serve_return
        if verbose:
            print("  [refresh] rebuilding engine …")
        pm, cur = ratings.build_all(("atp", "wta"))
        pm.to_parquet(PROCESSED / "ratings.parquet", index=False)
        cur.to_parquet(PROCESSED / "ratings_current.parquet", index=False)
        pm, cur = serve_return.build_all(("atp", "wta"))
        pm.to_parquet(PROCESSED / "serve_return.parquet", index=False)
        cur.to_parquet(PROCESSED / "serve_return_current.parquet", index=False)
        conditions.build_all(("atp", "wta")).to_parquet(
            PROCESSED / "conditions.parquet", index=False)
        matchups.build_all(("atp", "wta")).to_parquet(PROCESSED / "h2h.parquet", index=False)
        result["rebuilt"] = True

    result["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (PROCESSED / "last_refresh.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend the archive with recent results.")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-rebuild", action="store_true")
    args = ap.parse_args()

    if args.status:
        s = status()
        print(f"\nArchive freshness  ({s['checked_at']})")
        for tour, d in s["tours"].items():
            tag = "refreshable" if d["refreshable"] else "NO CURRENT SOURCE"
            print(f"  {tour.upper():<5} {d['matches']:>7,} matches  last {d['last_match']}"
                  f"  ({d['stale_days']} days stale)  [{tag}]")
        print()
        return

    r = refresh(rebuild=not args.no_rebuild)
    print("\n" + json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
