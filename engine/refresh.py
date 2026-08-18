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

    ATP  → ManTennisData (atptour.com scrape) — includes serve statistics
    WTA  → api.wtatennis.com, the tour's own public API — scores only, no serve
           statistics, so those matches move Elo but are skipped by the
           serve/return builder (which already ignores rows without a stat line).

An earlier version of this file claimed no WTA source existed. That was wrong —
one GitHub search is not a search. The WTA publishes the same JSON its own site
consumes, unauthenticated.

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
import gc
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

from engine.schema import REFRESH_LOOKBACK_DAYS, build_match_id  # noqa: F401
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
            # start < after - lookback, NOT start <= after: the strict test
            # skipped the in-progress event the archive had just reached, so its
            # later rounds never arrived. See schema.REFRESH_LOOKBACK_DAYS.
            if pd.isna(start) or start < after - pd.Timedelta(
                    days=REFRESH_LOOKBACK_DAYS):
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

    # These three lists hold the whole ManTennisData archive as Python dicts and
    # are the peak memory of a refresh. Drop them before building the frame — a
    # boot-time refresh runs inside a container that is already holding the
    # engine's own frames, and the container was OOM-killed with no log line.
    del rows, tours, src_players
    gc.collect()

    raw = pd.DataFrame(out)
    del out
    gc.collect()

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
        m = pd.read_parquet(p, columns=["tourney_date", "season", "w_svpt"])
        last = m["tourney_date"].max()
        recent = m[m["tourney_date"] >= last - pd.Timedelta(days=90)]
        out["tours"][tour] = {
            "matches": int(len(m)),
            "last_match": str(last.date()),
            "stale_days": int((pd.Timestamp.today().normalize() - last.normalize()).days),
            "refreshable": True,
            # Ratings only need a result; the serve/return model needs a stat
            # line. Those can diverge — a build that skips per-match stats
            # advances one and not the other — so coverage is reported, not assumed.
            "serve_stat_coverage_90d": (round(float(recent["w_svpt"].notna().mean()), 3)
                                        if len(recent) else None),
        }
    return out


def _fetch_new_wta(after: pd.Timestamp, verbose: bool = True,
                   with_stats: bool = True) -> tuple[pd.DataFrame, dict]:
    """Completed WTA main-draw singles after `after`, canonicalised."""
    from engine import wta_source
    from engine.schema import ROUND_ORD

    players = pd.read_parquet(RAW / "players_wta.parquet")
    resolver = PlayerResolver(players)
    raw, stats = wta_source.fetch(after, resolver, verbose=verbose,
                                  with_stats=with_stats)
    if raw.empty:
        return pd.DataFrame(), stats

    raw["_r"] = raw["round"].map(ROUND_ORD).fillna(99)
    raw = raw.sort_values(["tourney_id", "_r", "src_id"]).reset_index(drop=True)
    raw["match_num"] = raw.groupby("tourney_id").cumcount() + 1
    raw = raw.drop(columns=["_r"])

    canon = normalise_matches(raw, "wta")
    canon["source"] = "wta_api"
    if resolver.new_rows:
        resolver.merged_players().to_parquet(RAW / "players_wta.parquet", index=False)

    stats.update({
        "new": int(len(canon)),
        "players_matched": resolver.matched,
        "players_minted": resolver.minted,
        "date_range": [str(canon["tourney_date"].min().date()),
                       str(canon["tourney_date"].max().date())] if len(canon) else None,
    })
    return canon, stats


def drop_duplicate_pairings(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Collapse rows describing the same match, returning (frame, n_dropped).

    Deduping on `match_id` alone is not enough. That id ends in `match_num`, a
    positional cumcount within the tournament, so when new results arrive for an
    event already in the archive every later match_num shifts and the SAME match
    comes back under a new id — observed at Cincinnati 2026, where Swiatek beat
    Sakkari once but was stored twice, as match 65 and match 70. A duplicated
    result is counted twice by the ratings, so this is a correctness bug rather
    than untidiness.

    Two players cannot meet twice in the same event, so (event, unordered player
    pair) is the key that actually identifies a match. Keeps the FIRST occurrence,
    which on an existing archive is the row already published.
    """
    if df.empty:
        return df, 0
    pair = pd.DataFrame({
        "t": df["tourney_id"].astype(str),
        "lo": df[["winner_id", "loser_id"]].min(axis=1),
        "hi": df[["winner_id", "loser_id"]].max(axis=1),
    })
    dup = pair.duplicated(keep="first").to_numpy()
    return df[~dup], int(dup.sum())


def _merge(tour: str, new: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Append `new` to a tour's archive, refusing to lose rows silently.

    The null/duplicate check is not defensive padding: ManTennisData's
    `match_order` column is empty, which made every `match_id` NaN and let
    `drop_duplicates` collapse 2,334 fetched matches into ONE on the first run.
    A merge that quietly discards data is worse than a merge that fails.
    """
    path = RAW / f"matches_{tour}.parquet"
    existing = pd.read_parquet(path)
    last = existing["tourney_date"].max()
    before = {"matches": int(len(existing)), "last_match": str(last.date())}

    if new is None or new.empty:
        if verbose:
            print(f"  [refresh] {tour.upper()} already up to date")
        return {"before": before, "after": before, "added": 0}

    bad = int(new["match_id"].isna().sum())
    dupes = int(new["match_id"].duplicated().sum())
    if bad or dupes:
        raise ValueError(
            f"refusing to merge {tour}: {bad} null and {dupes} duplicate match_ids "
            f"in {len(new)} fetched rows — that silently collapses matches on merge."
        )

    for c in existing.columns:
        if c not in new.columns:
            new[c] = pd.NA
    if "source" not in existing.columns:
        existing["source"] = "sackmann"
    # Preserve src_id. Without it there is no way to map an archive row back to
    # the upstream match, which is exactly what a later stats backfill needs —
    # and builds now ingest WTA matches without stats.
    if "src_id" in new.columns and "src_id" not in existing.columns:
        existing["src_id"] = pd.NA

    # Re-key rows already in the archive onto the stable id before comparing.
    # Older rows were written when match_id ended in a positional cumcount; the
    # incoming rows now key on src_id, so without this the two would never match
    # and every refresh would re-add the same matches for the duplicate guard to
    # strip again. One pass and the archive converges.
    if "src_id" in existing.columns and existing["src_id"].notna().any():
        existing = existing.copy()
        existing["match_id"] = build_match_id(existing)

    keep = [c for c in existing.columns]
    merged = pd.concat([existing, new[keep]], ignore_index=True)
    n0 = len(merged)
    merged = merged.drop_duplicates(subset=["match_id"], keep="first")

    # match_id is NOT as stable as schema.py claims. It ends in `match_num`,
    # which is a positional cumcount within the tournament, so when new results
    # arrive for an event already in the archive every later match_num shifts
    # and the SAME match comes back under a new id. Deduping on match_id alone
    # then lets it through twice — observed at Cincinnati 2026, where Swiatek
    # beat Sakkari once but was recorded twice, under match numbers 65 and 70.
    #
    # Two players cannot meet twice in one event, so the pairing is the key that
    # actually holds. This catches the duplicate however the id was built, and
    # repairs rows already written by the old code.
    merged, dupes = drop_duplicate_pairings(merged)
    if dupes and verbose:
        print(f"  [refresh] {tour.upper()}: dropped {dupes} duplicate "
              f"pairing(s) — same two players, same event")

    merged = merged.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    merged.to_parquet(path, index=False)

    after = {"matches": int(len(merged)),
             "last_match": str(merged["tourney_date"].max().date())}
    if verbose:
        dupe_note = f"  ({n0 - len(merged)} already present)" if n0 - len(merged) else ""
        print(f"  [refresh] {tour.upper()}: {before['matches']:,} -> {after['matches']:,} "
              f"matches; now current to {after['last_match']}{dupe_note}")
    return {"before": before, "after": after,
            "added": int(after["matches"] - before["matches"])}


def refresh(rebuild: bool = True, verbose: bool = True,
            wta_stats: bool = True, backfill: bool = True,
            backfill_limit: int = 250) -> dict:
    """
    Append new results for BOTH tours, then rebuild.

    ATP comes from ManTennisData (with serve statistics); WTA from the tour's own
    public API (scores only, no serve stats — those matches move Elo but are
    skipped by the serve/return builder, which degrades cleanly).
    """
    result = {"tours": {}, "rebuilt": False}

    # Per-match WTA stats are ~1 request each with a politeness floor, so a
    # thousand new matches is minutes of wall clock. That is fine at runtime and
    # fatal inside a build: it is what pushed the Railway build past its deadline
    # (twice, and I first blamed the backtest for it). Builds pass wta_stats=False.
    def _wta(after, verbose=True):
        return _fetch_new_wta(after, verbose=verbose, with_stats=wta_stats)

    for tour, fetcher in (("atp", fetch_new_atp), ("wta", _wta)):
        path = RAW / f"matches_{tour}.parquet"
        if not path.exists():
            result["tours"][tour] = {"error": "no archive — run fetch_data.py first"}
            continue
        last = pd.read_parquet(path, columns=["tourney_date"])["tourney_date"].max()
        try:
            new, stats = fetcher(last, verbose=verbose)
        except Exception as e:
            # One tour's upstream being down must not block the other's refresh.
            msg = f"{type(e).__name__}: {e}"[:200]
            if verbose:
                print(f"  [refresh] {tour.upper()} fetch failed — {msg}")
            result["tours"][tour] = {"error": msg}
            continue
        merged = _merge(tour, new, verbose=verbose)
        merged["fetched"] = stats
        result["tours"][tour] = merged

    added = sum(t.get("added", 0) for t in result["tours"].values())
    result["added_total"] = added

    # Builds ingest WTA rows without stats, so top up the most recent gap before
    # rebuilding — otherwise the serve books never see those matches at all.
    if wta_stats and backfill:
        try:
            bf = backfill_wta_stats(limit=backfill_limit, verbose=verbose)
            result["wta_backfill"] = bf
            if bf.get("filled"):
                added += bf["filled"]          # force a rebuild; the books changed
        except Exception as e:
            result["wta_backfill"] = {"error": f"{type(e).__name__}: {e}"[:200]}

    if rebuild and added:
        from engine import conditions, matchups, rally, ratings, serve_return
        if verbose:
            print("  [refresh] rebuilding engine ...")
        # Each stage is written and released before the next begins. Holding all
        # four in memory at once is what makes this the heaviest thing the server
        # ever does.
        pm, cur = ratings.build_all(("atp", "wta"))
        pm.to_parquet(PROCESSED / "ratings.parquet", index=False)
        cur.to_parquet(PROCESSED / "ratings_current.parquet", index=False)
        del pm, cur
        gc.collect()

        pm, cur = serve_return.build_all(("atp", "wta"))
        pm.to_parquet(PROCESSED / "serve_return.parquet", index=False)
        cur.to_parquet(PROCESSED / "serve_return_current.parquet", index=False)
        del pm, cur
        gc.collect()

        conditions.build_all(("atp", "wta")).to_parquet(
            PROCESSED / "conditions.parquet", index=False)
        gc.collect()
        matchups.build_all(("atp", "wta")).to_parquet(PROCESSED / "h2h.parquet", index=False)
        gc.collect()

        rp, rc = rally.build_all(("atp", "wta"))
        if not rp.empty:
            rp.to_parquet(PROCESSED / "rally.parquet", index=False)
            rc.to_parquet(PROCESSED / "rally_current.parquet", index=False)
        del rp, rc
        gc.collect()
        result["rebuilt"] = True
    elif rebuild and verbose:
        print("  [refresh] nothing new — skipping rebuild")

    # Back-compat for callers that read the old flat shape.
    atp = result["tours"].get("atp", {})
    if "before" in atp:
        result["before"], result["after"] = atp["before"], atp["after"]

    result["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (PROCESSED / "last_refresh.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result



def backfill_wta_stats(limit: int = 250, verbose: bool = True) -> dict:
    """
    Fill in serve statistics for WTA rows that were ingested without them.

    The Railway build runs `refresh --no-wta-stats` because one HTTP request per
    match blows the build deadline. Those rows therefore advance the RATINGS but
    contribute nothing to the serve/return books, and the forward refresh will
    never revisit them — it only looks past the archive's last date. This closes
    that gap, newest first, bounded so a single run cannot go on indefinitely.

    Only `wta_api` rows are candidates. Sackmann rows missing stats are missing
    them upstream and no amount of fetching will help.
    """
    from engine import wta_source

    path = RAW / "matches_wta.parquet"
    if not path.exists():
        return {"filled": 0, "remaining": 0, "reason": "no WTA archive"}

    m = pd.read_parquet(path)
    if "source" not in m.columns:
        return {"filled": 0, "remaining": 0, "reason": "no source column"}

    missing = m[(m["source"] == "wta_api") & m["w_svpt"].isna()]
    total_missing = int(len(missing))
    if total_missing == 0:
        if verbose:
            print("  [backfill] every WTA row already has serve stats")
        return {"filled": 0, "remaining": 0}

    todo = missing.sort_values("tourney_date", ascending=False).head(limit)
    if verbose:
        print(f"  [backfill] {total_missing:,} WTA rows without serve stats; "
              f"attempting {len(todo):,} (newest first)")

    updates: dict = {}
    rate_limited = False
    for tid, grp in todo.groupby("tourney_id"):
        try:
            year_s, gid_s = str(tid).split("-W")
            year, gid = int(year_s), int(gid_s)
        except ValueError:
            continue
        wanted = {
            frozenset((_norm(r.winner_name), _norm(r.loser_name))): r.match_id
            for r in grp.itertuples(index=False)
        }
        try:
            got = wta_source.stats_for_matches(gid, year, wanted)
        except wta_source.RateLimited:
            rate_limited = True
            if verbose:
                print("  [backfill] rate limited — stopping and keeping what we have")
            break
        except Exception:
            continue
        updates.update(got)
        if verbose and got:
            print(f"    {grp['tourney_name'].iloc[0]}: {len(got)}/{len(grp)} filled")

    if not updates:
        return {"filled": 0, "remaining": total_missing, "rate_limited": rate_limited}

    m = m.set_index("match_id")
    for mid, vals in updates.items():
        for col, val in vals.items():
            if col in m.columns:
                m.loc[mid, col] = val
    m = m.reset_index()

    # Recompute the derived rates for the rows we touched. engine/schema.py
    # normally does this at ingest; a backfill writes the raw counts afterwards,
    # so spw/rpw would otherwise stay NaN and the serve books would still ignore
    # these matches — the exact problem this function exists to fix.
    touched = m["match_id"].isin(updates)
    for me in ("w", "l"):
        won = m.loc[touched, f"{me}_1stWon"] + m.loc[touched, f"{me}_2ndWon"]
        svpt = m.loc[touched, f"{me}_svpt"]
        m.loc[touched, f"{me}_spw"] = np.where(svpt > 0, won / svpt, np.nan)
    m.loc[touched, "w_rpw"] = 1.0 - m.loc[touched, "l_spw"]
    m.loc[touched, "l_rpw"] = 1.0 - m.loc[touched, "w_spw"]

    m.to_parquet(path, index=False)
    remaining = total_missing - len(updates)
    if verbose:
        print(f"  [backfill] filled {len(updates):,}; {remaining:,} still missing")
    return {"filled": int(len(updates)), "remaining": int(remaining),
            "rate_limited": rate_limited}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend the archive with recent results.")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--no-backfill", action="store_true",
                    help="skip topping up serve stats for WTA rows ingested "
                         "without them")
    ap.add_argument("--backfill-limit", type=int, default=250)
    ap.add_argument("--no-wta-stats", action="store_true",
                    help="skip per-match WTA serve statistics. One request per "
                         "match, so it is minutes for a large catch-up — used by "
                         "the Railway build, which has a deadline.")
    args = ap.parse_args()

    if args.status:
        s = status()
        print(f"\nArchive freshness  ({s['checked_at']})")
        for tour, d in s["tours"].items():
            tag = "refreshable" if d["refreshable"] else "NO CURRENT SOURCE"
            cov = d.get("serve_stat_coverage_90d")
            cov_s = f"  serve stats {cov*100:.0f}% (last 90d)" if cov is not None else ""
            print(f"  {tour.upper():<5} {d['matches']:>7,} matches  last {d['last_match']}"
                  f"  ({d['stale_days']} days stale)  [{tag}]{cov_s}")
        print()
        return

    r = refresh(rebuild=not args.no_rebuild, wta_stats=not args.no_wta_stats,
                backfill=not args.no_backfill,
                backfill_limit=args.backfill_limit)
    print("\n" + json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
