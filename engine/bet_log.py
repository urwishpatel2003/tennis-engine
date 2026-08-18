"""
Record every bet the dashboard recommends, then settle it against the archive.

Inputs : recommendations from engine/live.py, results from data/raw/matches_*.parquet
Outputs: an append-only JSONL log, and a settled profit/loss in units

Why it can only run forward
---------------------------
There is no honest way to backfill this. The Odds API serves CURRENT prices and
keeps no history, so a "record" reconstructed from past matches would pair
today's ratings with prices nobody was ever offered — hindsight wearing the
costume of a track record. The log therefore starts empty and fills as matches
are played.

That matters more here than it would elsewhere. This model has NOT beaten tennis
markets out of sample, so the expected outcome of a forward record is a slow
drift toward negative after the vig. Publishing it anyway is the point: a page
that names bets should be scored on them.

Persistence
-----------
The service has deliberately had no Railway Volume, because until now nothing
mutated at runtime. A bet log breaks that assumption. `BET_LOG_PATH` therefore
points wherever the operator wants; `is_persistent()` reports whether that
location survives a deploy, so the UI can say so instead of quietly resetting to
zero every time the image is rebuilt.

Sample size
-----------
Betting records are noisy far longer than people expect. At a ~5% hold, a
hundred flat bets is well inside the range a coin flip produces, so `summary()`
returns `meaningful` and the UI refuses to imply anything from a short run.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.schema import PROCESSED, RAW

# Default is INSIDE the image, which means it does not survive a deploy. Set
# BET_LOG_PATH to a mounted volume to keep a real record.
LOG_PATH = Path(os.environ.get("BET_LOG_PATH", str(PROCESSED / "bet_log.jsonl")))

# Below this, a profit/loss figure says nothing. Chosen from the arithmetic of
# the vig rather than taste: at ~5% hold and even money, the standard deviation
# of 100 flat bets is around 10 units, so anything inside +/-20 units is noise.
MEANINGFUL_AFTER = 200


def is_persistent() -> bool:
    """
    Whether the log lives somewhere that survives a redeploy.

    A record that silently resets is worse than no record, because it always
    looks like a fresh start rather than a lost one.
    """
    return bool(os.environ.get("BET_LOG_PATH"))


def _norm(name: object) -> str:
    return re.sub(r"[^a-z]", "", str(name or "").lower())


def key_for(rec: dict) -> str:
    """
    Stable identity for a recommendation: tour, pairing and date. ONE per match.

    Deliberately NOT keyed on the side. `side` is "A" or "B" relative to how the
    fixture happened to be ordered, so backing either player produced the same
    key and the second was silently dropped - a test logging opposite sides of
    one match got one of them recorded.

    Excluding side entirely is also the right rule rather than a workaround. As
    odds move the model can flip which player it prefers, and logging both would
    let one fixture contribute two entries to a record that is supposed to count
    decisions. The first recommendation stands.
    """
    a, b = sorted([_norm(rec.get("player_a")), _norm(rec.get("player_b"))])
    return f"{rec.get('tour')}|{a}|{b}|{str(rec.get('commence_time'))[:10]}"


def append(recs: list[dict]) -> int:
    """
    Log recommendations not already present. Returns how many were new.

    Deduped on `key_for`, because the Today page is refreshed repeatedly and the
    same fixture would otherwise be logged on every view — which would inflate a
    record with copies of one opinion.
    """
    if not recs:
        return 0
    existing = {r.get("key") for r in load()}
    fresh = []
    for r in recs:
        k = key_for(r)
        if k in existing:
            continue
        existing.add(k)
        fresh.append({**r, "key": k,
                      "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if not fresh:
        return 0
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        for r in fresh:
            fh.write(json.dumps(r) + "\n")
    return len(fresh)


def load() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a torn write must not destroy the whole record
    return out


def _results(tours=("atp", "wta")) -> pd.DataFrame:
    frames = []
    for t in tours:
        p = RAW / f"matches_{t}.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=["tour", "tourney_date", "winner_name",
                                            "loser_name", "completed", "retirement"])
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[df["completed"].fillna(False).astype(bool)]


def settle(tours=("atp", "wta")) -> list[dict]:
    """
    Attach outcomes to logged bets where the archive now knows the result.

    Matched on the normalised pairing within a few days of the scheduled start,
    because an event's `tourney_date` is its START date, not the match date.
    A bet with no result yet stays open rather than being scored as a loss.
    """
    bets = load()
    if not bets:
        return []
    res = _results(tours)
    if res.empty:
        return bets

    idx: dict[tuple[str, str], list] = {}
    for r in res.itertuples(index=False):
        w, l = _norm(r.winner_name), _norm(r.loser_name)
        idx.setdefault((min(w, l), max(w, l)), []).append((r.tourney_date, w))

    out = []
    for b in bets:
        a, bb = sorted([_norm(b.get("player_a")), _norm(b.get("player_b"))])
        start = pd.to_datetime(b.get("commence_time"), utc=True, errors="coerce")
        winner = None
        for date, w in idx.get((a, bb), []):
            d = pd.to_datetime(date, utc=True, errors="coerce")
            if pd.isna(start) or pd.isna(d) or abs((d - start).days) <= 21:
                winner = w
                break
        rec = dict(b)
        if winner is None:
            rec["settled"] = False
        else:
            won = _norm(b.get("player")) == winner
            odds = float(b.get("odds") or 0)
            rec["settled"] = True
            rec["won"] = won
            # Flat staking: one unit per bet, the standard way to report.
            rec["pl_units"] = (odds - 1.0) if won else -1.0
            # And sized the way the page suggested, so both are visible.
            stake = float(b.get("stake_pct") or 0) / 100.0
            rec["pl_kelly"] = stake * ((odds - 1.0) if won else -1.0)
        out.append(rec)
    return out


def summary(tours=("atp", "wta")) -> dict:
    rows = settle(tours)
    done = [r for r in rows if r.get("settled")]
    wins = [r for r in done if r.get("won")]
    pl = sum(r.get("pl_units", 0.0) for r in done)
    plk = sum(r.get("pl_kelly", 0.0) for r in done)
    staked_k = sum(float(r.get("stake_pct") or 0) / 100.0 for r in done)
    return {
        "logged": len(rows),
        "open": len(rows) - len(done),
        "settled": len(done),
        "won": len(wins),
        "lost": len(done) - len(wins),
        "pl_units": round(pl, 2),
        "roi_pct": round(100.0 * pl / len(done), 2) if done else None,
        "pl_kelly_units": round(plk, 3),
        "kelly_roi_pct": round(100.0 * plk / staked_k, 2) if staked_k else None,
        "avg_odds": round(sum(float(r.get("odds") or 0) for r in done) / len(done), 2)
        if done else None,
        # The UI must not draw conclusions from a short run; this says when it may.
        "meaningful": len(done) >= MEANINGFUL_AFTER,
        "meaningful_after": MEANINGFUL_AFTER,
        "persistent": is_persistent(),
        "path": str(LOG_PATH),
        "recent": sorted(rows, key=lambda r: str(r.get("logged_at")), reverse=True)[:15],
    }
