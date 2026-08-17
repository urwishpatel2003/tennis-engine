"""
Live fixtures with odds — today's and tomorrow's matches, priced by the model.

    python engine/live.py                      # every in-season tournament
    python engine/live.py --tour atp --top 20

Source: The Odds API (the-odds-api.com), the same provider and key the NFL engine
already uses. Reads ODDS_API_KEY from the environment — never hardcoded, never
committed, never logged.

Why this module exists
----------------------
The Sackmann archive is results-only and has no forward fixtures, so the engine
could rank and replay but never answer "who is playing today?". The Odds API
posts tennis as ONE SPORT KEY PER TOURNAMENT (`tennis_atp_cincinnati_open`), and
that set changes every week, so the keys are discovered at runtime from the free
`/sports` endpoint rather than hardcoded.

Credit discipline (mirrors ml/odds.py in the NFL engine)
--------------------------------------------------------
* `/sports` is free and does not count against the quota.
* Odds requests are cached for `_TTL` seconds, so repeated dashboard views do not
  re-spend credits.
* One request per tournament pulls every bookmaker at once.
* `x-requests-remaining` is surfaced in `meta` so the budget is always visible.

The honest caveat
-----------------
Ratings come from the archive, which currently ends months before today's play.
Every response carries `ratings_as_of` and `ratings_stale_days` for exactly this
reason: a model price built on stale ratings deserves to be labelled as such,
especially when it sits next to a live market price.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.predict import Engine
from engine.schema import RAW, normalise_surface

_BASE = "https://api.the-odds-api.com/v4"
_TTL = 600           # seconds; odds move, but not fast enough to justify re-spending
_SPORTS_TTL = 3600   # the in-season tournament list changes weekly at most

_cache: dict = {"sports": {"t": 0.0, "data": None},
                "odds": {}, "meta": {"remaining": None}}


def have_key() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def _get(path: str, **params) -> tuple[object, str | None]:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY is not set. Export it locally, or set it on the service "
            "with `railway variables --set ODDS_API_KEY=...`."
        )
    params["apiKey"] = key
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_BASE}{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "tennis-engine/0.1"})
    with urllib.request.urlopen(req, timeout=25) as r:
        remaining = r.headers.get("x-requests-remaining")
        _cache["meta"]["remaining"] = remaining
        return json.load(r), remaining


# ──────────────────────────────────────────────────────────────────────────────
# Name matching
# ──────────────────────────────────────────────────────────────────────────────
def norm_name(s: object) -> str:
    """
    Aggressively normalise a player name for cross-source matching.

    The two feeds disagree constantly in ways that carry no information:
    'Alex de Minaur' vs 'Alex De Minaur', 'Félix Auger-Aliassime' vs
    'Felix Auger Aliassime'. Strip accents, punctuation and case, collapse
    whitespace, and compare what is left.
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ").replace(".", " ").replace("'", "")
    return re.sub(r"[^a-z ]", "", s).strip()


def _surname_key(s: str) -> str:
    parts = norm_name(s).split()
    return parts[-1] if parts else ""


class NameIndex:
    """Maps a feed's player names onto our player_ids."""

    def __init__(self, eng: Engine, tour: str) -> None:
        self.tour = tour
        pool = eng.players[eng.players["tour"] == tour]
        counts = eng._match_counts()
        pool = pool.assign(nmatches=pool["player_id"].map(counts).fillna(0))
        # Most-played wins, which also resolves the duplicate-id problem upstream
        # has for ~858 ATP names.
        pool = pool.sort_values("nmatches", ascending=False)

        self.exact: dict[str, dict] = {}
        self.by_surname: dict[str, list] = {}
        for r in pool.itertuples(index=False):
            n = norm_name(r.name)
            if not n:
                continue
            self.exact.setdefault(n, {"id": int(r.player_id), "name": r.name, "n": r.nmatches})
            self.by_surname.setdefault(_surname_key(r.name), []).append(
                {"id": int(r.player_id), "name": r.name, "n": r.nmatches, "norm": n}
            )

    def lookup(self, name: str) -> dict | None:
        n = norm_name(name)
        if not n:
            return None
        if n in self.exact:
            return self.exact[n]

        # Fall back to surname + first-initial, which catches 'A. Zverev' style
        # renderings and reversed 'Zverev Alexander' orderings.
        parts = n.split()
        cands = self.by_surname.get(parts[-1], []) or self.by_surname.get(parts[0], [])
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        initial = parts[0][:1]
        narrowed = [c for c in cands if c["norm"].split()[0][:1] == initial]
        pick = (narrowed or cands)[0]
        return pick


# ──────────────────────────────────────────────────────────────────────────────
# Tournament context
# ──────────────────────────────────────────────────────────────────────────────
_SLAM_HINTS = ("australian open", "roland garros", "french open", "wimbledon", "us open")


def _archive_context(title: str, tour: str) -> dict:
    """
    Infer surface and format for a live event by looking it up in our own archive.

    The odds feed says 'ATP Cincinnati Open' and nothing about the court. We have
    fifteen years of Cincinnati; the surface it was played on last time is a far
    better guess than a global default.
    """
    name = re.sub(r"^(ATP|WTA)\s+", "", str(title)).strip()
    out = {"surface": "Hard", "best_of": 3, "level": None, "matched": None}

    p = RAW / f"matches_{tour}.parquet"
    if p.exists():
        m = pd.read_parquet(p, columns=["tourney_name", "surface", "best_of",
                                        "tourney_level", "season"])
        key = norm_name(name)
        m = m.assign(_k=m["tourney_name"].map(norm_name))
        hit = m[m["_k"] == key]
        if hit.empty:  # 'Cincinnati Open' vs 'Cincinnati Masters'
            first = key.split()[0] if key else ""
            hit = m[m["_k"].str.startswith(first, na=False)] if first else hit
        if not hit.empty:
            recent = hit.sort_values("season").tail(400)
            out["surface"] = normalise_surface(recent["surface"].mode().iloc[0])
            out["best_of"] = int(recent["best_of"].mode().iloc[0])
            out["level"] = recent["tourney_level"].mode().iloc[0]
            out["matched"] = recent["tourney_name"].iloc[-1]

    # Slams are best-of-5 on the men's tour regardless of what the archive says
    # about a same-named warm-up event.
    if tour == "atp" and any(h in name.lower() for h in _SLAM_HINTS):
        out["best_of"] = 5
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Odds handling
# ──────────────────────────────────────────────────────────────────────────────
def devig(price_a: float, price_b: float) -> tuple[float, float]:
    """Proportional de-vig of a two-way decimal market."""
    if not price_a or not price_b or price_a <= 1 or price_b <= 1:
        return float("nan"), float("nan")
    ia, ib = 1.0 / price_a, 1.0 / price_b
    tot = ia + ib
    return ia / tot, ib / tot


def _consensus(event: dict) -> dict:
    """Median h2h price per player across every bookmaker in the response."""
    px: dict[str, list] = {}
    for b in event.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []):
                if o.get("price"):
                    px.setdefault(o["name"], []).append(float(o["price"]))
    out = {}
    for name, prices in px.items():
        prices.sort()
        mid = len(prices) // 2
        out[name] = (prices[mid] if len(prices) % 2 else
                     (prices[mid - 1] + prices[mid]) / 2.0)
    return {"prices": out, "books": len(event.get("bookmakers", []))}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def in_season_tournaments(force: bool = False) -> list[dict]:
    """Tennis sport keys currently posted. Free endpoint — no quota cost."""
    c = _cache["sports"]
    if not force and c["data"] is not None and time.time() - c["t"] < _SPORTS_TTL:
        return c["data"]
    sports, _ = _get("/sports/")
    rows = [
        {"key": s["key"], "title": s["title"],
         "tour": "atp" if "_atp_" in s["key"] else ("wta" if "_wta_" in s["key"] else None)}
        for s in sports if "tennis" in s.get("key", "") and s.get("active")
    ]
    rows = [r for r in rows if r["tour"]]
    c.update({"t": time.time(), "data": rows})
    return rows


def _events(sport_key: str, regions: str = "us,uk,eu") -> list[dict]:
    c = _cache["odds"].get(sport_key)
    if c and time.time() - c["t"] < _TTL:
        return c["data"]
    data, _ = _get(f"/sports/{sport_key}/odds/", regions=regions,
                   markets="h2h", oddsFormat="decimal")
    _cache["odds"][sport_key] = {"t": time.time(), "data": data}
    return data


def fixtures(
    eng: Engine | None = None,
    tours: tuple[str, ...] = ("atp", "wta"),
    blend_market: bool = False,
    include_started: bool = False,
) -> dict:
    """
    Every posted fixture, priced by the model and compared to the market.

    `blend_market=False` by default and that is deliberate: blending the price
    into the model and then reporting the gap to that same price is circular —
    the disagreement would shrink toward zero for the wrong reason.

    `include_started=False` matters even more. The feed keeps returning a match
    after it begins, but the prices then become IN-PLAY prices, and comparing a
    pre-match model to a live price is meaningless: a set down at 1.02 produced a
    "+468% EV" in the first run of this, which is not an edge, it is a category
    error. Anything already under way is dropped unless explicitly asked for.
    """
    eng = eng or Engine()
    now = pd.Timestamp.utcnow()
    idx = {t: NameIndex(eng, t) for t in tours}

    # How old are the ratings we are about to price today's tennis with?
    #
    # PER TOUR, not overall. ATP is refreshed from a live source and WTA is not,
    # so a single pooled figure would report the ATP's freshness and quietly hide
    # that every WTA price is running on months-old ratings.
    staleness: dict[str, dict] = {}
    if not eng.ratings.empty and "last_played" in eng.ratings.columns:
        today = pd.Timestamp.today().normalize()
        for t, grp in eng.ratings.groupby("tour"):
            mx = pd.to_datetime(grp["last_played"]).max()
            if pd.notna(mx):
                staleness[str(t)] = {"as_of": mx.date().isoformat(),
                                     "stale_days": int((today - mx.normalize()).days)}
    worst = max((v["stale_days"] for v in staleness.values()), default=None)
    as_of = min((v["as_of"] for v in staleness.values()), default=None)
    stale = worst

    out, unmatched = [], []
    events_seen = started = 0
    for t in in_season_tournaments():
        if t["tour"] not in tours:
            continue
        ctx = _archive_context(t["title"], t["tour"])
        for ev in _events(t["key"]):
            events_seen += 1
            start = pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")
            if pd.notna(start) and start <= now and not include_started:
                started += 1
                continue
            cons = _consensus(ev)
            names = list(cons["prices"].keys()) or [ev.get("home_team"), ev.get("away_team")]
            if len(names) < 2:
                continue
            a_name, b_name = names[0], names[1]
            a = idx[t["tour"]].lookup(a_name)
            b = idx[t["tour"]].lookup(b_name)
            if not a or not b:
                unmatched.append(a_name if not a else b_name)
                continue

            pa = cons["prices"].get(a_name)
            pb = cons["prices"].get(b_name)
            mkt_a, mkt_b = devig(pa, pb)

            try:
                pred = eng.predict(
                    a["id"], b["id"], tour=t["tour"], surface=ctx["surface"],
                    best_of=ctx["best_of"], tournament=ctx["matched"] or t["title"],
                    market_prob_a=(mkt_a if blend_market and mkt_a == mkt_a else None),
                )
            except KeyError:
                unmatched.append(f"{a_name}/{b_name}")
                continue

            row = {
                "event_id": ev.get("id"),
                "tour": t["tour"],
                "tournament": t["title"],
                "surface": ctx["surface"],
                "best_of": ctx["best_of"],
                "commence_time": ev.get("commence_time"),
                "player_a": a["name"], "player_a_id": a["id"],
                "player_b": b["name"], "player_b_id": b["id"],
                "model_prob_a": pred["win_prob_a"],
                "fair_odds_a": pred["fair_odds_a"],
                "fair_odds_b": pred["fair_odds_b"],
                "expected_total_games": pred["expected_total_games"],
                "fair_total_line": pred["fair_total_games_line"],
                "fair_handicap_a": pred["fair_game_handicap_a"],
                "likely_score": (pred["likely_scorelines"][0]["score"]
                                 if pred["likely_scorelines"] else None),
                "data_quality": pred["data_quality"]["level"],
                "quality_flags": pred["data_quality"]["flags"],
                "books": cons["books"],
                "odds_a": pa, "odds_b": pb,
                "ratings_stale_days": staleness.get(t["tour"], {}).get("stale_days"),
            }
            if mkt_a == mkt_a:  # not NaN
                row.update({
                    "market_prob_a": mkt_a,
                    "disagreement_a": pred["win_prob_a"] - mkt_a,
                    "ev_a": pred["win_prob_a"] * pa - 1.0,
                    "ev_b": pred["win_prob_b"] * pb - 1.0,
                })
                row["best_side"] = "A" if row["ev_a"] >= row["ev_b"] else "B"
                row["best_ev"] = max(row["ev_a"], row["ev_b"])
            out.append(row)

    out.sort(key=lambda r: (r.get("commence_time") or "", r["tournament"]))
    return {
        "fixtures": out,
        "meta": {
            "tournaments": in_season_tournaments(),
            "events_seen": events_seen,
            "already_started": started,
            "matched": len(out),
            "unmatched": sorted(set(unmatched))[:20],
            "credits_remaining": _cache["meta"]["remaining"],
            "ratings_as_of": as_of,
            "ratings_stale_days": stale,
            "staleness_by_tour": staleness,
            "market_blended": blend_market,
        },
    }


def clear() -> None:
    _cache["sports"] = {"t": 0.0, "data": None}
    _cache["odds"] = {}


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Today's fixtures with odds.")
    ap.add_argument("--tour", nargs="+", default=["atp", "wta"])
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--include-started", action="store_true",
                    help="also show matches already under way. Their prices are "
                         "IN-PLAY, so model-vs-market on them is meaningless.")
    args = ap.parse_args()

    if not have_key():
        print("ODDS_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(2)

    d = fixtures(tours=tuple(args.tour), include_started=args.include_started)
    if args.json:
        print(json.dumps(d, indent=2, default=str))
        return

    m = d["meta"]
    print(f"\n{len(d['fixtures'])} fixtures from {len(m['tournaments'])} live "
          f"tournament(s) · {m['credits_remaining']} credits left")
    if m["ratings_stale_days"] and m["ratings_stale_days"] > 21:
        print(f"  ** ratings are {m['ratings_stale_days']} days old "
              f"(archive ends {m['ratings_as_of']}) — model prices are degraded **")
    if m["unmatched"]:
        print(f"  unmatched players: {', '.join(m['unmatched'][:6])}")

    print(f"\n{'start':<12}{'match':<44}{'model':>7}{'market':>8}{'diff':>7}"
          f"{'side':>7}{'EV':>8}")
    print("─" * 94)
    for r in d["fixtures"][:args.top]:
        t = (r["commence_time"] or "")[5:16].replace("T", " ")
        match = f"{r['player_a'][:19]} v {r['player_b'][:19]}"
        mk = f"{r['market_prob_a']*100:.1f}%" if "market_prob_a" in r else "   —"
        df = f"{r['disagreement_a']*100:+.1f}" if "disagreement_a" in r else "   —"
        side = "—"
        if r.get("best_side"):
            side = str(r["player_a"] if r["best_side"] == "A" else r["player_b"]).split()[-1][:6]
        ev = f"{r['best_ev']*100:+.1f}%" if "best_ev" in r else "   —"
        print(f"{t:<12}{match:<44}{r['model_prob_a']*100:>6.1f}%{mk:>8}{df:>7}{side:>7}{ev:>8}")

    print("\nNOTE: 'diff' is disagreement with the price, not a proven edge. Tennis\n"
          "      match markets are efficient; validate before treating it as signal.")


if __name__ == "__main__":
    main()
