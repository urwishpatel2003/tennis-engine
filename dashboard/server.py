"""
Flask dashboard for the tennis engine.

    python dashboard/server.py            # http://localhost:5000

Reads local parquet only — it never touches the network and never imports the
fetch-only dependencies. Everything it serves comes from data/processed/ plus the
raw match log, all built offline by run_engine.py.

Endpoints
---------
GET /                     the SPA
GET /api/health           liveness + data presence (Railway healthcheck)
GET /api/meta             tours, surfaces, data freshness, synthetic-data flag
GET /api/search           player autocomplete
GET /api/rankings         power rankings for a (tour, surface)
GET /api/player           one player's profile: splits, form, rating history
GET /api/matchup          a full prediction
GET /api/fixtures         today's matches with live odds (The Odds API)
GET /api/refresh/status   archive freshness + last refresh result
POST /api/refresh         trigger a refresh (guarded by REFRESH_TOKEN)
GET /api/tournaments      events for a (tour, season)
GET /api/tournament       one draw, with a prediction against every match
GET /api/tournament/bracket  pre-tournament title odds (lazy: it is expensive)
GET /api/backtest         the most recent backtest results

The Engine object and the heavier frames are cached in-process; `clear_caches()`
drops them so a rebuild can be picked up without a restart.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time as _time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.predict import Engine  # noqa: E402
from engine import bet_log  # noqa: E402
from engine import kalshi  # noqa: E402
from engine import kalshi_match  # noqa: E402
from engine import kalshi_order  # noqa: E402
from engine import risk  # noqa: E402
from engine import live_feed  # noqa: E402
from engine.live_state import leverage, win_prob_from_state  # noqa: E402
from engine.schema import (  # noqa: E402
    PROCESSED, RAW, SURFACES, TOURS, normalise_surface,
)
from engine import live  # noqa: E402
from engine import refresh as refresher  # noqa: E402
from engine.tournament import TournamentStore  # noqa: E402
from rankings import build_rankings  # noqa: E402  (repo root, added to sys.path above)

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE))
# Distribution endpoints send numeric-but-stringified keys ("-3", "24") in the
# order the model produced them. Flask sorts JSON keys by default, which would
# reorder those lexicographically and scramble every chart's x-axis.
app.json.sort_keys = False

_engine: Engine | None = None
_tstore: TournamentStore | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def tstore() -> TournamentStore:
    """Tournament store, built lazily — it holds the per-match frozen state."""
    global _tstore
    if _tstore is None:
        _tstore = TournamentStore()
    return _tstore


@lru_cache(maxsize=8)
def _matches(tour: str) -> pd.DataFrame:
    p = RAW / f"matches_{tour}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def _ratings_history() -> pd.DataFrame:
    p = PROCESSED / "ratings.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def clear_caches() -> None:
    """Drop every cached frame so a fresh build is picked up without a restart."""
    global _engine, _tstore
    _engine = None
    _tstore = None
    live.clear()
    _matches.cache_clear()
    _ratings_history.cache_clear()
    build_info.cache_clear()


@lru_cache(maxsize=1)
def build_info() -> dict:
    """The build manifest written by tools/build_report.py, or a stub if absent."""
    p = PROCESSED / "build_info.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"ok": None, "built_at": None, "synthetic": (RAW / "SYNTHETIC.marker").exists(),
            "source": "unknown", "tours": {}}


def data_ready() -> bool:
    """Is there enough on disk to answer a real query?"""
    return (PROCESSED / "ratings_current.parquet").exists() and any(
        (RAW / f"matches_{t}.parquet").exists() for t in TOURS
    )


def _no_data():
    """
    Uniform 503 for every data-backed endpoint when the build produced nothing.

    A deployed instance that lost its data should say so plainly rather than
    throw a 500 out of pandas — the UI can then render an explanation instead of
    an empty table that looks like a modelling result.
    """
    return jsonify({
        "error": "no data on this instance",
        "detail": "The build did not produce the processed tables. "
                  "Run `python fetch_data.py && python run_engine.py --build`, "
                  "or redeploy so the build step regenerates them.",
        "build": build_info(),
    }), 503


def _clean(obj):
    """JSON-safe: numpy scalars, NaN/Inf, Timestamps, tuple keys."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.date().isoformat()
    if obj is pd.NaT or (obj is not None and obj is pd.NA):
        return None
    return obj


@app.after_request
def _never_cache_money(resp):
    """
    Nothing under /api/kalshi may be cached, by anything.

    These endpoints price real orders against a live balance. A browser serving
    a cached ticket list sizes a bet against a bankroll that no longer exists,
    which is how tickets kept coming back at their old contract counts after a
    fill had already moved the balance. The index is already no-store for the
    same reason; the money endpoints needed it more.
    """
    if request.path.startswith("/api/kalshi"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # no-store, not merely no-cache. The dashboard is a single HTML file holding
    # all its own JavaScript, so a browser holding an old copy shows old columns
    # and old decision rules while the server has already been fixed - which is
    # indistinguishable, from the outside, from the fix not working.
    resp = send_from_directory(HERE, "dashboard.html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/health")
def api_health():
    """
    Railway healthcheck. Deliberately touches no parquet.

    It answers 200 as long as the process is serving, and reports data presence in
    the body. Gating the healthcheck on data would make a data problem look like a
    dead process and send Railway into a restart loop that can never fix it.
    """
    return jsonify({"status": "ok", "data_ready": data_ready(), "build": build_info()})


@app.route("/api/meta")
def api_meta():
    info = build_info()
    out = {
        "tours": list(TOURS),
        "surfaces": ["overall", *SURFACES],
        "synthetic": bool(info.get("synthetic")),
        "data_ready": data_ready(),
        "built_at": info.get("built_at"),
        "source": info.get("source"),
        "data": {},
    }
    for tour in TOURS:
        m = _matches(tour)
        if not m.empty:
            out["data"][tour] = {
                "matches": int(len(m)),
                "seasons": [int(m["season"].min()), int(m["season"].max())],
                "latest": m["tourney_date"].max(),
                "players": int(pd.concat([m["winner_id"], m["loser_id"]]).nunique()),
            }
    return jsonify(_clean(out))


@app.route("/api/search")
def api_search():
    if not data_ready():
        return jsonify([])
    q = (request.args.get("q") or "").lower().strip()
    tour = request.args.get("tour", "atp")
    eng = engine()
    if eng.players.empty or len(q) < 2:
        return jsonify([])

    pool = eng.players[eng.players["tour"] == tour]
    hits = pool[pool["name_lower"].str.contains(q, regex=False, na=False)]

    # Rank by how much tour history the player has, so the tour regular comes first.
    counts = (
        eng.ratings[eng.ratings["surface"] == "overall"]
        .set_index("player_id")["matches"].to_dict()
        if not eng.ratings.empty else {}
    )
    # Deduplicate by name before truncating. Upstream gives the same person more
    # than one player_id (~1,800 rows share a name), so without this the dropdown
    # shows the same player twice and picking the wrong row silently selects an
    # orphaned id with almost no career history.
    hits = (
        hits.assign(n=hits["player_id"].map(counts).fillna(0))
        .sort_values("n", ascending=False)
        .drop_duplicates(subset="name_lower", keep="first")
        .head(12)
    )
    return jsonify(_clean([
        {"id": int(r.player_id), "name": r.name, "matches": int(r.n), "ioc": r.ioc}
        for r in hits.itertuples(index=False)
    ]))


@app.route("/api/rankings")
def api_rankings():
    if not data_ready():
        return _no_data()
    tour = request.args.get("tour", "atp")
    surface = request.args.get("surface", "overall")
    min_matches = int(request.args.get("min_matches", 20))
    limit = int(request.args.get("limit", 100))

    df = build_rankings(tour, surface, min_matches).head(limit)
    cols = ["rank", "player_id", "name", "ioc", "hand", "height", "elo_overall",
            "elo_surface", "elo_blend", "hold_pct", "break_pct", "serve_excess",
            "return_excess", "form_90d", "trend", "matches_overall", "matches_surface"]
    cols = [c for c in cols if c in df.columns]
    return jsonify(_clean(df[cols].to_dict("records")))


@app.route("/api/player")
def api_player():
    if not data_ready():
        return _no_data()
    tour = request.args.get("tour", "atp")
    pid = request.args.get("id")
    if pid is None:
        return jsonify({"error": "id required"}), 400
    pid = int(pid)

    eng = engine()
    prow = eng.players[
        (eng.players["tour"] == tour) & (eng.players["player_id"] == pid)
    ]
    if prow.empty:
        return jsonify({"error": "unknown player"}), 404
    prow = prow.iloc[0]

    # Surface splits
    splits = []
    for surf in ["overall", *SURFACES]:
        r = eng._elo_idx.get((tour, pid, surf))
        s = eng._sr_idx.get((tour, pid, surf))
        if r is None and s is None:
            continue
        splits.append(
            {
                "surface": surf,
                "elo": r["elo"] if r else None,
                "matches": r["matches"] if r else 0,
                "serve_excess": s["serve_excess"] if s else None,
                "return_excess": s["return_excess"] if s else None,
                "last_played": r.get("last_played") if r else None,
            }
        )

    # Rating history + results
    hist = _ratings_history()
    hist = hist[(hist["tour"] == tour) &
                ((hist["winner_id"] == pid) | (hist["loser_id"] == pid))]
    hist = hist.sort_values("tourney_date")
    won = hist["winner_id"] == pid
    curve = pd.DataFrame(
        {
            "date": hist["tourney_date"],
            "elo": np.where(won, hist["w_elo"], hist["l_elo"]),
            "surface": hist["surface"],
            "won": won,
        }
    )

    m = _matches(tour)
    recent = m[(m["winner_id"] == pid) | (m["loser_id"] == pid)].nlargest(
        25, "tourney_date"
    )
    results = [
        {
            "date": r.tourney_date,
            "tournament": r.tourney_name,
            "surface": r.surface,
            "round": r.round,
            "won": bool(r.winner_id == pid),
            "opponent": r.loser_name if r.winner_id == pid else r.winner_name,
            "score": r.score,
        }
        for r in recent.itertuples(index=False)
    ]

    # Win rate by surface, from actual results
    by_surface = []
    for surf in SURFACES:
        sub = m[((m["winner_id"] == pid) | (m["loser_id"] == pid)) & (m["surface"] == surf)]
        if len(sub) == 0:
            continue
        w = int((sub["winner_id"] == pid).sum())
        by_surface.append({"surface": surf, "wins": w, "losses": int(len(sub) - w),
                           "win_pct": w / len(sub)})

    return jsonify(_clean({
        "id": pid,
        "name": prow["name"],
        "hand": prow.get("hand"),
        "height": prow.get("height"),
        "ioc": prow.get("ioc"),
        "dob": prow.get("dob"),
        "splits": splits,
        "curve": curve.tail(400).to_dict("records"),
        "results": results,
        "record_by_surface": by_surface,
    }))


@app.route("/api/matchup")
def api_matchup():
    if not data_ready():
        return _no_data()
    args = request.args
    try:
        p = engine().predict(
            args.get("a"), args.get("b"),
            tour=args.get("tour", "atp"),
            surface=args.get("surface", "Hard"),
            best_of=int(args.get("best_of", 3)),
            match_date=args.get("date") or None,
            tournament=args.get("tournament") or None,
            indoor=args.get("indoor") == "1",
            altitude=float(args.get("altitude") or 0.0),
            market_prob_a=float(args["market_prob_a"]) if args.get("market_prob_a") else None,
        )
    except KeyError as e:
        return jsonify({"error": str(e)}), 404

    games = p.pop("_games_joint", {})
    p["total_games_probs"] = {str(k): v for k, v in p["total_games_probs"].items()}
    # Margin distribution powers the handicap chart in the UI.
    margins: dict[int, float] = {}
    for (ga, gb), prob in games.items():
        margins[ga - gb] = margins.get(ga - gb, 0.0) + prob
    p["game_margin_probs"] = {str(k): v for k, v in sorted(margins.items())}
    return jsonify(_clean(p))


@app.route("/api/fixtures")
def api_fixtures():
    """
    Today's and tomorrow's matches with live bookmaker odds, priced by the model.

    Needs ODDS_API_KEY. Answers 200 with `available: false` rather than an error
    when the key is absent, so the UI can explain the situation instead of showing
    a failure for what is really just an unconfigured optional feature.
    """
    if not live.have_key():
        return jsonify({
            "available": False,
            "reason": "ODDS_API_KEY is not set on this service",
            "fixtures": [], "meta": {},
        })
    if not data_ready():
        return _no_data()

    tours = tuple(request.args.get("tours", "atp,wta").split(","))
    include_started = request.args.get("include_started") == "1"
    if request.args.get("refresh") == "1":
        live.clear()
    try:
        d = live.fixtures(engine(), tours=tours, include_started=include_started)
    except Exception as e:  # network/quota/provider problems must not 500 the page
        return jsonify({"available": False, "reason": str(e)[:200],
                        "fixtures": [], "meta": {}})
    # Log whatever the page is about to recommend, so the track record is built
    # from what was actually shown at the time rather than reconstructed later.
    # There is no way to backfill it: the odds provider keeps no history, so a
    # replayed record would pair today's ratings with prices nobody was offered.
    try:
        bet_log.append([
            {"player_a": f["player_a"], "player_b": f["player_b"],
             "tour": f.get("tour"), "surface": f.get("surface"),
             "tournament": f.get("tournament"),
             "commence_time": f.get("commence_time"),
             "side": f["bet"]["side"], "player": f["bet"]["player"],
             "odds": f["bet"]["odds"], "ev": f["bet"]["ev"],
             "stake_pct": f["bet"]["stake_pct"],
             "model_prob_a": f.get("model_prob_a"),
             "market_prob_a": f.get("market_prob_a")}
            for f in d.get("fixtures", []) if f.get("bet")
        ])
    except Exception as e:  # noqa: BLE001 - logging must never break the page
        print(f"[bets] could not log recommendations: {e}", flush=True)

    d["available"] = True
    return jsonify(_clean(d))


# ticker -> the odds feed's start time, filled by the last scan. Kalshi cannot
# be asked this: its occurrence_datetime is a session marker shared by dozens of
# matches, not a per-match start.
TICKET_STARTS: dict[str, str] = {}


@app.route("/api/kalshi/tickets")
def api_kalshi_tickets():
    """
    Costed order tickets for whatever the model currently recommends.

    Builds only. Nothing here can place an order; submission is a separate
    endpoint that requires an explicit confirmation.

    A fixture with no Kalshi market, or no resting offer, produces a REASON
    rather than a ticket. Both are ordinary - most fixtures have neither.
    """
    if not kalshi.configured():
        return jsonify({"available": False,
                        "reason": "Kalshi credentials are not set", "tickets": []})
    if not live.have_key():
        return jsonify({"available": False,
                        "reason": "ODDS_API_KEY is not set, so there are no picks",
                        "tickets": []})
    try:
        bankroll = float(kalshi.balance().get("dollars") or 0.0)
    except Exception as e:
        return jsonify({"available": False,
                        "reason": f"could not read balance: {str(e)[:120]}",
                        "tickets": []})

    budget = risk.budget(bankroll, kalshi_order.open_tickers())
    try:
        fixtures = live.fixtures(engine(), tours=("atp", "wta")).get("fixtures", [])
    except Exception as e:
        return jsonify({"available": False, "reason": str(e)[:160], "tickets": []})

    picks = [f for f in fixtures if f.get("bet")]
    events = {t: kalshi_match.open_events(t) for t in ("atp", "wta")}

    # What is ALREADY on this contract. Without this the same recommendation is
    # offered again every time the page is scanned, and the model does not say
    # "back Parks twice" - it says back Parks. Doubling a position by re-reading
    # the same advice is not a second bet, it is the same bet placed twice.
    #
    # Positions are authoritative because they reflect FILLS. The ledger is the
    # fallback for an order that was sent but has not filled yet, which would
    # otherwise look untouched.
    held: dict[str, float] = {}
    try:
        for pos in kalshi.positions(limit=200):
            c = kalshi.num(pos.get("position_fp")) or 0.0
            if c:
                held[str(pos.get("ticker"))] = c
    except Exception:
        pass          # a held-check failure must not stop tickets being shown
    try:
        for t in risk.placed_here()["tickers"]:
            held.setdefault(str(t), 0.0)
    except Exception:
        pass
    tickets, skipped = [], []

    def _ctx(f: dict) -> dict:
        """
        Context on a pick that did NOT become a ticket.

        The Today page judges a pick against the BOOKMAKER's price; a Kalshi
        ticket needs an edge at KALSHI's ask, which is a different number on a
        different book. Carrying the start time and the model's probability lets
        the page explain the gap instead of the pick just going missing.
        """
        return {"match": f"{f['player_a']} v {f['player_b']}",
                "backing": (f.get("bet") or {}).get("player"),
                "tour": str(f.get("tour", "")).lower(),
                "starts": f.get("commence_time"),
                "model_prob": (f.get("model_prob_a") if
                               (f.get("bet") or {}).get("side") == "A"
                               else (1.0 - f["model_prob_a"]
                                     if f.get("model_prob_a") is not None else None)),
                "book_odds": (f.get("bet") or {}).get("odds")}

    for f in picks:
        tour = str(f.get("tour", "atp")).lower()
        found = kalshi_match.find_market(f["player_a"], f["player_b"], tour,
                                         events=events.get(tour))
        if not found.get("ok"):
            skipped.append({**_ctx(f),
                            "reason": found.get("reason")})
            continue
        bet = f["bet"]
        backing_a = bet["side"] == "A"
        market = found["a"] if backing_a else found["b"]
        price = kalshi_match.ask_price(market)
        if price is None:
            skipped.append({**_ctx(f),
                            "reason": "no offer resting on that contract"})
            continue
        prob = f["model_prob_a"] if backing_a else 1.0 - f["model_prob_a"]
        sized = kalshi.size_position(prob, price, bankroll, maker=risk.MAKER,
                                     max_stake_pct=budget["ticket_pct"])
        if sized["contracts"] < 1:
            skipped.append({**_ctx(f),
                            "price": price,
                            "reason": sized.get("reason") or "sized to nothing"})
            continue
        # A match already under way is not a ticket. The model priced it before
        # it began, and Kalshi keeps the market ACTIVE through the match, so
        # nothing about the market's own status would stop this.
        if kalshi_order.started(market, starts=f.get("commence_time")):
            skipped.append({**_ctx(f),
                            "price": price,
                            "reason": "the match has already started"})
            continue
        # Never ask for more than is actually offered. Sizing past the resting
        # ask either partially fills or walks up the book, and the EV was
        # computed at THIS price, not at whatever the next level costs.
        offered = kalshi_match.ask_size(market)
        capped = False
        if offered is not None and sized["contracts"] > int(offered):
            if int(offered) < 1:
                skipped.append({**_ctx(f),
                                "price": price,
                                "reason": "nothing offered at the ask"})
                continue
            capped = True
            sized = kalshi.size_position(
                prob, price, bankroll, maker=risk.MAKER,
                max_stake_pct=min(budget["ticket_pct"],
                                  100.0 * int(offered) * price / max(bankroll, 1e-9)))
            if sized["contracts"] < 1:
                skipped.append({**_ctx(f),
                                "price": price, "offered": offered,
                                "reason": "offered size too small to trade"})
                continue
        tickets.append({
            "match": f"{f['player_a']} v {f['player_b']}",
            "tournament": f.get("tournament"), "tour": tour,
            "backing": bet["player"], "ticker": market.get("ticker"),
            "event": found["event"], "match_type": found["match_type"],
            "model_prob": round(prob, 4), "price": price,
            "contracts": sized["contracts"], "stake": sized["stake"],
            "fee": sized["fee"], "ev": sized["ev"], "ev_pct": sized["ev_pct"],
            # Created ONCE here and echoed back on confirm, so a retry after a
            # timeout is recognised by Kalshi rather than doubling the position.
            "offered": offered, "size_capped": capped,
            # When the match starts. The odds feed is the model's own source, so
            # it leads; Kalshi's scheduled start is the fallback for a fixture
            # whose feed row carries no time.
            "starts": f.get("commence_time"),
            "held": held.get(str(market.get("ticker"))),
            "already_placed": str(market.get("ticker")) in held,
            "client_order_id": kalshi_order.new_client_order_id(),
        })

    # Sorted by the model's own confidence, highest first. EV sorts longshots to
    # the top - a 12% pick at a big enough price can out-EV a 70% one - which
    # puts the least likely bets in the most prominent place. Ties break on EV,
    # so among equally likely picks the better-priced one still leads.
    for t in tickets:
        if t.get("starts"):
            TICKET_STARTS[str(t["ticker"])] = str(t["starts"])

    tickets.sort(key=lambda t: (-(t["model_prob"] or 0), -(t["ev"] or 0)))
    return jsonify(_clean({
        "available": True, "tickets": tickets, "skipped": skipped,
        "budget": budget, "armed": kalshi_order.armed(),
        "live": kalshi.live_mode(),
    }))


@app.route("/api/kalshi/report")
def api_kalshi_report():
    """
    Everything about the Kalshi side of the account, read-only.

    Four questions, four sources, deliberately kept apart because they answer
    different things and disagreeing is informative:

      orders      what was SENT           - includes the ones that never traded
      fills       what actually TRADED    - the only proof of a position
      positions   what is OPEN now        - with unrealised exposure
      settlements what RESOLVED           - the only source of realised P/L

    An order is not a fill. A submitted order that rests unfilled costs nothing
    and means nothing, and counting it as a bet would flatter the record. The
    record here is built from settlements, so it reports money that actually
    moved.
    """
    if not kalshi.configured():
        return jsonify({"available": False,
                        "reason": "Kalshi credentials are not set"})
    out = {"available": True, "live": kalshi.live_mode(),
           "armed": kalshi_order.armed(), "errors": {}}

    def pull(name, fn, default):
        # One failing section must not blank the page. Report the failure in
        # place and render everything that did load.
        try:
            return fn()
        except Exception as e:
            out["errors"][name] = f"{type(e).__name__}: {str(e)[:120]}"
            return default

    bal = pull("balance", kalshi.balance, {})
    bankroll = float(bal.get("dollars") or 0.0)
    out["balance"] = bal.get("dollars")
    out["budget"] = risk.budget(bankroll, kalshi_order.open_tickers())
    out["caps"] = {"ticket_pct": risk.MAX_TICKET_PCT, "daily_pct": risk.MAX_DAILY_PCT,
                   "maker": risk.MAKER,
                   "kelly_fraction": kalshi.KELLY_FRACTION,
                   "max_price_drift": kalshi_order.MAX_PRICE_DRIFT}

    raw_orders = pull("orders", lambda: kalshi.orders(limit=200), [])
    raw_fills = pull("fills", lambda: kalshi.fills(limit=200), [])
    raw_pos = pull("positions", lambda: kalshi.positions(limit=200), [])
    raw_settle = pull("settlements", lambda: kalshi.settlements(limit=200), [])

    # SCOPE. Kalshi's portfolio endpoints return the whole ACCOUNT - every
    # market, including trades placed by hand in the app and anything that is
    # not tennis. Reporting that as "the record" would credit the model with
    # trades it never made, so the default is the orders this page actually
    # sent, taken from our own ledger.
    #
    #   page     what this application sent, by client_order_id and ticker
    #   tennis   any ATP/WTA singles market, however it was traded
    #   account  everything, unfiltered
    scope = str(request.args.get("scope", "page")).lower()
    if scope not in ("page", "tennis", "account"):
        scope = "page"
    mine = pull("ledger", risk.placed_here, {"tickers": set(), "order_ids": set()})
    out["scope"] = scope
    out["scope_counts"] = {}

    def ours(rec: dict) -> bool:
        if scope == "account":
            return True
        t = str(rec.get("ticker") or rec.get("market_ticker") or "")
        if scope == "tennis":
            return t.startswith("KXATPMATCH") or t.startswith("KXWTAMATCH")
        # A resting order carries our client_order_id; a fill or settlement does
        # not, so those fall back to the ticker.
        coid = str(rec.get("client_order_id") or "")
        return (coid in mine["order_ids"]) or (t in mine["tickers"])

    for label, rows in (("orders", raw_orders), ("fills", raw_fills),
                        ("positions", raw_pos), ("settlements", raw_settle)):
        out["scope_counts"][label] = {"account": len(rows),
                                      "shown": sum(1 for r in rows if ours(r))}
    raw_orders = [r for r in raw_orders if ours(r)]
    raw_fills = [r for r in raw_fills if ours(r)]
    raw_pos = [r for r in raw_pos if ours(r)]
    raw_settle = [r for r in raw_settle if ours(r)]

    # Tickers are opaque; a person reading this wants player names. Resolve each
    # ticker once, and let a lookup failure degrade to the ticker rather than
    # taking the page down with it.
    mkts: dict[str, dict] = {}

    def market_of(ticker: str) -> dict:
        t = str(ticker or "")
        if t and t not in mkts:
            try:
                mkts[t] = kalshi.market(t) or {}
            except Exception:
                mkts[t] = {}
        return mkts.get(t, {})

    def name_of(ticker: str) -> str:
        t = str(ticker or "")
        return str(market_of(t).get("yes_sub_title") or t)

    def sport_of(ticker: str) -> str:
        """
        Series prefix -> a readable sport. Tickers look like
        KXWTAMATCH-26AUG27TAUPAR-PAR, so the series is everything before the
        first dash. Unknown series degrade to the series itself rather than
        being lumped into an 'other' bucket that hides what they were.
        """
        series = str(ticker or "").split("-")[0]
        known = {"KXATPMATCH": "ATP", "KXWTAMATCH": "WTA"}
        if series in known:
            return known[series]
        s2 = series[2:] if series.startswith("KX") else series
        return (s2[:-5] if s2.endswith("MATCH") else s2) or "other"

    seen = {str(r.get("ticker") or "") for r in
            (raw_orders + raw_fills + raw_pos + raw_settle)}
    for t in list(seen)[:60]:          # bounded: this is one HTTP call each
        name_of(t)

    n = kalshi.num
    out["orders"] = [{
        "order_id": o.get("order_id"), "ticker": o.get("ticker"),
        "backing": name_of(o.get("ticker")),
        "sport": sport_of(o.get("ticker")),
        "status": o.get("status"), "side": o.get("book_side") or o.get("side"),
        "price": n(o.get("yes_price_dollars")),
        "placed": n(o.get("initial_count_fp")),
        "filled": n(o.get("fill_count_fp")),
        "remaining": n(o.get("remaining_count_fp")),
        "fees": (n(o.get("taker_fees_dollars")) or 0.0)
                + (n(o.get("maker_fees_dollars")) or 0.0),
        "created": o.get("created_time"),
    } for o in raw_orders]

    out["fills"] = [{
        "fill_id": f.get("fill_id") or f.get("trade_id"),
        "order_id": f.get("order_id"),
        "ticker": f.get("ticker") or f.get("market_ticker"),
        "backing": name_of(f.get("ticker") or f.get("market_ticker")),
        "sport": sport_of(f.get("ticker") or f.get("market_ticker")),
        "side": f.get("outcome_side"), "taker": f.get("is_taker"),
        "count": n(f.get("count_fp")),
        "price": n(f.get("yes_price_dollars")),
        "fee": n(f.get("fee_cost")),
        "at": f.get("created_time"),
    } for f in raw_fills]

    out["positions"] = [{
        "ticker": p_.get("ticker"), "backing": name_of(p_.get("ticker")),
        "sport": sport_of(p_.get("ticker")),
        "contracts": n(p_.get("position_fp")),
        "exposure": n(p_.get("market_exposure_dollars")),
        "traded": n(p_.get("total_traded_dollars")),
        "realized": n(p_.get("realized_pnl_dollars")),
        "fees": n(p_.get("fees_paid_dollars")),
        # What the position could be sold for RIGHT NOW: contracts x the resting
        # bid. The bid, not the ask - the ask is what a buyer pays, and marking
        # a position at the price you would have to pay to buy it again
        # overstates what it is worth.
        "bid": n(market_of(p_.get("ticker")).get("yes_bid_dollars")),
        "mark": round((n(p_.get("position_fp")) or 0.0)
                      * (n(market_of(p_.get("ticker")).get("yes_bid_dollars")) or 0.0), 2),
        "updated": p_.get("last_updated_ts"),
    } for p_ in raw_pos if (n(p_.get("position_fp")) or 0) != 0]
    for p_ in out["positions"]:
        # Unrealised against the cost of getting in. On an illiquid book the bid
        # is often zero, which marks a live position at nothing - that is what
        # you could actually get for it, not a bug, but it makes the figure
        # jumpy and it is labelled as a mark rather than a value.
        p_["unrealised"] = round((p_["mark"] or 0.0) - (p_["traded"] or 0.0), 2)

    settled = []
    for s_ in raw_settle:
        # revenue is CENTS while the cost fields are dollars - mixing them is an
        # easy way to report a P/L that is off by a factor of a hundred.
        revenue = (n(s_.get("revenue")) or 0.0) / 100.0
        cost = ((n(s_.get("yes_total_cost_dollars")) or 0.0)
                + (n(s_.get("no_total_cost_dollars")) or 0.0))
        fee = n(s_.get("fee_cost")) or 0.0
        settled.append({
            "ticker": s_.get("ticker"), "backing": name_of(s_.get("ticker")),
            "sport": sport_of(s_.get("ticker")),
            "result": s_.get("market_result"),
            "contracts": n(s_.get("yes_count_fp")) or n(s_.get("no_count_fp")),
            "cost": round(cost, 2), "revenue": round(revenue, 2),
            "fee": round(fee, 2), "pl": round(revenue - cost - fee, 2),
            "at": s_.get("settled_time"),
        })
    settled.sort(key=lambda r: str(r.get("at") or ""), reverse=True)
    out["settlements"] = settled

    staked = sum(r["cost"] for r in settled)
    realised = sum(r["pl"] for r in settled)
    wins = sum(1 for r in settled if r["pl"] > 0)
    out["record"] = {
        "settled": len(settled), "won": wins, "lost": len(settled) - wins,
        "win_pct": round(100.0 * wins / len(settled), 1) if settled else None,
        "staked": round(staked, 2),
        "realised_pl": round(realised, 2),
        "fees": round(sum(r["fee"] for r in settled), 2),
        # ROI on money actually put at risk, which is the number that matters
        # more than a win rate: one big loser undoes a lot of small winners.
        "roi_pct": round(100.0 * realised / staked, 1) if staked else None,
        "open_positions": len(out["positions"]),
        "open_exposure": round(sum(p_["exposure"] or 0.0
                                   for p_ in out["positions"]), 2),
        "orders_sent": len(out["orders"]),
        "orders_filled": sum(1 for o in out["orders"] if (o["filled"] or 0) > 0),
        "orders_unfilled": sum(1 for o in out["orders"] if not (o["filled"] or 0)),
        "open_mark": round(sum(p_["mark"] or 0.0 for p_ in out["positions"]), 2),
        "open_unrealised": round(sum(p_["unrealised"] or 0.0
                                     for p_ in out["positions"]), 2),
        "avg_stake": round(staked / len(settled), 2) if settled else None,
        "best": max((r["pl"] for r in settled), default=None),
        "worst": min((r["pl"] for r in settled), default=None),
    }
    # Total return counts open positions at their mark. Realised P/L alone
    # reads as a pure loss while everything is still open, which is exactly the
    # state this account is in tonight.
    out["record"]["total_return"] = round(
        out["record"]["realised_pl"] + out["record"]["open_unrealised"], 2)

    # ── by sport ──────────────────────────────────────────────────────────────
    # Tennis is what this model does; anything else on the account is noise
    # against it. Keeping them apart is the only way the tennis numbers mean
    # anything once other markets are in the mix.
    sports: dict[str, dict] = {}

    def bucket(name: str) -> dict:
        return sports.setdefault(name, {
            "sport": name, "orders": 0, "filled": 0, "open": 0,
            "open_exposure": 0.0, "open_mark": 0.0,
            "settled": 0, "won": 0, "staked": 0.0, "realised_pl": 0.0,
            "fees": 0.0})

    for o in out["orders"]:
        b = bucket(o["sport"])
        b["orders"] += 1
        b["filled"] += 1 if (o["filled"] or 0) > 0 else 0
        b["fees"] += o["fees"] or 0.0
    for p_ in out["positions"]:
        b = bucket(p_["sport"])
        b["open"] += 1
        b["open_exposure"] += p_["exposure"] or 0.0
        b["open_mark"] += p_["mark"] or 0.0
    for r in settled:
        b = bucket(r["sport"])
        b["settled"] += 1
        b["won"] += 1 if r["pl"] > 0 else 0
        b["staked"] += r["cost"]
        b["realised_pl"] += r["pl"]

    for b in sports.values():
        for k in ("open_exposure", "open_mark", "staked", "realised_pl", "fees"):
            b[k] = round(b[k], 2)
        b["roi_pct"] = (round(100.0 * b["realised_pl"] / b["staked"], 1)
                        if b["staked"] else None)
        b["win_pct"] = (round(100.0 * b["won"] / b["settled"], 1)
                        if b["settled"] else None)
    out["by_sport"] = sorted(sports.values(),
                             key=lambda b: (-(b["settled"] + b["open"]), b["sport"]))
    return jsonify(_clean(out))



@app.route("/api/kalshi/submit", methods=["POST"])
def api_kalshi_submit():
    """
    Place ONE order that a person has just confirmed.

    Every limit is re-checked here rather than trusted from the browser: a
    client is the wrong place to enforce a spending cap. The request must carry
    confirm=true, which is what makes this a deliberate act rather than a
    consequence of loading a page.
    """
    body = request.get_json(silent=True) or {}
    if body.get("confirm") is not True:
        return jsonify({"ok": False, "error": "confirmation required"}), 400
    try:
        ticker = str(body["ticker"])
        count = int(body["contracts"])
        price = float(body["price"])
        coid = str(body["client_order_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False,
                        "error": "ticker, contracts, price and client_order_id "
                                 "are all required"}), 400

    result = kalshi_order.create_order(ticker, count, price, coid,
                                       starts=TICKET_STARTS.get(ticker))
    # Log the attempt, not just the sends. A refusal is the interesting event -
    # it is the guard doing its job - and until now it went only to the browser,
    # which made "it said refused" impossible to diagnose from the logs.
    if result.get("error"):
        mkt = result.get("market") or {}
        print(f"[kalshi] REFUSED {ticker} {count}@{price:.2f}: {result['error']}"
              + (f" (ask now {mkt.get('ask')}, offered {mkt.get('offered')})"
                 if mkt.get("ask") is not None or mkt.get("offered") is not None
                 else ""), flush=True)
        return jsonify({"ok": False, **result}), 400
    if result.get("sent"):
        print(f"[kalshi] SENT {ticker} {count}@{price:.2f} -> "
              f"{json.dumps(result.get('response'))[:300]}", flush=True)
    if result.get("sent"):
        try:
            bet_log.append([{
                "player_a": body.get("match", ""), "player_b": "",
                "tour": body.get("tour"), "commence_time": body.get("commence_time"),
                "side": "KALSHI", "player": body.get("backing"),
                "odds": round(1.0 / price, 4) if price else None,
                "stake_pct": None, "venue": "kalshi", "ticker": ticker,
                "contracts": count, "price": price,
            }])
        except Exception as e:
            print(f"[kalshi] order placed but not logged: {e}", flush=True)
    return jsonify({"ok": True, **result})


@app.route("/api/track_record")
def api_track_record():
    """
    Profit and loss, in units, on every bet this dashboard has recommended.

    Flat staking (one unit a bet) is the headline because it is how betting
    records are conventionally reported and cannot be flattered by sizing. The
    Kelly-weighted figure is shown beside it since that is what the page
    suggested staking.

    Settled against the archive, so a bet only scores once the result is in it -
    which for the ATP can lag by a couple of days behind the actual match.
    """
    try:
        return jsonify(_clean(bet_log.summary()))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)[:200], "logged": 0, "settled": 0})


@app.route("/api/live")
def api_live():
    """
    Matches in progress, with a win probability computed from the live score.

    Needs LIVE_TENNIS_API_KEY (the free tier is enough). Like /api/fixtures this
    answers 200 with `available: false` when the key is absent or the upstream
    is unhappy, because an unconfigured optional feature is not a server error.

    Each match is priced twice: `prematch` is what the engine said before a ball
    was struck, `live` is the same chain entered at the current score. Showing
    both is the point — the gap IS the story of the match so far.
    """
    if not live_feed.configured():
        return jsonify({"available": False,
                        "reason": "LIVE_TENNIS_API_KEY is not set on this service",
                        "matches": []})
    if not data_ready():
        return _no_data()

    tour = request.args.get("tour") or None
    try:
        raw = live_feed.live_matches(
            tour, force=request.args.get("refresh") == "1")
    except live_feed.LiveFeedError as e:
        # 429 is worth naming: on a free key it means the panel is being polled
        # harder than the tier allows, which is a configuration problem rather
        # than a fault, and the operator can act on it.
        reason = ("rate limited by the provider - raise LIVE_TTL_SECONDS or the tier"
                  if e.status == 429 else str(e)[:200])
        return jsonify({"available": False, "reason": reason, "matches": []})

    eng = engine()
    out = []
    for m in raw:
        state = live_feed.parse_state(m)
        if state is None:
            continue
        players = m.get("players") or {}
        p1 = (players.get("p1") or {}).get("name")
        p2 = (players.get("p2") or {}).get("name")
        if not p1 or not p2:
            continue
        try:
            pred = eng.predict(
                p1, p2,
                tour=str(m.get("tour", "atp")).lower(),
                surface=normalise_surface(m.get("surface")),
                best_of=state["best_of"],
                tournament=m.get("tournament"),
            )
        except Exception:
            # An unknown player must drop one row, not the whole panel.
            continue
        pa, pb = pred["point_win_serve_a"], pred["point_win_serve_b"]
        try:
            p_live = win_prob_from_state(pa, pb, **state)
            lev = leverage(pa, pb, **state)
        except Exception:
            continue
        out.append({
            "id": m.get("id"),
            "tour": m.get("tour"),
            "tournament": m.get("tournament"),
            "round": m.get("round_code") or m.get("round"),
            "surface": normalise_surface(m.get("surface")),
            "best_of": state["best_of"],
            "player_a": pred["player_a"]["name"],
            "player_b": pred["player_b"]["name"],
            "scoreline": live_feed.scoreline(m),
            "points": [state["points_a"], state["points_b"]],
            "a_serving": state["a_serving"],
            "in_tiebreak": state["in_tiebreak"],
            "sets": [state["sets_a"], state["sets_b"]],
            "games": [state["games_a"], state["games_b"]],
            "prematch_a": pred["win_prob_a"],
            "live_a": p_live,
            "swing_a": p_live - pred["win_prob_a"],
            "leverage": lev,
            "data_quality": pred["data_quality"]["level"],
            "feed_age_seconds": (m.get("score") or {}).get("age_seconds"),
            "stale": bool((m.get("score") or {}).get("stale")),
        })

    out.sort(key=lambda r: -(r["leverage"] or 0))
    return jsonify(_clean({
        "available": True,
        "matches": out,
        "meta": {"cache_age_seconds": live_feed.cache_age(tour),
                 "ttl_seconds": live_feed.DEFAULT_TTL,
                 "count": len(out)},
    }))


# ──────────────────────────────────────────────────────────────────────────────
# Archive refresh
# ──────────────────────────────────────────────────────────────────────────────
# Deliberately NO Railway Volume. The build already fetches current data, and a
# refresh on boot plus a daily job keeps it current afterwards, so a restart
# self-heals instead of resurrecting a stale seed. The NFL engine needed a volume
# because its refresh output could not be reproduced from the image; this one can.
_refresh_state = {"running": False, "last": None, "error": None, "started": None}


def _run_refresh(reason: str) -> dict:
    """Refresh the archive and rebuild. Never allowed to take the server down."""
    if _refresh_state["running"]:
        return {"ok": False, "reason": "a refresh is already running"}
    _refresh_state.update({"running": True, "error": None,
                           "started": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    try:
        print(f"[refresh] starting ({reason})", flush=True)
        # Release the cached frames FIRST. They are the biggest thing resident,
        # they are about to be invalidated anyway, and holding them through a
        # rebuild is what got the container OOM-killed on boot.
        clear_caches()
        import gc as _gc
        _gc.collect()
        result = refresher.refresh(rebuild=True, verbose=True)
        result["reason"] = reason
        _refresh_state["last"] = result
        clear_caches()
        print(f"[refresh] done — archive now to {result['after']['last_match']}", flush=True)
        return {"ok": True, **result}
    except Exception as e:
        # A bad upstream response or an OOM must not leave the dashboard dead;
        # the previously built data is still perfectly serveable.
        msg = f"{type(e).__name__}: {e}"[:300]
        _refresh_state["error"] = msg
        print(f"[refresh] FAILED — {msg}", flush=True)
        return {"ok": False, "error": msg}
    finally:
        _refresh_state["running"] = False


def _refresh_loop() -> None:
    """Daily refresh at REFRESH_HOUR UTC, plus an optional one on boot."""
    if os.environ.get("REFRESH_ON_BOOT") == "1":
        # 90s, not 5s. The healthcheck window is 2 minutes and the refresh is the
        # heaviest thing this process does; starting it while the platform is
        # still deciding whether the deploy is healthy risks losing the deploy to
        # a memory spike rather than to anything actually wrong.
        _time.sleep(90)
        _run_refresh("boot")

    if os.environ.get("REFRESH_DAILY") != "1":
        return
    hour = int(os.environ.get("REFRESH_HOUR", "6"))
    while True:
        now = datetime.now(timezone.utc)
        secs = ((hour - now.hour - 1) % 24) * 3600 + (60 - now.minute) * 60
        _time.sleep(max(secs, 300))
        if datetime.now(timezone.utc).hour == hour:
            _run_refresh("daily")


def _live_feed_selftest() -> None:
    """
    Call the live feed once at boot and say plainly whether it worked.

    This exists because the machine this is developed on cannot reach
    api.livetennisapi.com at all - the corporate proxy blocks it - nor can it
    reach this service's own public URL. So the live path could be verified
    against a captured payload and a mocked feed, and no further: whether real
    API calls work in production was genuinely unknown from the outside.

    One request per boot, out of a 100/day free tier. It reports the shape of
    what came back rather than the contents, never prints the key, and can never
    take the process down - an optional integration failing must not stop the
    dashboard serving everything else. Set LIVE_SELFTEST=0 to skip it.
    """
    if os.environ.get("LIVE_SELFTEST") == "0":
        return
    if not live_feed.configured():
        print("[live] LIVE_TENNIS_API_KEY not set - live panel disabled", flush=True)
        return
    try:
        rows = live_feed.live_matches()
        scoreable = 0
        sample = ""
        for m in rows:
            st = live_feed.parse_state(m)
            if st is None:
                continue
            scoreable += 1
            if not sample:
                sample = (f"{m.get('tour')} {m.get('tournament')} "
                          f"{m.get('round_code')} sets={st['sets_a']}-{st['sets_b']} "
                          f"games={st['games_a']}-{st['games_b']}")
        print(f"[live] OK - API returned {len(rows)} match(es), "
              f"{scoreable} scoreable singles"
              + (f"; e.g. {sample}" if sample else ""), flush=True)
    except live_feed.LiveFeedError as e:
        print(f"[live] FAILED - {e}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[live] FAILED - {type(e).__name__}: {e}", flush=True)


def _kalshi_selftest() -> None:
    """
    Report at boot whether the Kalshi credentials work, and what the balance is.

    Nothing outside the container can check this: the credentials live here, and
    a bad paste otherwise only surfaces when somebody presses a button expecting
    a trade ticket. This distinguishes the three failures that look alike from
    the outside - a key that will not parse, a key that parses but is rejected,
    and a key aimed at the wrong environment.

    Reads only. It never prints the key, and there is no code path here that
    places an order.
    """
    if not kalshi.configured():
        print("[kalshi] credentials not set - sizing disabled", flush=True)
        return
    mode = "LIVE (real money)" if kalshi.live_mode() else "demo"
    # Say the arm state out loud. Whether a confirmed ticket is sent or logged is
    # the single most consequential setting here, it is decided by an env var
    # nobody can see from outside the container, and "is it actually armed?" is
    # otherwise answerable only by pressing the button and finding out.
    print(f"[kalshi] ARMED - confirmed tickets WILL be sent, cap "
          f"{risk.MAX_TICKET_PCT}% per ticket" if kalshi_order.armed()
          else "[kalshi] not armed - confirmed tickets are logged, not sent",
          flush=True)
    try:
        bal = kalshi.balance()
        print(f"[kalshi] OK - {mode}, balance ${bal.get('dollars')}", flush=True)
        # Which portfolio sections actually return data, and whether our own
        # ledger knows about the orders. The Kalshi tab reads all of these; when
        # it shows nothing, this says which one is empty and which one errored,
        # instead of leaving it to guesswork from outside the container.
        for label, fn in (("orders", lambda: kalshi.orders(limit=50)),
                          ("fills", lambda: kalshi.fills(limit=50)),
                          ("positions", lambda: kalshi.positions(limit=50)),
                          ("settlements", lambda: kalshi.settlements(limit=50))):
            try:
                rows = fn()
                sample = str((rows[0] or {}).get("ticker")) if rows else "-"
                print(f"[kalshi]   {label}: {len(rows)} row(s), first {sample}",
                      flush=True)
            except Exception as e:
                print(f"[kalshi]   {label}: FAILED {type(e).__name__}: "
                      f"{str(e)[:150]}", flush=True)
        # A cash-out sells the contracts back, so the market never settles for
        # them and no settlement row exists. Find out where Kalshi DOES record
        # it - a sell fill, or a closed position carrying realised P/L - rather
        # than guessing, which is how the last two of these went wrong.
        try:
            fl = kalshi.fills(limit=100)
            sides = {}
            for f in fl:
                sides[str(f.get("book_side"))] = sides.get(str(f.get("book_side")), 0) + 1
            print(f"[kalshi]   fills by book_side: {sides}", flush=True)
            for f in fl[:3]:
                print(f"[kalshi]     fill {f.get('ticker')} side={f.get('book_side')}"
                      f"/{f.get('outcome_side')} n={f.get('count_fp')}"
                      f" yes=${f.get('yes_price_dollars')} fee=${f.get('fee_cost')}",
                      flush=True)
            pods = [f for f in fl if "POD" in str(f.get("ticker","")).upper()]
            print(f"[kalshi]   fills mentioning POD: {len(pods)}", flush=True)
            for f in pods[:6]:
                print(f"[kalshi]     POD {f.get('ticker')} side={f.get('book_side')}"
                      f" n={f.get('count_fp')} yes=${f.get('yes_price_dollars')}"
                      f" at={f.get('created_time')}", flush=True)
            pos = kalshi.positions(limit=100)
            nz = [x for x in pos if (kalshi.num(x.get("realized_pnl_dollars")) or 0) != 0]
            print(f"[kalshi]   positions: {len(pos)} row(s), "
                  f"{len(nz)} with realised P/L", flush=True)
            for x in nz[:3]:
                print(f"[kalshi]     {x.get('ticker')} pos={x.get('position_fp')}"
                      f" realised=${x.get('realized_pnl_dollars')}"
                      f" traded=${x.get('total_traded_dollars')}", flush=True)
        except Exception as e:
            print(f"[kalshi]   cashout probe FAILED {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
        try:
            mine = risk.placed_here()
            print(f"[kalshi]   ledger: {len(mine['tickers'])} ticker(s), "
                  f"{len(mine['order_ids'])} order id(s) at "
                  f"{risk._ledger_path()}", flush=True)
        except Exception as e:
            print(f"[kalshi]   ledger: FAILED {type(e).__name__}: {str(e)[:150]}",
                  flush=True)
    except kalshi.KalshiError as e:
        hint = ""
        if e.status == 401:
            hint = (" - the key was rejected. A demo key cannot sign production "
                    "requests, or vice versa; KALSHI_LIVE decides which is used")
        print(f"[kalshi] FAILED ({mode}) - {e}{hint}", flush=True)
    except Exception as e:  # noqa: BLE001 - never break boot over an optional feature
        print(f"[kalshi] FAILED ({mode}) - {type(e).__name__}: {str(e)[:140]}", flush=True)


def _start_refresh_thread() -> None:
    if os.environ.get("REFRESH_DAILY") == "1" or os.environ.get("REFRESH_ON_BOOT") == "1":
        threading.Thread(target=_refresh_loop, name="refresh", daemon=True).start()
        print("[refresh] scheduler armed "
              f"(boot={os.environ.get('REFRESH_ON_BOOT')=='1'}, "
              f"daily={os.environ.get('REFRESH_DAILY')=='1'} "
              f"@{os.environ.get('REFRESH_HOUR','6')}:00 UTC)", flush=True)


@app.route("/api/refresh/status")
def api_refresh_status():
    out = {"archive": refresher.status(), "running": _refresh_state["running"],
           "started": _refresh_state["started"], "error": _refresh_state["error"],
           "last": _refresh_state["last"]}
    p = PROCESSED / "last_refresh.json"
    if out["last"] is None and p.exists():
        try:
            out["last"] = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify(_clean(out))


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    token = os.environ.get("REFRESH_TOKEN")
    if not token:
        return jsonify({"error": "REFRESH_TOKEN is not configured on this service"}), 503
    supplied = request.headers.get("X-Refresh-Token") or request.args.get("token")
    if supplied != token:
        return jsonify({"error": "bad or missing token"}), 401
    if request.args.get("async") == "1":
        threading.Thread(target=_run_refresh, args=("api",), daemon=True).start()
        return jsonify({"ok": True, "started": True, "note": "running in background"})
    return jsonify(_clean(_run_refresh("api")))


@app.route("/api/tournaments")
def api_tournaments():
    """Events for a (tour, season), newest first."""
    if not data_ready():
        return _no_data()
    tour = request.args.get("tour", "atp")
    season = request.args.get("season")
    st = tstore()
    rows = st.list_tournaments(tour, int(season) if season else None)
    return jsonify(_clean({"seasons": st.seasons(tour), "tournaments": rows}))


@app.route("/api/tournament")
def api_tournament():
    """One event's full draw with a prediction against every match."""
    if not data_ready():
        return _no_data()
    tour = request.args.get("tour", "atp")
    tid = request.args.get("tourney_id")
    if not tid:
        return jsonify({"error": "tourney_id required"}), 400
    d = tstore().draw(tour, tid)
    if "error" in d:
        return jsonify(d), 404
    return jsonify(_clean(d))


@app.route("/api/tournament/accuracy")
def api_tournament_accuracy():
    """
    Model accuracy for a handful of events, for the tournament listing.

    Batched and capped rather than computed for a whole season at once. Even on
    the probability-only path an event costs a second or three, and a season is
    85 of them - one request for all of them would sit there for minutes and
    then time out. The page asks for a few at a time and fills the column in as
    the answers arrive.
    """
    if not data_ready():
        return _no_data()
    tour = request.args.get("tour", "atp")
    ids = [i for i in (request.args.get("ids") or "").split(",") if i][:8]
    if not ids:
        return jsonify({"accuracy": {}})
    st = tstore()
    out = {}
    for tid in ids:
        try:
            out[tid] = st.accuracy(tour, tid)
        except Exception as e:
            # One unbuildable event must not blank the whole column.
            out[tid] = {"error": f"{type(e).__name__}"}
    return jsonify(_clean({"accuracy": out}))


@app.route("/api/tournament/bracket")
def api_tournament_bracket():
    """
    Pre-tournament title odds, simulated forward through the reconstructed draw.

    Separate from /api/tournament because it evaluates every possible pairing in
    the tree — thousands of matchups for a 128 draw — and the draw itself should
    not wait on it.
    """
    if not data_ready():
        return _no_data()
    tour = request.args.get("tour", "atp")
    tid = request.args.get("tourney_id")
    if not tid:
        return jsonify({"error": "tourney_id required"}), 400
    return jsonify(_clean(tstore().bracket(tour, tid)))


@app.route("/api/backtest")
def api_backtest():
    """
    Backtest results, preferring a freshly generated file over the committed one.

    The committed copy in reports/ exists because data/ is excluded from both git
    and the Railway upload, so a locally generated file can never reach the
    server — the Model tab would read "no backtest yet" forever. Running it during
    the build was tried and pushed the build past its deadline, so it ships as a
    report instead. It measures the MODEL, which changes far less often than the
    data does.
    """
    local = PROCESSED / "backtest_results.csv"
    shipped = Path(__file__).resolve().parent.parent / "reports" / "backtest_results.csv"
    meta_path = shipped.parent / "backtest_meta.json"

    src = local if local.exists() else (shipped if shipped.exists() else None)
    if src is None:
        return jsonify({"error": "no backtest available — run python backtest.py"}), 404

    meta = {}
    if meta_path.exists() and src == shipped:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify(_clean({
        "rows": pd.read_csv(src).to_dict("records"),
        "meta": {**meta, "source": "generated on this instance" if src == local
                 else "committed report"},
    }))


_start_refresh_thread()
_live_feed_selftest()
_kalshi_selftest()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(f"Tennis engine dashboard → http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=a.debug)
