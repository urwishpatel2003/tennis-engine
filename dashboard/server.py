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

    budget = risk.budget(bankroll)
    try:
        fixtures = live.fixtures(engine(), tours=("atp", "wta")).get("fixtures", [])
    except Exception as e:
        return jsonify({"available": False, "reason": str(e)[:160], "tickets": []})

    picks = [f for f in fixtures if f.get("bet")]
    events = {t: kalshi_match.open_events(t) for t in ("atp", "wta")}
    tickets, skipped = [], []
    for f in picks:
        tour = str(f.get("tour", "atp")).lower()
        found = kalshi_match.find_market(f["player_a"], f["player_b"], tour,
                                         events=events.get(tour))
        if not found.get("ok"):
            skipped.append({"match": f"{f['player_a']} v {f['player_b']}",
                            "reason": found.get("reason")})
            continue
        bet = f["bet"]
        backing_a = bet["side"] == "A"
        market = found["a"] if backing_a else found["b"]
        price = kalshi_match.ask_price(market)
        if price is None:
            skipped.append({"match": f"{f['player_a']} v {f['player_b']}",
                            "reason": "no offer resting on that contract"})
            continue
        prob = f["model_prob_a"] if backing_a else 1.0 - f["model_prob_a"]
        sized = kalshi.size_position(prob, price, bankroll, maker=risk.MAKER,
                                     max_stake_pct=budget["ticket_pct"])
        if sized["contracts"] < 1:
            skipped.append({"match": f"{f['player_a']} v {f['player_b']}",
                            "reason": sized.get("reason") or "sized to nothing"})
            continue
        # A match already under way is not a ticket. The model priced it before
        # it began, and Kalshi keeps the market ACTIVE through the match, so
        # nothing about the market's own status would stop this.
        if kalshi_order.started(market):
            skipped.append({"match": f"{f['player_a']} v {f['player_b']}",
                            "reason": "the match has already started"})
            continue
        # Never ask for more than is actually offered. Sizing past the resting
        # ask either partially fills or walks up the book, and the EV was
        # computed at THIS price, not at whatever the next level costs.
        offered = kalshi_match.ask_size(market)
        capped = False
        if offered is not None and sized["contracts"] > int(offered):
            if int(offered) < 1:
                skipped.append({"match": f"{f['player_a']} v {f['player_b']}",
                                "reason": "nothing offered at the ask"})
                continue
            capped = True
            sized = kalshi.size_position(
                prob, price, bankroll, maker=risk.MAKER,
                max_stake_pct=min(budget["ticket_pct"],
                                  100.0 * int(offered) * price / max(bankroll, 1e-9)))
            if sized["contracts"] < 1:
                skipped.append({"match": f"{f['player_a']} v {f['player_b']}",
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
            "client_order_id": kalshi_order.new_client_order_id(),
        })

    tickets.sort(key=lambda t: -(t["ev"] or 0))
    return jsonify(_clean({
        "available": True, "tickets": tickets, "skipped": skipped,
        "budget": budget, "armed": kalshi_order.armed(),
        "live": kalshi.live_mode(),
    }))


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

    result = kalshi_order.create_order(ticker, count, price, coid)
    if result.get("error"):
        return jsonify({"ok": False, **result}), 400
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
    try:
        bal = kalshi.balance()
        print(f"[kalshi] OK - {mode}, balance ${bal.get('dollars')}", flush=True)
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
