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
    return send_from_directory(HERE, "dashboard.html")


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
    d["available"] = True
    return jsonify(_clean(d))


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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(f"Tennis engine dashboard → http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=a.debug)
