"""
Schema contracts, shared constants and normalisation helpers for the tennis engine.

Everything downstream imports paths, surface/level vocabularies and the tour-level
baseline rates from here, so there is exactly one place to change when the upstream
(Jeff Sackmann tennis_atp / tennis_wta) schema drifts.

Inputs : none (pure constants + functions)
Outputs: PATHS, vocabularies, `normalise_matches()` which turns a raw Sackmann
         match frame into the engine's canonical long-form columns.

Canonical match schema produced by `normalise_matches()` (one row per match):
    tour, season, tourney_id, tourney_name, tourney_date, tourney_level,
    surface, indoor, best_of, round, round_ord, match_num, match_id,
    winner_id, winner_name, winner_hand, winner_ht, winner_ioc, winner_age,
    winner_rank, winner_rank_points,   (…and the loser_* mirror)
    score, minutes, retirement,
    w_ace w_df w_svpt w_1stIn w_1stWon w_2ndWon w_SvGms w_bpSaved w_bpFaced (+ l_*)
    w_spw, w_rpw, l_spw, l_rpw        (derived serve/return point rates)
    sets_w, sets_l, games_w, games_l, total_games, completed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths — always derived from this file so the repo stays relocatable.
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

for _d in (RAW, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Vocabularies
# ──────────────────────────────────────────────────────────────────────────────
TOURS = ("atp", "wta")

# Sackmann writes surface as Hard/Clay/Grass/Carpet (and blanks for old/odd events).
SURFACES = ("Hard", "Clay", "Grass", "Carpet")
SURFACE_FALLBACK = "Hard"  # ~55% of the modern tour; safest prior for a blank

# tourney_level codes. ATP: G=Grand Slam, M=Masters 1000, A=other tour (250/500),
# F=Tour Finals, D=Davis Cup, C=Challenger, S=Satellite/ITF.
# WTA: G=Grand Slam, P=Premier/WTA-1000, PM=Premier Mandatory, I=International/250,
#      D=Fed Cup, T1..T4 legacy tiers.
LEVEL_WEIGHT = {
    # How much a result at this level should move a rating, relative to a tour event.
    # Bigger stages = more motivation, deeper fields, more reliable signal.
    "G": 1.10,   # Grand Slam — best-of-5 (ATP), full fields
    "F": 1.05,   # Tour Finals — elite round robin
    "M": 1.05,   # Masters 1000
    "PM": 1.05,  # WTA Premier Mandatory
    "P": 1.00,   # WTA Premier / 1000
    "A": 1.00,   # ATP 250/500 — the reference level
    "I": 1.00,   # WTA International / 250
    "D": 0.85,   # Davis/Fed Cup — team format, patchy motivation, odd surfaces
    "O": 0.85,   # Olympics
    "C": 0.80,   # Challenger — weaker fields, noisier stat lines
    "S": 0.70,   # Satellite / ITF
}
DEFAULT_LEVEL_WEIGHT = 1.00

# Round ordering — used for recency/importance and for the dashboard's draw view.
ROUND_ORDER = ["RR", "BR", "R128", "R64", "R32", "R16", "QF", "SF", "F"]
ROUND_ORD = {r: i for i, r in enumerate(ROUND_ORDER)}

# ──────────────────────────────────────────────────────────────────────────────
# Tour-level baselines
# ──────────────────────────────────────────────────────────────────────────────
# Average share of points won by the SERVER, by tour. These anchor the additive
# serve/return model in engine/serve_return.py: a player's rating is expressed as a
# deviation from this baseline, so the two tours never get mixed on one scale.
# (Men hold far more easily than women — ~0.64 vs ~0.565 of service points won.)
TOUR_BASE_SPW = {"atp": 0.640, "wta": 0.565}

# Surface shifts the server's edge: grass is fastest (serve dominates), clay slowest.
# Added to TOUR_BASE_SPW to get the (tour, surface) baseline.
SURFACE_SPW_SHIFT = {
    "Grass": +0.020,
    "Hard": 0.000,
    "Carpet": +0.015,
    "Clay": -0.020,
}

# Elo scale. 400 is the classic decade; tennis Elo conventionally keeps it.
ELO_SCALE = 400.0
ELO_INIT = 1500.0

# ──────────────────────────────────────────────────────────────────────────────
# Column contracts
# ──────────────────────────────────────────────────────────────────────────────
SERVE_STAT_COLS = [
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced",
]

RAW_MATCH_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    *[f"w_{c}" for c in SERVE_STAT_COLS],
    *[f"l_{c}" for c in SERVE_STAT_COLS],
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

NUMERIC_MATCH_COLS = [
    "draw_size", "match_num", "winner_id", "loser_id", "winner_ht", "loser_ht",
    "winner_age", "loser_age", "best_of", "minutes",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
    # Seeds MUST be coerced. Upstream writes them inconsistently — some seasons
    # as integers, others as strings — so concatenating years leaves a mixed
    # object column that pyarrow refuses to write ("Could not convert '5' with
    # type str: tried to convert to double"). Real data, not the fixtures, found
    # this one.
    "winner_seed", "loser_seed",
    *[f"w_{c}" for c in SERVE_STAT_COLS],
    *[f"l_{c}" for c in SERVE_STAT_COLS],
]

# Free-text columns that must be forced to string for the same reason: entry
# codes are 'WC'/'Q'/'LL'/'PR'/'SE' but read as NaN-heavy object columns, and a
# season where every value is blank comes back as float64.
STRING_MATCH_COLS = [
    "tourney_id", "tourney_name", "tourney_level", "surface", "round", "score",
    "winner_entry", "loser_entry", "winner_name", "loser_name",
    "winner_hand", "loser_hand", "winner_ioc", "loser_ioc",
]


# ──────────────────────────────────────────────────────────────────────────────
# Indoor detection
# ──────────────────────────────────────────────────────────────────────────────
# Sackmann has no indoor flag. These substrings catch the recurring indoor stops;
# everything else is treated as outdoor. Indoor removes wind/sun and speeds the
# court up slightly, which is a real (small) serve-side effect.
_INDOOR_HINTS = (
    "indoor", "paris masters", "bercy", "rolex paris", "tour finals",
    "masters cup", "vienna", "basel", "stockholm", "metz", "montpellier",
    "marseille", "rotterdam", "antwerp", "sofia", "st. petersburg",
    "st petersburg", "moscow", "zhuhai", "turin", "london finals",
    "next gen", "davis cup finals", "billie jean king cup finals",
    "linz", "ostrava", "luxembourg", "cluj", "transylvania",
)

# Notably high-altitude venues: thinner air = faster ball, serve gets a real boost.
# metres above sea level.
ALTITUDE_M = {
    "bogota": 2640, "quito": 2850, "la paz": 3640, "mexico city": 2240,
    "gstaad": 1050, "kitzbuhel": 760, "kitzbühel": 760, "madrid": 660,
    "denver": 1600, "johannesburg": 1750, "guadalajara": 1560,
    "sao paulo": 760, "são paulo": 760, "cali": 1000, "medellin": 1500,
    "medellín": 1500, "bastad": 20, "iasi": 100,
}


def is_indoor(tourney_name: object) -> bool:
    """Best-effort indoor flag from the tournament name."""
    if not isinstance(tourney_name, str):
        return False
    low = tourney_name.lower()
    return any(h in low for h in _INDOOR_HINTS)


def altitude_m(tourney_name: object) -> float:
    """Best-effort venue altitude in metres (0.0 when unknown / sea level)."""
    if not isinstance(tourney_name, str):
        return 0.0
    low = tourney_name.lower()
    for city, alt in ALTITUDE_M.items():
        if city in low:
            return float(alt)
    return 0.0


def normalise_surface(s: object) -> str:
    """Map a raw surface string onto SURFACES, defaulting to Hard."""
    if not isinstance(s, str) or not s.strip():
        return SURFACE_FALLBACK
    s = s.strip().title()
    return s if s in SURFACES else SURFACE_FALLBACK


def level_weight(level: object) -> float:
    if not isinstance(level, str):
        return DEFAULT_LEVEL_WEIGHT
    return LEVEL_WEIGHT.get(level.strip(), DEFAULT_LEVEL_WEIGHT)


def base_spw(tour: str, surface: str) -> float:
    """Baseline share of service points won for a (tour, surface)."""
    return TOUR_BASE_SPW.get(tour, 0.62) + SURFACE_SPW_SHIFT.get(surface, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Score-string parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_score(score: object) -> dict:
    """
    Turn a Sackmann score string into set/game counts from the WINNER's perspective.

    Handles the messy real-world forms: '6-4 7-6(3)', '7-6(5) 6-7(4) 6-3',
    'W/O', '6-2 RET', '6-4 2-1 RET', 'DEF', 'ABN', and stray unicode dashes.

    Returns dict with sets_w, sets_l, games_w, games_l, completed, retirement.
    `completed` is False for walkovers/retirements/defaults — those rows must be
    excluded from any serve/return or score-distribution fitting, because the
    scoreline does not reflect a finished match.
    """
    out = {
        "sets_w": 0, "sets_l": 0, "games_w": 0, "games_l": 0,
        "completed": False, "retirement": False,
    }
    if not isinstance(score, str) or not score.strip():
        return out

    s = score.replace("–", "-").replace("—", "-").upper().strip()

    # Non-matches: walkover / default / abandoned carry no playable score.
    if any(tag in s for tag in ("W/O", "WO ", "DEF", "ABN", "UNFINISHED")) or s in {"WO", "W/O"}:
        return out

    if "RET" in s:
        out["retirement"] = True
        s = s.replace("RET", " ")

    for token in s.split():
        token = token.strip()
        if not token or "-" not in token:
            continue
        # strip the tiebreak detail: '7-6(3)' -> '7-6'
        token = token.split("(")[0]
        parts = token.split("-")
        if len(parts) != 2:
            continue
        try:
            gw, gl = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        # sanity: no legitimate set has a double-digit game count above 20
        if not (0 <= gw <= 30 and 0 <= gl <= 30):
            continue
        out["games_w"] += gw
        out["games_l"] += gl
        if gw > gl:
            out["sets_w"] += 1
        elif gl > gw:
            out["sets_l"] += 1

    # A completed match is one the winner actually finished by winning enough sets.
    out["completed"] = (not out["retirement"]) and out["sets_w"] > out["sets_l"] and out["sets_w"] >= 2
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Frame normalisation
# ──────────────────────────────────────────────────────────────────────────────
def normalise_matches(df: pd.DataFrame, tour: str) -> pd.DataFrame:
    """
    Coerce a raw Sackmann match frame into the engine's canonical schema.

    Defensive by design: upstream drops/renames columns between seasons (older
    seasons have no serve stats at all), so every expected column is created if
    missing rather than raising.
    """
    df = df.copy()

    for col in RAW_MATCH_COLS:
        if col not in df.columns:
            df[col] = np.nan

    for col in NUMERIC_MATCH_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in STRING_MATCH_COLS:
        df[col] = df[col].astype("string")

    df["tour"] = tour
    df["surface"] = df["surface"].map(normalise_surface)

    # tourney_date is YYYYMMDD as an int.
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype("string").str.slice(0, 8),
        format="%Y%m%d", errors="coerce",
    )
    df["season"] = df["tourney_date"].dt.year

    df["indoor"] = df["tourney_name"].map(is_indoor)
    df["altitude"] = df["tourney_name"].map(altitude_m)
    df["level_w"] = df["tourney_level"].map(level_weight)
    df["round_ord"] = df["round"].map(ROUND_ORD).astype("Float64")

    df["best_of"] = df["best_of"].fillna(3).clip(3, 5).astype("Int64")

    # Parsed scoreline
    parsed = pd.DataFrame([parse_score(s) for s in df["score"]], index=df.index)
    df = pd.concat([df, parsed], axis=1)
    df["total_games"] = df["games_w"] + df["games_l"]

    # Derived point rates. svpt = service points played, so:
    #   spw = (1stWon + 2ndWon) / svpt
    #   rpw = 1 - opponent's spw
    for me, opp in (("w", "l"), ("l", "w")):
        won = df[f"{me}_1stWon"] + df[f"{me}_2ndWon"]
        svpt = df[f"{me}_svpt"]
        df[f"{me}_spw"] = np.where(svpt > 0, won / svpt, np.nan)
    df["w_rpw"] = 1.0 - df["l_spw"]
    df["l_rpw"] = 1.0 - df["w_spw"]

    # Stable unique key. tourney_id already encodes the season.
    df["match_id"] = (
        df["tour"].astype(str) + "-"
        + df["tourney_id"].astype(str) + "-"
        + df["match_num"].astype("Int64").astype(str)
    )

    df = df.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    df["winner_id"] = df["winner_id"].astype("int64")
    df["loser_id"] = df["loser_id"].astype("int64")

    return df.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Explode one-row-per-match into two-rows-per-match (one per player).

    Produces player-centric columns: player_id, player_name, opp_id, opp_name,
    won, spw, rpw, plus the shared match context. This is the shape the rating
    and serve/return builders want.
    """
    shared = [
        "tour", "season", "tourney_id", "tourney_name", "tourney_date",
        "tourney_level", "level_w", "surface", "indoor", "altitude",
        "best_of", "round", "round_ord", "match_id", "minutes",
        "total_games", "completed", "retirement", "score",
    ]
    frames = []
    for me, opp, won in (("winner", "loser", True), ("loser", "winner", False)):
        pre = "w" if me == "winner" else "l"
        opre = "l" if me == "winner" else "w"
        out = matches[shared].copy()
        out["player_id"] = matches[f"{me}_id"]
        out["player_name"] = matches[f"{me}_name"]
        out["player_hand"] = matches[f"{me}_hand"]
        out["player_ht"] = matches[f"{me}_ht"]
        out["player_ioc"] = matches[f"{me}_ioc"]
        out["player_age"] = matches[f"{me}_age"]
        out["player_rank"] = matches[f"{me}_rank"]
        out["player_rank_points"] = matches[f"{me}_rank_points"]
        out["opp_id"] = matches[f"{opp}_id"]
        out["opp_name"] = matches[f"{opp}_name"]
        out["opp_hand"] = matches[f"{opp}_hand"]
        out["opp_rank"] = matches[f"{opp}_rank"]
        out["won"] = won
        out["spw"] = matches[f"{pre}_spw"]
        out["rpw"] = matches[f"{pre}_rpw"]
        out["svpt"] = matches[f"{pre}_svpt"]
        out["rtpt"] = matches[f"{opre}_svpt"]  # return points played = opp service points
        out["ace"] = matches[f"{pre}_ace"]
        out["df"] = matches[f"{pre}_df"]
        out["bpSaved"] = matches[f"{pre}_bpSaved"]
        out["bpFaced"] = matches[f"{pre}_bpFaced"]
        out["SvGms"] = matches[f"{pre}_SvGms"]
        out["games_for"] = matches["games_w"] if me == "winner" else matches["games_l"]
        out["games_against"] = matches["games_l"] if me == "winner" else matches["games_w"]
        out["sets_for"] = matches["sets_w"] if me == "winner" else matches["sets_l"]
        out["sets_against"] = matches["sets_l"] if me == "winner" else matches["sets_w"]
        frames.append(out)

    long = pd.concat(frames, ignore_index=True)
    long["player_id"] = long["player_id"].astype("int64")
    long["opp_id"] = long["opp_id"].astype("int64")
    return long.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)
