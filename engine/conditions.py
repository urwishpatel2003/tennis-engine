"""
Match conditions and player state — fatigue, rest, venue, home crowd.

Inputs : data/raw/matches_{tour}.parquet
Outputs: data/processed/conditions.parquet — one row per (match, player) with the
         player's physical state entering that match, plus the venue context.

Everything here is strictly backward-looking (rolling windows over matches already
played), so it inherits the same no-leakage guarantee as ratings.py.

Feature notes
-------------
* `days_rest`      — days since the player's previous match. The tennis calendar
                     makes this genuinely variable: a qualifier may play five days
                     running while a seed rests four.
* `minutes_7d/14d` — court time, the honest fatigue measure. Three consecutive
                     five-setters is a different load from three straight-set wins,
                     and match COUNT cannot see the difference.
* `matches_event`  — matches already played at this event. Deep runs accumulate.
* `is_home`        — player's IOC matches the tournament's country.
* `altitude`/`indoor` — venue physics: thin air and still air both speed the ball
                     up, which shifts the serve/return balance (applied in predict.py).

All fatigue effects are deliberately modest. The literature (and our own backtests)
finds real but small effects — a tired player is maybe 2-4% less likely to win, not
20%. Over-weighting fatigue is a classic way to make a tennis model worse.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow both `python engine/conditions.py` and `python -m engine.conditions`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import PROCESSED, RAW, TOURS, to_long

# ──────────────────────────────────────────────────────────────────────────────
# Venue → country (IOC code), for the home-crowd flag.
# Sackmann gives no country column, so this maps the recurring tour stops by name.
# Substring match, lowercase. Unlisted events simply get no home flag.
# ──────────────────────────────────────────────────────────────────────────────
VENUE_COUNTRY = {
    # Grand Slams
    "australian open": "AUS", "roland garros": "FRA", "french open": "FRA",
    "wimbledon": "GBR", "us open": "USA",
    # Masters / 1000
    "indian wells": "USA", "miami": "USA", "monte carlo": "MON", "madrid": "ESP",
    "rome": "ITA", "canada": "CAN", "toronto": "CAN", "montreal": "CAN",
    "cincinnati": "USA", "shanghai": "CHN", "paris": "FRA", "bercy": "FRA",
    # Regular tour stops
    "doha": "QAT", "dubai": "UAE", "acapulco": "MEX", "rio": "BRA",
    "buenos aires": "ARG", "santiago": "CHI", "cordoba": "ARG", "rio de janeiro": "BRA",
    "barcelona": "ESP", "estoril": "POR", "munich": "GER", "hamburg": "GER",
    "stuttgart": "GER", "halle": "GER", "queen": "GBR", "eastbourne": "GBR",
    "s-hertogenbosch": "NED", "rotterdam": "NED", "amsterdam": "NED",
    "gstaad": "SUI", "basel": "SUI", "geneva": "SUI", "kitzbuhel": "AUT",
    "kitzbühel": "AUT", "vienna": "AUT", "umag": "CRO", "zagreb": "CRO",
    "bastad": "SWE", "stockholm": "SWE", "winston-salem": "USA",
    "atlanta": "USA", "washington": "USA", "newport": "USA", "delray beach": "USA",
    "san diego": "USA", "houston": "USA", "dallas": "USA", "san jose": "USA",
    "tokyo": "JPN", "osaka": "JPN", "beijing": "CHN", "zhuhai": "CHN",
    "chengdu": "CHN", "hangzhou": "CHN", "wuhan": "CHN", "guangzhou": "CHN",
    "seoul": "KOR", "sydney": "AUS", "brisbane": "AUS", "adelaide": "AUS",
    "melbourne": "AUS", "perth": "AUS", "auckland": "NZL", "chennai": "IND",
    "pune": "IND", "marseille": "FRA", "montpellier": "FRA", "metz": "FRA",
    "lyon": "FRA", "antwerp": "BEL", "brussels": "BEL", "sofia": "BUL",
    "bucharest": "ROU", "cluj": "ROU", "iasi": "ROU", "budapest": "HUN",
    "moscow": "RUS", "st. petersburg": "RUS", "st petersburg": "RUS",
    "istanbul": "TUR", "antalya": "TUR", "bogota": "COL", "quito": "ECU",
    "lima": "PER", "sao paulo": "BRA", "são paulo": "BRA", "los cabos": "MEX",
    "monterrey": "MEX", "guadalajara": "MEX", "turin": "ITA", "florence": "ITA",
    "naples": "ITA", "sardinia": "ITA", "palermo": "ITA", "parma": "ITA",
    "mallorca": "ESP", "marbella": "ESP", "valencia": "ESP", "seville": "ESP",
    "prague": "CZE", "ostrava": "CZE", "linz": "AUT", "luxembourg": "LUX",
    "lausanne": "SUI", "warsaw": "POL", "tenerife": "ESP", "almaty": "KAZ",
    "astana": "KAZ", "nur-sultan": "KAZ", "jeddah": "KSA", "riyadh": "KSA",
}

# ──────────────────────────────────────────────────────────────────────────────
# Tuned constants
# ──────────────────────────────────────────────────────────────────────────────
# Typical match durations, used to impute `minutes` when Sackmann has none
# (common before ~1990 and for some smaller events).
DEFAULT_MINUTES = {3: 95.0, 5: 165.0}

FATIGUE_WINDOWS = (7, 14, 28)  # days

# What counts as "heavy" recent load, for the normalised fatigue index.
# ~6 hours of tennis in a week is a deep run at a 1000-level event.
HEAVY_MINUTES_7D = 360.0
HEAVY_MINUTES_14D = 660.0


def _minutes_or_default(minutes: float, best_of: int) -> float:
    if np.isfinite(minutes) and minutes > 0:
        return float(minutes)
    return DEFAULT_MINUTES.get(int(best_of) if np.isfinite(best_of) else 3, 95.0)


def venue_country(tourney_name: object) -> str | None:
    if not isinstance(tourney_name, str):
        return None
    low = tourney_name.lower()
    for city, ioc in VENUE_COUNTRY.items():
        if city in low:
            return ioc
    return None


def build_conditions(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Per (match, player) physical state + venue context, computed walk-forward.

    One chronological pass with a per-player deque of (date, minutes) keeps this
    O(n) rather than the O(n²) a per-row date-window filter would cost.
    """
    long = to_long(matches)
    long = long.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)

    history: dict[int, deque] = defaultdict(deque)   # pid -> (date, minutes)
    last_played: dict[int, pd.Timestamp] = {}
    event_matches: dict[tuple[int, str], int] = defaultdict(int)
    event_minutes: dict[tuple[int, str], float] = defaultdict(float)

    rows = []
    for r in long.itertuples(index=False):
        pid = int(r.player_id)
        when = r.tourney_date
        mins = _minutes_or_default(r.minutes, r.best_of)

        # ── Prune the window and read state BEFORE this match ──────────────────
        dq = history[pid]
        while dq and (when - dq[0][0]).days > max(FATIGUE_WINDOWS):
            dq.popleft()

        windowed = {w: 0.0 for w in FATIGUE_WINDOWS}
        counts = {w: 0 for w in FATIGUE_WINDOWS}
        for d, m in dq:
            gap = (when - d).days
            for w in FATIGUE_WINDOWS:
                if gap <= w:
                    windowed[w] += m
                    counts[w] += 1

        prev = last_played.get(pid)
        days_rest = (when - prev).days if prev is not None else np.nan

        ev_key = (pid, str(r.tourney_id))
        played_here = event_matches[ev_key]
        minutes_here = event_minutes[ev_key]

        home_ioc = venue_country(r.tourney_name)
        is_home = bool(home_ioc and isinstance(r.player_ioc, str) and r.player_ioc == home_ioc)

        # Normalised 0-1 fatigue index; 1.0 = a very heavy recent load.
        fatigue = float(
            np.clip(
                0.6 * (windowed[7] / HEAVY_MINUTES_7D)
                + 0.4 * (windowed[14] / HEAVY_MINUTES_14D),
                0.0, 1.5,
            )
        )

        rows.append(
            {
                "match_id": r.match_id, "tour": r.tour, "season": r.season,
                "tourney_date": when, "tourney_id": r.tourney_id,
                "tourney_name": r.tourney_name, "surface": r.surface,
                "indoor": bool(r.indoor), "altitude": float(r.altitude),
                "best_of": int(r.best_of) if np.isfinite(r.best_of) else 3,
                "player_id": pid, "opp_id": int(r.opp_id), "won": bool(r.won),
                "days_rest": days_rest,
                "matches_7d": counts[7], "matches_14d": counts[14], "matches_28d": counts[28],
                "minutes_7d": windowed[7], "minutes_14d": windowed[14],
                "minutes_28d": windowed[28],
                "matches_this_event": played_here,
                "minutes_this_event": minutes_here,
                "fatigue_index": fatigue,
                "is_home": is_home,
                "venue_country": home_ioc,
            }
        )

        # ── Update ────────────────────────────────────────────────────────────
        dq.append((when, mins))
        last_played[pid] = when
        event_matches[ev_key] = played_here + 1
        event_minutes[ev_key] = minutes_here + mins

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction-time adjustments
# ──────────────────────────────────────────────────────────────────────────────
# All values are in Elo points, applied to the rating gap before the win
# probability is formed. 25 Elo ≈ 3.5 percentage points at even money — the right
# order of magnitude for a conditions effect.

# MEASURED, not reasoned. tools/validate_adjustments.py scores each of these
# against 34.5k matches with a walk-forward holdout. The first version of this
# file was hand-tuned and two of its four terms were actively making the model
# WORSE — the whole conditions block cost 0.0016 log loss.
#
# What the measurement found:
#
#   fatigue      best multiplier -1.8   BACKWARDS
#   short rest   best multiplier -2.0   BACKWARDS
#   layoff       best multiplier +4.0   real, but 4x under-weighted
#   home         best multiplier  0.8   about right, negligible
#
# Fatigue and short rest are backwards because they are not measuring tiredness
# at all — they are measuring HAVING BEEN WINNING. Court time accumulates by
# advancing through a draw, and playing on consecutive days means you keep
# progressing. When player A is much fresher than B, A wins only 44.4% of the
# time. Penalising the tired player was penalising form.
#
# The honest fix is not to invert them — "tired players win" is causally wrong and
# would mislead on a genuinely exhausted player. It is to drop the confounded
# proxy and use the real variable underneath it: how far into this event each
# player has already played. That IS in-tournament form the ratings have not yet
# absorbed, and it is worth ~80 Elo per match already won.
FATIGUE_ELO_PER_INDEX = 0.0     # retired: measured backwards (see above)
SHORT_REST_PENALTY = 0.0        # retired: same confound

# Rust is real and was badly under-weighted. -30 measured at x4.
LONG_LAYOFF_PENALTY = -120.0

# Elo per match already won at THIS event, applied as a difference between the
# two players. Chosen from the flat region (60-100) of the holdout curve; a clean
# train-only fit gave 100, which still held up out of sample. Fires on 23% of
# matches — those where the two players have taken different paths into the round,
# which is mostly seeds with byes against players who had to qualify through.
EVENT_PROGRESS_ELO = 80.0

HOME_ELO_BONUS = 18.0           # measured at x0.8 — near enough, and it is small
ALTITUDE_SERVE_BONUS = 0.010    # +1.0pt of service points won at ~2000m+


def conditions_elo_delta(state: dict) -> tuple[float, dict]:
    """
    Convert one player's condition state into an Elo adjustment.

    Returns (delta, breakdown) so the dashboard can show *why* a player was
    marked down, not just that they were.
    """
    parts: dict[str, float] = {}

    # In-tournament progress: matches already won at THIS event. Replaces the
    # fatigue and short-rest terms, which measured the same thing with the wrong
    # sign. Absent for a fixture at an event we have no results for yet, in which
    # case it simply does not fire.
    played = state.get("matches_this_event")
    if played is not None and np.isfinite(played) and played > 0:
        parts["event_progress"] = EVENT_PROGRESS_ELO * float(played)

    rest = state.get("days_rest")
    if rest is not None and np.isfinite(rest):
        if rest >= 60:
            # Rust is real but self-limiting; scale it in rather than a cliff.
            parts["layoff"] = LONG_LAYOFF_PENALTY * min((rest - 60) / 120.0 + 0.5, 1.0)

    if state.get("is_home"):
        parts["home"] = HOME_ELO_BONUS

    return float(sum(parts.values())), parts


def altitude_serve_delta(altitude_m: float) -> float:
    """
    Extra share of service points won from thin air.

    Ramps linearly from 500m and saturates at 2500m. Bogota (2640m) is the extreme
    case and plays roughly a full point of service-points-won faster than sea level.
    """
    if not np.isfinite(altitude_m) or altitude_m <= 500:
        return 0.0
    return ALTITUDE_SERVE_BONUS * min((altitude_m - 500.0) / 2000.0, 1.0)


INDOOR_SERVE_BONUS = 0.004  # no wind or sun: marginally easier to hold


def build_all(tours: tuple[str, ...] = TOURS) -> pd.DataFrame:
    out = []
    for tour in tours:
        path = RAW / f"matches_{tour}.parquet"
        if not path.exists():
            print(f"  [conditions] no {path.name}, skipping {tour}")
            continue
        m = pd.read_parquet(path)
        print(f"  [conditions] {tour}: {len(m):,} matches")
        out.append(build_conditions(m))
    if not out:
        raise FileNotFoundError(
            f"No match parquets found in {RAW}. Run `python fetch_data.py` first."
        )
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build match conditions / fatigue table.")
    ap.add_argument("--tours", nargs="+", default=list(TOURS))
    args = ap.parse_args()

    df = build_all(tuple(args.tours))
    df.to_parquet(PROCESSED / "conditions.parquet", index=False)
    print(f"  [conditions] wrote {len(df):,} player-match rows")


if __name__ == "__main__":
    main()
