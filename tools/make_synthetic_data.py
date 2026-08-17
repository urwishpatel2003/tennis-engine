"""
Generate a synthetic Sackmann-shaped tennis archive.

    python tools/make_synthetic_data.py --seasons 2018-2026 --players 220

Why this exists
---------------
Two jobs, both real:

1. **Pipeline validation without the network.** Every stage of the engine can be
   exercised end to end on data whose schema is byte-compatible with the real
   archive, so a broken join or a leaked future value shows up immediately.

2. **Ground-truth recovery.** Because the generator knows each player's TRUE
   latent serve/return skill and surface affinity, the backtest can be checked
   against a known answer: a correct engine should recover the true skill ordering
   and produce calibrated probabilities. That is a far stronger test than "the
   code ran".

Matches are simulated point by point through real scoring rules (deuce, tiebreaks,
best-of-5 at slams), so scorelines, service statistics and match durations all have
the right shape and inter-correlations.

The output is written to data/raw/ exactly like fetch_data.py, with a marker file
so nothing confuses synthetic data for the real archive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import RAW, SURFACES, TOUR_BASE_SPW, normalise_matches  # noqa: E402

# A plausible tour calendar: (name, level, surface, draw_size, best_of, week)
CALENDAR = [
    ("Brisbane", "A", "Hard", 32, 3, 1),
    ("Auckland", "A", "Hard", 32, 3, 2),
    ("Australian Open", "G", "Hard", 128, 5, 3),
    ("Rotterdam", "A", "Hard", 32, 3, 7),
    ("Rio de Janeiro", "A", "Clay", 32, 3, 8),
    ("Dubai", "A", "Hard", 32, 3, 9),
    ("Acapulco", "A", "Hard", 32, 3, 9),
    ("Indian Wells Masters", "M", "Hard", 64, 3, 11),
    ("Miami Masters", "M", "Hard", 64, 3, 13),
    ("Monte Carlo Masters", "M", "Clay", 64, 3, 16),
    ("Barcelona", "A", "Clay", 32, 3, 17),
    ("Madrid Masters", "M", "Clay", 64, 3, 19),
    ("Rome Masters", "M", "Clay", 64, 3, 20),
    ("Roland Garros", "G", "Clay", 128, 5, 22),
    ("Stuttgart", "A", "Grass", 32, 3, 25),
    ("Queen's Club", "A", "Grass", 32, 3, 25),
    ("Halle", "A", "Grass", 32, 3, 26),
    ("Wimbledon", "G", "Grass", 128, 5, 27),
    ("Hamburg", "A", "Clay", 32, 3, 30),
    ("Washington", "A", "Hard", 32, 3, 31),
    ("Canada Masters", "M", "Hard", 64, 3, 32),
    ("Cincinnati Masters", "M", "Hard", 64, 3, 33),
    ("US Open", "G", "Hard", 128, 5, 35),
    ("Tokyo", "A", "Hard", 32, 3, 40),
    ("Shanghai Masters", "M", "Hard", 64, 3, 41),
    ("Vienna", "A", "Hard", 32, 3, 43),
    ("Basel", "A", "Hard", 32, 3, 43),
    ("Paris Masters", "M", "Hard", 64, 3, 44),
    ("Tour Finals", "F", "Hard", 8, 3, 46),
    ("Bogota", "A", "Hard", 32, 3, 30),
    ("Gstaad", "A", "Clay", 32, 3, 29),
]

IOCS = ["ESP", "USA", "FRA", "ITA", "SRB", "GER", "ARG", "AUS", "GBR", "RUS",
        "SUI", "CAN", "JPN", "CRO", "POL", "GRE", "NOR", "AUT", "BEL", "NED"]

FIRST = ["Carlos", "Jannik", "Novak", "Daniil", "Andrey", "Stefanos", "Casper",
         "Taylor", "Hubert", "Alex", "Grigor", "Holger", "Tommy", "Frances",
         "Ben", "Sebastian", "Karen", "Alexander", "Felix", "Lorenzo", "Iga",
         "Aryna", "Coco", "Elena", "Jessica", "Ons", "Marketa", "Qinwen",
         "Maria", "Barbora", "Jelena", "Daria", "Beatriz", "Liudmila"]
LAST = ["Alvarez", "Bennett", "Castellan", "Duarte", "Eriksson", "Falk",
        "Garin", "Haas", "Ivanov", "Jansen", "Kovac", "Lindqvist", "Moreau",
        "Novak", "Olsen", "Petrov", "Quintero", "Rossi", "Sandberg", "Torres",
        "Ueda", "Vargas", "Wolff", "Xu", "Yilmaz", "Zeman", "Ackland",
        "Brandt", "Calvo", "Dvorak", "Engel", "Fontana", "Grabowski"]


def make_players(rng: np.random.Generator, n: int, tour: str, start_id: int) -> pd.DataFrame:
    """Latent skills: a serve/return talent pair plus per-surface affinities."""
    # Talent is normally distributed; the tour's depth means most players cluster.
    serve = rng.normal(0.0, 0.030, n)
    ret = rng.normal(0.0, 0.026, n)
    # Serve and return talent are mildly negatively correlated in reality
    # (big servers tend to be worse returners), so bake that in.
    ret = ret - 0.25 * serve

    rows = []
    for i in range(n):
        rows.append(
            {
                "player_id": start_id + i,
                "name_first": FIRST[rng.integers(len(FIRST))],
                "name_last": LAST[rng.integers(len(LAST))] + (
                    "" if i < len(LAST) else str(i // len(LAST))
                ),
                "hand": "L" if rng.random() < 0.13 else "R",
                "dob": pd.Timestamp("1990-01-01") + pd.Timedelta(days=int(rng.integers(0, 4000))),
                "ioc": IOCS[rng.integers(len(IOCS))],
                "height": float(np.clip(
                    rng.normal(185 if tour == "atp" else 173, 7), 155, 210
                )),
                "true_serve": serve[i],
                "true_return": ret[i],
                # Surface affinity: some players are genuine clay or grass specialists.
                **{
                    f"aff_{s}": float(rng.normal(0, 0.010))
                    for s in SURFACES
                },
                # Career arc: peak age and how fast they improve/decline.
                "peak_age": float(rng.normal(26.0, 2.5)),
                "arc": float(abs(rng.normal(0.0016, 0.0008))),
            }
        )
    df = pd.DataFrame(rows)
    df["name"] = df["name_first"] + " " + df["name_last"]
    # Make names unique so the resolver has an unambiguous target.
    dup = df["name"].duplicated(keep=False)
    df.loc[dup, "name"] = df.loc[dup, "name"] + " " + df.loc[dup, "player_id"].astype(str)
    df["name_last"] = df["name"].str.split(" ", n=1).str[1]
    return df


def _skill_at(row: pd.Series, surface: str, age: float) -> tuple[float, float]:
    """Latent serve/return excess for a player on a surface at a given age."""
    decline = -row["arc"] * (age - row["peak_age"]) ** 2
    s = row["true_serve"] + row[f"aff_{surface}"] + decline
    r = row["true_return"] + 0.5 * row[f"aff_{surface}"] + decline
    return s, r


def _simulate_game(rng: np.random.Generator, p: float) -> tuple[bool, int, int]:
    """Play one service game. Returns (server_won, points_played, points_won_by_server)."""
    a = b = 0
    won = pw = 0
    while True:
        won += 1
        if rng.random() < p:
            a += 1
            pw += 1
        else:
            b += 1
        if a >= 4 and a - b >= 2:
            return True, won, pw
        if b >= 4 and b - a >= 2:
            return False, won, pw


def _simulate_tiebreak(
    rng: np.random.Generator, pa: float, pb: float
) -> tuple[bool, list[int], list[int]]:
    """Returns (A won, [svpt_a, svpt_b], [spw_a, spw_b])."""
    a = b = 0
    svpt = [0, 0]
    spw = [0, 0]
    while True:
        i = a + b
        a_serves = ((i + 1) // 2) % 2 == 0
        p = pa if a_serves else pb
        srv = 0 if a_serves else 1
        svpt[srv] += 1
        server_won = rng.random() < p
        if server_won:
            spw[srv] += 1
        a_won_point = server_won if a_serves else not server_won
        if a_won_point:
            a += 1
        else:
            b += 1
        if a >= 7 and a - b >= 2:
            return True, svpt, spw
        if b >= 7 and b - a >= 2:
            return False, svpt, spw


def simulate_match(
    rng: np.random.Generator, pa: float, pb: float, best_of: int
) -> dict:
    """Point-by-point simulation through real scoring rules."""
    sets_to_win = 3 if best_of == 5 else 2
    sets_a = sets_b = 0
    set_scores: list[tuple[int, int]] = []
    svpt = [0, 0]
    spw = [0, 0]
    a_serves_first = rng.random() < 0.5

    while sets_a < sets_to_win and sets_b < sets_to_win:
        ga = gb = 0
        while True:
            if ga == 6 and gb == 6:
                a_won, tsv, tsp = _simulate_tiebreak(rng, pa, pb)
                svpt[0] += tsv[0]; svpt[1] += tsv[1]
                spw[0] += tsp[0]; spw[1] += tsp[1]
                ga, gb = (7, 6) if a_won else (6, 7)
                break
            a_serving = (((ga + gb) % 2) == 0) == a_serves_first
            p = pa if a_serving else pb
            srv = 0 if a_serving else 1
            server_won, pts, pts_won = _simulate_game(rng, p)
            svpt[srv] += pts
            spw[srv] += pts_won
            if (server_won and a_serving) or (not server_won and not a_serving):
                ga += 1
            else:
                gb += 1
            if ga >= 6 and ga - gb >= 2:
                break
            if gb >= 6 and gb - ga >= 2:
                break
            if ga == 7 or gb == 7:
                break

        set_scores.append((ga, gb))
        if ga > gb:
            sets_a += 1
        else:
            sets_b += 1
        # Serve rotation carries into the next set.
        if (ga + gb) % 2 == 1:
            a_serves_first = not a_serves_first

    return {
        "a_won": sets_a > sets_b,
        "set_scores": set_scores,
        "svpt": svpt,
        "spw": spw,
        "minutes": int(sum(svpt) * 0.62 + rng.normal(0, 8)),
    }


def _stat_line(rng: np.random.Generator, svpt: int, spw: int) -> dict:
    """Split a serve line into the Sackmann sub-statistics, keeping totals exact."""
    first_in = int(round(svpt * rng.uniform(0.58, 0.66)))
    second = svpt - first_in
    # First serves win at a much higher clip than seconds; split spw accordingly.
    share = min(max(rng.normal(0.72, 0.03), 0.55), 0.88)
    first_won = int(round(min(spw * share, first_in)))
    second_won = max(0, min(spw - first_won, second))
    aces = int(round(first_won * rng.uniform(0.10, 0.24)))
    dfs = int(round(second * rng.uniform(0.06, 0.14)))
    sv_gms = max(1, int(round(svpt / 6.4)))
    bp_faced = max(0, int(round(sv_gms * rng.uniform(0.15, 0.55))))
    bp_saved = int(round(bp_faced * rng.uniform(0.45, 0.75)))
    return {
        "ace": aces, "df": dfs, "svpt": svpt, "1stIn": first_in,
        "1stWon": first_won, "2ndWon": second_won, "SvGms": sv_gms,
        "bpSaved": bp_saved, "bpFaced": bp_faced,
    }


def generate_tour(
    rng: np.random.Generator, tour: str, seasons: list[int], n_players: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = make_players(rng, n_players, tour, 100_000 if tour == "atp" else 200_000)
    pmap = {int(r["player_id"]): r for _, r in players.iterrows()}
    base_tour = TOUR_BASE_SPW[tour]

    rows = []
    for season in seasons:
        # Entry ranking proxy = current latent strength; used to seed the draw.
        for tname, level, surface, draw, best_of, week in CALENDAR:
            # WTA plays best-of-3 everywhere.
            if tour == "wta":
                best_of = 3
            date = pd.Timestamp(f"{season}-01-01") + pd.Timedelta(weeks=week - 1)

            strength = {}
            for pid, row in pmap.items():
                age = (date - row["dob"]).days / 365.25
                s, r = _skill_at(row, surface, age)
                strength[pid] = s + r

            eligible = [pid for pid in pmap if 17 <= (date - pmap[pid]["dob"]).days / 365.25 <= 39]
            if len(eligible) < draw:
                continue
            # Better players are likelier to enter the bigger events.
            ranked = sorted(eligible, key=lambda p: -strength[p])
            pool_size = min(len(ranked), draw * 2)
            field = list(rng.choice(ranked[:pool_size], size=draw, replace=False))

            match_num = 0
            alive = field
            round_names = {128: "R128", 64: "R64", 32: "R32", 16: "R16",
                           8: "QF", 4: "SF", 2: "F"}
            while len(alive) > 1:
                rnd = round_names.get(len(alive), "RR")
                nxt = []
                for i in range(0, len(alive), 2):
                    pa_id, pb_id = int(alive[i]), int(alive[i + 1])
                    ra, rb = pmap[pa_id], pmap[pb_id]
                    age_a = (date - ra["dob"]).days / 365.25
                    age_b = (date - rb["dob"]).days / 365.25
                    sa, rra = _skill_at(ra, surface, age_a)
                    sb, rrb = _skill_at(rb, surface, age_b)

                    # Per-match form noise — nobody plays at their mean every day.
                    na, nb = rng.normal(0, 0.012), rng.normal(0, 0.012)
                    pa = float(np.clip(base_tour + sa - rrb + na, 0.40, 0.82))
                    pb = float(np.clip(base_tour + sb - rra + nb, 0.40, 0.82))

                    res = simulate_match(rng, pa, pb, best_of)
                    match_num += 1

                    if res["a_won"]:
                        w, l, wi, li = pa_id, pb_id, 0, 1
                        sets = res["set_scores"]
                    else:
                        w, l, wi, li = pb_id, pa_id, 1, 0
                        sets = [(b, a) for a, b in res["set_scores"]]

                    score = " ".join(f"{x}-{y}" for x, y in sets)
                    wr, lr = pmap[w], pmap[l]
                    ws = _stat_line(rng, res["svpt"][wi], res["spw"][wi])
                    ls = _stat_line(rng, res["svpt"][li], res["spw"][li])

                    rows.append(
                        {
                            "tourney_id": f"{season}-{tname[:4].upper()}",
                            "tourney_name": tname,
                            "surface": surface,
                            "draw_size": draw,
                            "tourney_level": level,
                            "tourney_date": int(date.strftime("%Y%m%d")),
                            "match_num": match_num,
                            "winner_id": w, "winner_name": wr["name"],
                            "winner_hand": wr["hand"], "winner_ht": wr["height"],
                            "winner_ioc": wr["ioc"],
                            "winner_age": round((date - wr["dob"]).days / 365.25, 1),
                            "winner_seed": np.nan, "winner_entry": np.nan,
                            "loser_id": l, "loser_name": lr["name"],
                            "loser_hand": lr["hand"], "loser_ht": lr["height"],
                            "loser_ioc": lr["ioc"],
                            "loser_age": round((date - lr["dob"]).days / 365.25, 1),
                            "loser_seed": np.nan, "loser_entry": np.nan,
                            "score": score, "best_of": best_of, "round": rnd,
                            "minutes": max(45, res["minutes"]),
                            **{f"w_{k}": v for k, v in ws.items()},
                            **{f"l_{k}": v for k, v in ls.items()},
                            "winner_rank": np.nan, "winner_rank_points": np.nan,
                            "loser_rank": np.nan, "loser_rank_points": np.nan,
                        }
                    )
                    nxt.append(w)
                alive = nxt

    raw = pd.DataFrame(rows)
    return normalise_matches(raw, tour), players


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic tennis data.")
    ap.add_argument("--seasons", default="2018-2026")
    ap.add_argument("--players", type=int, default=220)
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    lo, hi = (args.seasons.split("-") + [args.seasons])[:2]
    seasons = list(range(int(lo), int(hi) + 1))

    rng = np.random.default_rng(args.seed)
    for tour in args.tours:
        print(f"── generating {tour.upper()} {seasons[0]}–{seasons[-1]} …")
        matches, players = generate_tour(rng, tour, seasons, args.players)
        matches.to_parquet(RAW / f"matches_{tour}.parquet", index=False)

        keep = ["player_id", "name", "name_first", "name_last", "hand", "dob",
                "ioc", "height"]
        players[keep].to_parquet(RAW / f"players_{tour}.parquet", index=False)
        # The latent truth, for tools/validate_recovery.py.
        players.to_parquet(RAW / f"_truth_{tour}.parquet", index=False)

        print(f"   {len(matches):,} matches, {len(players)} players")

    (RAW / "SYNTHETIC.marker").write_text(
        "Data in this directory was generated by tools/make_synthetic_data.py.\n"
        "It is NOT real tennis data. Delete this file and re-run fetch_data.py\n"
        "to replace it with the real Sackmann archive.\n",
        encoding="utf-8",
    )
    print("\nWrote data/raw/SYNTHETIC.marker — this is not real data.")


if __name__ == "__main__":
    main()
