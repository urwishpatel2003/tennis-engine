"""
Parsing tests against the REAL Sackmann file formats.

    python tests/test_fetch_parsing.py

Why this exists: the development machine's network blocks GitHub's raw hosts, so
`fetch_data.py` has never been run end to end against the live archive. The
download is untestable here, but the *parsing* is not — and parsing is where the
bugs would be. These fixtures reproduce the upstream column order, header quirks
and messy real-world values byte for byte, so a schema mistake fails here rather
than halfway through a Railway build.

Covered:
  * exact atp_matches_YYYY.csv column order (49 columns)
  * headerless players/rankings files (older releases shipped without a header)
  * walkovers, retirements, defaults, abandoned matches
  * blank surfaces, missing heights, missing serve stats, missing ranks
  * unicode names and accented characters
  * best-of-5 and best-of-3, tiebreak scorelines
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import normalise_matches, parse_score, to_long  # noqa: E402
from fetch_data import PLAYER_COLS, RANKING_COLS, _read_csv  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# The exact upstream header, in order.
MATCH_HEADER = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,match_num,"
    "winner_id,winner_seed,winner_entry,winner_name,winner_hand,winner_ht,winner_ioc,"
    "winner_age,loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,"
    "loser_ioc,loser_age,score,best_of,round,minutes,"
    "w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,"
    "l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced,"
    "winner_rank,winner_rank_points,loser_rank,loser_rank_points"
)

MATCH_ROWS = [
    # normal best-of-5 slam win with full stats
    "2024-520,Roland Garros,Clay,128,G,20240526,101,207989,3,,Carlos Alcaraz,R,183,ESP,"
    "21.1,106421,,,Novak Djokovic,R,188,SRB,37.0,6-3 7-6(4) 6-4,5,SF,168,"
    "12,3,110,68,52,23,17,4,6,8,5,105,60,41,22,17,7,11,3,9800,1,9900",
    # tiebreak-heavy grass match, best of 3
    "2024-540,Wimbledon,Grass,128,G,20240701,55,104925,,,Jannik Sinner,R,191,ITA,"
    "22.9,105777,7,,Grigor Dimitrov,R,190,BUL,33.0,7-6(5) 6-7(3) 7-6(9),3,QF,190,"
    "22,4,140,90,70,25,20,8,10,18,6,138,85,66,26,20,9,12,2,8700,10,3200",
    # RETIREMENT — real result, unusable scoreline
    "2024-419,Miami Masters,Hard,96,M,20240320,30,126094,,,Ben Shelton,L,193,USA,"
    "21.4,100644,,Q,Alexander Zverev,R,198,GER,26.9,6-4 2-1 RET,3,R32,62,"
    "8,2,55,33,26,11,9,2,3,5,3,48,29,20,10,8,4,6,15,2100,5,4900",
    # WALKOVER — no play at all
    "2024-407,Indian Wells Masters,Hard,96,M,20240306,12,111815,,,Denis Shapovalov,L,"
    "185,CAN,25.0,105453,,WC,Kei Nishikori,R,178,JPN,34.2,W/O,3,R64,,"
    ",,,,,,,,,,,,,,,,,,120,540,,",
    # blank surface, missing height, missing stats, missing ranks (old/small event)
    "2024-M020,Davis Cup F,,4,D,20240210,3,200000,,,Aslan Karatsev,R,,RUS,30.5,"
    "200001,,,Marc-Andrea Huesler,L,,SUI,27.8,7-5 4-6 6-3,3,RR,,"
    ",,,,,,,,,,,,,,,,,,,,,",
    # unicode / accented names
    "2024-580,US Open,Hard,128,G,20240826,7,106233,,,Félix Auger-Aliassime,R,193,CAN,"
    "24.0,105138,,,Botic van de Zandschulp,R,188,NED,28.7,6-2 6-4 3-6 7-5,5,R16,201,"
    "14,5,120,74,58,20,19,6,8,9,7,118,70,50,24,18,5,9,20,2400,60,900",
    # DEF (default) — like a walkover
    "2024-321,Vienna,Hard,32,A,20241021,9,200002,,,Player Aa,R,185,AUT,23.0,"
    "200003,,,Player Bb,R,180,POL,25.0,DEF,3,R32,,,,,,,,,,,,,,,,,,,,80,700,90,650",
]


print("\n1. score parsing")
cases = [
    ("6-3 7-6(4) 6-4", 3, 0, 19, 13, True, False),
    ("7-6(5) 6-7(3) 7-6(9)", 2, 1, 20, 19, True, False),
    ("6-4 2-1 RET", 2, 0, 8, 5, False, True),
    ("W/O", 0, 0, 0, 0, False, False),
    ("DEF", 0, 0, 0, 0, False, False),
    ("6-2 6-4 3-6 7-5", 3, 1, 22, 17, True, False),
    ("", 0, 0, 0, 0, False, False),
]
for score, sw, sl, gw, gl, completed, ret in cases:
    r = parse_score(score)
    ok = (r["sets_w"] == sw and r["sets_l"] == sl and r["games_w"] == gw
          and r["games_l"] == gl and r["completed"] == completed
          and r["retirement"] == ret)
    check(f"{score or '(blank)':<22} -> {sw}-{sl} sets, {gw}-{gl} games",
          ok, f"got {r}")

# Unicode dash, sometimes present in scraped variants.
check("en-dash score parses", parse_score("6–4 6–3")["sets_w"] == 2)

print("\n2. real-format match CSV")
csv = MATCH_HEADER + "\n" + "\n".join(MATCH_ROWS) + "\n"
raw = pd.read_csv(io.StringIO(csv), low_memory=False)
check("upstream header has 49 columns", len(MATCH_HEADER.split(",")) == 49,
      f"{len(MATCH_HEADER.split(','))}")
check("all rows read", len(raw) == len(MATCH_ROWS), f"{len(raw)}")

m = normalise_matches(raw, "atp")
check("normalises without error", len(m) == len(MATCH_ROWS), f"{len(m)}")
check("dates parsed", m["tourney_date"].notna().all())
check("seasons derived", (m["season"] == 2024).all())
check("blank surface defaults to Hard",
      m.loc[m["tourney_name"] == "Davis Cup F", "surface"].iloc[0] == "Hard")
check("Roland Garros is Clay",
      m.loc[m["tourney_name"] == "Roland Garros", "surface"].iloc[0] == "Clay")
check("best_of preserved",
      set(m["best_of"].dropna().unique()) == {3, 5}, f"{m['best_of'].unique()}")

wo = m[m["score"] == "W/O"].iloc[0]
check("walkover: not completed, not retirement",
      (not wo["completed"]) and (not wo["retirement"]))
check("walkover: zero games", wo["total_games"] == 0)
dfl = m[m["score"] == "DEF"].iloc[0]
check("default: not completed", not dfl["completed"])
ret = m[m["score"].str.contains("RET")].iloc[0]
check("retirement flagged", ret["retirement"] and not ret["completed"])

print("\n3. derived serve/return rates")
full = m[m["w_svpt"].notna() & (m["w_svpt"] > 0)]
check("spw computed where stats exist", full["w_spw"].notna().all())
check("spw in (0,1)", full["w_spw"].between(0, 1).all())
check("rpw is 1 - opponent spw",
      ((full["w_rpw"] - (1 - full["l_spw"])).abs() < 1e-12).all())
sparse = m[m["w_svpt"].isna()]
check("no stats -> NaN spw, not a crash", sparse["w_spw"].isna().all())

print("\n4. unicode survives")
uni = m[m["winner_name"].astype(str).str.contains("Auger", na=False)]
check("accented winner name preserved", len(uni) == 1
      and "é" in uni["winner_name"].iloc[0], f"{uni['winner_name'].tolist()}")
check("van de Zandschulp preserved",
      (m["loser_name"].astype(str) == "Botic van de Zandschulp").any())

print("\n5. long-form explode")
long = to_long(m)
check("two rows per match", len(long) == 2 * len(m), f"{len(long)} vs {2*len(m)}")
check("every match has exactly one winner",
      (long.groupby("match_id")["won"].sum() == 1).all())
check("player/opp are never equal", (long["player_id"] != long["opp_id"]).all())

print("\n6. headerless players file (older releases)")
players_headerless = (
    "104925,Jannik,Sinner,R,20010816,ITA,191,Q1234\n"
    "207989,Carlos,Alcaraz,R,20030505,ESP,183,Q5678\n"
    "200004,Félix,Auger-Aliassime,R,20000808,CAN,193,\n"
)
p = _read_csv(players_headerless.encode(), "player_id", PLAYER_COLS)
check("headerless players parsed", len(p) == 3, f"{len(p)}")
check("positional columns applied", list(p.columns) == PLAYER_COLS)
check("dob column present", "dob" in p.columns)

players_with_header = "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n" \
                      + players_headerless
p2 = _read_csv(players_with_header.encode(), "player_id", PLAYER_COLS)
check("headered players parsed identically", len(p2) == 3 and
      list(p2.columns) == PLAYER_COLS)
check("header row not treated as data",
      not (p2["player_id"].astype(str) == "player_id").any())

print("\n7. headerless rankings file")
rk_headerless = "20240101,1,104925,9500\n20240101,2,207989,8800\n"
r = _read_csv(rk_headerless.encode(), "ranking_date", RANKING_COLS)
check("headerless rankings parsed", len(r) == 2 and list(r.columns) == RANKING_COLS)
rk_headered = "ranking_date,rank,player,points\n" + rk_headerless
r2 = _read_csv(rk_headered.encode(), "ranking_date", RANKING_COLS)
check("headered rankings parsed", len(r2) == 2)
check("'player' column present for later rename", "player" in r2.columns)

print("\n8. the whole engine runs on this frame")
from engine.conditions import build_conditions  # noqa: E402
from engine.matchups import build_h2h  # noqa: E402
from engine.ratings import build_ratings  # noqa: E402
from engine.serve_return import build_serve_return  # noqa: E402

pm, cur = build_ratings(m)
check("ratings build on real-format data", len(pm) == len(m) and len(cur) > 0)
sp, sc = build_serve_return(m)
check("serve/return build", len(sp) == len(m))
cond = build_conditions(m)
check("conditions build", len(cond) == 2 * len(m))
h2h = build_h2h(m)
check("h2h build excludes walkovers/defaults", len(h2h) == 2 * int(
    (m["completed"] | m["retirement"]).sum()))

# ──────────────────────────────────────────────────────────────────────────────
print("")
print("9. the refresh can still see an in-progress tournament")
# Reported as "yesterday's results are not updated". All three bugs guarded
# here lived in the incremental refresh path, which had never once run against
# a live event - so nothing had ever exercised them.
from engine import refresh as rf  # noqa: E402
from engine.schema import REFRESH_LOOKBACK_DAYS  # noqa: E402
from engine.wta_source import _round_label  # noqa: E402

# `tourney_date` is a tournament's START date. Filtering incoming events to
# "starts after the archive's last date" excluded the very event the archive
# had just reached, so an in-progress tournament could never gain a round. The
# lookback has to cover the longest event on tour - a Slam is 14 days.
check("refresh lookback covers a two-week event", REFRESH_LOOKBACK_DAYS >= 14,
      str(REFRESH_LOOKBACK_DAYS))

# Round labels come from RoundID + draw size, never from how many matches have
# FINISHED. The old count-based rule called a part-played R16 "SF", and gave
# R128 and R64 the same name in a bye draw.
for rid, draw, want in [("1", 96, "R128"), ("2", 96, "R64"), ("3", 96, "R32"),
                        ("4", 96, "R16"), ("Q", 96, "QF"), ("S", 96, "SF"),
                        ("F", 96, "F"), ("1", 32, "R32"), ("2", 32, "R16"),
                        ("1", 64, "R64")]:
    check(f"RoundID {rid} in a {draw}-draw is {want}",
          _round_label(rid, draw) == want, _round_label(rid, draw))

# The same match arriving under a shifted match_id must not be stored twice;
# the ratings would count the result twice.
dup = pd.DataFrame({
    "tourney_id": ["2026-W1017"] * 3 + ["2026-W999"],
    "winner_id": [10, 10, 12, 10],
    "loser_id": [11, 11, 13, 11],
    "match_id": ["wta-2026-W1017-65", "wta-2026-W1017-70",
                 "wta-2026-W1017-71", "wta-2026-W999-1"],
})
out, n = rf.drop_duplicate_pairings(dup)
check("the same pairing twice in one event collapses",
      n == 1 and len(out) == 3, f"n={n} rows={len(out)}")
check("the surviving row is the one already published",
      out["match_id"].tolist()[0] == "wta-2026-W1017-65")
check("the same pairing in a DIFFERENT event is kept",
      int((out["tourney_id"] == "2026-W999").sum()) == 1)
rev = pd.DataFrame({"tourney_id": ["t", "t"], "winner_id": [10, 11],
                    "loser_id": [11, 10], "match_id": ["a", "b"]})
_, n2 = rf.drop_duplicate_pairings(rev)
check("a reversed result is the same pairing", n2 == 1, f"n={n2}")

# match_id must survive a refresh. The whole duplicate problem was that it
# ended in a positional cumcount, so a later round being played renumbered
# earlier matches. Keyed on the upstream src_id it cannot move.
from engine.schema import build_match_id  # noqa: E402

before = pd.DataFrame({
    "tour": ["wta"] * 2, "tourney_id": ["2026-W1017"] * 2,
    "match_num": [65, 66],
    "src_id": ["1017-2026-RS034", "1017-2026-RS035"],
})
# The same two matches after five more results shifted their position.
after = before.copy()
after["match_num"] = [70, 71]
check("match_id is unchanged when match_num shifts",
      build_match_id(before).tolist() == build_match_id(after).tolist(),
      str(build_match_id(after).tolist()))

# Historical rows have no upstream id; theirs comes from the source and is
# stable, so they keep the match_num form rather than losing an id entirely.
legacy = pd.DataFrame({"tour": ["atp"], "tourney_id": ["2010-339"],
                       "match_num": [7], "src_id": [None]})
check("rows without an upstream id fall back to match_num",
      build_match_id(legacy).tolist() == ["atp-2010-339-7"],
      str(build_match_id(legacy).tolist()))
check("ids stay unique across both forms",
      build_match_id(pd.concat([before, legacy], ignore_index=True)).nunique() == 3)

# The Tournaments page hides second-tier events. ATP Challengers have their own
# level code ("C"); WTA 125s do not - they arrive as "I", the same as a WTA 250
# - so they are matched on the name, which needs to be exact about it.
from engine.tournament import MINOR_LEVELS, _is_minor_name  # noqa: E402

check("ATP Challenger level is hidden", "C" in MINOR_LEVELS)
check("a WTA 125 is recognised", _is_minor_name("Warsaw 125"))
check("a WTA 125 mid-name is recognised", _is_minor_name("Targu Mures 125"))
check("a main-tour event is not", not _is_minor_name("Cincinnati"))
# The reason this tokenises instead of substring-matching.
check("digits inside a longer token do not count", not _is_minor_name("Open 1250"))
check("a 125 in a sponsor number does not false-positive",
      not _is_minor_name("ATP 1250 Trophy"))

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
