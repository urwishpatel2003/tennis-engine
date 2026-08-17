"""
End-to-end smoke test: does a prediction come out, and is it internally coherent?

    python tests/test_pipeline.py

Requires a built engine (`python run_engine.py --build`). Where test_markov.py
checks the maths and test_no_leakage.py checks the temporal invariant, this checks
the seams — that the tables join, the fallbacks fire instead of raising, and the
headline probability agrees with the score distribution that is printed beside it.

That last one is the coherence property `markov.invert_to_target` exists to
guarantee, and it is the easiest thing in the whole engine to break silently: an
edit that reports the blended probability but derives the games line from the raw
serve rates produces two numbers that quietly contradict each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.predict import Engine  # noqa: E402
from engine.schema import PROCESSED, RAW  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


for p in ["ratings_current.parquet", "serve_return_current.parquet", "h2h.parquet"]:
    if not (PROCESSED / p).exists():
        print(f"Missing {p} — run `python run_engine.py --build` first.")
        sys.exit(2)

eng = Engine()
ratings = pd.read_parquet(PROCESSED / "ratings_current.parquet")
players = pd.read_parquet(RAW / "players_atp.parquet")

top = (
    ratings[(ratings["tour"] == "atp") & (ratings["surface"] == "overall")]
    .nlargest(6, "elo").merge(players[["player_id", "name"]], on="player_id")
)
A, B, C = top["name"].iloc[0], top["name"].iloc[1], top["name"].iloc[4]

print("\n1. resolution and basic output")
p = eng.predict(A, B, tour="atp", surface="Hard", best_of=3)
check("returns a probability in (0,1)", 0.0 < p["win_prob_a"] < 1.0)
check("probabilities sum to 1", abs(p["win_prob_a"] + p["win_prob_b"] - 1.0) < 1e-9)
check("resolves by partial name", eng.predict(A.split()[-1], B, tour="atp")["player_a"]["name"] == A)
check("resolves by player id",
      eng.predict(int(top["player_id"].iloc[0]), B, tour="atp")["player_a"]["name"] == A)

print("\n2. symmetry")
ab = eng.predict(A, B, tour="atp", surface="Clay", best_of=3)["win_prob_a"]
ba = eng.predict(B, A, tour="atp", surface="Clay", best_of=3)["win_prob_b"]
check("P(A beats B) = P(A beats B) with arguments swapped", abs(ab - ba) < 2e-3,
      f"{ab:.5f} vs {ba:.5f}")

print("\n3. score distribution is coherent with the headline probability")
for surface in ("Hard", "Clay", "Grass"):
    for bo in (3, 5):
        q = eng.predict(A, C, tour="atp", surface=surface, best_of=bo)
        sets_prob = sum(
            v for k, v in q["set_score_probs"].items()
            if int(k.split("-")[0]) > int(k.split("-")[1])
        )
        check(f"{surface} bo{bo}: set-score mass = win prob",
              abs(sets_prob - q["win_prob_a"]) < 5e-3,
              f"{sets_prob:.4f} vs {q['win_prob_a']:.4f}")
        check(f"{surface} bo{bo}: set scores sum to 1",
              abs(sum(q["set_score_probs"].values()) - 1.0) < 1e-9)
        check(f"{surface} bo{bo}: totals sum to 1",
              abs(sum(q["total_games_probs"].values()) - 1.0) < 1e-9)
        # The favourite must be expected to win more games than they lose.
        fav_more_games = (q["win_prob_a"] > 0.5) == (q["game_margin_a"] > 0)
        check(f"{surface} bo{bo}: game margin agrees with the favourite",
              fav_more_games,
              f"p={q['win_prob_a']:.3f} margin={q['game_margin_a']:+.2f}")

print("\n4. best-of-5 sharpens the favourite")
p3 = eng.predict(A, C, tour="atp", surface="Hard", best_of=3)
p5 = eng.predict(A, C, tour="atp", surface="Hard", best_of=5)
check("bo5 win prob ≥ bo3 for the favourite", p5["win_prob_a"] >= p3["win_prob_a"] - 1e-9,
      f"bo3 {p3['win_prob_a']:.4f} bo5 {p5['win_prob_a']:.4f}")
check("bo5 has more expected games",
      p5["expected_total_games"] > p3["expected_total_games"])

print("\n5. market blending moves the number toward the price")
base = eng.predict(A, C, tour="atp", surface="Hard", best_of=3)["win_prob_a"]
pulled = eng.predict(A, C, tour="atp", surface="Hard", best_of=3,
                     market_prob_a=0.20)["win_prob_a"]
check("a 20% market price drags the model down", pulled < base,
      f"{base:.3f} → {pulled:.3f}")
pushed = eng.predict(A, C, tour="atp", surface="Hard", best_of=3,
                     market_prob_a=0.95)["win_prob_a"]
check("a 95% market price drags the model up", pushed > base,
      f"{base:.3f} → {pushed:.3f}")

print("\n6. fallbacks degrade rather than raise")
try:
    eng.predict("Definitely Not A Real Player", B, tour="atp")
    check("unknown player raises KeyError", False, "no exception raised")
except KeyError:
    check("unknown player raises KeyError", True)

q = eng.predict(A, B, tour="atp", surface="Carpet", best_of=3)
check("unseen surface still returns a prediction", 0.0 < q["win_prob_a"] < 1.0)
check("data_quality is reported", q["data_quality"]["level"] in ("low", "medium", "high"))

print("\n7. duplicate player ids resolve to the real career")
# Upstream assigns the same person more than one player_id (~1,800 player rows
# share a name). Taking the first exact name match returned the orphaned id, so a
# well-known player came back with a near-default rating and a "low data quality"
# warning. Whichever id carries the career must win.
dupe_names = (
    eng.players[eng.players["tour"] == "atp"]
    .groupby("name_lower").filter(lambda g: len(g) > 1)
)
if dupe_names.empty:
    check("duplicate-name players exist to test", True, "(none in this archive)")
else:
    counts = eng._match_counts()
    worse = 0
    for name, grp in dupe_names.groupby("name_lower"):
        best_id = int(grp.assign(n=grp["player_id"].map(counts).fillna(0))
                      .sort_values("n", ascending=False)["player_id"].iloc[0])
        got = eng.resolve_player(grp["name"].iloc[0], "atp")
        if got is None or int(got["player_id"]) != best_id:
            worse += 1
    check(f"all {dupe_names['name_lower'].nunique()} duplicated names pick the "
          f"id with the most matches", worse == 0, f"{worse} picked a lesser id")

# And the resolved player must be the one whose stats are reported.
top_name = top["name"].iloc[0]
pq = eng.predict(top_name, top["name"].iloc[3], tour="atp")
check("reported match count matches the resolved id",
      pq["player_a"]["matches"] == int(
          eng.ratings[(eng.ratings["tour"] == "atp")
                      & (eng.ratings["surface"] == "overall")
                      & (eng.ratings["player_id"] == pq["player_a"]["id"])]["matches"].iloc[0]))

print("\n8. data quality is per-player and names the weak link")
q_elite = eng.predict(top["name"].iloc[0], top["name"].iloc[1], tour="atp",
                      surface="Hard")["data_quality"]
check("two elite players on a live surface are 'high'", q_elite["level"] == "high",
      f"{q_elite['level']} {q_elite['flags']}")
q_carpet = eng.predict(top["name"].iloc[0], top["name"].iloc[1], tour="atp",
                       surface="Carpet")["data_quality"]
# Carpet was retired from the tour in 2009, so nobody has carpet history. That is
# a fact about the surface and must not be charged against the players.
check("carpet does not downgrade two elite players", q_carpet["level"] == "high",
      f"{q_carpet['level']} {q_carpet['flags']}")
check("carpet explains itself once",
      sum("carpet" in f.lower() for f in q_carpet["flags"]) == 1, str(q_carpet["flags"]))
check("quality flags are readable sentences, not codes",
      all(":" not in f or " " in f.split(":")[0] or f.split(":")[0].istitle()
          or " " in f for f in q_elite["flags"] + q_carpet["flags"]),
      str(q_elite["flags"] + q_carpet["flags"]))

print("\n9. venue effects move serve, not the winner")
flat = eng.predict(A, B, tour="atp", surface="Hard", best_of=3, altitude=0)
high = eng.predict(A, B, tour="atp", surface="Hard", best_of=3, altitude=2600)
check("altitude raises hold probability", high["hold_prob_a"] > flat["hold_prob_a"],
      f"{flat['hold_prob_a']:.4f} → {high['hold_prob_a']:.4f}")
# Thin air lifts BOTH players' serve, so breaks get rarer and more sets run to
# 6-6. Easier holding means LONGER matches in games, not shorter — at the limit
# of certain holds every set is a 13-game tiebreak set. (Altitude also barely
# moves the winner, which is the point: it is a serve effect, not a skill edge.)
check("altitude lengthens the match in games",
      high["expected_total_games"] > flat["expected_total_games"],
      f"{flat['expected_total_games']:.2f} → {high['expected_total_games']:.2f}")
check("altitude barely moves the winner",
      abs(high["win_prob_a"] - flat["win_prob_a"]) < 0.05,
      f"{flat['win_prob_a']:.4f} → {high['win_prob_a']:.4f}")

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
