"""
The invariant that makes the backtest worth anything: no future leakage.

    python tests/test_no_leakage.py

Run this after touching ANYTHING in engine/ratings.py, engine/serve_return.py,
engine/conditions.py or engine/matchups.py.

A prediction model that can see the result it is predicting will look brilliant
and be worthless. The NFL engine in this workspace has the scar: an inverted
spread convention faked an ~86% cover rate against a true out-of-sample ~49%.
The defence there was a convention test; the defence here is this file.

What is checked
---------------
1. **Prefix independence.** Rebuild the ratings on a truncated match log. Every
   pre-match value for the surviving matches must be bit-identical to the value
   from the full-history build. If a future match could influence an earlier
   row, truncating the future would change it.
2. **First-appearance neutrality.** A player's very first match must carry the
   default rating and a zero serve/return excess — no information can exist
   before their debut.
3. **Head-to-head windowing.** `h2h_record(before=D)` must never count a meeting
   on or after D.
4. **Conditions are backward-looking.** A player's first-ever match must show no
   rest days and no accumulated fatigue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.conditions import build_conditions  # noqa: E402
from engine.matchups import build_h2h, h2h_record  # noqa: E402
from engine.ratings import ELO_INIT, build_ratings  # noqa: E402
from engine.schema import RAW  # noqa: E402
from engine.serve_return import build_serve_return  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


matches_path = RAW / "matches_atp.parquet"
if not matches_path.exists():
    print("No matches_atp.parquet — run fetch_data.py or "
          "tools/make_synthetic_data.py first.")
    sys.exit(2)

full = pd.read_parquet(matches_path)
# A few thousand matches is plenty to expose leakage and keeps the test quick.
full = full.sort_values(["tourney_date", "tourney_id", "match_num"]).head(6000)
cut_date = full["tourney_date"].quantile(0.6)
prefix = full[full["tourney_date"] < cut_date]

print(f"\nfull: {len(full):,} matches   prefix: {len(prefix):,} "
      f"(before {pd.Timestamp(cut_date).date()})")

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. prefix independence — ratings")
r_full, _ = build_ratings(full)
r_pre, _ = build_ratings(prefix)

merged = r_pre.merge(r_full, on="match_id", suffixes=("_pre", "_full"))
check("prefix build covers the same matches", len(merged) == len(r_pre),
      f"{len(merged)} vs {len(r_pre)}")
for col in ["w_elo", "l_elo", "w_elo_surface", "l_elo_surface", "elo_exp_blend"]:
    diff = (merged[f"{col}_pre"] - merged[f"{col}_full"]).abs().max()
    check(f"{col} identical on the shared prefix", diff < 1e-9, f"max diff {diff}")

print("\n1b. prefix independence — serve/return")
s_full, _ = build_serve_return(full)
s_pre, _ = build_serve_return(prefix)
sm = s_pre.merge(s_full, on="match_id", suffixes=("_pre", "_full"))
for col in ["w_serve_excess", "l_serve_excess", "w_return_excess", "l_return_excess"]:
    diff = (sm[f"{col}_pre"] - sm[f"{col}_full"]).abs().max()
    check(f"{col} identical on the shared prefix", diff < 1e-9, f"max diff {diff}")

print("\n1c. prefix independence — conditions")
c_full = build_conditions(full)
c_pre = build_conditions(prefix)
cm = c_pre.merge(c_full, on=["match_id", "player_id"], suffixes=("_pre", "_full"))
for col in ["minutes_7d", "minutes_14d", "matches_this_event", "fatigue_index"]:
    diff = (cm[f"{col}_pre"] - cm[f"{col}_full"]).abs().max()
    check(f"{col} identical on the shared prefix", diff < 1e-9, f"max diff {diff}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n2. first appearance carries no information")
first_seen: dict[int, str] = {}
for r in full.itertuples(index=False):
    for pid in (int(r.winner_id), int(r.loser_id)):
        first_seen.setdefault(pid, r.match_id)

debut_ok_elo = debut_ok_sr = 0
debut_bad_elo: list[str] = []
debut_bad_sr: list[str] = []
r_idx = r_full.set_index("match_id")
s_idx = s_full.set_index("match_id")

for pid, mid in list(first_seen.items())[:400]:
    row = r_idx.loc[mid]
    srow = s_idx.loc[mid]
    is_winner = int(row["winner_id"]) == pid
    elo = row["w_elo"] if is_winner else row["l_elo"]
    serve = srow["w_serve_excess"] if is_winner else srow["l_serve_excess"]
    ret = srow["w_return_excess"] if is_winner else srow["l_return_excess"]

    if abs(float(elo) - ELO_INIT) < 1e-9:
        debut_ok_elo += 1
    else:
        debut_bad_elo.append(f"{pid}:{elo:.2f}")
    if abs(float(serve)) < 1e-12 and abs(float(ret)) < 1e-12:
        debut_ok_sr += 1
    else:
        debut_bad_sr.append(f"{pid}:{serve:.5f}/{ret:.5f}")

check("every debut starts at ELO_INIT", not debut_bad_elo,
      f"{len(debut_bad_elo)} bad, e.g. {debut_bad_elo[:3]}")
check("every debut starts at zero serve/return excess", not debut_bad_sr,
      f"{len(debut_bad_sr)} bad, e.g. {debut_bad_sr[:3]}")

# ──────────────────────────────────────────────────────────────────────────────
print("\n3. head-to-head respects the `before` cutoff")
h2h = build_h2h(full)
pairs = (
    h2h.groupby(["player_id", "opp_id"]).size()
    .sort_values(ascending=False).head(20).index.tolist()
)
bad_window = 0
for pid, oid in pairs:
    meetings = h2h[(h2h["player_id"] == pid) & (h2h["opp_id"] == oid)]
    mid_date = meetings["tourney_date"].iloc[len(meetings) // 2]
    rec = h2h_record(h2h, pid, oid, before=mid_date)
    expected = int((meetings["tourney_date"] < mid_date).sum())
    if rec["n"] != expected:
        bad_window += 1
check("h2h_record(before=D) counts only meetings strictly before D",
      bad_window == 0, f"{bad_window} pairs wrong")

full_rec = h2h_record(h2h, pairs[0][0], pairs[0][1])
early_rec = h2h_record(h2h, pairs[0][0], pairs[0][1],
                       before=h2h["tourney_date"].min())
check("cutoff at the very start yields an empty record", early_rec["n"] == 0)
check("no cutoff yields the full record", full_rec["n"] > early_rec["n"])

# ──────────────────────────────────────────────────────────────────────────────
print("\n4. conditions are backward-looking")
# NOTE: do NOT re-sort by tourney_date here. Every match at a tournament shares
# one tourney_date, so sorting on it alone reorders rows within an event and the
# "first" row per player stops being their debut — it can land on their R16 match,
# which legitimately has rest days and accumulated minutes. build_conditions
# already emits rows in processing order, which is the order that matters.
c = c_full
firsts = c.groupby("player_id", as_index=False).head(1)
check("debut has no recorded rest days", firsts["days_rest"].isna().all(),
      f"{int(firsts['days_rest'].notna().sum())} debuts had rest days")
check("debut has zero recent minutes", (firsts["minutes_7d"] == 0).all(),
      f"max {firsts['minutes_7d'].max()}")
check("debut has zero matches at the event",
      (firsts["matches_this_event"] == 0).all())
check("fatigue is never negative", (c["fatigue_index"] >= 0).all())

# ──────────────────────────────────────────────────────────────────────────────
print("\n5. the head-to-head replay never sees the present")
# engine/replay.py rebuilds head-to-head for a whole frame at once so the
# backtest can score the adjustment layer. Rebuilding history in bulk is exactly
# where leakage creeps in, and the term is invisible to the rest of the backtest
# if it goes wrong — this is the assertion that would catch it.
from engine import replay  # noqa: E402

dates = pd.to_datetime(["2020-01-06", "2020-02-03", "2020-03-02", "2020-04-06"])
a_ids = np.array([1, 1, 1, 1])
b_ids = np.array([2, 2, 2, 2])
surf = np.array(["Hard"] * 4)
gapv = np.zeros(4)
# A wins every meeting. The FIRST match must score 0 — there is no prior history
# — and each later one may only reflect the meetings strictly before it.
d = replay.h2h_elo(list(dates), a_ids, b_ids, surf, gapv, np.array([1, 1, 1, 1]))
check("first-ever meeting gets no head-to-head credit", abs(d[0]) < 1e-12, f"{d[0]}")
check("credit grows with each prior win", d[1] > 0 and d[3] > d[1],
      f"{d.tolist()}")

# Feeding the same matches in a shuffled order must give the same answer: the
# function sorts by date internally, so results cannot depend on row order.
order = [2, 0, 3, 1]
d2 = replay.h2h_elo([dates[i] for i in order], a_ids[order], b_ids[order],
                    surf[order], gapv[order], np.array([1, 1, 1, 1])[order])
check("row order does not change the result",
      np.allclose(d2, d[order], atol=1e-12), f"{d2.tolist()} vs {d[order].tolist()}")

# If A loses every meeting instead, the sign must flip.
d3 = replay.h2h_elo(list(dates), a_ids, b_ids, surf, gapv, np.array([0, 0, 0, 0]))
check("losing the prior meetings flips the sign", d3[3] < 0, f"{d3[3]}")

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
