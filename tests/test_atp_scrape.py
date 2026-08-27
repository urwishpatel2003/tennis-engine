"""
Tests for reading ATP results off atptour.com.

    python tests/test_atp_scrape.py

The mirror files main-draw results late - on 2026-08-27 it held nothing past
qualifying for Winston-Salem four days in, and had never filed the second half
of Cincinnati including the final. This parser reads the same pages directly.

The fixture below reproduces the three structures that actually broke it, all
found by running it rather than reading it:

  1. NESTING. The first version closed a player on any `</div>`, so the inner
     `name` and `country` divs ended the player before its scores were read and
     the page parsed to ZERO matches. Depth is now tracked per region.
  2. THE HEADER HOLDS TWO TEXTS. A match header carries the round AND the match
     duration ("01:37:13"). Reading the duration as a round label cleared every
     round and again parsed to zero. Only the first text counts.
  3. QUALIFYING INHERITS. A qualifying block's header is not a round this parser
     knows, and leaving the previous label standing filed 36 qualifying matches
     as "Round of 128" - a 96 draw has 32. An unrecognised header clears it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.atp_scrape import parse_results  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def _match(round_label, duration, a, acode, ascore, b, bcode, bscore, a_wins=True):
    def side(name, code, scores, winner):
        cells = "".join(
            f'<div class="score-item">{"".join(f"<span>{x}</span>" for x in c)}</div>'
            for c in scores)
        win = '<div class="winner"><span class="icon-checkmark"></span></div>' if winner else ""
        return (f'<div class="stats-item"><div class="player-info">'
                f'<div class="profile"><img src="x"/></div>'
                f'<div class="country"><svg></svg></div>'
                f'<div class="name"><a href="/en/players/{code}-slug/{code}/overview">{name}</a>'
                f'<span>(15)</span></div>{win}</div>'
                f'<div class="scores">{cells}</div></div>')
    return (f'<div class="match"><div class="match-header">'
            f'<span><strong>{round_label} - Stadium Court</strong></span>'
            f'<span>{duration}</span></div>'
            f'<div class="match-content"><div class="match-stats">'
            f'{side(a, acode, ascore, a_wins)}{side(b, bcode, bscore, not a_wins)}'
            f'</div></div></div>')


HTML = (
    '<div class="match-group">'
    + _match("Final", "01:52:10",
             "Arthur Fils", "f0f1", [[""], ["6"], ["1"], ["6"]],
             "Frances Tiafoe", "td51", [[""], ["3"], ["6"], ["0"]])
    + _match("Quarterfinals", "02:10:00",
             "Brandon Nakashima", "n0ae", [[""], ["7", "9"], ["4"], ["6"]],
             "Taylor Fritz", "fb98", [[""], ["6", "7"], ["6"], ["3"]])
    + '</div><div class="match-group">'
    # A qualifying block: its header is not a round we map.
    + _match("Qualifying", "00:58:00",
             "Some Qualifier", "q001", [[""], ["6"], ["6"]],
             "Another Qualifier", "q002", [[""], ["2"], ["3"]])
    + '</div>'
)

rows = parse_results(HTML)

print("\n1. the page parses at all")
check("two main-draw matches found", len(rows) == 2, f"{len(rows)}")

print("\n2. rounds, not durations")
got = sorted(r["round"] for r in rows)
check("rounds are F and QF", got == ["F", "QF"], str(got))
check("the duration was not read as a round",
      all(r["round"] in ("F", "QF") for r in rows))

print("\n3. qualifying does not inherit a main-draw round")
check("the qualifying match is dropped",
      not any("Qualifier" in (r["winner_name"] or "") for r in rows))

print("\n4. players, codes and winners")
f = next(r for r in rows if r["round"] == "F")
check("winner is the side with the marker", f["winner_name"] == "Arthur Fils",
      str(f["winner_name"]))
check("loser is the other side", f["loser_name"] == "Frances Tiafoe")
check("winner code captured from the profile link", f["winner_code"] == "f0f1")
check("loser code captured", f["loser_code"] == "td51")
check("a seed in the name block is not read as a name",
      "(" not in (f["winner_name"] or ""))

print("\n5. scores, winner's games first")
check("final score is 6-3 1-6 6-0", f["score"] == "6-3 1-6 6-0", f["score"])
q = next(r for r in rows if r["round"] == "QF")
check("a tiebreak keeps the loser's points", q["score"].startswith("7-6(7)"), q["score"])

print("\n6. an in-progress match is not filed as finished")
live = _match("Semifinals", "00:31:00",
              "Player One", "p001", [[""], ["3"]],
              "Player Two", "p002", [[""], ["2"]], a_wins=True)
live = live.replace('<div class="winner"><span class="icon-checkmark"></span></div>', "")
check("no winner marked means no row", len(parse_results(live)) == 0)

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
