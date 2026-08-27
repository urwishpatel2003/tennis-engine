"""
Tests for matching a fixture to its Kalshi market.

    python tests/test_kalshi_match.py

Matching the wrong contract spends real money on the wrong thing, so these
assert that it FAILS CLOSED. The fixtures below are shaped from live Kalshi
data captured on 2026-08-27, including the two traps that data actually
contained.

Trap 1 — the ticker's player code is ambiguous. Live markets held:

    KXWTAMATCH-26AUG27TAUPAR-PAR   Diane Parry
    KXWTAMATCH-26AUG27PARANN-PAR   Alycia Parks

The same three letters for different players. Matching on the code backs Parks
when the model meant Parry, so names come from yes_sub_title instead.

Trap 2 — series prefixes catch the wrong sport. `KXWTABLETENNISMATCH` is
"Women's Table Tennis Match" and begins with KXWTA.

No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.kalshi_match import (  # noqa: E402
    SERIES,
    ask_price,
    ask_size,
    find_market,
    norm,
    surname,
)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def mkt(event: str, code: str, name: str, ask: int | None = 50) -> dict:
    return {"ticker": f"{event}-{code}", "event_ticker": event,
            "yes_sub_title": name, "title": f"{name} wins", "yes_ask": ask}


# Real shapes: note both events use the code PAR for different players.
EVENTS = {
    "KXWTAMATCH-26AUG27TAUPAR": [
        mkt("KXWTAMATCH-26AUG27TAUPAR", "TAU", "Clara Tauson", 62),
        mkt("KXWTAMATCH-26AUG27TAUPAR", "PAR", "Diane Parry", 41),
    ],
    "KXWTAMATCH-26AUG27PARANN": [
        mkt("KXWTAMATCH-26AUG27PARANN", "PAR", "Alycia Parks", 55),
        mkt("KXWTAMATCH-26AUG27PARANN", "ANN", "Ann Li", 48),
    ],
    "KXWTAMATCH-26AUG27MERSTA": [
        mkt("KXWTAMATCH-26AUG27MERSTA", "MER", "Elise Mertens", 70),
        mkt("KXWTAMATCH-26AUG27MERSTA", "STA", "Yuliia Starodubtseva", 33),
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. only the exact match-winner series are used")
check("ATP series is exactly KXATPMATCH", SERIES["atp"] == "KXATPMATCH")
check("WTA series is exactly KXWTAMATCH", SERIES["wta"] == "KXWTAMATCH")
# The trap: a prefix match on KXWTA would sweep in table tennis.
check("table tennis is not the WTA series",
      SERIES["wta"] != "KXWTABLETENNISMATCH"
      and "KXWTABLETENNISMATCH".startswith(SERIES["wta"][:5]))

print("\n2. a real fixture resolves to the right contract")
r = find_market("Diane Parry", "Clara Tauson", "wta", events=EVENTS)
check("the fixture is found", r["ok"], str(r))
check("it is the Tauson/Parry event", r.get("event") == "KXWTAMATCH-26AUG27TAUPAR")
check("backing Parry buys the Parry contract",
      r["a"]["ticker"].endswith("-PAR") and r["a_name"] == "Diane Parry", str(r.get("a")))
check("the opponent contract is Tauson's", r["b_name"] == "Clara Tauson")

print("\n3. the ambiguous player code does not confuse them")
r2 = find_market("Alycia Parks", "Ann Li", "wta", events=EVENTS)
check("Parks resolves to her own event",
      r2["ok"] and r2["event"] == "KXWTAMATCH-26AUG27PARANN", str(r2))
check("and to the Parks contract, not Parry's",
      r2["a_name"] == "Alycia Parks", str(r2.get("a_name")))
check("Parry and Parks are different keys", norm("Diane Parry") != norm("Alycia Parks"))

print("\n4. it fails closed rather than guessing")
bad = find_market("Diane Parry", "Alycia Parks", "wta", events=EVENTS)
check("two players from different events are refused",
      not bad["ok"], str(bad))
check("and it gives a reason", bool(bad.get("reason")))
half = find_market("Diane Parry", "Someone Unlisted", "wta", events=EVENTS)
check("one side matching is not enough", not half["ok"], str(half))
check("an unknown fixture is refused", not find_market(
    "Nobody Atall", "Alsonot Real", "wta", events=EVENTS)["ok"])
check("a missing name is refused", not find_market("", "Clara Tauson", "wta",
                                                   events=EVENTS)["ok"])
check("an unknown tour is refused", not find_market(
    "Diane Parry", "Clara Tauson", "padel", events=None)["ok"])

print("\n5. surnames only rescue an UNAMBIGUOUS case")
# Kalshi and our archive disagree on given names more often than surnames.
r3 = find_market("D. Parry", "C. Tauson", "wta", events=EVENTS)
check("initials still resolve via surname", r3["ok"] and r3["match_type"] == "surname",
      str(r3))
check("and to the right contracts", r3["a_name"] == "Diane Parry", str(r3.get("a_name")))
# Two events sharing both surnames must refuse rather than pick one.
dupe = dict(EVENTS)
dupe["KXWTAMATCH-26AUG28TAUPAR"] = [
    mkt("KXWTAMATCH-26AUG28TAUPAR", "TAU", "Clara Tauson", 60),
    mkt("KXWTAMATCH-26AUG28TAUPAR", "PAR", "Diane Parry", 43),
]
amb = find_market("D. Parry", "C. Tauson", "wta", events=dupe)
check("two candidate events are refused, not guessed between",
      not amb["ok"] and "ambiguous" in (amb.get("reason") or ""), str(amb))
check("surname helper takes the last token", surname("Daniel Merida Aguilar") == "aguilar")

print("")
print("6. prices come from the V2 dollar fields")
# The bug this guards: Kalshi V2 carries fixed-point dollar STRINGS and leaves
# the legacy cent field null. Reading the cent field made every live market
# look like it had no offer, so every ticket was refused and the feature could
# never have produced a trade at all.
check("a dollar string is read", ask_price({"yes_ask_dollars": "0.3700"}) == 0.37)
check("dollars win over a null cent field",
      ask_price({"yes_ask_dollars": "0.6400", "yes_ask": None}) == 0.64)
check("the cent field still works as a fallback", ask_price({"yes_ask": 41}) == 0.41)
check("no ask at all means no price", ask_price({"yes_ask": None}) is None)
check("a missing field is refused", ask_price({}) is None)
check("a nonsense ask is refused",
      ask_price({"yes_ask_dollars": "0.0000"}) is None
      and ask_price({"yes_ask_dollars": "1.0000"}) is None)
check("junk is refused rather than raising",
      ask_price({"yes_ask_dollars": "abc"}) is None)

print("")
print("7. offered size caps what we can ask for")
check("size is read as a number", ask_size({"yes_ask_size_fp": "106.70"}) == 106.7)
check("a missing size is None", ask_size({}) is None)
check("junk size is None", ask_size({"yes_ask_size_fp": "x"}) is None)

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
