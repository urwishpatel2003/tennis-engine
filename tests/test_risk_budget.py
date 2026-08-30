"""
Tests for the daily exposure budget.

    python tests/test_risk_budget.py

The cap bounds money AT RISK, not money staked over the day: once a market
settles the stake has come back, win or lose, so it stops consuming the
allowance. That is a deliberate loosening — on a fast-settling day the same
allowance can be spent more than once — so the gross figure stays visible
beside it.

The failure that matters is the API one. If positions cannot be read, "nothing
is open" and "could not tell" look identical from the outside, and treating the
second as the first hands back the entire daily budget at the worst possible
moment.

No network: the ledger is a temp file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tmp = Path(tempfile.mkdtemp())
os.environ["BET_LOG_PATH"] = str(tmp / "bet_log.jsonl")

from engine import risk  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


DAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
OPEN_T, SETTLED_T = "KXWTAMATCH-OPEN-A", "KXWTAMATCH-DONE-A"
ROWS = [
    {"day": DAY, "stake": 30.0, "ticker": OPEN_T, "contracts": 100,
     "client_order_id": "a"},
    {"day": DAY, "stake": 50.0, "ticker": SETTLED_T, "contracts": 100,
     "client_order_id": "b"},
    # A different day must never count towards today's allowance.
    {"day": "2020-01-01", "stake": 999.0, "ticker": "OLD", "contracts": 1},
]
ledger = tmp / "kalshi_exposure.jsonl"
ledger.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")

BANK = 1000.0

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. without position data nothing is released")
check("the gross figure is used", risk.committed_today() == 80.0,
      str(risk.committed_today()))
b = risk.budget(BANK)
check("budget counts both stakes", b["committed"] == 80.0, str(b))
check("another day is never counted", risk.staked_today() == 80.0,
      str(risk.staked_today()))

print("\n2. a settled market releases its stake")
b2 = risk.budget(BANK, {OPEN_T})
check("only the open stake is at risk", b2["committed"] == 30.0, str(b2))
check("the released amount is reported", b2["released"] == 50.0, str(b2))
check("the gross figure is still shown", b2["staked_today"] == 80.0, str(b2))
check("the allowance grows back",
      b2["daily_remaining"] > b["daily_remaining"], str(b2))

print("\n3. everything settled frees the whole day")
b3 = risk.budget(BANK, set())
check("nothing is at risk", b3["committed"] == 0.0, str(b3))
check("the full cap is available", b3["daily_remaining"] == b3["daily_cap"], str(b3))
check("but the gross stake is not erased", b3["staked_today"] == 80.0, str(b3))

print("\n4. an unreadable account must not release anything")
# The dangerous case: None means "could not tell", and must behave like the
# safe default rather than like an empty set.
b4 = risk.budget(BANK, None)
check("None keeps the gross figure", b4["committed"] == 80.0, str(b4))
check("None is not the same as an empty set",
      b4["committed"] != risk.budget(BANK, set())["committed"])
check("nothing is reported as released", b4["released"] == 0.0, str(b4))

print("\n5. the caps still bind")
check("the ticket cap is still a percentage",
      b2["ticket_pct"] <= risk.MAX_TICKET_PCT, str(b2))
small = risk.budget(80.0 / (risk.MAX_DAILY_PCT / 100.0), {OPEN_T, SETTLED_T})
check("a full day is still exhaustible", small["exhausted"], str(small))
check("a zero bankroll is exhausted", risk.budget(0.0, set())["exhausted"])
check("a released day cannot exceed its cap",
      b3["daily_remaining"] <= b3["daily_cap"], str(b3))

print("\n6. a torn ledger line does not hide exposure")
ledger.write_text(json.dumps(ROWS[0]) + "\n{not json\n"
                  + json.dumps(ROWS[1]) + "\n", encoding="utf-8")
check("readable rows still count", risk.committed_today() == 80.0,
      str(risk.committed_today()))

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
