"""
Tests for the freshness guard on the order path.

    python tests/test_kalshi_order_guard.py

A ticket is a snapshot. Between building one and confirming it, the match can
start, the book can empty, and the price can move — all three were observed in
live data within minutes of a dry run on 2026-08-27:

    ticket: Alycia Parks 75 @ 0.31      minutes later the ask was 0.55
    ticket: Diane Parry  64 @ 0.36      minutes later there was no ask at all

The dangerous direction is DOWN. A bid left at a stale-high price fills at once
against a market that fell because the player is losing, at a price that only
looked like value against a probability computed before the match began.

No network: every case passes an explicit market snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.kalshi_order import (  # noqa: E402
    MAX_PRICE_DRIFT,
    build_payload,
    new_client_order_id,
    verify_market,
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


def snap(ask="0.3100", size="500.00", status="active") -> dict:
    return {"status": status, "yes_ask_dollars": ask, "yes_ask_size_fp": size}


T = "KXWTAMATCH-26AUG27PARANN-PAR"

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. an unchanged market still trades")
r = verify_market(T, 75, 0.31, market=snap())
check("the ticket passes", r["ok"], str(r))
check("it reports the ask it saw", r["ask"] == 0.31, str(r))
check("and the depth", r["offered"] == 500.0, str(r))
check("a move inside tolerance is fine",
      verify_market(T, 75, 0.31, market=snap(ask="0.3200"))["ok"])

print("\n2. a price that moved DOWN is refused")
# The expensive case: fills instantly, against news the model has not seen.
down = verify_market(T, 75, 0.31, market=snap(ask="0.2000"))
check("the stale bid is refused", not down["ok"], str(down))
check("and the reason names the direction", "down" in down["reason"], down["reason"])
check("the real observed drop is refused",
      not verify_market(T, 64, 0.36, market=snap(ask="0.1500"))["ok"])

print("\n3. a price that moved UP is refused too")
# Harmless economically - it would not fill - but it leaves a resting order at
# a price nobody re-examined.
up = verify_market(T, 75, 0.31, market=snap(ask="0.5500"))
check("the ticket is refused", not up["ok"], str(up))
check("and the reason names the direction", "up" in up["reason"], up["reason"])

print("\n4. the market has to still be there")
check("a closed market is refused",
      not verify_market(T, 75, 0.31, market=snap(status="closed"))["ok"])
check("a settled market is refused",
      not verify_market(T, 75, 0.31, market=snap(status="settled"))["ok"])
check("a vanished market is refused", not verify_market(T, 75, 0.31, market={})["ok"])
empty = verify_market(T, 75, 0.31, market=snap(ask=None))
check("an empty book is refused", not empty["ok"], str(empty))
check("and says nothing is offered", "offered" in empty["reason"], empty["reason"])

print("\n5. depth is re-checked, not assumed")
thin = verify_market(T, 75, 0.31, market=snap(size="21.00"))
check("more contracts than are offered is refused", not thin["ok"], str(thin))
check("and the reason gives both numbers",
      "21" in thin["reason"] and "75" in thin["reason"], thin["reason"])
check("exactly the offered size is allowed",
      verify_market(T, 21, 0.31, market=snap(size="21.00"))["ok"])

print("\n6. every refusal explains itself")
for label, m, cnt in [("closed", snap(status="closed"), 75),
                      ("no ask", snap(ask=None), 75),
                      ("moved", snap(ask="0.5500"), 75),
                      ("thin", snap(size="2.00"), 75),
                      ("gone", {}, 75)]:
    r = verify_market(T, cnt, 0.31, market=m)
    check(f"{label} refuses with a reason", not r["ok"] and bool(r["reason"]))

print("\n7. the tolerance is a sane band")
check("it is a positive number of cents", 0.0 < MAX_PRICE_DRIFT < 0.25,
      str(MAX_PRICE_DRIFT))

print("\n8. idempotency is unchanged by the guard")
cid = new_client_order_id()
p1 = build_payload(T, 75, 0.31, cid)
p2 = build_payload(T, 75, 0.31, cid)
check("the same ticket builds the same order id", p1 == p2, str(p1))
check("two tickets get different ids", new_client_order_id() != new_client_order_id())

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
