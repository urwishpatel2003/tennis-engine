"""
Tests for the freshness guard on the order path.

    python tests/test_kalshi_order_guard.py

A ticket is a snapshot. Between building one and confirming it the price moves,
the book empties, and the match starts. All of that was seen in live data within
minutes of the first dry run on 2026-08-27:

    ticket: Alycia Parks 75 @ 0.31      minutes later the ask was 0.55
    ticket: Diane Parry  64 @ 0.36      minutes later there was no ask at all

The start check is separate from the price check and cannot be folded into it.
Kalshi keeps a match market ACTIVE right through the match - its own
early_close_condition reads "this market will close and expire after a winner is
declared" - so an open market is not evidence that play has not begun.

WHERE the start time comes from is the part that went wrong. It was read off
Kalshi's `occurrence_datetime`, which is NOT a per-match start: a live pull had
all 125 open tennis markets reporting a start in the past, dozens of different
matches sharing one identical timestamp. Gating on it refused nearly every
ticket, including matches three hours away whose own countdown was still
running beside the refusal. The odds feed's commence_time is the trusted source
and is passed in explicitly.

No network: every case passes an explicit market snapshot and a fixed clock.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.kalshi_order import (  # noqa: E402
    MAX_PRICE_DRIFT,
    SELF_TRADE_PREVENTION,
    TIME_IN_FORCE_MAKER,
    TIME_IN_FORCE_TAKER,
    build_payload,
    new_client_order_id,
    started,
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


# A fixed clock, so "before the match" never depends on when this is run.
NOW = datetime(2026, 8, 27, 17, 30, tzinfo=timezone.utc)
SOON = (NOW + timedelta(hours=11)).strftime("%Y-%m-%dT%H:%M:%SZ")   # tonight
GONE = (NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")    # underway

T = "KXWTAMATCH-26AUG27PARANN-PAR"


def snap(ask="0.3100", size="500.00", status="active", start=None) -> dict:
    """
    A market. `start` sets Kalshi's occurrence_datetime, which must be IGNORED:
    it is a session marker shared by dozens of matches, not a per-match start.
    """
    m = {"status": status, "yes_ask_dollars": ask, "yes_ask_size_fp": size}
    if start is not None:
        m["occurrence_datetime"] = start
    return m


def verify(count, price, market, starts=SOON):
    """The odds feed's commence_time is the trusted start, so it is passed in."""
    return verify_market(T, count, price, market=market, now=NOW, starts=starts)


# ──────────────────────────────────────────────────────────────────────────────
print("\n1. an unchanged market still trades")
r = verify(75, 0.31, snap())
check("the ticket passes", r["ok"], str(r))
check("it reports the ask it saw", r["ask"] == 0.31, str(r))
check("and the depth", r["offered"] == 500.0, str(r))
check("a move inside tolerance is fine", verify(75, 0.31, snap(ask="0.3200"))["ok"])

print("\n2. a price that moved DOWN is refused")
# The expensive case: fills instantly, against news the model has not seen.
down = verify(75, 0.31, snap(ask="0.2000"))
check("the stale bid is refused", not down["ok"], str(down))
check("and the reason names the direction", "down" in down["reason"], down["reason"])
check("the real observed drop is refused", not verify(64, 0.36, snap(ask="0.1500"))["ok"])

print("\n3. a price that moved UP is refused too")
# Harmless economically - it would not fill - but it leaves a resting order at a
# price nobody re-examined.
up = verify(75, 0.31, snap(ask="0.5500"))
check("the ticket is refused", not up["ok"], str(up))
check("and the reason names the direction", "up" in up["reason"], up["reason"])

print("\n4. the market has to still be there")
check("a closed market is refused", not verify(75, 0.31, snap(status="closed"))["ok"])
check("a settled market is refused", not verify(75, 0.31, snap(status="settled"))["ok"])
check("a vanished market is refused", not verify(75, 0.31, {})["ok"])
empty = verify(75, 0.31, snap(ask=None))
check("an empty book is refused", not empty["ok"], str(empty))
check("and says nothing is offered", "offered" in empty["reason"], empty["reason"])

print("\n5. depth is re-checked, not assumed")
thin = verify(75, 0.31, snap(size="21.00"))
check("more contracts than are offered is refused", not thin["ok"], str(thin))
check("and the reason gives both numbers",
      "21" in thin["reason"] and "75" in thin["reason"], thin["reason"])
check("exactly the offered size is allowed", verify(21, 0.31, snap(size="21.00"))["ok"])

print("\n6. a match already underway is refused")
live = verify(75, 0.31, snap(), starts=GONE)
check("an in-play ticket is refused", not live["ok"], str(live))
check("and the reason says why", "started" in live["reason"], live["reason"])
check("an in-play market is still 'active', so status alone would miss it",
      snap()["status"] == "active")
check("a match starting tonight is fine", verify(75, 0.31, snap(), starts=SOON)["ok"])

print("\n7. Kalshi's occurrence_datetime is NOT trusted")
# The bug this guards. Every open tennis market reported a start in the past,
# dozens sharing one timestamp, so gating on it refused matches hours away.
stale = verify(75, 0.31, snap(start=GONE), starts=SOON)
check("a stale market timestamp does not refuse a future match", stale["ok"], str(stale))
check("the feed decides, not the market",
      not verify(75, 0.31, snap(start=SOON), starts=GONE)["ok"])
check("the market field alone means nothing",
      started(snap(start=GONE), now=NOW) is None)

print("\n8. the start time itself is read correctly")
check("a future start has not begun", started({}, now=NOW, starts=SOON) is False)
check("a past start has begun", started({}, now=NOW, starts=GONE) is True)
check("the moment of the start counts as begun",
      started({}, now=NOW, starts="2026-08-27T17:30:00Z") is True)
check("a naive timestamp is read as UTC",
      started({}, now=NOW, starts="2026-08-27T19:00:00") is False)
check("a missing start time is unknown", started({}, now=NOW) is None)
check("an unparseable start time is unknown",
      started({}, now=NOW, starts="tomorrow") is None)
# Not knowing is not the same as knowing it is fine - this is a money path, and
# a ticket built by this server always carries a start time.
unknown = verify(75, 0.31, snap(), starts=None)
check("an unknown start time is REFUSED, not waved through", not unknown["ok"],
      str(unknown))
check("and it says to re-scan", "re-scan" in unknown["reason"], unknown["reason"])

print("\n9. every refusal explains itself")
for label, m, cnt, st in [("closed", snap(status="closed"), 75, SOON),
                          ("no ask", snap(ask=None), 75, SOON),
                          ("moved", snap(ask="0.5500"), 75, SOON),
                          ("thin", snap(size="2.00"), 75, SOON),
                          ("in play", snap(), 75, GONE),
                          ("no start time", snap(), 75, None),
                          ("gone", {}, 75, SOON)]:
    res = verify(cnt, 0.31, m, starts=st)
    check(f"{label} refuses with a reason", not res["ok"] and bool(res["reason"]))

print("\n10. the tolerance is a sane band")
check("it is a positive number of cents", 0.0 < MAX_PRICE_DRIFT < 0.25,
      str(MAX_PRICE_DRIFT))

print("\n11. idempotency is unchanged by the guard")
cid = new_client_order_id()
check("the same ticket builds the same order id",
      build_payload(T, 75, 0.31, cid) == build_payload(T, 75, 0.31, cid))
check("two tickets get different ids", new_client_order_id() != new_client_order_id())

print("\n12. the payload carries every field V2 requires")
# The first live attempt failed with 400 missing_parameters:
#   'CreateOrderV2Request.SelfTradePreventionType' failed on the 'required' tag
#   'CreateOrderV2Request.TimeInForce'             failed on the 'required' tag
pay = build_payload(T, 109, 0.32, "cid-1")
for field in ("ticker", "side", "count", "price", "client_order_id",
              "time_in_force", "self_trade_prevention_type"):
    check(f"{field} is present", field in pay, str(pay))
check("count is a fixed-point string", pay["count"] == "109.00", str(pay["count"]))
check("price is a 4dp string", pay["price"] == "0.3200", str(pay["price"]))
check("side is a bid", pay["side"] == "bid")
check("self-trade prevention is a documented value",
      pay["self_trade_prevention_type"] in ("taker_at_cross", "maker"),
      pay["self_trade_prevention_type"])
check("time in force is a documented value",
      pay["time_in_force"] in ("fill_or_kill", "good_till_canceled",
                               "immediate_or_cancel"), pay["time_in_force"])

print("\n13. a taker order never rests")
# The drift guard exists so no order sits at a price nobody re-examined;
# good_till_canceled on a taker order would reintroduce exactly that.
check("the default order is immediate_or_cancel",
      pay["time_in_force"] == "immediate_or_cancel", pay["time_in_force"])
check("it is not good_till_canceled", pay["time_in_force"] != "good_till_canceled")
check("post_only is off by default", pay["post_only"] is False)
maker = build_payload(T, 109, 0.32, "cid-1", post_only=True)
check("a post-only order is allowed to rest",
      maker["time_in_force"] == TIME_IN_FORCE_MAKER, maker["time_in_force"])
check("post_only and immediate_or_cancel are never combined",
      not (maker["post_only"] and maker["time_in_force"] == TIME_IN_FORCE_TAKER))
check("the taker and maker settings differ",
      TIME_IN_FORCE_TAKER != TIME_IN_FORCE_MAKER)
check("self-trade prevention is set either way",
      maker["self_trade_prevention_type"] == SELF_TRADE_PREVENTION)

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
