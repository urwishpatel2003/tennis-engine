"""
Tests for the Kalshi reporting endpoint.

    python tests/test_kalshi_report.py

Two things here are easy to get wrong and expensive to get wrong quietly:

1. UNITS. Kalshi sends `revenue` in CENTS while every cost field beside it is in
   fixed-point DOLLARS. Mixing them reports a P/L off by a factor of a hundred,
   in a panel whose entire job is to tell you whether you are up or down.

2. WHAT COUNTS AS A BET. An order that rested and never filled cost nothing and
   proved nothing. Counting sent orders as bets would flatter the record, so the
   record is built from settlements only.

No network: the Kalshi client is replaced with fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import server  # noqa: E402
from engine import kalshi, risk  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# A won market and a lost one, in Kalshi's real shapes.
#   won : 100 contracts at 0.40 = $40 cost, settles at $1 each = $100 revenue
#         revenue arrives as 10000 CENTS, fee $0.35  ->  P/L +$59.65
#   lost: 50 contracts at 0.60 = $30 cost, settles worthless -> P/L -$30.00
SETTLEMENTS = [
    {"ticker": "KXWTAMATCH-26AUG27TAUPAR-PAR", "market_result": "yes",
     "yes_count_fp": "100.00", "yes_total_cost_dollars": "40.0000",
     "no_count_fp": "0.00", "no_total_cost_dollars": "0.0000",
     "revenue": 10000, "fee_cost": "0.3500",
     "settled_time": "2026-08-28T06:00:00Z"},
    {"ticker": "KXWTAMATCH-26AUG27PARANN-PAR", "market_result": "no",
     "yes_count_fp": "50.00", "yes_total_cost_dollars": "30.0000",
     "no_count_fp": "0.00", "no_total_cost_dollars": "0.0000",
     "revenue": 0, "fee_cost": "0.0000",
     "settled_time": "2026-08-28T02:00:00Z"},
]
ORDERS = [
    {"order_id": "o1", "ticker": "KXWTAMATCH-26AUG27TAUPAR-PAR", "status": "executed",
     "yes_price_dollars": "0.4000", "initial_count_fp": "100.00",
     "fill_count_fp": "100.00", "remaining_count_fp": "0.00",
     "taker_fees_dollars": "0.3500", "maker_fees_dollars": "0.0000",
     "created_time": "2026-08-27T18:00:00Z"},
    # Sent, rested, never traded. Not a bet.
    {"order_id": "o2", "ticker": "KXATPMATCH-26AUG27AAABBB-AAA", "status": "canceled",
     "yes_price_dollars": "0.3000", "initial_count_fp": "80.00",
     "fill_count_fp": "0.00", "remaining_count_fp": "80.00",
     "taker_fees_dollars": "0.0000", "maker_fees_dollars": "0.0000",
     "created_time": "2026-08-27T18:05:00Z"},
]
POSITIONS = [
    {"ticker": "KXWTAMATCH-26AUG28XXXYYY-XXX", "position_fp": "25.00",
     "market_exposure_dollars": "12.5000", "total_traded_dollars": "12.5000",
     "realized_pnl_dollars": "0.0000", "fees_paid_dollars": "0.1000",
     "last_updated_ts": "2026-08-27T19:00:00Z"},
    {"ticker": "KXWTAMATCH-26AUG28ZZZWWW-ZZZ", "position_fp": "0.00",
     "market_exposure_dollars": "0.0000", "total_traded_dollars": "5.0000",
     "realized_pnl_dollars": "0.0000", "fees_paid_dollars": "0.0000",
     "last_updated_ts": "2026-08-27T19:00:00Z"},
]

kalshi.configured = lambda: True
kalshi.live_mode = lambda: True
kalshi.balance = lambda: {"dollars": 1168.21, "live": True}
kalshi.orders = lambda limit=200, status=None: ORDERS
kalshi.fills = lambda limit=200: []
kalshi.positions = lambda limit=200: POSITIONS
kalshi.settlements = lambda limit=200: SETTLEMENTS
kalshi.market = lambda t: {"yes_sub_title": "Diane Parry"}

# The report defaults to "what this page sent", so the fixtures need a ledger
# saying these tickers came from here. Section 8 varies this deliberately.
ALL_TICKERS = {str(r.get("ticker")) for r in
               (SETTLEMENTS + ORDERS + POSITIONS)}
risk.placed_here = lambda: {"tickers": set(ALL_TICKERS), "order_ids": set()}

d = server.app.test_client().get("/api/kalshi/report").get_json()

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. the report renders")
check("it is available", d.get("available") is True, str(d)[:200])
check("the balance comes through", d.get("balance") == 1168.21)
check("the arm state is reported", "armed" in d)
check("the caps are reported", (d.get("caps") or {}).get("ticket_pct") is not None)

print("\n2. cents and dollars are not mixed")
won = [s for s in d["settlements"] if s["pl"] > 0][0]
check("a $40 stake returning $100 nets +$59.65", won["pl"] == 59.65, str(won))
check("revenue is read as dollars, not cents", won["revenue"] == 100.0, str(won))
check("the cost is unchanged", won["cost"] == 40.0, str(won))
lost = [s for s in d["settlements"] if s["pl"] < 0][0]
check("a losing market returns nothing", lost["revenue"] == 0.0, str(lost))
check("and loses exactly the stake", lost["pl"] == -30.0, str(lost))

print("\n3. the record totals correctly")
r = d["record"]
check("both markets are counted", r["settled"] == 2, str(r))
check("one won, one lost", r["won"] == 1 and r["lost"] == 1, str(r))
check("staked is the sum of costs", r["staked"] == 70.0, str(r))
check("realised P/L is +$29.65", r["realised_pl"] == 29.65, str(r))
check("ROI is on money risked", r["roi_pct"] == round(100 * 29.65 / 70.0, 1), str(r))
check("fees are totalled", r["fees"] == 0.35, str(r))

print("\n4. an unfilled order is not a bet")
check("both orders are listed", r["orders_sent"] == 2, str(r))
check("only the filled one counts as filled", r["orders_filled"] == 1, str(r))
check("the resting one is counted separately", r["orders_unfilled"] == 1, str(r))
check("the record ignores orders entirely", r["settled"] == len(SETTLEMENTS))
# The trap: 2 orders and 2 settlements are the same count here by coincidence.
# What must not happen is the unfilled order appearing as a settled bet.
check("no settlement came from the unfilled ticker",
      all("AAABBB" not in s["ticker"] for s in d["settlements"]))

print("\n5. a closed-out position is not shown as open")
check("only the non-zero position is listed", len(d["positions"]) == 1, str(d["positions"]))
check("open exposure matches it", r["open_exposure"] == 12.5, str(r))

print("\n6. tickers are resolved to names")
check("a position shows who is backed",
      d["positions"][0]["backing"] == "Diane Parry", str(d["positions"][0]))

print("\n7. one broken section does not blank the page")
def boom(*a, **k):
    raise RuntimeError("kalshi is down")
kalshi.settlements = boom
d2 = server.app.test_client().get("/api/kalshi/report").get_json()
check("the report still renders", d2.get("available") is True)
check("the failure is reported", "settlements" in (d2.get("errors") or {}), str(d2.get("errors")))
check("the sections that worked still have data", len(d2.get("orders") or []) == 2)
check("the record degrades to empty rather than lying",
      d2["record"]["settled"] == 0 and d2["record"]["realised_pl"] == 0, str(d2["record"]))

print("")
print("8. the record is scoped to what THIS page sent")
# Kalshi's portfolio endpoints return the whole account. A manual trade placed
# by hand in the app must not be credited to the model.
MANUAL = {"ticker": "KXNBA-26AUG27LALBOS-LAL", "market_result": "yes",
          "yes_count_fp": "10.00", "yes_total_cost_dollars": "5.0000",
          "no_count_fp": "0.00", "no_total_cost_dollars": "0.0000",
          "revenue": 1000, "fee_cost": "0.0000",
          "settled_time": "2026-08-28T04:00:00Z"}
kalshi.settlements = lambda limit=200: SETTLEMENTS + [MANUAL]
kalshi.orders = lambda limit=200, status=None: ORDERS
risk.placed_here = lambda: {
    "tickers": {"KXWTAMATCH-26AUG27TAUPAR-PAR", "KXWTAMATCH-26AUG27PARANN-PAR",
                "KXATPMATCH-26AUG27AAABBB-AAA"},
    "order_ids": set()}

cli = server.app.test_client()
page = cli.get("/api/kalshi/report").get_json()
check("the default scope is this page", page.get("scope") == "page", str(page.get("scope")))
check("the manual NBA trade is excluded", len(page["settlements"]) == 2,
      str([s["ticker"] for s in page["settlements"]]))
check("its winnings are NOT in the P/L", page["record"]["realised_pl"] == 29.65,
      str(page["record"]))
check("the hidden count is reported",
      page["scope_counts"]["settlements"]["account"] == 3
      and page["scope_counts"]["settlements"]["shown"] == 2,
      str(page["scope_counts"]))

acct = cli.get("/api/kalshi/report?scope=account").get_json()
check("the whole account can still be seen", len(acct["settlements"]) == 3,
      str(len(acct["settlements"])))
check("and then the manual trade does count",
      acct["record"]["realised_pl"] == round(29.65 + 5.0, 2), str(acct["record"]))

tennis = cli.get("/api/kalshi/report?scope=tennis").get_json()
check("tennis scope keeps both tennis markets", len(tennis["settlements"]) == 2)
check("tennis scope drops the NBA market",
      all("KXNBA" not in s["ticker"] for s in tennis["settlements"]))

check("an unknown scope falls back to the safe one",
      cli.get("/api/kalshi/report?scope=nonsense").get_json()["scope"] == "page")

print("")
print("9. an empty ledger shows nothing rather than everything")
# The dangerous failure: a missing ledger reading as "no filter" and reporting
# the entire account as the model's record.
risk.placed_here = lambda: {"tickers": set(), "order_ids": set()}
empty = cli.get("/api/kalshi/report").get_json()
check("no ledger means no claimed trades", empty["record"]["settled"] == 0,
      str(empty["record"]))
check("and the account total is still visible for comparison",
      empty["scope_counts"]["settlements"]["account"] == 3,
      str(empty["scope_counts"]))


print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
