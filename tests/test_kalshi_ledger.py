"""
Tests for realised P/L: the fills-based ledger.

    python tests/test_kalshi_ledger.py

The bug these exist for: the record was built from SETTLEMENTS alone, and a
settlement is only one of the ways a position closes. A cash-out - selling the
contracts back before the match resolves - never settles, so it produced no
settlement row, and because the position was then closed it also dropped out of
open positions. A real cashed-out winner vanished from the page completely.

The fixtures are the exact shapes of a real account, taken from the live API:

    BUY  132 @ 0.36   ->  cost      $47.52
    SELL 132 @ 0.92   ->  proceeds $121.44
                          gross     $73.92   == Kalshi's realized_pnl_dollars

Two traps the arithmetic has to avoid:

1. DOUBLE-COUNTING COST. Fills carry a cost and settlements carry one too.
   Adding both charges the stake twice.
2. UNITS. Settlement `revenue` is in CENTS while every cost field beside it is
   in fixed-point DOLLARS.

No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import kalshi, kalshi_order, risk  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


CASH = "KXWTAMATCH-26AUG30PODWAL-POD"     # bought and sold back: a cash-out
SETL = "KXWTAMATCH-26AUG30GOLPAR-PAR"     # bought and left to settle
PART = "KXATPMATCH-26AUG30DARWEN-DAR"     # bought 52, sold 20, 32 still open
NAMES = {CASH: "Podoroska", SETL: "Golubic", PART: "Darderi"}

kalshi.configured = lambda: True
kalshi.live_mode = lambda: True
kalshi.balance = lambda: {"dollars": 1755.44, "live": True}
kalshi.market = lambda t: {"yes_sub_title": NAMES.get(t, t),
                           "yes_bid_dollars": "0.5000"}
kalshi.orders = lambda limit=200, status=None: []
kalshi.fills = lambda limit=200: [
    # EXACT rows from the live account. The sell fee is 0.6801 and the buy fee
    # 2.1289; they sum to 2.8090, which is also what the settlement reports.
    {"ticker": CASH, "book_side": "ask", "outcome_side": "no", "count_fp": "132.00",
     "yes_price_dollars": "0.9200", "no_price_dollars": "0.0800",
     "fee_cost": "0.680100", "created_time": "2026-08-31T21:15:21Z"},
    {"ticker": CASH, "book_side": "bid", "outcome_side": "yes", "count_fp": "132.00",
     "yes_price_dollars": "0.3600", "no_price_dollars": "0.6400",
     "fee_cost": "2.128900", "created_time": "2026-08-31T16:10:37Z"},
    {"ticker": SETL, "book_side": "bid", "outcome_side": "yes", "count_fp": "77.00",
     "yes_price_dollars": "0.5200", "fee_cost": "1.345400",
     "created_time": "2026-08-31T15:00:00Z"},
    {"ticker": PART, "book_side": "bid", "outcome_side": "yes", "count_fp": "52.00",
     "yes_price_dollars": "0.7400", "fee_cost": "0.700400",
     "created_time": "2026-08-31T18:00:00Z"},
    {"ticker": PART, "book_side": "ask", "outcome_side": "no", "count_fp": "20.00",
     "yes_price_dollars": "0.8400", "fee_cost": "0.200000",
     "created_time": "2026-08-31T20:00:00Z"},
]
kalshi.settlements = lambda limit=200: [
    {"ticker": SETL, "market_result": "yes", "yes_count_fp": "77.00",
     "yes_total_cost_dollars": "40.0400", "no_count_fp": "0.00",
     "no_total_cost_dollars": "0.0000", "revenue": 7700, "fee_cost": "0.0000",
     "settled_time": "2026-08-31T19:00:00Z"},
    # The trap: a market that was CASHED OUT and then settled anyway. Kalshi
    # reports a settlement row for it even though the position was already
    # flat. Counting those contracts as closed on top of the ones already sold
    # books more closed than were ever bought, and turns a winner into a loss.
    # The real settlement: it reports BOTH sides (yes_n and no_n are each 132,
    # because Kalshi models the sell as buying 132 NO), zero revenue, and a fee
    # of 2.8090 that repeats the fills rather than adding to them.
    {"ticker": CASH, "market_result": "yes", "yes_count_fp": "132.00",
     "yes_total_cost_dollars": "47.520000", "no_count_fp": "132.00",
     "no_total_cost_dollars": "10.560000", "revenue": 0, "fee_cost": "2.809000",
     "settled_time": "2026-08-31T23:00:00Z"},
]
kalshi.positions = lambda limit=200: [
    {"ticker": CASH, "position_fp": "0.00", "market_exposure_dollars": "0.0000",
     "total_traded_dollars": "58.0800", "realized_pnl_dollars": "73.920000",
     "fees_paid_dollars": "1.2401"},
    {"ticker": PART, "position_fp": "32.00", "market_exposure_dollars": "23.6800",
     "total_traded_dollars": "23.6800", "realized_pnl_dollars": "0.0000",
     "fees_paid_dollars": "0.9004"},
]
risk.placed_here = lambda: {"tickers": {CASH, SETL, PART}, "order_ids": set()}
kalshi_order.open_tickers = lambda: {PART}

from dashboard import server  # noqa: E402

d = server.app.test_client().get("/api/kalshi/report").get_json()
by = {r["backing"]: r for r in d["settlements"]}
rec = d["record"]

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. a cash-out reaches the record")
check("the cashed-out trade is listed", "Podoroska" in by, str(list(by)))
pod = by.get("Podoroska", {})
check("it is labelled as a cash-out", "cashed out" in pod.get("kind", ""), str(pod))
check("cost is the BUY price", pod.get("cost") == 47.52, str(pod.get("cost")))
check("returned is the SELL proceeds", pod.get("revenue") == 121.44, str(pod))
# Kalshi's own realized_pnl_dollars for this market is 73.92, before fees.
check("P/L is gross minus fees", abs(pod.get("pl", 0) - (73.92 - 2.809)) < 0.02,
      str(pod.get("pl")))
check("the record counts it", rec["cashed_out"] >= 1, str(rec))

print("\n2. a settled market is unchanged")
gol = by.get("Golubic", {})
check("it is labelled settled", gol.get("kind") == "settled", str(gol))
check("77 contracts at 0.52 cost $40.04", gol.get("cost") == 40.04, str(gol))
check("revenue is read as dollars, not cents", gol.get("revenue") == 77.0, str(gol))
check("P/L is 77.00 - 40.04 - 1.35", abs(gol.get("pl", 0) - 35.61) < 0.02, str(gol))

print("\n3. cost is never counted twice")
# The settled market has BOTH a buy fill and a settlement carrying a cost.
# Charging the stake twice would put P/L at about -$4.43 instead of +$35.61.
check("the settled stake is charged once", gol.get("cost") == 40.04, str(gol))
check("and not doubled", gol.get("cost") != 80.08, str(gol))
check("staked totals each trade once",
      abs(rec["staked"] - (47.52 + 40.04 + 14.80)) < 0.02, str(rec["staked"]))

print("\n4. a partial exit realises only the part sold")
part = by.get("Darderi", {})
check("only the 20 sold are closed", part.get("contracts") == 20, str(part))
check("at the average cost of 0.74", abs(part.get("cost", 0) - 14.80) < 0.02, str(part))
check("P/L is on the sold portion", abs(part.get("pl", 0) - 1.10) < 0.02, str(part))
check("the rest stays open", rec["open_positions"] == 1, str(rec))
check("open exposure is the remaining 32", rec["open_exposure"] == 23.68, str(rec))

print("\n5. the totals add up")
total = round(sum(r["pl"] for r in d["settlements"]), 2)
check("realised is the sum of the closed rows", rec["realised_pl"] == total,
      f'{rec["realised_pl"]} vs {total}')
check("three trades are closed", rec["settled"] == 3, str(rec))
check("by-sport P/L still reconciles",
      abs(sum(b["realised_pl"] for b in d["by_sport"]) - rec["realised_pl"]) < 0.02,
      str(d["by_sport"]))

print("\n6. an open position is not counted as closed")
check("the open ticker has no settled row for its full size",
      by.get("Darderi", {}).get("contracts") != 52, str(part))
check("nothing unsold is in the realised figure",
      rec["realised_pl"] < 47.52 + 77.0, str(rec))

print("")
print("7. the real Podoroska trade, from the live rows")
# The deployed code produced -$26.74 on this exact data:
#   closed = 132 sold + 264 settled = 396 -> basis 0.36*396 = 142.56
#   fees   = 2.809 fills + 2.809 settlement = 5.618  (counted twice)
#   121.44 - 142.56 - 5.618 = -26.74
# The truth: bought 132 at 0.36, sold at 0.92, gross 73.92, fees 2.809.
GROSS, FEES = 73.92, 2.809
check("contracts closed never exceed contracts bought",
      pod.get("contracts") == 132, str(pod.get("contracts")))
check("cost basis is the buy, charged once",
      pod.get("cost") == 47.52, str(pod.get("cost")))
check("returned is the sell proceeds",
      pod.get("revenue") == 121.44, str(pod.get("revenue")))
check("the settlement fee is not added on top of the fill fees",
      abs(pod.get("fee", 0) - FEES) < 0.01, str(pod.get("fee")))
check("gross matches Kalshi's own realized_pnl_dollars",
      abs((pod.get("revenue", 0) - pod.get("cost", 0)) - GROSS) < 0.01,
      str(pod.get("revenue", 0) - pod.get("cost", 0)))
check("net P/L is +$71.11",
      abs(pod.get("pl", 0) - (GROSS - FEES)) < 0.02, str(pod.get("pl")))
check("it is NOT the -$26.74 that shipped", pod.get("pl", 0) > 0,
      str(pod.get("pl")))
check("it reads as a cash-out", "cashed out" in pod.get("kind", ""),
      str(pod.get("kind")))

print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
