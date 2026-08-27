"""
Submit ONE confirmed order to Kalshi. Deliberately a separate module.

Inputs : a ticket a person has confirmed, plus credentials from the environment
Outputs: Kalshi's order response, and a record of the commitment

Why this is its own file
------------------------
engine/kalshi.py reads the account and prices a position; nothing in it can
spend money. This module is the only place that can, and keeping it separate
makes that boundary visible in the file listing rather than buried in a branch.

There is no scheduler here, no retry daemon and no batch mode. `create_order`
places exactly one order per call and is reachable only from an endpoint that
requires an explicit confirmation.

Dry run is the default
----------------------
This code has NOT been exercised against the live API, because testing it means
placing a real order. `dry_run=True` logs precisely what would be sent and sends
nothing, which is the honest default for an untested write path against real
money. KALSHI_ARM=1 turns it off.

The ticket is a snapshot, so the market is re-read
------------------------------------------------
Between building a ticket and confirming it, the match can start, the book can
empty and the price can move. All three happened within minutes of the first
dry run. `verify_market` re-reads the market before anything is sent and refuses
if it no longer resembles the one the ticket was priced in. The caps say how
much may be risked; this says whether the thing being bought is still the thing
that was priced.

Idempotency is the part that bites
----------------------------------
A network timeout can happen AFTER Kalshi accepts an order and before the
response arrives. Retrying with a fresh id then doubles the position. The
client_order_id is therefore created ONCE when the ticket is built, travels with
it through the confirmation, and is reused on every attempt - so a retry is
recognised by Kalshi as the same order rather than a second one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

from engine import kalshi, kalshi_match, risk

ORDER_PATH = "/portfolio/events/orders"      # V2; /portfolio/orders is deprecated

# Both are REQUIRED by CreateOrderV2Request. Omitting them is a 400
# missing_parameters, which is how the first live attempt failed.
#
# TAKER orders are immediate_or_cancel: take what is resting at our price and
# cancel the rest. Never good_till_canceled - a resting order is exactly what
# the drift guard exists to prevent, an order sitting at a price nobody
# re-examined while the match moves. The cost is that a partial fill is
# possible, which is the right trade: less exposure than intended is a far
# smaller problem than an unattended order on a book that swings 20c.
#
# fill_or_kill was the alternative. It rules out partials, but on books this
# thin it would mostly just fail, and a partial fill of a sized position is
# still a position we wanted.
TIME_IN_FORCE_TAKER = "immediate_or_cancel"
TIME_IN_FORCE_MAKER = "good_till_canceled"   # a post-only order must be able to rest

# Cancels OUR taker order if it would cross our own resting order, rather than
# trading with ourselves.
SELF_TRADE_PREVENTION = "taker_at_cross"

# How far the ask may move between building a ticket and confirming it, in
# dollars. A ticket is priced at a moment; a tennis book moves while a match is
# being played. Two cents is inside the edge the model needs to justify a bet,
# so a drift past it means the ticket is being confirmed against a market that
# no longer resembles the one it was priced in.
MAX_PRICE_DRIFT = float(os.environ.get("KALSHI_MAX_PRICE_DRIFT", "0.02"))


def armed() -> bool:
    """False means build the payload and send nothing."""
    return os.environ.get("KALSHI_ARM") == "1"


def new_client_order_id() -> str:
    """Created once per ticket, reused across retries. Never regenerate on retry."""
    return str(uuid.uuid4())


def build_payload(ticker: str, count: int, price: float,
                  client_order_id: str, post_only: bool = False) -> dict:
    """
    The V2 order body.

    `price` is a fixed-point dollar string and `count` a string, which is what
    the V2 shape expects. `time_in_force` and `self_trade_prevention_type` are
    both required; leaving them out returns 400 missing_parameters. Buying a contract is a bid on that ticker: to back a
    player you bid on THEIR market, never the opponent's.
    """
    return {
        "ticker": ticker,
        "side": "bid",
        "count": f"{int(count)}.00",
        "price": f"{float(price):.4f}",
        "client_order_id": client_order_id,
        "post_only": bool(post_only),
        "time_in_force": TIME_IN_FORCE_MAKER if post_only else TIME_IN_FORCE_TAKER,
        "self_trade_prevention_type": SELF_TRADE_PREVENTION,
    }



def started(market: dict, now: datetime | None = None) -> bool | None:
    """
    Has the match begun? None means the market did not say.

    Kalshi carries the scheduled start as `occurrence_datetime`, and a match
    market stays ACTIVE right through the match - its own early_close_condition
    reads "this market will close and expire after a winner is declared". So an
    open market is not evidence that play has not started, and the status check
    alone would let an in-play ticket through.
    """
    raw = market.get("occurrence_datetime")
    if not raw:
        return None
    try:
        t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) >= t


def verify_market(ticker: str, count: int, price: float,
                  market: dict | None = None,
                  now: datetime | None = None) -> dict:
    """
    Re-read the market and decide whether this ticket is still the trade.

    Returns {"ok": bool, "reason": str, "ask": float|None, "offered": float|None}.
    Pass `market` to check a snapshot instead of fetching one.

    Three ways a ticket goes stale, all seen in live data within minutes of the
    tickets being built:

    1. THE MARKET CLOSED. The match started or settled. Nothing to buy.
    2. THE BOOK EMPTIED. No ask resting, so the order would rest rather than
       fill, at a price set before whatever emptied the book.
    3. THE PRICE MOVED. This is the one that costs money. If a player drops a
       set the ask falls, and a stale bid fills immediately at a price that only
       looked like value against a probability computed before the match began.
       A bid that has moved the other way simply will not fill, which is merely
       useless - but it is refused too, so that a resting order is never left
       behind at a price nobody re-examined.

    It fails closed: if the market cannot be read, that is a refusal, not a
    shrug. Not knowing the price is not the same as the price being fine.
    """
    out = {"ok": False, "reason": "", "ask": None, "offered": None}
    try:
        m = kalshi.market(ticker) if market is None else market
    except Exception as e:
        out["reason"] = f"could not re-read the market: {type(e).__name__}"
        return out
    if not m:
        out["reason"] = "the market no longer exists"
        return out

    status = str(m.get("status") or "").lower()
    if status and status != "active" and status != "open":
        out["reason"] = f"the market is {status}, not open"
        return out

    # The model priced this before the match. Once play starts the probability
    # the ticket carries is stale in a way no price band can detect: a set down,
    # the ask falls, and a bid built on a pre-match number fills against news the
    # model has never seen. A missing start time is a refusal too - on a money
    # path, not knowing is not the same as knowing it is fine.
    began = started(m, now)
    if began is None:
        out["reason"] = "the market did not say when the match starts"
        return out
    if began:
        out["reason"] = ("the match has already started; this ticket was priced "
                         "before it began")
        return out

    ask = kalshi_match.ask_price(m)
    offered = kalshi_match.ask_size(m)
    out["ask"], out["offered"] = ask, offered
    if ask is None:
        out["reason"] = "nothing is offered at the ask any more"
        return out

    drift = ask - float(price)
    if abs(drift) > MAX_PRICE_DRIFT:
        direction = "up" if drift > 0 else "down"
        out["reason"] = (f"the ask moved {direction} from {float(price):.2f} to "
                         f"{ask:.2f}; rebuild the ticket at the current price")
        return out

    if offered is not None and int(count) > int(offered):
        out["reason"] = (f"only {int(offered)} contracts are offered, "
                         f"not {int(count)}")
        return out

    out["ok"] = True
    return out


def create_order(ticker: str, count: int, price: float, client_order_id: str,
                 dry_run: bool | None = None, post_only: bool = False) -> dict:
    """
    Place one order. Returns what happened, including the payload either way.

    Refuses rather than sends when anything looks wrong: no credentials, a
    nonsensical size or price, or the day's exposure budget already spent. The
    caps are re-checked HERE and not trusted from the caller, because a client
    is the wrong place to enforce a spending limit.
    """
    out = {"sent": False, "dry_run": True, "payload": None,
           "response": None, "error": None, "market": None}

    if not kalshi.configured():
        out["error"] = "Kalshi credentials are not set"
        return out
    if int(count) < 1:
        out["error"] = "count must be at least one contract"
        return out
    if not (0.01 <= float(price) <= 0.99):
        out["error"] = "price must be between 0.01 and 0.99"
        return out
    if not client_order_id:
        out["error"] = "a client_order_id is required for safe retries"
        return out

    cost = int(count) * float(price)
    try:
        bal = kalshi.balance()
        bankroll = float(bal.get("dollars") or 0.0)
    except Exception as e:
        out["error"] = f"could not read balance: {type(e).__name__}: {str(e)[:100]}"
        return out

    b = risk.budget(bankroll)
    if b["exhausted"]:
        out["error"] = (f"daily exposure already used: ${b['committed']:.2f} of "
                        f"${b['daily_cap']:.2f}")
        return out
    ticket_cap = bankroll * risk.MAX_TICKET_PCT / 100.0
    if cost > ticket_cap + 0.01:
        out["error"] = (f"${cost:.2f} exceeds the per-ticket cap of "
                        f"${ticket_cap:.2f}")
        return out
    if cost > b["daily_remaining"] + 0.01:
        out["error"] = (f"${cost:.2f} exceeds what remains today "
                        f"(${b['daily_remaining']:.2f})")
        return out

    # Re-check the market itself, not just our own limits. The caps say how
    # much may be risked; this says whether the thing being bought is still the
    # thing that was priced. A dry run runs it too - a guard that only engages
    # with real money on the line has never been tested.
    fresh = verify_market(ticker, count, price)
    out["market"] = fresh
    if not fresh["ok"]:
        out["error"] = fresh["reason"]
        return out

    payload = build_payload(ticker, count, price, client_order_id, post_only)
    out["payload"] = payload
    dry = (not armed()) if dry_run is None else bool(dry_run)
    out["dry_run"] = dry
    if dry:
        print(f"[kalshi] DRY RUN, nothing sent: {json.dumps(payload)}", flush=True)
        return out

    body = json.dumps(payload).encode()
    path = "/trade-api/v2" + ORDER_PATH
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "tennis-engine/0.1"}
    headers.update(kalshi._sign("POST", path))
    req = urllib.request.Request(kalshi.base_url() + ORDER_PATH, data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out["response"] = json.loads(r.read().decode("utf-8", "replace"))
            out["sent"] = True
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:800]
        except Exception:
            pass
        out["error"] = f"HTTP {e.code}: {detail}"
        return out
    except Exception as e:
        # A timeout here may mean Kalshi ACCEPTED the order. Say so rather than
        # implying nothing happened, and never retry automatically - the same
        # client_order_id must be reused deliberately.
        out["error"] = (f"{type(e).__name__}: {str(e)[:120]} - the order may have "
                        f"been accepted; check the portfolio before retrying, and "
                        f"reuse client_order_id {client_order_id} if you do")
        return out

    # Only a confirmed send consumes the day's budget.
    risk.record_commit(cost, ticker, int(count))
    return out
