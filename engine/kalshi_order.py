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

from engine import kalshi, risk

ORDER_PATH = "/portfolio/events/orders"      # V2; /portfolio/orders is deprecated


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
    the V2 shape expects. Buying a contract is a bid on that ticker: to back a
    player you bid on THEIR market, never the opponent's.
    """
    return {
        "ticker": ticker,
        "side": "bid",
        "count": f"{int(count)}.00",
        "price": f"{float(price):.4f}",
        "client_order_id": client_order_id,
        "post_only": bool(post_only),
    }


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
           "response": None, "error": None}

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
            detail = e.read().decode("utf-8", "replace")[:250]
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
