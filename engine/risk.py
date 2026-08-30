"""
Position limits, enforced in code rather than left to discipline.

Inputs : the account bankroll, and what has already been committed today
Outputs: how much a single ticket may stake, and whether the day is used up

Two caps, both refused server-side:

    MAX_TICKET_PCT   the most one ticket may stake, as a % of bankroll
    MAX_DAILY_PCT    the most all tickets may stake in a day, as a % of bankroll

They are separate because they fail differently. The per-ticket cap stops one
confident-looking price taking an outsized position. The daily cap stops a run
of individually reasonable tickets adding up to a bad day — which is the failure
mode that actually empties accounts, because each step looks defensible.

Why the ledger is on the volume
-------------------------------
Daily exposure has to survive a redeploy or the cap resets to zero every time the
service restarts, which on this project has been roughly twenty times in a day.
A cap that forgets is not a cap. It lives beside the bet log at BET_LOG_PATH.

What counts as committed
------------------------
Only what a person actually confirmed. Tickets that were generated and not taken
do not consume the day's budget, because they never risked anything.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Defaults are the operator's stated limits; both may be overridden per-service.
MAX_TICKET_PCT = float(os.environ.get("KALSHI_MAX_TICKET_PCT", "2.0"))
MAX_DAILY_PCT = float(os.environ.get("KALSHI_MAX_DAILY_PCT", "20.0"))

# Taker unless told otherwise. Makers pay about a quarter of the fee but only if
# they are filled; a resting order that never fills is not a cheaper trade, it is
# no trade, and the model's edge is tied to a price that moves.
MAKER = os.environ.get("KALSHI_MAKER") == "1"


def _ledger_path() -> Path:
    base = Path(os.environ.get("BET_LOG_PATH", "data/processed/bet_log.jsonl")).parent
    return base / "kalshi_exposure.jsonl"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _today_rows() -> list[dict]:
    """Today's confirmed commitments. Unreadable lines are skipped."""
    p = _ledger_path()
    if not p.exists():
        return []
    day, out = _today(), []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # a torn write must not hide the day's exposure
        if str(r.get("day")) == day:
            out.append(r)
    return out


def staked_today() -> float:
    """Everything confirmed today, settled or not. The gross figure."""
    return round(sum(float(r.get("stake") or 0.0) for r in _today_rows()), 2)


def committed_today(open_tickers: set | None = None) -> float:
    """
    Dollars STILL AT RISK today.

    Pass the tickers with an open position and a stake whose market has since
    settled is released back into the day's allowance: the money has come back,
    win or lose, so it is no longer exposure. Without the argument this is the
    gross figure and nothing is released, which is the safe default for any
    caller that cannot see the account.

    The trade-off is deliberate and worth naming. The daily cap exists to stop a
    run of individually reasonable tickets adding up to a bad day. Releasing
    settled stakes weakens that: on a fast-settling day the same allowance can
    be spent more than once. It bounds money AT RISK AT ONCE rather than money
    staked over the day, and `staked_today` keeps the gross figure visible so
    the difference is never hidden.
    """
    rows = _today_rows()
    if open_tickers is not None:
        rows = [r for r in rows if str(r.get("ticker")) in open_tickers]
    return round(sum(float(r.get("stake") or 0.0) for r in rows), 2)


def record_commit(stake: float, ticker: str, contracts: int,
                  client_order_id: str | None = None) -> None:
    """
    Record a CONFIRMED commitment. Called after a person presses the button.

    The client_order_id is stored alongside the ticker because this ledger is
    the only record of which orders came from THIS application. The account
    history cannot answer that: Kalshi does not distinguish an order this page
    sent from one placed by hand in the app, and a record that quietly absorbed
    manual trades would not be a record of the model.
    """
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"day": _today(), "stake": round(float(stake), 2),
           "ticker": ticker, "contracts": int(contracts),
           "client_order_id": client_order_id,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def budget(bankroll: float, open_tickers: set | None = None) -> dict:
    """
    What is left to risk today, and the cap a single ticket may use.

    `ticket_pct` is the smaller of the per-ticket cap and whatever remains of the
    day, so the daily limit cannot be stepped over by one large ticket.
    """
    if bankroll <= 0:
        return {"bankroll": 0.0, "committed": 0.0, "daily_remaining": 0.0,
                "ticket_pct": 0.0, "exhausted": True}
    committed = committed_today(open_tickers)
    daily_cap = bankroll * MAX_DAILY_PCT / 100.0
    remaining = max(0.0, daily_cap - committed)
    ticket_pct = min(MAX_TICKET_PCT, 100.0 * remaining / bankroll)
    return {"bankroll": round(bankroll, 2),
            "committed": round(committed, 2),
            "staked_today": staked_today(),
            "released": round(max(0.0, staked_today() - committed), 2),
            "daily_cap": round(daily_cap, 2),
            "daily_remaining": round(remaining, 2),
            "ticket_pct": round(max(ticket_pct, 0.0), 4),
            "exhausted": remaining <= 0.0}



def placed_here() -> dict:
    """
    What this application actually sent: {"tickers": set, "order_ids": set}.

    Order ids are exact. Tickers are the fallback for rows written before ids
    were stored, and for fills, positions and settlements, which carry a ticker
    but never our client_order_id.
    """
    out = {"tickers": set(), "order_ids": set()}
    p = _ledger_path()
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ticker"):
            out["tickers"].add(str(row["ticker"]))
        if row.get("client_order_id"):
            out["order_ids"].add(str(row["client_order_id"]))
    return out