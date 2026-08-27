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


def committed_today() -> float:
    """Dollars a person has confirmed today. Unreadable lines are skipped."""
    p = _ledger_path()
    if not p.exists():
        return 0.0
    day, total = _today(), 0.0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # a torn write must not hide the day's exposure
        if str(r.get("day")) == day:
            total += float(r.get("stake") or 0.0)
    return round(total, 2)


def record_commit(stake: float, ticker: str, contracts: int) -> None:
    """Record a CONFIRMED commitment. Called after a person presses the button."""
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"day": _today(), "stake": round(float(stake), 2),
           "ticker": ticker, "contracts": int(contracts),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def budget(bankroll: float) -> dict:
    """
    What is left to risk today, and the cap a single ticket may use.

    `ticket_pct` is the smaller of the per-ticket cap and whatever remains of the
    day, so the daily limit cannot be stepped over by one large ticket.
    """
    if bankroll <= 0:
        return {"bankroll": 0.0, "committed": 0.0, "daily_remaining": 0.0,
                "ticket_pct": 0.0, "exhausted": True}
    committed = committed_today()
    daily_cap = bankroll * MAX_DAILY_PCT / 100.0
    remaining = max(0.0, daily_cap - committed)
    ticket_pct = min(MAX_TICKET_PCT, 100.0 * remaining / bankroll)
    return {"bankroll": round(bankroll, 2),
            "committed": round(committed, 2),
            "daily_cap": round(daily_cap, 2),
            "daily_remaining": round(remaining, 2),
            "ticket_pct": round(max(ticket_pct, 0.0), 4),
            "exhausted": remaining <= 0.0}
