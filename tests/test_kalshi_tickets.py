"""
Tests for ticket generation, and for not offering the same bet twice.

    python tests/test_kalshi_tickets.py

The bug these exist for: the tickets endpoint had no idea what the account
already held, so a recommendation stayed on screen with a Place button after it
had been filled. Scanning again re-offered it. The model does not say "back
Parks twice" - it says back Parks, and re-reading the same advice is the same
bet, not a new one.

Positions are the authoritative source because they reflect FILLS. The ledger is
the fallback for an order that was sent but has not filled yet, which would
otherwise look untouched.

No network: the odds feed, the Kalshi client and the ledger are all replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import kalshi, kalshi_match, kalshi_order, risk  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


HELD = "KXWTAMATCH-26AUG27PARANN-PAR"      # 109 contracts already filled
SENT = "KXWTAMATCH-26AUG27TAUPAR-PAR"      # sent, not filled
FRESH = "KXWTAMATCH-26AUG27STAMER-MER"     # untouched

FIXTURES = {"fixtures": [
    {"player_a": "Alycia Parks", "player_b": "Ann Li", "tour": "wta",
     "tournament": "Cleveland", "model_prob_a": 0.46,
     "bet": {"side": "A", "odds": 3.0, "player": "Alycia Parks"}},
    {"player_a": "Diane Parry", "player_b": "Clara Tauson", "tour": "wta",
     "tournament": "Cleveland", "model_prob_a": 0.45,
     "bet": {"side": "A", "odds": 2.68, "player": "Diane Parry"}},
    {"player_a": "Elise Mertens", "player_b": "Yuliia Starodubtseva", "tour": "wta",
     "tournament": "Cleveland", "model_prob_a": 0.76,
     "commence_time": "2030-01-01T05:40:00Z",
     "bet": {"side": "A", "odds": 1.41, "player": "Elise Mertens"}},
]}
FIXTURES["fixtures"][0]["commence_time"] = "2030-01-01T00:00:00Z"
FIXTURES["fixtures"][1]["commence_time"] = "2030-01-01T04:30:00Z"
MARKETS = {
    ("Alycia Parks", "Ann Li"): HELD,
    ("Diane Parry", "Clara Tauson"): SENT,
    ("Elise Mertens", "Yuliia Starodubtseva"): FRESH,
}


def fake_find(a, b, tour, events=None):
    t = MARKETS.get((a, b))
    if not t:
        return {"ok": False, "reason": "no market"}
    mk = lambda tk, nm, ask: {"ticker": tk, "yes_sub_title": nm,
                              "yes_ask_dollars": ask, "yes_ask_size_fp": "5000.00",
                              "status": "active"}
    return {"ok": True, "event": t.rsplit("-", 1)[0], "match_type": "full name",
            "a": mk(t, a, "0.3200"), "b": mk(t + "-OPP", b, "0.6600"),
            "a_name": a, "b_name": b}


kalshi.configured = lambda: True
kalshi.live_mode = lambda: True
kalshi.balance = lambda: {"dollars": 1131.67, "live": True}
kalshi.positions = lambda limit=200: [
    {"ticker": HELD, "position_fp": "109.00"},
    # A closed-out market must not read as still held.
    {"ticker": FRESH, "position_fp": "0.00"},
]
risk.placed_here = lambda: {"tickers": {SENT}, "order_ids": set()}
kalshi_match.open_events = lambda tour: {"e": []}
kalshi_match.find_market = fake_find
kalshi_order.armed = lambda: True

from dashboard import server  # noqa: E402

server.engine = lambda: None
server.live.have_key = lambda: True
server.live.fixtures = lambda eng, tours=None: FIXTURES

d = server.app.test_client().get("/api/kalshi/tickets").get_json()
by = {t["backing"]: t for t in (d.get("tickets") or [])}

# ──────────────────────────────────────────────────────────────────────────────
print("\n1. tickets are produced")
check("the endpoint is available", d.get("available") is True, str(d)[:200])
check("all three picks made tickets", len(by) == 3, str(list(by)))

print("\n2. a filled position is marked, not re-offered blind")
p = by.get("Alycia Parks") or {}
check("the held ticket is flagged", p.get("already_placed") is True, str(p))
check("and says how much is held", p.get("held") == 109.0, str(p.get("held")))

print("\n3. a sent-but-unfilled order also counts as placed")
# Otherwise an order resting on the book looks untouched and gets sent twice.
q = by.get("Diane Parry") or {}
check("the ledger ticker is flagged", q.get("already_placed") is True, str(q))

print("\n4. an untouched market is offered normally")
r = by.get("Elise Mertens") or {}
check("it is not flagged", r.get("already_placed") is False, str(r))
check("and has no holding", not r.get("held"), str(r.get("held")))
# The trap: this ticker IS in positions, with a position of zero.
check("a closed-out position does not count as held",
      r.get("already_placed") is False, str(r))

print("\n5. the flag never blocks a ticket from appearing")
# Marking is the point, not hiding: seeing the holding is what stops the
# accidental second bet, and hiding it would just look like the pick vanished.
check("held tickets still appear", "Alycia Parks" in by)
check("every ticket carries the flag",
      all("already_placed" in t for t in by.values()))

print("\n6. a failure to read positions does not kill the panel")
def boom(*a, **k):
    raise RuntimeError("portfolio unavailable")
kalshi.positions = boom
d2 = server.app.test_client().get("/api/kalshi/tickets").get_json()
check("tickets still render", d2.get("available") is True, str(d2)[:160])
check("and the ledger still flags what it knows",
      any(t["already_placed"] for t in d2["tickets"]), str(d2["tickets"]))

print("")
print("7. tickets are ordered by the model's win probability")
# EV sorting put longshots first: a 12% pick at a long enough price out-EVs a
# 76% one, which parks the least likely bet in the most prominent row.
order = [t["backing"] for t in d["tickets"]]
probs = [t["model_prob"] for t in d["tickets"]]
check("the most likely pick leads", order[0] == "Elise Mertens", str(order))
check("probabilities descend", probs == sorted(probs, reverse=True), str(probs))
check("the longshot is not first", order[0] != "Alycia Parks", str(order))

print("")
print("8. each ticket says when the match starts")
starts = {t["backing"]: t["starts"] for t in d["tickets"]}
check("the odds feed time is used",
      starts["Elise Mertens"] == "2030-01-01T05:40:00Z", str(starts))
check("every ticket has one", all(t.get("starts") for t in d["tickets"]), str(starts))


print("")
print("9. a pick that is not tradeable says why, per pick")
FUTURE = "2030-01-01T00:00:00Z"   # fixed, so the clock cannot age it out
# The question this answers is "where did MY pick go", so every skipped pick
# keeps its own row: who was backed, when it starts, and the two prices that
# disagreed. Grouping by reason read as a summary and lost the pick.
kalshi_match.find_market = lambda a, b, tour, events=None: (
    {"ok": True, "event": "E", "match_type": "full name",
     "a": {"ticker": "KXWTAMATCH-Z-A", "yes_sub_title": a,
           "yes_ask_dollars": "0.9000", "yes_ask_size_fp": "9000.00",
           "status": "active", "occurrence_datetime": FUTURE},
     "b": {"ticker": "KXWTAMATCH-Z-B", "yes_sub_title": b,
           "yes_ask_dollars": "0.1000", "yes_ask_size_fp": "9000.00",
           "status": "active"}, "a_name": a, "b_name": b}
    if a == "Alycia Parks" else {"ok": False, "reason": "no Kalshi market"})
d5 = server.app.test_client().get("/api/kalshi/tickets").get_json()
sk = {x.get("backing"): x for x in d5["skipped"]}
check("every pick is listed separately", len(d5["skipped"]) == 3, str(len(d5["skipped"])))
check("each row names who was backed", all(x.get("backing") for x in d5["skipped"]),
      str(sk.keys()))
check("each row carries the reason", all(x.get("reason") for x in d5["skipped"]))
check("each row carries the start time", all(x.get("starts") for x in d5["skipped"]))
check("each row carries the model probability",
      all(x.get("model_prob") is not None for x in d5["skipped"]))
check("each row carries the bookmaker price",
      all(x.get("book_odds") is not None for x in d5["skipped"]))

# The distinction that matters: a Kalshi price present means the model liked it
# and the exchange price removed the edge. Absent means there was nothing to buy.
priced = sk.get("Alycia Parks") or {}
check("a priced-out pick shows Kalshi's ask", priced.get("price") == 0.90, str(priced))
check("and blames the price, not the market",
      "edge" in str(priced.get("reason")), str(priced.get("reason")))
unlisted = sk.get("Elise Mertens") or {}
check("a fixture with no market has no Kalshi price",
      unlisted.get("price") is None, str(unlisted))


print(f"\n{'='*54}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*54}\n")
sys.exit(1 if FAIL else 0)
