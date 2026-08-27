"""
Read ATP results straight from atptour.com, for events the mirror has not filed.

Inputs : a tournament results URL (tournaments.csv supplies one per event)
Outputs: match dicts in the same shape engine/refresh.py builds from the mirror

Why this exists
---------------
`msolonskyi/ManTennisData` scrapes atptour.com and is what the ATP archive is
built from, but it files main-draw results late. On 2026-08-27 it held nothing
for Winston-Salem beyond qualifying, four days after the event began, and had
not filed the second half of Cincinnati at all - including the final. Its
TOURNAMENT list is current; only the matches lag. So this reads the same pages
the mirror reads, for the handful of events whose results are missing.

Parsing is stdlib only. bs4 and lxml are available on a development machine, but
the daily refresh runs on the server and requirements.txt deliberately installs
nothing beyond pandas/flask - adding a parser to the runtime for this would be a
poor trade when html.parser handles a regular document fine.

What the page gives us
----------------------
Each match block carries the round, both players with their ATP player CODES in
the profile links, a winner marker, and per-set games. The codes matter: the
mirror keys players the same way, so a scraped match resolves to the same
player_id by code rather than by fuzzy name match.

Serve statistics are NOT here - they sit behind a per-match stats page. Scraped
matches therefore update the ratings but not the point model, which is the same
trade any results-only source forces.

Courtesy
--------
robots.txt allows crawling (`Allow: /` for all agents). Requests identify
themselves honestly, are made one event at a time with a delay between, and only
for events whose results are actually missing.
"""

from __future__ import annotations

import re
import time
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "Mozilla/5.0 (compatible; tennis-engine/0.1; +results ingest)"
REQUEST_DELAY = 1.5          # seconds between event pages
TIMEOUT = 60

# "Round of 16 - Stadium Court" -> R16. The court suffix varies and is dropped.
ROUND_TEXT = {
    "round of 128": "R128", "round of 64": "R64", "round of 32": "R32",
    "round of 16": "R16", "quarterfinals": "QF", "quarter-finals": "QF",
    "semifinals": "SF", "semi-finals": "SF", "finals": "F", "final": "F",
    "round robin": "RR", "bronze": "BR",
}
PLAYER_HREF = re.compile(r"/en/players/[^/]+/([a-z0-9]+)/overview", re.I)


def _class_of(attrs) -> str:
    for k, v in attrs:
        if k == "class":
            return v or ""
    return ""


class _Results(HTMLParser):
    """
    Walks a results page into match dicts.

    Depth-tracked rather than a flat state machine. The first version closed a
    player on any `</div>`, which meant the inner `name` and `country` divs ended
    the player before its scores were read and the page parsed to zero matches.
    Each region therefore records the div depth it opened at and closes only when
    that depth is unwound.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.matches: list[dict] = []
        self._depth = 0
        self._m = self._p = None
        self._m_depth = self._p_depth = self._score_depth = None
        self._round = None
        self._in_header = self._in_name = False
        self._header_taken = False
        self._header_depth = self._name_depth = None
        self._score_buf = ""

    # ── regions ──────────────────────────────────────────────────────────────
    def handle_starttag(self, tag, attrs):
        if tag == "a" and self._in_name and self._p is not None:
            for k, v in attrs:
                if k == "href":
                    hit = PLAYER_HREF.search(v or "")
                    if hit:
                        self._p["code"] = hit.group(1).lower()
            return
        if tag != "div":
            return

        self._depth += 1
        cls = _class_of(attrs)
        classes = cls.split()
        if "match-header" in classes:
            self._in_header, self._header_depth = True, self._depth
            self._header_taken = False
        elif "match" in classes and self._m is None:
            self._m, self._m_depth = {"round": self._round, "players": []}, self._depth
        elif "stats-item" in classes and self._m is not None and self._p is None:
            self._p = {"name": None, "code": None, "winner": False, "sets": []}
            self._p_depth = self._depth
        elif "winner" in classes and self._p is not None:
            self._p["winner"] = True
        elif "name" in classes and self._p is not None:
            self._in_name, self._name_depth = True, self._depth
        elif "score-item" in classes and self._p is not None:
            self._score_depth, self._score_buf = self._depth, ""

    def handle_endtag(self, tag):
        if tag != "div":
            return
        d = self._depth
        self._depth -= 1

        if self._score_depth == d:
            nums = re.findall(r"\d+", self._score_buf)
            if nums and self._p is not None:
                self._p["sets"].append(tuple(nums[:2]))
            self._score_depth = None
        if self._header_depth == d:
            self._in_header, self._header_depth = False, None
        if self._name_depth == d:
            self._in_name, self._name_depth = False, None
        if self._p_depth == d and self._p is not None:
            if self._p.get("name"):
                self._m["players"].append(self._p)
            self._p, self._p_depth = None, None
        if self._m_depth == d and self._m is not None:
            if len(self._m["players"]) == 2:
                self.matches.append(self._m)
            self._m, self._m_depth = None, None

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._score_depth is not None:
            self._score_buf += " " + text
        elif self._in_header and not self._header_taken:
            # ONLY the first text in a header is the round. The block also holds
            # the match duration ("01:37:13"), and reading that as a round label
            # cleared every round and parsed the page to zero.
            #
            # An unrecognised FIRST label does clear the round rather than
            # leaving the previous one standing: qualifying blocks carry headers
            # this map has no entry for, and inheriting the last main-draw label
            # filed 36 qualifying matches as "Round of 128" on a 96 draw.
            self._header_taken = True
            key = text.split(" - ")[0].strip().lower()
            self._round = ROUND_TEXT.get(key)
            if self._m is not None:
                self._m["round"] = self._round
        elif self._in_name and self._p is not None and not self._p["name"]:
            if not text.startswith("("):          # "(15)" is a seed, not a name
                self._p["name"] = text


def _score_string(winner: dict, loser: dict) -> str:
    """Per-set cells to a Sackmann-style score, winner's games first."""
    out = []
    for w, l in zip(winner["sets"], loser["sets"]):
        wg, lg = w[0], l[0]
        tb = ""
        # A tiebreak set shows the loser's tiebreak points on the losing line.
        if len(l) > 1:
            tb = f"({l[1]})"
        elif len(w) > 1:
            tb = f"({w[1]})"
        out.append(f"{wg}-{lg}{tb}")
    return " ".join(out)


def parse_results(html_text: str) -> list[dict]:
    """Every completed singles match on a results page."""
    p = _Results()
    p.feed(html_text)
    out = []
    for m in p.matches:
        if len(m["players"]) != 2:
            continue
        a, b = m["players"]
        win, lose = (a, b) if a["winner"] else (b, a)
        if not win["winner"]:
            continue                       # still in progress: no winner marked
        score = _score_string(win, lose)
        if not score or not m.get("round"):
            continue          # qualifying, or a header we do not recognise
        out.append({
            "round": m.get("round"),
            "winner_name": win["name"], "winner_code": win["code"],
            "loser_name": lose["name"], "loser_code": lose["code"],
            "score": score,
        })
    return out


def fetch_results(url: str, delay: bool = True) -> list[dict]:
    """Fetch and parse one event's results page. Never raises on a bad page."""
    if delay:
        time.sleep(REQUEST_DELAY)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return parse_results(r.read().decode("utf-8", "replace"))
    except Exception as e:                  # noqa: BLE001
        print(f"  [atp] {url.rsplit('/', 3)[-3]}: {type(e).__name__} {str(e)[:60]}",
              flush=True)
        return []
