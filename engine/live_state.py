"""
Win probability from a match ALREADY IN PROGRESS.

Inputs : the two point-on-serve probabilities, plus the current score
Outputs: P(player A wins the match) given that score

Everything in engine/markov.py starts from 0-0. That is the right shape for
prediction, and useless the moment a match is under way: `match_win_prob` cannot
tell you that a break in the third set moved a player from 62% to 81%, because
it has no way to be told the score.

This module adds the score-conditional entry point. It is the same exact
Barnett-Clarke chain, entered part-way through rather than at the start, so it
inherits the base model's calibration instead of introducing a second one. There
is deliberately no new estimation here and no smoothing: given the score and the
two serve probabilities, the answer is arithmetic, not a fit.

Kept out of markov.py on purpose. That module's set-level DP is the hot path —
`invert_to_target` calls it ~40 times per bisection — and none of this belongs in
it.

The live feed supplies exactly the state this needs: server, point score, game
score, set score and a tiebreak flag.
"""

from __future__ import annotations

from functools import lru_cache

from engine.markov import (
    _next_set_first_server,
    _q,
    _set_outcome_split,
    _tb_rec,
    _tb_server_is_a,
    game_prob,
    tiebreak_prob,
)

# Tennis point names to a count of points won. "AD" is a fourth point held with
# the advantage, which the recursion below treats as 4 against 3.
POINT_INDEX = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4, "ADV": 4}


def point_index(value: object) -> int:
    """Coerce a scoreboard point value ('0', '15', '40', 'AD', 3) to a count."""
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    return POINT_INDEX.get(str(value).strip().upper(), 0)


@lru_cache(maxsize=200_000)
def game_prob_from(p: float, a: int, b: int) -> float:
    """
    P(the SERVER wins this game) from point counts a-b, server's points first.

    a and b are counts, not scoreboard values: 0/15/30/40 are 0/1/2/3, and a
    player holding advantage is on 4 against 3. Once both are past 40 the state
    collapses onto the standard deuce formula, so 40-40, AD-40 and 40-AD are the
    only three cases that matter however long the game runs.
    """
    p = _q(p)
    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0
    if a >= 3 and b >= 3:
        # P(win from deuce): win two in a row, or split and return to deuce.
        denom = p * p + (1.0 - p) * (1.0 - p)
        p_deuce = (p * p) / denom if denom > 0 else 0.5
        diff = a - b
        if diff == 0:
            return p_deuce
        if diff == 1:                       # server holds advantage
            return p + (1.0 - p) * p_deuce
        return p * p_deuce                  # receiver holds advantage
    return p * game_prob_from(p, a + 1, b) + (1.0 - p) * game_prob_from(p, a, b + 1)


@lru_cache(maxsize=200_000)
def _set_from(
    ha: float, hb: float, tb_a: float, ga: int, gb: int,
    a_serving: bool, has_tb: bool,
) -> tuple[tuple[bool, bool, float], ...]:
    """
    Set outcomes from the START of a game at score ga-gb, `a_serving` serving it.

    Returns the same shape as markov._set_outcome_split — (A won the set, next
    set's first server is A, probability) — so the match-level DP below can
    consume either interchangeably.
    """
    # Set already decided.
    if ga >= 6 and ga - gb >= 2:
        return ((True, a_serving, 1.0),)
    if gb >= 6 and gb - ga >= 2:
        return ((False, a_serving, 1.0),)
    if ga == 7 or gb == 7:                  # 7-5 and 7-6 are terminal
        return ((ga > gb, a_serving, 1.0),)

    if ga == 6 and gb == 6:
        if has_tb:
            # A tiebreak counts as one more game, so the serve rotates once.
            return ((True, not a_serving, tb_a), (False, not a_serving, 1.0 - tb_a))
        # Advantage set: no tiebreak, play on until someone leads by two.
        return _set_from(ha, hb, tb_a, 5, 5, a_serving, False)

    p_a_wins_game = ha if a_serving else (1.0 - hb)
    agg: dict[tuple[bool, bool], float] = {}
    for a_won, p in ((True, p_a_wins_game), (False, 1.0 - p_a_wins_game)):
        if p <= 0.0:
            continue
        nga, ngb = (ga + 1, gb) if a_won else (ga, gb + 1)
        for won_set, nxt_first, ps in _set_from(
            ha, hb, tb_a, nga, ngb, not a_serving, has_tb
        ):
            key = (won_set, nxt_first)
            agg[key] = agg.get(key, 0.0) + p * ps
    return tuple((k[0], k[1], v) for k, v in agg.items())


@lru_cache(maxsize=200_000)
def _match_from(
    ha: float, hb: float, tb_a: float, sa: int, sb: int,
    a_first: bool, sets_to_win: int, final_set_tb: bool,
) -> float:
    """P(A wins the match) from a set score, with sets starting at 0-0 games."""
    if sa >= sets_to_win:
        return 1.0
    if sb >= sets_to_win:
        return 0.0
    is_decider = (sa == sets_to_win - 1) and (sb == sets_to_win - 1)
    total = 0.0
    for a_won, next_first, p in _set_outcome_split(
        ha, hb, tb_a, a_first, True if not is_decider else final_set_tb
    ):
        total += p * _match_from(
            ha, hb, tb_a, sa + (1 if a_won else 0), sb + (0 if a_won else 1),
            next_first, sets_to_win, final_set_tb,
        )
    return total


def win_prob_from_state(
    pa: float,
    pb: float,
    *,
    sets_a: int = 0,
    sets_b: int = 0,
    games_a: int = 0,
    games_b: int = 0,
    points_a: object = 0,
    points_b: object = 0,
    a_serving: bool = True,
    in_tiebreak: bool = False,
    best_of: int = 3,
    final_set_tb: bool = True,
) -> float:
    """
    P(A wins the match) given the current score.

    `pa`/`pb` are the point-on-serve probabilities the rest of the engine already
    produces — pass the RECONCILED ones from a prediction, not the raw serve
    rates, so the live number agrees with the headline price.

    `a_serving` refers to the point about to be played. Point values accept
    either scoreboard strings ('0', '15', '40', 'AD') or counts.

    With every score argument left at its default this agrees with
    markov.match_win_prob to within ~5e-6 — the invariant that says the
    recursion was entered correctly, and the first thing
    tests/test_live_state.py checks. Not bit-exact, because the base chain
    quantises to 1e-4 and accumulates that rounding differently through an
    iterative set construction than through this recursion. Five parts per
    million is far below the model's real precision and invisible at the one
    decimal place a win probability is ever shown to.
    """
    pa, pb = _q(pa), _q(pb)
    sets_to_win = 3 if int(best_of) == 5 else 2
    # Quantise the hold/tiebreak probabilities exactly as markov.set_distribution
    # does. Skipping this is not a rounding nicety: the base chain quantises to
    # 1e-4 for cache friendliness, so an un-quantised recursion lands on slightly
    # different set probabilities and the live number drifts from the headline
    # price by ~1e-4 per set. A live figure that disagrees with the price beside
    # it is the same defect as a fair line disagreeing with its own probability.
    ha = _q(game_prob(pa))
    hb = _q(game_prob(pb))
    tb_a = _q(tiebreak_prob(pa, pb))

    sa, sb = int(sets_a), int(sets_b)
    if sa >= sets_to_win:
        return 1.0
    if sb >= sets_to_win:
        return 0.0

    ga, gb = int(games_a), int(games_b)
    pta, ptb = point_index(points_a), point_index(points_b)
    is_decider = (sa == sets_to_win - 1) and (sb == sets_to_win - 1)
    has_tb = True if not is_decider else bool(final_set_tb)

    # ── the point in progress ────────────────────────────────────────────────
    if in_tiebreak:
        target = 10 if (is_decider and not final_set_tb) else 7
        # _tb_rec assumes A served the tiebreak's first point. Which player that
        # was is recoverable rather than assumed: the rotation fixes who serves
        # point n, so comparing that against who is serving NOW identifies it.
        played = pta + ptb
        a_started_tb = (a_serving == _tb_server_is_a(played))
        if a_started_tb:
            p_a_takes_set = _tb_rec(pta, ptb, pa, pb, target)
        else:
            p_a_takes_set = 1.0 - _tb_rec(ptb, pta, pb, pa, target)
        # A tiebreak is the set's last game; the loser of it serves first next.
        outcomes = ((True, not a_serving, p_a_takes_set),
                    (False, not a_serving, 1.0 - p_a_takes_set))
    else:
        if a_serving:
            p_a_wins_game = game_prob_from(pa, pta, ptb)
        else:
            p_a_wins_game = 1.0 - game_prob_from(pb, ptb, pta)

        agg: dict[tuple[bool, bool], float] = {}
        for a_won, p in ((True, p_a_wins_game), (False, 1.0 - p_a_wins_game)):
            if p <= 0.0:
                continue
            nga, ngb = (ga + 1, gb) if a_won else (ga, gb + 1)
            for won_set, nxt_first, ps in _set_from(
                ha, hb, tb_a, nga, ngb, not a_serving, has_tb
            ):
                key = (won_set, nxt_first)
                agg[key] = agg.get(key, 0.0) + p * ps
        outcomes = tuple((k[0], k[1], v) for k, v in agg.items())

    # ── carry each set outcome into the match ────────────────────────────────
    total = 0.0
    for won_set, nxt_first, p in outcomes:
        total += p * _match_from(
            ha, hb, tb_a, sa + (1 if won_set else 0), sb + (0 if won_set else 1),
            nxt_first, sets_to_win, bool(final_set_tb),
        )
    return float(min(max(total, 0.0), 1.0))


def leverage(pa: float, pb: float, **state) -> float:
    """
    How much the next point is worth: |P(A wins | A takes it) − P(A wins | B does)|.

    This is what makes a live win-probability curve worth looking at. A point at
    30-0 in a 5-1 set is worth almost nothing; break point at 4-5 in a decider
    can be worth twenty points of win probability, and the number says which is
    which rather than leaving it to commentary.
    """
    pta = point_index(state.get("points_a", 0))
    ptb = point_index(state.get("points_b", 0))
    a_wins = dict(state, points_a=pta + 1, points_b=ptb)
    b_wins = dict(state, points_a=pta, points_b=ptb + 1)
    return abs(win_prob_from_state(pa, pb, **a_wins)
               - win_prob_from_state(pa, pb, **b_wins))
