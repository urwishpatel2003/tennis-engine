"""
Hierarchical tennis scoring model — point → game → tiebreak → set → match.

This is the Barnett–Clarke Markov chain, solved EXACTLY (closed form + memoised
recursion), not by Monte Carlo. Exactness matters: the whole point of this module
is to hand `predict.py` a full joint distribution over set scores, from which the
game handicap, total-games line and exact-scoreline market all fall out
analytically. A simulation would put noise on all three.

Inputs : two point-win-on-serve probabilities, pa (player A serving) and
         pb (player B serving), plus best_of and the tiebreak convention.
Outputs: hold probabilities, set-score distributions, match win probability and
         the derived games/sets distributions.

Key modelling assumption (standard, and empirically close): points are i.i.d.
given the server. Real tennis has mild point-importance and momentum effects;
they are second-order next to the serve/return gap and are deliberately not
modelled here — the calibration step in predict.py absorbs the residual.

All functions are pure and cached, so the ~20k-call backtest loop stays fast.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Probabilities are rounded to this many decimals before entering the cached
# recursions. 1e-4 granularity is far finer than the model's real precision and
# keeps the LRU caches from exploding during a full-season backtest.
_Q = 4


def _q(p: float) -> float:
    """Clamp into a safe open interval and quantise for cache friendliness."""
    return round(min(max(float(p), 1e-6), 1.0 - 1e-6), _Q)


# ──────────────────────────────────────────────────────────────────────────────
# Game
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=200_000)
def game_prob(p: float) -> float:
    """
    Probability the SERVER wins a service game, given point-win probability `p`.

    Closed form over the four ways to win by two from 0-0:
        40-0   : p^4
        40-15  : C(4,1) p^4 q
        40-30  : C(5,2) p^4 q^2
        deuce  : C(6,3) p^3 q^3 · p^2/(p^2+q^2)
    where q = 1-p and the deuce term is the standard two-point-swing geometric sum.
    """
    p = _q(p)
    q = 1.0 - p
    deuce = (p * p) / (p * p + q * q)
    return (
        p**4
        + 4 * p**4 * q
        + 10 * p**4 * q**2
        + 20 * p**3 * q**3 * deuce
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tiebreak
# ──────────────────────────────────────────────────────────────────────────────
def _tb_server_is_a(points_played: int) -> bool:
    """
    Who serves point `points_played` (0-indexed) of a tiebreak?

    Convention: A serves point 0, then serve alternates in pairs (B,B / A,A / …).
    server(i) is A  iff  ((i+1)//2) % 2 == 0.
    """
    return ((points_played + 1) // 2) % 2 == 0


@lru_cache(maxsize=500_000)
def _tb_rec(a: int, b: int, pa: float, pb: float, target: int) -> float:
    """P(A wins the tiebreak) from score a-b. `target` is 7 (or 10 for a super-TB)."""
    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0

    # At any level tie from (target-1) onward the chain becomes periodic: over the
    # next two points A serves once and B serves once regardless of parity, so
    #   P(A) = w / (w + l),  w = pa(1-pb) (A wins both), l = (1-pa)pb (A loses both).
    # (The "split" case returns to an identical state, so it cancels out.)
    if a == b and a >= target - 1:
        w = pa * (1.0 - pb)
        l = (1.0 - pa) * pb
        return w / (w + l) if (w + l) > 0 else 0.5

    # A's probability of winning the next point depends on who is serving it.
    p_point = pa if _tb_server_is_a(a + b) else (1.0 - pb)
    return (
        p_point * _tb_rec(a + 1, b, pa, pb, target)
        + (1.0 - p_point) * _tb_rec(a, b + 1, pa, pb, target)
    )


def tiebreak_prob(pa: float, pb: float, target: int = 7) -> float:
    """P(A wins a tiebreak), A serving the first point."""
    return _tb_rec(0, 0, _q(pa), _q(pb), int(target))


# ──────────────────────────────────────────────────────────────────────────────
# Set
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=100_000)
def set_distribution(
    ha: float,
    hb: float,
    tb_a: float,
    a_serves_first: bool,
    final_set_tb: bool = True,
) -> tuple[tuple[tuple[int, int], float], ...]:
    """
    Full distribution over set scores, as a tuple of ((games_a, games_b), prob).

    Args:
        ha: P(A holds serve)          hb: P(B holds serve)
        tb_a: P(A wins a tiebreak)
        a_serves_first: who serves game 1 of the set
        final_set_tb: if False, 6-6 goes to an advantage set (modelled as a
            long-run hold battle rather than a tiebreak — see below)

    Returned scores are the real ones: (6,0)…(6,4), (7,5), (7,6) and mirrors.
    A tiebreak set is recorded as 13 games total, which is also what the
    serve-rotation bookkeeping in `match_distribution` needs.
    """
    ha, hb, tb_a = _q(ha), _q(hb), _q(tb_a)
    out: dict[tuple[int, int], float] = {}

    def walk(ga: int, gb: int, prob: float) -> None:
        if prob < 1e-12:
            return

        # Straight wins at 6-x (x<=4) and 7-5.
        if ga == 6 and gb <= 4:
            out[(6, gb)] = out.get((6, gb), 0.0) + prob
            return
        if gb == 6 and ga <= 4:
            out[(ga, 6)] = out.get((ga, 6), 0.0) + prob
            return
        if ga == 7:
            out[(7, gb)] = out.get((7, gb), 0.0) + prob
            return
        if gb == 7:
            out[(ga, 7)] = out.get((ga, 7), 0.0) + prob
            return

        if ga == 6 and gb == 6:
            if final_set_tb:
                out[(7, 6)] = out.get((7, 6), 0.0) + prob * tb_a
                out[(6, 7)] = out.get((6, 7), 0.0) + prob * (1.0 - tb_a)
            else:
                # Advantage set: from 6-6 both players keep serving until someone
                # breaks and consolidates. Over a two-game cycle A wins the set
                # with w = ha(1-hb) and loses with l = (1-ha)hb; anything else
                # returns to level, so P(A) = w/(w+l). Score is unbounded, so we
                # record a nominal 8-6 / 6-8 — the games line for a no-tiebreak
                # final set is not meaningful anyway.
                w = ha * (1.0 - hb)
                l = (1.0 - ha) * hb
                pa_set = w / (w + l) if (w + l) > 0 else 0.5
                out[(8, 6)] = out.get((8, 6), 0.0) + prob * pa_set
                out[(6, 8)] = out.get((6, 8), 0.0) + prob * (1.0 - pa_set)
            return

        # Serve alternates every game; A served game 0 iff a_serves_first.
        a_serving = (((ga + gb) % 2) == 0) == a_serves_first
        p_a_wins_game = ha if a_serving else (1.0 - hb)
        walk(ga + 1, gb, prob * p_a_wins_game)
        walk(ga, gb + 1, prob * (1.0 - p_a_wins_game))

    walk(0, 0, 1.0)
    return tuple(sorted(out.items()))


def set_prob(ha: float, hb: float, tb_a: float, a_serves_first: bool = True) -> float:
    """P(A wins the set)."""
    return sum(p for (ga, gb), p in set_distribution(ha, hb, tb_a, a_serves_first) if ga > gb)


# ──────────────────────────────────────────────────────────────────────────────
# Match
# ──────────────────────────────────────────────────────────────────────────────
def _next_set_first_server(a_served_first: bool, total_games: int) -> bool:
    """
    Serve rotation carries across sets: the player who would have served the next
    game serves first in the next set. After `total_games` games the server flips
    iff total_games is odd. (A tiebreak counts as one game, which is exactly the
    real rule — the tiebreak's first receiver serves first in the next set.)
    """
    return a_served_first if total_games % 2 == 0 else (not a_served_first)


MAX_SET_GAMES = 15  # 7-6 is 13; advantage sets are recorded as 8-6 = 14


@lru_cache(maxsize=100_000)
def _set_outcome_split(
    ha: float, hb: float, tb_a: float, a_first: bool, final_set_tb: bool
) -> tuple[tuple[bool, bool, float], ...]:
    """
    Collapse a set distribution onto what the SET-level DP actually needs:
    (A won the set, next set's first server is A, probability).

    Dropping the game counts here is what makes `match_win_prob` cheap — it turns
    a distribution over ~14 scorelines into one over 4 classes.
    """
    agg: dict[tuple[bool, bool], float] = {}
    for (sga, sgb), p in set_distribution(ha, hb, tb_a, a_first, final_set_tb):
        key = (sga > sgb, _next_set_first_server(a_first, sga + sgb))
        agg[key] = agg.get(key, 0.0) + p
    return tuple((k[0], k[1], v) for k, v in agg.items())


@lru_cache(maxsize=400_000)
def match_win_prob(
    pa: float, pb: float, best_of: int = 3, final_set_tb: bool = True
) -> float:
    """
    P(A wins the match) — a pure set-level DP, with no game bookkeeping.

    This is the hot path: `invert_to_target` calls it ~40 times per bisection, and
    a backtest inverts once per match. Tracking the joint games distribution here
    (as `match_distribution` must) made best-of-5 cost 300ms a call, which is what
    turned a 20k-match backtest into an overnight job. Sets alone cost microseconds.
    """
    pa, pb = _q(pa), _q(pb)
    sets_to_win = 3 if best_of == 5 else 2
    ha, hb, tb_a = game_prob(pa), game_prob(pb), tiebreak_prob(pa, pb)

    # state (sets_a, sets_b, a_serves_first) -> probability
    states = {(0, 0, True): 0.5, (0, 0, False): 0.5}
    win = 0.0

    while states:
        nxt: dict[tuple[int, int, bool], float] = {}
        for (sa, sb, a_first), prob in states.items():
            is_decider = (sa == sets_to_win - 1) and (sb == sets_to_win - 1)
            for a_won, next_first, sp in _set_outcome_split(
                ha, hb, tb_a, a_first, True if not is_decider else final_set_tb
            ):
                nsa = sa + (1 if a_won else 0)
                nsb = sb + (0 if a_won else 1)
                p = prob * sp
                if nsa == sets_to_win:
                    win += p
                elif nsb == sets_to_win:
                    pass  # B has won; nothing to accumulate
                else:
                    key = (nsa, nsb, next_first)
                    nxt[key] = nxt.get(key, 0.0) + p
        states = nxt
    return win


def match_distribution(
    pa: float,
    pb: float,
    best_of: int = 3,
    final_set_tb: bool = True,
    average_first_server: bool = True,
    track_scorelines: bool = False,
) -> dict:
    """
    Exact match-level distribution from the two point-on-serve probabilities.

    Args:
        pa: P(A wins a point on A's serve)
        pb: P(B wins a point on B's serve)
        best_of: 3 or 5
        final_set_tb: whether the deciding set has a tiebreak at 6-6
        average_first_server: average over the coin toss (True) or assume A serves
            first (False). Averaging is right for a pre-match prediction.
        track_scorelines: also accumulate the full set-by-set scoreline
            distribution. Off by default — it multiplies the state space by ~14
            per set, which is fine for a single prediction and wasteful across a
            20k-match backtest that only needs the marginals.

    Returns dict with:
        win_prob        P(A wins the match)
        hold_a, hold_b  service hold probabilities
        tb_a            P(A wins a tiebreak)
        set_scores      {(sets_a, sets_b): prob}   e.g. (2,0), (2,1), (0,2)…
        games           {(games_a, games_b): prob} full joint over total games
        exp_games_a/b   expected games won by each
        exp_total_games expected total games in the match
        game_margin     expected games_a - games_b  (the handicap line)
        p_straight      P(A wins in straight sets)
        scorelines      {((6,4),(7,6)): prob}  — only when track_scorelines
    """
    pa, pb = _q(pa), _q(pb)
    sets_to_win = 3 if best_of == 5 else 2
    max_sets = 2 * sets_to_win - 1

    ha = game_prob(pa)
    hb = game_prob(pb)
    tb_a = tiebreak_prob(pa, pb)

    set_scores: dict[tuple[int, int], float] = {}
    scorelines: dict[tuple, float] = {}

    # ── Games distribution by array DP ────────────────────────────────────────
    # State is (sets_a, sets_b, a_serves_first); the value is a 2-D array over
    # (games_a, games_b). Adding a set is a shifted add of that array — 14 of them
    # per state per level. The old formulation enumerated every root-to-leaf path,
    # which is 14^5 ≈ 540k leaves for a five-setter; this is a few hundred array ops.
    size = max_sets * MAX_SET_GAMES + 1
    games_arr = np.zeros((size, size))

    start = np.zeros((size, size))
    if average_first_server:
        start[0, 0] = 0.5
        states = {(0, 0, True): start.copy(), (0, 0, False): start.copy()}
    else:
        start[0, 0] = 1.0
        states = {(0, 0, True): start}

    while states:
        nxt: dict[tuple[int, int, bool], np.ndarray] = {}
        for (sa, sb, a_first), arr in states.items():
            is_decider = (sa == sets_to_win - 1) and (sb == sets_to_win - 1)
            dist = set_distribution(
                ha, hb, tb_a, a_first,
                final_set_tb=True if not is_decider else final_set_tb,
            )
            for (sga, sgb), sp in dist:
                a_won = sga > sgb
                nsa = sa + (1 if a_won else 0)
                nsb = sb + (0 if a_won else 1)
                shifted = np.zeros((size, size))
                shifted[sga:, sgb:] = arr[: size - sga, : size - sgb] * sp

                if nsa == sets_to_win or nsb == sets_to_win:
                    set_scores[(nsa, nsb)] = set_scores.get((nsa, nsb), 0.0) + shifted.sum()
                    games_arr += shifted
                else:
                    key = (nsa, nsb, _next_set_first_server(a_first, sga + sgb))
                    if key in nxt:
                        nxt[key] += shifted
                    else:
                        nxt[key] = shifted
        states = nxt

    nz = np.nonzero(games_arr)
    games = {
        (int(i), int(j)): float(games_arr[i, j])
        for i, j in zip(*nz) if games_arr[i, j] > 1e-12
    }

    idx = np.arange(size)
    exp_ga = float((games_arr.sum(axis=1) * idx).sum())
    exp_gb = float((games_arr.sum(axis=0) * idx).sum())

    win_prob = sum(p for (sa, sb), p in set_scores.items() if sa > sb)
    p_straight = sum(p for (sa, sb), p in set_scores.items() if sa == sets_to_win and sb == 0)

    # ── Optional set-by-set scoreline enumeration ─────────────────────────────
    # Only for single predictions that want "most likely scorelines" — it is a
    # genuine path enumeration and cannot be collapsed. The prune threshold is
    # deliberately loose: paths under 1e-7 can never reach a top-6 display, and
    # dropping them cuts a best-of-5 enumeration by well over an order of magnitude.
    if track_scorelines:
        def walk_paths(sa: int, sb: int, a_first: bool, prob: float, path: tuple) -> None:
            if prob < 1e-7:
                return
            if sa == sets_to_win or sb == sets_to_win:
                scorelines[path] = scorelines.get(path, 0.0) + prob
                return
            is_dec = (sa == sets_to_win - 1) and (sb == sets_to_win - 1)
            for (sga, sgb), sp in set_distribution(
                ha, hb, tb_a, a_first, True if not is_dec else final_set_tb
            ):
                won = sga > sgb
                walk_paths(
                    sa + (1 if won else 0), sb + (0 if won else 1),
                    _next_set_first_server(a_first, sga + sgb),
                    prob * sp, path + ((sga, sgb),),
                )

        if average_first_server:
            walk_paths(0, 0, True, 0.5, ())
            walk_paths(0, 0, False, 0.5, ())
        else:
            walk_paths(0, 0, True, 1.0, ())

    out = {
        "win_prob": win_prob,
        "hold_a": ha,
        "hold_b": hb,
        "tb_a": tb_a,
        "set_scores": set_scores,
        "games": games,
        "exp_games_a": exp_ga,
        "exp_games_b": exp_gb,
        "exp_total_games": exp_ga + exp_gb,
        "game_margin": exp_ga - exp_gb,
        "p_straight": p_straight,
    }
    if track_scorelines:
        out["scorelines"] = scorelines
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Derived market views
# ──────────────────────────────────────────────────────────────────────────────
def total_games_distribution(games: dict) -> dict[int, float]:
    """Collapse the joint games distribution onto total games played."""
    out: dict[int, float] = {}
    for (ga, gb), p in games.items():
        out[ga + gb] = out.get(ga + gb, 0.0) + p
    return dict(sorted(out.items()))


def prob_over_games(games: dict, line: float) -> float:
    """P(total games > line). Half-point lines have no push; integer lines do."""
    return sum(p for (ga, gb), p in games.items() if (ga + gb) > line)


def prob_cover_handicap(games: dict, line: float) -> float:
    """
    P(A covers a game handicap `line`), stated as games added to A.
    e.g. line = -3.5 means A must win by 4+ games.
    """
    return sum(p for (ga, gb), p in games.items() if (ga - gb + line) > 0)


@lru_cache(maxsize=200_000)
def match_summary(
    pa: float, pb: float, best_of: int = 3, final_set_tb: bool = True
) -> tuple[float, float, float]:
    """
    Cached (win_prob, expected_total_games, expected_game_margin).

    The three scalars a backtest or picks run actually consumes, without paying to
    materialise and then discard the full joint distribution on every row.
    """
    d = match_distribution(_q(pa), _q(pb), best_of, final_set_tb)
    return d["win_prob"], d["exp_total_games"], d["game_margin"]


def invert_to_target(
    pa: float,
    pb: float,
    target_win_prob: float,
    best_of: int = 3,
    final_set_tb: bool = True,
) -> tuple[float, float]:
    """
    Nudge (pa, pb) symmetrically so the Markov match probability equals `target`.

    This is the keystone that keeps the engine self-consistent. `predict.py`
    forms its match win probability from a BLEND of surface Elo, the serve/return
    model and (when available) the market — a number the raw point rates do not
    reproduce on their own. Feeding those raw rates to the Markov chain would emit
    a set-score distribution and games line that disagree with the headline
    probability.

    So instead we solve for the scalar δ in
        pa' = pa + δ/2,   pb' = pb - δ/2
    such that match_distribution(pa', pb') gives exactly the target. The serve
    *level* (pa + pb, which drives total games) is preserved; only the serve *gap*
    (which drives who wins) moves. Bisection on δ — the match probability is
    monotone in δ, so it always converges.

    Inputs are quantised before hitting the cached solver, so a backtest sweep
    reuses work across the many matches that land on the same rounded triple.
    """
    return _invert_cached(_q(pa), _q(pb), _q(target_win_prob), int(best_of), bool(final_set_tb))


@lru_cache(maxsize=200_000)
def _invert_cached(
    pa: float, pb: float, target_win_prob: float, best_of: int, final_set_tb: bool,
    tol: float = 1e-4, max_iter: int = 60,
) -> tuple[float, float]:
    lo, hi = -0.45, 0.45

    # Uses the cached scalar `match_win_prob`, never `match_distribution` — the
    # bisection needs ~40 evaluations per call, and building the full joint
    # distribution each time made a 20k-match backtest take hours.
    def f(d: float) -> float:
        a = min(max(pa + d / 2.0, 0.01), 0.99)
        b = min(max(pb - d / 2.0, 0.01), 0.99)
        return match_win_prob(a, b, best_of, final_set_tb)

    if f(lo) > target_win_prob:
        return _q(pa + lo / 2.0), _q(pb - lo / 2.0)
    if f(hi) < target_win_prob:
        return _q(pa + hi / 2.0), _q(pb - hi / 2.0)

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if f(mid) < target_win_prob:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    d = (lo + hi) / 2.0
    return (
        _q(min(max(pa + d / 2.0, 0.01), 0.99)),
        _q(min(max(pb - d / 2.0, 0.01), 0.99)),
    )
