# Tennis Engine

Match prediction for the ATP and WTA tours: win probability, set scores, game
handicap and total games, from surface-aware ratings plus a point-level serve/return
model.

**Live:** [tennis-engine-production.up.railway.app](https://tennis-engine-production.up.railway.app)

```
  Carlos Alcaraz  vs  Jannik Sinner
  Roland Garros · Clay · best of 5 · ATP
==========================================================================
  WIN PROBABILITY   Carlos Alcaraz:  37.4%   (2.67)
                    Jannik Sinner:  62.6%   (1.60)

  components        elo 0.416 | serve/return 0.310 | market   —
  elo blend          1994.9 vs  2076.9   (gap -82.0 → -58.7 adjusted)
  adjustments       conditions -2.4  head_to_head +25.7
  head-to-head      10-7 in 17 meetings

  SERVE             hold 73.4% vs 77.4%   (pts on serve 59.9% / 61.9%)
  SET SCORES        0-3 18%  1-3 24%  2-3 21%  3-0 8%  3-1 14%  3-2 16%
  GAMES             Carlos Alcaraz 19.7 - 21.4 Jannik Sinner   (total 41.0)
  FAIR LINES        handicap Carlos Alcaraz +2.5 games   ·   total 41.5 games
  DATA QUALITY      high
```

## Quick start

```bash
pip install -r requirements.txt
pip install -r requirements-fetch.txt      # only needed for the data pull

python fetch_data.py --seasons 2010-2026   # needs internet
python run_engine.py --build
python engine/predict.py --a "Alcaraz" --b "Sinner" --surface Clay --best-of 5
python dashboard/server.py                 # http://localhost:5000
```

If every source fails, check whether the upstream still exists before assuming a
network problem — a deleted repo 404s exactly like a blocked host:

```bash
gh api repos/OWNER/REPO --jq '.full_name, .default_branch'
```

Any mirror can be cloned and side-loaded with
`python fetch_data.py --from-clone /path/to/parent_of_atp_and_wta`.

To try the engine with no data at all, generate a schema-compatible synthetic
archive (clearly marked as synthetic everywhere it surfaces):

```bash
python tools/make_synthetic_data.py --seasons 2018-2026
python run_engine.py --build
```

## The Tournaments view

Pick a tour, a season and an event, and the dashboard replays the whole draw:
every match round by round with the model's prediction beside the actual result
and a hit/miss marker, an accuracy summary for the event, the biggest upsets it
missed, and — on demand — pre-tournament title odds simulated forward through the
reconstructed bracket.

Two things make this honest rather than decorative:

- **Predictions use pre-match state.** Each match is priced from the ratings
  frozen before it was played, so the hit/miss column is a genuine out-of-sample
  record. Replaying a 2024 draw with today's ratings would be hindsight.
- **The bracket is reconstructed, not assumed.** A player in a round-k match must
  have won a round-(k-1) match, which identifies exactly which two earlier matches
  feed each later one. Title odds then convolve over *every* pairing that could
  have occurred, using each player's rating as of the start of the event.

Note the archive is **results-only** — tennis draws are made a day or two ahead
and are not published in it, so there are no forward fixtures. The Tournaments
view covers completed and in-progress events.

```bash
python engine/tournament.py --tour atp --season 2025 --list
python engine/tournament.py --tour atp --tourney-id 2025-540
```

## How it works

Two independent views of a match are formed and then reconciled.

**Rating view.** Elo, run separately for each surface and for the tour overall, with
a K-factor that decays with career matches (a 20-match teenager moves far more per
result than a 900-match veteran) and regression toward the mean across long injury
layoffs. Surface and overall are blended, because a pure grass Elo built from 12
career grass matches overfits. The resulting gap is adjusted in Elo points for
fatigue, rest, home crowd, handedness, and head-to-head *surprise* — how a pair's
record compares to what the ratings alone predicted, heavily shrunk.

**Point view.** Every player carries a serve and a return rating expressed as an
excess over the (tour, surface) baseline — a raw 64% of service points won means
something completely different on the ATP grass court and the WTA clay court. Those
excesses are opponent-adjusted using the opponent's rating *as it stood before the
match*, then combined additively into two point-on-serve probabilities and run
through an exact Barnett–Clarke Markov chain: point → game → tiebreak → set → match.

**Reconciliation.** The two views are blended in log-odds space. The blended number
is not what the raw point rates produce on their own, so the engine solves for the
symmetric shift in the two serve probabilities that makes the Markov chain reproduce
it exactly — preserving the serve *level* (which drives total games) while moving
the serve *gap* (which drives who wins). Only then are the set-score distribution,
game handicap and totals read off. That is why the headline probability and the
games line can never contradict each other.

## Validation

`backtest.py` is walk-forward and out-of-sample by construction: it reads the
pre-match rating columns that the build steps froze before each match was played,
rather than recomputing anything. Player A is fixed as the lower player id so row
order carries no information about the result.

**Real data, 46,482 matches (2015–2026, both players with ≥20 prior matches):**

| model | log loss | Brier | accuracy |
|---|---|---|---|
| coin flip | 0.6931 | 0.2500 | 50.0% |
| Elo, overall only | 0.6190 | 0.2155 | 65.1% |
| Elo, surface blend | 0.6180 | 0.2151 | 65.3% |
| serve/return simulation | 0.6244 | 0.2172 | 64.8% |
| **blended model** | **0.6120** | **0.2126** | **65.6%** |

The blend beats every component it is built from. The **optimal Elo spread
multiplier is 0.96 — "well scaled"** — so `ELO_SPREAD_MULT` stays at 1.0. (The
synthetic archive suggested 0.86; that really was rating warm-up, and reading the
pooled number would have damped good ratings to fix a transient that does not
exist in the real data.)

Calibration is good but consistently a shade over-confident — every decile lands
1–4 points below its predicted rate, largest gap 3.6pp around the coin-flip band.

Per tournament, the same engine replayed over completed draws:

| event | accuracy | log loss |
|---|---|---|
| Roland Garros 2025 | 87/119 = 73.1% | 0.5445 |
| US Open 2025 | 85/120 = 70.8% | 0.5528 |
| Wimbledon 2025 | 80/122 = 65.6% | 0.5906 |

Grass being the hardest to predict is a real, well-known property of the surface,
not an artefact — short points and big serving compress the skill gap.

### Score markets

Everything above scores the win probability. The **fair total** and **fair
handicap** are separate outputs, they appear on every matchup page, and for a long
time nothing graded them — `backtest.py` does not look at scorelines at all.

`tools/validate_score_markets.py` replays 10,000 completed matches and asks the one
question a "fair" line actually promises: does each side win half the time?

| | before | after |
|---|---|---|
| fair total goes over | 42.0% | **50.4%** |
| fair handicap covers | 48.1% | **49.7%** |
| worst gap in stated P(over) | −8.3pp | **−1.2pp** |
| pushes (line landed on a whole number) | 403 | **0** |

The cause was a category error, not a bad constant: `calibrate_total_games` is
fitted on the *expectation* of total games and was being applied to the *median*
that the line is derived from. `engine/score_calib.py` now calibrates centre and
width as two separate parameters — the single-slope map had to use one number for
both, and the 0.73 slope needed to centre the line was also shrinking the
distribution's width by 0.73, cutting the right tail short. Fitted freely, the
width comes out at 0.98: the chain's dispersion was right all along.

One thing this deliberately does not fix. Real total-games is more right-skewed
than the model's — many straight-sets matches, then a long thin tail — and a
location-scale correction has no skew parameter. So the **fair line sits below the
expected total**, and totals carry two calibrations on purpose. The game margin is
near-symmetric and needs only one.

Constants are fitted by `tools/fit_score_calibration.py` on an odd/even split and
scored on the held-out half. A per-quantile fit was tried and rejected: it did not
improve the 50/50 rate and made the distribution's shape worse.

### Recovering known truth

`tools/validate_recovery.py` runs against the synthetic generator, where the
answer is known, and checks the estimator itself rather than its accuracy:

| quantity | Spearman |
|---|---|
| Elo vs current strength | +0.91 |
| serve rating vs true serve talent | +0.80 |
| return rating vs true return talent | +0.81 |

with clean discriminant separation — the serve book tracks serve talent (+0.80)
and not return talent (−0.38), so the two skills stay distinct rather than
collapsing into one strength number.

### What is NOT validated

The backtest covers the Elo + serve/return blend. The conditions, head-to-head and
height adjustments are applied per-query in `predict.py` and are **not** measured
by it — see "Known gap" in `CLAUDE.md`.

## Tests

```bash
python tests/test_markov.py       # scoring maths against known identities
python tests/test_no_leakage.py   # the temporal invariant
python tests/test_pipeline.py     # end-to-end coherence
```

`test_markov.py` pins the closed forms against hand-checked values and checks
symmetry and monotonicity throughout. `test_no_leakage.py` rebuilds the ratings on a
truncated match log and requires every surviving pre-match value to be bit-identical
— if a future match could influence an earlier row, truncating the future would
change it. `test_pipeline.py` checks that the printed probability and the printed
score distribution actually agree.

## A note on betting

`weekly_picks.py` compares the model against market prices and reports the
disagreement. Disagreement is not edge. Tennis match markets are efficient, and the
honest expectation is a well-calibrated price rather than a beatable one — the same
conclusion already on record for the NFL engine in this workspace. Validate against
results before treating any of it as a signal.

## Staying current

The Sackmann mirror stopped updating in May 2026, so `engine/refresh.py` extends
the ATP archive from [ManTennisData](https://github.com/msolonskyi/ManTennisData)
(an atptour.com scrape), which carries fuller serve statistics and is current to
within a few days. It appends only — the historical base is never restated.

```bash
python -m engine.refresh --status      # per-tour freshness
python -m engine.refresh               # append + rebuild
```

WTA comes from **api.wtatennis.com**, the tour's own public API — unauthenticated,
the same JSON the official site consumes. It carries scores but no serve
statistics, so those matches update Elo while the serve/return model relies on
the historical stat lines.

The dashboard reports freshness per tour, since the two sources can drift apart.

Why it matters, concretely: on stale ratings the model priced Tabilo v Jodar at
48.6% against a 31.8% market — a 17-point "edge". After refreshing, it prices the
same match at 35.8% against 32.9%. The edge was the staleness.

On Railway a refresh runs on boot and daily (`REFRESH_ON_BOOT`, `REFRESH_DAILY`,
`REFRESH_HOUR`), or on demand via `POST /api/refresh` with `REFRESH_TOKEN`.

## Deployment

Hosted on Railway, deployed from the CLI:

```bash
railway up
```

The data is **fetched and built during the Railway build**, not committed — the
machine this was developed on sits behind a proxy that 404s
`raw.githubusercontent.com`, while the build container has open internet. So
`nixpacks.toml` runs `fetch_data.py` → `run_engine.py --build` →
`tools/build_report.py`, and the image ships with real ATP/WTA data.

There is deliberately **no Railway Volume**: nothing on the server mutates data at
runtime, so there is no state worth persisting, and every redeploy simply
refreshes the archive. `data/` is in both `.gitignore` and `.railwayignore` so a
local (possibly synthetic) dataset can never reach the public URL.

| Variable | Default | Purpose |
|---|---|---|
| `FETCH_SEASONS` | `2005-2026` | Season range built into the image. Wider = better-converged Elo, longer build. |
| `PORT` | set by Railway | gunicorn bind port |

`tools/build_report.py` **fails the build** if no usable data was produced —
shipping an empty dashboard silently is worse than a failed deploy. `/api/health`
is the healthcheck and touches no parquet, so a data problem never turns into a
restart loop.

## Data

Jeff Sackmann's ATP and WTA archives, CC BY-NC-SA. Match results and serve/return
statistics; the engine is built from 2010 onward by default.

The original `JeffSackmann/tennis_atp` and `tennis_wta` repositories were taken
down (404 as of August 2026), so `fetch_data.py` reads from
[Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive),
an archival mirror with identical file names and schema. The original URLs are
still attempted first, so the fetcher switches back automatically if they return.

## Layout

```
fetch_data.py            network ingest (the only script that goes online)
run_engine.py            orchestrator + status
engine/
  schema.py              paths, vocabularies, tour baselines, score parsing
  markov.py              exact point → game → tiebreak → set → match chain
  ratings.py             surface-aware Elo, walk-forward
  serve_return.py        opponent-adjusted serve/return books
  conditions.py          fatigue, rest, venue, home crowd
  matchups.py            head-to-head and style interactions
  predict.py             blending and reconciliation
rankings.py              power rankings
weekly_picks.py          model vs market
backtest.py              walk-forward OOS validation
dashboard/               Flask server + single-page UI
tools/                   synthetic data generator, recovery validator
tests/                   three standalone test scripts
```
