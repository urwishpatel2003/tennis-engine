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

On the synthetic reference archive (~19k matches, both players with ≥20 prior matches):

| model | log loss | Brier | accuracy |
|---|---|---|---|
| coin flip | 0.6931 | 0.2500 | 50.0% |
| Elo, overall only | 0.6392 | 0.2238 | 63.6% |
| Elo, surface blend | 0.6408 | 0.2248 | 63.4% |
| serve/return simulation | 0.6415 | 0.2254 | 62.8% |
| **blended model** | **0.6359** | **0.2228** | **64.0%** |

The blend beats every component it is built from, and calibration is tight (largest
decile gap 2.8pp). Total games MAE 5.77, game margin MAE 4.09.

`tools/validate_recovery.py` goes further and asks whether the engine recovers the
*known* latent skills the generator drew:

| quantity | Spearman |
|---|---|
| Elo vs current strength | +0.91 |
| serve rating vs true serve talent | +0.80 |
| return rating vs true return talent | +0.81 |

with clean discriminant separation — the serve book tracks serve talent (+0.80) and
not return talent (−0.38), so the two skills stay distinct rather than collapsing
into one strength number.

**These are synthetic-data numbers.** They demonstrate that the estimator is
unbiased and the plumbing is sound. They say nothing about accuracy on real tennis
— re-run both after pulling the real archive.

Broken down by season, the picture is more interesting than the headline:

| season | log loss | accuracy | Elo spread multiplier |
|---|---|---|---|
| 2021 | 0.6599 | 62.1% | 0.67 |
| 2022 | 0.6477 | 61.9% | 0.80 |
| 2023 | 0.6366 | 64.1% | 0.84 |
| 2024 | 0.6345 | 64.0% | 0.86 |
| 2025 | 0.6221 | 65.7% | 0.98 |
| 2026 | 0.6204 | 65.7% | 0.98 |

The multiplier climbs from 0.67 to 0.98 as the ratings mature. So the
over-confidence in the 0.86 headline is a **warm-up artefact, not a property of
the model** — the archive only starts in 2018, and early seasons are dominated by
players whose Elo has not converged. On mature ratings the spread is essentially
correct, which is why `ELO_SPREAD_MULT` in `engine/predict.py` is left at 1.0.

Two caveats before reading too much into any of it:

- These are synthetic numbers. Re-run the backtest on the real archive; if the
  *late* seasons still show a multiplier well below 1.0 there, that is a genuine
  signal to lower `K0`, and the early seasons should be ignored either way.
- The backtest covers the Elo + serve/return blend only. The conditions,
  head-to-head and height adjustments are applied per-query in `predict.py` and are
  **not** measured by it — see the "Known gap" section in `CLAUDE.md`.

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
