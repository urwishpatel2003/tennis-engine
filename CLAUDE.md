# CLAUDE.md

Operational guide for working in this repo. Read this first.

## What this is

A tennis match-prediction engine covering the ATP and WTA tours. It ingests Jeff
Sackmann's public match archive, builds surface-aware player ratings and
point-level serve/return models, and predicts match win probability, set scores,
game handicaps and total games. A picks layer compares those to market prices, a
backtest layer validates them walk-forward, and a Flask dashboard visualises them.

- Language: Python 3 (pandas / numpy / pyarrow)
- Data format: Parquet (raw + processed), CSV for rankings/backtest outputs
- Not a package — scripts are run directly.

## Pipeline (data flow)

```
fetch_data.py ──▶ data/raw/*.parquet ──▶ engine/ build steps ──▶ data/processed/*.parquet ──▶ predict ──▶ rankings / picks / backtest / dashboard
```

1. **Fetch** — `python fetch_data.py --seasons 2000-2026`
   Run LOCALLY (needs internet). Writes `matches_{tour}.parquet`,
   `players_{tour}.parquet`, `rankings_{tour}.parquet` to `data/raw/`.
   Needs `pip install -r requirements-fetch.txt`.
2. **Build engine** (orchestrated by `run_engine.py --build`; the four steps are
   independent and each reads only the raw match log, so any one can be re-run alone):
   - `engine/ratings.py` → `ratings.parquet` + `ratings_current.parquet`
   - `engine/serve_return.py` → `serve_return.parquet` + `serve_return_current.parquet`
   - `engine/conditions.py` → `conditions.parquet`
   - `engine/matchups.py` → `h2h.parquet`
3. **Predict** — `engine/predict.py` combines everything into a full forecast.
4. **Use** — `rankings.py`, `weekly_picks.py`, `backtest.py`, `dashboard/`.

## Common commands

```bash
# One-time data pull, then build
python fetch_data.py --seasons 2000-2026
python run_engine.py --build

# What is built and how fresh
python run_engine.py --status

# Predict a match
python engine/predict.py --a "Carlos Alcaraz" --b "Jannik Sinner" \
    --tour atp --surface Clay --best-of 5 --tournament "Roland Garros"

# Power rankings
python rankings.py --tour atp --surface Clay --top 30
python rankings.py --tour wta --all-surfaces          # writes a CSV per surface

# Walk-forward out-of-sample backtest (the honest one)
python backtest.py --seasons 2015-2026
python backtest.py --tours atp --surface Grass --no-scores    # much faster

# Model vs market
python weekly_picks.py --slate slate.csv --top 10

# Dashboard
python dashboard/server.py            # http://localhost:5000

# Tests — run all three after touching the engine
python tests/test_markov.py
python tests/test_no_leakage.py
python tests/test_pipeline.py
```

`run_engine.py --build` flags: `--skip-ratings`, `--skip-serve-return`,
`--skip-conditions`, `--skip-matchups`.

## Key directories / files

- `engine/` — the model.
  - `markov.py` — the scoring maths. Exact Barnett-Clarke point→game→tiebreak→set→match.
  - `predict.py` — the heart. Blends the two views and reconciles the score model.
  - `ratings.py`, `serve_return.py`, `conditions.py`, `matchups.py` — build steps.
  - `schema.py` — paths, vocabularies, tour baselines, score-string parsing.
- `data/raw/` — fetched source parquets. `data/processed/` — built model outputs.
- `dashboard/` — Flask `server.py` + hash-routed SPA `dashboard.html`.
- `tools/make_synthetic_data.py` — generates a schema-compatible fake archive.
- `tools/validate_recovery.py` — checks the engine recovers known latent truth.
- `fetch_data.py` — the only thing that touches the network.

## How a prediction is assembled (engine/predict.py)

Two independent views, blended in log-odds space, then reconciled:

- **A. Rating view** — blended surface Elo (`SURFACE_BLEND = 0.55` surface / overall),
  adjusted in Elo points for conditions (fatigue, rest, home crowd), handedness and
  head-to-head surprise. → `p_elo`
- **B. Point view** — serve/return excesses (surface-blended, opponent-adjusted),
  shifted for height/style and venue physics, combined additively into two
  point-on-serve probabilities, run through the exact Markov chain. → `p_sim`
- **Blend** — `W_ELO = 0.62`, log-odds weighted. Market gets `W_MARKET = 0.55` when supplied.
- **C. Reconcile** — `markov.invert_to_target` solves for the symmetric shift in the
  two serve probabilities that makes the Markov chain reproduce the blended
  probability exactly. Only then are set scores, handicap and totals read off.

Most constants are hand-tuned with inline rationale. When changing a weight, keep
the comment explaining the *why* next to the number.

## Invariants (do not break these)

- **No future leakage.** `ratings.py` / `serve_return.py` / `conditions.py` each walk
  the match log once, chronologically, and write the state as it stood BEFORE each
  match. `backtest.py` reads those pre-match columns rather than recomputing, so it
  is honest by construction. `tests/test_no_leakage.py` guards this — run it after
  touching any build step.
- **Backtest row orientation.** Raw tables are winner-first. `backtest.py` fixes
  player A as the LOWER player id, which is independent of the result. Never
  evaluate on winner-first rows; it scores 100% and means nothing.
- **Score/probability coherence.** The headline win probability and the games line
  must come from the same reconciled point probabilities. If you compute the games
  line from raw serve rates while reporting a blended probability, they will
  silently disagree. `tests/test_pipeline.py` section 3 guards this.
- **The server reads local parquet only.** Nothing in `engine/` or `dashboard/`
  may import anything from `requirements-fetch.txt`.

## Known gap: what the backtest does NOT cover

`backtest.py` evaluates the Elo + serve/return blend. It does **not** exercise the
conditions (fatigue/rest/home), head-to-head or height/style adjustments, because
those live in `predict.py` and are applied per-query rather than stored per-match.
So the reported log loss validates the core, not the adjustment layer.

Those adjustments are therefore held to *reasoning*, not measurement, and their
magnitudes are deliberately conservative. If you want to validate them, extend
`score_frame()` to reconstruct them from `conditions.parquet` and `h2h.parquet` —
both are keyed by `match_id` and already leak-free. Until then, treat any change
to their constants as unvalidated.

A concrete instance of why that matters: head-to-head originally used a prior of 6
with the full ~695 Elo-per-probability conversion, which turned a single 1-0
meeting into a +47 Elo swing — larger than most rating gaps. It backtested exactly
the same as the fixed version, because the backtest never sees the term.

## Gotchas

- **Windows console encoding.** `engine/__init__.py` forces UTF-8 on stdout/stderr
  at import. Without it every CLI dies on its first box-drawing character under
  cp1252. Any new entry point must import something from `engine` before printing.
- **Higher hold ⇒ MORE games.** Fast conditions (altitude, indoor, grass) raise both
  players' hold probability, which makes breaks rarer and pushes sets to 6-6. It is
  natural to assume fast conditions shorten matches; in games, they lengthen them.
- **`tourney_date` is tournament-level.** Every match at an event shares one date.
  Sorting by it alone does not order matches within an event — sort by
  `["tourney_date", "tourney_id", "match_num"]`, or leave build outputs in the order
  they were emitted.
- **`best_of` matters a lot for cost.** A best-of-5 `match_distribution` is ~75× a
  best-of-3. Use `markov.match_win_prob` (set-level DP, no game bookkeeping) or
  `markov.match_summary` in any loop; reserve the full distribution for single
  predictions.
- **Retirements vs walkovers.** `parse_score` marks walkovers/defaults as
  `completed=False, retirement=False` and they are skipped entirely by every build
  step. Retirements update ratings at half weight but never feed the serve/return
  books.
- **Surface fallbacks.** A player with no history on a surface inherits their
  overall rating and overall serve/return book, not a 1500 default. Preserve that —
  a top hard-courter is not average the first time they play on clay.
- **Windows + OneDrive.** Repo lives under a OneDrive path with spaces. Quote paths.
  Shell is PowerShell; a Bash tool is also available.

## Hosting & deployment (Railway)

Deployed on Railway via the CLI (`railway up`), same as the NFL engine — the repo
is NOT connected to Railway, so a `git push` alone does not deploy.

**The data is built at BUILD TIME, and there is no Volume.** This is the one
significant difference from the NFL engine, and it is deliberate:

- The dev machine's proxy 404s `raw.githubusercontent.com`, so the real archive
  cannot be fetched locally. Railway's build container has unrestricted internet,
  so `nixpacks.toml` runs `fetch_data.py` → `run_engine.py --build` there.
- Nothing on this server mutates data at runtime (there is no `/api/refresh`
  equivalent), so there is no state worth persisting between deploys. A Volume
  would only add a stale-seed failure mode. **Every redeploy refreshes the data.**
- `data/` is in both `.gitignore` and `.railwayignore`, so the local (possibly
  SYNTHETIC) dataset can never reach the server.

Service variables:

- `FETCH_SEASONS` — season range for the build, default `2005-2026`. Wider means
  better-converged Elo and a longer build.
- `PORT` — supplied by Railway; gunicorn binds it via the Procfile.

The build ends with `tools/build_report.py`, which writes
`data/processed/build_info.json` and **fails the build** if no usable data was
produced. Shipping an empty dashboard silently is worse than a failed deploy.

`/api/health` is the healthcheck and deliberately touches no parquet — it reports
data presence in the body instead. Gating it on data would turn a data problem
into a restart loop that can never resolve itself.

```bash
railway up                 # deploy
railway logs               # build + runtime logs
railway variables          # inspect/set FETCH_SEASONS
```

Do NOT curl the live URL from this laptop to verify — it is unreachable from here
(same as the NFL engine). Verify on localhost, then deploy and read `railway logs`.

## Data source

**The original `JeffSackmann/tennis_atp` and `tennis_wta` repos no longer exist** —
they returned 404 as of 2026-08-17, from three independent networks, and the
account's only remaining public repo is `tennis_MatchChartingProject`. Data now
comes from `Aneeshers/tennis-sackmann-archive`, an archival mirror carrying the
identical file names and the identical 49-column match schema, but nested under
`atp/` and `wta/` directories on branch `main` rather than at the repo root on
`master`. The original URLs are still tried first at no cost (a 404 is not
retried), so if Jeff restores them they are picked up automatically.

Data remains CC BY-NC-SA (Jeff Sackmann). See LICENSE.

**Debugging note worth keeping.** Those 404s were first misdiagnosed here as a
corporate proxy blocking `raw.githubusercontent.com`, which cost hours and sent
the project down a synthetic-data detour. A deleted repo and a blocked host look
identical if you only read status codes. Before believing any network theory:

```bash
gh api repos/OWNER/REPO --jq '.full_name, .default_branch'   # gone, or unreachable?
gh search repos tennis_atp                                    # find a surviving mirror
```

and fetch a control URL on the same host that you know exists.

## Conventions

- Heavy module docstrings describing inputs/outputs, section banners (`# ──── …`),
  inline comments justifying every tuned constant.
- Scripts use `argparse` with `--tour`/`--tours`, `--surface`, `--seasons`.
  Outputs go to `data/processed/`.
- Paths always derive from `Path(__file__).parent` — keep it relocatable.
- Engine modules carry a `sys.path` bootstrap so both `python engine/x.py` and
  `python -m engine.x` work.
