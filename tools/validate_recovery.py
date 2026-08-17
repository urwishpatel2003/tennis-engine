"""
Does the engine actually recover the truth?

    python tools/validate_recovery.py --tour atp

Only meaningful on synthetic data (tools/make_synthetic_data.py), because only
there is the answer known. The generator draws each player a TRUE latent serve
skill, return skill and surface affinity, then simulates matches point by point.
This script asks whether the engine, seeing only the match results, recovers them.

Passing this does not prove the engine is right about real tennis. It proves the
estimator is unbiased and the plumbing is sound — that Elo tracks true strength,
that the serve/return books recover the latent serve and return talent separately
rather than smearing them together, and that the surface ratings pick up genuine
surface affinity. Those are exactly the failures that are invisible in a log-loss
number but fatal in a player profile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.schema import PROCESSED, RAW, SURFACES  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default="atp")
    ap.add_argument("--min-matches", type=int, default=40)
    args = ap.parse_args()

    truth_path = RAW / f"_truth_{args.tour}.parquet"
    if not truth_path.exists():
        print("No ground truth available — this check only works on synthetic data.\n"
              "Run: python tools/make_synthetic_data.py")
        sys.exit(2)

    truth = pd.read_parquet(truth_path)
    ratings = pd.read_parquet(PROCESSED / "ratings_current.parquet")
    sr = pd.read_parquet(PROCESSED / "serve_return_current.parquet")
    ratings = ratings[ratings["tour"] == args.tour]
    sr = sr[sr["tour"] == args.tour]

    ov = ratings[ratings["surface"] == "overall"][
        ["player_id", "elo", "matches", "last_played"]
    ]
    srov = sr[sr["surface"] == "overall"][["player_id", "serve_excess", "return_excess"]]

    df = truth.merge(ov, on="player_id").merge(srov, on="player_id", how="left")
    df = df[df["matches"] >= args.min_matches]

    # Two different notions of "true strength", and the distinction matters:
    #
    #   static  = the player's career-peak talent (true_serve + true_return)
    #   current = that talent AFTER the generator's age-arc decline, evaluated at
    #             the player's most recent match
    #
    # Elo estimates CURRENT strength — a 34-year-old past their peak is correctly
    # rated below their prime self. Scoring Elo against static talent therefore
    # understates it badly (~0.43 vs ~0.91 on the reference dataset) and says
    # nothing about whether the estimator works. `current` is the honest target;
    # `static` is reported alongside only to show the size of the age effect.
    from tools.make_synthetic_data import _skill_at

    df["true_static"] = df["true_serve"] + df["true_return"]
    current = []
    for _, row in df.iterrows():
        ref = pd.Timestamp(row["last_played"]) if pd.notna(row["last_played"]) else None
        age = ((ref - row["dob"]).days / 365.25) if ref is not None else row["peak_age"]
        s, r = _skill_at(row, "Hard", age)
        current.append(s + r)
    df["true_current"] = current

    print(f"\n{args.tour.upper()} — {len(df)} players with ≥{args.min_matches} matches")
    print("=" * 66)

    rows = [
        ("Elo  vs CURRENT strength",      df["true_current"], df["elo"]),
        ("Elo  vs static career talent",  df["true_static"],  df["elo"]),
        ("serve_excess  vs true serve",   df["true_serve"],   df["serve_excess"]),
        ("return_excess vs true return",  df["true_return"],  df["return_excess"]),
    ]
    results = {}
    for label, t, e in rows:
        m = np.isfinite(t) & np.isfinite(e)
        rho = spearman(t[m].to_numpy(), e[m].to_numpy())
        r = float(np.corrcoef(t[m], e[m])[0, 1])
        results[label] = rho
        print(f"  {label:<34} spearman {rho:+.3f}   pearson {r:+.3f}")

    # ── Discriminant validity ─────────────────────────────────────────────────
    # The serve book must track SERVE talent more closely than RETURN talent, and
    # vice versa. If both correlate equally with both, the model has collapsed two
    # distinct skills into one strength number and every player profile is a lie.
    print("\n  discriminant check (the estimate should track its OWN skill best)")
    m = df["serve_excess"].notna()
    ss = spearman(df.loc[m, "true_serve"].to_numpy(), df.loc[m, "serve_excess"].to_numpy())
    sr_ = spearman(df.loc[m, "true_return"].to_numpy(), df.loc[m, "serve_excess"].to_numpy())
    rr = spearman(df.loc[m, "true_return"].to_numpy(), df.loc[m, "return_excess"].to_numpy())
    rs = spearman(df.loc[m, "true_serve"].to_numpy(), df.loc[m, "return_excess"].to_numpy())
    print(f"    serve_excess :  own {ss:+.3f}   cross {sr_:+.3f}   "
          f"{'OK' if ss > sr_ else 'FAILED — skills are entangled'}")
    print(f"    return_excess:  own {rr:+.3f}   cross {rs:+.3f}   "
          f"{'OK' if rr > rs else 'FAILED — skills are entangled'}")

    # ── Surface affinity ──────────────────────────────────────────────────────
    print("\n  surface affinity recovery (surface Elo minus overall Elo)")
    for surf in SURFACES:
        s = ratings[ratings["surface"] == surf][["player_id", "elo", "matches"]]
        s = s.rename(columns={"elo": "elo_surf", "matches": "m_surf"})
        d = df.merge(s, on="player_id")
        d = d[d["m_surf"] >= 15]
        if len(d) < 25:
            print(f"    {surf:<7} — too few players with surface history")
            continue
        rho = spearman(d[f"aff_{surf}"].to_numpy(),
                       (d["elo_surf"] - d["elo"]).to_numpy())
        print(f"    {surf:<7} spearman {rho:+.3f}   (n={len(d)})")

    print("\n" + "=" * 66)
    # Surface affinity is deliberately NOT a gate. The generator draws affinities
    # an order of magnitude smaller than core talent, and grass supplies ~18
    # matches per player across the whole archive — the signal is real but too
    # thin to hold to a threshold without making this check flaky.
    ok = (results["Elo  vs CURRENT strength"] > 0.80
          and results["serve_excess  vs true serve"] > 0.55
          and results["return_excess vs true return"] > 0.55
          and ss > sr_ and rr > rs)
    print("  VERDICT:", "engine recovers the latent truth" if ok
          else "recovery is WEAKER than expected — investigate")
    print("=" * 66 + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
