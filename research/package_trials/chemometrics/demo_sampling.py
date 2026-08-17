"""Trial (brief): astartes / kennard-stone sample selection on the 800 cohort.

Question: if a 190-filter calibration subset were drawn by Kennard-Stone (X
only) or SPXY (X and y) instead of at random, how does its coverage of the TOR
EC loading distribution differ?

Run:  python demo_sampling.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trial_common import load_cohort_800

N_SELECT = 190
RANDOM_SEEDS = range(20)


def y_coverage(y_sub, y_all, deciles):
    """Simple y-coverage summary for a subset against the full cohort."""
    return {
        "n": len(y_sub),
        "y_min": float(np.min(y_sub)),
        "y_max": float(np.max(y_sub)),
        "range_covered_%": 100 * (np.max(y_sub) - np.min(y_sub)) / np.ptp(y_all),
        "deciles_hit": int(pd.Series(np.digitize(y_sub, deciles)).nunique()),
        "y_mean": float(np.mean(y_sub)),
        "y_sd": float(np.std(y_sub, ddof=1)),
    }


def main():
    import astartes
    import kennard_stone
    print(f"astartes {astartes.__version__}, kennard-stone {kennard_stone.__version__}")

    X, y, sites, _, _ = load_cohort_800()
    deciles = np.quantile(y, np.linspace(0.1, 0.9, 9))
    rows = {"full cohort": y_coverage(y, y, deciles)}

    # astartes: KS on spectra only, SPXY on spectra + y jointly.
    from astartes import train_test_split as astartes_split
    for sampler in ("kennard_stone", "spxy"):
        *_, idx_train, idx_test = astartes_split(
            X, y, sampler=sampler, train_size=N_SELECT / len(y),
            return_indices=True)
        rows[f"astartes {sampler}"] = y_coverage(y[idx_train], y, deciles)
        rows[f"astartes {sampler}"]["n_sites"] = int(
            pd.Series(sites[idx_train]).nunique())

    # kennard-stone package: same KS ordering, sklearn-signature API.
    from kennard_stone import train_test_split as ks_split
    _, _, y_train, _ = ks_split(X, y, train_size=N_SELECT / len(y))
    rows["kennard_stone pkg (KS)"] = y_coverage(np.asarray(y_train), y, deciles)

    # random baseline: 20 seeds
    rng_rows = []
    for seed in RANDOM_SEEDS:
        idx = np.random.default_rng(seed).choice(len(y), N_SELECT, replace=False)
        rng_rows.append(y_coverage(y[idx], y, deciles))
    rand = pd.DataFrame(rng_rows).mean()
    rand["n"] = N_SELECT
    rows[f"random (mean of {len(rng_rows)} seeds)"] = rand.to_dict()

    table = pd.DataFrame(rows).T
    table["n"] = table["n"].astype(int)
    for col in ("y_min", "y_max", "range_covered_%", "y_mean", "y_sd"):
        table[col] = table[col].astype(float).round(2)
    print(f"\ny-coverage of {N_SELECT}-sample subsets "
          f"(y = TOR EC loading, µg/filter; cohort range "
          f"{y.min():.2f}–{y.max():.2f}):")
    print(table.to_string())


if __name__ == "__main__":
    main()
