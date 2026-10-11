"""Per-filter predictions for the whole IMPROVE held-out split (Ann, 9 Oct 2026).

The meeting tab's three crossplots draw only the ~100 IMPROVE analogs of each
season that fall in the held-out split. Ann asked to flip to every held-out
IMPROVE filter (~2,000) as well, to see whether the analogs behave like the
rest of the network. The rank-70 meeting run saved summary statistics for that
population (panel_stats.csv, population ``pool_test``) but not the points.

This rebuilds them from the same models, without refitting anything new:
- "all-IMPROVE lot 251": PLS on the lot-251 train split at the saved k (no CV;
  PLS is deterministic, so this is the saved model)
- "all-IMPROVE both lots (frozen Colab)": the completed rank-70 coefficients
Each model's TOR R² on the held-out split must reproduce fits.csv, or nothing is
written.

Run from the repo root:
    uv run python research/ftir_hips_chem/workflows/run_pool_test_points_20261009.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import analog_bias as ab  # noqa: E402
from vibes_current import CURRENT, MIGRATION, MODEL_NAMES, load_context  # noqa: E402

LOT = 251
SRC = MIGRATION / "meeting"
OUT = SRC / "pool_test_predictions.csv"
REF_COLS = ["filter_id", "Site", "lot", "split", "date", "y", "tor_volume_m3", "tor_ec_ugm3",
            "fabs_Mm1", "tor_oc_ec", "op_tor_frac"]


def current_model(method):
    z = np.load(CURRENT / f"benchmark/pls_full_pool_{MODEL_NAMES[method]}.npz")
    coef, xm, b = z["coefficient"].ravel(), z["x_mean"], float(z["intercept"][0])
    return lambda X: (np.asarray(X, float) - xm) @ coef + b


def main():
    ctx, _ = load_context()
    fits = pd.read_csv(SRC / "fits.csv").set_index("fit_id")
    lot = ctx.lib.lot.eq(LOT).to_numpy()
    test = ctx.lib.split.eq("test").to_numpy()
    out = []
    for method in ctx.methods:
        idx = np.flatnonzero(lot)
        sub = ctx.lib.iloc[idx]
        fid = f"allimprove|{method}|all-IMPROVE lot {LOT}"
        f = ab.fit_cohort(ctx.X[method][idx], sub.y, sub.Site,
                          train=sub.split.eq("train").to_numpy(), k=int(fits.loc[fid, "k"]))
        models = {fid: (f.predict, test & lot),
                  f"allimprove|{method}|all-IMPROVE both lots (frozen Colab)": (current_model(method), test)}
        for fit_id, (predict, rows) in models.items():
            pos = np.flatnonzero(rows)
            loading = predict(ctx.X[method][pos])
            r2 = ab.xy_stats(ctx.lib.y.to_numpy()[pos], loading).get("R2")
            saved = float(fits.loc[fit_id, "TOR_test_R2"])
            if len(pos) != int(fits.loc[fit_id, "test_n"]) or not np.isclose(r2, saved, atol=1e-6):
                raise ValueError(f"{fit_id}: n {len(pos)} R² {r2:.6f} does not reproduce the saved {saved:.6f}")
            a = ctx.lib.iloc[pos][REF_COLS].copy()
            a["ftir_ec_ugm3"] = loading / a.tor_volume_m3.to_numpy(float)
            a["fit_id"] = fit_id
            out.append(a)
            print(f"{fit_id}: {len(pos)} held-out filters, TOR R² {r2:.4f} (saved {saved:.4f})", flush=True)
    pd.concat(out).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(AREA.parents[1])}")


if __name__ == "__main__":
    main()
