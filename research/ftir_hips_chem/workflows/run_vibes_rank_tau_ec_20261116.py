"""How far does fixed-model EC move when the VIBES rank or tau changes? Network scans, solved to convergence.

    uv run python run_vibes_rank_tau_ec_20261116.py --area research/ftir_hips_chem --out OUT

Every scan saved under the 87-blank background in the 19 Oct network bundle is refitted at each rank
(the first components of the rank-70 background) and tau, with the converged solver. EC is then read
with the frozen k 4 IMPROVE model, which was trained on spectra corrected at rank 70 and tau 0.1. So
the table measures what happens when the correction and the calibration no longer match; it does not
rank the settings. Ranking them needs the calibration refitted at each setting.

One file per setting is cached in OUT, so a rerun only recomputes what is missing.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vibes_batch as vb  # noqa: E402

RANKS = (70, 40, 20, 10)
TAUS = (0.1, 0.05, 0.2)
REFERENCE = (70, 0.1)
BANDS = {"featureless 2000-2500": (2000, 2500), "C-H 2800-3000": (2800, 3000), "carbonyl 1650-1800": (1650, 1800)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", required=True, help="research/ftir_hips_chem")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    area, out = Path(a.area), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    src = area / "output/tables/network_baselines_20261019/vibes_bundle"
    v2 = area / "output/tables/network_baselines_v2_20261103"
    spectra = np.asarray(np.load(src / "X_model_grid.npy"), dtype=np.float64)
    cases = pd.read_csv(src / "cases.csv", dtype={"analysis_id": str, "media_id": str})
    model = np.load(src / "background_model.npz")
    wn, mean, components = model["wn"], model["mean"].astype(float), model["components"].astype(float)
    diagnostics = pd.read_csv(v2 / "vibes_fit_diagnostics.csv")
    rows = np.flatnonzero(diagnostics.vibes_model.eq("b87").to_numpy())
    X = spectra[rows]
    pls = np.load(area / "output/tables/vibes_fullrange/interpretation_20261002/benchmark/pls_full_pool_uncapped_full.npz")
    coefficient, x_mean, intercept = pls["coefficient"].ravel(), pls["x_mean"].ravel(), float(pls["intercept"].ravel()[0])
    windows = {name: (wn >= lo) & (wn <= hi) for name, (lo, hi) in BANDS.items()}

    cases.iloc[rows][["case_row", "analysis_id", "filter_id", "filter_key", "code", "filter_type", "qc"]].assign(
        premature_stop=diagnostics.premature_stop.to_numpy()[rows]).to_csv(out / "scans.csv", index=False)

    reference = None
    for rank, tau in [REFERENCE] + [(r, t) for r in RANKS for t in TAUS if (r, t) != REFERENCE]:
        cache = out / f"fit_rank{rank}_tau{tau}.npz"
        if cache.exists() and (rank, tau) != REFERENCE:
            continue
        t0 = time.perf_counter()
        basis = components[:, :rank]
        elbo = vb.fit_elbo_newton(X, mean, basis, tau=tau, chunk=512)
        solved = vb.solve_map_batch(X, mean, basis, sigma=elbo.sigma, tau=tau, chunk=256)
        baseline = solved.baseline
        corrected = X - baseline
        if (rank, tau) == REFERENCE:
            reference = baseline
            saved = np.load(v2 / "vibes_baselines_model_grid.npy")[rows].astype(float)
            print("reference against the 3 Nov converged run: largest baseline difference",
                  float(np.abs(baseline - saved).max()), flush=True)
        np.savez(
            cache, rank=rank, tau=tau, sigma=elbo.sigma, ok=elbo.success & solved.converged,
            ec=(corrected - x_mean) @ coefficient + intercept,
            corrected_rms=np.sqrt((corrected ** 2).mean(1)),
            baseline_change_rms=np.sqrt(((baseline - reference) ** 2).mean(1)),
            **{f"mean {name}": corrected[:, w].mean(1) for name, w in windows.items()},
        )
        print(f"rank {rank} tau {tau}: {int((elbo.success & solved.converged).sum())} of {len(X)} certified, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
