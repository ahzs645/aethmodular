"""Build vibes_newton_refit_20261103.ipynb from the outputs of vibes_newton_refit_20261103.py (no fitting here)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(True)}


cells = [
    md("""
# VIBES with the ELBO solved to convergence (3 Nov 2026)

Every step here reads saved outputs of `workflows/vibes_newton_refit_20261103.py`; nothing is refitted.
Conclusions and caveats are in `docs/vibes-newton-adoption-tests-2026-11-03.md`.
"""),
    code("""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
AREA = Path(os.environ.get("AETH_AREA", "."))
T = AREA / "output/tables"; N = T / "vibes_newton_20261103"; G = T / "vibes_gpu_bench_20261103"
SAVED, NEWTON = "#2a78d6", "#eb6834"; INK = "#3a3a38"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a95", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": .6, "lines.linewidth": 1.2})
pd.set_option("display.width", 200, "display.max_columns", 30)
"""),
    md("## 1. L-BFGS-B port against the saved run\nFused SciPy is the noise floor: same algorithm, last bits of the objective differ."),
    code("""
pd.read_csv(N / "port/port_vs_saved.csv").rename(columns={"s1": "sigma >1 %", "s10": "sigma >10 %", "s2x": "sigma >2x", "b5": "baselines >5 %", "b20": "baselines >20 %", "short": "stops <30 its", "short_shared": "shared with saved (155)", "same_it": "same iteration count"})
"""),
    md("Does a tighter `ftol` alone cure the premature stops? 60 premature-stop and 60 other spectra, SciPy, maxiter 10,000."),
    code("""
d = pd.read_csv(N / "port/ftol.csv")
d.groupby(["ftol", "short"]).agg(n=("row", "size"), median_iterations=("nit", "median"), seconds_total=("s", "sum"), median_gap_to_optimum=("gap", "median"), gap_over_1=("gap", lambda v: int((v > 1).sum()))).round(3)
"""),
    md("## 2. Calibration rebuilt on Newton baselines\nSame procedure as the frozen model. On the saved baselines it returns the frozen coefficients to 6e-15."),
    code("""
pd.read_csv(N / "calibration_scores.csv").round(4)
"""),
    md("## 3. Transfer scoring"),
    code("""
r = pd.read_csv(N / "transfer/ratios_by_loading.csv")
w = ["IMPROVE out-of-site MAE (µg)", "Addis OLS slope vs HIPS", "Addis OLS intercept vs HIPS (µg)"]
r[r.what.isin(w) & r.model.isin(["vibes", "vibes_newton", "anchorhp_newton"])].pivot_table(index=["what", "k"], columns="model", values="median").round(3)
"""),
    md("## 4. Functional-group band areas of the corrected spectra"),
    code("""
pd.read_csv(N / "band_areas.csv").round(4)
"""),
    code("""
z = np.load(N / "band_areas_per_spectrum.npz")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, b in zip(axs, ["carbonyl 1650-1800", "1600 band 1500-1650"]):
    a = z[f"addis|{b}"]; lim = [min(a.min(), 0), a.max() * 1.05]
    ax.plot(lim, lim, color="#9a9a95", lw=.8)
    ax.scatter(a[:, 0], a[:, 1], s=14, color=NEWTON, edgecolor="white", linewidth=.4)
    ax.set(xlim=lim, ylim=lim, xlabel="saved baseline, band area", ylabel="Newton baseline, band area", title=f"Addis, {b} cm⁻¹", aspect="equal")
fig.tight_layout(); plt.show()
"""),
    md("## 5. Lots, blanks and spikes\nSame scoring code as `vibes_lot241_20261026`."),
    code("""
A = pd.read_csv(T / "vibes_lot241_20261026/blank_residue_by_lot.csv"); B = pd.read_csv(N / "lots/blank_residue_by_lot.csv")
m = A.merge(B, on=["family", "lot"], suffixes=("_saved", "_newton"))
print("failed fits, all banks: saved", int(A.failed.sum()), " Newton", int(B.failed.sum()))
m[m.family.isin(["b87", "lot241_spartan_r70", "lobo_r70"])][["family", "lot", "n_saved", "failed_saved", "failed_newton", "rms_all_saved", "rms_all_newton", "rms_ptfe_saved", "rms_ptfe_newton"]].round(6)
"""),
    code("""
A = pd.read_csv(T / "vibes_lot241_20261026/spike_recovery_by_lot.csv"); B = pd.read_csv(N / "lots/spike_recovery_by_lot.csv")
m = A.merge(B, on=["family", "lot", "amplitude"], suffixes=("_saved", "_newton"))
m[m.family.eq("b87")][["lot", "amplitude", "rel_rmse_saved", "rel_rmse_newton", "recovered_2920_saved", "recovered_2920_newton"]].round(4)
"""),
    md("## 6. Continuum injection"),
    code("""
pd.read_csv(N / "inject/summary.csv").round(5)
"""),
    md("## 7. Blank banks\nSame scoring as `run_oct1_remaining_20261007`; the capped 87-blank arrays stay the saved ones."),
    code("""
from IPython.display import display
for f in sorted((N / "banks").glob("*.csv"))[:8]:
    print(f.name)
    display(pd.read_csv(f).head(8).round(4))
"""),
    md("## 8. The Addis premature-stop scans\nResidual from the Addis EC-HIPS line fitted on filters whose baseline did not change (µg/m³)."),
    code("""
r = pd.read_csv(N / "within_site_residuals.csv")
print("all changed filters, closer to their site line:", r.closer.value_counts().to_dict())
chk = pd.read_csv(N / "addis_premature_checks.csv")  # every Addis PM2.5 early stop, not only the largest movers
a = r[r["filter"].isin(chk.filter_key)].sort_values("dEC")
print("the %d Addis early stops:" % len(a), a.closer.value_counts().to_dict(), "| mean residual saved %.3f, Newton %.3f; mean |residual| saved %.3f, Newton %.3f" % (a.res_saved.mean(), a.res_newton.mean(), a.res_saved.abs().mean(), a.res_newton.abs().mean()))
a[["filter", "it", "es", "en", "h", "res_saved", "res_newton"]].round(3)
"""),
    code("""
fig, ax = plt.subplots(figsize=(6.4, 3.2))
y = np.arange(len(a))
ax.hlines(y, a.res_saved, a.res_newton, color="#c3c2b7", lw=2, zorder=1)
ax.scatter(a.res_saved, y, s=46, color=SAVED, edgecolor="white", linewidth=1, zorder=2, label="Saved baseline")
ax.scatter(a.res_newton, y, s=46, color=NEWTON, edgecolor="white", linewidth=1, zorder=2, label="Newton baseline")
ax.axvline(0, color="#9a9a95", lw=.8); ax.set_yticks(y, a["filter"]); ax.grid(axis="y", visible=False)
ax.set(xlabel="EC minus site line (µg/m³)", title="Addis early stops: 6 of 10 move toward the site line"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); plt.show()
"""),
    code("""
sv = np.load(G / "bundle_v2/saved_baselines.npy", mmap_mode="r"); nw = np.load(T / "network_baselines_v2_20261103/vibes_baselines_model_grid.npy", mmap_mode="r")
X = np.load(G / "bundle_v2/X_model_grid.npy", mmap_mode="r"); wn = np.load(G / "bundle_v2/background_model.npz")["wn"]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(wn, np.asarray(X[1043]) - np.asarray(sv[1043]), color=SAVED, label="Saved baseline")
ax.plot(wn, np.asarray(X[1043]) - np.asarray(nw[1043]), color=NEWTON, label="Newton baseline")
ax.set(xlim=(4000, 500), xlabel="Wavenumber (cm⁻¹)", ylabel="Corrected absorbance", title="ETAD-0053: stopped at iteration 13 in the saved run"); ax.legend(frameon=False)
fig.tight_layout(); plt.show()
"""),
    md("## 9. The v2 run"),
    code("""
print(json.dumps(json.loads((T / "network_baselines_v2_20261103/MANIFEST.json").read_text()), indent=1))
d = pd.read_csv(T / "network_baselines_v2_20261103/vibes_fit_diagnostics.csv")
d.groupby("filter_type").agg(analyses=("case_row", "size"), premature=("premature_stop", "sum"), moved_gt5pct=("baseline_change_frac_rms", lambda v: int((v > .05).sum())), moved_gt20pct=("baseline_change_frac_rms", lambda v: int((v > .2).sum())))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
out = HERE / "vibes_newton_refit_20261103.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, file=sys.stderr)
