"""Builds 09_component_selection_fixed.ipynb. Run once, then nbconvert --execute.

Corrects the component-selection graph from notebook 06. It does NOT redefine the
CAL-0..CAL-5 variants (same masks as 06); it fixes HOW their RMSECV is computed, scaled,
plotted, and how k is chosen:

  1. 10-fold INTERLEAVED CV (not 5-fold shuffled) — smoother, and it matched R's pls
     exactly this session. Per-component predictions reconstructed from one fit/fold.
  2. %RMSECV (÷ each subset's mean EC) so the six variants are COMPARABLE — the absolute-µg
     overlay in 06 is dominated by the high-EC diagnostic variants and squishes CAL-0/CAL-5.
  3. Candidate calibrations (CAL-0/1/5) and unstable diagnostics (CAL-2/3/4) in SEPARATE
     panels, so the diagnostics stop visually dominating.
  4. ONE tolerance for k-selection everywhere (06 used 2% in the table but 5% in the
     2nd-derivative section — giving 21 vs 13 for the same CAL-0 curve).

Also shows raw vs 2nd-derivative for CAL-0: the 2nd-derivative curve is far smoother, which
is the real argument for picking k on 2nd-derivative spectra (Weakley).

Grounded in the same data as 06: spartan_ec_2026_06_16/data/rds_EC_{X,Ymeasured}.csv.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Component selection, corrected (fixes the notebook-06 RMSECV graph)

Notebook 06's per-variant RMSECV graph is **misleading**, though the underlying PLS math is
fine and the CAL-0 minimum (k≈21) is robust. Four issues, fixed here:

| # | Problem in 06 | Fix |
|---|---------------|-----|
| 1 | 5-fold *shuffled* CV → jagged curves | **10-fold interleaved** CV (smoother; matched R exactly) |
| 2 | absolute-µg RMSECV overlaid → not comparable across variants | **%RMSECV** (÷ subset mean EC) |
| 3 | diagnostics (CAL-2/3/4) dominate the plot | candidates vs diagnostics in **separate panels** |
| 4 | 2% tol in the table but 5% in the 2nd-deriv section (21 vs 13) | **one tolerance** everywhere |

**This notebook does not change the variant definitions** — it rebuilds the *identical*
CAL-0…CAL-5 masks from notebook 06 and only corrects the component analysis on top.""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter

PRED = Path("../spartan_ec_2026_06_16")
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
y = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
X = Xdf[[c for c in Xdf.columns if c != "id"]].to_numpy(float)
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()
assert len(WN) == X.shape[1]
print("EC set:", X.shape, "| EC range", round(y.min(), 2), "-", round(y.max(), 2),
      "| samples EC>150:", int((y > 150).sum()), "(these few dominate absolute RMSE)")""")

md(r"""## Rebuild the identical CAL-0…CAL-5 masks (verbatim from notebook 06)

`BASELINE_K = 21` is notebook 06's `k0` (its `first_major_min` of the CAL-0 curve); the
baseline model at that k defines the cleaned / removed / below-1:1 sets. Reproducing it here
keeps the variant sets byte-for-byte identical to 06 (906 / 860 / 46 / 469 / 22 / 480).""")

code(r"""BASELINE_K = 21
base = PLSRegression(n_components=BASELINE_K, scale=False).fit(X, y)
pred0 = base.predict(X).ravel()

keep = np.ones(len(y), bool)              # CAL-1: iterative 3-sigma trim at BASELINE_K
for _ in range(3):
    m = PLSRegression(n_components=BASELINE_K, scale=False).fit(X[keep], y[keep])
    rr = y - m.predict(X).ravel()
    newkeep = np.abs(rr) <= 3 * rr[keep].std()
    if (newkeep == keep).all():
        break
    keep = keep & newkeep

ETH_LOW, ETH_HIGH = 10.0, 100.0           # CAL-5 placeholder band (unchanged from 06)
MASKS = {
    "CAL-0 all-nofilter": np.ones(len(y), bool),
    "CAL-1 cleaned":      keep,
    "CAL-2 removed-only": ~keep,
    "CAL-3 below-1:1":    pred0 < y,
    "CAL-4 EC-high>=70":  y >= 70.0,
    "CAL-5 Eth-range":    (y >= ETH_LOW) & (y <= ETH_HIGH),
}
CANDIDATES  = ["CAL-0 all-nofilter", "CAL-1 cleaned", "CAL-5 Eth-range"]
DIAGNOSTICS = ["CAL-2 removed-only", "CAL-3 below-1:1", "CAL-4 EC-high>=70"]
for k, m in MASKS.items():
    print(f"{k:22s} n={int(m.sum()):4d}   mean EC={y[m].mean():6.1f} µg")""")

md(r"""## Corrected CV — 10-fold interleaved, per-component reconstruction

Interleaved folds (`i % 10`) match the app's R `pls` scheme; per-component predictions are
reconstructed from one fit per fold (so the whole curve costs 10 fits, not 10×kmax). `maxc`
is capped below the smallest training-fold size so small variants don't fit degenerate PLS.""")

code(r"""def pred_all_ncomps(m, Xnew, mc):
    Xc = Xnew - m._x_mean
    out = np.empty((Xnew.shape[0], mc))
    for k in range(1, mc + 1):
        b = m.x_rotations_[:, :k] @ m.y_loadings_[:, :k].T
        out[:, k - 1] = (Xc @ b).ravel() + m._y_mean
    return out

def cv_rmsecv(Xs, ys, folds=10, maxc=30):
    f = np.arange(len(ys)) % folds
    mc = int(min(maxc, min((f != i).sum() for i in range(folds)) - 1, Xs.shape[1]))
    press = np.zeros(mc)
    for i in range(folds):
        tr, te = f != i, f == i
        m = PLSRegression(n_components=mc, scale=False).fit(Xs[tr], ys[tr])
        press += np.sum((pred_all_ncomps(m, Xs[te], mc) - ys[te][:, None])**2, axis=0)
    ks = np.arange(1, mc + 1)
    return ks, np.sqrt(press / len(ys))

TOL = 0.05                                  # ONE tolerance, used everywhere
def pick_k(ks, rmsecv, tol=TOL):
    return int(ks[np.argmax(rmsecv <= rmsecv.min() * (1 + tol))])

curves = {}
for name, mask in MASKS.items():
    ks, rc = cv_rmsecv(X[mask], y[mask])
    pct = 100 * rc / y[mask].mean()
    curves[name] = dict(ks=ks, rmsecv=rc, pct=pct, k=pick_k(ks, rc),
                        n=int(mask.sum()), meanEC=float(y[mask].mean()))
    print(f"{name:22s} k*={curves[name]['k']:2d}  RMSECV_min={rc.min():6.2f} µg  "
          f"%RMSECV_min={pct.min():5.1f}%")""")

md(r"""## Figure 1 — %RMSECV, candidates vs. diagnostics (the corrected graph)

Left: the three variants that are actual calibration candidates (CAL-0/1/5). Right: the
small, unstable inverse/threshold diagnostics (CAL-2/3/4) — shown but kept out of the
candidates' scale. Dashed line = the k picked by the single 5%-tolerance rule.""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, group, title in [(axes[0], CANDIDATES, "Candidate calibrations"),
                         (axes[1], DIAGNOSTICS, "Diagnostics (small n — unstable)")]:
    for name in group:
        c = curves[name]
        ax.plot(c["ks"], c["pct"], "-o", ms=3, lw=1.2,
                label=f"{name}  (n={c['n']}, k*={c['k']})")
        ax.axvline(c["k"], ls=":", lw=0.8, color=ax.get_lines()[-1].get_color())
    ax.set_title(title); ax.set_xlabel("n PLS components")
    ax.set_ylabel("%RMSECV  (= RMSECV ÷ subset mean EC)")
    ax.legend(fontsize=8)
fig.suptitle("Corrected component selection — %RMSECV, 10-fold interleaved CV", y=1.02, fontsize=13)
plt.tight_layout(); plt.savefig("figures/fig09_component_selection_pct.png", dpi=140, bbox_inches="tight")
print("saved figures/fig09_component_selection_pct.png"); plt.show()""")

md(r"""## Figure 2 — raw vs. 2nd-derivative (CAL-0): why the raw curve is jagged

The raw curve wobbles (low components model the PTFE baseline, and a handful of extreme-EC
samples dominate the error). The **2nd-derivative** curve is far smoother and near its floor
from very few components — the Weakley argument for selecting k on 2nd-derivative spectra.""")

code(r"""X2 = savgol_filter(X, window_length=11, polyorder=2, deriv=2, axis=1)
ks_raw, rc_raw = cv_rmsecv(X, y)
ks_d2,  rc_d2  = cv_rmsecv(X2, y)
pct_raw = 100 * rc_raw / y.mean(); pct_d2 = 100 * rc_d2 / y.mean()
k_raw, k_d2 = pick_k(ks_raw, rc_raw), pick_k(ks_d2, rc_d2)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(ks_raw, pct_raw, "-o", ms=3, color="#1f77b4", label=f"raw  (k*={k_raw})")
ax.plot(ks_d2,  pct_d2,  "-o", ms=3, color="#d62728", label=f"2nd-derivative  (k*={k_d2})")
ax.axvline(k_raw, ls=":", lw=1, color="#1f77b4"); ax.axvline(k_d2, ls=":", lw=1, color="#d62728")
ax.set_xlabel("n PLS components"); ax.set_ylabel("%RMSECV (CAL-0, n=906)")
ax.set_title("CAL-0: raw spectra are jagged; 2nd-derivative is smooth (one 5% tol → consistent k)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig09_raw_vs_d2.png", dpi=140, bbox_inches="tight")
print(f"raw k*={k_raw} (min {rc_raw.min():.2f} µg) | 2nd-deriv k*={k_d2} (min {rc_d2.min():.2f} µg)")
print("saved figures/fig09_raw_vs_d2.png"); plt.show()""")

md(r"""## Corrected summary table

Compare on **%RMSECV** (comparable across variants), with the single-rule `k*`. Contrast
with notebook 06's `k_firstmin` (2% tol on the jagged 5-fold curve) to see what changed.""")

code(r"""old = pd.read_csv("tables/component_selection_by_variant.csv").set_index("name")
rows = []
for name, c in curves.items():
    rows.append({"name": name, "n": c["n"], "mean_EC": round(c["meanEC"], 1),
                 "k*_corrected": c["k"], "RMSECV_min_ug": round(float(c["rmsecv"].min()), 2),
                 "pctRMSECV_min": round(float(c["pct"].min()), 1),
                 "k_firstmin_nb06": int(old.loc[name, "k_firstmin"]) if name in old.index else None})
tbl = pd.DataFrame(rows)
tbl.to_csv("tables/component_selection_corrected.csv", index=False)
print(tbl.to_string(index=False))
print("\nwrote tables/component_selection_corrected.csv")""")

md(r"""### What changed vs. notebook 06
- The **honest comparison is %RMSECV** — on it, CAL-0 (keep-everything) and CAL-5 are the
  only stable candidates; CAL-2/3/4 have large %RMSECV (small-n instability), confirming they
  are diagnostics, not calibrations.
- `k*` here uses **one** 5% tolerance on the smoother 10-fold curve; it no longer flips
  between 21 and 13 depending on which section you read.
- **Adopt 2nd-derivative preprocessing before finalizing k** — Figure 2 shows the raw curve's
  jaggedness is a preprocessing artifact, not real model structure.
- Variant definitions are unchanged; only the component-selection analysis is corrected.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("09_component_selection_fixed.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 09_component_selection_fixed.ipynb")
