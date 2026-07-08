"""Builds 09_ethiopia_component_ec.ipynb. Run once, then nbconvert --execute.

A few EC calibrations that differ ONLY in the number of PLS components (the notebook-06
"components" theme), each trained on the full 906-sample EC set (all samples, no over-
filtering = the meeting's CAL-0 direction), then applied to the 319 ETAD (Addis/Ethiopia)
spectra (notebook-07's job). Output: predicted Ethiopia EC per filter for each component
count -> tables/ethiopia_new_ec_by_component.csv ("our new Ethiopia data").

Question this answers: how much does the *component count* alone move the Ethiopia EC?
(Companion to 07, which varied the training-sample set instead.)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Ethiopia (ETAD) EC across a few component-count calibrations

Same idea as notebook `07`, but instead of changing *which samples* the calibration trains on,
we hold the training set fixed (**all 906 EC samples, no filtering** — the meeting's CAL-0
direction) and change **only the number of PLS components**. Each calibration is applied to the
**319 ETAD spectra** (identical 2722-pt grid, 3998→500 cm⁻¹) and we read off the Ethiopia EC.

**Why:** component count is decision #6 from the meeting ("pick the first major RMSECV minimum,
be consistent"). The offline reproduction work showed the app's operators often pick a high `k`,
where cross-validated error is climbing — so the honest question is *how much does that choice
alone swing the Ethiopia numbers?* This gives the answer, and hands back a per-filter table.

> Same caveat as `06`/`07`: trained on the IMPROVE-based 906-sample set; the biomass-only outer
> filter is still blocked on Sean's smoke classifier. First-pass application, not the final model.""")

code(r"""from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

PRED = Path("../spartan_ec_2026_06_16")
sys.path.insert(0, str(PRED))
from ftir_pls_calibration import _rmsep_by_ncomp
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

# --- training data (906 x 2722) + the real wavenumber grid ---
Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
y = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Vcols = [c for c in Xdf.columns if c != "id"]
Xtr = Xdf[Vcols].to_numpy(float)
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()
assert len(WN) == Xtr.shape[1]
print("EC training:", Xtr.shape, "| measured EC (ug) range", round(y.min(), 1), "-", round(y.max(), 1),
      "| wavenumber", WN[0], "->", WN[-1])""")

md(r"""## Load the ETAD spectra + metadata and align to the training grid""")

code(r"""ETAD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                      "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
meta = pd.read_csv(ETAD / "ETAD_metadata.csv")

wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")],
               key=lambda c: -float(c))                     # descending wavenumber = training order
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != training grid"
Xeth = spec[wcols].to_numpy(float)
media = spec["MediaId"].to_numpy()

vol = meta.set_index("MediaId")["SampleVolume_m3"]
volume = pd.Series(media).map(vol).to_numpy()               # m3 per ETAD filter (for ug -> ug/m3)
print("ETAD spectra aligned:", Xeth.shape,
      "| filters with sample volume:", int(np.isfinite(volume).sum()), "of", len(media))""")

md(r"""## Pick the component counts

Use the meeting's rule (`first_major_min`: first k within 2% of the global RMSECV minimum) as the
principled choice, then bracket it with a low / mid / high ladder so the *sensitivity* to component
count is visible. All calibrations train on the full 906 EC set (no filtering).""")

code(r"""def first_major_min(rmse, rel_tol=0.02):
    r = np.asarray(rmse, float)
    return int(np.argmax(r <= r.min() * (1 + rel_tol))) + 1

# CV-RMSEP curve (5-fold, same helper as notebook 06)
kmax = max(2, min(30, int(len(y) * 4 / 5) - 2))
rc = _rmsep_by_ncomp(Xtr, y, range(1, kmax + 1), cv=5, seed=0)
ks_cv = np.array(sorted(rc)); rv_cv = np.array([rc[k] for k in ks_cv])
kmin = first_major_min(rv_cv)

# a few component counts to build: the CV pick + a bracketing ladder (high k probes overfitting)
LADDER = sorted(set([3, 10, 18, 25, 40, kmin]))
LADDER = [k for k in LADDER if k < min(Xtr.shape)]
print(f"first_major_min (recommended) k = {kmin}   |   building calibrations at k = {LADDER}")""")

md(r"""## Build each calibration and predict Ethiopia EC

`calib_id = EC_all_nofilt_k<K>` (naming scheme). For each: fit PLS on all 906, predict loading (µg)
for every ETAD filter, then divide by sample volume for concentration (µg/m³).""")

code(r"""load_pred, conc_pred, ids = {}, {}, []
for k in LADDER:
    cid = f"EC_all_nofilt_k{k}" + ("_kmin" if k == kmin else "")
    ids.append((k, cid))
    mdl = PLSRegression(n_components=k, scale=False).fit(Xtr, y)
    load = mdl.predict(Xeth).ravel()
    load_pred[cid] = load
    with np.errstate(divide="ignore", invalid="ignore"):
        conc_pred[cid] = np.where(np.isfinite(volume) & (volume > 0), load / volume, np.nan)

load_df = pd.DataFrame(load_pred, index=media); load_df.index.name = "MediaId"
conc_df = pd.DataFrame(conc_pred, index=media); conc_df.index.name = "MediaId"
KMIN_ID = [cid for k, cid in ids if k == kmin][0]

print("Predicted ETAD EC concentration (µg/m³) by component count — summary:")
print(conc_df.describe().loc[["mean", "50%", "min", "max"]].round(2).to_string())""")

md(r"""## Our new Ethiopia data — per-filter EC table

One row per ETAD filter, with site/date/volume metadata and the predicted EC (µg/m³) from every
component-count calibration, plus the recommended (`first_major_min`) column called `EC_recommended`.""")

code(r"""# ETAD spectra can have several analyses per MediaId, so build positionally (one row per
# spectrum, unique SampleAnalysisId) and map metadata by MediaId — never a many-to-many join.
meta_by_id = meta.drop_duplicates("MediaId").set_index("MediaId")
eth = pd.DataFrame({"SampleAnalysisId": spec["SampleAnalysisId"].values, "MediaId": media})
for col in ["SiteCode", "ExternalFilterId", "SamplingStartDate", "SamplingEndDate",
            "MassCollectedOnFilter_ug", "SampleVolume_m3"]:
    eth[col] = eth["MediaId"].map(meta_by_id[col])
for _, cid in ids:
    eth["EC_ugm3__" + cid] = conc_pred[cid]               # arrays aligned to spectra rows
eth["EC_recommended_ugm3"] = conc_pred[KMIN_ID]           # first_major_min pick

eth.round(4).to_csv("tables/ethiopia_new_ec_by_component.csv", index=False)
conc_df.round(4).to_csv("tables/etad_ec_by_component_conc.csv")
load_df.round(4).to_csv("tables/etad_ec_by_component_loading.csv")
print(f"wrote tables/ethiopia_new_ec_by_component.csv  ({len(eth)} filters, "
      f"recommended = {KMIN_ID})")
eth.head(8)""")

md(r"""## How much does the component count alone move the Ethiopia EC?""")

code(r"""summary = pd.DataFrame({
    "median_EC_ugm3": conc_df.median(),
    "mean_EC_ugm3": conc_df.mean(),
    "pct_negative": (conc_df < 0).mean() * 100,          # negative EC = extrapolation warning
    "ratio_vs_kmin": conc_df.median() / conc_df[KMIN_ID].median(),
}).round(3)
print(summary.to_string())
summary.to_csv("tables/etad_ec_component_summary.csv")
swing = conc_df.median().max() / conc_df.median().min()
print(f"\nComponent choice alone swings the median Ethiopia EC by {swing:.2f}x "
      f"(from {conc_df.median().min():.2f} to {conc_df.median().max():.2f} µg/m³).")""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) RMSECV curve with the chosen ks marked
axes[0].plot(ks_cv, rv_cv, "-o", ms=3, color="#1f77b4")
for k, cid in ids:
    axes[0].axvline(k, color=("#d62728" if k == kmin else "#999999"),
                    ls=("-" if k == kmin else ":"), lw=1)
axes[0].set_xlabel("n components"); axes[0].set_ylabel("RMSECV (µg, 5-fold)")
axes[0].set_title(f"RMSECV — recommended k={kmin} (red)")

# (b) box plot of Ethiopia EC by component count
labels = [cid.replace("EC_all_nofilt_", "") for _, cid in ids]
axes[1].boxplot([conc_df[cid].dropna().values for _, cid in ids], labels=labels, showfliers=False)
axes[1].axhline(0, color="red", ls=":", lw=1)
axes[1].set_ylabel("predicted EC (µg/m³)"); axes[1].tick_params(axis="x", labelrotation=30)
axes[1].set_title("ETAD EC by component count (n=319)")

# (c) median Ethiopia EC vs k
axes[2].plot([k for k, _ in ids], [conc_df[cid].median() for _, cid in ids], "-o", color="#2ca02c")
axes[2].axvline(kmin, color="#d62728", ls="-", lw=1, label=f"recommended k={kmin}")
axes[2].set_xlabel("n components"); axes[2].set_ylabel("median ETAD EC (µg/m³)")
axes[2].set_title("Median Ethiopia EC vs component count"); axes[2].legend(fontsize=8)

plt.tight_layout(); plt.savefig("figures/fig09_ethiopia_ec_by_component.png", dpi=140, bbox_inches="tight")
print("saved figures/fig09_ethiopia_ec_by_component.png"); plt.show()""")

md(r"""### Reading it
- **`tables/ethiopia_new_ec_by_component.csv`** is the deliverable: one row per ETAD filter, EC
  (µg/m³) under each component-count calibration, plus `EC_recommended_ugm3` (the `first_major_min`
  pick) as the single best-estimate column.
- The **swing factor** says how much the component choice alone changes Addis EC — companion to `07`'s
  finding that the *sample-selection* choice swings it ~2×.
- **High-k calibrations** (e.g. k=40) sit where cross-validated error is climbing; watch for negative
  or inflated EC (`pct_negative`) — those are overfitting/extrapolation, not better estimates.
- Next: once Sean's smoke classifier lands, refit on the biomass-only subset; and set the recommended
  `k` with the 2nd-derivative preprocessing test from `06` before finalizing.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("09_ethiopia_component_ec.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 09_ethiopia_component_ec.ipynb")
