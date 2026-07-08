"""Builds 10_ethiopia_ec_both_lots.ipynb. Run once, then nbconvert --execute.

Companion to notebook 09. Notebook 09 trained on the older 906-sample EC set, which turns
out to be **lot 251 only**. The fresh 2026-07-07 export spans **both** ETAD lots
(906 lot-251 + 29 lot-248 = 935). This rebuilds the same component-count EC calibrations on
the both-lots set and applies them to the 319 ETAD spectra, so we can see whether *adding
lot 248* moves the Ethiopia EC.

Three training sets, identical pipeline, to separate the effects:
  - old_906_lot251 : the notebook-09 baseline (spartan rds_EC, 906, lot 251)
  - fresh_lot251   : the fresh 2026-07-07 export with lot 248 dropped (906, lot 251) -> isolates data-vintage
  - fresh_both_lots: the fresh export, all 935 (lots 251+248)       -> isolates the lot-248 effect
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Ethiopia EC — does spanning both filter lots change it?

Notebook `09` trained on the older **906-sample** EC set. Joining it to the filter-lot metadata
shows that set is **100% lot 251** — but the Ethiopia (ETAD) filters use **lots 251 and 248**, so
the calibration did *not* span both lots.

The **fresh 2026-07-07 export** does: **906 lot-251 + 29 lot-248 = 935**. Here we rebuild the same
component-count EC calibrations on the both-lots set and re-predict the 319 ETAD filters, comparing
against the lot-251-only baseline. Because the fresh set is exactly *the lot-251 samples plus 29
lot-248 samples*, we can cleanly isolate the lot-248 contribution.

> Same caveats as `06`/`07`/`09`: IMPROVE-based training; the biomass-only outer filter still waits
> on Sean's smoke classifier. First-pass application.""")

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

# --- old 906-sample set (lot 251 only) — the notebook-09 baseline ---
Xold = pd.read_csv(PRED / "data/rds_EC_X.csv")
yold = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Vold = [c for c in Xold.columns if c != "id"]
Xold = Xold[Vold].to_numpy(float)
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()

# --- fresh 2026-07-07 set (both lots) ---
Xf = pd.read_csv("data/ec0707_X.csv")
yf = pd.read_csv("data/ec0707_Y.csv")["Y_measured"].to_numpy()
wn_f = pd.read_csv("data/ec0707_wn.csv")["wavenumber"].to_numpy()
ids_f = Xf["id"].to_numpy()
Vf = [c for c in Xf.columns if c != "id"]
Xf = Xf[Vf].to_numpy(float)
assert np.allclose(wn_f, WN), "fresh grid != old grid"

# lot label per fresh sample (join to the DB metadata pull)
meta = pd.read_csv(Path.home() / "Downloads/ftir_metadata.csv", dtype={"LotNumber": str})
lut = meta.drop_duplicates("AnalysisId").set_index("AnalysisId")["LotNumber"]
lot_f = pd.Series(ids_f).map(lut).to_numpy()
is251 = (lot_f == "251")
print(f"old  set: {Xold.shape[0]} (lot 251 only)")
print(f"fresh set: {Xf.shape[0]} = {int(is251.sum())} lot-251 + {int((~is251).sum())} lot-248")""")

md(r"""## Align the ETAD spectra to the (shared) training grid""")

code(r"""ETAD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                      "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
emeta = pd.read_csv(ETAD / "ETAD_metadata.csv")
wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")], key=lambda c: -float(c))
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != training grid"
Xeth = spec[wcols].to_numpy(float)
media = spec["MediaId"].to_numpy()
volume = pd.Series(media).map(emeta.drop_duplicates("MediaId").set_index("MediaId")["SampleVolume_m3"]).to_numpy()
print("ETAD aligned:", Xeth.shape, "| with volume:", int(np.isfinite(volume).sum()), "of", len(media))""")

md(r"""## The three training sets and the component ladder

Component counts from the meeting's `first_major_min` rule on the both-lots set, bracketed by a
low/mid/high ladder (identical to notebook 09 so the numbers line up).""")

code(r"""def first_major_min(rmse, rel_tol=0.02):
    r = np.asarray(rmse, float)
    return int(np.argmax(r <= r.min() * (1 + rel_tol))) + 1

kmax = max(2, min(30, int(len(yf) * 4 / 5) - 2))
rc = _rmsep_by_ncomp(Xf, yf, range(1, kmax + 1), cv=5, seed=0)
ks_cv = np.array(sorted(rc)); rv_cv = np.array([rc[k] for k in ks_cv])
kmin = first_major_min(rv_cv)
LADDER = sorted(set([3, 10, 18, 25, 40, kmin]))
LADDER = [k for k in LADDER if k < min(Xf.shape)]

TRAINSETS = {
    "old_906_lot251":  (Xold, yold),
    "fresh_lot251":    (Xf[is251], yf[is251]),
    "fresh_both_lots": (Xf, yf),
}
print(f"first_major_min (both-lots) k = {kmin}   |   ladder = {LADDER}")
for name, (Xt, yt) in TRAINSETS.items():
    print(f"  {name:16s} n={len(yt)}")""")

md(r"""## Predict ETAD EC for every (training set × component count)""")

code(r"""def conc(load):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(volume) & (volume > 0), load / volume, np.nan)

records = {}   # (trainset, k) -> conc array over ETAD
for name, (Xt, yt) in TRAINSETS.items():
    for k in LADDER:
        mdl = PLSRegression(n_components=k, scale=False).fit(Xt, yt)
        records[(name, k)] = conc(mdl.predict(Xeth).ravel())

# median ETAD EC table: rows = component count, cols = training set
med = pd.DataFrame({name: [np.nanmedian(records[(name, k)]) for k in LADDER] for name in TRAINSETS},
                   index=LADDER)
med.index.name = "ncomp"
print("Median ETAD EC (µg/m³) by training set × components:")
print(med.round(3).to_string())""")

md(r"""## Our new Ethiopia data — from the both-lots calibration

One row per ETAD filter, EC (µg/m³) at each component count from the **both-lots** calibration, plus
`EC_recommended_ugm3` (the `first_major_min` pick) and, for context, the lot-251-only recommended EC.""")

code(r"""m = emeta.drop_duplicates("MediaId").set_index("MediaId")
eth = pd.DataFrame({"SampleAnalysisId": spec["SampleAnalysisId"].values, "MediaId": media})
for c in ["SiteCode", "ExternalFilterId", "SamplingStartDate", "SampleVolume_m3"]:
    eth[c] = eth["MediaId"].map(m[c])
for k in LADDER:
    eth[f"EC_bothlots_k{k}"] = records[("fresh_both_lots", k)]
eth["EC_recommended_ugm3"]      = records[("fresh_both_lots", kmin)]   # both-lots, first_major_min
eth["EC_lot251only_recommended"] = records[("fresh_lot251", kmin)]     # same k, lot-251 only
eth.round(4).to_csv("tables/ethiopia_new_ec_both_lots.csv", index=False)
med.round(4).to_csv("tables/etad_ec_both_lots_median_by_trainset.csv")
print(f"wrote tables/ethiopia_new_ec_both_lots.csv  ({len(eth)} filters, recommended k={kmin})")
eth.head(6)""")

md(r"""## How much does spanning both lots move the Ethiopia EC?""")

code(r"""ref251, refboth = records[("fresh_lot251", kmin)], records[("fresh_both_lots", kmin)]
refold = records[("old_906_lot251", kmin)]
lot248_effect = np.nanmedian(refboth) / np.nanmedian(ref251)      # fresh: +lot248 vs lot251, same data
vintage_effect = np.nanmedian(ref251) / np.nanmedian(refold)      # fresh lot251 vs old lot251
print(f"At the recommended k={kmin}, median ETAD EC (µg/m³):")
print(f"  old 906 (lot 251)      : {np.nanmedian(refold):.3f}")
print(f"  fresh lot 251 (n={int(is251.sum())})   : {np.nanmedian(ref251):.3f}")
print(f"  fresh both lots (n={len(yf)}) : {np.nanmedian(refboth):.3f}")
print(f"\nAdding the 29 lot-248 samples changes median Addis EC by {lot248_effect:.3f}x "
      f"({(lot248_effect-1)*100:+.1f}%).")
print(f"Data-vintage alone (fresh vs old lot-251) changes it {vintage_effect:.3f}x "
      f"({(vintage_effect-1)*100:+.1f}%).")""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
colors = {"old_906_lot251": "#7f7f7f", "fresh_lot251": "#1f77b4", "fresh_both_lots": "#d62728"}
for name in TRAINSETS:
    axes[0].plot(LADDER, [np.nanmedian(records[(name, k)]) for k in LADDER],
                 "-o", color=colors[name], label=name)
axes[0].axvline(kmin, color="k", ls=":", lw=1, label=f"recommended k={kmin}")
axes[0].set_xlabel("n components"); axes[0].set_ylabel("median ETAD EC (µg/m³)")
axes[0].set_title("Median Ethiopia EC vs components, by training set"); axes[0].legend(fontsize=8)

# paired per-filter: both-lots vs lot-251-only at recommended k
axes[1].scatter(ref251, refboth, s=12, alpha=0.5, color="#d62728")
lim = np.nanpercentile(np.concatenate([ref251, refboth]), 99) * 1.1
axes[1].plot([0, lim], [0, lim], "k--", lw=1)
axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim)
axes[1].set_xlabel("lot-251-only EC (µg/m³)"); axes[1].set_ylabel("both-lots EC (µg/m³)")
axes[1].set_title(f"Per-filter, k={kmin}: adding lot 248 (median {lot248_effect:.2f}×)")
plt.tight_layout(); plt.savefig("figures/fig10_ethiopia_both_lots.png", dpi=140, bbox_inches="tight")
print("saved figures/fig10_ethiopia_both_lots.png"); plt.show()""")

md(r"""### Reading it
- **`tables/ethiopia_new_ec_both_lots.csv`** is the deliverable — per-filter Ethiopia EC from the
  both-lots calibration (recommended column = `first_major_min`).
- **lot-248 effect**: the fresh set is the lot-251 samples **plus 29 lot-248**, so the fresh-both vs
  fresh-lot251 comparison isolates exactly what adding lot 248 does — at only ~3% of samples, expect a
  small shift, and the size of it is the headline number.
- **vintage effect**: fresh-lot251 vs old-906 shows any drift from re-pulling the data itself (should
  be ≈1× if it's the same samples).
- Next: once there are *more* lot-248 samples (or Sean's smoke classifier), re-run — 29 samples is a
  first look at whether the lot matters, not the final word.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("10_ethiopia_ec_both_lots.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 10_ethiopia_ec_both_lots.ipynb")
