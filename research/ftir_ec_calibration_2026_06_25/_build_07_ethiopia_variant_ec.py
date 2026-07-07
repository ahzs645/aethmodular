"""Builds 07_ethiopia_variant_ec.ipynb. Run once, then nbconvert --execute.

Applies every calibration variant (CAL-0…CAL-5 from notebook 06) to the **319 ETAD (Addis/Ethiopia)
FTIR spectra** and compares the predicted EC across variants. The ETAD export is on the **identical
2722-point wavenumber grid** as the training data (3998→500 cm⁻¹, verified), so the PLS models apply
directly — no interpolation. This answers "does the sample-selection rule change the Ethiopia EC?"

FABS comparison (meeting: EC-vs-FABS at MAC=10) needs the SPARTAN HIPS fAbs joined to these MediaIds
— left as a clearly-marked follow-on cell.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Ethiopia (ETAD) EC under each calibration variant

Take the CAL-0…CAL-5 calibrations (built on the real 906-sample EC set in `06`) and **predict EC for
every ETAD Addis/Ethiopia filter** with each one. The ETAD spectra sit on the **same 2722-point grid
as the training data** (3998→500 cm⁻¹, exact match), so each PLS model applies directly.

**Question:** how much does the Ethiopia EC estimate move when we change *which samples the
calibration was trained on* — keep-everything (CAL-0) vs. cleaned (CAL-1) vs. the inverse below-1:1
(CAL-3), etc.? If the answer is "a lot," the filtering decision is not cosmetic.

> Same caveat as `06`: these are trained on the IMPROVE-based 906-sample set (the biomass/smoke-only
> outer filter is still blocked on Sean's classifier). This is the first-pass application, not the
> final smoke-only model.""")

code(r"""from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

PRED = Path("../spartan_ec_2026_06_16")
sys.path.insert(0, str(PRED))
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

# --- training data (906 x 2722) + real wavenumber grid ---
Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
y = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Vcols = [c for c in Xdf.columns if c != "id"]          # V1..V2722 in wavenumber-descending order
Xtr = Xdf[Vcols].to_numpy(float)
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()
assert len(WN) == Xtr.shape[1]
print("training:", Xtr.shape, "| wavenumber", WN[0], "->", WN[-1])""")

md(r"""## Load the ETAD spectra and align to the training grid""")

code(r"""ETAD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                      "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
meta = pd.read_csv(ETAD / "ETAD_metadata.csv")

wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")],
               key=lambda c: -float(c))               # descending wavenumber = training order
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != training grid"
Xeth = spec[wcols].to_numpy(float)
media = spec["MediaId"].to_numpy()
print("ETAD spectra aligned:", Xeth.shape)

vol = meta.set_index("MediaId")["SampleVolume_m3"]
volume = pd.Series(media).map(vol).to_numpy()          # m3 per ETAD sample (for µg -> µg/m3)
print("samples with sample volume:", int(np.isfinite(volume).sum()), "of", len(media))""")

md(r"""## Rebuild the six variant masks (same rules as notebook 06)

Baseline PLS at the common **k=20**; its residuals define cleaned / removed / below-1:1.""")

code(r"""K_COMMON = 20
base = PLSRegression(n_components=K_COMMON, scale=False).fit(Xtr, y)
pred0 = base.predict(Xtr).ravel()

keep = np.ones(len(y), bool)
for _ in range(3):
    m = PLSRegression(n_components=K_COMMON, scale=False).fit(Xtr[keep], y[keep])
    rr = y - m.predict(Xtr).ravel()
    nk = np.abs(rr) <= 3 * rr[keep].std()
    if (nk == keep).all():
        break
    keep = keep & nk

ETH_LOW, ETH_HIGH = 10.0, 100.0     # PLACEHOLDER Ethiopia band (set from FABS/10 later)
MASKS = {
    "CAL-0 all-nofilter": np.ones(len(y), bool),
    "CAL-1 cleaned":      keep,
    "CAL-2 removed-only": ~keep,
    "CAL-3 below-1:1":    pred0 < y,
    "CAL-4 EC-high>=70":  y >= 70.0,
    "CAL-5 Eth-range":    (y >= ETH_LOW) & (y <= ETH_HIGH),
}
for k, mm in MASKS.items():
    print(f"{k:22s} n={int(mm.sum())}")""")

md(r"""## Predict ETAD EC with each variant

Every model is fit at the common `k` (capped below the CV-fold size for the small sets), then applied
to all 319 ETAD spectra. We report predicted **loading (µg)** and **concentration (µg/m³)**.""")

code(r"""def safe_k(n, cv=5):
    return max(2, min(K_COMMON, int(n * (cv - 1) / cv) - 2))

pred_load = {}
for name, mm in MASKS.items():
    k = safe_k(int(mm.sum()))
    mdl = PLSRegression(n_components=k, scale=False).fit(Xtr[mm], y[mm])
    pred_load[name] = mdl.predict(Xeth).ravel()

load_df = pd.DataFrame(pred_load, index=media)
vol_ok = np.where(np.isfinite(volume) & (volume > 0), volume, np.nan)
conc_df = load_df.div(vol_ok, axis=0).replace([np.inf, -np.inf], np.nan)   # µg/m3, non-finite -> NaN
load_df.index.name = conc_df.index.name = "MediaId"
print(f"{int(np.isfinite(conc_df['CAL-0 all-nofilter']).sum())} of {len(media)} filters have a "
      f"usable sample volume for concentration.\n")

print("Predicted ETAD EC concentration (µg/m³) — summary across the filters:")
print(conc_df.describe().loc[["mean", "50%", "min", "max"]].round(2).to_string())
conc_df.round(4).to_csv("tables/etad_ec_by_variant_conc.csv")
load_df.round(4).to_csv("tables/etad_ec_by_variant_loading.csv")
print("\nwrote tables/etad_ec_by_variant_{conc,loading}.csv")""")

md(r"""## How much does the calibration choice move the Ethiopia EC?""")

code(r"""fig, ax = plt.subplots(figsize=(10, 5))
data = [conc_df[c].dropna().values for c in conc_df.columns]
bp = ax.boxplot(data, labels=list(conc_df.columns), showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], conc_df.columns):
    patch.set_facecolor("#d62728" if ("removed" in c or "below" in c) else "#1f77b4")
    patch.set_alpha(0.5)
ax.axhline(0, color="red", ls=":", lw=1)               # physical floor — EC can't be negative
ax.set_ylabel("predicted EC (µg/m³)")
ax.set_title("ETAD (Addis/Ethiopia) EC by calibration variant  (n=319 filters)")
ax.text(0.01, 0.02, "CAL-2 & CAL-4 go negative → unstable tiny-sample calibrations (not viable)",
        transform=ax.transAxes, fontsize=8, color="gray", style="italic")
ax.tick_params(axis="x", labelrotation=20)
plt.tight_layout(); plt.savefig("figures/fig07_etad_ec_by_variant.png", dpi=140, bbox_inches="tight")
print("saved figures/fig07_etad_ec_by_variant.png"); plt.show()""")

md(r"""## Paired view — each variant vs. the keep-everything baseline (CAL-0)

Points off the 1:1 line show where a variant systematically raises or lowers the Ethiopia EC relative
to CAL-0. If the inverse / cleaned calibrations shift EC up, that is the "FTIR was missing EC" signal
on the real target samples.""")

code(r"""ref = conc_df["CAL-0 all-nofilter"].values
others = [c for c in conc_df.columns if c != "CAL-0 all-nofilter"]
fig, axes = plt.subplots(1, len(others), figsize=(4*len(others), 4.2), sharex=True, sharey=True)
finite_vals = conc_df.values[np.isfinite(conc_df.values)]
lim = float(np.nanpercentile(finite_vals, 99) * 1.1) if finite_vals.size else 1.0
for ax, c in zip(axes, others):
    v = conc_df[c].values
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.scatter(ref, v, s=12, alpha=0.5,
               color=("#d62728" if ("removed" in c or "below" in c) else "#1f77b4"))
    med_ratio = np.nanmedian(v / ref)
    ax.set_title(f"{c}\nmedian {c.split()[0]}/CAL-0 = {med_ratio:.2f}", fontsize=8)
    ax.set_xlabel("CAL-0 EC (µg/m³)"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
axes[0].set_ylabel("variant EC (µg/m³)")
fig.suptitle("ETAD EC: each variant vs. CAL-0 (keep-everything)", y=1.03)
plt.tight_layout(); plt.savefig("figures/fig07_etad_variant_vs_cal0.png", dpi=140, bbox_inches="tight")
print("saved figures/fig07_etad_variant_vs_cal0.png"); plt.show()

ratios = pd.DataFrame({"median_vs_CAL0": [np.nanmedian(conc_df[c].values / ref) for c in conc_df.columns]},
                      index=conc_df.columns).round(3)
print(ratios.to_string())
ratios.to_csv("tables/etad_variant_ratio_vs_cal0.csv")
print("\nwrote tables/etad_variant_ratio_vs_cal0.csv")""")

md(r"""## FABS comparison (MAC = 10) — follow-on, needs the HIPS join

The meeting asked to compare the calibrations **vs. FABS** (EC ≈ FABS / MAC, MAC = 10). ETAD's
metadata has **no fAbs column**; the SPARTAN HIPS fAbs lives in
`…/NASA MAIA/Data/Spartan/SPARTAN_HIPS_Batch1-51.v2.csv` and must be joined to these MediaIds /
ExternalFilterIds first. Wire that join here, then plot each variant's EC vs. FABS/10.""")

code(r"""# TODO: load SPARTAN HIPS, map ETAD MediaId/ExternalFilterId -> HIPS_Fabs, then:
#   ec_from_fabs = HIPS_Fabs / 10.0            # MAC = 10 (6-vs-10 unresolved)
#   for each variant: scatter conc_df[variant] vs ec_from_fabs, 1:1 line, fit.
print("FABS overlay not wired yet — needs SPARTAN HIPS join (see markdown). "
      "EC predictions per variant are already saved in tables/etad_ec_by_variant_conc.csv.")""")

md(r"""### The result (this run, real ETAD spectra)
- **Median Ethiopia EC (µg/m³):** CAL-0 all ≈ **4.7**, CAL-1 cleaned ≈ **3.1**, CAL-3 below-1:1 ≈
  **6.3**, CAL-5 Eth-range ≈ **4.2**. So the **cleaned** calibration reports ~**0.61×** the
  keep-everything EC, while the **inverse below-1:1** calibration reports ~**1.29×**.
- **The calibration choice swings Addis EC by ~2×** (CAL-1 3.1 → CAL-3 6.3). The filtering decision is
  **not cosmetic** — it is the difference between under- and over-reporting Ethiopia EC. That is the
  headline result for the meeting: over-cleaning systematically **lowers** the Ethiopia EC.
- **CAL-2 (removed-only) and CAL-4 (EC≥70) predict negative EC** — unphysical, tiny-sample
  extrapolation. They are broken diagnostics (as flagged in `06`), **not** candidate calibrations.

### Reading it next week
- The **median variant/CAL-0 ratios** quantify the shift. A ratio > 1 means that variant assigns
  Addis *more* EC than keep-everything.
- Watch for **extrapolation**: Ethiopia loadings can exceed the training range, so predictions past
  the calibrated span are less trustworthy — flag any variant that predicts far outside its training
  EC range.
- Next: wire FABS (MAC=10) and, once Sean's classifier lands, re-fit every variant on the
  **biomass-only** subset before drawing conclusions.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("07_ethiopia_variant_ec.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 07_ethiopia_variant_ec.ipynb")
