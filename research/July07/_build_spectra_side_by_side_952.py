"""Builds spectra_side_by_side_952_improve_vs_ethiopia.ipynb. Run once, then nbconvert --execute.

Same comparison as `_build_spectra_side_by_side.py`, but the IMPROVE calibration set is the **new
2026-07-07 952-sample pull** (`ftir-spectra-2026-07-07.csv`) instead of the older 906-sample
`rds_EC_X.csv`. Everything else — ALS baseline, side-by-side format, ETAD load — is identical, so the
two notebooks are directly comparable.

IMPROVE calibration spectra:  My Drive/ftir-spectra-2026-07-07.csv   (952 samples x 2722 wavenumbers)
IMPROVE full DB metadata:     My Drive/ftir_metadata.csv            (join by AnalysisId -> Lot/Site)
Ethiopia (prediction) spectra: .../DAVIS/ETAD FTIR/ETAD_FTIR_spectra.csv  (319 x 2722, same grid)

The 952-sample export already carries the wavenumber grid in its column names (3998->500 cm-1) — the
identical grid the ETAD export uses — so the overlay needs no interpolation. The pls-EC / pls-OC RDS
files are R PLS *models* (for applying a calibration); they are not needed for this spectral-similarity
figure and are left untouched.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# IMPROVE calibration (952-sample, 2026-07-07) vs. Ethiopia (ETAD) — side by side, baselined

Same first-principles comparison Ann asked for, now on the **new 952-sample IMPROVE calibration pull**
(`ftir-spectra-2026-07-07.csv`) rather than the older 906-sample set. The question is unchanged: do the
IMPROVE samples we calibrate with actually *look* spectrally like the Ethiopia (ETAD/Addis) samples we
predict? A spectral mismatch is a **physical** reason a general calibration would fail on Ethiopia —
the argument for a smoke-only calibration.

**Inputs**
- IMPROVE calibration spectra — `My Drive/ftir-spectra-2026-07-07.csv` (**952** samples × 2722 cm⁻¹)
- IMPROVE DB metadata — `My Drive/ftir_metadata.csv` (join by `AnalysisId` → LotNumber / Site)
- Ethiopia spectra — `.../DAVIS/ETAD FTIR/ETAD_FTIR_spectra.csv` (**319** × 2722, identical grid)

**Baseline** = asymmetric least squares (ALS), the same stand-in for Satoshi's **AirSpec** used in the
906-sample notebook — swap only `baseline_correct()` once AirSpec runs locally. The `pls-EC/OC RDS`
models aren't needed here (this figure is about spectral *shape*, not applying a calibration).""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "0.92", "figure.dpi": 110})

DRIVE = Path.home() / "Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive"
IMP_SPECTRA = DRIVE / "ftir-spectra-2026-07-07.csv"      # 952-sample IMPROVE calibration pull
IMP_META    = DRIVE / "ftir_metadata.csv"                # full DB metadata (AnalysisId -> Lot/Site)
ETAD = (DRIVE / "University/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)
print("IMPROVE 952 spectra:", IMP_SPECTRA.exists(), "| metadata:", IMP_META.exists(),
      "| ETAD dir:", ETAD.exists())


# --- baseline correction (ALS, Eilers & Boelens 2005) — AirSpec stand-in ------------------
def als_baseline(y, lam=1e6, p=0.01, niter=10):
    y = np.asarray(y, float); L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve(W + D, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_correct(X, lam=1e6, p=0.01, niter=10):
    # baseline-subtracted spectra (rows = samples); AirSpec swap-in point
    out = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        out[i] = X[i] - als_baseline(X[i], lam=lam, p=p, niter=niter)
    return out

print("ALS baseline ready (stand-in for AirSpec).")""")

md(r"""## 1. Load the 952-sample IMPROVE calibration spectra

The export carries its wavenumber grid in the column names (everything after the four ID columns
`AnalysisId, FilterId, SampleDate, Site`). Columns are already ordered 3998 → 500 cm⁻¹.""")

code(r"""IDCOLS = ["AnalysisId", "FilterId", "SampleDate", "Site"]
imp = pd.read_csv(IMP_SPECTRA)
wn_cols = [c for c in imp.columns if c not in IDCOLS]
WN = np.array([float(c) for c in wn_cols])                # 3998 -> 500, descending
assert np.all(np.diff(WN) < 0), "IMPROVE wavenumbers not strictly descending"
Ximp = imp[wn_cols].to_numpy(float)
print(f"IMPROVE calibration spectra: {Ximp.shape[0]} samples x {Ximp.shape[1]} wavenumbers "
      f"({WN.max():.0f} -> {WN.min():.0f} cm-1)")
print("sites represented:", imp['Site'].nunique(),
      "| e.g.", ", ".join(imp['Site'].value_counts().head(6).index))""")

md(r"""### What are these 952 samples? (join to DB metadata by AnalysisId)

Quick context on the calibration population — lot numbers and site spread — so we know whether this is
a broad IMPROVE set or a narrower (e.g. biomass) subset. Guarded so a metadata hiccup can't break the
figure.""")

code(r"""try:
    meta = pd.read_csv(IMP_META, usecols=["AnalysisId", "LotNumber", "Site", "SampleDate"])
    info = imp[["AnalysisId"]].merge(meta.drop_duplicates("AnalysisId"), on="AnalysisId", how="left")
    print(f"matched metadata for {info['LotNumber'].notna().sum()}/{len(info)} samples\n")
    print("top lot numbers:\n", info["LotNumber"].value_counts().head(8).to_string())
    yr = pd.to_datetime(info["SampleDate"], errors="coerce").dt.year
    print("\nsample-year range:", int(yr.min()), "->", int(yr.max()))
except Exception as e:
    print("metadata context skipped:", e)""")

md(r"""## 2. Load the Ethiopia (ETAD) spectra — assert identical grid""")

code(r"""spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")],
               key=lambda c: -float(c))                    # descending, same order as WN
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != IMPROVE 952 grid"
Xeth = spec[wcols].to_numpy(float)
print(f"ETAD (Ethiopia) spectra: {Xeth.shape[0]} samples x {Xeth.shape[1]} wavenumbers  "
      f"(grid matches IMPROVE 952: True)")""")

md(r"""## 3. Baseline-correct both sets (ALS)

Pull the sloped baseline (PTFE filter + EC electronic absorption) out of every spectrum so the
functional-group peaks — OH (~3400), CH stretch (~2920/2850), C=O (~1700), etc. — stand up on a flat
floor and the two sets become directly comparable.""")

code(r"""Bimp = baseline_correct(Ximp)
Beth = baseline_correct(Xeth)
print("baselined:", Bimp.shape, "IMPROVE (952) |", Beth.shape, "ETAD")""")

md(r"""## 4. The side-by-side figure — identical format, raw (top) and baselined (bottom)

Left = **IMPROVE 952-sample calibration**, right = **Ethiopia (ETAD)**. Within each row the x-axis
(4000→500 cm⁻¹) and y-axis limits are **shared** so intensities compare at a glance. Thin lines =
individual spectra; black = mean. Grey band = the PTFE (CF) artifact region (~1150–1250) — an artifact,
not analyte, so don't read chemistry into it.""")

code(r"""def overlay(ax, X, color, label, ylim=None):
    for row in X:
        ax.plot(WN, row, color=color, lw=0.25, alpha=0.15)
    ax.plot(WN, X.mean(0), color="black", lw=1.6, label=f"mean (n={len(X)})")
    ax.axvspan(1150, 1250, color="0.5", alpha=0.10)         # PTFE / CF artifact
    ax.set_xlim(WN.max(), WN.min())                          # FTIR convention: high -> low
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

raw_lim  = (min(Ximp.min(), Xeth.min()), max(np.percentile(Ximp, 99.9), np.percentile(Xeth, 99.9)))
base_lim = (min(Bimp.min(), Beth.min()), max(np.percentile(Bimp, 99.9), np.percentile(Beth, 99.9)))

fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
overlay(ax[0, 0], Ximp, "#1f77b4", "IMPROVE 952 calibration — RAW",             raw_lim)
overlay(ax[0, 1], Xeth, "#d62728", "Ethiopia (ETAD) — RAW",                    raw_lim)
overlay(ax[1, 0], Bimp, "#1f77b4", "IMPROVE 952 calibration — BASELINED (ALS)", base_lim)
overlay(ax[1, 1], Beth, "#d62728", "Ethiopia (ETAD) — BASELINED (ALS)",         base_lim)
for a in ax[1, :]: a.set_xlabel("wavenumber (cm$^{-1}$)")
for a in ax[:, 0]: a.set_ylabel("absorbance (a.u.)")
fig.suptitle("Do the 952-sample calibration spectra look like the Ethiopia spectra?  "
             "IMPROVE (blue) vs. ETAD (red) — same axes, same scale", y=0.995, fontsize=13)
plt.tight_layout()
plt.savefig("figures/spectra_side_by_side_952.png", dpi=150, bbox_inches="tight")
print("saved figures/spectra_side_by_side_952.png"); plt.show()""")

md(r"""## 5. Bonus — baselined mean shapes, area-normalized and overlaid

- **IMPROVE 952 vs Ethiopia shape:** normalize each mean spectrum to unit area and overlay — aligned
  peaks = shared chemistry; peaks present in one and not the other = the mismatch that would physically
  justify a smoke-only calibration.
- **Ethiopia internal homogeneity:** overlay every area-normalized ETAD spectrum — collapse together =
  homogeneous (one custom cal could serve all); fan out = heterogeneous.""")

code(r"""def area_norm(X):
    a = np.trapz(np.clip(X, 0, None), WN[::-1], axis=1)[:, None]
    a[a == 0] = np.nan
    return np.clip(X, 0, None) / np.abs(a)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))
axL.plot(WN, area_norm(Bimp).mean(0), color="#1f77b4", lw=1.8, label=f"IMPROVE 952 mean (n={len(Bimp)})")
axL.plot(WN, area_norm(Beth).mean(0), color="#d62728", lw=1.8, label=f"ETAD mean (n={len(Beth)})")
axL.axvspan(1150, 1250, color="0.5", alpha=0.10)
axL.set_xlim(WN.max(), WN.min()); axL.set_xlabel("wavenumber (cm$^{-1}$)")
axL.set_ylabel("area-normalized absorbance"); axL.legend(fontsize=9)
axL.set_title("Mean baselined shape — IMPROVE 952 vs. ETAD", fontsize=11)

Neth = area_norm(Beth)
for row in Neth:
    axR.plot(WN, row, color="#d62728", lw=0.25, alpha=0.15)
axR.plot(WN, Neth.mean(0), color="black", lw=1.5, label="ETAD mean")
axR.axvspan(1150, 1250, color="0.5", alpha=0.10)
axR.set_xlim(WN.max(), WN.min()); axR.set_xlabel("wavenumber (cm$^{-1}$)")
axR.set_ylabel("area-normalized absorbance"); axR.legend(fontsize=9)
axR.set_title(f"ETAD internal homogeneity (all {len(Neth)} normalized)", fontsize=11)

plt.tight_layout()
plt.savefig("figures/spectra_normalized_shapes_952.png", dpi=150, bbox_inches="tight")
print("saved figures/spectra_normalized_shapes_952.png"); plt.show()""")

md(r"""### How to read this / next steps

- **Compare to the 906-sample notebook** (`spectra_side_by_side_improve_vs_ethiopia.ipynb`): the 952
  pull should tell the same story — shared dominant bands, with any IMPROVE-only feature (e.g. the
  ~1700 cm⁻¹ carbonyl shoulder) marking where the calibration population differs from Ethiopia.
- **PTFE caveat:** the shaded ~1150–1250 band is a filter artifact and the tallest baselined feature —
  mask it before judging relative functional-group intensities (easy next step).
- **AirSpec:** replace the body of `baseline_correct()` with Satoshi's AirSpec baseline once it runs
  locally; nothing else changes. AirSpec also gives functional-group quantities from the baselined
  spectra — the natural follow-on.
- **Extend:** add the KBr charcoal spectra as a third panel (IMPROVE vs Ethiopia vs charcoal).""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("spectra_side_by_side_952_improve_vs_ethiopia.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote spectra_side_by_side_952_improve_vs_ethiopia.ipynb")
