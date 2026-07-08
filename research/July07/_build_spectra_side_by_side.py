"""Builds spectra_side_by_side_improve_vs_ethiopia.ipynb. Run once, then nbconvert --execute.

Ann's action item (2026-07-07 meeting): plot the **IMPROVE calibration spectra** side-by-side with the
**Ethiopia (ETAD/Addis) spectra** in the *same format* to see how similar they actually look — from
first principles (chemistry/spectroscopy), not just statistically (R²/slope). If the calibration set
does not spectrally resemble the prediction set, that's a *physical* reason a general calibration would
fail on Ethiopia — motivating a smoke-only calibration.

Two things to see:
  1. Do the peak POSITIONS, relative INTENSITIES, and overall SHAPES match between the two sets?
  2. (bonus) Is the Ethiopia set internally homogeneous? (Ann eyeballed the raw spectra as all looking
     alike — normalize to unit area and check whether they collapse onto each other.)

Baselining ("so the peaks pop"): the raw spectra have a sloped baseline (a property of the PTFE filter
+ EC's broad electronic absorption above ~3500), which buries the functional-group peaks and makes the
two sets hard to compare by eye. Ann ruled out 2nd derivatives for THIS purpose (they wash out the
broad OH band and the EC slope, which is signal we want) and asked for **baseline correction** instead.
Satoshi Takahama's **AirSpec** has the exact routine; until it's running locally we use **asymmetric
least squares (ALS, Eilers & Boelens 2005)** — a standard, self-contained baseline that produces the
same visual effect (pulls the slope down, peaks pop). Swap in AirSpec's routine after the Satoshi
meeting; the figure code doesn't change, only `baseline_correct()`.

Both sets sit on the identical 2722-point wavenumber grid (3998->500 cm-1), so side-by-side needs no
interpolation — same axes, same scaling, like-for-like.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# IMPROVE calibration spectra vs. Ethiopia (ETAD) spectra — side by side, baselined

**Why (2026-07-07 meeting with Ann).** The hypothesis is that the standard/general calibration doesn't
suit the Ethiopia samples because they're spectrally *atypical* (cf. Weakley — atypical spectra may
need their own calibration). So far that's been argued **statistically** (R², slopes, intercepts). Ann
wants it grounded in **first principles**: do the IMPROVE samples we calibrate with actually *look*
like the Ethiopia samples we predict? If the calibration set doesn't resemble the prediction set,
that's a **physical** reason a general calibration fails on Ethiopia — not just an artifact of which
samples got removed. This feeds directly into whether a **smoke-only calibration is justified**.

**What this notebook does.**
1. Load the **IMPROVE calibration spectra** (the 906-sample EC training set = the "regular all-data"
   calibration) and the **319 ETAD (Addis/Ethiopia)** spectra.
2. **Baseline-correct** both (ALS, standing in for AirSpec) so the sloped filter/EC baseline is pulled
   down and the functional-group peaks pop.
3. Plot each set **overlaid in identical format** (same axes, same scaling), **side by side** — raw on
   top, baselined below — so peak positions / intensities / shapes can be compared by eye.
4. **Bonus:** area-normalize and overlay the mean shapes to test (a) how different IMPROVE vs Ethiopia
   really are, and (b) whether the Ethiopia set is internally homogeneous (Ann's observation).

> **On the baseline method.** Satoshi's **AirSpec** has the baseline + functional-group code; this
> notebook uses **asymmetric least squares (ALS)** as a drop-in until AirSpec runs locally (Ann is
> setting up a meeting with Satoshi). ALS gives the same visual result Ann's after. To switch, replace
> only the body of `baseline_correct()` — everything downstream is unchanged.""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "0.92", "figure.dpi": 110})

PRED = Path("../spartan_ec_2026_06_16")           # IMPROVE 906-sample training set lives here
ETAD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                      "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)
print("IMPROVE training dir:", PRED.exists(), "| ETAD dir:", ETAD.exists())


# --- baseline correction -----------------------------------------------------------------
# ALS (Eilers & Boelens 2005): a smooth, asymmetric-weighted baseline that hugs the underside of the
# spectrum. lam = smoothness (bigger -> stiffer baseline); p = asymmetry (small -> baseline stays below
# peaks). This is the stand-in for Satoshi's AirSpec baseline; swap the body to use AirSpec's routine.
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
    # Return baseline-subtracted spectra (rows = samples). AirSpec swap-in point.
    out = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        out[i] = X[i] - als_baseline(X[i], lam=lam, p=p, niter=niter)
    return out

print("ALS baseline ready (stand-in for AirSpec).")""")

md(r"""## 1. Load the IMPROVE calibration spectra (906-sample EC training set)

This is the **"regular all-data" calibration** set — the IMPROVE-based 906-sample matrix the general
PLS EC model is fit on. Columns are already in wavenumber order matching the coefficient grid.""")

code(r"""Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
WN  = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy(float)  # 3998 -> 500, desc
Vcols = [c for c in Xdf.columns if c != "id"]
Ximp = Xdf[Vcols].to_numpy(float)
assert Ximp.shape[1] == len(WN), "IMPROVE grid mismatch"
print(f"IMPROVE calibration spectra: {Ximp.shape[0]} samples x {Ximp.shape[1]} wavenumbers "
      f"({WN.max():.0f} -> {WN.min():.0f} cm-1)")""")

md(r"""## 2. Load the Ethiopia (ETAD) spectra — same grid

The ETAD export sits on the **identical** 2722-point grid, so no interpolation is needed. We assert the
grids match before overlaying anything.""")

code(r"""spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")],
               key=lambda c: -float(c))                     # descending = same order as WN
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != IMPROVE grid"
Xeth = spec[wcols].to_numpy(float)
print(f"ETAD (Ethiopia) spectra: {Xeth.shape[0]} samples x {Xeth.shape[1]} wavenumbers  "
      f"(grid matches IMPROVE: {np.allclose([float(c) for c in wcols], WN)})")""")

md(r"""## 3. Baseline-correct both sets

Pull the sloped baseline (PTFE filter + EC electronic absorption) out of every spectrum so the
functional-group peaks — OH (~3400), CH stretch (~2920/2850), C=O (~1700), etc. — stand up on a flat
floor and the two sets become directly comparable.""")

code(r"""Bimp = baseline_correct(Ximp)
Beth = baseline_correct(Xeth)
print("baselined:", Bimp.shape, "IMPROVE |", Beth.shape, "ETAD")""")

md(r"""## 4. The side-by-side figure — identical format, raw (top) and baselined (bottom)

Left column = **IMPROVE calibration**, right column = **Ethiopia (ETAD)**. Within each row the x-axis
(4000->500 cm-1) and y-axis limits are **shared** so intensities are comparable at a glance. Thin lines
= individual spectra; black = mean spectrum. Grey band = the PTFE (CF) artifact region.""")

code(r"""def overlay(ax, X, color, label, ylim=None):
    for row in X:
        ax.plot(WN, row, color=color, lw=0.25, alpha=0.15)
    ax.plot(WN, X.mean(0), color="black", lw=1.6, label=f"mean (n={len(X)})")
    ax.axvspan(1150, 1250, color="0.5", alpha=0.10)          # PTFE / CF artifact
    ax.set_xlim(WN.max(), WN.min())                           # FTIR convention: high -> low
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

# shared y-limits per row so the two sides are on the same scale
raw_lim  = (min(Ximp.min(), Xeth.min()), max(np.percentile(Ximp, 99.9), np.percentile(Xeth, 99.9)))
base_lim = (min(Bimp.min(), Beth.min()), max(np.percentile(Bimp, 99.9), np.percentile(Beth, 99.9)))

fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
overlay(ax[0, 0], Ximp, "#1f77b4", "IMPROVE calibration — RAW",            raw_lim)
overlay(ax[0, 1], Xeth, "#d62728", "Ethiopia (ETAD) — RAW",               raw_lim)
overlay(ax[1, 0], Bimp, "#1f77b4", "IMPROVE calibration — BASELINED (ALS)", base_lim)
overlay(ax[1, 1], Beth, "#d62728", "Ethiopia (ETAD) — BASELINED (ALS)",     base_lim)
for a in ax[1, :]: a.set_xlabel("wavenumber (cm$^{-1}$)")
for a in ax[:, 0]: a.set_ylabel("absorbance (a.u.)")
fig.suptitle("Do the calibration spectra look like the Ethiopia spectra?  "
             "IMPROVE (blue) vs. ETAD (red) — same axes, same scale", y=0.995, fontsize=13)
plt.tight_layout()
plt.savefig("figures/spectra_side_by_side.png", dpi=150, bbox_inches="tight")
print("saved figures/spectra_side_by_side.png"); plt.show()""")

md(r"""## 5. Bonus — baselined mean shapes, area-normalized and overlaid

Two questions in one plot:
- **IMPROVE vs Ethiopia shape:** normalize each mean spectrum to unit area and overlay them. Peaks that
  line up = shared chemistry; peaks present in one and not the other = the spectral mismatch that would
  physically justify a smoke-only calibration.
- **Ethiopia internal homogeneity (Ann's note):** overlay every *area-normalized* ETAD spectrum. If
  they collapse onto each other, the Ethiopia set is homogeneous and a single custom calibration could
  serve all of it; if they fan out, it isn't.""")

code(r"""def area_norm(X):
    a = np.trapz(np.clip(X, 0, None), WN[::-1], axis=1)[:, None]   # positive area over the grid
    a[a == 0] = np.nan
    return np.clip(X, 0, None) / np.abs(a)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))

# left: mean shapes overlaid
axL.plot(WN, area_norm(Bimp).mean(0), color="#1f77b4", lw=1.8, label=f"IMPROVE mean (n={len(Bimp)})")
axL.plot(WN, area_norm(Beth).mean(0), color="#d62728", lw=1.8, label=f"ETAD mean (n={len(Beth)})")
axL.axvspan(1150, 1250, color="0.5", alpha=0.10)
axL.set_xlim(WN.max(), WN.min()); axL.set_xlabel("wavenumber (cm$^{-1}$)")
axL.set_ylabel("area-normalized absorbance"); axL.legend(fontsize=9)
axL.set_title("Mean baselined shape — IMPROVE vs. ETAD", fontsize=11)

# right: every ETAD spectrum normalized -> internal homogeneity
Neth = area_norm(Beth)
for row in Neth:
    axR.plot(WN, row, color="#d62728", lw=0.25, alpha=0.15)
axR.plot(WN, Neth.mean(0), color="black", lw=1.5, label="ETAD mean")
axR.axvspan(1150, 1250, color="0.5", alpha=0.10)
axR.set_xlim(WN.max(), WN.min()); axR.set_xlabel("wavenumber (cm$^{-1}$)")
axR.set_ylabel("area-normalized absorbance"); axR.legend(fontsize=9)
axR.set_title(f"ETAD internal homogeneity (all {len(Neth)} normalized)", fontsize=11)

plt.tight_layout()
plt.savefig("figures/spectra_normalized_shapes.png", dpi=150, bbox_inches="tight")
print("saved figures/spectra_normalized_shapes.png"); plt.show()""")

md(r"""### How to read this / next steps

- **The comparison to make (Ann's ask):** eyeball the two baselined panels. Do the peak **positions**
  line up? Are the relative **intensities** similar? Does Ethiopia have features the IMPROVE
  calibration set **lacks** (or vice versa)? Any systematic mismatch is a *physical* argument that the
  general calibration is being applied outside its spectral domain — motivating a smoke-only cal.
- **Homogeneity:** if the normalized ETAD spectra collapse together, one custom calibration could serve
  the whole Ethiopia set (supports the smoke-only plan); if they fan out, it's heterogeneous.
- **AirSpec:** replace the body of `baseline_correct()` with Satoshi's AirSpec baseline once it runs
  locally (Ann is arranging the Satoshi meeting) and re-run — nothing else changes. AirSpec also
  computes functional groups from the baselined spectra, the natural follow-on.
- **Extend:** add the **KBr charcoal** spectra as a third panel (Ann's longer-term plan: IMPROVE vs
  Ethiopia vs charcoal) to see whether the charcoal samples carry features absent from the calibration
  set.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("spectra_side_by_side_improve_vs_ethiopia.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote spectra_side_by_side_improve_vs_ethiopia.ipynb")
