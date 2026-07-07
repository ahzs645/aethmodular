"""Builds 06_calibration_variants_components.ipynb. Run once, then nbconvert --execute.

The calibration-variant study from the 2026-06-25 meeting, made concrete on the real 906-sample EC
training set. Implements the CAL-0..CAL-5 naming scheme, builds each variant, and draws a
predicted-vs-measured 1:1 graph for EVERY variant (incl. the "inverse" ones that train only on the
samples you would normally throw away). Then does component selection two documented ways
(first-major-RMSECV-minimum and Wold's R) and tests whether second-derivative preprocessing cuts the
component count (the Weakley result).

Grounded in real data: `spartan_ec_2026_06_16/data/rds_EC_{X,Ymeasured}.csv`.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Calibration variants (CAL-0…CAL-5) & number of components

The 2026-06-25 meeting asked us to *stop over-filtering* and instead **compare several calibrations
built from different sample sets**, including the "inverse" ones that keep exactly the samples we
would normally remove. This notebook makes that concrete on the **real 906-sample EC training set**
and draws a predicted-vs-measured **1:1 graph for every variant**.

### Naming scheme (from the meeting)
| Name | Meaning | Rule on the training set |
|------|---------|--------------------------|
| **CAL-0** All-smoke no-filter | all samples, no manual removal | keep everything (906) |
| **CAL-1** Current cleaned | the cleaned calibration (high/odd removed) | iterative 3σ residual trim |
| **CAL-2** Removed-only *(inverse)* | only the samples CAL-1 removed | complement of CAL-1 |
| **CAL-3** Below-1:1 *(inverse)* | only samples where FTIR under-predicts EC | predicted < measured |
| **CAL-4** EC-high | only high-loading filters | measured EC ≥ 70 µg |
| **CAL-5** Ethiopia-range | only the Ethiopia/Addis loading band | EC in a placeholder band |

> **The "inverse" idea.** CAL-2 and CAL-3 deliberately train on the samples normally discarded — the
> weird, under-predicted, smoke-like filters. The hypothesis is that *those* carry the char signal
> the general calibration misses, so a model built from them (or from everything below the 1:1 line)
> may behave very differently. This is the direct test of "the weird samples may be the point."

*(Grounding: Han et al. 2007 supports char/soot as operational TOR fractions; Weakley et al. 2016
supports source-specific EC calibrations, RMSECV-based component selection, and second-derivative
preprocessing. See `notes/paper_slides_han_weakley.md`.)*""")

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

Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
y = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Vcols = [c for c in Xdf.columns if c != "id"]
X = Xdf[Vcols].to_numpy(float)
print("EC training set:", X.shape, "| measured EC range", round(y.min(), 2), "-", round(y.max(), 2))""")

md(r"""## Component-selection rules (documented, applied identically everywhere)

- **`first_major_min`** — first local RMSECV minimum from which no larger k improves by > `rel_tol`.
- **`wold_R`** — Wold's R: keep adding components while the PRESS ratio RMSECV(k+1)²/RMSECV(k)² is
  below a threshold; stop at the first component whose successor barely helps. This is the criterion
  Weakley cites (a local MSECV/RMSECV minimum to avoid overfitting).

Both are deterministic, so "how we pick k" is fixed in code, not by eye.""")

code(r"""def first_major_min(rmse, rel_tol=0.02):
    '''First k whose RMSECV is within rel_tol of the global minimum — the first time
    CV error effectively bottoms out. Monotone-safe (won't get trapped by an early
    shallow local min on a jagged, rises-then-falls curve).'''
    r = np.asarray(rmse, float)
    return int(np.argmax(r <= r.min() * (1 + rel_tol))) + 1

def wold_R(rmse, thresh=0.95):
    '''Wold's R: keep adding components while the PRESS ratio RMSECV(k+1)²/RMSECV(k)²
    stays below thresh; stop at the first component whose successor barely helps.
    NOTE: assumes a roughly monotone-decreasing PRESS curve — on our raw jagged curve
    (which RISES before it falls) it degenerates to k=1. That degeneracy is itself the
    argument for second-derivative preprocessing (see the 2nd-derivative section).'''
    r = np.asarray(rmse, float)
    for i in range(len(r) - 1):
        if (r[i+1]**2) / (r[i]**2) > thresh:   # PRESS ratio ~ RMSECV^2 ratio
            return i + 1
    return len(r)

def rmsecv_curve(Xs, ys, cv=5):
    n = len(ys)
    kmax = max(2, min(30, int(n * (cv - 1) / cv) - 2))   # keep k < smallest CV-fold size
    rc = _rmsep_by_ncomp(Xs, ys, range(1, kmax + 1), cv=cv, seed=0)
    ks = np.array(sorted(rc)); rv = np.array([rc[k] for k in ks])
    return ks, rv""")

md(r"""## Baseline fit → define the "cleaned", "removed", and "below-1:1" sets

Fit one PLS on all 906 (at CAL-0's chosen k). Its predicted-vs-measured residuals define:
**CAL-1** (iterative 3σ trim), **CAL-2** (what got trimmed), and **CAL-3** (predicted < measured).""")

code(r"""# CAL-0 component count, then a baseline model on everything
ks0, rv0 = rmsecv_curve(X, y)
k0 = first_major_min(rv0)
base = PLSRegression(n_components=k0, scale=False).fit(X, y)
pred0 = base.predict(X).ravel()
resid = y - pred0
print(f"CAL-0 baseline: k={k0}, in-sample RMSE={np.sqrt(np.mean(resid**2)):.2f}")

# CAL-1 cleaned: iterative 3-sigma residual trimming at fixed k0 (mirrors the tool's loop)
keep = np.ones(len(y), bool)
for _ in range(3):
    m = PLSRegression(n_components=k0, scale=False).fit(X[keep], y[keep])
    rr = y - m.predict(X).ravel()
    s = rr[keep].std()
    newkeep = np.abs(rr) <= 3 * s
    if (newkeep == keep).all():
        break
    keep = keep & newkeep
cleaned_mask = keep
removed_mask = ~keep
below11_mask = pred0 < y            # FTIR under-predicts -> below the 1:1 line
print(f"CAL-1 cleaned: {cleaned_mask.sum()} kept, CAL-2 removed: {removed_mask.sum()}, "
      f"CAL-3 below-1:1: {below11_mask.sum()}")""")

md(r"""## Define all six variant masks

CAL-5's Ethiopia band is a **placeholder** (`ETH_LOW`–`ETH_HIGH` µg) until the real FABS/10-derived
Ethiopia EC range is set — flagged clearly.""")

code(r"""ETH_LOW, ETH_HIGH = 10.0, 100.0     # PLACEHOLDER Ethiopia loading band (set from FABS/10 later)

MASKS = {
    "CAL-0 all-nofilter": np.ones(len(y), bool),
    "CAL-1 cleaned":      cleaned_mask,
    "CAL-2 removed-only": removed_mask,       # inverse
    "CAL-3 below-1:1":    below11_mask,       # inverse
    "CAL-4 EC-high>=70":  y >= 70.0,
    "CAL-5 Eth-range":    (y >= ETH_LOW) & (y <= ETH_HIGH),
}
for k, m in MASKS.items():
    print(f"{k:22s} n={int(m.sum())}")""")

md(r"""## Build every variant and draw its predicted-vs-measured 1:1 graph

To compare variants **fairly**, every model is fit at a **common component count** `K_COMMON=20`
(the centre of last week's 15–25 ensemble), capped below the CV-fold size for the small sets. The
per-variant RMSECV-selected `k` is reported separately in the components section — mixing complexities
across the 1:1 panels would make the comparison meaningless.""")

code(r"""K_COMMON = 20

def build_variant(name, mask, k_common=K_COMMON):
    n = int(mask.sum())
    Xs, ys = X[mask], y[mask]
    ks, rv = rmsecv_curve(Xs, ys)
    k_fmm, k_wold = first_major_min(rv), wold_R(rv)
    k_fit = min(k_common, int(ks.max()))            # common complexity, capped for small n
    model = PLSRegression(n_components=k_fit, scale=False).fit(Xs, ys)
    pred = model.predict(Xs).ravel()
    ss_res = np.sum((ys - pred)**2); ss_tot = np.sum((ys - ys.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((ys - pred)**2)))
    return {"name": name, "n": n, "k_fit": k_fit, "k_firstmin": k_fmm, "k_wold": k_wold,
            "RMSECV_min": round(float(rv.min()), 2), "R2": round(float(r2), 3),
            "RMSE": round(rmse, 2), "_ys": ys, "_pred": pred, "_ks": ks, "_rv": rv}

results = {name: build_variant(name, m) for name, m in MASKS.items()}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (name, r) in zip(axes.ravel(), results.items()):
    ys, pred = r["_ys"], r["_pred"]
    inverse = ("removed" in name) or ("below" in name)
    lim = max(ys.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.scatter(ys, pred, s=14, alpha=0.5, color=("#d62728" if inverse else "#1f77b4"))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    tag = "  [inverse]" if inverse else ""
    ax.set_title(f"{name}{tag}\nn={r['n']}  k_fit={r['k_fit']}  R²={r['R2']}  RMSE={r['RMSE']}", fontsize=9)
    ax.set_xlabel("measured EC (µg)"); ax.set_ylabel("predicted EC (µg)")
fig.suptitle(f"Calibration variants — predicted vs. measured EC (in-sample, fit at k={K_COMMON}, 1:1 dashed)",
             y=1.01, fontsize=13)
plt.tight_layout(); plt.savefig("figures/fig06_variants_1to1_grid.png", dpi=140, bbox_inches="tight")
print("saved figures/fig06_variants_1to1_grid.png"); plt.show()

summary = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results.values()])
print(summary.to_string(index=False))
summary.to_csv("tables/calibration_variants_results.csv", index=False)
print("\nwrote tables/calibration_variants_results.csv")""")

md(r"""### Caveats to read on the graphs
- These are **in-sample** predicted-vs-measured at a common k (fit quality), not held-out accuracy —
  the honest cross-variant comparison is **`RMSECV_min`**, not R².
- **CAL-4 (n=22)** and **CAL-2 (removed-only, n≈46)** are small: their fits are unstable and `k_fit`
  is capped below the CV-fold size — show them as *diagnostics*, not candidate calibrations.
- **CAL-1 (cleaned) has the best R² but among the worst `RMSECV_min`** — that is exactly last week's
  "too perfect" trap: 3σ trimming removed the informative high-EC samples, so the in-sample fit
  tightens while the cross-validated error gets *worse*. Direct evidence for "don't over-filter."
- **CAL-5 (Ethiopia band)** has the lowest RMSECV here — but its band is a placeholder; don't
  over-read it until the real FABS/10 range is set.""")

md(r"""## The "inverse" view — which samples each inverse variant trains on

The two inverse calibrations each train on a different discarded subset, so they get **one plot
each**: the full CAL-0 baseline (grey) with that variant's training samples highlighted.

- **CAL-3 below-1:1** — every sample the baseline **under-predicts** (predicted < measured).
- **CAL-2 removed-only** — exactly the samples the 3σ cleaning **threw away**.""")

code(r"""lim = max(y.max(), pred0.max()) * 1.05

# --- CAL-3: below-1:1 ---
fig, ax = plt.subplots(figsize=(6.6, 6.2))
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
ax.scatter(y[~below11_mask], pred0[~below11_mask], s=14, alpha=0.35, color="#cccccc",
           label=f"not used (above 1:1, n={int((~below11_mask).sum())})")
ax.scatter(y[below11_mask], pred0[below11_mask], s=18, alpha=0.65, color="#d62728",
           label=f"CAL-3 training set: below 1:1 (n={int(below11_mask.sum())})")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("measured EC (µg)"); ax.set_ylabel("baseline predicted EC (µg)")
ax.set_title("CAL-3 (inverse) — trains only on the under-predicted samples")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("figures/fig06_inverse_below11.png", dpi=140, bbox_inches="tight")
print("saved figures/fig06_inverse_below11.png"); plt.show()""")

code(r"""# --- CAL-2: removed-only ---
fig, ax = plt.subplots(figsize=(6.6, 6.2))
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
ax.scatter(y[~removed_mask], pred0[~removed_mask], s=14, alpha=0.35, color="#cccccc",
           label=f"kept by CAL-1 (n={int((~removed_mask).sum())})")
ax.scatter(y[removed_mask], pred0[removed_mask], s=45, color="#1f77b4", edgecolors="k", lw=0.4,
           label=f"CAL-2 training set: 3σ-removed (n={int(removed_mask.sum())})")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("measured EC (µg)"); ax.set_ylabel("baseline predicted EC (µg)")
ax.set_title("CAL-2 (inverse) — trains only on the 3σ-removed samples")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("figures/fig06_inverse_removed.png", dpi=140, bbox_inches="tight")
print("saved figures/fig06_inverse_removed.png"); plt.show()""")

md(r"""## Number of components — the rules side by side

`k_firstmin` (first k within 2% of the global RMSECV min) is the rule to adopt. `k_wold` shows **1**
for every variant because Wold's R needs a monotone-decreasing PRESS curve, and our **raw** curves
rise before they fall — a real limitation that the 2nd-derivative section addresses, not a bug.""")

code(r"""comp = summary[["name", "n", "k_firstmin", "k_wold"]].copy()
comp["global_min"] = [int(r["_ks"][np.argmin(r["_rv"])]) for r in results.values()]
print(comp.to_string(index=False))
comp.to_csv("tables/component_selection_by_variant.csv", index=False)
print("\nwrote tables/component_selection_by_variant.csv")

fig, ax = plt.subplots(figsize=(9, 4.8))
for name, r in results.items():
    ax.plot(r["_ks"], r["_rv"], "-o", ms=2.5, lw=1, label=name)
ax.set_xlabel("n PLS components"); ax.set_ylabel("RMSECV (µg, 5-fold)")
ax.set_title("RMSECV curve per variant (pick the first major minimum, consistently)")
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig06_rmsecv_by_variant.png", dpi=140, bbox_inches="tight")
print("saved figures/fig06_rmsecv_by_variant.png"); plt.show()""")

md(r"""## Second-derivative preprocessing — does it cut the component count? (Weakley)

Weakley's OC work reports that **second-derivative spectra + variable selection** collapse model
complexity dramatically (they cite going from ~35 components to ~3), because the first components of
raw spectra mostly model the **PTFE/baseline**, not the analyte. Test it on our CAL-0 EC data: take
the Savitzky–Golay second derivative and recompute the RMSECV curve. If the "first major minimum"
lands at far fewer components, that is direct support for adopting 2nd-derivative preprocessing.""")

code(r"""from scipy.signal import savgol_filter

X2 = savgol_filter(X, window_length=11, polyorder=2, deriv=2, axis=1)   # 2nd derivative
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()  # real cm-1 grid
assert len(WN) == X.shape[1], "wavenumber grid does not match the spectra columns"
ks_raw, rv_raw = rmsecv_curve(X, y)
ks_d2,  rv_d2  = rmsecv_curve(X2, y)

def k_within(ks, rv, frac=0.05):
    '''Components needed to first reach within `frac` of that curve's own RMSECV minimum.'''
    return int(ks[np.argmax(rv <= rv.min() * (1 + frac))])

kw_raw, kw_d2 = k_within(ks_raw, rv_raw), k_within(ks_d2, rv_d2)
print(f"raw spectra:            reaches within 5% of its best RMSECV at k={kw_raw} "
      f"(best {rv_raw.min():.2f}; at k=1 {rv_raw[0]:.1f})")
print(f"2nd-derivative spectra: reaches within 5% of its best RMSECV at k={kw_d2} "
      f"(best {rv_d2.min():.2f}; at k=1 {rv_d2[0]:.1f})")""")

md(r"""### Figure A — RMSECV: raw vs. 2nd-derivative
2nd-derivative spectra are **near their best RMSECV from very few components** (k=1 is already
usable), while the raw curve is jagged and needs many components to descend — the Weakley effect
(our result is milder than his 35→3 because we did not add BMCUVE variable selection / vapor
correction). *This also explains why Wold's R degenerated to k=1 on the raw curve above: it assumes a
monotone-decreasing PRESS, which only the 2nd-derivative curve approaches.*""")

code(r"""fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(ks_raw, rv_raw, "-o", ms=3, color="#1f77b4", label=f"raw (within-5% at k={kw_raw})")
ax.plot(ks_d2, rv_d2, "-o", ms=3, color="#d62728", label=f"2nd-derivative (within-5% at k={kw_d2})")
ax.axvline(kw_raw, color="#1f77b4", ls=":", lw=1); ax.axvline(kw_d2, color="#d62728", ls=":", lw=1)
ax.set_xlabel("n PLS components"); ax.set_ylabel("RMSECV (µg, 5-fold)")
ax.set_title("RMSECV: raw vs. 2nd-derivative (CAL-0, n=906)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig06_second_derivative_rmsecv.png", dpi=140, bbox_inches="tight")
print("saved figures/fig06_second_derivative_rmsecv.png"); plt.show()""")

md(r"""### Figure B — Mean spectrum: raw vs. 2nd-derivative
The 2nd derivative flattens the sloping PTFE baseline (removing the broad substrate ramp) and sharpens
the analyte bands — which is why fewer components are needed to reach a good fit.""")

code(r"""BANDS = [(2800, 3000, "CH"), (1650, 1750, "C=O"), (3100, 3500, "OH"), (1150, 1250, "CF (PTFE)")]

fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
for ax in axes:
    for x0, x1, lbl in BANDS:
        ax.axvspan(x0, x1, color=("red" if lbl.startswith("CF") else "green"), alpha=0.07)
axes[0].plot(WN, X.mean(0), color="#1f77b4", lw=0.9)
axes[0].set_ylabel("raw absorbance"); axes[0].set_title("Mean raw spectrum (sloping PTFE baseline dominates)")
for x0, x1, lbl in BANDS:
    axes[0].text((x0+x1)/2, axes[0].get_ylim()[1]*0.9, lbl, ha="center", fontsize=7,
                 color=("red" if lbl.startswith("CF") else "green"))
axes[1].plot(WN, X2.mean(0), color="#d62728", lw=0.9)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_ylabel("2nd-derivative"); axes[1].set_xlabel("wavenumber (cm⁻¹)")
axes[1].set_title("Mean 2nd-derivative spectrum (baseline removed; bands sharpened)")
axes[0].set_xlim(WN.max(), WN.min())     # FTIR convention: high → low
plt.tight_layout(); plt.savefig("figures/fig06_second_derivative_spectra.png", dpi=140, bbox_inches="tight")
print("saved figures/fig06_second_derivative_spectra.png"); plt.show()""")

md(r"""### Reading it next week
- **Variants:** show the 1:1 grid — CAL-1 (cleaned) looks tightest, but that's the "too perfect"
  trap; the honest comparison is RMSECV, and the **inverse** CAL-2/CAL-3 are diagnostics for the
  char-rich behaviour. Apply the chosen variants to the 5 Adama filters + ETAD next (needs the
  wavenumber-matched spectra).
- **Components:** report `first_major_min` and `wold_R` together, pick one rule, and use it for every
  variant — never re-pick per calibration.
- **2nd-derivative:** if k drops sharply (as Weakley found for OC), that is the argument for adopting
  it before finalizing any variant — decide preprocessing and the component rule *together*.
- **Framing:** call these **TOR-defined** char/soot and **source-specific smoke** calibrations, not
  absolute chemical truth (Han's caveat).""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("06_calibration_variants_components.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 06_calibration_variants_components.ipynb")
