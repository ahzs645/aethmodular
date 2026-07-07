"""Builds 05_calibration_variants.ipynb. Run once, then nbconvert --execute.

The calibration-experiment scaffold from the 2026-06-25 meeting. The *heavy* part (training PLS on
the full IMPROVE+smoke spectra with each filtering rule) is **blocked** on: (a) the comments export
from the Shiny app, (b) Sean's smoke/non-smoke classifier, and (c) agreeing the below-1:1 sample
set. So this notebook is a runnable **scaffold**: it fixes the variant list + naming scheme, provides
a deterministic **"first major RMSECV minimum"** component-picker (the meeting's consistency ask),
and lays out the real training harness behind a `RUN_HEAVY` flag pointing at the predecessor's
`ftir_pls_calibration.build_calibration` and the `rds_EC_*` training data.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Calibration variants — scaffold & component-selection rule

**Status: scaffold (partly blocked).** The meeting's direction was to *stop over-filtering* and
build several calibrations to compare. This notebook nails down (1) the **variant list** and
**naming scheme**, (2) a **consistent, deterministic component-selection rule**, and (3) the
**training harness** wired to real code — ready to run once the training data + smoke labels + the
comments export are in hand.

> Blocked on: comments export from the Shiny app · Sean's smoke classifier · the agreed below-1:1
> sample set. See `TASKS.md`.

> **Not a repeat of last week.** Last week's deck already showed the 906-sample calibration, the
> outlier-removal before/after, the jagged-K variance, and *recommended a K-ensemble (15–25)*. This
> notebook does the opposite of last week's cleaning — it fixes the variant list around **keeping**
> samples (`nofilt`, `below11`, `removed`) — and replaces the ensemble pitch with a single
> reproducible component rule plus the **Weakley 2nd-derivative** direction. See
> `notes/whats_new_vs_last_week.md`.""")

md(r"""## 1. The variants to build (naming scheme)

One `calib_id` per variant (full spec in `NAMING_SCHEME.md`). These are the experiments the meeting
asked for — each is a different answer to "which samples does the calibration get to see?".""")

code(r"""import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

variants = pd.DataFrame([
    ["EC_all_absres_lot251_k17",   "all",     "absres (old)", "the 'beautiful but too perfect' baseline being replaced"],
    ["EC_all_nofilt_lot251_kmin",  "all",     "nofilt",       "KEEP EVERYTHING — the primary new baseline"],
    ["EC_biomass_nofilt_lot251_kmin","biomass","nofilt",      "smoke-classifier=yes only — the 'FTIR misses char' test"],
    ["EC_below11_nofilt_lot251_kmin","below11","nofilt",      "train only on samples BELOW the 1:1 Tor line ('weird = the point')"],
    ["EC_removed_nofilt_lot251_kmin","removed","nofilt",      "train only on the previously-removed samples"],
    ["EC_all_ecthr70_lot251_kmin", "all",     "ecthr70",      "drop EC < 70 µg low-signal filters"],
    ["EC_all_ecthrX_lot251_kmin",  "all",     "ecthrX",       "drop below Ethiopia EC range ÷ 10"],
], columns=["calib_id", "trainset", "filter", "purpose"])
print(variants.to_string(index=False))
variants.to_csv("tables/calibration_variants_plan.csv", index=False)
print("\nwrote tables/calibration_variants_plan.csv")""")

md(r"""## 2. Consistent component-selection: first major RMSECV minimum

The meeting rule: pick the **first major minimum** of the RMSECV curve, and — more important — apply
the **same rule everywhere** so the calibrations stay comparable. Below is a deterministic
implementation so "how we pick" is fixed in code, not by eye.

`first_major_min(rmse)` = the smallest `k` that is a local minimum *and* from which no larger `k`
improves RMSECV by more than `rel_tol` (default 2%). That formalizes "first major minimum".

We run it on the **real EC training set** — the 906 samples in the predecessor's `rds_EC_*` export
(the same data behind last week's calibration), cross-validated with the predecessor's own
`_rmsep_by_ncomp`.""")

code(r"""import sys
PRED = Path("../spartan_ec_2026_06_16")
sys.path.insert(0, str(PRED))
from ftir_pls_calibration import _rmsep_by_ncomp

def first_major_min(rmse, rel_tol=0.02):
    '''Return n_components (1-indexed) at the first *major* RMSECV minimum.

    rmse: array-like, rmse[i] = RMSECV for (i+1) components.
    A point k is chosen if it is a local minimum and min(rmse[k:]) is within
    rel_tol*rmse[k] of rmse[k] (i.e. adding components barely helps after k).
    '''
    r = np.asarray(rmse, dtype=float)
    n = len(r)
    for i in range(n):
        left_ok  = (i == 0) or (r[i] <= r[i-1])
        right_ok = (i == n-1) or (r[i] <= r[i+1])
        if left_ok and right_ok:
            if (r[i] - r[i:].min()) <= rel_tol * r[i]:
                return i + 1
    return int(np.argmin(r)) + 1""")

code(r"""# real EC training data (906 samples x 2722 spectral channels)
Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
y = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Vcols = [c for c in Xdf.columns if c != "id"]
Xec = Xdf[Vcols].to_numpy(float)
print("EC training set:", Xec.shape, "| y range", round(y.min(), 2), "-", round(y.max(), 2))

# real cross-validated RMSECV curve (5-fold), as the tool/predecessor computes it
KRANGE = range(1, 31)
rmsecv = _rmsep_by_ncomp(Xec, y, KRANGE, cv=5, seed=0)   # ~15-20 s
ks = np.array(sorted(rmsecv)); rv = np.array([rmsecv[k] for k in ks])

k_first = first_major_min(rv, rel_tol=0.02)     # our consistent rule
k_min   = int(ks[np.argmin(rv)])                # plain global min (what build_calibration uses)
print(f"first major RMSECV min -> {k_first} components (RMSECV {rv[k_first-1]:.2f})")
print(f"global RMSECV min       -> {k_min} components (RMSECV {rv[k_min-1]:.2f})")

pd.DataFrame({"n_components": ks, "RMSECV": rv}).to_csv("tables/ec_rmsecv_curve.csv", index=False)
print("wrote tables/ec_rmsecv_curve.csv")""")

code(r"""fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(ks, rv, "-o", ms=3, color="#1f77b4")
ax.axvline(k_first, color="red", ls="--", label=f"first major min = {k_first}")
ax.axvline(k_min, color="green", ls=":", label=f"global min = {k_min}")
ax.set_xlabel("n PLS components"); ax.set_ylabel("RMSECV (µg, 5-fold)")
ax.set_title(f"Real EC-training RMSECV curve (n={Xec.shape[0]} samples)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig05_ec_rmsecv_curve.png", dpi=140, bbox_inches="tight")
print("saved figures/fig05_ec_rmsecv_curve.png"); plt.show()""")

md(r"""**Note the real curve rises before it dips** — exactly the "started ~40, rose, then dipped"
shape the advisors described. It is also **jagged**: the first deep dip is around **k≈13**
(RMSECV≈17), it bumps back up at k=14, then settles into a broad noisy minimum out to **k=21**
(RMSECV≈16.6). Because k=13 is within ~2% of the k=21 minimum, the **tolerance you set is what decides
13 vs 21** — which is precisely why the rule must be fixed once (`first_major_min`, `rel_tol`) and
applied to **every** variant identically, rather than eyeballed per calibration.

> **Read Weakley (CSN EC paper) before finalizing this.** He uses **second-derivative** spectra and
> needs only **~4 components**, showing the first PLS components mostly model the **Teflon**, not the
> analyte. If we second-derivative-preprocess, this jagged curve may collapse to a few — so the
> *preprocessing* choice and the component rule must be decided together and applied identically.""")

md(r"""## 3. Build the unblocked variants for real, on the same 906-sample data

Some planned variants need **no external labels**, so we build them now on the real EC data. First,
how many of the 906 samples survive each EC threshold — this alone answers whether the meeting's
"≥70 µg" idea is viable:""")

code(r"""THRESHOLDS = [70, 50, 30, 20, 10]
retain = pd.DataFrame({"EC_threshold_ug": THRESHOLDS,
                       "n_retained": [int((y >= t).sum()) for t in THRESHOLDS],
                       "of_total": len(y)})
print(retain.to_string(index=False))
retain.to_csv("tables/ec_threshold_retention.csv", index=False)
print("\n>= 70 µg keeps only", int((y >= 70).sum()),
      "samples — too few for a real calibration. Use a lower cut (or Ethiopia-range ÷ 10).")""")

md(r"""Now build the two variants that have enough samples: **`nofilt`** (all 906 — the meeting's
"stop over-filtering" baseline) and **`ecthr10`** (drop measured EC < 10 µg → 491 samples). Each
picks its components with the **same** `first_major_min` rule, then fits a plain PLS at that `k`.
The label-dependent variants (`biomass`, `below11`, `removed`) stay **blocked** until Sean's smoke
classifier / the below-1:1 set arrive (`TASKS.md`).""")

code(r"""from sklearn.cross_decomposition import PLSRegression

def trainset_mask(name, y):
    '''Boolean mask over the 906 rows for a variant. Only label-free rules run now.'''
    y = np.asarray(y, float)
    if name == "nofilt":   return np.ones(len(y), bool)
    if name.startswith("ecthr"): return y >= float(name[len("ecthr"):])
    raise NotImplementedError(f"'{name}' needs external labels — see TASKS.md")

def build_variant(name, cv=5):
    m = trainset_mask(name, y); n = int(m.sum())
    Xs, ys = Xec[m], y[m]
    kmax = min(30, int(n * (cv - 1) / cv) - 2)     # keep k below the smallest CV-fold size
    rc = _rmsep_by_ncomp(Xs, ys, range(1, kmax + 1), cv=cv, seed=0)
    kk = np.array(sorted(rc)); rr = np.array([rc[k] for k in kk])
    k = first_major_min(rr)
    model = PLSRegression(n_components=k, scale=False).fit(Xs, ys)
    rmse = float(np.sqrt(np.mean((ys - model.predict(Xs).ravel()) ** 2)))
    return {"calib_id": f"EC_all_{name}_lot251_k{k}", "n_train": n,
            "k": k, "RMSECV_at_k": round(float(rr[k - 1]), 2), "insample_RMSE": round(rmse, 2)}

built = pd.DataFrame([build_variant(n) for n in ["nofilt", "ecthr10"]])
print(built.to_string(index=False))
built.to_csv("tables/ec_variants_built.csv", index=False)
print("\nwrote tables/ec_variants_built.csv")
print("blocked (need labels): biomass, below11, removed — see TASKS.md")""")

md(r"""## 4. What each variant feeds into
Once built, apply every calibration to:
- the **5 Adama spectra** → overlay on `02_adama_tor_vs_ftir` (biomass column) and check which move
  toward Tor;
- the **ETAD spectra** (`04`) → EC vs **FABS** (MAC = 10) per season.

Keep the `calib_id` as the column name / legend label throughout so every figure is self-documenting
(`NAMING_SCHEME.md`).""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("05_calibration_variants.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 05_calibration_variants.ipynb")
