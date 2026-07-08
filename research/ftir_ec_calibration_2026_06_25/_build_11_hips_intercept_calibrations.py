"""Builds 11_hips_intercept_calibrations.ipynb. Run once, then nbconvert --execute.

Goal: find a calibration whose ETAD New-EC-vs-HIPS regression passes closer to (0,0) than the
notebook-08 baseline (CAL-0 k=20: slope 2.27, intercept -5.25). Sweeps two levers — component
count and the TRAINING EC RANGE — because ETAD's HIPS EC is a narrow low band (~3-7 µg/m³)
while the full IMPROVE set is anchored by a few extreme >150 µg smoke filters that force the
steep slope and the big negative intercept.

Honest framing kept front-and-centre:
  - Intercept alone is the WRONG target: a near-flat calibration (CAL-0 k=2) gives intercept≈0
    with slope 0.2 — meaningless. The real goal is slope≈1 AND intercept≈0 (agreement).
  - This tunes toward HIPS, which is NOT ground truth (HIPS EC = Fabs/MAC, MAC assumed). The
    calibration is trained on TOR EC.
  - The intercept is MAC-INDEPENDENT (scaling Fabs/MAC moves the slope, not the x=0 crossing).

Data: training spartan_ec_2026_06_16/data/rds_EC_*; ETAD spectra + SPARTAN HIPS on Google Drive
(same join as notebook 08). Pipeline validated to reproduce nb08's CAL-0 fit exactly.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Calibrations that bring the ETAD New-EC-vs-HIPS line toward (0,0)

Notebook 08's ETAD New-EC-vs-HIPS fit is `y = 2.27x − 5.25` for the CAL-0 keep-everything
calibration — a **steep slope and a −5.25 µg/m³ intercept**. This notebook asks: *which
calibration brings that line closer to the origin?*

**Why the intercept is big:** ETAD's HIPS EC is a narrow low band (~3–7 µg/m³), but the full
IMPROVE calibration is anchored by a handful of extreme >150 µg smoke filters that set a steep
slope. Applied to ETAD's low range, that steepness is what produces the large negative intercept.

**Read these caveats before trusting any low-intercept number:**
1. **Intercept alone is the wrong target.** A flat calibration (e.g. CAL-0 at k=2) trivially gives
   intercept ≈ 0 with slope ≈ 0.2 and R² ≈ 0.2 — a line through the origin that says nothing. The
   meaningful goal is **slope ≈ 1 *and* intercept ≈ 0**.
2. **This tunes toward HIPS, which is not ground truth.** The calibration is trained on TOR EC;
   HIPS EC = Fabs/MAC carries an assumed MAC.
3. **The intercept is MAC-independent** — Fabs/6 vs Fabs/10 changes the slope, not where the line
   crosses zero. Only the calibration's low-loading behaviour moves the intercept.""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "0.9"})
PRED = Path("../spartan_ec_2026_06_16")
GD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                    "/Research/Grad/UC Davis Ann/NASA MAIA/Data")
ETAD = GD / "DAVIS/ETAD FTIR"
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

# --- training EC set + wavenumber grid ---
Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
y = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Xtr = Xdf[[c for c in Xdf.columns if c != "id"]].to_numpy(float)
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()

# --- ETAD spectra (same grid) + volumes ---
spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
meta = pd.read_csv(ETAD / "ETAD_metadata.csv")
wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")], key=lambda c: -float(c))
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != training grid"
Xeth = spec[wcols].to_numpy(float)
media = spec["MediaId"].to_numpy()
vol = pd.Series(media).map(meta.set_index("MediaId")["SampleVolume_m3"]).to_numpy()
vol_ok = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)

# --- SPARTAN HIPS Fabs for ETAD ---
hips = pd.read_csv(GD / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv", usecols=["Site", "FilterId", "Fabs"])
hips = hips[hips["Site"] == "ETAD"].rename(columns={"FilterId": "ExternalFilterId"})
media_ext = meta[["MediaId", "ExternalFilterId"]]
print("training:", Xtr.shape, "| ETAD:", Xeth.shape, "| ETAD HIPS filters:", int(hips["Fabs"].notna().sum()))""")

md(r"""## Helpers: fit a calibration, predict ETAD EC, regress vs HIPS (AGENTS.md style)""")

code(r"""def reg_stats(x, y_):
    x = np.asarray(x, float); y_ = np.asarray(y_, float)
    m = np.isfinite(x) & np.isfinite(y_); x, y_ = x[m], y_[m]
    slope, intercept = np.polyfit(x, y_, 1)
    yh = slope * x + intercept
    r2 = 1 - np.sum((y_ - yh)**2) / np.sum((y_ - y_.mean())**2)
    return dict(n=int(len(x)), slope=float(slope), intercept=float(intercept), r2=float(r2))

def etad_ec(mask, k, Xt=Xtr, Xe=Xeth):
    mdl = PLSRegression(n_components=k, scale=False).fit(Xt[mask], y[mask])
    return mdl.predict(Xe).ravel() / vol_ok           # µg/m³ per ETAD filter

def hips_fit(conc):
    d = (pd.DataFrame({"MediaId": media, "newec": conc})
         .merge(media_ext, on="MediaId").merge(hips, on="ExternalFilterId"))
    d["EC_hips"] = d["Fabs"] / 10.0
    d = d[np.isfinite(d["EC_hips"]) & np.isfinite(d["newec"])]
    st = reg_stats(d["EC_hips"], d["newec"]); st["_x"] = d["EC_hips"].to_numpy(); st["_y"] = d["newec"].to_numpy()
    return st

# validate the pipeline reproduces notebook 08's CAL-0 fit
v = hips_fit(etad_ec(np.ones(len(y), bool), 20))
print(f"CAL-0 k=20 check: slope={v['slope']:.3f} intercept={v['intercept']:.3f} "
      f"r2={v['r2']:.3f} n={v['n']}  (nb08: 2.265 / -5.253 / 0.657)")
ec_hips = v["_x"]
print(f"ETAD HIPS EC band: median {np.median(ec_hips):.2f}, 5-95% "
      f"{np.percentile(ec_hips,5):.2f}-{np.percentile(ec_hips,95):.2f} µg/m³ (a narrow low range)")""")

md(r"""## Sweep — component count and training EC range

Fewer components flatten the model; restricting the training EC range removes the extreme
high-EC leverage points. Both pull the ETAD intercept toward 0 — the question is which keeps a
**usable slope** (≈1) rather than collapsing to a flat line.""")

code(r"""allm = np.ones(len(y), bool)
CANDS = [
    ("CAL-0 all",        "k=20 (current)", allm,       20),
    ("CAL-0 all",        "k=15",           allm,       15),
    ("CAL-0 all",        "k=10",           allm,       10),
    ("CAL-0 all",        "k=7",            allm,       7),
    ("CAL-0 all",        "k=5",            allm,       5),
    ("CAL-0 all",        "k=3",            allm,       3),
    ("CAL-0 all",        "k=2",            allm,       2),
    ("CAL-7 no-extremes","k=20",           y <= 250,   20),
    ("train EC<=100",    "k=10",           y <= 100,   10),
    ("train EC<=50",     "k=15",           y <= 50,    15),
    ("train EC<=50",     "k=10",           y <= 50,    10),
    ("train EC<=50",     "k=5",            y <= 50,    5),
    ("train EC<=30",     "k=10",           y <= 30,    10),
    ("train EC<=30",     "k=5",            y <= 30,    5),
]
rows = []
for name, setting, mask, k in CANDS:
    st = hips_fit(etad_ec(mask, k))
    flag = "flat/degenerate" if st["slope"] < 0.5 else ("near-1:1" if 0.75 <= st["slope"] <= 1.3 else "")
    rows.append({"calibration": name, "setting": setting, "n_train": int(mask.sum()),
                 "slope": round(st["slope"], 3), "intercept": round(st["intercept"], 3),
                 "r2": round(st["r2"], 3), "flag": flag})
sweep = pd.DataFrame(rows)
sweep["abs_int"] = sweep["intercept"].abs()
sweep = sweep.sort_values("abs_int").drop(columns="abs_int").reset_index(drop=True)
sweep.to_csv("tables/hips_intercept_sweep.csv", index=False)
print(sweep.to_string(index=False))
print("\nwrote tables/hips_intercept_sweep.csv")""")

md(r"""## Before / after — the crossplot

Three calibrations: the **current** CAL-0 (steep, −5.25); the **range-matched** EC≤50/k=10 that
lands near 1:1 with a much smaller intercept; and the **smallest-real-intercept** EC≤30/k=5
(near the origin but with a slope well below 1 — the residual under-agreement).""")

code(r"""picks = [("CAL-0 all  k=20  (current)", allm, 20, "#1f77b4"),
         ("train EC<=50  k=10  (near 1:1)", y <= 50, 10, "#2ca02c"),
         ("train EC<=30  k=5  (min |intercept|)", y <= 30, 5, "#d62728")]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (title, mask, k, col) in zip(axes, picks):
    st = hips_fit(etad_ec(mask, k))
    x, yv = st["_x"], st["_y"]
    hi = max(x.max(), yv.max()) * 1.1
    lo = min(0, yv.min()) * 1.1
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1)                       # 1:1
    ax.scatter(x, yv, s=45, alpha=0.55, color=col, edgecolors="k", linewidths=0.3)
    xs = np.array([0, hi])
    ax.plot(xs, st["slope"] * xs + st["intercept"], color=col, lw=1.9)       # regression
    sign = "+" if st["intercept"] >= 0 else "−"
    ax.text(0.03, 0.97, f"y = {st['slope']:.2f}x {sign} {abs(st['intercept']):.2f}\n"
                        f"R² = {st['r2']:.3f}   n = {st['n']}", transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.scatter([0], [0], marker="*", s=180, color="black", zorder=5)         # the (0,0) target
    ax.set_xlim(0, hi); ax.set_ylim(lo, hi); ax.set_title(title, fontsize=10)
    ax.set_xlabel("HIPS EC = Fabs/10 (µg/m³)"); ax.set_ylabel("New FTIR-EC (µg/m³)")
fig.suptitle("ETAD New-EC vs HIPS — steep IMPROVE cal (−5.25)  →  range-matched cal (~−1.3)", y=1.02, fontsize=13)
plt.tight_layout(); plt.savefig("figures/fig11_hips_intercept_comparison.png", dpi=140, bbox_inches="tight")
print("saved figures/fig11_hips_intercept_comparison.png"); plt.show()""")

md(r"""## Does a baseline offset explain the remaining −1.3? (exploratory)

The range-matched calibration still crosses at ≈ −1.3. ETAD spectra are transmission and carry a
non-zero baseline (the "absorbance at 4000 cm⁻¹" Teflon artefact from notebook 01). Subtract, per
spectrum, the mean absorbance in the non-absorbing **3900–4000 cm⁻¹** window from BOTH training and
ETAD, then re-fit — if the intercept closes, the offset was a baseline mismatch, not chemistry.""")

code(r"""bwin = WN >= 3900                      # non-absorbing high-wavenumber window
def offset_correct(M): return M - M[:, bwin].mean(axis=1, keepdims=True)
Xtr_bc, Xeth_bc = offset_correct(Xtr), offset_correct(Xeth)

for label, mask, k in [("EC<=50 k=10", y <= 50, 10), ("CAL-0 k=20", allm, 20)]:
    raw = hips_fit(etad_ec(mask, k))
    bc  = hips_fit(etad_ec(mask, k, Xt=Xtr_bc, Xe=Xeth_bc))
    print(f"{label:14s}  raw: slope={raw['slope']:.2f} intercept={raw['intercept']:+.2f} r2={raw['r2']:.2f}"
          f"   | baseline-corrected: slope={bc['slope']:.2f} intercept={bc['intercept']:+.2f} r2={bc['r2']:.2f}")""")

md(r"""### Reading it
- **The lever that works is training range, not just k.** The full IMPROVE calibration is steep
  because a few >150 µg smoke filters anchor it; ETAD lives at 3–7 µg/m³, so a calibration trained
  on the **matched low-EC range (≤50 µg)** applies without that extrapolation and lands near 1:1
  with a much smaller intercept (~−1.3 vs −5.25).
- **Don't chase the intercept to zero blindly** — EC≤30/k=5 gets the intercept smallest but at
  slope ≈ 0.4, i.e. it stops tracking HIPS. `EC≤50, k=10` (slope ≈ 0.9, intercept ≈ −1.3) is the
  honest sweet spot.
- **Whatever the baseline-correction cell shows** is the tell on the last −1.3: if it closes, the
  offset was spectral baseline; if not, it is a real method/MAC difference and not something to
  calibrate away.
- Still trained on TOR, still compared to a MAC-10 HIPS reference — treat this as *agreement
  tuning*, not a new ground-truth calibration.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("11_hips_intercept_calibrations.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 11_hips_intercept_calibrations.ipynb")
