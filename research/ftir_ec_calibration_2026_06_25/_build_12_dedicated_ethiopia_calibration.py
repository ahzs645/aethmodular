"""Builds 12_dedicated_ethiopia_calibration.ipynb. Run once, then nbconvert --execute.

Works through the three "make Ethiopia work better" ideas on the NEW local IMPROVE mirror
(~/My Drive/FTIR/local_db) — 13k+ spectra across lots 248+251 (the ETAD lots, spanning BOTH),
matched to TOR EC by Site+SampleDate, validated to the app export. This replaces the narrow
906-sample rds_EC set used in notebooks 06-11.

  Idea 1  dedicated / spectrally-matched calibration (Weakley "atypical->dedicated"; Ann's PCA):
          (a) the lot-248+251 full set is ALREADY a dedicated set; (b) PCA Addis-neighborhood.
  Idea 3  Weakley pipeline: 2nd-derivative + biomass-band selection (drop the Teflon/CF region).

Honest headline (verified in-session): the DATA upgrade — lot-248+251, EC-representative — is the
real lever (intercept -5.25 -> -2.3, slope 2.27 -> 1.2). PCA neighborhood is marginal on top.
Ethiopia has no TOR truth, so "better" = HIPS agreement + physical sensibility (no negative EC).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# A dedicated Ethiopia (ETAD) calibration — from the local lot-248+251 mirror

Notebooks 06-11 trained on the narrow 906-sample `rds_EC` set and got a steep, offset ETAD-vs-HIPS
fit (`y = 2.27x − 5.25`). Here we use the **new local IMPROVE mirror** (`~/My Drive/FTIR/local_db`):
**13k+ spectra across lots 248 + 251** — the two lots ETAD actually spans — matched to TOR EC and
validated to the app export. Then we work the three ideas:

1. **Dedicated / spectrally-matched calibration** (Weakley: *atypical sites deserve their own EC
   calibration*; Ann: pull in lot 248 by PCA-similarity). Two forms: (a) the lot-248+251 set is
   itself a dedicated set; (b) a PCA Addis-neighborhood subset.
2. **Weakley pipeline** — 2nd-derivative + biomass-band selection (drop the Teflon/CF region that
   drives the baseline offset).

**Judged by** HIPS agreement (does `y=mx+b` approach 1:1 through the origin?) and physical
sensibility (no negative EC), because **Ethiopia has no TOR ground truth**.""")

code(r"""from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "0.9"})
LOCAL = Path.home() / "Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/FTIR/local_db"
GD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                    "/Research/Grad/UC Davis Ann/NASA MAIA/Data")
PRED = Path("../spartan_ec_2026_06_16")
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

# --- IMPROVE lot 248+251 EC set from the local mirror (X, y) ---
sys.path.insert(0, str(LOCAL)); import local_calib
df, wn = local_calib.assemble("EC")
df = df[df["y"] > 0].reset_index(drop=True)                 # drop measured-negative EC (app rule)
Ximp = df[wn].to_numpy(np.float32); yimp = df["y"].to_numpy(float)
WN = np.array([float(c) for c in wn])
print(f"local_db EC set: {Ximp.shape}  EC {yimp.min():.1f}-{yimp.max():.0f} µg (median {np.median(yimp):.1f})")""")

md(r"""## ETAD spectra + HIPS reference (same join as notebook 08/11)""")

code(r"""ET = GD / "DAVIS/ETAD FTIR"
spec = pd.read_csv(ET / "ETAD_FTIR_spectra.csv"); meta = pd.read_csv(ET / "ETAD_metadata.csv")
wc = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")], key=lambda c: -float(c))
assert np.allclose([float(c) for c in wc], WN), "ETAD grid != local_db grid"
Xeth = spec[wc].to_numpy(np.float32); media = spec["MediaId"].to_numpy()
vol = pd.Series(media).map(meta.set_index("MediaId")["SampleVolume_m3"]).to_numpy()
volok = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
hips = pd.read_csv(GD / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv", usecols=["Site", "FilterId", "Fabs"])
hips = hips[hips["Site"] == "ETAD"].rename(columns={"FilterId": "ExternalFilterId"})
m2 = meta[["MediaId", "ExternalFilterId"]]

def hips_fit(load):
    d = (pd.DataFrame({"MediaId": media, "c": load / volok})
         .merge(m2, on="MediaId").merge(hips, on="ExternalFilterId"))
    d["ec"] = d["Fabs"] / 10.0; d = d[np.isfinite(d["ec"]) & np.isfinite(d["c"])]
    s, b = np.polyfit(d["ec"], d["c"], 1); yh = s * d["ec"] + b
    r2 = 1 - np.sum((d["c"] - yh)**2) / np.sum((d["c"] - d["c"].mean())**2)
    return dict(slope=float(s), intercept=float(b), r2=float(r2), n=int(len(d)),
                _x=d["ec"].to_numpy(), _y=d["c"].to_numpy(), pct_neg=float(100*(d["c"] < 0).mean()))

def fit_hips(Xt, yt, Xe, k):
    m = PLSRegression(k, scale=False).fit(Xt.astype(float), yt)
    return hips_fit(m.predict(Xe.astype(float)).ravel())
print("ETAD:", Xeth.shape, "| HIPS-joined filters:", hips_fit(np.zeros(len(media)))["n"])""")

md(r"""## Baseline: the current (rds_EC 906) calibration, for reference""")

code(r"""Xr = pd.read_csv(PRED / "data/rds_EC_X.csv"); Xr = Xr[[c for c in Xr.columns if c != "id"]].to_numpy(float)
yr = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
b0 = fit_hips(Xr, yr, Xeth, 20)
print(f"rds_EC 906  k=20 (current):  slope={b0['slope']:.2f} intercept={b0['intercept']:+.2f} "
      f"r2={b0['r2']:.2f}  neg-EC={b0['pct_neg']:.0f}%")""")

md(r"""## Idea 1a — the dedicated lot-248+251 set (the data upgrade)""")

code(r"""allm = np.ones(len(yimp), bool)
for k in [10, 15, 20]:
    f = fit_hips(Ximp[allm], yimp[allm], Xeth, k)
    print(f"local_db 248+251 (n={allm.sum()})  k={k}:  slope={f['slope']:.2f} "
          f"intercept={f['intercept']:+.2f} r2={f['r2']:.2f}  neg-EC={f['pct_neg']:.0f}%")""")

md(r"""## Idea 1b — PCA Addis-neighborhood (spectrally-matched subset)

PCA the pooled spectra, whiten, and keep the IMPROVE samples closest to the ETAD centroid.
Test a few neighborhood sizes.""")

code(r"""pool = np.vstack([Ximp, Xeth]).astype(np.float32); mu = pool.mean(0)
P = PCA(n_components=20, random_state=0).fit(pool - mu)
Simp, Seth = P.transform(Ximp - mu), P.transform(Xeth - mu)
w = 1.0 / np.sqrt(P.explained_variance_)
dist = np.sqrt((((Simp - Seth.mean(0)) * w)**2).sum(1))
order = np.argsort(dist)

nbhd = {}
for N in [1000, 2000, 4000]:
    mask = np.zeros(len(yimp), bool); mask[order[:N]] = True
    f = fit_hips(Ximp[mask], yimp[mask], Xeth, 10)
    nbhd[N] = (mask, f)
    print(f"Addis-neighborhood N={N} (EC med {np.median(yimp[mask]):.1f})  k=10:  "
          f"slope={f['slope']:.2f} intercept={f['intercept']:+.2f} r2={f['r2']:.2f}")""")

md(r"""## Idea 3 — Weakley pipeline: 2nd-derivative + biomass-band selection

2nd-derivative removes the sloping Teflon baseline; band selection keeps the organic proxy bands
(CH, C=O, aromatic/char C=C, OH) and drops the Teflon **CF** region (~1100-1300 cm⁻¹) that drives
the offset. Applied to the full lot-248+251 set.""")

code(r"""KEEP = ((WN >= 2800) & (WN <= 3050)) | ((WN >= 1550) & (WN <= 1800)) | \
       ((WN >= 3100) & (WN <= 3500)) | ((WN >= 1580) & (WN <= 1620))
CF   = (WN >= 1100) & (WN <= 1300)
print(f"band-selection keeps {KEEP.sum()} of {len(WN)} wavenumbers; CF region dropped ({CF.sum()} pts)")

Ximp_d2 = savgol_filter(Ximp.astype(float), 11, 2, deriv=2, axis=1)
Xeth_d2 = savgol_filter(Xeth.astype(float), 11, 2, deriv=2, axis=1)

variants = {
    "2nd-deriv (all bands)":  (Ximp_d2, Xeth_d2, slice(None)),
    "2nd-deriv + band-select": (Ximp_d2[:, KEEP], Xeth_d2[:, KEEP], slice(None)),
}
for name, (Xi, Xe, _) in variants.items():
    for k in [8, 12]:
        f = fit_hips(Xi, yimp, Xe, k)
        print(f"{name:26s} k={k}: slope={f['slope']:.2f} intercept={f['intercept']:+.2f} "
              f"r2={f['r2']:.2f} neg-EC={f['pct_neg']:.0f}%")""")

md(r"""## Progression crossplot — current → dedicated data → Weakley pipeline""")

code(r"""picks = [
    ("rds_EC 906  k=20  (current)", fit_hips(Xr, yr, Xeth, 20), "#1f77b4"),
    ("local_db 248+251  k=10  (dedicated data)", fit_hips(Ximp, yimp, Xeth, 10), "#2ca02c"),
    ("2nd-deriv + band-select  k=8", fit_hips(Ximp_d2[:, KEEP], yimp, Xeth_d2[:, KEEP], 8), "#d62728"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (title, f, col) in zip(axes, picks):
    x, yv = f["_x"], f["_y"]; hi = max(x.max(), yv.max()) * 1.1; lo = min(0, yv.min()) * 1.1
    ax.axhline(0, color="0.6", lw=0.8); ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1)
    ax.scatter(x, yv, s=42, alpha=0.5, color=col, edgecolors="k", linewidths=0.3)
    xs = np.array([0, hi]); ax.plot(xs, f["slope"] * xs + f["intercept"], color=col, lw=1.9)
    sign = "+" if f["intercept"] >= 0 else "−"
    ax.text(0.03, 0.97, f"y = {f['slope']:.2f}x {sign} {abs(f['intercept']):.2f}\n"
                        f"R² = {f['r2']:.3f}  n={f['n']}  neg={f['pct_neg']:.0f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.scatter([0], [0], marker="*", s=180, color="black", zorder=5)
    ax.set_xlim(0, hi); ax.set_ylim(lo, hi); ax.set_title(title, fontsize=10)
    ax.set_xlabel("HIPS EC = Fabs/10 (µg/m³)"); ax.set_ylabel("New FTIR-EC (µg/m³)")
fig.suptitle("ETAD New-EC vs HIPS — current (−5.25) → dedicated lot-248+251 data → +Weakley pipeline", y=1.02)
plt.tight_layout(); plt.savefig("figures/fig12_dedicated_progression.png", dpi=140, bbox_inches="tight")
print("saved figures/fig12_dedicated_progression.png"); plt.show()""")

md(r"""### What the data actually says
- **The dedicated *dataset* is the win, not the fancy selection.** Moving from the 906-sample
  rds_EC set to the full **lot-248+251 mirror** (both ETAD lots, EC-representative) takes the ETAD
  intercept from **−5.25 → ~−2.3** and the slope from **2.27 → ~1.2** on its own. This is Weakley's
  "dedicated atypical calibration" and Ann's "span both lots" — realized via the *right training
  population*, not PCA subsetting.
- **PCA Addis-neighborhood is marginal** on top of the lot-matched set — the lot-248+251 samples are
  already Addis-relevant, so re-selecting inside them adds little. Honest null-ish result worth
  reporting rather than overselling.
- **2nd-derivative + band-selection** tests the residual ~−2.3 offset (the Teflon-CF baseline). Read
  the printed slope/intercept: if it closes the gap without wrecking the slope, adopt it; if it just
  flattens the slope, the residual offset is method/MAC, not spectral.
- **Still no Ethiopia TOR truth** — this is HIPS-agreement + physical-sensibility (neg-EC %) tuning,
  not a validated ground-truth calibration. Say so on the slide.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("12_dedicated_ethiopia_calibration.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 12_dedicated_ethiopia_calibration.ipynb")
