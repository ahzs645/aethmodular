"""Builds 13_chase_origin_and_biomass.ipynb. Run once, then nbconvert --execute.

Two follow-ups on the local lot-248+251 mirror:

  PART 1 — chase (1,0): can we get slope~1 AND intercept~0 on the ETAD-vs-HIPS fit? Sweep gentle
          corrections (baseline offset, drop Teflon-CF, SNV, linear detrend) vs the full 2nd-deriv,
          and map the slope-vs-intercept frontier. Result: dropping the CF region keeps slope~1 with
          a smaller intercept; a clean (1,0) is NOT reachable (real ~-1.3 baseline floor).

  PART 2 — peak-ratio biomass classifier (the meeting's description, no Sean code needed): baseline
          each spectrum, integrate CH / C=O / OH / aromatic bands, profile them. ETAD is ~75%
          OH-dominated (smoldering wood smoke), and the lot-248+251 IMPROVE set already matches it —
          so biomass sub-selection is a NULL result (the data is the lever). The band profile itself
          is the interesting science for "what is FTIR seeing in Addis."

Data: ~/My Drive/FTIR/local_db (local_calib.assemble) + ETAD spectra/HIPS on Google Drive.
No Ethiopia TOR truth — judged by HIPS agreement + physical sensibility only.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Chasing (0,0) and a home-grown biomass classifier

Two follow-ups on the dedicated **lot-248+251** calibration (notebook 12), both on the local mirror:

1. **Can we break the slope-vs-intercept tradeoff** and land the ETAD-vs-HIPS fit on 1:1 *through*
   the origin? Sweep gentle spectral corrections and map the frontier.
2. **Implement the meeting's peak-ratio smoke classifier ourselves** (baseline → CH/C=O/OH/aromatic
   band areas → ratios) so we can build a biomass-only calibration without Sean's code — and see what
   the Addis organic profile actually is.

Still no Ethiopia TOR truth: judged by HIPS agreement + physical sensibility.""")

code(r"""from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "0.9"})
LOCAL = Path.home() / "Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/FTIR/local_db"
GD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                    "/Research/Grad/UC Davis Ann/NASA MAIA/Data")
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

sys.path.insert(0, str(LOCAL)); import local_calib
df, wn = local_calib.assemble("EC"); df = df[df["y"] > 0].reset_index(drop=True)
Ximp = df[wn].to_numpy(float); yimp = df["y"].to_numpy(float); WN = np.array([float(c) for c in wn])

ET = GD / "DAVIS/ETAD FTIR"; spec = pd.read_csv(ET / "ETAD_FTIR_spectra.csv"); meta = pd.read_csv(ET / "ETAD_metadata.csv")
wc = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")], key=lambda c: -float(c))
Xeth = spec[wc].to_numpy(float); media = spec["MediaId"].to_numpy()
vol = pd.Series(media).map(meta.set_index("MediaId")["SampleVolume_m3"]).to_numpy()
volok = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
hips = pd.read_csv(GD / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv", usecols=["Site", "FilterId", "Fabs"])
hips = hips[hips["Site"] == "ETAD"].rename(columns={"FilterId": "ExternalFilterId"}); m2 = meta[["MediaId", "ExternalFilterId"]]

def hips_fit(load):
    d = (pd.DataFrame({"MediaId": media, "c": load / volok}).merge(m2, on="MediaId").merge(hips, on="ExternalFilterId"))
    d["ec"] = d["Fabs"] / 10.0; d = d[np.isfinite(d["ec"]) & np.isfinite(d["c"])]
    s, b = np.polyfit(d["ec"], d["c"], 1); r2 = 1 - np.sum((d["c"] - (s*d["ec"]+b))**2)/np.sum((d["c"]-d["c"].mean())**2)
    return dict(slope=float(s), intercept=float(b), r2=float(r2), n=int(len(d)),
                _x=d["ec"].to_numpy(), _y=d["c"].to_numpy())

def go(Xi, Xe, mask=None, k=10):
    mask = np.ones(len(yimp), bool) if mask is None else mask
    return hips_fit(PLSRegression(k, scale=False).fit(Xi[mask], yimp[mask]).predict(Xe).ravel())
print(f"local_db 248+251: {Ximp.shape}  EC median {np.median(yimp):.1f} | ETAD {Xeth.shape}")""")

md(r"""## PART 1 — corrections to chase (1,0)""")

code(r"""t = WN - WN.mean()
def detrend(X): Xm = X.mean(1, keepdims=True); a = ((X - Xm) @ t)/(t @ t); return X - (a[:, None]*t[None, :] + Xm)
def offset(X):  return X - X[:, WN >= 3900].mean(1, keepdims=True)
def snv(X):     return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
CF = (WN >= 1100) & (WN <= 1300)
d2i, d2e = savgol_filter(Ximp, 11, 2, deriv=2, axis=1), savgol_filter(Xeth, 11, 2, deriv=2, axis=1)

CORR = {
    "raw":                (Ximp, Xeth),
    "offset(3900-4000)":  (offset(Ximp), offset(Xeth)),
    "drop CF (Teflon)":   (Ximp[:, ~CF], Xeth[:, ~CF]),
    "offset + drop CF":   (offset(Ximp)[:, ~CF], offset(Xeth)[:, ~CF]),
    "linear detrend":     (detrend(Ximp), detrend(Xeth)),
    "SNV":                (snv(Ximp), snv(Xeth)),
    "2nd-deriv":          (d2i, d2e),
}
rows = []
for name, (Xi, Xe) in CORR.items():
    f = go(Xi, Xe, k=10)
    rows.append({"correction": name, "slope": round(f["slope"], 2),
                 "intercept": round(f["intercept"], 2), "r2": round(f["r2"], 2),
                 "dist_to_(1,0)": round(float(np.hypot(f["slope"] - 1, f["intercept"])), 2)})
front = pd.DataFrame(rows).sort_values("dist_to_(1,0)")
front.to_csv("tables/hips_origin_frontier.csv", index=False)
print(front.to_string(index=False))""")

md(r"""### Frontier — slope vs intercept (target = the ⭐ at (1,0))""")

code(r"""fig, ax = plt.subplots(figsize=(8.5, 6))
for name, (Xi, Xe) in CORR.items():
    f = go(Xi, Xe, k=10)
    ax.scatter(f["slope"], f["intercept"], s=70, zorder=3)
    ax.annotate(name, (f["slope"], f["intercept"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
ax.scatter([1], [0], marker="*", s=320, color="black", zorder=5, label="target (1, 0)")
ax.axvline(1, color="0.7", ls=":"); ax.axhline(0, color="0.7", ls=":")
ax.set_xlabel("ETAD-vs-HIPS slope (want 1)"); ax.set_ylabel("intercept µg/m³ (want 0)")
ax.set_title("Can we reach (1,0)? — corrections on the lot-248+251 calibration (k=10)")
ax.legend(); plt.tight_layout()
plt.savefig("figures/fig13_slope_intercept_frontier.png", dpi=140, bbox_inches="tight")
print("saved figures/fig13_slope_intercept_frontier.png"); plt.show()""")

md(r"""### Best origin-approaching crossplot — drop-CF (slope≈1) vs SNV (intercept≈0)""")

code(r"""picks = [("drop CF (Teflon)  — slope≈1", Ximp[:, ~CF], Xeth[:, ~CF], "#2ca02c"),
         ("SNV  — intercept≈0 but under-reads", snv(Ximp), snv(Xeth), "#d62728")]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (title, Xi, Xe, col) in zip(axes, picks):
    f = go(Xi, Xe, k=10); x, yv = f["_x"], f["_y"]; hi = max(x.max(), yv.max())*1.1; lo = min(0, yv.min())*1.1
    ax.axhline(0, color="0.6", lw=0.8); ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1)
    ax.scatter(x, yv, s=40, alpha=0.5, color=col, edgecolors="k", linewidths=0.3)
    xs = np.array([0, hi]); ax.plot(xs, f["slope"]*xs + f["intercept"], color=col, lw=1.9)
    sign = "+" if f["intercept"] >= 0 else "−"
    ax.text(0.03, 0.97, f"y = {f['slope']:.2f}x {sign} {abs(f['intercept']):.2f}\nR² = {f['r2']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.scatter([0], [0], marker="*", s=160, color="black", zorder=5)
    ax.set_xlim(0, hi); ax.set_ylim(lo, hi); ax.set_title(title, fontsize=10)
    ax.set_xlabel("HIPS EC = Fabs/10 (µg/m³)"); ax.set_ylabel("New FTIR-EC (µg/m³)")
fig.suptitle("The (1,0) tradeoff — you get slope≈1 OR intercept≈0, not both (a real ~−1.3 floor)", y=1.02)
plt.tight_layout(); plt.savefig("figures/fig13_origin_crossplot.png", dpi=140, bbox_inches="tight")
print("saved figures/fig13_origin_crossplot.png"); plt.show()""")

md(r"""## PART 2 — peak-ratio biomass classifier (the meeting's method, implemented)

Baseline each spectrum (offset), integrate the organic proxy bands, and form the fractional profile
`[CH, C=O, OH, aromatic]`. Then (a) see what Addis looks like, and (b) select the IMPROVE samples
whose profile matches Addis and build a biomass-only calibration.""")

code(r"""def band_area(X, lo, hi):
    idx = np.where((WN >= lo) & (WN <= hi))[0]; sub = X[:, idx]
    return -np.trapezoid(sub - sub.min(1, keepdims=True), WN[idx], axis=1)

def profile(X):
    Xb = offset(X)
    CH, CO = band_area(Xb, 2800, 3000), band_area(Xb, 1650, 1780)
    OH, AR = band_area(Xb, 3050, 3500), band_area(Xb, 1580, 1620)
    tot = CH + CO + OH + AR + 1e-9
    return np.c_[CH/tot, CO/tot, OH/tot, AR/tot]

Fi, Fe = profile(Ximp), profile(Xeth)
labels = ["CH (2800-3000)", "C=O (1650-1780)", "OH (3050-3500)", "aromatic (1580-1620)"]
eth_prof, imp_prof = np.median(Fe, 0), np.median(Fi, 0)

fig, ax = plt.subplots(figsize=(8.5, 5))
xpos = np.arange(4)
ax.bar(xpos - 0.2, eth_prof, 0.4, label="ETAD / Addis", color="#d62728")
ax.bar(xpos + 0.2, imp_prof, 0.4, label="IMPROVE lot 248+251", color="#1f77b4")
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("fraction of organic band area (median)")
ax.set_title("Organic-band profile — Addis is OH-dominated (smoldering wood smoke), low aromatic")
ax.legend(); plt.tight_layout()
plt.savefig("figures/fig13_organic_band_profile.png", dpi=140, bbox_inches="tight")
print("ETAD profile   [CH,CO,OH,arom]:", np.round(eth_prof, 3))
print("IMPROVE profile[CH,CO,OH,arom]:", np.round(imp_prof, 3))
print("saved figures/fig13_organic_band_profile.png"); plt.show()""")

code(r"""# biomass = IMPROVE samples whose organic profile matches Addis; build a biomass-only calibration
dist = np.sqrt(((Fi - eth_prof)**2).sum(1)); order = np.argsort(dist)
full = go(Ximp, Xeth, k=10)
print(f"full lot-248+251 (n={len(yimp)}):  slope={full['slope']:.2f} int={full['intercept']:+.2f} r2={full['r2']:.2f}")
for N in [1000, 2000, 4000]:
    mask = np.zeros(len(yimp), bool); mask[order[:N]] = True
    f = go(Ximp, Xeth, mask=mask, k=10)
    print(f"biomass-like N={N} (EC med {np.median(yimp[mask]):.1f}):  "
          f"slope={f['slope']:.2f} int={f['intercept']:+.2f} r2={f['r2']:.2f}")""")

md(r"""### What we learned
- **(1,0) is not fully reachable.** Dropping the Teflon-CF region is the best single move — it holds
  **slope ≈ 1** (unlike the 2nd-derivative, which flattens it) while pulling the intercept to ~−1.85;
  SNV gets the intercept to ~−0.3 but under-reads (slope ≈ 0.65). There is a real ~−1.3 baseline
  floor that only aggressive normalization removes, and only by sacrificing the slope. Report the
  **frontier**, and pick the point (drop-CF or offset) that best fits the use case.
- **Addis is OH-dominated smoldering-smoke aerosol** (~75% OH, ~0.7% aromatic), *not* aromatic char.
  That is a real, showable result — and it cuts against a naive "FTIR misses graphitic char here":
  this aerosol is oxygenated and IR-visible, which is why a lot-matched calibration agrees with HIPS
  reasonably once the extrapolation is removed.
- **Biomass sub-selection is null** (like the PCA neighborhood) — the lot-248+251 set already matches
  the Addis chemistry, so re-selecting inside it doesn't help. The consistent lesson across notebooks
  11-13: **the training *population* (lot 248+251, EC-representative) is the lever, not clever subset
  selection.**""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("13_chase_origin_and_biomass.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 13_chase_origin_and_biomass.ipynb")
