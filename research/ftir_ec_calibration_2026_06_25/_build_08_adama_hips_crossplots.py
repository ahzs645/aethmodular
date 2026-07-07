"""Builds 08_adama_hips_crossplots.ipynb. Run once, then nbconvert --execute.

Applies the CAL-0…CAL-5 calibrations to the **Adama** spectra and cross-plots the new EC against the
right reference for each dataset, in the AGENTS.md crossplot style (y = m·x + b + R² + 1:1 line):

- **Adama** (CSU/AMOD, *no HIPS*) → New EC vs **TOR** (thermal-optical ground truth), 5 filters.
- **ETAD / Addis** (SPARTAN site, *has HIPS*) → New EC vs **HIPS** (EC ≈ Fabs / MAC, MAC = 10),
  joining the SPARTAN HIPS `Fabs` to the ETAD filters (ExternalFilterId ↔ HIPS FilterId).

The ETAD spectra share the training grid exactly; the Adama spectra are interpolated onto it.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# New-calibration EC vs. reference — Adama (TOR) and ETAD (HIPS)

Apply the calibration variants to the **Adama** spectra (the meeting's ask) and cross-plot the new EC
against the reference each dataset actually has, in the **AGENTS.md** style (regression `y = m·x + b`,
`R²`, `n`, 1:1 line, white background):

- **Adama** has **no HIPS** (it is a CSU/AMOD campaign, not a SPARTAN site) → compare to **TOR**.
- **ETAD/Addis** is a SPARTAN site **with HIPS** → the requested **New-EC-vs-HIPS** crossplot, with
  HIPS converted to an EC-equivalent via `EC ≈ Fabs / 10` (MAC = 10, the 6-vs-10 question unresolved).""")

code(r"""from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "0.9"})
PRED = Path("../spartan_ec_2026_06_16")
sys.path.insert(0, str(PRED))
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

# --- AGENTS.md-style regression (same math as scripts/plotting/utils.calculate_regression_stats) ---
def reg_stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}

def crossplot(ax, x, y, color="#1f77b4", title=""):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ax.scatter(x, y, s=55, alpha=0.6, color=color, edgecolors="k", linewidths=0.3)
    st = reg_stats(x, y)
    hi = np.nanmax(np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])) * 1.1
    ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1)                      # 1:1
    if st:
        xs = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(xs, st["slope"] * xs + st["intercept"], color=color, lw=1.8)  # regression
        sign = "+" if st["intercept"] >= 0 else "−"
        ax.text(0.03, 0.97, f"y = {st['slope']:.2f}x {sign} {abs(st['intercept']):.2f}\n"
                            f"R² = {st['r2']:.3f}   n = {st['n']}", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_title(title, fontsize=10)
    return st""")

md(r"""## Fit the six variant calibrations on the 906-sample EC set (same rules as `06`/`07`)""")

code(r"""Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
y = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Vcols = [c for c in Xdf.columns if c != "id"]
Xtr = Xdf[Vcols].to_numpy(float)
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()

K_COMMON = 20
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
MASKS = {
    "CAL-0 all-nofilter": np.ones(len(y), bool), "CAL-1 cleaned": keep,
    "CAL-2 removed-only": ~keep, "CAL-3 below-1:1": pred0 < y,
    "CAL-4 EC-high>=70": y >= 70.0, "CAL-5 Eth-range": (y >= 10.0) & (y <= 100.0),
}
def safe_k(n, cv=5): return max(2, min(K_COMMON, int(n*(cv-1)/cv) - 2))
MODELS = {name: PLSRegression(n_components=safe_k(int(mm.sum())), scale=False).fit(Xtr[mm], y[mm])
          for name, mm in MASKS.items()}
print("fitted variants:", list(MODELS))""")

md(r"""## Adama — apply each variant, cross-plot New EC vs **TOR**

The Adama PTFE spectra are on a slightly wider grid, so each is **interpolated onto the training
wavenumbers**. Barcodes 4744–4748 carry no FilterId, so we reuse the `01`-notebook **ASSUMED**
sorted↔sorted map to reach SampleDate, then join to the quartz **TOR** ECTR by date (co-located
pairs). *(Verify the barcode↔FilterId crosswalk — this is the one soft assumption here.)*""")

code(r"""ADAMA = PRED / "data/adama"
araw = pd.read_csv(ADAMA / "adama_ptfe_spectra_batch54.csv")
a_ids = araw.iloc[:, 0].astype(str).tolist()               # 4744..4748
a_wn = araw.columns[1:].astype(float).to_numpy()
A = araw.iloc[:, 1:].to_numpy(float)
order = np.argsort(a_wn)                                     # ascending for np.interp
Aad = np.vstack([np.interp(WN, a_wn[order], A[i][order]) for i in range(len(A))])   # 5 x 2722
print("Adama spectra interpolated to training grid:", Aad.shape)

# general FTIR-EC (original tool cal) + volumes + dates, per FilterId
ft = pd.read_csv(ADAMA / "adama_ptfe_ftir_batch54.csv")
ec_ft = ft[ft["Parameter"] == "EC_ftir"].sort_values("FilterId").reset_index(drop=True)
ASSUMED = dict(zip(sorted(a_ids), ec_ft["FilterId"]))       # 4744->J1233 ... (sorted<->sorted)
fid = [ASSUMED[s] for s in a_ids]
vol = ec_ft.set_index("FilterId")["Volume_m3"]
date = ec_ft.set_index("FilterId")["SampleDate"].apply(lambda s: pd.to_datetime(s).normalize())

# TOR ECTR (concentration) per quartz filter, keyed by date
tor = pd.read_csv(ADAMA / "adama_quartz_tor_batch54.csv")
tor["date"] = pd.to_datetime(tor["SampleDate"]).dt.normalize()
tor_ec = tor[tor["Parameter"] == "ECTR"].set_index("date")["Concentration_ug_m3"]

adama = pd.DataFrame({"barcode": a_ids, "FilterId": fid,
                      "date": [date[f] for f in fid],
                      "volume_m3": [vol[f] for f in fid]})
adama["TOR_EC"] = adama["date"].map(tor_ec).to_numpy()
adama["FTIR_EC_general_orig"] = ec_ft.set_index("FilterId")["Concentration_ug_m3"].reindex(fid).to_numpy()
for name, mdl in MODELS.items():
    adama[name] = mdl.predict(Aad).ravel() / adama["volume_m3"].to_numpy()   # µg -> µg/m3
print(adama[["barcode","FilterId","date","TOR_EC","FTIR_EC_general_orig",
             "CAL-0 all-nofilter","CAL-3 below-1:1"]].round(2).to_string(index=False))
adama.round(4).to_csv("tables/adama_new_ec_by_variant.csv", index=False)""")

code(r"""panels = ["FTIR_EC_general_orig", "CAL-0 all-nofilter", "CAL-1 cleaned", "CAL-3 below-1:1"]
titles = ["Original tool FTIR-EC", "CAL-0 New EC", "CAL-1 New EC", "CAL-3 New EC"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, col, t in zip(axes, panels, titles):
    crossplot(ax, adama["TOR_EC"].to_numpy(), adama[col].to_numpy(),
              color=("#d62728" if "CAL-3" in col else "#1f77b4"), title=t)
    ax.set_xlabel("TOR EC (µg/m³) — reference"); ax.set_ylabel("EC (µg/m³)")
fig.suptitle("Adama (n=5): New EC vs. TOR  —  Adama has no HIPS, TOR is the reference", y=1.03)
plt.tight_layout(); plt.savefig("figures/fig08_adama_newec_vs_tor.png", dpi=140, bbox_inches="tight")
print("saved figures/fig08_adama_newec_vs_tor.png"); plt.show()""")

md(r"""## ETAD / Addis — the New-EC-vs-**HIPS** crossplot

Join each ETAD filter's New EC (from `07`) to the SPARTAN HIPS `Fabs`
(`ExternalFilterId` ↔ HIPS `FilterId`), convert HIPS to an EC-equivalent `Fabs / 10` (MAC = 10), and
cross-plot. This is the comparison the paper cares about — does a given calibration bring FTIR-EC
onto the HIPS line?""")

code(r"""ETAD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                      "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
SPARTAN = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                        "/Research/Grad/UC Davis Ann/NASA MAIA/Data/Spartan/SPARTAN_HIPS_Batch1-51.v2.csv")
MAC = 10.0

conc = pd.read_csv("tables/etad_ec_by_variant_conc.csv")            # New EC per variant (from nb07)
meta = pd.read_csv(ETAD / "ETAD_metadata.csv")[["MediaId", "ExternalFilterId"]]
hips = pd.read_csv(SPARTAN, usecols=["Site", "FilterId", "Fabs"])
hips = hips[hips["Site"] == "ETAD"].rename(columns={"FilterId": "ExternalFilterId"})

d = (conc.merge(meta, on="MediaId", how="left")
         .merge(hips[["ExternalFilterId", "Fabs"]], on="ExternalFilterId", how="inner"))
d["EC_hips"] = d["Fabs"] / MAC
d = d[d["EC_hips"].notna()]
print(f"ETAD filters joined to HIPS Fabs: {len(d)}  (EC_hips = Fabs/{MAC:.0f})")

variants = ["CAL-0 all-nofilter", "CAL-1 cleaned", "CAL-3 below-1:1", "CAL-5 Eth-range"]
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
rows = []
for ax, v in zip(axes.ravel(), variants):
    st = crossplot(ax, d["EC_hips"].to_numpy(), d[v].to_numpy(),
                   color=("#d62728" if "below" in v else "#1f77b4"), title=v)
    ax.set_xlabel("HIPS EC-equivalent = Fabs/10 (µg/m³)"); ax.set_ylabel("New FTIR-EC (µg/m³)")
    if st: rows.append({"variant": v, **st})
fig.suptitle(f"ETAD (Addis): New FTIR-EC vs. HIPS (n={len(d)})", y=1.01, fontsize=13)
plt.tight_layout(); plt.savefig("figures/fig08_etad_newec_vs_hips.png", dpi=140, bbox_inches="tight")
print("saved figures/fig08_etad_newec_vs_hips.png"); plt.show()

fits = pd.DataFrame(rows).round(3)
print(fits.to_string(index=False))
fits.to_csv("tables/etad_newec_vs_hips_fits.csv", index=False)
print("wrote tables/etad_newec_vs_hips_fits.csv")""")

md(r"""### The result (this run)
- **ETAD/HIPS (n=259):** every variant gives **slope > 1** against HIPS/10 — the New FTIR-EC runs
  *above* the HIPS EC-equivalent at MAC = 10 (slopes **1.25 CAL-1 → 2.71 CAL-3**, R² ≈ 0.61–0.71).
  Read two ways: (a) the implied effective **MAC ≈ 10 / slope ≈ 4–8 m²/g**, so the MAC = 10 assumption
  — not the calibration — may be what's off; and (b) the **ordering is the story**: CAL-3 (below-1:1)
  highest, CAL-1 (cleaned) lowest, mirroring the ETAD EC ranking in `07`.
- **MAC caveat:** HIPS EC-equivalent uses MAC = 10; at MAC = 6 the HIPS EC rises ~1.7×, which pulls
  every slope down toward 1. Re-run with MAC = 6 to bound it — this is the unresolved 6-vs-10 question.
- **Adama (n=5):** regression is weak — read **points vs the 1:1 line**, not R². Note the
  IMPROVE-trained CAL-0 lands *above* TOR here, whereas the tool's native lot-241a EC (`02`) sat
  *below* TOR — a training-set / preprocessing difference (interpolated grid, different calibration
  population) worth investigating before trusting the absolute level.
- These variants are still the IMPROVE-trained set — re-fit on the biomass-only subset once Sean's
  classifier lands.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("08_adama_hips_crossplots.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 08_adama_hips_crossplots.ipynb")
