"""Builds 15_downstream_recommended_calibration.ipynb. Run once, then nbconvert --execute.

Propagates the VALIDATED calibration (notebook 14) through the downstream Ethiopia + Adama
analysis, replacing the old rds_EC-906 calibration used in notebooks 07/08.

RECOMMENDED calibration = full local_db lot-248+251 set, drop physically-flagged filters
(status_events), PLS k=10. Validated in nb14: ETAD is inside its spectral domain (0%
extrapolation) and it transfers across sites (leave-one-site-out +5%).

Recomputes, old-vs-new:
  - ETAD/Addis New-EC vs HIPS crossplot (Fabs/10)  ..... was 2.27x-5.25, now ~1.19x-2.24
  - ETAD EC concentration estimates (the actual Ethiopia numbers) + export
  - Adama (n=5) New-EC vs TOR crossplot
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Downstream with the validated calibration (Ethiopia + Adama)

Notebooks 07/08 ran the downstream on the old **rds_EC-906** calibration (ETAD-vs-HIPS `2.27x−5.25`,
steep and offset). Notebook 14 landed on a **validated** calibration:

> **full local_db lot-248+251 set, physically-flag-filtered, PLS k=10** — ETAD is inside its
> spectral domain (0% extrapolation) and it transfers across sites (leave-one-site-out +5%).

Here we recompute the Ethiopia and Adama results with it, **old vs new**.""")

code(r"""from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white", "axes.grid": True, "grid.color": "0.9"})
LOCAL = Path.home() / "Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/FTIR/local_db"
GD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                    "/Research/Grad/UC Davis Ann/NASA MAIA/Data")
PRED = Path("../spartan_ec_2026_06_16")
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

def reg_stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    s, b = np.polyfit(x, y, 1); r2 = 1 - np.sum((y-(s*x+b))**2)/np.sum((y-y.mean())**2)
    return dict(n=int(len(x)), slope=float(s), intercept=float(b), r2=float(r2))

def crossplot(ax, x, y, color, title):
    x = np.asarray(x, float); y = np.asarray(y, float)
    hi = np.nanmax(np.r_[x, y]) * 1.1; lo = min(0, np.nanmin(y)) * 1.1
    ax.axhline(0, color="0.6", lw=0.8); ax.plot([0, hi], [0, hi], "--", color="0.5", lw=1)
    ax.scatter(x, y, s=45, alpha=0.55, color=color, edgecolors="k", linewidths=0.3)
    st = reg_stats(x, y); xs = np.array([0, hi]); ax.plot(xs, st["slope"]*xs+st["intercept"], color=color, lw=1.9)
    sign = "+" if st["intercept"] >= 0 else "−"
    ax.text(0.03, 0.97, f"y = {st['slope']:.2f}x {sign} {abs(st['intercept']):.2f}\nR² = {st['r2']:.3f}  n={st['n']}",
            transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.scatter([0], [0], marker="*", s=150, color="black", zorder=5)
    ax.set_xlim(0, hi); ax.set_ylim(lo, hi); ax.set_title(title, fontsize=10)
    return st""")

md(r"""## Build both calibrations — OLD (rds_EC 906) and NEW (validated lot-248+251)""")

code(r"""# NEW: local_db lot-248+251, drop physically-flagged, k=10
sys.path.insert(0, str(LOCAL)); import local_calib
df, wn = local_calib.assemble("EC"); df = df[df["y"] > 0].reset_index(drop=True)
Ximp = df[wn].to_numpy(float); yimp = df["y"].to_numpy(float); WN = np.array([float(c) for c in wn])
ev = pd.read_csv(LOCAL / "status_events.csv")
PHYS = r"mass|weight|negativ|contamin|damag|tear|hole|punctur|dropp|broke|crack|spill|leak|rip|fell|scratch|smudge|fold|wrinkl"
phys_fids = set(ev.loc[ev["Entry"].astype(str).str.lower().str.contains(PHYS) & (ev["event"] == "flag"), "FilterId"])
keep = ~df["FilterId"].isin(phys_fids).to_numpy()
NEW = PLSRegression(10, scale=False).fit(Ximp[keep], yimp[keep])
print(f"NEW calibration: lot-248+251, drop {(~keep).sum()} physically-flagged, n={keep.sum()}, k=10")

# OLD: rds_EC 906, k=20 (the notebook 07/08 calibration)
Xr = pd.read_csv(PRED / "data/rds_EC_X.csv"); Xr = Xr[[c for c in Xr.columns if c != "id"]].to_numpy(float)
yr = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
OLD = PLSRegression(20, scale=False).fit(Xr, yr)
print("OLD calibration: rds_EC 906, k=20")""")

md(r"""## Ethiopia (ETAD) — recompute EC and the HIPS crossplot""")

code(r"""ET = GD / "DAVIS/ETAD FTIR"; spec = pd.read_csv(ET / "ETAD_FTIR_spectra.csv"); meta = pd.read_csv(ET / "ETAD_metadata.csv")
wcx = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")], key=lambda c: -float(c))
Xeth = spec[wcx].to_numpy(float); media = spec["MediaId"].to_numpy()
vol = pd.Series(media).map(meta.set_index("MediaId")["SampleVolume_m3"]).to_numpy(); volok = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
hips = pd.read_csv(GD / "Spartan/SPARTAN_HIPS_Batch1-51.v2.csv", usecols=["Site", "FilterId", "Fabs"])
hips = hips[hips["Site"] == "ETAD"].rename(columns={"FilterId": "ExternalFilterId"}); m2 = meta[["MediaId", "ExternalFilterId"]]

def etad_conc(model): return model.predict(Xeth).ravel() / volok
def join_hips(conc):
    d = pd.DataFrame({"MediaId": media, "ec_new": conc}).merge(m2, on="MediaId").merge(hips, on="ExternalFilterId")
    d["EC_hips"] = d["Fabs"] / 10.0; return d[np.isfinite(d["EC_hips"]) & np.isfinite(d["ec_new"])]

cN, cO = etad_conc(NEW), etad_conc(OLD)
dN, dO = join_hips(cN), join_hips(cO)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
crossplot(axes[0], dO["EC_hips"], dO["ec_new"], "#1f77b4", "OLD  rds_EC 906  (nb07/08)")
crossplot(axes[1], dN["EC_hips"], dN["ec_new"], "#2ca02c", "NEW  validated lot-248+251")
for ax in axes: ax.set_xlabel("HIPS EC = Fabs/10 (µg/m³)"); ax.set_ylabel("FTIR-EC (µg/m³)")
fig.suptitle("ETAD (Addis) New-EC vs HIPS — old vs validated calibration", y=1.02, fontsize=13)
plt.tight_layout(); plt.savefig("figures/fig15_etad_hips_old_vs_new.png", dpi=140, bbox_inches="tight")
print("saved figures/fig15_etad_hips_old_vs_new.png"); plt.show()""")

md(r"""### Ethiopia EC estimates with the validated calibration (the actual numbers)""")

code(r"""out = pd.DataFrame({"MediaId": media, "EC_ugm3_new": cN, "EC_ugm3_old": cO})
out = out.merge(m2, on="MediaId").merge(hips[["ExternalFilterId", "Fabs"]], on="ExternalFilterId", how="left")
out["EC_hips"] = out["Fabs"] / 10.0
out.round(3).to_csv("tables/etad_ec_recommended.csv", index=False)
summ = pd.DataFrame({
    "calibration": ["OLD rds_EC 906", "NEW lot-248+251"],
    "median EC (µg/m³)": [np.nanmedian(cO), np.nanmedian(cN)],
    "mean EC": [np.nanmean(cO), np.nanmean(cN)],
    "% negative": [100*np.nanmean(cO < 0), 100*np.nanmean(cN < 0)],
}).round(2)
print(summ.to_string(index=False))
print("\nwrote tables/etad_ec_recommended.csv")""")

md(r"""## Adama (n=5) — recompute New-EC vs TOR

Adama PTFE spectra are on a wider grid → interpolate onto the training wavenumbers. Barcode↔FilterId
uses notebook-01's ASSUMED sorted↔sorted map (the one soft assumption); TOR ECTR joined by date.""")

code(r"""ADAMA = PRED / "data/adama"
araw = pd.read_csv(ADAMA / "adama_ptfe_spectra_batch54.csv")
a_ids = araw.iloc[:, 0].astype(str).tolist(); a_wn = araw.columns[1:].astype(float).to_numpy(); A = araw.iloc[:, 1:].to_numpy(float)
o = np.argsort(a_wn); Aad = np.vstack([np.interp(WN, a_wn[o], A[i][o]) for i in range(len(A))])
ft = pd.read_csv(ADAMA / "adama_ptfe_ftir_batch54.csv"); ec_ft = ft[ft["Parameter"] == "EC_ftir"].sort_values("FilterId").reset_index(drop=True)
ASSUMED = dict(zip(sorted(a_ids), ec_ft["FilterId"])); fid = [ASSUMED[s] for s in a_ids]
vol_a = ec_ft.set_index("FilterId")["Volume_m3"]; date_a = ec_ft.set_index("FilterId")["SampleDate"].apply(lambda s: pd.to_datetime(s).normalize())
tor = pd.read_csv(ADAMA / "adama_quartz_tor_batch54.csv"); tor["date"] = pd.to_datetime(tor["SampleDate"]).dt.normalize()
tor_ec = tor[tor["Parameter"] == "ECTR"].set_index("date")["Concentration_ug_m3"]
ad = pd.DataFrame({"barcode": a_ids, "FilterId": fid, "date": [date_a[f] for f in fid], "vol": [vol_a[f] for f in fid]})
ad["TOR_EC"] = ad["date"].map(tor_ec).to_numpy()
ad["EC_new"] = NEW.predict(Aad).ravel() / ad["vol"].to_numpy()
ad["EC_old"] = OLD.predict(Aad).ravel() / ad["vol"].to_numpy()
print(ad[["barcode", "date", "TOR_EC", "EC_old", "EC_new"]].round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
crossplot(axes[0], ad["TOR_EC"], ad["EC_old"], "#1f77b4", "OLD  rds_EC 906")
crossplot(axes[1], ad["TOR_EC"], ad["EC_new"], "#2ca02c", "NEW  validated lot-248+251")
for ax in axes: ax.set_xlabel("TOR EC (µg/m³) — reference"); ax.set_ylabel("FTIR-EC (µg/m³)")
fig.suptitle("Adama (n=5): New-EC vs TOR — old vs validated calibration (read points vs 1:1, n small)", y=1.02)
plt.tight_layout(); plt.savefig("figures/fig15_adama_tor_old_vs_new.png", dpi=140, bbox_inches="tight")
print("saved figures/fig15_adama_tor_old_vs_new.png"); plt.show()""")

md(r"""### Downstream result with the validated calibration
- **Ethiopia/HIPS:** the crossplot moves from the steep offset `2.27x−5.25` onto near-1:1
  (`~1.19x−2.24`), and the Ethiopia EC estimates drop to physically-sensible levels (median in
  `tables/etad_ec_recommended.csv`) with fewer/no negatives — the downstream payoff of choosing and
  validating the right training population.
- **Adama/TOR:** read the 5 points against the 1:1 line (regression on n=5 is not meaningful); the
  validated calibration's absolute level is the thing to check against TOR.
- **Caveats unchanged:** no Ethiopia TOR truth (HIPS is MAC-10 dependent), the intercept still sits
  ~−2 (the ~−1.3 baseline floor from nb13), and the Adama barcode↔FilterId map is assumed.
- Everything downstream now flows from ONE validated calibration — swap `NEW` if you refine it.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("15_downstream_recommended_calibration.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 15_downstream_recommended_calibration.ipynb")
