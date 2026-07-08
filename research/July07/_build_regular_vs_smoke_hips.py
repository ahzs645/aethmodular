"""Builds regular_vs_smoke_ec_vs_hips.ipynb. Run once, then nbconvert --execute.

Core question (2026-07-07): **does calibrating with smoke samples actually improve anything?**
i.e. does a biomass-burning-only calibration bring FTIR-EC closer to the HIPS ground truth than the
regular all-data calibration — on the real Ethiopia (ETAD/Addis) filters?

This is the plot that was missing: the **old/original data (the regular all-data calibration) EC vs
HIPS**, rendered in the *same* crossplot format as `ftir_ec_calibration_2026_06_25/fig08` so it can be
laid next to the smoke calibration and compared like-for-like.

Both calibrations are applied to the **same 319 ETAD spectra** and joined to the **same** SPARTAN HIPS
Fabs (n=259), so the only thing that changes between panels is the calibration itself:

- **Regular / all-data (CAL-0 "keep everything")** — PLS(k=20) fit on the full 906-sample EC training
  set (same model as fig08's CAL-0). This is the "regular all-data calibration."
- **Smoke-only (biomass lot-251)** — the real biomass-burning calibration coefficients
  (`tool_EC_coeffs_lot251_biomass.csv`), applied to the ETAD spectra directly (same 2722-pt grid).
- **Deployed general-cal EC (reported)** — a guarded cross-check pulling the EC the SPARTAN tool
  actually reported for these filters from `unified_filter_dataset.pkl` (never breaks the notebook).

Axis convention (per 2026-07-07 standardization): **HIPS (ground truth) on x, FTIR-EC on y**, equal
0-20 µg/m3 range on both axes (fit to the ETAD data; the 0-400 tool standard is for the
predicted-vs-measured EC plots, not this HIPS crossplot). HIPS -> EC-equivalent via EC = Fabs / MAC,
MAC = 10.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Does the smoke-only calibration actually help? — Regular vs. biomass EC vs. HIPS (ETAD/Addis)

**The core question:** SPARTAN uses one *general, all-data* calibration for every filter. Our own
FTIR-EC paper argues smoke samples should be calibrated with *smoke* samples. So — on the real
Ethiopia filters — does a **biomass-burning-only** calibration move FTIR-EC **closer to the
independent HIPS** ground truth than the **regular all-data** calibration, or not?

To answer it we put the **old/original data (regular all-data calibration) EC vs HIPS** crossplot in
the *same format* as `ftir_ec_calibration_2026_06_25/figures/fig08_etad_newec_vs_hips.png`, next to the
smoke calibration. Both calibrations are applied to the **same 319 ETAD spectra** and joined to the
**same** SPARTAN HIPS (n=259) — only the calibration differs.

**Axis standard (2026-07-07):** HIPS (ground truth) on **x**, FTIR-EC on **y**, equal **0-20 µg/m³**
range (fit to these data — the 0-400 tool standard is for the predicted-vs-measured EC plots).
HIPS → EC-equivalent as `EC = Fabs / 10` (MAC = 10; the 6-vs-10 question is still open).

> **Read it honestly.** "Regular" here is the all-data PLS on the IMPROVE-based 906-sample set;
> "smoke" is the SPARTAN lot-251 biomass tool calibration. They differ in *sample selection* (the
> thing we're testing) but also in *training population* (IMPROVE vs SPARTAN lot-251), so treat the
> **direction** (does smoke move toward the 1:1 / HIPS line?) as the result, not the absolute level.""")

code(r"""from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "0.9"})

PRED = Path("../spartan_ec_2026_06_16")            # training data + biomass coeffs live here
sys.path.insert(0, str(PRED))
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

MAC = 10.0            # m²/g — HIPS Fabs -> EC-equivalent
AXIS_MAX = 20.0       # µg/m³, equal on both axes (fit to ETAD; 0-400 is the tool pred-vs-meas standard)

ETAD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                      "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
SPARTAN = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                        "/Research/Grad/UC Davis Ann/NASA MAIA/Data/Spartan/SPARTAN_HIPS_Batch1-51.v2.csv")


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


def crossplot(ax, hips, ec, color="#1f77b4", title=""):
    '''HIPS (ground truth) on x, FTIR-EC on y, equal 0-AXIS_MAX range, 1:1 + regression + stats box.'''
    hips = np.asarray(hips, float); ec = np.asarray(ec, float)
    m = np.isfinite(hips) & np.isfinite(ec)
    ax.scatter(hips[m], ec[m], s=45, alpha=0.55, color=color, edgecolors="k", linewidths=0.3, zorder=3)
    ax.plot([0, AXIS_MAX], [0, AXIS_MAX], "--", color="0.5", lw=1, zorder=2)          # 1:1
    st = reg_stats(hips, ec)
    if st:
        xs = np.array([0.0, AXIS_MAX])
        ax.plot(xs, st["slope"] * xs + st["intercept"], color=color, lw=1.9, zorder=4)  # regression
        mac_eff = MAC / st["slope"] if st["slope"] > 0 else np.nan
        sign = "+" if st["intercept"] >= 0 else "−"
        ax.text(0.03, 0.97,
                f"y = {st['slope']:.2f}x {sign} {abs(st['intercept']):.2f}\n"
                f"R² = {st['r2']:.3f}   n = {st['n']}\n"
                f"implied MAC = {mac_eff:.1f} m²/g",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlim(0, AXIS_MAX); ax.set_ylim(0, AXIS_MAX); ax.set_aspect("equal")
    ax.set_xlabel("HIPS EC-equivalent = Fabs/10 (µg/m³)  —  ground truth")
    ax.set_ylabel("FTIR-EC (µg/m³)")
    ax.set_title(title, fontsize=11)
    return st

print("MAC =", MAC, "| axis 0 –", AXIS_MAX, "µg/m³ (HIPS x, FTIR-EC y)")
print("ETAD dir:", ETAD.exists(), "| SPARTAN HIPS:", SPARTAN.exists())""")

md(r"""## 1. Load the ETAD spectra + training grid

The ETAD export sits on the **identical 2722-point wavenumber grid** (3998→500 cm⁻¹) as the training
data, so every calibration applies directly — no interpolation.""")

code(r"""# training set (906 x 2722) + the real wavenumber grid
Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv")
ytr = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
Vcols = [c for c in Xdf.columns if c != "id"]
Xtr = Xdf[Vcols].to_numpy(float)
WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()
assert len(WN) == Xtr.shape[1]

# ETAD spectra + per-filter sample volume (µg loading -> µg/m³)
spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
meta = pd.read_csv(ETAD / "ETAD_metadata.csv")
wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")],
               key=lambda c: -float(c))                          # descending = training order
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != training grid"
Xeth = spec[wcols].to_numpy(float)
media = spec["MediaId"].to_numpy()
vol = pd.Series(media).map(meta.set_index("MediaId")["SampleVolume_m3"]).to_numpy()
vol_ok = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)     # guard 0-volume filters
print(f"training {Xtr.shape} | ETAD {Xeth.shape} | volume ok for "
      f"{int(np.isfinite(vol_ok).sum())}/{len(media)} filters")""")

md(r"""## 2. Apply the two calibrations to the ETAD spectra

**Regular / all-data (CAL-0)** — PLS(k=20) on all 906 training samples (identical to fig08's CAL-0).
**Smoke-only (biomass lot-251)** — the tool's biomass coefficients: `EC = b₀ + Σ bᵢ·absorbanceᵢ`.
Both predict filter **loading (µg)**; divide by sample volume for **concentration (µg/m³)**.""")

code(r"""# --- Regular all-data calibration: CAL-0 keep-everything (same as fig08 top-left) ---
K_COMMON = 20
cal0 = PLSRegression(n_components=K_COMMON, scale=False).fit(Xtr, ytr)
ec_cal0 = cal0.predict(Xeth).ravel() / vol_ok                    # µg -> µg/m³

# --- Smoke-only calibration: biomass lot-251 tool coefficients ---
bc = pd.read_csv(PRED / "data/tool_EC_coeffs_lot251_biomass.csv")
bc.columns = ["idx", "Wavenumber", "b"]
b0 = float(bc.loc[bc["Wavenumber"] == 0, "b"].iloc[0])          # intercept row (Wavenumber == 0)
coef = bc[bc["Wavenumber"] != 0].copy()
assert np.allclose(coef["Wavenumber"].to_numpy(), WN), "biomass coeff grid != training grid"
ec_bio = (b0 + Xeth @ coef["b"].to_numpy(float)) / vol_ok       # µg -> µg/m³

print(f"biomass intercept b0 = {b0:.3f} µg | {len(coef)} wavenumber coefficients")
print(f"regular  (CAL-0)  EC µg/m³:  median {np.nanmedian(ec_cal0):.2f}  "
      f"[{np.nanmin(ec_cal0):.2f}, {np.nanmax(ec_cal0):.2f}]")
print(f"smoke (biomass)   EC µg/m³:  median {np.nanmedian(ec_bio):.2f}  "
      f"[{np.nanmin(ec_bio):.2f}, {np.nanmax(ec_bio):.2f}]  "
      f"({int(np.nansum(ec_bio < 0))} negative)")""")

md(r"""## 3. Join to the SPARTAN HIPS ground truth (same join as fig08 → n≈259)

`MediaId → ExternalFilterId → SPARTAN HIPS FilterId`, then `EC_hips = Fabs / 10`.""")

code(r"""master = pd.DataFrame({"MediaId": media, "EC_regular": ec_cal0, "EC_smoke": ec_bio})
master = master.merge(meta[["MediaId", "ExternalFilterId"]], on="MediaId", how="left")

hips = pd.read_csv(SPARTAN, usecols=["Site", "FilterId", "Fabs"])
hips = hips[hips["Site"] == "ETAD"].rename(columns={"FilterId": "ExternalFilterId"})
master = master.merge(hips[["ExternalFilterId", "Fabs"]], on="ExternalFilterId", how="inner")
master["EC_hips"] = master["Fabs"] / MAC
master = master[master["EC_hips"].notna()].reset_index(drop=True)
print(f"ETAD filters joined to HIPS: n = {len(master)}")

# --- guarded cross-check: the EC the deployed general calibration actually REPORTED ---
try:
    def _root(p=Path.cwd()):
        for c in [p, *p.parents]:
            if (c / "AGENTS.md").exists() and (c / "research").exists():
                return c
        raise RuntimeError("repo root not found")
    sp = pd.read_pickle(_root() / "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl")
    rep = (sp[(sp["Site"] == "ETAD") & (sp["Parameter"] == "EC_ftir")]
           .groupby("FilterId")["Concentration"].first())
    master["EC_reported"] = master["ExternalFilterId"].map(rep)
    n_rep = int(master["EC_reported"].notna().sum())
    print(f"deployed general-cal EC (reported) matched for {n_rep}/{len(master)} filters "
          f"(cross-check only)")
except Exception as e:
    master["EC_reported"] = np.nan
    print("reported-EC cross-check skipped:", e)

master.to_csv("tables/etad_regular_vs_smoke_ec.csv", index=False)
print("wrote tables/etad_regular_vs_smoke_ec.csv")""")

md(r"""## 4. The crossplot — regular all-data vs. smoke-only, same axes, HIPS on x

If the smoke calibration *helps*, its panel should sit **closer to the 1:1 line** (slope → 1,
intercept → 0) and/or show a **smaller HIPS−EC gap** than the regular panel.""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
st_reg = crossplot(axes[0], master["EC_hips"], master["EC_regular"],
                   color="#1f77b4", title="Regular — all-data calibration (CAL-0)")
st_smk = crossplot(axes[1], master["EC_hips"], master["EC_smoke"],
                   color="#d62728", title="Smoke-only — biomass lot-251 calibration")
fig.suptitle(f"ETAD (Addis): does the smoke calibration move FTIR-EC toward HIPS?  (n={len(master)})",
             y=1.00, fontsize=13)
plt.tight_layout()
plt.savefig("figures/regular_vs_smoke_ec_vs_hips.png", dpi=140, bbox_inches="tight")
print("saved figures/regular_vs_smoke_ec_vs_hips.png"); plt.show()""")

md(r"""## 5. The verdict — closer to HIPS, or not?

Three yardsticks for "closer to HIPS ground truth":
- **slope → 1** (and **implied MAC = 10/slope → a physical ~6–10 m²/g**),
- **|median (HIPS − EC)| gap → 0** (does the calibration sit on the HIPS level?),
- **R²** (tightness — a tie-breaker, not the main point).""")

code(r"""def scorecard(name, ec):
    st = reg_stats(master["EC_hips"].values, ec.values)
    gap = float(np.nanmedian(master["EC_hips"].values - ec.values))
    return {"calibration": name, "n": st["n"], "slope": round(st["slope"], 2),
            "intercept": round(st["intercept"], 2), "R2": round(st["r2"], 3),
            "implied_MAC": round(MAC / st["slope"], 1) if st["slope"] > 0 else np.nan,
            "median_EC": round(float(np.nanmedian(ec.values)), 2),
            "median_HIPS_minus_EC": round(gap, 2),
            "dist_from_1to1(|slope-1|)": round(abs(st["slope"] - 1), 2)}

rows = [scorecard("Regular (all-data CAL-0)", master["EC_regular"]),
        scorecard("Smoke-only (biomass)",     master["EC_smoke"])]
if master["EC_reported"].notna().sum() >= 3:
    rows.append(scorecard("Deployed general (reported)", master["EC_reported"]))
score = pd.DataFrame(rows)
print(score.to_string(index=False)); print()
score.to_csv("tables/regular_vs_smoke_verdict.csv", index=False)

reg, smk = rows[0], rows[1]
closer_slope = abs(smk["slope"] - 1) < abs(reg["slope"] - 1)
closer_gap   = abs(smk["median_HIPS_minus_EC"]) < abs(reg["median_HIPS_minus_EC"])
tighter      = smk["R2"] > reg["R2"]
verdict = ("HELPS" if (closer_slope and closer_gap) else
           "MIXED" if (closer_slope or closer_gap) else "does NOT help")
print(f"→ Smoke calibration {verdict}: "
      f"slope {'closer to 1' if closer_slope else 'not closer to 1'} "
      f"({reg['slope']}→{smk['slope']}); "
      f"HIPS−EC gap {'smaller' if closer_gap else 'not smaller'} "
      f"({reg['median_HIPS_minus_EC']}→{smk['median_HIPS_minus_EC']}); "
      f"R² {'higher' if tighter else 'not higher'} ({reg['R2']}→{smk['R2']}).")
print("wrote tables/regular_vs_smoke_verdict.csv")""")

md(r"""### How to read this

- **Both calibrations run above the 1:1 line vs HIPS/10** (slope > 1) — FTIR-EC reads *higher* than
  the MAC=10 HIPS EC-equivalent. Two readings, exactly as in fig08: (a) the implied **MAC ≈ 10/slope**
  may be the thing that's off (a MAC nearer 6 would raise the HIPS EC-equivalent and pull slopes
  toward 1), and (b) the **relative** movement between the two panels is the calibration signal.
- **"Does smoke help" = does the biomass panel move toward the 1:1 / HIPS line** relative to the
  regular panel — smaller `|slope-1|` and smaller `|HIPS−EC|` gap. The scorecard answers it numerically.
- **Caveat (don't over-read the absolute level):** the regular model is the IMPROVE-based 906-sample
  PLS; the smoke model is the SPARTAN lot-251 biomass tool cal — different *training populations*, not
  only different *sample selection*. The clean, confound-free version is a **general lot-251 vs biomass
  lot-251** pair (same lot). When Mona/Sean's general lot-251 coefficients or the smoke classifier
  land, drop them in here — the harness is unit-for-unit ready.
- **MAC sweep:** re-run with `MAC = 6` to bound the 6-vs-10 question; every implied MAC and gap will
  shift, but the regular-vs-smoke *ordering* should not.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("regular_vs_smoke_ec_vs_hips.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote regular_vs_smoke_ec_vs_hips.ipynb")
