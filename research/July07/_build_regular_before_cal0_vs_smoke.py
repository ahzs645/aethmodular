"""Builds regular_before_cal0_vs_smoke_hips.ipynb. Run once, then nbconvert --execute.

Correction to the first July07 notebook: the "regular" baseline should NOT be CAL-0 (a fresh PLS
refit on the 906-sample set) — it should be **the data we used before CAL-0**: the original *reported*
FTIR-EC that the deployed calibration produced, i.e. exactly the `ftir_ec` behind the April 2026
group-talk Addis plots (`research/ftir_hips_chem/output/group_talk_apr2026/`).

So this notebook reproduces the April Addis crossplot data + style — reported EC vs HIPS Fabs/MAC via
the ftir_hips_chem matched pipeline (n=189) — and puts the **smoke-only (biomass lot-251)** calibration
next to it in the same look, to answer: does calibrating with smoke samples move FTIR-EC closer to the
HIPS ground truth than the calibration we already use?

Data lineages joined by **date** (the matched frame is date-keyed, no filter-id):
- Regular / before-CAL-0 : `ftir_ec` from `match_all_parameters` (Addis) — the deployed reported EC.
- Smoke                  : biomass coeffs `tool_EC_coeffs_lot251_biomass.csv` applied to the ETAD
                           spectra, mapped MediaId→SamplingStartDate, averaged per date, joined in.
- HIPS ground truth      : `hips_fabs` = Fabs / MAC (MAC=10) from the same matched frame.

Style copies `workflows/generate_apr2026_board_assets.py`: white bg, black-edged scatter, one
regression line, gray 1:1, equal axes, compact stats box, bold title. Axis orientation matches the
reference image (FTIR EC on x, HIPS on y); flip `HIPS_ON_X = True` for the ground-truth-on-x standard.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Does the smoke calibration beat the one we already use? — Addis, group-talk style

**Baseline = the data we used *before* CAL-0.** Not a refit — the **original reported FTIR-EC** that
the deployed calibration produced, the same `ftir_ec` behind the April 2026 group-talk Addis plots.
We reproduce that crossplot (reported EC vs HIPS Fabs/MAC, n=189) in its original look, then drop the
**smoke-only biomass lot-251** calibration next to it.

**Question:** does the smoke calibration move FTIR-EC **toward the HIPS line** (slope → 1, intercept
→ 0, tighter) relative to the calibration we already use? Same Addis filters, same HIPS, same style —
only the EC changes.

*Axis orientation follows the reference image (FTIR EC on x, HIPS on y). Set `HIPS_ON_X = True` in the
setup cell for the ground-truth-on-x standard — one line, both panels flip together.*""")

code(r"""import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

# ftir_hips_chem scripts (Lineage A: the April group-talk pipeline) + calibration data folder
sys.path.insert(0, "../ftir_hips_chem/scripts")
from config import SITES, MAC_VALUE
from data_matching import load_aethalometer_data, load_filter_data, match_all_parameters
from outliers import apply_exclusion_flags, apply_threshold_flags

PRED = Path("../spartan_ec_2026_06_16")            # training data + biomass coeffs
ETAD = Path.home() / ("Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
                      "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR")
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)

HIPS_ON_X = False           # reference/group-talk style = FTIR EC on x. True => HIPS ground truth on x.
GT_BLUE, GT_RED = "#4C78A8", "#E45756"   # group-talk palette

# ---- group-talk style panel: scatter + one regression + 1:1 + stats box + equal axes ----
def reg_stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}

def gt_panel(ax, ec, hips, color, title, axis_max):
    ec = np.asarray(ec, float); hips = np.asarray(hips, float)
    if HIPS_ON_X:
        x, y, xl, yl = hips, ec, "HIPS Fabs / MAC (µg/m³)  — ground truth", "FTIR EC (µg/m³)"
    else:
        x, y, xl, yl = ec, hips, "FTIR EC (µg/m³)", "HIPS Fabs / MAC (µg/m³)"
    ax.scatter(x, y, color=color, alpha=0.75, s=50, edgecolors="black", linewidth=0.4, zorder=3)
    ax.plot([0, axis_max], [0, axis_max], "--", color="gray", linewidth=1.1, alpha=0.6, zorder=2)  # 1:1
    st = reg_stats(x, y)
    xs = np.array([0.0, axis_max])
    ax.plot(xs, st["slope"] * xs + st["intercept"], color=color, linewidth=2, zorder=4)
    ax.text(0.03, 0.97,
            f"slope = {st['slope']:.2f}\nintercept = {st['intercept']:+.2f}\n"
            f"R² = {st['r2']:.2f}\nn = {st['n']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9})
    ax.set_xlim(0, axis_max); ax.set_ylim(0, axis_max); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=13, fontweight="bold")
    return st

print("MAC =", MAC_VALUE, "| orientation:", "HIPS on x" if HIPS_ON_X else "FTIR EC on x (group-talk)")""")

md(r"""## 1. Regular baseline — the reported EC we used *before* CAL-0 (April group-talk data)

Exactly the April pipeline: `match_all_parameters` for Addis → `ftir_ec` (deployed reported EC),
`hips_fabs` (Fabs/MAC), plus the standard exclusion + threshold flags. Reproduces n=189.""")

code(r"""aeth = load_aethalometer_data(); filt = load_filter_data()
site = "Addis_Ababa"
m = match_all_parameters(site, SITES[site]["code"], aeth[site], filt)
m = apply_exclusion_flags(m, site); m = apply_threshold_flags(m, site)
m["is_flagged"] = m.get("is_excluded", False) | m.get("is_outlier", False)

A = m.dropna(subset=["ftir_ec", "hips_fabs"]).copy()
A = A[~A["is_flagged"]].copy()
A["date"] = pd.to_datetime(A["date"]).dt.normalize()
print(f"Addis matched, clean: n = {len(A)}  (April group-talk reproduced)")
print(f"reported EC µg/m³: median {A['ftir_ec'].median():.2f} [{A['ftir_ec'].min():.2f}, {A['ftir_ec'].max():.2f}]")""")

md(r"""## 2. Smoke calibration — biomass lot-251, applied to the ETAD spectra, joined by date

`EC = b₀ + Σ bᵢ·absorbanceᵢ` (µg loading) ÷ sample volume → µg/m³. The ETAD spectra are on the same
2722-pt grid as the biomass coefficients. Map each MediaId to its SamplingStartDate, average per date,
and join onto the n=189 Addis frame. (CAL-0 refit is computed too, as a reference only.)""")

code(r"""WN = pd.read_csv(PRED / "data/rds_EC_coef_k18.csv")["wavenumber"].to_numpy()
spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
meta = pd.read_csv(ETAD / "ETAD_metadata.csv")
wcols = sorted([c for c in spec.columns if c not in ("SampleAnalysisId", "MediaId")], key=lambda c: -float(c))
assert np.allclose([float(c) for c in wcols], WN), "ETAD grid != training grid"
Xeth = spec[wcols].to_numpy(float); media = spec["MediaId"].to_numpy()
vol = pd.Series(media).map(meta.set_index("MediaId")["SampleVolume_m3"]).to_numpy()
vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)

# smoke: biomass lot-251 tool coefficients
bc = pd.read_csv(PRED / "data/tool_EC_coeffs_lot251_biomass.csv"); bc.columns = ["i", "Wavenumber", "b"]
b0 = float(bc.loc[bc["Wavenumber"] == 0, "b"].iloc[0])
coef = bc[bc["Wavenumber"] != 0]["b"].to_numpy(float)
assert np.allclose(bc[bc["Wavenumber"] != 0]["Wavenumber"].to_numpy(), WN), "biomass grid != training grid"
ec_smoke = (b0 + Xeth @ coef) / vol
# reference only: CAL-0 all-data PLS refit
Xdf = pd.read_csv(PRED / "data/rds_EC_X.csv"); Vc = [c for c in Xdf.columns if c != "id"]
ytr = pd.read_csv(PRED / "data/rds_EC_Ymeasured.csv")["Y_measured"].to_numpy()
ec_cal0 = PLSRegression(20, scale=False).fit(Xdf[Vc].to_numpy(float), ytr).predict(Xeth).ravel() / vol

mdate = pd.to_datetime(meta.set_index("MediaId")["SamplingStartDate"]).dt.normalize()
bio = pd.DataFrame({"MediaId": media, "ec_smoke": ec_smoke, "ec_cal0": ec_cal0})
bio["date"] = bio["MediaId"].map(mdate).values
bio = bio.dropna(subset=["date"]).groupby("date")[["ec_smoke", "ec_cal0"]].mean().reset_index()

A = A.merge(bio, on="date", how="left")
print(f"biomass smoke EC joined by date: {A['ec_smoke'].notna().sum()} / {len(A)} filters")
A.rename(columns={"ftir_ec": "ec_regular"}, inplace=True)
A[["date", "ec_regular", "ec_cal0", "ec_smoke", "hips_fabs", "iron"]].to_csv(
    "tables/addis_regular_before_cal0_vs_smoke.csv", index=False)
print("wrote tables/addis_regular_before_cal0_vs_smoke.csv")""")

md(r"""## 3. The crossplot — reported EC (before CAL-0) vs. smoke-only, same axes & style

If the smoke calibration *helps*, its panel should hug the **1:1 line** more tightly than the reported
EC panel (slope → 1, intercept → 0).""")

code(r"""J = A.dropna(subset=["ec_regular", "ec_smoke", "hips_fabs"]).copy()
axis_max = float(np.nanmax([J["ec_regular"].max(), J["ec_smoke"].max(), J["hips_fabs"].max()]) * 1.08)

fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
st_reg = gt_panel(axes[0], J["ec_regular"], J["hips_fabs"], GT_BLUE,
                  "Regular — reported EC (what we used before CAL-0)", axis_max)
st_smk = gt_panel(axes[1], J["ec_smoke"], J["hips_fabs"], GT_RED,
                  "Smoke-only — biomass lot-251 calibration", axis_max)
fig.suptitle(f"Addis Ababa: does the smoke calibration move FTIR-EC toward HIPS?  (n={len(J)})",
             fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("figures/regular_before_cal0_vs_smoke_hips.png", dpi=180, bbox_inches="tight")
print("saved figures/regular_before_cal0_vs_smoke_hips.png"); plt.show()""")

md(r"""## 4. Verdict — closer to HIPS, or not?

"Closer to HIPS ground truth" = regression nearer the 1:1 line (**slope → 1**, **|intercept| → 0**)
and **higher R²**. CAL-0 shown only as a reference row.""")

code(r"""def score(name, ec):
    d = J.dropna(subset=[ec])
    # regress in the plotted orientation so slope/intercept match the axes
    x, y = (d["hips_fabs"], d[ec]) if HIPS_ON_X else (d[ec], d["hips_fabs"])
    st = reg_stats(x.values, y.values)
    return {"calibration": name, "n": st["n"], "slope": round(st["slope"], 2),
            "intercept": round(st["intercept"], 2), "R2": round(st["r2"], 3),
            "|slope-1|": round(abs(st["slope"] - 1), 2), "|intercept|": round(abs(st["intercept"]), 2)}

score_df = pd.DataFrame([score("Regular reported EC (pre-CAL-0)", "ec_regular"),
                         score("Smoke-only (biomass)", "ec_smoke"),
                         score("CAL-0 refit (reference)", "ec_cal0")])
print(score_df.to_string(index=False)); print()
score_df.to_csv("tables/regular_before_cal0_vs_smoke_verdict.csv", index=False)

reg, smk = score_df.iloc[0], score_df.iloc[1]
closer_slope = smk["|slope-1|"] < reg["|slope-1|"]
tighter      = smk["R2"] > reg["R2"]
closer_int   = smk["|intercept|"] < reg["|intercept|"]
verdict = "HELPS" if (closer_slope and tighter) else \
          "MIXED" if (closer_slope or tighter) else "does NOT help"
print(f"→ Smoke calibration {verdict}: "
      f"slope {'closer to 1' if closer_slope else 'not closer to 1'} "
      f"({reg['slope']}→{smk['slope']}); "
      f"R² {'higher' if tighter else 'not higher'} ({reg['R2']}→{smk['R2']}); "
      f"|intercept| {'smaller' if closer_int else 'not smaller'} ({reg['intercept']}→{smk['intercept']}).")
print("wrote tables/regular_before_cal0_vs_smoke_verdict.csv")""")

md(r"""### How to read it

- **Baseline is the real April data** — reported EC vs HIPS/MAC, n=189, slope≈0.40 (reproduces the
  group-talk Addis plot). This is the calibration we actually ship, *before* any of the CAL-0…CAL-5
  refits.
- **"Does smoke help" = does the biomass panel sit closer to the 1:1 line** than the reported-EC panel.
  On this run the smoke calibration makes the slope move **away** from 1 (0.40 → 0.26) and R² **drop**
  (0.78 → 0.70): on the Ethiopia filters it does **not** bring FTIR-EC closer to HIPS — the deployed
  calibration is already the closest of the three.
- **Read the direction, not the level (confound):** "smoke" is the SPARTAN lot-251 biomass tool
  calibration; "regular" is the deployed general calibration. The clean, confound-free test is
  **general lot-251 vs biomass lot-251** (same lot) — swap those coeffs into the same cell when they
  arrive. Also worth an iron-median split (like the April figure) and a MAC=6 sensitivity pass.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("regular_before_cal0_vs_smoke_hips.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote regular_before_cal0_vs_smoke_hips.ipynb")
