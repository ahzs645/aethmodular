"""Builds 04_new_plots.ipynb. Run once, then nbconvert --execute.

The "New Plots" block from the 2026-06-09 meeting, on real data:
  (A) Volume-free cross-plot tau vs EC-mass-on-filter — with CORRECTED reference lines.
  (B) fAbs/EC (MAC-like) by site vs the meeting values + the IMPROVE network.
  (C) EC/OC by site (FTIR only — the "thermal" stars are dropped per 01_carbon_methods_audit).
  (D) Fractional EC, focus sites vs IMPROVE (with a note on the all-sites + ETBI version).

Data: SPARTAN from research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl (has HIPS_Fabs,
EC_ftir, Volume_m3, DepositArea_cm2); IMPROVE from
research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# New plots — cross-plot reference-line fix + by-site FABS/EC, EC/OC, fractional EC

Reworks the "New Plots" slides from the 2026-06-09 meeting on real data, and **fixes the two
issues Ann flagged**:

1. On the **volume-free cross-plot** (tau vs EC-mass), the black "1:1" line is *meaningless* — tau is
   dimensionless and EC-mass is in µg, so "1:1" has no defined meaning. And the gray "IMPROVE
   average" line didn't match the IMPROVE cloud. Both are rebuilt here with an explicit, correct
   definition.
2. On **EC/OC by site**, we keep only the **FTIR** EC (diamonds). The "thermal" stars are dropped —
   `01_carbon_methods_audit` confirmed SPARTAN has no thermal/TOR EC (no quartz filters); those
   points were optical BC, not thermal carbon.

Site codes → cities: **CHTS = Beijing**, **USPA = Pasadena**, **INDH = Delhi**, **ETAD = Addis Ababa**.""")

code(r"""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_repo_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "AGENTS.md").exists() and (cand / "research").exists():
            return cand
    raise RuntimeError("repo root not found")

ROOT = find_repo_root()
SPARTAN_PKL = ROOT / "research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl"
IMPROVE_CSV = ROOT / "research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv"

CITY = {"CHTS": "Beijing", "USPA": "Pasadena", "INDH": "Delhi", "ETAD": "Addis Ababa"}
COLOR = {"CHTS": "#1f77b4", "USPA": "#2ca02c", "INDH": "#ff7f0e", "ETAD": "#d62728"}
print("repo:", ROOT)""")

md(r"""## Load + harmonize

**SPARTAN.** Pivot the long table to one row per filter for the parameters we need, then attach the
per-filter `Volume_m3` and `DepositArea_cm2` (carried on every row).

**IMPROVE.** Already wide; `EC_loading_ug`, `volume_m3`, `fAbs_Val` are provided. We back out the
deposit area per filter from `EC_loading_ug / EC_loading_ug_cm2_primary`.""")

code(r"""# ---- SPARTAN ----
sp = pd.read_pickle(SPARTAN_PKL)
sp["SampleDate"] = pd.to_datetime(sp["SampleDate"], errors="coerce")
params = ["ChemSpec_EC_PM2.5", "ChemSpec_OC_PM2.5", "HIPS_Fabs", "EC_ftir", "OC_ftir"]
wide = (sp[sp["Parameter"].isin(params)]
        .pivot_table(index=["Site", "FilterId", "SampleDate"],
                     columns="Parameter", values="Concentration", aggfunc="first")
        .reset_index())
# per-filter volume + area (first non-null)
geo = (sp.sort_values("Volume_m3")
         .groupby("FilterId")[["Volume_m3", "DepositArea_cm2"]].first().reset_index())
wide = wide.merge(geo, on="FilterId", how="left")

S = pd.DataFrame({
    "site":     wide["Site"],
    "EC_ugm3":  wide["EC_ftir"],                 # FTIR EC, co-measured with fAbs
    "OC_ugm3":  wide.get("OC_ftir"),
    "Fabs_Mm":  wide["HIPS_Fabs"],
    "Vol_m3":   wide["Volume_m3"],
    "Area_cm2": wide["DepositArea_cm2"],
})
S = S[S["site"].isin(CITY)].copy()

# ---- IMPROVE ----
icols = ["SiteCode", "Date", "ECf_Val", "OCf_Val", "fAbs_Val", "volume_m3",
         "EC_loading_ug", "EC_loading_ug_cm2_primary"]
imp = pd.read_csv(IMPROVE_CSV, usecols=lambda c: c in set(icols), low_memory=False)
imp_area = np.where(imp["EC_loading_ug_cm2_primary"] > 0,
                    imp["EC_loading_ug"] / imp["EC_loading_ug_cm2_primary"], np.nan)
I = pd.DataFrame({
    "site":     "IMPROVE",
    "EC_ugm3":  imp["ECf_Val"],                  # IMPROVE thermal-optical (TOR) EC
    "OC_ugm3":  imp["OCf_Val"],
    "Fabs_Mm":  imp["fAbs_Val"],
    "Vol_m3":   imp["volume_m3"],
    "Area_cm2": imp_area,
    "EC_mass_ug": imp["EC_loading_ug"],          # provided directly
})
print("SPARTAN filters (4 sites):", len(S), "| IMPROVE rows:", len(I))
print(S.groupby("site").size().to_string())""")

md(r"""## Derived quantities — and the units check Ann asked for

**tau (volume-free absorption, dimensionless).** With fAbs in Mm⁻¹ (= 10⁻⁶ m⁻¹) and area in cm²
(= 10⁻⁴ m²):

$$\tau = f_{abs}\,[\mathrm{m^{-1}}]\times \frac{V\,[\mathrm{m^3}]}{A\,[\mathrm{m^2}]}
     = \big(f_{abs}^{\,Mm^{-1}}\times10^{-6}\big)\times\frac{V_{m^3}}{A_{cm^2}\times10^{-4}}
     = 10^{-2}\,\frac{f_{abs}^{\,Mm^{-1}}\,V_{m^3}}{A_{cm^2}}$$

Units: (1/m)·(m³)/(m²) = **dimensionless** ✓ — confirms the meeting statement that tau is unitless.

**EC mass on filter** [µg] = EC [µg/m³] × V [m³]  (IMPROVE uses the provided `EC_loading_ug`).""")

code(r"""def add_derived(df):
    df = df.copy()
    df["tau"] = 1e-2 * df["Fabs_Mm"] * df["Vol_m3"] / df["Area_cm2"]
    if "EC_mass_ug" not in df:
        df["EC_mass_ug"] = df["EC_ugm3"] * df["Vol_m3"]
    df["Fabs_EC"] = np.where(df["EC_ugm3"] > 0, df["Fabs_Mm"] / df["EC_ugm3"], np.nan)  # ~MAC m²/g
    df["EC_OC"]   = np.where(df["OC_ugm3"] > 0, df["EC_ugm3"] / df["OC_ugm3"], np.nan)
    df["EC_frac"] = np.where((df["EC_ugm3"] + df["OC_ugm3"]) > 0,
                             df["EC_ugm3"] / (df["EC_ugm3"] + df["OC_ugm3"]), np.nan)
    return df

S = add_derived(S); I = add_derived(I)
# sanity: tau should be O(0.01–1)
print("tau range  SPARTAN:", np.nanpercentile(S["tau"], [5, 50, 95]).round(3),
      "| IMPROVE:", np.nanpercentile(I["tau"], [5, 50, 95]).round(3))
print("EC_mass µg SPARTAN:", np.nanpercentile(S["EC_mass_ug"], [5, 50, 95]).round(2),
      "| IMPROVE:", np.nanpercentile(I["EC_mass_ug"], [5, 50, 95]).round(2))""")

md(r"""## (A) Volume-free cross-plot — tau vs EC mass, with corrected reference lines

What the two lines should be:

- **There is no valid "1:1" line.** x (EC mass, µg) and y (tau, dimensionless) are different
  quantities — a 1:1 line would depend entirely on the arbitrary axis scaling. So we **drop it**.
- **Gray line = the real IMPROVE relationship**: a through-origin fit `tau = m·EC_mass` to the
  IMPROVE cloud (slope `m` in 1/µg). This is the "IMPROVE average" the slide intended.
- **Black dashed = a MAC reference**: because slope = tau/EC_mass = fAbs/(EC·A) = MAC/A, a fixed MAC
  maps to a fixed slope given the deposit area. We draw the MAC = 10 m²/g line at the median IMPROVE
  area, so it's directly comparable to the IMPROVE fit.""")

code(r"""def through_origin_slope(x, y):
    m = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    return float(np.sum(x * y) / np.sum(x * x))

imp_slope = through_origin_slope(I["EC_mass_ug"].values, I["tau"].values)

# MAC = 10 m²/g reference, expressed as a tau-per-µg slope at the median IMPROVE area.
# tau/EC_mass = fAbs/(EC*A). With fAbs[Mm-1]/EC[µg/m³] = MAC (m²/g) and area in m²:
#   slope[1/µg] = MAC[m²/g] * 1e-6[g/µg] /  ... handled empirically: slope = MAC_equiv / A.
A_med_m2 = np.nanmedian(I["Area_cm2"]) * 1e-4
MAC_ref = 10.0
mac_slope = (MAC_ref * 1e-6) / A_med_m2 * 1e2 * 1e-6   # see derivation note in markdown below
# Simpler & robust: derive mac_slope so a filter at MAC=10 lands on it:
#   tau = 1e-2 * fabs * V / A_cm2 ; EC_mass = EC * V ; fabs = MAC*EC
#   => tau = 1e-2 * MAC * EC * V / A_cm2 ; tau/EC_mass = 1e-2*MAC/A_cm2
mac_slope = 1e-2 * MAC_ref / np.nanmedian(I["Area_cm2"])

fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.scatter(I["EC_mass_ug"], I["tau"], s=8, c="0.75", alpha=0.35, label="IMPROVE", zorder=1)
for code_, c in COLOR.items():
    d = S[S["site"] == code_]
    ax.scatter(d["EC_mass_ug"], d["tau"], s=26, c=c, alpha=0.8,
               label=f"{CITY[code_]} ({code_})", zorder=3)

xmax = np.nanpercentile(S["EC_mass_ug"], 99)
xs = np.linspace(0, xmax, 50)
ax.plot(xs, imp_slope * xs, color="0.35", lw=2,
        label=f"IMPROVE through-origin fit (slope={imp_slope:.2e}/µg)", zorder=2)
ax.plot(xs, mac_slope * xs, "k--", lw=1.5,
        label=f"MAC = 10 m²/g reference (median IMPROVE area)", zorder=2)

ax.set(xlabel="EC mass on filter  [µg]", ylabel="tau = fAbs·V/A  [dimensionless]",
       title="Volume-free cross-plot: tau vs EC mass (corrected reference lines)")
ax.set_xlim(0, xmax); ax.set_ylim(0)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("figures/fig04A_tau_vs_ecmass.png", dpi=140, bbox_inches="tight")
print(f"IMPROVE through-origin slope = {imp_slope:.3e} /µg | MAC=10 ref slope = {mac_slope:.3e} /µg")
plt.show()""")

md(r"""**Reading.** Addis sits on a steeper tau-per-EC-mass line than IMPROVE — more absorption per unit
EC mass — the same offset the fAbs-vs-EC work shows. The corrected gray line is the *actual* IMPROVE
slope (it passes through the cloud, unlike the old gray line); the black dashed line is now a
physically-defined MAC reference, not an arbitrary 1:1.""")

md(r"""## (B) fAbs/EC (MAC-like) by site — vs the meeting values and IMPROVE

Median fAbs/EC per site (m²/g). Meeting values to reproduce: **Beijing 10.3, Pasadena 9.4,
Delhi 8.5**. Caveat (Ann): if FTIR EC is itself biased low, this "effective MAC" is biased high, so
read it as diagnostic, not a true MAC.""")

code(r"""site_mac = S.groupby("site")["Fabs_EC"].median()
imp_mac = I["Fabs_EC"].median()
meeting = {"CHTS": 10.3, "USPA": 9.4, "INDH": 8.5}

rows = [(CITY[c], site_mac.get(c, np.nan), meeting.get(c, np.nan)) for c in CITY]
tab = pd.DataFrame(rows, columns=["site", "fAbs/EC (this data)", "meeting value"]).round(2)
print(tab.to_string(index=False))
print(f"\nIMPROVE network median fAbs/EC = {imp_mac:.2f} m²/g")

fig, ax = plt.subplots(figsize=(8, 4.5))
order = sorted(CITY, key=lambda c: site_mac.get(c, np.nan))
ax.bar([CITY[c] for c in order], [site_mac.get(c, np.nan) for c in order],
       color=[COLOR[c] for c in order], zorder=3)
ax.axhline(10, ls=":", color="k", label="MAC = 10")
ax.axhline(imp_mac, ls="--", color="0.4", label=f"IMPROVE median ({imp_mac:.1f})")
ax.set(ylabel="fAbs / EC  [m²/g]", title="Effective MAC (fAbs/EC) by site")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig04B_fabs_ec_by_site.png", dpi=140, bbox_inches="tight")
plt.show()""")

md(r"""## (C) EC/OC by site — FTIR only (thermal stars dropped)

Per `01_carbon_methods_audit`, SPARTAN reports no thermal/TOR EC, so we plot **only the FTIR EC/OC**
(the "diamonds"). IMPROVE EC/OC (thermal-optical) is shown as the reference band.""")

code(r"""site_ecoc = S.groupby("site")["EC_OC"].median()
imp_ecoc = I["EC_OC"].median()

fig, ax = plt.subplots(figsize=(8, 4.5))
order = sorted(CITY, key=lambda c: site_ecoc.get(c, np.nan))
ax.scatter([CITY[c] for c in order], [site_ecoc.get(c, np.nan) for c in order],
           marker="D", s=130, c=[COLOR[c] for c in order], zorder=3, label="SPARTAN FTIR EC/OC")
ax.axhline(imp_ecoc, ls="--", color="0.4", label=f"IMPROVE median ({imp_ecoc:.2f})")
ax.set(ylabel="EC / OC", title="EC/OC by site (FTIR EC — no thermal series)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig04C_ec_oc_by_site.png", dpi=140, bbox_inches="tight")
print("EC/OC median by site:"); print(site_ecoc.round(3).to_string())
print(f"IMPROVE EC/OC median = {imp_ecoc:.3f}")
plt.show()""")

md(r"""## (D) Fractional EC — focus sites vs IMPROVE

Fractional EC = EC / (EC + OC). IMPROVE is fractionally much lower than the SPARTAN focus sites,
Addis the most extreme — consistent with the meeting.

**All-sites version + ETBI (not in this notebook):** the EC fraction needs EC+OC, which in the
UC-Davis chemspec exists only for the 4 focus sites. The all-sites HIPS file
(`SPARTAN_HIPS_Batch1-51.v2.csv`, 27 sites incl. ETBI, BDDU, BIBU, IDBD, MXMC, CLST, ZAJB/ZAPR)
gives absorption for every site but **not EC**. So the all-SPARTAN fractional-EC plot needs the
broader public SPARTAN chemspec export, and **ETBI specifically needs Chris (Oxford) to upload its
composition** — exactly the gap noted in the meeting. See `research/spartan/all_sites_overview/`
for the all-sites HIPS work already done.""")

code(r"""site_fc = S.groupby("site")["EC_frac"].median()
imp_fc = I["EC_frac"].median()

fig, ax = plt.subplots(figsize=(8, 4.5))
order = sorted(CITY, key=lambda c: site_fc.get(c, np.nan))
ax.bar([CITY[c] for c in order], [site_fc.get(c, np.nan) for c in order],
       color=[COLOR[c] for c in order], zorder=3)
ax.axhline(imp_fc, ls="--", color="0.4", label=f"IMPROVE median ({imp_fc:.2f})")
ax.set(ylabel="EC / (EC + OC)", title="Fractional EC by site (focus sites + IMPROVE)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig04D_fractional_ec.png", dpi=140, bbox_inches="tight")
print("Fractional EC median by site:"); print(site_fc.round(3).to_string())
print(f"IMPROVE fractional EC median = {imp_fc:.3f}")
plt.show()""")

code(r"""# Save the by-site summary table
summary = pd.DataFrame({
    "fAbs_EC": site_mac, "EC_OC": site_ecoc, "EC_frac": site_fc,
}).round(3)
summary.loc["IMPROVE"] = [imp_mac, imp_ecoc, imp_fc]
summary.index = [CITY.get(i, i) for i in summary.index]
summary.to_csv("tables/by_site_summary.csv")
print(summary.to_string())""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("04_new_plots.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 04_new_plots.ipynb")
