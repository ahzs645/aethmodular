"""Builds 02_biomass_calibration_comparison.ipynb. Run once, then nbconvert --execute.

Top-priority action item: compare FTIR-EC from the general (all-samples) lot-251 calibration
vs a biomass-burning-only lot-251 calibration for Addis. The biomass-only calibration must be
produced in the FTIR calibration tool (with Mona) and the per-filter EC predictions pasted into
the INPUT cell below — that part can't run here. Everything else (the real lot-251 baseline vs
HIPS, and the comparison/plotting harness) runs now on repo data.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Biomass-burning calibration vs general calibration — Addis FTIR EC

**The experiment (2026-06-09 meeting, top priority).** Our original FTIR-EC paper says smoke samples
should be calibrated with *smoke* samples, not a general calibration — but SPARTAN uses one general
calibration for everything. Hypothesis: a **biomass-burning-only** calibration predicts **higher**
EC for Addis (recovering the char that a wood-smoke-poor general calibration misses), moving FTIR-EC
toward the independent HIPS absorption.

**Workflow**
1. In the FTIR calibration tool, select the lot with the most Addis samples — **lot 251** (confirmed
   below; Ann guessed 256, but the data says 251; add lot 248 / PCA-similar lots if too few smoke
   samples).
2. Build a calibration from **biomass-burning samples only** (Chelsea's guidance: screen outliers,
   choose the number of PLS factors).
3. Apply both calibrations (general lot-251 and biomass-only) to the Addis spectra → two EC
   predictions per filter (Mona helps with multiplying spectra × calibration correctly).
4. **Compare** the two EC predictions, and each against HIPS. Expectation: biomass EC > general EC,
   and closer to HIPS.

**What runs here vs what's blocked:** the tool/spectra steps (1–3) happen with Mona. This notebook
provides (a) the real lot-251 baseline — current general-calibration FTIR-EC vs HIPS — and (b) the
comparison + plotting harness, with a clearly-marked INPUT cell for the biomass-EC predictions.""")

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
MAC = 10.0   # m²/g, SPARTAN HIPS convention — to put fAbs on a BC-equivalent µg/m³ axis
print("repo:", ROOT)""")

md(r"""## 1. Confirm the lot and build the real baseline

Pivot the Addis (ETAD) filters and confirm the lot distribution, then assemble the lot-251 set with
its **current general-calibration FTIR-EC**, **OC**, and the independent **HIPS fAbs**.""")

code(r"""sp = pd.read_pickle(SPARTAN_PKL)
et = sp[sp["Site"] == "ETAD"]
wide = (et[et["Parameter"].isin(["EC_ftir", "OC_ftir", "HIPS_Fabs"])]
        .pivot_table(index=["FilterId", "LotId", "SampleDate"],
                     columns="Parameter", values="Concentration", aggfunc="first")
        .reset_index())

print("Addis FTIR-EC samples by lot:")
print(et[et["Parameter"].eq("EC_ftir")].groupby("LotId").size().sort_values(ascending=False).to_string())

LOT = 251
base = wide[wide["LotId"] == LOT].dropna(subset=["EC_ftir", "HIPS_Fabs"]).copy()
base["fabs_bc_ugm3"] = base["HIPS_Fabs"] / MAC          # HIPS as BC-equiv µg/m³
base["EC_general"]   = base["EC_ftir"]                  # current general-calibration EC
print(f"\nLot {LOT}: {len(base)} Addis filters with EC + HIPS")
base[["EC_general", "OC_ftir", "HIPS_Fabs", "fabs_bc_ugm3"]].describe().round(2)""")

code(r"""# Baseline relationship: current general-cal FTIR-EC vs HIPS BC-equivalent
x = base["EC_general"].values
y = base["fabs_bc_ugm3"].values
slope, intercept = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]
gap = np.median(y - x)
print(f"HIPS(BC-eq) = {slope:.2f}·EC_general + {intercept:.2f}   r={r:.3f}")
print(f"median (HIPS_BC − EC_general) = {gap:.2f} µg/m³  "
      f"(>0 ⇒ HIPS sees more absorber than FTIR EC ⇒ room for biomass cal to close the gap)")""")

md(r"""## 2. ⬇️ INPUT — biomass-burning calibration results from the tool (with Mona)

Two things come from the FTIR calibration tool / Mona and must be pasted here:

- `BIOMASS_FILTER_IDS` — the FilterIds flagged biomass-burning by the FTIR algorithm (the labels
  behind the "biomass burning" data in the tool).
- `EC_BIOMASS` — per-FilterId EC predicted by the **biomass-only lot-251 calibration**.

Until then, set `PLACEHOLDER = True` to exercise the harness with an *illustrative* biomass
prediction (general EC nudged up for biomass-flagged filters) so the plots render. **This is not a
result** — replace with the tool output and set `PLACEHOLDER = False`.""")

code(r"""PLACEHOLDER = True   # <-- set False once EC_BIOMASS is pasted from the tool

if PLACEHOLDER:
    rng_ids = base["FilterId"].tolist()
    # Illustrative: flag the higher-OC half as "biomass" and bump their EC ~+18% (char recovery).
    med_oc = base["OC_ftir"].median()
    BIOMASS_FILTER_IDS = base.loc[base["OC_ftir"] >= med_oc, "FilterId"].tolist()
    EC_BIOMASS = base.set_index("FilterId")["EC_general"].copy()
    EC_BIOMASS.loc[BIOMASS_FILTER_IDS] *= 1.18
    EC_BIOMASS = EC_BIOMASS.to_dict()
    print("⚠️  PLACEHOLDER biomass EC in use — illustrative only, NOT a calibration result.")
else:
    # --- paste real values from the tool below ---
    BIOMASS_FILTER_IDS = [
        # "ETAD-0123", ...
    ]
    EC_BIOMASS = {
        # "ETAD-0123": 7.4,   # µg/m³ from the biomass-only lot-251 calibration
    }

print(f"biomass-flagged filters: {len(BIOMASS_FILTER_IDS)} | biomass-EC values: {len(EC_BIOMASS)}")""")

md(r"""## 3. Comparison harness — general vs biomass calibration, and vs HIPS""")

code(r"""d = base.copy()
d["EC_biomass"] = d["FilterId"].map(EC_BIOMASS)
d["is_biomass"] = d["FilterId"].isin(BIOMASS_FILTER_IDS)
cmp = d.dropna(subset=["EC_biomass"]).copy()
cmp["delta"] = cmp["EC_biomass"] - cmp["EC_general"]
cmp["ratio"] = cmp["EC_biomass"] / cmp["EC_general"]

print(f"n compared = {len(cmp)}  ({cmp['is_biomass'].sum()} biomass-flagged)")
print(f"median EC_general = {cmp['EC_general'].median():.2f}  "
      f"median EC_biomass = {cmp['EC_biomass'].median():.2f}  "
      f"median Δ = {cmp['delta'].median():+.2f} µg/m³")
print(f"median ratio (biomass/general) = {cmp['ratio'].median():.3f}  "
      f"(expect > 1 for the hypothesis)")
print(f"\nGap to HIPS(BC-eq):")
print(f"  general: median |HIPS − EC| = {(cmp['fabs_bc_ugm3'] - cmp['EC_general']).median():+.2f}")
print(f"  biomass: median |HIPS − EC| = {(cmp['fabs_bc_ugm3'] - cmp['EC_biomass']).median():+.2f}"
      "   (smaller ⇒ biomass cal moves FTIR toward HIPS)")""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
ax.scatter(cmp["EC_general"], cmp["EC_biomass"], c=cmp["is_biomass"].map({True:"#d62728",False:"0.6"}),
           s=30, zorder=3)
lim = max(cmp["EC_general"].max(), cmp["EC_biomass"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 (no change)")
ax.set(xlabel="EC — general lot-251 cal [µg/m³]", ylabel="EC — biomass-only cal [µg/m³]",
       title="Biomass vs general EC")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(cmp["delta"], bins=20, color="#d62728", alpha=0.8, zorder=3)
ax.axvline(0, color="k", lw=1); ax.axvline(cmp["delta"].median(), color="navy", ls="--",
           label=f"median Δ = {cmp['delta'].median():+.2f}")
ax.set(xlabel="EC_biomass − EC_general [µg/m³]", ylabel="filters", title="EC change (expect > 0)")
ax.legend(fontsize=8)

ax = axes[2]
ax.scatter(cmp["EC_general"], cmp["fabs_bc_ugm3"], s=28, c="0.6", label="general", zorder=3)
ax.scatter(cmp["EC_biomass"], cmp["fabs_bc_ugm3"], s=28, c="#d62728", label="biomass", zorder=3)
lim2 = max(cmp[["EC_general","EC_biomass"]].max().max(), cmp["fabs_bc_ugm3"].max()) * 1.05
ax.plot([0, lim2], [0, lim2], "k--", lw=1, label="1:1 (EC = HIPS BC-eq)")
ax.set(xlabel="FTIR EC [µg/m³]", ylabel="HIPS BC-equivalent [µg/m³]",
       title="Each calibration vs HIPS")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig02_biomass_vs_general.png", dpi=140, bbox_inches="tight")
tag = " (PLACEHOLDER)" if PLACEHOLDER else ""
print("saved figures/fig02_biomass_vs_general.png" + tag)
plt.show()""")

md(r"""## 4. Tie-in — OC/EC and the functional-group calibration (Satoshi's point)

The FTIR calibration leans on functional groups; if Addis's OC/EC is far from the calibration
samples, the functional groups help less. Check whether OC/EC within the biomass-flagged subset is
closer to a "normal" calibration range than the full lot — supporting the smoke-specific approach.""")

code(r"""d["OC_EC"] = d["OC_ftir"] / d["EC_general"]
print("OC/EC median — all lot-251 Addis:", round(d["OC_EC"].median(), 2))
print("OC/EC median — biomass-flagged   :", round(d.loc[d["is_biomass"], "OC_EC"].median(), 2))
print("OC/EC median — non-biomass        :", round(d.loc[~d["is_biomass"], "OC_EC"].median(), 2))
cmp.to_csv("tables/biomass_vs_general_ec.csv", index=False)
print("\nwrote tables/biomass_vs_general_ec.csv",
      "(PLACEHOLDER)" if PLACEHOLDER else "(real)")""")

md(r"""## Summary / next actions

- **Lot is 251**, not 256 — use it (add 248 / PCA-similar lots for more smoke samples).
- Baseline is real: lot-251 Addis general-cal FTIR-EC vs HIPS is loaded; the median HIPS−EC gap
  quantifies how much room a biomass calibration has to close.
- **Blocked on Mona / the tool:** the biomass-only calibration and its per-filter EC predictions.
  Paste them into the INPUT cell and set `PLACEHOLDER = False` to get the real comparison.
- Feeds `03_adama_han_char_soot` (predict EC for the 5 Adama filters with both calibrations) — the
  cross-check on independent TOR data.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("02_biomass_calibration_comparison.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 02_biomass_calibration_comparison.ipynb")
