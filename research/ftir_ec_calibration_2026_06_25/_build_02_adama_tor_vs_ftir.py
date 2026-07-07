"""Builds 02_adama_tor_vs_ftir.ipynb. Run once, then nbconvert --execute.

Adama step 2 (2026-06-25 meeting): cross-plot **Tor EC (thermal-optical ground truth) vs.
FTIR-EC** for the 5 Adama samples. The meeting was explicit: compare against **Tor, not
FTIR-vs-FTIR** — Tor is what we are trying to reproduce, so it goes on the x-axis and the 1:1 line
is the reference. The quartz-TOR and PTFE-FTIR filters are co-located pairs from the same day but
carry different J-numbers, so we join on **sample date**.

The general-calibration FTIR-EC is real (from the batch file). The biomass-calibration FTIR-EC is a
clearly-marked INPUT cell to paste once those predictions exist.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Adama — Tor EC vs. FTIR-EC (ground-truth cross-plot)

**Goal (2026-06-25 meeting, Adama step 2).** For the five Adama samples with thermal-optical
reflectance (**Tor**) carbon, plot **Tor EC on x** (the ground truth) against **FTIR-EC on y**, with
the **1:1 line**. Points **below 1:1** are where **FTIR under-reports EC vs. Tor** — the central
hypothesis. Do this for the **general** calibration now; overlay the **biomass** calibration when its
predictions land.

> The advisors' correction: *don't* plot FTIR-vs-FTIR. Tor is the reference we are trying to
> reproduce, so everything is measured against it.""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANDS = [Path("../spartan_ec_2026_06_16/data/adama"),
         Path("../../spartan_ec_2026_06_16/data/adama"),
         Path("spartan_ec_2026_06_16/data/adama")]
ADAMA = next((p for p in CANDS if (p / "adama_quartz_tor_batch54.csv").exists()), None)
assert ADAMA is not None, "adama_quartz_tor_batch54.csv not found in: " + ", ".join(map(str, CANDS))
print("Adama data dir:", ADAMA)
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)""")

md(r"""## Tor EC per quartz filter

`ECTR` = EC by **reflectance** (the Tor convention) = `EC1+EC2+EC3 − OPTR`. We take the file's
`Concentration_ug_m3` (ambient concentration) so it is directly comparable to the FTIR-EC
concentration, even though the two measurements are on different substrates (quartz vs. Teflon).""")

code(r"""tor = pd.read_csv(ADAMA / "adama_quartz_tor_batch54.csv")
tor["date"] = pd.to_datetime(tor["SampleDate"]).dt.normalize()
tor_ec = (tor[tor["Parameter"] == "ECTR"]
          .set_index("FilterId")[["date", "Concentration_ug_m3"]]
          .rename(columns={"Concentration_ug_m3": "Tor_EC_ug_m3"}))
tor_ec.index.name = "quartz_FilterId"
print("Tor EC (reflectance) per quartz filter:")
print(tor_ec.to_string())""")

md(r"""## General FTIR-EC per PTFE filter, joined to Tor by date""")

code(r"""ft = pd.read_csv(ADAMA / "adama_ptfe_ftir_batch54.csv")
ft["date"] = pd.to_datetime(ft["SampleDate"]).dt.normalize()
ftir_ec = (ft[ft["Parameter"] == "EC_ftir"]
           .set_index("FilterId")[["date", "Concentration_ug_m3"]]
           .rename(columns={"Concentration_ug_m3": "FTIR_EC_general_ug_m3"}))
ftir_ec.index.name = "ptfe_FilterId"

pair = (tor_ec.reset_index().merge(ftir_ec.reset_index(), on="date", how="inner")
        .sort_values("date").reset_index(drop=True))
assert len(pair) == 5, f"expected 5 co-located pairs, got {len(pair)}"
print("Paired quartz(Tor) <-> PTFE(FTIR) by sample date:")
print(pair.to_string(index=False))""")

md(r"""## INPUT — biomass-calibration FTIR-EC (paste when available)

Fill in the biomass-only calibration's EC prediction (µg/m³) for each **PTFE FilterId**. Leave as
`None` to skip the biomass overlay. Once `05_calibration_variants` produces these, paste them here
and re-run — the plot and table update automatically.""")

code(r"""BIOMASS_EC_ug_m3 = {
    "J1233": None,
    "J1266": None,
    "J1269": None,
    "J1270": None,
    "J1285": None,
}
pair["FTIR_EC_biomass_ug_m3"] = pair["ptfe_FilterId"].map(BIOMASS_EC_ug_m3)
has_biomass = pair["FTIR_EC_biomass_ug_m3"].notna().any()
print("biomass overlay:", "ON" if has_biomass else "OFF (placeholders still None)")
pair""")

md(r"""## Cross-plot — Tor EC (x) vs. FTIR-EC (y), 1:1 reference""")

code(r"""fig, ax = plt.subplots(figsize=(6.6, 6.2))
lim = float(np.nanmax([pair["Tor_EC_ug_m3"].max(),
                       pair["FTIR_EC_general_ug_m3"].max(),
                       (pair["FTIR_EC_biomass_ug_m3"].max() if has_biomass else 0)])) * 1.15
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 (FTIR = Tor)")

ax.scatter(pair["Tor_EC_ug_m3"], pair["FTIR_EC_general_ug_m3"],
           s=90, color="#1f77b4", zorder=3, label="general FTIR-EC")
if has_biomass:
    ax.scatter(pair["Tor_EC_ug_m3"], pair["FTIR_EC_biomass_ug_m3"],
               s=90, color="#d62728", marker="^", zorder=3, label="biomass FTIR-EC")

for _, r in pair.iterrows():
    ax.annotate(f'{r["ptfe_FilterId"]}/{r["quartz_FilterId"]}',
                (r["Tor_EC_ug_m3"], r["FTIR_EC_general_ug_m3"]),
                textcoords="offset points", xytext=(7, 3), fontsize=8)

ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Tor EC (reflectance, µg/m³)  — ground truth")
ax.set_ylabel("FTIR-EC (µg/m³)")
ax.set_title("Adama — Tor EC vs. FTIR-EC (n=5)")
ax.text(lim*0.97, lim*0.03, "below 1:1 → FTIR under-reports", ha="right", va="bottom",
        fontsize=8, color="gray", style="italic")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("figures/fig02_adama_tor_vs_ftir.png", dpi=140, bbox_inches="tight")
print("saved figures/fig02_adama_tor_vs_ftir.png"); plt.show()""")

md(r"""## The numbers — ratio to Tor, and where each sample sits""")

code(r"""out = pair.copy()
out["general/Tor"] = out["FTIR_EC_general_ug_m3"] / out["Tor_EC_ug_m3"]
if has_biomass:
    out["biomass/Tor"] = out["FTIR_EC_biomass_ug_m3"] / out["Tor_EC_ug_m3"]
out["side_of_1:1(general)"] = np.where(out["general/Tor"] < 1, "below (FTIR<Tor)", "above (FTIR>Tor)")
cols = ["date", "quartz_FilterId", "ptfe_FilterId", "Tor_EC_ug_m3",
        "FTIR_EC_general_ug_m3", "general/Tor", "side_of_1:1(general)"]
print(out[cols].round(3).to_string(index=False))
out.round(4).to_csv("tables/adama_tor_vs_ftir.csv", index=False)
print("\nwrote tables/adama_tor_vs_ftir.csv")
print("\nmedian general/Tor ratio:", round(float(out['general/Tor'].median()), 3))""")

md(r"""### Reading it next week
- Points **below the 1:1 line** are the "FTIR misses EC" cases — call them out by FilterId.
- The **J1269/J1693** pair (2024-07-09) is the anomalous high-EC day flagged in the meeting — see
  where it lands and whether the biomass calibration moves it *toward* Tor (expected) or the wrong
  way (the meeting noted one sample moves opposite).
- Pair this with `03_adama_char_soot`: samples that are **char-rich** should be the ones FTIR
  under-reports most. That linkage is the story.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("02_adama_tor_vs_ftir.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 02_adama_tor_vs_ftir.ipynb")
