"""Builds 03_adama_han_char_soot.ipynb. Run once, then nbconvert --execute.

Han et al. char-EC / soot-EC classification of the five Adama (AMOD) quartz-filter
TOR samples Alex sent. The five samples' TOR fractions are NOT in the repo yet
(they arrived by email), so the input table below is a clearly-marked PLACEHOLDER
that lets the notebook run end-to-end. Paste Alex's real OC1-4 / OP / EC1-3 values
into that one cell and re-run; everything downstream updates automatically.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Adama AMOD quartz filters — char-EC vs soot-EC (Han et al.)

**Goal (2026-06-09 meeting).** Alex sent TOR carbon fractions (OC1–OC4, OP, EC1–EC3) for five
**Adama, Ethiopia** quartz filters — the same airshed as Addis, and the *only* thermal-optical
EC we have anywhere near our SPARTAN sites (SPARTAN itself has no quartz → no TOR, see
`01_carbon_methods_audit`). Using the **Han et al.** char/soot framework, classify each sample as
**wood/char-dominated** vs **diesel/soot-dominated**. This tells us whether the Adama (≈Addis)
aerosol is the kind of biomass-heavy, char-rich mixture that the FTIR functional-group
calibration is expected to *under*-predict — the central hypothesis of the SPARTAN-EC work.

### Han char/soot definitions (IMPROVE/IMPROVE_A TOR protocol)
For TOR fractions OC1–OC4 (He steps), EC1–EC3 (He/O₂ steps), and OP (the pyrolysed/optical
charring correction):

- **char-EC = EC1 − OP**  — lower-temperature EC; the pyrolysis/charring residue. Enriched in
  **biomass / wood / coal** smoke.
- **soot-EC = EC2 + EC3** — higher-temperature EC; graphitic soot formed in the gas phase.
  Enriched in **traffic / diesel / high-temperature fossil** combustion.
- **EC(TOR) = EC1 + EC2 + EC3 − OP = char-EC + soot-EC**
- **OC(TOR) = OC1 + OC2 + OC3 + OC4 + OP**

The **char-EC/soot-EC ratio** is the source discriminator: high (≫1) → biomass/wood-dominated;
low (≲1) → diesel/traffic-dominated (Han et al. 2007, 2010).

> **Citation note:** the Han et al. paper Ann sent is *not in the Zotero library yet* — add it so
> we can cite these definitions. (Han, Y.M. et al., *Chemosphere* 2007, "char-EC vs soot-EC";
> Han et al. 2010 on atmospheric char vs soot.) The exact OP convention (OPC/OPT) should be
> matched to Alex's lab protocol when the data lands.""")

md(r"""## Load the Adama TOR data directly from the Batch54 file

Read `Carbon_concs_Batch54.csv` (device AD00412, Adama; 5 quartz filters; IMPROVE_A TOR protocol)
and pivot the long table to one row per filter. No numbers are typed in — everything is read from
the file. We take **mass loading (µg C)** (ratios are unit-free) and map **OP = OPTR** (pyrolysed
carbon by **reflectance**, the TOR convention). The load is verified against the file's own `ECTR`
(`EC1+EC2+EC3 − OPTR == ECTR`).""")

code(r"""import os
from pathlib import Path
import pandas as pd, numpy as np

# Canonical source (Google Drive). Falls back to a repo-local copy if present.
CANDIDATES = [
    Path.home() / "Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive"
        "/University/Research/Grad/UC Davis Ann/NASA MAIA/Data/Adama TOR",
    Path("data/adama_tor"),
    Path("../../data/adama_tor"),
]
ADAMA_DIR = next((p for p in CANDIDATES if (p / "Carbon_concs_Batch54.csv").exists()), None)
if ADAMA_DIR is None:
    raise FileNotFoundError(
        "Carbon_concs_Batch54.csv not found. Checked:\n  " +
        "\n  ".join(str(p) for p in CANDIDATES))
print("Adama TOR source:", ADAMA_DIR)

raw = pd.read_csv(ADAMA_DIR / "Carbon_concs_Batch54.csv")
piv = raw.pivot_table(index="FilterId", columns="Parameter",
                      values="MassLoading_ug", aggfunc="first")

adama = pd.DataFrame({
    "OC1": piv["OC1"], "OC2": piv["OC2"], "OC3": piv["OC3"], "OC4": piv["OC4"],
    "OP":  piv["OPTR"],                       # OP = pyrolysed C by reflectance (TOR)
    "EC1": piv["EC1"], "EC2": piv["EC2"], "EC3": piv["EC3"],
})
adama.index.name = "sample"

# verify the load against the file's reported ECTR
chk = (adama[["EC1","EC2","EC3"]].sum(1) - adama["OP"])
assert np.allclose(chk.values, piv["ECTR"].values, atol=1e-2), "EC(TOR) != reported ECTR"
print(f"Loaded {len(adama)} Adama filters; EC(TOR)=EC1+EC2+EC3-OP matches file ECTR ✓")
adama.round(3)""")

md(r"""## Compute char-EC, soot-EC, totals, and ratios""")

code(r"""d = adama.copy()
d["char_EC"]  = d["EC1"] - d["OP"]
d["soot_EC"]  = d["EC2"] + d["EC3"]
d["EC_TOR"]   = d["EC1"] + d["EC2"] + d["EC3"] - d["OP"]
d["OC_TOR"]   = d[["OC1","OC2","OC3","OC4","OP"]].sum(axis=1)
d["char_soot_ratio"] = d["char_EC"] / d["soot_EC"]
d["OC_EC_ratio"]     = d["OC_TOR"] / d["EC_TOR"]
d["char_frac_of_EC"] = d["char_EC"] / d["EC_TOR"]

d[["char_EC","soot_EC","EC_TOR","OC_TOR","char_soot_ratio","OC_EC_ratio","char_frac_of_EC"]].round(3)""")

md(r"""## Classify each sample

Thresholds follow the Han-et-al. interpretation (refine against Alex's protocol / the Han values
once available):

- **char/soot ≥ 2** → *wood / biomass-dominated* (char-rich) — the regime FTIR is expected to miss.
- **char/soot ≤ 1** → *diesel / traffic-dominated* (soot-rich).
- **between** → *mixed*.

A high **OC/EC** ratio independently corroborates biomass burning.""")

code(r"""def classify(r):
    cs = r["char_soot_ratio"]
    if cs >= 2:   return "wood / biomass-dominated (char-rich)"
    if cs <= 1:   return "diesel / traffic-dominated (soot-rich)"
    return "mixed"

d["source_class"] = d.apply(classify, axis=1)
summary = d[["char_soot_ratio","OC_EC_ratio","char_frac_of_EC","source_class"]].round(2)
print(summary.to_string())
print("\nClass counts:")
print(d["source_class"].value_counts().to_string())""")

md(r"""### Honest reading — two metrics disagree, and that's the finding

The real Batch54 samples sit in a genuinely interesting place:

- **char-EC/soot-EC is low (0.02–0.58, all < 1)** → the *EC that is present* is **soot-like** (the
  high-temperature EC2 fraction dominates; EC1 is largely pyrolysed-OC removed as OP).
- **OC/EC is high (4.6–7.2)** and pyrolysed carbon (OP) is large → the aerosol is very
  **organic-rich**, the classic biomass-burning signature.

So these are **not clean diesel samples** — they're organic-rich (biomass-influenced) aerosol whose
EC speciation skews to soot. Don't over-read the char/soot label alone. The combination (high OC/EC,
high OP, soot-leaning EC) is exactly the regime where the FTIR functional-group calibration is
expected to struggle — which is *why* the biomass-calibration test (notebook 02) matters.""")

code(r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.scatter(d["soot_EC"], d["char_EC"], s=90, zorder=3)
for s, r in d.iterrows():
    ax.annotate(s, (r["soot_EC"], r["char_EC"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
lim = max(d["soot_EC"].max(), d["char_EC"].max()) * 1.1
ax.plot([0, lim], [0, lim], "k--", lw=1, label="char = soot (1:1)")
ax.plot([0, lim], [0, 2*lim], color="green", lw=1, ls=":", label="char = 2×soot (biomass)")
ax.set(xlabel="soot-EC = EC2 + EC3", ylabel="char-EC = EC1 − OP",
       title="Char vs soot — Adama AMOD")
ax.legend(fontsize=8); ax.set_xlim(0); ax.set_ylim(0)

ax = axes[1]
order = d.sort_values("char_soot_ratio").index
ax.barh(range(len(order)), d.loc[order, "char_soot_ratio"], zorder=3)
ax.axvline(1, color="red", ls="--", lw=1, label="diesel/traffic ≤ 1")
ax.axvline(2, color="green", ls="--", lw=1, label="biomass ≥ 2")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set(xlabel="char-EC / soot-EC", title="Source discriminator (higher = more wood/char)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig03_adama_char_soot.png", dpi=140, bbox_inches="tight")
print("saved figures/fig03_adama_char_soot.png")
plt.show()""")

# (placeholder logic removed — data is read directly from the Batch54 file)

md(r"""## Next step — predict FTIR EC for these five and compare to TOR

Once Mona sends the **spectra** for these five Adama filters, predict EC two ways and compare to
the **EC(TOR)** computed above:

1. general lot-251 calibration (all samples), and
2. wood-smoke-only lot-251 calibration (from `02_biomass_calibration_comparison`).

**Expectation:** for the char-rich (biomass-dominated) samples, FTIR-EC < EC(TOR), and the
wood-smoke calibration narrows that gap. That is the direct test of "FTIR misses char." Save the
classified table for that comparison.""")

code(r"""d.round(4).to_csv("tables/adama_char_soot_classification.csv")
print("wrote tables/adama_char_soot_classification.csv")""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("03_adama_han_char_soot.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 03_adama_han_char_soot.ipynb")
