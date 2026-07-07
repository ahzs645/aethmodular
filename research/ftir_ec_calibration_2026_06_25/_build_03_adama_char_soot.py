"""Builds 03_adama_char_soot.ipynb. Run once, then nbconvert --execute.

Adama step 3 (2026-06-25 meeting): a simple **per-sample bar plot of char-EC (EC1 − OP) vs.
soot-EC (EC2 + EC3)** for the 5 Adama quartz-TOR filters, computed **both ways** — OP = OPTR
(reflectance, the Tor default) and OP = OPTT (transmittance) — because the meeting asked to try
**OPT vs OPR** and see whether the source story changes.

Reads the repo-local `adama_quartz_tor_batch54.csv` directly (no typed-in numbers) and self-verifies
against the file's own ECTR/ECTT.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Adama — char-EC vs. soot-EC (Han et al.), OPR **and** OPT

**Goal (2026-06-25 meeting, Adama step 3).** Classify each of the five Adama quartz filters as
**wood/char-dominated** vs. **diesel/soot-dominated** using the Han et al. framework, as a simple
**per-sample bar plot**. Compute it with **both** pyrolysis conventions:

- **char-EC = EC1 − OP**  (lower-temp EC; biomass/wood/coal charring residue)
- **soot-EC = EC2 + EC3** (higher-temp graphitic soot; diesel/traffic)
- **OP = OPTR** (reflectance — the Tor default) *or* **OP = OPTT** (transmittance — the meeting's
  "try OPT" ask)

High char/soot (≫1) → biomass; low (≲1) → diesel. A high **OC/EC** independently corroborates
biomass. *(Double-check these definitions against Han et al. — still to add to Zotero.)*""")

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

md(r"""## Load & pivot the TOR fractions (mass loading, µg C)""")

code(r"""raw = pd.read_csv(ADAMA / "adama_quartz_tor_batch54.csv")
piv = raw.pivot_table(index="FilterId", columns="Parameter",
                      values="MassLoading_ug", aggfunc="first")
need = ["OC1","OC2","OC3","OC4","OPTR","OPTT","EC1","EC2","EC3","ECTR","ECTT"]
missing = [c for c in need if c not in piv.columns]
assert not missing, f"missing TOR params: {missing}"
print("filters:", list(piv.index))
piv[need].round(3)""")

md(r"""## Compute char / soot / ratios for OPR and OPT, and self-verify""")

code(r"""def derive(op_col):
    d = pd.DataFrame(index=piv.index)
    OP = piv[op_col]
    d["char_EC"] = piv["EC1"] - OP
    d["soot_EC"] = piv["EC2"] + piv["EC3"]
    d["EC_TOR"]  = piv["EC1"] + piv["EC2"] + piv["EC3"] - OP
    d["OC_TOR"]  = piv[["OC1","OC2","OC3","OC4"]].sum(1) + OP
    d["char_soot"] = d["char_EC"] / d["soot_EC"]
    d["OC_EC"]     = d["OC_TOR"] / d["EC_TOR"]
    return d

opr = derive("OPTR")   # reflectance (Tor default)
opt = derive("OPTT")   # transmittance

# self-check EC_TOR vs the file's own ECTR / ECTT
assert np.allclose(opr["EC_TOR"], piv["ECTR"], atol=1e-2), "OPR EC_TOR != file ECTR"
assert np.allclose(opt["EC_TOR"], piv["ECTT"], atol=1e-2), "OPT EC_TOR != file ECTT"
print("self-check OK: EC_TOR(OPR)=ECTR, EC_TOR(OPT)=ECTT ✓\n")

comp = pd.DataFrame({
    "char_EC_opr": opr["char_EC"], "soot_EC_opr": opr["soot_EC"], "char_soot_opr": opr["char_soot"],
    "char_EC_opt": opt["char_EC"], "soot_EC_opt": opt["soot_EC"], "char_soot_opt": opt["char_soot"],
    "OC_EC_opr": opr["OC_EC"],
})
print(comp.round(3).to_string())
comp.round(4).to_csv("tables/adama_char_soot_opr_opt.csv")
print("\nwrote tables/adama_char_soot_opr_opt.csv")""")

md(r"""## Per-sample bar plot — char vs. soot (the meeting's simple plot)""")

code(r"""samples = list(piv.index)
x = np.arange(len(samples)); w = 0.38

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, d, tag in [(axes[0], opr, "OP = OPTR (reflectance, Tor default)"),
                   (axes[1], opt, "OP = OPTT (transmittance)")]:
    ax.bar(x - w/2, d["char_EC"], w, label="char-EC = EC1 − OP", color="#2ca02c")
    ax.bar(x + w/2, d["soot_EC"], w, label="soot-EC = EC2 + EC3", color="#7f7f7f")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(samples, rotation=0)
    ax.set_title(tag, fontsize=10)
    ax.set_xlabel("quartz FilterId")
    for xi, r in zip(x, d["char_soot"]):
        ax.annotate(f"c/s={r:.2f}", (xi, ax.get_ylim()[1]*0.9), ha="center", fontsize=7, color="#2ca02c")
axes[0].set_ylabel("EC mass loading (µg C)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Adama — char-EC vs. soot-EC per sample (OPR vs OPT)", y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig("figures/fig03_adama_char_soot.png", dpi=140, bbox_inches="tight")
print("saved figures/fig03_adama_char_soot.png"); plt.show()""")

md(r"""## Does OPT change the story? (char/soot ratio, both conventions)""")

code(r"""fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(x - w/2, opr["char_soot"], w, label="char/soot (OPR)", color="#1f77b4")
ax.bar(x + w/2, opt["char_soot"], w, label="char/soot (OPT)", color="#ff7f0e")
ax.axhline(1, color="red", ls="--", lw=1, label="char=soot (1:1)")
ax.axhline(2, color="green", ls="--", lw=1, label="biomass ≥ 2")
ax.set_xticks(x); ax.set_xticklabels(samples)
ax.set_ylabel("char-EC / soot-EC"); ax.set_xlabel("quartz FilterId")
ax.set_title("Source discriminator — OPR vs OPT")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig03b_char_soot_ratio_opr_opt.png", dpi=140, bbox_inches="tight")
print("saved figures/fig03b_char_soot_ratio_opr_opt.png"); plt.show()""")

md(r"""### Reading it next week
- Expected (from last week's Batch54 read): **char/soot is low (<1) for most samples → soot/diesel-
  leaning EC**, yet **OC/EC is high** — organic-rich aerosol whose EC speciation skews soot. That is
  the regime the FTIR functional-group calibration is expected to miss.
- **OPR vs OPT:** if the char/soot ranking is stable across conventions, the classification is
  robust; if a sample flips across the 1-line, flag it — the OP convention matters there.
- Cross-reference with `02_adama_tor_vs_ftir`: the char-rich samples should be the ones most
  under-reported by FTIR. That link is the punchline.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("03_adama_char_soot.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 03_adama_char_soot.ipynb")
