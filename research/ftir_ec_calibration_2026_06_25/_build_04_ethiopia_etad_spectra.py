"""Builds 04_ethiopia_etad_spectra.ipynb. Run once, then nbconvert --execute.

Ethiopia action item (2026-06-25 meeting): plot ALL the ETAD (Addis/Ethiopia) FTIR spectra, then
split them by month so **Navid's high-biomass vs. diesel months** can be contrasted as two spectra
clouds. Loads the Google-Drive `ETAD FTIR` export directly.

The month split is parameterized (`BIOMASS_MONTHS` / `DIESEL_MONTHS`) with a clearly-marked
PLACEHOLDER default — replace with Navid's actual month list and re-run.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Ethiopia (ETAD) — all FTIR spectra, split by month

**Goal (2026-06-25 meeting).** (1) Plot **all** the ETAD Addis/Ethiopia FTIR spectra to orient us to
the dataset. (2) Split them into **high-biomass months** vs. **diesel months** (per Navid's earlier
segregation) and show the two spectra clouds side by side. Later these feed the calibration-vs-FABS
comparison (MAC = 10 for now).""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ETAD FTIR export on Google Drive (with fallbacks)
CANDS = [
    Path.home() / "Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/My Drive/University"
        "/Research/Grad/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR",
    Path("data/etad_ftir"),
]
ETAD = next((p for p in CANDS if (p / "ETAD_FTIR_spectra.csv").exists()), None)
assert ETAD is not None, "ETAD_FTIR_spectra.csv not found in: " + ", ".join(map(str, CANDS))
print("ETAD FTIR dir:", ETAD)
Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)""")

md(r"""## Load spectra + metadata, join on MediaId to get sampling dates""")

code(r"""spec = pd.read_csv(ETAD / "ETAD_FTIR_spectra.csv")
meta = pd.read_csv(ETAD / "ETAD_metadata.csv")
print("spectra:", spec.shape, "| metadata:", meta.shape)

idcols = ["SampleAnalysisId", "MediaId"]
# wavenumber columns sorted high -> low (FTIR convention)
wcols = sorted([c for c in spec.columns if c not in idcols], key=lambda c: -float(c))
X = spec.set_index("MediaId")[wcols]
X.columns = [float(c) for c in wcols]
print("spectra matrix:", X.shape, "| wavenumber", X.columns.max(), "->", X.columns.min())

meta["month"] = pd.to_datetime(meta["SamplingStartDate"], errors="coerce").dt.month
m = meta.set_index("MediaId")[["SiteCode", "SamplingStartDate", "month",
                               "MassCollectedOnFilter_ug", "SampleVolume_m3"]]
Xm = X.join(m, how="left")
print("joined; with month:", Xm["month"].notna().sum(), "of", len(Xm))
print("month histogram:\n", Xm["month"].value_counts().sort_index().to_string())""")

md(r"""## Plot 1 — all ETAD spectra overlaid""")

code(r"""wnv = np.array(X.columns, dtype=float)
fig, ax = plt.subplots(figsize=(11, 5.5))
for mid, row in X.iterrows():
    ax.plot(wnv, row.to_numpy(dtype=float), color="steelblue", lw=0.3, alpha=0.25)
ax.plot(wnv, X.mean(axis=0).to_numpy(dtype=float), color="black", lw=1.5, label="mean spectrum")
ax.axvspan(1150, 1250, color="red", alpha=0.06)     # CF (Teflon) artifact
ax.text(1200, ax.get_ylim()[1]*0.95, "CF (PTFE)", color="red", ha="center", va="top", fontsize=8)
ax.set_xlim(wnv.max(), wnv.min())
ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_ylabel("absorbance (a.u.)")
ax.set_title(f"All ETAD (Addis/Ethiopia) FTIR spectra (n={len(X)})")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig04_etad_spectra_all.png", dpi=140, bbox_inches="tight")
print("saved figures/fig04_etad_spectra_all.png"); plt.show()""")

md(r"""## INPUT — Navid's high-biomass vs. diesel months

**PLACEHOLDER** month split below — replace with **Navid's actual** biomass-burning vs. diesel
months for Addis/Ethiopia. Default here is a dry-season (biomass) vs. wet-season (cleaner/diesel)
guess purely so the plot renders; it is **not** Navid's classification.""")

code(r"""# TODO: replace with Navid's real month lists.
BIOMASS_MONTHS = [10, 11, 12, 1, 2]     # PLACEHOLDER (dry-season guess)
DIESEL_MONTHS  = [6, 7, 8]              # PLACEHOLDER (wet-season guess)

Xm["season"] = np.where(Xm["month"].isin(BIOMASS_MONTHS), "high-biomass",
                 np.where(Xm["month"].isin(DIESEL_MONTHS), "diesel", "other/unassigned"))
print(Xm["season"].value_counts().to_string())""")

md(r"""## Plot 2 — spectra by season (biomass vs diesel)""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
palette = {"high-biomass": "#2ca02c", "diesel": "#7f7f7f"}
for ax, season in zip(axes, ["high-biomass", "diesel"]):
    sub = Xm[Xm["season"] == season]
    for mid, row in sub[X.columns].iterrows():
        ax.plot(wnv, row.to_numpy(dtype=float), color=palette[season], lw=0.3, alpha=0.25)
    if len(sub):
        ax.plot(wnv, sub[X.columns].mean(axis=0).to_numpy(dtype=float),
                color="black", lw=1.5, label=f"mean (n={len(sub)})")
    ax.set_xlim(wnv.max(), wnv.min())
    ax.set_title(f"{season} months  ({', '.join(map(str, BIOMASS_MONTHS if season=='high-biomass' else DIESEL_MONTHS))})",
                 fontsize=10)
    ax.set_xlabel("wavenumber (cm⁻¹)"); ax.legend(fontsize=8)
axes[0].set_ylabel("absorbance (a.u.)")
fig.suptitle("ETAD spectra by month split  (PLACEHOLDER split — use Navid's months)", y=1.02)
plt.tight_layout(); plt.savefig("figures/fig04_etad_spectra_by_season.png", dpi=140, bbox_inches="tight")
print("saved figures/fig04_etad_spectra_by_season.png"); plt.show()""")

md(r"""## Save a small per-sample summary table""")

code(r"""summ = Xm[["SiteCode", "SamplingStartDate", "month", "season",
           "MassCollectedOnFilter_ug", "SampleVolume_m3"]].copy()
summ.to_csv("tables/etad_sample_month_summary.csv")
print("wrote tables/etad_sample_month_summary.csv  (", len(summ), "rows )")""")

md(r"""### Next steps
- Replace the placeholder months with **Navid's** biomass vs. diesel split and re-run.
- Predict EC on these spectra with each calibration variant (`05_calibration_variants`) and compare
  to **FABS** (MAC = 10 for now; the 6-vs-10 question is unresolved).
- Overlay the Adama spectra (`01`) on the ETAD mean to check whether Adama looks like Addis.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("04_ethiopia_etad_spectra.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 04_ethiopia_etad_spectra.ipynb")
