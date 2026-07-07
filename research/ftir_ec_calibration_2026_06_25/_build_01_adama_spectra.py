"""Builds 01_adama_spectra.ipynb. Run once, then nbconvert --execute.

Step 1 of the Adama action items (2026-06-25 meeting): plot the FTIR spectra of all 5 Adama
PTFE (Teflon) filters, and annotate the *substrate* artifacts (CF double peak; sloping baseline /
non-zero absorbance at 4000) so they aren't mistaken for aerosol signal. Sets up the meeting's hook:
"does the highest-EC sample show the highest peaks?" (Expectation: no — something is missing.)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Adama — the 5 PTFE FTIR spectra

**Goal (2026-06-25 meeting, Adama step 1).** Plot the FTIR spectra of all five Adama PTFE (Teflon)
filters and mark the **Teflon/substrate artifacts** so we don't read them as aerosol:

- **CF double peak** (carbon–fluorine bonds in the PTFE) around **~1150–1250 cm⁻¹**.
- **Sloping baseline** — non-zero absorbance at **4000 cm⁻¹** from **Teflon scattering**, which
  grows with aerosol loading. It is a confounder, *not* signal.

Then the hook for next week: **does the highest-EC sample show the highest peaks?** If the calibration
is missing char (the central hypothesis), the answer should be *no*.""")

code(r"""from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- locate the Adama data (lives in the predecessor folder) ---
CANDS = [Path("../spartan_ec_2026_06_16/data/adama"),
         Path("../../spartan_ec_2026_06_16/data/adama"),
         Path("spartan_ec_2026_06_16/data/adama")]
ADAMA = next((p for p in CANDS if (p / "adama_ptfe_spectra_batch54.csv").exists()), None)
assert ADAMA is not None, "adama_ptfe_spectra_batch54.csv not found in: " + ", ".join(map(str, CANDS))
print("Adama data dir:", ADAMA)

Path("figures").mkdir(exist_ok=True); Path("tables").mkdir(exist_ok=True)""")

md(r"""## Load the spectra

The file stores one **row per sample** (its first column, oddly named `Wavelength`, actually holds
the sample row-id 4744–4748) and one **column per wavenumber** (3998 → 420 cm⁻¹). We transpose to
the natural FTIR layout: wavenumber on the index, one column per sample.""")

code(r"""raw = pd.read_csv(ADAMA / "adama_ptfe_spectra_batch54.csv")
sample_ids = raw.iloc[:, 0].astype(str).tolist()          # 4744..4748 (file's own row ids)
wn = raw.columns[1:].astype(float).to_numpy()             # wavenumbers, 3998 -> 420
S = raw.iloc[:, 1:].to_numpy(dtype=float)                 # shape (5, n_wn), abs per sample
spec = pd.DataFrame(S.T, index=wn, columns=sample_ids)    # index=wavenumber, cols=sample
spec.index.name = "wavenumber_cm-1"
print("spectra:", spec.shape, "| samples:", sample_ids)
print("wavenumber range:", wn.max(), "->", wn.min(), "| absorbance range:",
      round(float(np.nanmin(S)), 3), "to", round(float(np.nanmax(S)), 3))
spec.iloc[:3]""")

md(r"""## Pull each PTFE filter's general FTIR-EC (for the ranking hook)

`adama_ptfe_ftir_batch54.csv` has the general-calibration FTIR EC/OC for the five PTFE filters. The
**spectra file uses row-ids 4744–4748**, which do **not** match the FTIR file's 228xxx barcodes, so
the definitive spectrum→FilterId link needs a barcode/MediaId crosswalk (**action item**). Until
then we use an explicit `ASSUMED_MAP` (spectra sorted ↔ FilterId sorted) so the plot is usable — the
mapping is clearly flagged and trivially corrected in one place.""")

code(r"""ft = pd.read_csv(ADAMA / "adama_ptfe_ftir_batch54.csv")
ec = (ft[ft["Parameter"] == "EC_ftir"]
      .sort_values("FilterId")[["FilterId", "Barcode", "SampleDate", "Concentration_ug_m3"]]
      .reset_index(drop=True))
ec = ec.rename(columns={"Concentration_ug_m3": "EC_ftir_ug_m3"})
print("General FTIR-EC per PTFE filter (µg/m³):")
print(ec.to_string(index=False))

# ASSUMED spectrum-row -> FilterId map (sorted<->sorted). CORRECT ME with the real crosswalk.
ASSUMED_MAP = dict(zip(sorted(sample_ids), ec["FilterId"].tolist()))
print("\nASSUMED spectrum->FilterId map (verify!):", ASSUMED_MAP)""")

md(r"""## Plot — raw spectra with the Teflon artifacts marked""")

code(r"""fig, ax = plt.subplots(figsize=(11, 5.5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(sample_ids)))
for c, sid in zip(colors, sample_ids):
    fid = ASSUMED_MAP.get(sid, sid)
    e = ec.set_index("FilterId")["EC_ftir_ug_m3"].get(fid, np.nan)
    ax.plot(spec.index, spec[sid], color=c, lw=1.0,
            label=f"{sid} (≈{fid}, EC_ftir={e:.2f})")

# Teflon/substrate artifacts
ymax = ax.get_ylim()[1]
ax.axvspan(1150, 1250, color="red", alpha=0.08)
ax.annotate("CF double peak\n(PTFE, artifact)", xy=(1200, ymax*0.80),
            xytext=(1550, ymax*0.88), color="red", ha="center", va="top", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
ax.axvline(4000, color="gray", ls=":", lw=1)
ax.annotate("non-zero abs @4000\n(Teflon scattering)", xy=(3960, ymax*0.42),
            xytext=(3500, ymax*0.62), color="gray", ha="left", va="center", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
# functional-group bands of interest (aerosol)
for x0, x1, lbl in [(2800, 3000, "CH"), (1650, 1750, "C=O"), (3100, 3500, "OH")]:
    ax.axvspan(x0, x1, color="green", alpha=0.05)
    ax.text((x0+x1)/2, ax.get_ylim()[1]*0.02, lbl, color="green", ha="center", fontsize=7)

ax.set_xlim(spec.index.max(), spec.index.min())          # FTIR convention: high -> low
ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_ylabel("absorbance (a.u.)")
ax.set_title("Adama PTFE FTIR spectra (n=5) — substrate artifacts marked")
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig("figures/fig01_adama_spectra.png", dpi=140, bbox_inches="tight")
print("saved figures/fig01_adama_spectra.png"); plt.show()""")

md(r"""## Baseline-flatten + peak-area ranking vs. EC ranking (the hook)

Subtract a straight baseline anchored at the two ends of each spectrum (a crude scatter correction),
integrate the CH band (2800–3000 cm⁻¹) as a rough "organic peak height", and compare that ranking to
the FTIR-EC ranking. If the highest-EC sample is **not** the highest-peak sample, that is the
"something is missing" story to show next week. *(A proper baseline is Sean's job; this is a quick
look, clearly labelled.)*""")

code(r"""def flatten(y, x):
    # linear baseline through the first/last points (x is descending)
    x0, x1 = x[0], x[-1]; y0, y1 = y[0], y[-1]
    base = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return y - base

xi = spec.index.to_numpy()
flat = pd.DataFrame({sid: flatten(spec[sid].to_numpy(), xi) for sid in sample_ids}, index=spec.index)

def band_area(df, lo, hi):
    m = (df.index >= lo) & (df.index <= hi)
    sub = df.loc[m]
    return pd.Series({c: np.trapz(sub[c][::-1], sub.index[::-1]) for c in df.columns})

rank = pd.DataFrame({
    "CH_area_2800_3000": band_area(flat, 2800, 3000),
    "CO_area_1650_1750": band_area(flat, 1650, 1750),
})
rank["FilterId_assumed"] = [ASSUMED_MAP.get(s, s) for s in rank.index]
rank = rank.merge(ec.set_index("FilterId")["EC_ftir_ug_m3"],
                  left_on="FilterId_assumed", right_index=True, how="left")
rank["peak_rank"] = rank["CH_area_2800_3000"].rank(ascending=False).astype(int)
rank["EC_rank"]   = rank["EC_ftir_ug_m3"].rank(ascending=False).astype(int)
print(rank.round(3).to_string())
rank.round(4).to_csv("tables/adama_spectra_peak_vs_ec.csv")
print("\nwrote tables/adama_spectra_peak_vs_ec.csv")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(rank["EC_ftir_ug_m3"], rank["CH_area_2800_3000"], s=80, zorder=3)
for i, r in rank.iterrows():
    ax.annotate(r["FilterId_assumed"], (r["EC_ftir_ug_m3"], r["CH_area_2800_3000"]),
                textcoords="offset points", xytext=(6, 3), fontsize=8)
ax.set_xlabel("general FTIR-EC (µg/m³)"); ax.set_ylabel("CH-band area (2800–3000, flattened)")
ax.set_title("Does the highest-EC sample have the biggest CH peak?")
plt.tight_layout(); plt.savefig("figures/fig01b_adama_peak_vs_ec.png", dpi=140, bbox_inches="tight")
print("saved figures/fig01b_adama_peak_vs_ec.png"); plt.show()""")

md(r"""### Notes / caveats
- **Spectrum→FilterId mapping is ASSUMED** (sorted↔sorted). Get the barcode/MediaId crosswalk from
  Sean/Mona before trusting the per-sample EC labels — this is the one thing to confirm here.
- The baseline flatten here is a **crude linear** correction, not Sean's routine. Use it only to
  motivate the "peaks don't track EC" point, then redo with the real baseline.
- Next: overlay Addis SPARTAN spectra on the same axes to see whether Adama looks like Addis.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("01_adama_spectra.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 01_adama_spectra.ipynb")
