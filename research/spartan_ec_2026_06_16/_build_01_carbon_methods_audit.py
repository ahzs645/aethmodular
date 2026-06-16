"""Builds 01_carbon_methods_audit.ipynb. Run once, then nbconvert --execute.

Confirms — directly from the SPARTAN public ChemSpec files — what SPARTAN's
carbon parameters actually are: which are FTIR, which are optical (HIPS / SSR),
and that there is NO thermal-optical (TOR/TOT) EC anywhere in the data. This is
the "thermal vs non-thermal EC" question Ann asked Ahmad to confirm.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# SPARTAN carbon methods audit — what is "thermal" EC, really?

**Question (from Ann, 2026-06-09 meeting).** In the EC/OC-by-site plot the markers were split
into "thermal EC" (stars) and "FTIR EC" (diamonds). Ann's correction: SPARTAN has **no quartz
filters**, so thermal-optical reflectance/transmittance (TOR/TOT) is *impossible* — whatever was
labelled "thermal" cannot be TOR. Ahmad was to send the exact terminology SPARTAN uses and
confirm it.

**This notebook answers that straight from the data** — the public SPARTAN `FilterBased
ChemSpecPM25` files for the four focus sites (ETAD, CHTS, INDH, USPA) — by reading the
`Parameter_Name`, `Method_Code`, and `Analysis_Description` columns. No interpretation needed:
the files state the analysis method for every carbon value.

**Bottom line (verified below):**
- `EC PM2.5` and `OC PM2.5` → **FTIR** (method codes 217, 218).
- `BC PM2.5` → **HIPS** (optical; methods 219, 221) or *estimated from the HIPS–SSR BC curve* (220).
- `Equivalent BC PM2.5` → **Smoke Stain Reflectometer (SSR)** (methods 212, 214, 215, 216).
- **No TOR / TOT / thermal-optical EC exists in the SPARTAN data.** So the "thermal" stars were a
  mislabel — every SPARTAN EC value is FTIR; every BC value is optical. Use the FTIR EC.""")

code(r"""import glob, os
import pandas as pd

DATA = "../filter_combine"
files = sorted(glob.glob(f"{DATA}/FilterBased_ChemSpecPM25_*.csv"))
print("ChemSpec files found:")
for f in files:
    print("  ", os.path.basename(f))""")

md(r"""## 1. Every carbon parameter, with its declared analysis method

We pull all rows whose `Parameter_Name` mentions carbon/EC/OC/BC and list the distinct
(parameter, method code, analysis description) combinations the files actually contain.""")

code(r"""rows = []
for f in files:
    site = os.path.basename(f).split("_")[-1].replace(".csv", "")
    df = pd.read_csv(f, comment="#")          # 3 leading '#' metadata lines
    m = df["Parameter_Name"].str.contains(
        "carbon|EC|OC|BC|black|elemental|organic", case=False, na=False)
    sub = df[m][["Parameter_Code", "Parameter_Name", "Method_Code",
                 "Analysis_Description", "Units"]].drop_duplicates()
    sub.insert(0, "Site", site)
    rows.append(sub)

carbon = pd.concat(rows, ignore_index=True)
carbon["Analysis_Description"] = carbon["Analysis_Description"].astype(str).str.slice(0, 55)
carbon.sort_values(["Parameter_Name", "Method_Code", "Site"])""")

md(r"""## 2. The method-code → technique map (the answer for Ann)

Collapsed across sites: every distinct carbon measurement method present in SPARTAN's public
release, and which technique family it belongs to.""")

code(r"""def family(desc):
    d = str(desc).lower()
    if "ftir" in d:                                   return "FTIR (functional groups)"
    if "hips" in d and "ssr" in d:                    return "Optical — HIPS-SSR curve estimate"
    if "hips" in d:                                   return "Optical — HIPS"
    if "smoke stain" in d or "ssr" in d:              return "Optical — Smoke Stain Reflectometer"
    if "thermal" in d or "tor" in d or "tot" in d:    return "THERMAL-OPTICAL (TOR/TOT)"
    return "other / check"

mp = (carbon[["Parameter_Name", "Method_Code", "Analysis_Description"]]
      .drop_duplicates()
      .assign(Technique=lambda d: d["Analysis_Description"].map(family))
      .sort_values(["Parameter_Name", "Method_Code"])
      .reset_index(drop=True))
mp""")

code(r"""thermal = mp[mp["Technique"].str.contains("THERMAL")]
print("Carbon methods that are thermal-optical (TOR/TOT):", len(thermal))
print()
print("=> SPARTAN EC PM2.5 is measured by:",
      sorted(mp.loc[mp.Parameter_Name.eq('EC PM2.5'), 'Technique'].unique()))
print("=> SPARTAN BC PM2.5 is measured by:",
      sorted(mp.loc[mp.Parameter_Name.eq('BC PM2.5'), 'Technique'].unique()))
print("=> SPARTAN Equivalent BC PM2.5 is measured by:",
      sorted(mp.loc[mp.Parameter_Name.eq('Equivalent BC PM2.5'), 'Technique'].unique()))
assert len(thermal) == 0, "Found a thermal-optical method — re-examine!"
print("\nCONFIRMED: no thermal-optical (TOR/TOT) EC in the SPARTAN public data.")""")

md(r"""**Reading of the result.** The only EC in SPARTAN is `EC PM2.5` by **FTIR** (methods 217/218 —
the UC Davis functional-group prediction, the same method this whole project is scrutinising).
The values that *look* like an independent "thermal EC" are actually optical **BC** (HIPS, or the
HIPS–SSR curve) and **Equivalent BC** (Smoke Stain Reflectometer) — light-absorption measures, not
thermal carbon. So on the EC/OC-by-site plot, the FTIR diamonds are the EC; the "thermal" stars
should be dropped or relabelled as optical BC. This matches Ann exactly: no quartz → no TOR.""")

md(r"""## 3. Cross-check against the in-house unified dataset

The combined working dataset distinguishes the public ChemSpec parameters from the in-house FTIR
predictions. Confirm the FTIR side carries `EC_ftir`/`OC_ftir` (plus functional groups) and the
ChemSpec side carries the optical `BC`.""")

code(r"""u = pd.read_pickle(f"{DATA}/unified_filter_dataset.pkl")
print("Parameters by data source:")
print(u.groupby(["DataSource", "Parameter"]).size()
       .reset_index(name="n")
       .query("Parameter.str.contains('EC|OC|BC|carbon', case=False)", engine="python")
       .to_string(index=False))""")

md(r"""## 4. Which lot has the most Addis samples? (grounding the calibration work)

Ann thought it was **lot 256** ("but I'm not sure"). The data says otherwise — worth flagging
before building the biomass-burning calibration, since the whole experiment is run on a chosen lot.""")

code(r"""ftir_ec = u[u["Parameter"].eq("EC_ftir")]
addis = ftir_ec[ftir_ec["Site"].astype(str).str.contains("ETAD|Addis", case=False, na=False)]
by_lot = addis.groupby("LotId").size().sort_values(ascending=False)
print("Addis (ETAD) FTIR-EC samples by LotId:")
print(by_lot.to_string())
print(f"\n=> Lot with the most Addis samples: {int(by_lot.index[0])} "
      f"({int(by_lot.iloc[0])} samples) — NOT 256.")""")

md(r"""## Summary to send Ann

| SPARTAN parameter | Method code(s) | Technique | Independent of FTIR? |
|---|---|---|---|
| `EC PM2.5` | 217, 218 | **FTIR** functional-group prediction | no — *this is the method under review* |
| `OC PM2.5` | 217, 218 | **FTIR** | no |
| `BC PM2.5` | 219, 221 | **HIPS** (optical absorption) | yes |
| `BC PM2.5` | 220 | estimated from the **HIPS–SSR** curve | partly |
| `Equivalent BC PM2.5` | 212, 214, 215, 216 | **Smoke Stain Reflectometer** | yes |

- **There is no thermal/TOR EC in SPARTAN** — confirmed across ETAD, CHTS, INDH, USPA. The
  "thermal" label was wrong; those points are optical BC, not thermal carbon.
- For the EC/OC-by-site figure: keep the **FTIR EC** (`EC PM2.5`), drop the "thermal" series (or
  relabel it "optical BC — HIPS/SSR" and plot it as a different quantity, not as EC).
- The only genuinely *independent* check on FTIR EC at these sites is the **optical** BC (HIPS),
  which is exactly why the fAbs-vs-EC and tau-vs-EC cross-plots carry the weight of this argument.
- **Calibration lot:** the lot with the most Addis samples is **251**, not 256 — use 251 (and add
  248 / PCA-similar lots if more smoke samples are needed).""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("01_carbon_methods_audit.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 01_carbon_methods_audit.ipynb")
