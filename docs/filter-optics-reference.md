# Filter and optics reference — SPARTAN, IMPROVE, and the Warren White corrections

A single technical reference for the filter hardware and optical processing used
by SPARTAN and IMPROVE, plus the published Warren White corrections that bridge
the two networks. Every numeric value carries a `[source]` tag so you can
follow it back to its origin.

This file lives in `docs/` so it is easy to find, and is meant to be appended
to as we learn new values — particularly for the SPARTAN-specific quantities
that Warren White's corrections need but that no public document has confirmed
yet (`H`, `α`).

---

## 1. SPARTAN sampling filter

| Property | Value | Source |
|---|---|---|
| Filter material | PTFE (Teflon) membrane | `[SPARTAN EBC SOP v2.0, §5.0]` |
| Diameter | 25 mm | `[SPARTAN EBC SOP v2.0, §5.0]` |
| Active deposit area `A` | **3.1 cm²** | `[SPARTAN EBC SOP v2.0, §2.0 equation]` |
| Vendor part number | **PT25DMCAN-PF03A** (Measurement Technology Labs / Mesa Labs) | `[SPARTAN EBC SOP v2.0, §5.0]` |
| Pore size | **3 μm** | `[SPARTAN EBC SOP v2.0, §5.0]` |
| Support ring | FEP polymer | `[SPARTAN EBC SOP v2.0, §5.0]` |
| Sampler | AirPhoton SS5i with BGI cyclone PM2.5 inlet | `[Public SPARTAN CSV "Collection_Description"]` |
| Nominal collection cadence | 1 filter every 9 days | `[Observed in public coverage analysis — research/spartan/inventory/coverage.csv]` |
| Field-blank convention | Replicate number `-7` in the filter ID | `[HIPS Drive file inspection — research/spartan/inventory/HIPS_BRIDGE.md]` |

> **Note on the 3 μm pore.** PTFE membrane filters at 3 μm still collect PM2.5
> at >99% efficiency. Submicron particles are captured by impaction,
> interception, and Brownian diffusion rather than by sieving. The 3 μm
> rating is chosen because it gives **low pressure drop**, which extends
> pump life and improves flow stability — both important for a global,
> remotely-operated network.

> **Why nitrate is reported as qualitative.** The open membrane and the
> ambient-to-lab transit time let ammonium nitrate volatilize between
> collection and analysis. SPARTAN flags this in the public CSV
> `Analysis_Description` column ("These nitrate concentrations collected on
> Teflon filters are likely biased low.") and in the Known Issues sheet that
> cites Snider et al. 2015.

---

## 2. SPARTAN optical analysis chain on the PTFE filter

Two devices read the same filter:

### 2a. Smoke Stain Reflectometer (SSR) — Equivalent Black Carbon (EBC)

| Quantity | Value | Source |
|---|---|---|
| Equation | `[EBC] = -A / (q·v) · ln(R / R₀)`, in μg | `[SPARTAN EBC SOP v2.0, §2.0]` |
| `A` (filter area) | 3.1 cm² | `[SOP §2.0]` |
| `q` | 0.5 · σ_SSR (reflectivity path × MAC) | `[SOP §2.0]` |
| σ_SSR (mass absorption cross-section) | **0.1 cm² μg⁻¹** ≡ **10 m² g⁻¹** | `[SOP §2.0]` |
| Thickness factor (PTFE in-depth deposition) | ×1.5 | `[SOP §2.0]` |
| Calibration target on grey plate | 36 ± 1.5 | `[SOP §6.1]` |
| Valid reflectance window | 20 ≤ R ≤ 90 (linear EBC regime) | `[SOP §7.1]` |
| Required triplicate std. dev. | ≤ 1.5 | `[SOP §6.2]` |
| Order of analysis | Post-weighing → FTIR → HIPS → SSR (SSR is last because subsequent methods are destructive) | `[SOP §1.0]` |
| Document version | EBC SOP **Revision 2.0**, dated 23 Oct 2019, by Crystal Weagle (Washington University, St Louis) | `[Cover page]` |

### 2b. Hybrid Integrating-Plate System (HIPS) — `Fabs`

| Quantity | Value | Source |
|---|---|---|
| Output reported | `Fabs` (light absorption coefficient, **Mm⁻¹**) | `[HIPS Drive CSV — column "Fabs"]` |
| Reported MDL (median across files) | ≈ **1.6 Mm⁻¹** | `[research/spartan/inventory/hips_coverage_by_site.csv]` |
| Raw quantities exposed | T₁, R₁, t, r, tau, intercept, slope (lot-specific blank line) | `[HIPS Drive CSV columns]` |
| MAC used to convert Fabs → public "BC PM2.5" | **10 m² g⁻¹** (i.e. `BC_public = Fabs / 10`) | `[Empirical, derived in research/spartan/inventory/hips_vs_bc_linear_fits.csv — slope = 10.0 ± 0.05 m² g⁻¹ at every site, R² ≈ 1.000]` |
| Method code in public CSV | 221 ("HIPS") | `[Public SPARTAN ChemSpecPM25 — Method_Code]` |
| Wavelength | **OPEN** — not stated in either the EBC SOP or the public CSV header. SPARTAN HIPS uses a near-IR laser of comparable wavelength to IMPROVE (~635 nm), but please confirm with SPARTAN before quoting | `[Open question — RESEARCH_PROGRESS.md notes that raw HIPS records were never released]` |

> **Critical:** the public-dataset `BC PM2.5` is **not an independent
> measurement**. It is computed from the same HIPS `Fabs` divided by a
> network-wide MAC of 10 m² g⁻¹. Use `Fabs` directly when MAC sensitivity
> matters.

---

## 3. IMPROVE sampling filter and HIPS

| Property | Value | Source |
|---|---|---|
| Filter material | PTFE (Teflon) membrane | `[McDade et al. 2009; reproduced in research/ftir_hips_chem/workflows/build_warren_for_warren.py]` |
| Diameter | 25 mm filter holder support screen | `[McDade et al. 2009, Fig. 1]` |
| Deposit pattern | Discrete "dots" formed over each support-screen hole. Each dot ≈ **0.013 inch** (≈ **0.33 mm**) across | `[McDade et al. 2009, Fig. 2 caption — quoted in build_warren_for_warren.py]` |
| HIPS export wavelength | **635 nm** (reflectance and transmittance ratios in the FED parameter table) | `[research/improve_hips_offset/improve_addis_analog_audit.ipynb; improve_fed_rt_proxy_figure2.ipynb]` |
| Stable-calibration epoch | **2003-present** (used as `year >= 2003` filter in all IMPROVE comparisons in this repo) | `[IMPROVE Steering Committee 2015 minutes; White et al. 2016; reaffirmed in White et al. 2024/2025 reanalysis. Cited in research/improve_hips_offset/improve_first_order_loading_range_analysis_executed.ipynb]` |
| Earlier-epoch transition | Apparent HIPS-response change inferred at **2002-2003** | `[White et al. 2016 — DOI 10.1080/02786826.2016.1211615]` |
| Lots used in 2024/2025 reanalysis | Filters from 2003-2016 | `[White et al. 2025 patterned-deposit paper, summarised in research/improve_hips_offset]` |

> The IMPROVE FED public export gives **ratio** fields (final/initial/min
> R and T at 635 nm) — not raw registered HIPS R, T detector outputs, not
> field-blank rows, and not lot-specific blank-line coefficients. Anything
> requiring those raw quantities (e.g. Warren's Figure-2 reproduction)
> needs an off-database data request to UC Davis.

---

## 4. Warren White references — exact values and what they mean

This section catalogues what each Warren White paper actually contains. Where
a number is reported in the published paper that we have transcribed, it is
listed. Where the paper introduces a parameter whose SPARTAN-specific value
has not been published, it is flagged **OPEN** rather than guessed.

### 4a. White et al. 2016 — HIPS calibration / PTFE T-R interpretation

- DOI: `10.1080/02786826.2016.1211615` `[addis_06_hips_ptfe_operating_envelope.ipynb literature map]`
- Topic: HIPS calibration over time and the field-blank "blank-line" used to
  convert raw R, T into `fAbs`.
- Key qualitative finding: HIPS response shifted between ~2002 and 2003;
  reported data are reliably comparable from 2003 onward.
- Quantitative correction parameters: **none transcribed in this repo yet** —
  the executed comparisons (`improve_first_order_loading_range_analysis_executed.ipynb`)
  only use the date filter, not numeric corrections.

### 4b. White et al. 2024/2025 — patterned deposits / "pixelation" / loading

- DOI: `10.1080/10962247.2024.2442634` `[addis_06_hips_ptfe_operating_envelope.ipynb]`
- Topic: at high mass loading, the deposit-pattern dots (from the IMPROVE
  support screen) bias HIPS toward higher `fAbs` than a linear blank-line
  model predicts. Warren's framework parameterises the effect with:

| Parameter | Meaning | IMPROVE value | SPARTAN value |
|---|---|---|---|
| `H` | Support-screen **hole fraction** (open area / total filter area) | Reported in McDade et al. 2009 figures; deposit dot ≈ 0.013" implies a specific H but the exact numeric ratio is not transcribed here | **OPEN** — no published number for the SPARTAN holder. Open ask logged in `research/ftir_hips_chem/RESEARCH_PROGRESS.md` ("Contact SPARTAN / Christopher Ockfort … ask for support-screen hole fraction H, mesh geometry/spec sheet"). |
| `α` | Pixelation / loading exponent in the nonlinearity correction | Reported in White 2025; the executed notebook explicitly says **"correction not applied because SPARTAN-specific H and alpha are unknown"** | **OPEN** — same open ask. |

- Reanalysis-period filters used: 2003-2016 lots from IMPROVE archive
  `[improve_first_order_loading_range_analysis_executed.ipynb]`.
- Warren's Figure 2 is plotted in **(R, T) space at 635 nm**, with field
  blanks defining the **blank line** that all sample R, T pairs are projected
  onto to compute `fAbs`. Reproducing it requires:
  1. Raw registered HIPS R and T detector outputs (not exposed in FED).
  2. Lot-keyed field-blank rows.
  3. Lot-specific blank-line OLS coefficients (intercept `a0`, slope `a1`).
  
  All three are missing from the public IMPROVE FED export and from the
  SPARTAN public CSVs; only the SPARTAN HIPS Drive file partially exposes
  them (it includes `Intercept`, `Slope`, `t`, `r`, `tau` per filter).
  `[research/improve_hips_offset/warren_cena_improve_prep_analysis_executed.ipynb]`.

### 4c. Other Warren-adjacent references

| Reference | DOI | What it's used for |
|---|---|---|
| Bond et al. 1999 | (general filter-photometer artefacts) | Comparison of OLS, through-origin, and robust fits for filter-based absorption |
| Bond & Bergstrom 2006 | (MAC review) | MAC = 7.5 m² g⁻¹ vs the 10 m² g⁻¹ used by SPARTAN HIPS / SSR — sensitivity to network choice |
| Petzold et al. 2013 | (terminology) | Distinguish EC, eBC, BC; SPARTAN public "BC PM2.5" is eBC by Petzold's nomenclature |
| McDade et al. 2009 | (IMPROVE deposit geometry) | Source of the 0.013″ deposit-dot figure and the support-screen photograph |
| Snider et al. 2015 | (SPARTAN nitrate caveat) | Basis for the SOP-level "nitrate is qualitative" statement |
| Pandey, Shetty & Chakrabarty 2019 | `10.5194/amt-12-1365-2019` | PTFE optical-depth / SSA dependence — used in Addis residual diagnostics |

`[All from addis_06_hips_ptfe_operating_envelope.ipynb literature mechanism map]`

---

## 5. Cross-network bridge points

### MAC values used in each chain

- **SPARTAN SSR EBC**: σ_SSR = 0.1 cm² μg⁻¹ ≡ 10 m² g⁻¹, with a 1.5× thickness
  factor applied to ln(R/R₀) `[EBC SOP §2.0]`.
- **SPARTAN HIPS public BC**: `BC = Fabs / 10` (empirically verified network-wide;
  slope = 10.0 m² g⁻¹, R² ≈ 1.000 at 25/27 sites) `[hips_vs_bc_linear_fits.csv]`.
- **IMPROVE**: not converted to BC in the public FED export; `fAbs` is reported
  directly. Conversion to BC requires choosing a MAC and applying any
  Warren 2025 nonlinearity correction.

### Filter-ID matching across products

- The **base filter ID** `SITE-NNNN` ties together every product on the same
  physical filter:
  - SPARTAN public `FilterBased/ChemSpec*` (chemistry + BC),
  - SPARTAN public `FilterBased/ReconstrPM25` (reconstructed mass),
  - SPARTAN HIPS Drive file (`Fabs`, raw T₁/R₁/t/r/tau),
  - SPARTAN FTIR EC (in the local `unified_filter_dataset.pkl`).
- Replicate suffix conventions:
  - `-1` … `-6`: routine samples (replicate slot in the cassette).
  - **`-7`: field blank**. Drop these before any bridge analysis. `[HIPS Drive
    inspection]`.
  - `*-LB*`: lab blanks. Same — drop.

### Sites without HIPS

12 SPARTAN public sites have **no HIPS row** in the Drive file currently:
CADO, CAKE, CALE, CODC, INKA, NGIL, PHMO, SGSU, VNHN, ARCB, USBO, USMC. For
those, the public `BC PM2.5` column is either (a) coming from an earlier
batch we don't have, or (b) absent. Check
`research/spartan/inventory/master_connections.csv` before assuming HIPS is
available. `[research/spartan/inventory/master_connections.csv]`.

---

## 6. Open questions (not yet answered in this repo)

These are the things still needed before a defensible network-bridging
correction can be written. Each is paired with the place we've already logged
the ask.

1. **SPARTAN support-screen hole fraction `H`**. `[RESEARCH_PROGRESS.md
   open-tasks table — "Contact SPARTAN / Christopher Ockfort"]`.
2. **SPARTAN pixelation exponent `α`**. `[RESEARCH_PROGRESS.md — "Ask Ann to
   contact Warren White … clarify alpha and high-loading interpretation"]`.
3. **Exact wavelength of SPARTAN HIPS** (likely 635 nm to match IMPROVE,
   but unconfirmed). `[PAPER_SKELETON_Apr2026.md]`.
4. **SPARTAN HIPS sphere/plate geometry** — needed to know whether Warren's
   IMPROVE-derived correction is geometrically transferable.
5. **Raw IMPROVE Figure-2 R, T detector outputs + field-blank rows + lot
   coefficients** — needed to reproduce Warren's blank-line calibration; not
   in FED. `[warren_cena_improve_prep_analysis_executed.ipynb "Questions for
   Warren White"]`.
6. **Whether SPARTAN has observed HIPS nonlinearity on heavily loaded
   filters** at sites like ETAD, INDH, CHTS. `[PAPER_SKELETON_Apr2026.md]`.

---

## 7. Source documents

- **SPARTAN Determining EBC, Revision 2.0**, Crystal Weagle, Washington
  University in St Louis, dated 23 October 2019. Cover document, 6 pages.
  Local copy uploaded for this session, not committed to the repo (private
  SOP).
- **SPARTAN public CSV headers**, especially the
  `Collection_Description`, `Analysis_Description`, and `Method_Code` columns
  of `data/spartan/raw/FilterBased/ChemSpecPM25/*.csv`.
- **SPARTAN HIPS Drive file**: `SPARTAN_HIPS_Batch1-51.v2.csv` in Drive
  folder `1YVmkYP_0pzs5TQwwbcTQi7gEJ-rTl9LZ`; mirrored to
  `data/drive_bridge/Spartan/` (gitignored).
- **IMPROVE FED public export** (parameter table at 635 nm) — referenced by
  the IMPROVE notebooks in `research/improve_hips_offset/`.
- **Repository notebooks and notes** referenced inline above; key ones:
  - `research/ftir_hips_chem/RESEARCH_PROGRESS.md`
  - `research/ftir_hips_chem/PAPER_SKELETON_Apr2026.md`
  - `research/ftir_hips_chem/COMPLETE_RESEARCH_SUMMARY.md`
  - `research/ftir_hips_chem/addis_06_hips_ptfe_operating_envelope.ipynb`
  - `research/improve_hips_offset/warren_cena_improve_prep_analysis_executed.ipynb`
  - `research/improve_hips_offset/improve_first_order_loading_range_analysis_executed.ipynb`
  - `research/spartan/inventory/HIPS_BRIDGE.md` and `hips_vs_bc_linear_fits.csv`
