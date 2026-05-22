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

## 0. Quick side-by-side: SPARTAN vs IMPROVE

This is the at-a-glance comparison. Every value links to the detailed
section below where the citation is recorded.

### Filter hardware

| Property | SPARTAN | IMPROVE |
|---|---|---|
| Filter material | PTFE membrane | Expanded PTFE membrane, ring-mounted |
| Diameter | **25 mm** | **25 mm** |
| Active deposit area `A` | **3.1 cm²** (SOP equation) | not transcribed (similar order; deposit pattern dots) |
| Pore size | **3 μm** | not transcribed in this repo |
| Vendor part number | **PT25DMCAN-PF03A** (Measurement Technology Labs / Mesa Labs) | not transcribed in this repo |
| Support ring | **FEP polymer** (not metal) | metal screen with perforations (see hole fraction) |
| Sampler | AirPhoton SS5i + BGI cyclone PM2.5 inlet | IMPROVE Module A |

### Support screen geometry (the heart of the Warren-2025 correction)

| Property | SPARTAN | IMPROVE |
|---|---|---|
| Support-screen **hole fraction `h`** | **OPEN** — not published; logged as an ask to Christopher Ockfort | **0.638** (63.8% open / 36.2% blocked) — McDade, Dillner & Indresand 2009; adopted in White 2025 |
| Deposit-dot size | not applicable (FEP ring, not perforated screen) | ≈ **0.013 inch** ≈ **0.33 mm** per dot — McDade et al. 2009 |
| Pixelation **relative amplitude** (continuous, 0 = uniform → 1 = fully pixelated) | **OPEN** — no published value | continuous family in White 2025; per-sample fits in the paper, **not transcribed** here |

### Optical-photometer readout

| Property | SPARTAN | IMPROVE |
|---|---|---|
| Wavelength | **OPEN** — not stated in any SPARTAN document we have. Most likely also red ~633 nm, but unconfirmed | **633 nm** — White 2025 abstract verbatim ("absorptance of red (633-nm) light") |
| Wavelength label in public exports | n/a — public CSV reports `Fabs` in Mm⁻¹ without a wavelength field | **635 nm** is the FED export label — treat as a reporting convention for the 633 nm channel |
| Output reported | `Fabs` (Mm⁻¹) in HIPS file; **BC PM2.5 (μg/m³) = Fabs / 10** in the public CSV | `fAbs` plus R/T ratio fields in FED |
| Median MDL | ≈ **1.6 Mm⁻¹** (HIPS Drive file) | not transcribed |
| Raw quantities exposed | T₁, R₁, t, r, tau, lot-specific intercept/slope (HIPS Drive file) | FED gives R/T ratios only; raw R, T detectors + blank rows + lot coefficients are off-database |

### Mass absorption cross-section (MAC) conventions

| Convention | SPARTAN | IMPROVE |
|---|---|---|
| MAC used by network's published BC | **10 m² g⁻¹** (equivalently σ_SSR = **0.1 cm² μg⁻¹**, with a ×1.5 thickness factor in the SSR EBC equation) | none — IMPROVE reports `fAbs`, not BC. The user picks a MAC at analysis time. |
| Empirical check on `BC_public = Fabs / 10` | **Confirmed network-wide**: slope = 10.0 m² g⁻¹, intercept ≈ 0, R² ≈ 1.000 at 25/27 sites with both files (see `research/spartan/inventory/hips_vs_bc_linear_fits.csv`) | n/a |

### Sampling protocol

| Property | SPARTAN | IMPROVE |
|---|---|---|
| Sampling cadence | nominal **1 filter every 9 days** (recent sites sample more frequently — see `research/spartan/inventory/coverage.csv`) | **every third day** at ~150 rural sites — White 2025 abstract |
| Sample integration | 24 h | 24 h |
| Field-blank convention | replicate `-7` in `FilterId` (e.g. `ETAD-0001-7`); also `*-LB*` for lab blanks | not transcribed (raw blanks not in FED) |

### Stable-calibration epoch

| Network | Epoch | Source |
|---|---|---|
| SPARTAN | EBC SOP **Revision 2.0** dated 23 Oct 2019; HIPS Drive file covers 2022-2026 | EBC SOP cover page; HIPS Drive coverage from this repo |
| IMPROVE | **2003-present** (use `year >= 2003` for comparisons) | IMPROVE Steering Committee 2015; White et al. 2016; reaffirmed in White et al. 2025 reanalysis of 2003-2016 lots |

> **Bottom line.** For *filter hardware* (25 mm PTFE, 24 h sampling) the
> two networks look comparable. For *support-screen geometry and pixelation*
> they almost certainly are not — IMPROVE uses a perforated metal screen with
> `h = 0.638`; SPARTAN uses a 25 mm PTFE with an FEP ring and the screen
> behind it has not been characterised. Don't transfer Warren's IMPROVE
> correction onto SPARTAN without that number.

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
| Wavelength | **OPEN** — not stated in the EBC SOP, the public CSV header, or the HIPS Drive file. Most likely **633 nm** to match the IMPROVE photometer (White 2025 abstract), but **unconfirmed** in any SPARTAN-side document we have seen. Do not quote without checking with SPARTAN. | `[Open question — RESEARCH_PROGRESS.md notes that raw HIPS records were never released]` |

> **Critical:** the public-dataset `BC PM2.5` is **not an independent
> measurement**. It is computed from the same HIPS `Fabs` divided by a
> network-wide MAC of 10 m² g⁻¹. Use `Fabs` directly when MAC sensitivity
> matters.

---

## 3. IMPROVE sampling filter and HIPS

| Property | Value | Source |
|---|---|---|
| Filter material | PTFE (Teflon) membrane, ring-mounted, expanded PTFE | `[White et al. 2025 abstract — paraphrased: "ring-mounted membranes of expanded PTFE"]` |
| Diameter | 25 mm filter holder support screen | `[McDade et al. 2009, Fig. 1]` |
| Deposit pattern | Discrete "dots" formed over each support-screen hole. Each dot ≈ **0.013 inch** (≈ **0.33 mm**) across | `[McDade et al. 2009, Fig. 2 caption — quoted in build_warren_for_warren.py]` |
| **Support-screen hole fraction `h`** | **h = 0.638** (i.e. **63.8%** of the illuminated filter area sits over an open hole; **36.2%** sits over a screen strut and is therefore blocked from collecting deposit) | `[McDade, Dillner & Indresand 2009 — explicitly adopted as h = 0.638 in White et al. 2025]` |
| HIPS photometer wavelength | **633 nm** (red light) | `[White et al. 2025 abstract — "absorptance of red (633-nm) light by filter photometry"]` |
| Note on the "635 nm" label in FED | The FED RT export labels its R/T ratio fields at 635 nm. This is the documented FED reporting convention; the physical HIPS measurement is the 633 nm channel reported by White et al. 2025. Treat the 2 nm difference as a labelling artefact, not two different instruments. | `[improve_addis_analog_audit.ipynb says "635 nm reflectance/transmittance ratios"; White 2025 abstract gives 633 nm]` |
| Sampling cadence | **every third day** at **~150 rural sites** | `[White et al. 2025 abstract]` |
| Sample integration | 24 h | `[White et al. 2025 abstract]` |
| Stable-calibration epoch | **2003-present** (used as `year >= 2003` filter in all IMPROVE comparisons in this repo) | `[IMPROVE Steering Committee 2015 minutes; White et al. 2016; reaffirmed in White et al. 2025 reanalysis. Cited in research/improve_hips_offset/improve_first_order_loading_range_analysis_executed.ipynb]` |
| Earlier-epoch transition | Apparent HIPS-response change inferred at **2002-2003** | `[White et al. 2016 — DOI 10.1080/02786826.2016.1211615]` |
| Lots used in 2025 reanalysis | Filters from 2003-2016 | `[improve_first_order_loading_range_analysis_executed.ipynb]` |

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

- **Title**: "A critical review of filter transmittance measurements for
  aerosol light absorption, and de novo calibration for a decade of
  monitoring on PTFE membranes"
- **DOI**: `10.1080/02786826.2016.1211615`
- **Topic**: HIPS calibration over time and the field-blank "blank-line" used
  to convert raw R, T into `fAbs`.
- **Key qualitative finding**: HIPS response shifted between ~2002 and 2003;
  reported data are reliably comparable from 2003 onward.
- **Quantitative correction parameters**: this repo only uses the date filter
  (year ≥ 2003), not numeric corrections. The paper itself contains the
  blank-line slope/intercept (`a0`, `a1`) for each lot-period that we have
  **not transcribed**; reproducing them requires reading the paper directly.

### 4b. White et al. 2025 — patterned deposits / pixelation

- **Title**: "Absorption photometry of patterned deposits on IMPROVE PTFE
  filters"
- **Authors**: Warren H. White, Scott A. Copeland, Jason Giacomo, Nicole P.
  Hyslop, Lindsay M. Kline, William Malm, Sean Raffuse, Bret A. Schichtel,
  Nicholas J. Spada, Christopher D. Wallis, Xiaolu Zhang
- **DOI**: `10.1080/10962247.2024.2442634`
- **Journal**: J. Air & Waste Manage. Assoc.
- **Published online**: 3 February 2025
- **Topic**: a deposit patterned by the perforated support screen ("pixelated"
  deposit) yields a higher absorption cross-section per unit `fAbs` than the
  Beer-Lambert formula assumes for a uniform deposit. The paper extends the
  Beer-Lambert model to a one-parameter family that spans **uniform → fully
  pixelated**.
- **Letter to the editor**: DOI `10.1080/10962247.2025.2473457` (J. Air Waste
  Manag. Assoc. **75**, 2025) raised methodological objections; the authors'
  response is at PMID **40172300** (same journal, pages 351-352, May 2025).
  Neither is open access.

**Numeric values we can verify from the abstract / public snippets:**

| Parameter | Value | Source / phrasing |
|---|---|---|
| Wavelength | **633 nm** ("red light") | abstract — verbatim "absorptance of red (633-nm) light" |
| Network site count | **~150** rural sites | abstract |
| Sampling cadence | **every third day**, 24-h integrated | abstract |
| **Support-screen hole fraction `h`** | **0.638** (63.8%) | adopted from McDade, Dillner & Indresand (2009); used as the upper bound for a fully pixelated deposit ("the fraction of laser power absorbed by a fully pixelated deposit cannot exceed 0.638 of the illuminated filter area") |
| Pixelation parameter (what we've been calling "α") | **a continuous "relative amplitude"** — not a single number | paper text snippet: "intermediate pixelations forming a one-dimensional family parameterized by a relative amplitude" |

**Important nuance.** The pixelation correction in White 2025 is not a single
α value applied to every IMPROVE sample. The family of patterned-deposit
solutions is parameterised by a continuous **relative amplitude** that runs
from 0 (uniform Beer-Lambert deposit) to 1 (fully pixelated, all mass in the
hole footprints). The paper presumably reports either (a) a typical
relative-amplitude value for routine IMPROVE samples or (b) sample-by-sample
fits — but we **have not transcribed those values from the paper text**;
reading the full paper is the only reliable way to get them. The repo's
existing claim that "the correction is not applied because SPARTAN-specific
H and α are unknown" reflects this: we have IMPROVE `h` but no IMPROVE
*sample-specific* amplitude, and we have **nothing** for SPARTAN.

### 4c. Other Warren-adjacent references

| Reference | DOI | What it's used for |
|---|---|---|
| Bond et al. 1999 | (general filter-photometer artefacts) | Comparison of OLS, through-origin, and robust fits for filter-based absorption |
| Bond & Bergstrom 2006 | (MAC review) | MAC = 7.5 m² g⁻¹ vs the 10 m² g⁻¹ used by SPARTAN HIPS / SSR — sensitivity to network choice |
| Petzold et al. 2013 | (terminology) | Distinguish EC, eBC, BC; SPARTAN public "BC PM2.5" is eBC by Petzold's nomenclature |
| McDade, Dillner & Indresand 2009 | (IMPROVE deposit geometry) | Source of `h = 0.638` and the 0.013″ deposit-dot figure |
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
the ask. Items marked **[needs the paper]** can be closed by anyone who can
read the full text of White et al. 2025 — they're verbatim transcription, not
new science.

1. **SPARTAN support-screen hole fraction**. The IMPROVE value `h = 0.638` is
   adopted from McDade 2009. For SPARTAN, no published number exists.
   `[RESEARCH_PROGRESS.md open-tasks table — "Contact SPARTAN / Christopher
   Ockfort"]`.
2. **Pixelation relative-amplitude distribution for routine IMPROVE samples.**
   White 2025 introduces it as a continuous parameter, but the actual values
   the paper fits to its IMPROVE dataset are not in any public snippet I can
   reach. **[needs the paper]**
3. **Pixelation relative-amplitude for SPARTAN samples.** Even if we adopt
   IMPROVE's `h = 0.638` as a proxy (questionable: SPARTAN uses a 25 mm
   PTFE in an AirPhoton SS5i, not the IMPROVE holder), we still need a
   relative amplitude. No published value. `[RESEARCH_PROGRESS.md — "Ask
   Ann to contact Warren White"]`.
4. **Exact wavelength of SPARTAN HIPS**. IMPROVE is 633 nm; SPARTAN is
   strongly implied to be in the same red band but is **not stated** in the
   EBC SOP, the public CSV header, or the HIPS Drive file. `[PAPER_SKELETON_Apr2026.md]`.
5. **SPARTAN HIPS sphere/plate geometry**. Needed to know whether the IMPROVE
   patterned-deposit correction is geometrically transferable to SPARTAN
   instruments.
6. **Raw IMPROVE Figure-2 R, T detector outputs + field-blank rows + lot
   coefficients**. Needed to reproduce Warren's blank-line calibration; not
   in the FED export. `[warren_cena_improve_prep_analysis_executed.ipynb
   "Questions for Warren White"]`.
7. **Lot-period blank-line slope/intercept (`a0`, `a1`) from White 2016**.
   Reported in the paper but not transcribed into this repo. **[needs the
   paper]**
8. **Whether SPARTAN has observed HIPS nonlinearity on heavily loaded
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
