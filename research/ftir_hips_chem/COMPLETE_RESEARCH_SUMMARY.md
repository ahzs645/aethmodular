# Complete Research Summary — Multi-Site Black Carbon Measurement Comparison

**Project:** PhD Research — Comparing BC measurement methods across SPARTAN network sites
**Student:** Ahmed | **Advisor:** Anne (UC Davis) | **Co-supervisor:** Hossein (UNBC)
**Last Updated:** April 1, 2026
**Branch:** `claude/review-research-progress-4qorl`

---

## What We Are Trying to Do

### The Core Research Question

Three instruments measure black carbon (BC) or elemental carbon (EC) at four global sites. They should agree. Do they?

| Instrument | Method | What it measures | Resolution |
|-----------|--------|-----------------|------------|
| **FTIR EC** | Filter-based thermal analysis | Elemental carbon mass | 24-hour filters |
| **HIPS** | Filter-based optical absorption (405 nm) | Light absorption → BC equivalent | 24-hour filters |
| **Aethalometer (MA350)** | Real-time optical absorption (5 wavelengths) | BC concentration | 1-minute |

### The Four Sites

| Site | Code | Location | Concentration Level | BC Median (µg/m³) |
|------|------|----------|--------------------|--------------------|
| Beijing | CHTS | China | Low-moderate | 1.02 |
| JPL/Pasadena | USPA | USA | Low | 0.56 |
| Delhi | INDH | India | High | 10.96 |
| Addis Ababa | ETAD | Ethiopia | Moderate-high | 4.94 |

### Why This Matters

If the instruments don't agree, we need to understand why — otherwise BC measurements worldwide are unreliable. The SPARTAN network uses these instruments globally, so any systematic issues affect air quality and climate research everywhere.

---

## What We Found

### The Big Picture: Three Sites Work, Addis Is Anomalous

**This is the paper's central finding.**

- **Beijing + JPL** (low concentration): HIPS and FTIR EC agree nearly perfectly. Slope ≈ 1.0 with MAC = 10 m²/g. Near-zero intercepts.
- **Delhi** (high concentration): Also works well (slope = 0.71, intercept = 0.93). This is critical — it proves the anomaly is **site-specific**, not **concentration-dependent**.
- **Addis Ababa**: Two co-occurring problems with no identified physical mechanism:
  1. **Large positive intercept** (~2.83 µg/m³) — HIPS reads ~3 µg/m³ even when FTIR EC is near zero
  2. **Compressed dynamic range** — HIPS squeezed into ~2-4 µg/m³ while FTIR EC spans ~2-15 µg/m³

### Cross-Plot Statistics (After Filtering)

| Site | Comparison | n | Slope | Intercept | R² |
|------|-----------|---|-------|-----------|-----|
| Beijing | HIPS vs FTIR EC | 144 | 0.60 | 0.54 | 0.66 |
| Delhi | HIPS vs FTIR EC | 57 | 0.71 | 0.93 | 0.78 |
| JPL | HIPS vs FTIR EC | 60 | 0.84 | 0.06 | 0.30 |
| **Addis** | **HIPS vs FTIR EC** | **190** | **0.40** | **2.83** | **0.76** |

---

## What We Did (This Session)

### Data Pipeline Built

1. **Downloaded** minute-resolution aethalometer data for all 4 sites (4.16 million rows total)
2. **Built QC pipeline**: Remove negatives → IQR outlier removal → 10-min rolling median smoothing
3. **Justified smoothing**: Compared raw vs instrument-smoothed vs 5/10/15/30-min rolling median. 10-min best matches instrument (R²=0.993, wins 7/11 site-wavelength comparisons)
4. **Applied exclusion filters** from outliers.py: below-MDL, negative EC, extreme outliers, measurement mismatches (126 points excluded from 1416 total)

### Analyses Completed

#### 1. Multi-Site Diurnal Wavelength Analysis
**Notebook:** `multisite_diurnal_wavelength_analysis.ipynb`

Extended the Addis-only wavelength analysis to all 4 sites with site-specific seasonal definitions.

**Key Finding — Addis Green Channel Anomaly Is Unique:**

| Site | Green/IR Ratio | Expected? |
|------|---------------|-----------|
| Beijing | ~1.10–1.15 | Yes |
| Delhi | ~1.05–1.10 | Yes |
| JPL | ~1.05–1.10 | Yes |
| **Addis** | **~0.52–0.55** | **NO — half of IR** |

No other site shows this. The advisor said: "Blue and red should not be identical to IR — that's not an aerosol phenomenon." This confirms an instrument or data processing issue specific to the ETAD MA350-0238.

**Other wavelength findings:**
- AAE (UV/IR) at all sites: 0.15–0.63 (literature expects 1.0–2.0). Likely because data is eBC (MAC-corrected), not raw absorption coefficients.
- Delta-C tracks expected biomass burning patterns: Delhi post-monsoon highest, Addis year-round elevated, JPL minimal.
- Seasonal breakdown is essential — aggregated data hides real patterns at all sites.

#### 2. Smoothing Comparison
**Notebook:** `smoothing_comparison.ipynb`

Compared raw BCc, instrument-smoothed BCc, and rolling median variants across all sites.

**Key Finding:** 10-min rolling median is the only consistent approach because instrument-smoothed columns are missing for Green/Red/UV at Beijing, Delhi, and JPL. Only Addis has all 5 wavelengths smoothed by the instrument.

#### 3. Cross-Plots and Distributions
**Notebook:** `cross_plots_and_distributions.ipynb`

Three cross-plots (HIPS vs FTIR EC, Aeth vs FTIR EC, HIPS vs Aeth) for all 4 sites with:
- Per-site regression lines showing slope, intercept, R²
- Exclusion filters applied per outliers.py registry
- Distribution plots (histograms + violin plots) showing HIPS compression at Addis

**Delhi outlier fixed:** INDH-0172-4 (EC=60.6 µg/m³) was 5x higher than any other Delhi point and was dragging Delhi's slope from 0.71 to 0.38. Added to exclusion registry.

#### 4. AERONET Sequential Exclusion
**Notebook:** `cross_plots_and_distributions.ipynb` (section 4)

Downloaded AERONET Level 1.5 daily AOD data for Addis (AAU Jackros station). Matched 157 filter days with AERONET AOD. Applied sequential exclusion (0/20/40/60% lowest AOD removed) to the three cross-plots.

**Purpose:** Test whether removing surface-column decoupled days improves instrument comparisons at Addis.

#### 5. Seasonal Definitions
**Config:** `src/config/multi_site_seasons.py`

| Site | Seasons |
|------|---------|
| Addis | Dry (Oct-Feb), Belg (Mar-May), Kiremt (Jun-Sep) |
| Beijing | Spring (MAM), Summer (JJA), Autumn (SON), Winter (DJF) — Sun et al. 2022 |
| Delhi | Winter (Nov-Feb), Pre-monsoon (Mar-Jun), Monsoon (Jul-Sep), Post-monsoon (Oct) — provisional |
| JPL | Winter (DJF), Spring (MAM), Summer (JJA), Autumn (SON) |

#### 6. Comprehensive Progress Tracking
**Document:** `RESEARCH_PROGRESS.md`

Cross-referenced all action items from 10 advisor meetings (Jan 7 – Mar 18, 2026). Tracked 23 completed, 10 pending, 5 deprioritized.

#### 7. Presentations
- **`weekly_progress_presentation.pptx`** (18 slides): Everything done this week including WIP
- **`state_of_knowledge_presentation.pptx`** (15 slides): Confident results only, with presenter notes

---

## What We Tried That Didn't Explain the Addis Anomaly

11 hypotheses tested — all negative. Framed as scientific narrowing for the Cena meeting:

| # | Approach | Result | Meeting |
|---|----------|--------|---------|
| 1 | Relative humidity thresholds | Sample sizes too small when split by RH | Feb 11, Feb 18 |
| 2 | Temperature (T_min, T_max) | Counterintuitive: biomass burning peaks when warmer | Feb 18 |
| 3 | Optical saturation | No curvature — quadratic ≈ linear fits | Feb 18 |
| 4 | Influential point removal | Dataset too tight, no leverage points | Feb 18 |
| 5 | OC/EC ratio analysis | No additional insight | Feb 18 |
| 6 | Dust (AERONET coarse AOD) | r = −0.33 (negative) — **RULES OUT** dust | Mar 4 |
| 7 | Iron/EC ratio thresholds | Iron not causing slope discrepancies | Jan 7–28 |
| 8 | Source apportionment (PMF, 5 factors) | All combustion sources show same slopes (~0.3-0.4) | Jan 28, Feb 4, Feb 11 |
| 9 | Combined source grouping (biomass vs fossil) | Fully overlap — no separation | Feb 4 |
| 10 | All-season aggregation | Masks patterns but doesn't explain | Mar 18 |
| 11 | Flow fix before/after | No significant difference at JPL | Jan 14 |

### Important Clarification on Dust

The HIPS/FTIR ratio vs coarse AOD correlation of r ≈ −0.33 means: **higher dust → HIPS reads LOWER relative to FTIR**. If dust were causing the HIPS anomaly (absorbing at 405nm → inflating Fabs), the correlation would be positive. The negative correlation **rules out** dust as the mechanism. This is a "null result" — valuable because it eliminates a hypothesis using independent data (AERONET).

### The RH Discovery (Feb 11) — Then Retracted (Feb 18)

An interesting pattern was found: HIPS performs better at low RH (<50%), aethalometer performs better at high RH. But when broken by season, sample sizes were too small to draw conclusions. The advisor concluded RH is a "red herring" — it correlates with season but doesn't independently explain the discrepancy.

---

## What We Know About the Data Quality

### Aethalometer Data
- All sites have minute-resolution data with 5 wavelengths (UV, Blue, Green, Red, IR)
- QC pipeline: negatives removed, IQR outliers capped, 10-min rolling median smoothing
- Addis UV channel: 22% of raw values are negative (worst of any site/wavelength)
- Addis Green channel: reads at 0.55x IR — unique instrument issue
- Raw ATN columns exist for all sites → could recalculate absorption coefficients for proper AAE

### Filter Data
- 1416 total filter samples across 4 sites
- 126 excluded (MDL + outliers) → 1290 clean samples
- Specific exclusions documented in `outliers.py`:
  - EC < 0.5 µg/m³ (below MDL): 121 points
  - Negative EC values: 6 points
  - Delhi INDH-0172-4 (EC=60.6): 1 point (likely contamination)
  - Beijing extreme aeth: threshold at >4 µg/m³
  - Delhi high aeth + low EC mismatch
  - JPL pre-flow-fix outliers

### AERONET Data
- Level 1.0, 1.5, and 2.0 available for Addis (AAU Jackros station)
- Daily and all-points (15-min) resolution
- 700 days of data, 157 matched with filter samples
- Level 2 limited due to cloud cover → using Level 1.5

---

## Files Created This Session

### Notebooks (`research/ftir_hips_chem/`)

| File | Purpose | Status |
|------|---------|--------|
| `multisite_diurnal_wavelength_analysis.ipynb` | Diurnal BC by wavelength, ratios, AAE, Delta-C, CV — all 4 sites, all seasons | Executed |
| `smoothing_comparison.ipynb` | Raw vs instrument-smoothed vs rolling median justification | Executed |
| `cross_plots_and_distributions.ipynb` | Cross-plots, distributions, AERONET sequential exclusion | Executed |
| `results_summary.ipynb` | Quick figure viewer for presentation prep | Executed |

### Config (`src/config/`)

| File | Purpose |
|------|---------|
| `multi_site_seasons.py` | Seasonal definitions for all 4 sites with helper functions |

### Documents & Presentations

| File | Purpose |
|------|---------|
| `RESEARCH_PROGRESS.md` | Action item tracker across all 10 meetings |
| `COMPLETE_RESEARCH_SUMMARY.md` | This document — full narrative of everything done and next steps |
| `output/weekly_progress_presentation.pptx` | 18-slide weekly progress deck (includes WIP) |
| `output/state_of_knowledge_presentation.pptx` | 15-slide confident results deck with presenter notes |

### Updated Files

| File | What Changed |
|------|-------------|
| `scripts/outliers.py` | Added Delhi INDH-0172-4 exclusion, `high_ec` and `negative_ec` threshold types |

### Data Downloaded (`data/`)

| File | Size | Content |
|------|------|---------|
| `df_cleaned_Beijing_manual_BCc.pkl` | 1.3 GB | Beijing 1-min aethalometer (770K rows) |
| `df_cleaned_Delhi_manual_BCc.pkl` | 1.1 GB | Delhi 1-min aethalometer (561K rows) |
| `df_cleaned_JPL_manual_BCc.pkl` | 548 MB | JPL 1-min aethalometer (1.27M rows) |
| `df_jacros_cleaned_API_and_OG_manual_BC_all_wl.pkl` | 3.8 GB | Addis 1-min aethalometer (1.56M rows) |
| `aeronet/daily/` | 18 MB | AERONET L1.0/1.5/2.0 daily (AOD, SDA, inversions) |
| `aeronet/all/` | 40 MB | AERONET all-points (15-min resolution) |

---

## Next Steps

### HIGH PRIORITY (Before Next Advisor Meeting)

1. **Paper skeleton**
   - The advisor has asked for this multiple times (Mar 4, Mar 18)
   - Format: Define conclusions first, then build story toward them
   - Follow Nagendra's approach: brief intro, methods listed not detailed, expected outcomes
   - Do NOT write about everything done — focus on what tells the story
   - Proposed framing: "Three sites work well + Addis is anomalous + here's what we tried + anomaly remains open"

2. **Contact Amina for Delhi seasonal definitions**
   - Navid's new grad student in Pakistan
   - Ask about seasonal definitions for Delhi/Punjab: winter, pre-monsoon, monsoon, post-monsoon months
   - Currently using literature defaults (Sebastian et al. 2022)
   - Advisor will send her email and CC Navid

3. **Remind advisor → schedule Cena meeting**
   - Advisor has a proposal due 2 days after next meeting
   - Plan to connect with Cena shortly after the biweekly meeting
   - Show: confident results + 11 approaches tried + open questions
   - Ask about: Green channel issue, HIPS root cause, filter loading effects

4. **Queens College comparison image**
   - Advisor specifically asked for Sean Zeta's Queens College plot showing wavelength divergence
   - Add to the wavelength analysis for visual comparison
   - Need the actual image file

### MEDIUM PRIORITY

5. **Data exclusion registry** — Formalize the exclusion log (outliers.py has the data but TODO dates need filling with actual values)

6. **Literature search: absorption measurements**
   - Keywords: "filter-based absorption measurement," PSAP, MAAP, photoacoustic
   - Look for corrections applied to intercept/slope issues similar to ours

7. **Warren White HIPS/IMPROVE papers**
   - Look for comparisons at high-concentration sites
   - Check mass loading on filter vs concentration — could filter overloading cause the compression?

8. **Calculate AAE from raw b_abs**
   - Raw ATN columns exist for all sites (UV/Blue/Green/Red/IR ATN1 & ATN2)
   - Current AAE from eBC is 0.15-0.63 (below literature range of 1.0-2.0)
   - Recalculating from raw absorption coefficients should give literature-range values
   - Formula: b_abs = (ATN × A) / (Q × t × C) where A=spot area, Q=flow, t=timebase, C=correction

### LOW PRIORITY / DEPRIORITIZED

9. **AERONET at other SPARTAN sites** — Only Addis has AERONET data currently
10. **Purple Air data** — Decided not a priority; overlaps with Cena's team work
11. **Iron absorption hypothesis for wavelength stacking** — Advisor: "too speculative"
12. **Wind rose analysis** — Discussed but implementation not found in codebase
13. **Multi-site PMF** — Addis done, other sites would need source apportionment data

---

## Key References

1. Sun et al. (2022), ACP — Beijing seasonal AAE analysis
2. Sebastian et al. (2022), ACP — Delhi/Punjab seasonal aerosol
3. Zotter et al. (2017), ACP — Aethalometer model wavelength choice (370nm artifact-sensitive)
4. Drinovec et al. (2018), ACP — AE33 corrections and eBC vs b_abs
5. AMT (2023) — Micro-aethalometer evaluation showing UV channel sensitivity
6. AMT (2024) — Aethalometer intercomparison
7. Ohm et al. — AERONET sequential exclusion methodology (Level 1.5)
8. Park et al. (2025) — SPARTAN/AERONET composition-scattering study finding Addis surface-column coupling issues

---

## Open Questions for Cena

1. **Green channel at 0.55×IR on MA350-0238**: Is this a known issue? Is there a correction or recalibration available? No other site shows this.

2. **AAE from eBC vs b_abs**: Is the data in the pkl files eBC (MAC-corrected) or raw absorption? The AAE values (0.15-0.63) are far below the expected range (1.0-2.0), which would be consistent with MAC compression.

3. **HIPS anomaly root cause**: We've tested humidity, temperature, optical saturation, dust, iron, source composition, seasonality, influential points, OC/EC ratios, and flow corrections. All negative. What else could cause a persistent intercept + compression at one specific site?

4. **Filter loading at high BC**: Could the high BC loading at Addis (~5 µg/m³ median vs ~0.5 at JPL) affect the HIPS optical measurement differently? Is there a known nonlinearity at high filter loadings?

5. **Minute-resolution data for other sites**: Can we get the same temporal resolution for Delhi and Beijing to extend the multi-site wavelength analysis beyond what we have in the 9am-resampled datasets?
