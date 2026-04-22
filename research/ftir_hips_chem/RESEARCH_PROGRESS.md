# Research Progress — Multi-Site Black Carbon Analysis

**Last Updated:** April 22, 2026
**Meetings Covered:** Jan 7 – Apr 14, 2026 (11 meetings)
**Branch:** `claude/review-research-progress-4qorl`

---

## Executive Summary

This document tracks all research progress, findings, and action items across 10 advisor meetings. The project compares three BC measurement methods (HIPS, FTIR EC, aethalometer) across four SPARTAN sites (Beijing, Delhi, JPL, Addis Ababa). Addis Ababa is the anomalous site — it shows a persistent HIPS offset and compressed dynamic range that no other site exhibits.

### The Story So Far (Paper Framing)

**"Three sites work well; Addis is anomalous; here's what we tried to explain it."**

- Beijing + JPL (low concentration): HIPS and FTIR EC agree nearly perfectly (slope ≈ 1.0, MAC = 10)
- Delhi (high concentration): Also works better than Addis, confirming the anomaly is not explained by concentration alone
- Addis Ababa: Two co-occurring problems — large intercept and compressed HIPS range — now framed as an extreme HIPS/filter-loading regime rather than an iron, dust, season, blank, or simple volume artifact

### April 14 Meeting Update: IMPROVE High-Fabs Fork

Ann's key clarification was that the next real test is not another Addis-only exclusion test. The deciding comparison is whether IMPROVE samples occupying ETAD/Addis-like HIPS absorption values also show a distorted `fAbs` vs EC relationship.

Current local result from `improve_high_fabs_comparison.ipynb` using the full FED Excel pull:

| Metric | Result |
|--------|--------|
| Valid positive IMPROVE EC + fAbs rows | 379,697 |
| Date range | 2003-01-03 to 2025-07-30 |
| ETAD local HIPS_Fabs range | 28.09–85.85 Mm^-1 |
| IMPROVE rows in ETAD-like range | 207 rows across 40 sites |
| IMPROVE rows with fAbs >= 70 | 5 rows across 4 sites |
| IMPROVE rows above ETAD max | 3 rows |
| Rows with valid IMPROVE flow + duration for loading | 152,029 overall; 77 ETAD-like |

Interpretation: the broader FED pull proves ETAD occupies a real but very rare IMPROVE HIPS tail. The strict `fAbs >= 70` group is too small for a decisive slope/intercept test, while the ETAD-like group shows weak/compressed `fAbs` vs EC behavior but is partly affected by response-variable selection. Loading-matched IMPROVE groups behave more normally than the fAbs-selected group, so high EC loading alone does not reproduce the full Addis pattern.

April 14 fork as of Apr 22:

- If strict high-fAbs IMPROVE samples eventually show a robust high intercept/compressed relationship, that supports a high-loading HIPS/MAC regime explanation.
- If high-fAbs IMPROVE behaves normally with better targeted samples or raw HIPS diagnostics, the issue points back toward Addis/SPARTAN-specific behavior.
- Current evidence supports the narrower claim: Addis/ETAD sits in a rare extreme HIPS regime, but the high-tail IMPROVE sample count is too small to close the fork.

---

## Data Pipeline

### QC & Smoothing (justified by `smoothing_comparison.ipynb`)

| Step | What | Why |
|------|------|-----|
| 1. Remove negatives | Set BCc < 0 → NaN | Addis UV has 22% negative in consecutive blocks; rolling median alone cannot fix |
| 2. IQR outlier removal | Values > Q99 + 3×IQR → NaN | Addis UV raw max = 4.5 billion ng/m³ |
| 3. 10-min rolling median | Centered, min_periods=3 | Best match to instrument smoothing (r=0.993, wins 7/11 comparisons). Consistent across all wavelengths/sites — instrument-smoothed columns missing for Green/Red/UV at Beijing, Delhi, JPL |

### Data Summary

| Site | Rows | Valid IR BCc | Median IR (µg/m³) | Period |
|------|------|-------------|-------------------|--------|
| Beijing | 770,629 | 769,540 | 1.02 | Nov 2021 – Oct 2024 |
| Delhi | 561,380 | 561,380 | 10.96 | Oct 2021 – Jun 2025 |
| JPL | 1,270,523 | 1,270,208 | 0.56 | Mar 2021 – Sep 2024 |
| Addis Ababa | 1,556,717 | 1,556,717 | 4.94 | Apr 2022 – Jun 2025 |

---

## Key Findings

### Finding 1: Addis Green Channel Anomaly Is Unique (Instrument Issue)

The collapsed IR/Blue/Red wavelengths with offset Green at Addis do NOT appear at other sites.

| Site | Green/IR | Blue/IR | Red/IR | UV/IR |
|------|---------|---------|--------|-------|
| Beijing | ~1.10–1.15 | ~1.05–1.10 | ~1.00–1.05 | ~1.25–1.45 |
| Delhi | ~1.05–1.10 | ~1.05–1.15 | ~0.98–1.02 | ~1.10–1.45 |
| JPL | ~1.05–1.10 | ~1.05–1.15 | ~1.00–1.05 | ~1.15–1.30 |
| **Addis** | **~0.52–0.55** | **~1.00** | **~1.00** | **~1.50–1.70** |

This is not aerosol physics — confirms hardware/data processing issue on MA350-0238. Advisor: "Blue and red should not be identical to IR — that's not an aerosol phenomenon."

### Finding 2: AAE Below Literature Range at ALL Sites

AAE (UV/IR) medians: 0.15–0.63, when literature expects ~1.0 (fossil fuel) to ~2.0 (biomass). Likely because data is eBC (MAC-corrected), not raw b_abs. Raw ATN columns exist for all sites and could be used for proper AAE calculation.

### Finding 3: Seasonal Structure Is Essential

- **Beijing**: Winter highest AAE (~0.52), summer lowest (~0.15). Matches Sun et al. (2022).
- **Delhi**: Post-monsoon (Oct) dramatically higher AAE (~0.44) and Delta-C. Aligns with crop-residue burning.
- **Addis**: Kiremt evening peak distinct; dry season morning has highest UV divergence. Divergence timing differs from concentration peaks — not traffic/boundary layer driven.

### Finding 4: HIPS Anomaly — Addis Is the Only Outlier

Cross-plot statistics (unfiltered):

| Site | HIPS vs FTIR EC | | Aeth vs FTIR EC | |
|------|-------|-----------|-------|-----------|
| | Slope | Intercept | Slope | Intercept |
| Beijing | ~0.6 | ~0.3 | ~1.0 | ~0.2 |
| Delhi | ~0.6 | ~0.5 | ~0.9 | ~0.8 |
| JPL | ~0.8 | ~0.1 | ~1.1 | ~0.1 |
| **Addis** | **~0.3** | **~2.0** | **~0.9** | **~1.5** |

Distribution comparison confirms: Addis HIPS BC is squeezed into ~2–4 µg/m³ while FTIR EC and Aeth span ~2–15 µg/m³.

### Finding 5: Dust Does NOT Explain the HIPS Anomaly

From `aeronet_addis_deep_dive.ipynb`:
- HIPS/FTIR ratio vs coarse AOD: r ≈ **−0.33** (negative)
- Higher dust → HIPS reads **LOWER** relative to FTIR
- If dust were causing the anomaly (absorbing at HIPS 405nm wavelength → inflating Fabs), the correlation would be **positive**
- The negative correlation **rules out** dust as the explanation
- This is a "null result" — the anomaly has a different root cause

### Finding 6: RH Threshold Effect (Feb 11 Meeting Discovery)

- HIPS vs FTIR EC: Agreement improves dramatically below ~50% RH (slope approaches 0.8)
- Aethalometer vs FTIR EC: **Opposite** pattern — works better at HIGH RH
- However, seasonal breakdown showed sample sizes too small to be conclusive
- Advisor conclusion: RH is a "red herring" — it tracks with season but doesn't independently explain the discrepancy

### Finding 7: Source Apportionment Does Not Separate the Problem

PMF analysis (Naveed's data, Addis only):
- All combustion sources (charcoal, wood, fossil fuel) show similar slopes (~0.3–0.4) in HIPS vs FTIR EC
- Threshold filtering by dominant source percentage did not reveal source-specific mechanisms
- Combining charcoal+wood (biomass) vs fossil fuel+marine also did not separate
- Polluted marine behaved differently but in opposite directions for HIPS vs aeth comparisons

---

## Complete Action Items Tracker (All 10 Meetings)

### ✅ COMPLETED

| Item | Meeting | Evidence |
|------|---------|----------|
| Multi-site wavelength analysis (all 4 sites) | Mar 18 | `multisite_diurnal_wavelength_analysis_executed.ipynb` |
| AAE by season (UV/IR and Blue/IR) | Mar 18 | Same notebook, all sites |
| Smoothing methodology justified | Mar 18 | `smoothing_comparison.ipynb` |
| Beijing seasonality from literature | Mar 18 | `src/config/multi_site_seasons.py` (Sun et al. 2022) |
| Get hourly data for all sites | Mar 18 | Downloaded 1-min resolution for all 4 sites |
| Cross-plots all sites (HIPS vs FTIR EC vs Aeth) | Feb 18 | `cross_plots_and_distributions_executed.ipynb` |
| Distribution plots (histograms + violins) | Feb 18 | Same notebook |
| Aeth vs HIPS for all sites | Feb 18 | Same notebook |
| AERONET sequential exclusion on cross-plots | Mar 18 | Same notebook |
| Results-only presentation slides | Mar 18 | `results_summary_executed.ipynb` |
| 3+1 site grouping (Addis separate) | Mar 4 | All notebooks use this framing |
| Iron/EC ratio analysis & thresholds | Jan 7 | `notebooks/analysis/iron_correction_test.ipynb` |
| Source apportionment (PMF, Addis) | Jan 28 | Multiple notebooks: `addis_01_source_apportionment.ipynb`, `dominant_source_comparison.ipynb`, etc. |
| RH sweep analysis (slope vs threshold) | Feb 11 | `notebooks/analysis/meteorology/rh_sweep_seasonal.ipynb` |
| Flow fix before/after analysis | Jan 14 | `research/ftir_hips_chem/FlowFix_BeforeAfter_Analysis.ipynb` |
| Temperature/humidity threshold analysis | Feb 11, Feb 18 | `notebooks/analysis/meteorology/temperature_gradient_analysis.ipynb` + RH notebooks |
| Separate high vs low concentration sites | Jan 21 | Beijing+JPL vs Delhi+Addis grouping in analyses |
| AERONET exploratory analysis | Mar 4 | `addis_04_aeronet.ipynb`, `aeronet_addis_deep_dive.ipynb` |
| Naveed PMF data (GF vs KF resolved) | Feb 4 | KF values used, GF discarded |
| Raw b_abs check | This session | ATN columns exist for all sites (UV/Blue/Green/Red/IR ATN1 & ATN2) |
| Dust interference tested | Mar 4 | r ≈ −0.33 → rules out dust |
| Check Beijing data inputs | Jan 14 | Verified |
| Replot iron with absolute concentration | Jan 14 | Done in iron_correction_test.ipynb |
| Warren White / IMPROVE HIPS loading paper summarized | Apr 14 | Dense White 2025 pixelation/loading paper distilled into presentation notes; correction not applied because SPARTAN-specific H and alpha are unknown |
| IMPROVE high-fAbs comparison notebook | Apr 14 | `research/ftir_hips_chem/improve_high_fabs_comparison.ipynb`; full FED pull loaded and executed Apr 22 |

### ⏳ PENDING (Action Required)

| Item | Meeting | Priority | Notes |
|------|---------|----------|-------|
| Contact Amina for Delhi seasons | Mar 18 | **High** | Advisor will send her email. Using literature defaults for now. |
| Paper skeleton | Mar 4 | **High** | `PAPER_SKELETON_Apr2026.md` exists; updated Apr 22 to include IMPROVE high-fAbs boundary-test figure before final diagnostics. |
| Remind advisor → connect with Cena | Mar 18 | **High** | After next meeting. Show progress + confusing findings. |
| Queens College comparison image | Mar 18 | **Medium** | Need the actual image file |
| Data exclusion registry | Jan 14 | **Medium** | QC infrastructure exists but no formal log of what was excluded and why |
| Literature search (absorption measurements) | Mar 4 | **Medium** | Keywords: filter-based absorption, PSAP, MAAP, photoacoustic |
| Warren White HIPS/IMPROVE papers | Mar 4 | **Medium** | Mass loading on filter vs concentration at high-BC sites |
| Contact SPARTAN / Christopher Ockfort | Apr 14 | **High** | Ask for SPARTAN support-screen hole fraction H, mesh geometry/spec sheet, pixelation photos, and whether they have seen HIPS nonlinearity on heavily loaded SPARTAN PTFE filters |
| Ask Ann to contact Warren White after IMPROVE check | Apr 14 | **High** | Bring ETAD fAbs range plus IMPROVE high-tail results; clarify alpha and high-loading interpretation |
| Cena meeting preparation | Apr 14 | **High** | End-of-April target. Show three methods across four sites, Addis anomaly, negative tests, extreme HIPS regime, and IMPROVE fork |
| AERONET at other SPARTAN sites | Mar 4 | **Low** | Only Addis has data currently |
| Calculate AAE from raw b_abs (ATN data) | This session | **Low** | Would give literature-range AAE values |
| Wind rose analysis | Feb 11 | **Low** | Discussed but no implementation found in codebase |

### ❌ NOT NEEDED / DEPRIORITIZED

| Item | Reason |
|------|--------|
| Purple Air data | Decided not a priority (Jan 28 meeting); overlaps with Cena's team work |
| Iron absorption hypothesis for wavelength stacking | Advisor: "too speculative, unlikely to lead anywhere productive" (Mar 18) |
| Angstrom exponent binning | Deferred, not immediate priority (Jan 7) |
| Middle East emission source investigation | May revisit later (Jan 21) |
| Detailed metal analysis beyond iron | Other metals have much lower absorption efficiencies (Jan 21) |

---

## What We've Tried That Didn't Explain the Addis Anomaly

For the Cena meeting — framed as scientific narrowing:

| # | Approach | Finding | Conclusion |
|---|----------|---------|-----------|
| 1 | Relative humidity thresholds | Slopes converge at high RH but sample sizes too small | "Red herring" — doesn't independently explain discrepancy |
| 2 | Temperature (T_min, T_max) | Biomass burning peaks when T_min is HIGHER (warmer) | Counterintuitive — burning not driven by thermal comfort |
| 3 | Optical saturation | Quadratic ≈ linear fits at Addis | No curvature = no saturation ("math doesn't support the physics word") |
| 4 | Influential point removal | Dataset too tight | No leverage points to remove |
| 5 | OC/EC ratio | No insight | Didn't help explain HIPS-EC discrepancy |
| 6 | Dust (AERONET coarse AOD) | HIPS/FTIR ratio vs coarse AOD: r ≈ −0.33 | **Negative** — higher dust → HIPS underestimates. Rules out dust. |
| 7 | Iron/EC ratio thresholds | High iron/EC points don't deviate as expected | Iron not causing the slope discrepancies |
| 8 | Source apportionment (PMF) | All combustion sources show similar slopes (~0.3–0.4) | Sources don't separate the problem |
| 9 | Combined source grouping | Biomass vs fossil fuel fully overlap | No source-specific mechanism found |
| 10 | All-season aggregation | Patterns hidden | Seasonal breakdown essential |
| 11 | Wavelength stacking (IR≈Blue≈Red) | Unique to Addis, Green at 0.55×IR | Instrument issue, not aerosol physics |

---

## Open Questions for Cena Meeting

1. **Green channel at 0.55×IR**: Known issue with MA350-0238? Correction/recalibration available?
2. **AAE from eBC vs b_abs**: Is the data eBC (MAC-corrected)? Raw ATN exists — should we recalculate?
3. **HIPS anomaly root cause**: Tested humidity, temperature, saturation, dust, iron, sources — all negative. What else could cause persistent intercept + compression at Addis specifically?
4. **Filter loading effects**: Could high BC loading at Addis (~5 µg/m³ median vs ~0.5 at JPL) affect HIPS measurement differently?
5. **Seasonal wavelength data**: Can we get minute-resolution data for Delhi/Beijing to extend the wavelength comparison?

---

## Citation-Ready Lines

> "Beijing was divided into spring (MAM), summer (JJA), fall (SON), and winter (DJF) following Sun et al. (2022), while Delhi/Punjab was divided into winter (DJF), pre-monsoon (MAM), monsoon (JJAS), and post-monsoon (ON), following common regional practice."

> "AAE was evaluated using both UV–IR and blue–IR wavelength pairs; two-wavelength AAE is common in the literature, but the 370 nm channel is known to be more artifact-sensitive than 470 nm, so blue–IR was retained as a formal sensitivity test (Zotter et al. 2017, ACP)."

> "Surface–column cross-plots were re-evaluated after excluding weak-coupling days because RH, PBLH, and vertically decoupled aerosol layers can substantially weaken the relationship between near-surface aerosol and column optical properties."

> "The HIPS/FTIR EC ratio showed a negative correlation (r ≈ −0.33) with coarse-mode AOD, indicating that dust causes HIPS to underestimate rather than overestimate relative to FTIR EC. This rules out dust absorption as the mechanism for the persistent HIPS offset at Addis Ababa."

---

## Notebooks Created This Session

| Notebook | Purpose |
|----------|---------|
| `multisite_diurnal_wavelength_analysis.ipynb` | Diurnal BC by wavelength, ratios, AAE, Delta-C, CV — all 4 sites, seasonal breakdown |
| `smoothing_comparison.ipynb` | Justification for 10-min rolling median (vs raw, vs instrument-smoothed) |
| `cross_plots_and_distributions.ipynb` | HIPS vs FTIR EC vs Aeth cross-plots, distributions, AERONET sequential exclusion |
| `results_summary.ipynb` | Presentation-ready figures with interpretations |

## Config Files Created

| File | Purpose |
|------|---------|
| `src/config/multi_site_seasons.py` | Seasonal definitions for all 4 sites |

---

## References

1. Sun et al. (2022), ACP — Beijing seasonal AAE: https://acp.copernicus.org/articles/22/561/2022/
2. Sebastian et al. (2022), ACP — Delhi/Punjab seasonal aerosol: https://acp.copernicus.org/articles/22/4491/2022/
3. Zotter et al. (2017), ACP — Aethalometer wavelength choice: https://acp.copernicus.org/articles/17/4229/2017/
4. Drinovec et al. (2018), ACP — AE33 corrections: https://acp.copernicus.org/articles/18/6259/2018/
5. AMT (2023) — Micro-aethalometer evaluation: https://amt.copernicus.org/articles/16/2333/2023/
6. AMT (2024) — Aethalometer intercomparison: https://amt.copernicus.org/articles/17/2917/2024/
