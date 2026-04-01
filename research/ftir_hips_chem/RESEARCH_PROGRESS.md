# Multi-Site Aethalometer Wavelength Analysis — Research Progress & Findings

## Data Pipeline

### QC & Smoothing (justified by `smoothing_comparison.ipynb`)

| Step | What | Why |
|------|------|-----|
| 1. Remove negatives | Set BCc < 0 → NaN | Addis UV has 22% negative values in consecutive blocks; rolling median cannot fix this alone |
| 2. IQR outlier removal | Values > Q99 + 3×IQR → NaN | Addis UV raw max = 4.5 billion ng/m³; caps extreme spikes while preserving real variability |
| 3. 10-min rolling median | Centered, min_periods=3 | Best match to instrument's built-in smoothing (r=0.993, wins 7/11 site-wavelength comparisons). Consistent across all 5 wavelengths and all 4 sites — instrument-smoothed columns are missing for Green/Red/UV at Beijing, Delhi, and JPL |

### Data Summary (after QC + smoothing)

| Site | Rows | Valid IR BCc | Median IR (µg/m³) | Period |
|------|------|-------------|-------------------|--------|
| Beijing | 770,629 | 769,540 | 1.02 | Nov 2021 – Oct 2024 |
| Delhi | 561,380 | 561,380 | 10.96 | Oct 2021 – Jun 2025 |
| JPL | 1,270,523 | 1,270,208 | 0.56 | Mar 2021 – Sep 2024 |
| Addis Ababa | 1,556,717 | 1,556,717 | 4.94 | Apr 2022 – Jun 2025 |

---

## Key Finding 1: Addis Green Channel Anomaly Is Unique (Instrument Issue)

**The identical IR/Blue/Red wavelengths with offset Green seen at Addis do NOT appear at other sites.**

| Site | Green/IR Ratio | Blue/IR | Red/IR | UV/IR |
|------|---------------|---------|--------|-------|
| Beijing | ~1.10–1.15 | ~1.05–1.10 | ~1.00–1.05 | ~1.25–1.45 |
| Delhi | ~1.05–1.10 | ~1.05–1.15 | ~0.98–1.02 | ~1.10–1.45 |
| JPL | ~1.05–1.10 | ~1.05–1.15 | ~1.00–1.05 | ~1.15–1.30 |
| **Addis** | **~0.52–0.55** | **~1.00** | **~1.00** | **~1.50–1.70** |

- Beijing, Delhi, JPL all show expected spectral ordering: UV > Blue > Green > Red ≈ IR
- Addis shows Green at **half** the IR value, with Blue/Red/IR collapsed together
- This is not consistent with any aerosol physics — strongly suggests a hardware or data processing issue specific to the ETAD MA350-0238 instrument
- **This should be raised with Cena at the upcoming joint meeting**

---

## Key Finding 2: AAE Well Below Literature Range at ALL Sites

AAE (UV/IR) medians across all sites: **0.15–0.63**, when literature expects ~1.0 (fossil fuel) to ~2.0 (biomass).

| Site | AAE UV/IR range | AAE Blue/IR range |
|------|----------------|------------------|
| Beijing | 0.15–0.52 | 0.06–0.23 |
| Delhi | 0.08–0.46 | 0.06–0.36 |
| JPL | 0.15–0.35 | 0.09–0.22 |
| Addis | 0.45–0.63 | **−0.01 to 0.08** |

**Likely explanation**: These pkl files contain **eBC (equivalent Black Carbon)**, not raw absorption coefficients (b_abs). The MA350 applies wavelength-dependent MAC values when converting b_abs → eBC, which compresses spectral dependence toward AAE ≈ 0. This is consistent with the literature (ACP 2018, Drinovec et al.).

- Addis Blue/IR AAE near zero or negative confirms the wavelength correction issue
- **Action needed**: Check if raw b_abs data is available, or note this as a limitation. Per Zotter et al. (2017), the 370 nm (UV) channel is more artifact-sensitive than 470 nm (Blue), so Blue/IR should be treated as a formal sensitivity test.

---

## Key Finding 3: Seasonal Structure Confirmed Essential

Aggregated all-season plots hide real patterns. Seasonal breakdown reveals:

### Beijing (seasons per Sun et al. 2022, ACP)
- Winter: highest AAE (~0.52 midday), strongest wavelength divergence
- Summer: lowest AAE (~0.15), least divergence
- Pattern matches literature exactly

### Delhi (provisional seasons; confirm with Amina)
- **Post-monsoon (Oct)**: dramatically higher AAE (~0.44–0.47) and UV/IR ratio (~1.46–1.49)
- **Monsoon (Jul–Sep)**: lowest AAE (~0.08–0.13), consistent with washout
- Aligns with crop-residue burning after October–November harvest (Sebastian et al. 2022, ACP)

### Addis Ababa
- Kiremt evening/late shows highest concentrations (~10 µg/m³ median IR)
- Dry season morning has highest UV/IR divergence
- Divergence timing differs from concentration peaks — not traffic or boundary layer driven

### JPL/Pasadena
- Relatively flat across seasons, lowest BC concentrations
- Summer shows slightly higher concentrations (wildfire influence?)

---

## Key Finding 4: Delta-C Tracks Expected Burning Patterns

Delta-C (UV BCc − IR BCc) as biomass burning tracer:

| Site | Delta-C range (µg/m³) | Pattern |
|------|----------------------|---------|
| Delhi | 0.2–9.0 | Massive in post-monsoon/winter, drops to ~0.2 in monsoon |
| Addis | 2–7 | Highest absolute of any site, consistent with biomass fuel use |
| Beijing | 0.1–0.6 | Moderate, winter highest |
| JPL | −0.1 to 0.2 | Very low, as expected for US urban site |

---

## Key Finding 5: Cross-Site CV Patterns

Wavelength coefficient of variation (spread between wavelengths):

- **Delhi**: Lowest CV (~0.05–0.13) — wavelengths most similar, consistent with a more uniform source mix
- **Addis**: Highest CV (~0.40) — driven by the anomalous Green channel offset
- **Beijing/JPL**: Moderate (~0.12–0.25)

---

## Status vs. Meeting Action Items

### From March 18 Meeting

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Prepare "State of Knowledge" summary | **Pending** | Need to compile confident results slides (no AERONET/wavelength WIP) |
| 2 | Calculate AAE by season (UV/IR) | **Done** | All 4 sites, all seasons. Values below expected range — eBC issue |
| 3 | Check shared drive for hourly data | **Done** | Downloaded Beijing, Delhi, JPL, Addis 1-min resolution data |
| 4 | Request hourly data from Cena | **Not needed** | Already have minute-by-minute data for all sites |
| 5 | Contact Amina for Delhi seasons | **Pending** | Using literature defaults in the meantime |
| 6 | Look up Beijing seasonality | **Done** | Using Sun et al. 2022 (MAM/JJA/SON/DJF) |
| 7 | Re-run wavelength analyses for other sites | **Done** | Full notebook: diurnal, ratios, AAE, Delta-C, CV, seasonal breakdown |
| 8 | AERONET 60% exclusion filter for cross-plots | **Pending** | Need to apply AERONET filtering to FTIR/HIPS/aeth cross-plots |
| 9 | Add Queens College comparison image | **Pending** | Need image file |
| 10 | Remind advisor to connect with Cena | **Pending** | After next meeting |

### From March 4 Meeting

| Task | Status | Notes |
|------|--------|-------|
| Explore AERONET data | **Partially done** | Previous analysis exists; sequential exclusion done |
| Check AERONET at other SPARTAN sites | **Pending** | |
| Literature search on absorption measurements | **Pending** | Filter-based absorption, PSAP, MAAP corrections |
| Read Warren White HIPS/IMPROVE papers | **Pending** | |
| Start paper skeleton | **Pending** | Advisor emphasized: define story first, write results-first |
| Redo plots for 3+1 site grouping | **Done** | Multi-site notebook separates Addis from group |

### From Feb 18 Meeting

| Task | Status | Notes |
|------|--------|-------|
| Redo comparison plots including Delhi | **Done** | All 4 sites in multi-site analysis |
| Distribution plots (EC, HIPS, aeth) | **Pending** | Not yet in current notebooks |
| Plot aeth vs HIPS for all sites | **Pending** | Cross-plot work not yet done in this session |
| Explore AERONET data | **Done** | Existing notebook `addis_04_aeronet.ipynb` |

---

## Immediate Next Steps (Priority Order)

1. **Contact Amina** for Delhi/Punjab seasonal confirmation
2. **AERONET cross-plot filtering** — apply 40% exclusion to FTIR/HIPS/aeth cross-plots (directly tests whether AERONET filtering explains comparison discrepancies)
3. **Paper skeleton** — define the story: "3 sites work well + Addis is anomalous + here's what we tried to explain it + here's what wavelength data tells us"
4. **Prepare "results only" slides** for next advisor meeting
5. **Remind advisor** to connect with Cena after next meeting
6. **Investigate raw b_abs availability** — needed for proper AAE calculation

## Citation-Ready Lines

> "Beijing was divided into spring (MAM), summer (JJA), fall (SON), and winter (DJF) following Sun et al. (2022), while Delhi/Punjab was divided into winter (DJF), pre-monsoon (MAM), monsoon (JJAS), and post-monsoon (ON), following common regional practice."

> "AAE was evaluated using both UV–IR and blue–IR wavelength pairs; two-wavelength AAE is common in the literature, but the 370 nm channel is known to be more artifact-sensitive than 470 nm, so blue–IR was retained as a formal sensitivity test (Zotter et al. 2017, ACP)."

> "Surface–column cross-plots were re-evaluated after excluding weak-coupling days because RH, PBLH, and vertically decoupled aerosol layers can substantially weaken the relationship between near-surface aerosol and column optical properties."

## References

1. Sun et al. (2022), ACP — Beijing seasonal AAE analysis: https://acp.copernicus.org/articles/22/561/2022/
2. Sebastian et al. (2022), ACP — Delhi/Punjab seasonal aerosol: https://acp.copernicus.org/articles/22/4491/2022/
3. Zotter et al. (2017), ACP — Aethalometer model wavelength choice: https://acp.copernicus.org/articles/17/4229/2017/
4. Drinovec et al. (2018), ACP — AE33 corrections: https://acp.copernicus.org/articles/18/6259/2018/
5. AMT (2023) — Micro-aethalometer evaluation: https://amt.copernicus.org/articles/16/2333/2023/
6. AMT (2024) — Aethalometer intercomparison: https://amt.copernicus.org/articles/17/2917/2024/
