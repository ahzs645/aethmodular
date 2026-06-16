# SPARTAN EC — week of 2026-06-16

Kickoff of the **SPARTAN-network EC paper** workstream (advisor meeting Tue 2026-06-09).
Scope has grown: Addis → four focus sites → the whole SPARTAN network. The paper presents
the EC data, highlights how extreme Addis/Ethiopia is (and fractional EC vs. IMPROVE),
argues why the data is good, and candidly addresses the measurement limitations now under
investigation. **AAAR abstract deadline: 2026-07-24** (topic: SPARTAN EC).

This folder continues from `research/spartan/` (SPARTAN↔IMPROVE comparison) and
`research/addis_fabs_ec_deming/` (the fAbs-vs-FTIR-EC errors-in-variables fit + the additive
+7 µg/m³ EC shift that zeros the ~28 Mm⁻¹ intercept).

## Working hypothesis
FTIR may **underestimate EC in charcoal/wood-smoke-heavy environments** because fully-charred
carbon lacks IR-active bonds, and because the FTIR functional-group calibration (trained on
IMPROVE samples, low wood smoke) hasn't "seen" Addis-like OC/EC ratios. HIPS (independent,
field-blank-zeroed) looks fine per Warren White's review, so investigation has shifted to the
FTIR-EC side.

## This week's action items (from the meeting)
1. **FTIR calibration tool (top priority).** Learn the tool; pick the lot with the most Addis
   samples (likely **lot 256** — confirm); build a calibration from **biomass-burning samples
   only** (add similar lots via Mona's PCA plot if too few); apply to Addis samples; compare
   EC from all-sample lot-256 calibration vs. biomass-only. *Expectation: biomass calibration
   → higher EC.* Meet with Mona this week.
2. **Five AMOD quartz-filter samples (Adama, Ethiopia)** with TOR fractions (OC1–4, OP,
   EC1–3, from Alex). Use Han et al. to classify wood/char- vs. diesel/soot-dominated; predict
   EC with general lot-256 vs. wood-smoke calibration; compare spectra to Addis (ask Alex,
   then Mona, for Addis spectra).
3. **Charcoal KBr pellet:** ~0.4–0.5 mg sample / 300 mg KBr; dry in muffle furnace separately
   from others' samples.
4. **Plot fixes:** re-plot site map including **ETBI** (drop the chem-spec filter for it);
   double-check the 1:1 black line and gray IMPROVE-average line on the tau-vs-EC cross plot.
5. Send advisor the SPARTAN terminology for the "thermal EC" measurement (the star markers —
   *not* TOR; no quartz filters at SPARTAN sites).

## Notebooks (authored via `_build_*.py`, then `nbconvert --execute`)

| # | Notebook | Status | What it does |
|---|---|---|---|
| 01 | `01_carbon_methods_audit` | ✅ runs | Confirms from the SPARTAN ChemSpec files that **EC = FTIR** (methods 217/218), BC = HIPS optical (219/221) / HIPS-SSR curve (220), Equivalent BC = SSR (212/214/215/216), and **no thermal/TOR EC exists**. Answers Ann's "thermal vs non-thermal" question. Also flags the lot finding. |
| 02 | `02_biomass_calibration_comparison` | ⏳ scaffold + real baseline | Real lot-251 Addis general-cal FTIR-EC vs HIPS baseline + the general-vs-biomass EC comparison/plot harness. **Blocked on Mona/the tool** for the biomass-only calibration EC (paste into the INPUT cell, set `PLACEHOLDER=False`). |
| 03 | `03_adama_han_char_soot` | ✅ runs (real data) | Han et al. char-EC/soot-EC classifier for the 5 Adama AMOD TOR filters. **Reads `Carbon_concs_Batch54.csv` directly** (no hardcoded numbers); self-verifies against the file's `ECTR`. |
| 04 | `04_new_plots` | ✅ runs | The "New Plots": volume-free cross-plot (tau vs EC-mass) with **corrected reference lines**, fAbs/EC by site (reproduces meeting values), EC/OC by site (FTIR only), fractional EC. |

**Slides:** `spartan_ec_weekly_2026_06_16.pptx` (built by `_build_slides.py` from the figures above) — 10-slide simple update deck: confirmation points, the 4 new plots, the biomass-calibration plan, and the Adama TOR result.

**Adama TOR finding (real Batch54 data):** the 5 Adama quartz filters have **low char-EC/soot-EC (0.02–0.58 → soot-leaning EC)** but **high OC/EC (4.6–7.2 → organic-rich/biomass-influenced)** with large pyrolysis (OP). The two metrics disagree — organic-rich aerosol whose EC speciation skews soot, i.e. the regime where the FTIR functional-group calibration is expected to struggle.

### Key grounded findings this week
- **Lot is 251, not 256.** The lot with the most Addis FTIR-EC samples is **251** (184), then 248 (40). Use 251.
- **No thermal/TOR EC in SPARTAN** — every EC is FTIR; the "thermal" stars on the EC/OC plot were optical BC (HIPS/SSR). Drop or relabel them.
- **fAbs/EC by site reproduces the meeting exactly**: Beijing 10.30, Pasadena 9.41, Delhi 8.53; IMPROVE median ≈ 12.
- **The FTIR/HIPS gap is in the intercept**: lot-251 fit `HIPS(BC-eq)=0.43·EC+2.65` (r=0.89); the +2.65 µg/m³ ≈ 26.5 Mm⁻¹ matches the ~28 Mm⁻¹ Deming intercept.
- The all-sites fractional-EC plot + **ETBI** need the broader public SPARTAN chemspec (only the 4 focus sites have EC in the UC-Davis files; the 27-site HIPS file has ETBI but no EC — exactly why ETBI dropped out).

### Data sources
- SPARTAN: `research/ftir_hips_chem/Filter Data/unified_filter_dataset.pkl` (HIPS_Fabs, EC_ftir, OC_ftir, Volume_m3, DepositArea_cm2; 4 focus sites).
- IMPROVE: `research/ftir_hips_chem/output/improve_high_fabs_comparison/improve_valid_cleaned.csv`.
- Canonical/raw (Google Drive): `…/UC Davis Ann/NASA MAIA/Data/` — `Spartan/SPARTAN_HIPS_Batch1-51.v2.csv` (27 sites incl. ETBI), `Combine csv files/Chemical Speciation/PM2.5/` (4 focus sites), and `Adama TOR/Carbon_concs_Batch54.csv` (the 5 Adama quartz-filter TOR fractions; read directly by notebook 03).

## Literature (Zotero) — references for the write-up
Pulled from the **UCDavis** collection tree in `~/Zotero/zotero.sqlite`. Map of papers → use:

| Topic / notebook | Zotero paper(s) |
|---|---|
| 2003 HIPS cutoff, filter transmittance (01, 04) | White et al. 2016 (DOI 10.1080/02786826.2016.1211615); White et al. 2025 |
| FTIR-EC method = functional groups (01, 02) | Reggente/Takahama/Dillner 2016 (TOR-from-IR); Weakley/Takahama/Dillner (IR + PLS in CSN) |
| SPARTAN EC underestimated in the global south (big picture) | Ren/Dillner et al. 2025 ("Black carbon emissions generally underestimated in the global south") |
| Deposit geometry / face velocity / `h`=0.638 (04) | McDade, Dillner & Indresand 2009 |
| SPARTAN network (intro) | Snider et al. 2015 |
| OC/EC & BC analysis methods (01) | Chow et al. 2007; Bond & Bergstrom 2006 |
| **char-EC vs soot-EC (03)** | **Han et al. — NOT in Zotero yet; add it** |
| **KBr-pellet ratio (charcoal prep)** | **Not in Zotero yet; add the ~0.4–0.5 mg / 300 mg KBr paper** |
