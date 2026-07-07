# FTIR-EC Calibration — week of 2026-06-25

Continuation of the **SPARTAN-network EC paper** workstream (`research/spartan_ec_2026_06_16/`),
following the **2026-06-25 advising meeting** (`notes/meeting_notes_2026_06_25.md`). This folder is
the prep workspace for the **presentation on/around 2026-07-02**.

Where the last folder built the first biomass-vs-general calibration and the Adama char/soot
classifier, this meeting was a **review of that work** — and the headline was that the calibration
had been **over-filtered**. The direction for this week is to *stop throwing samples away*, evaluate
everything **against Tor EC (the thermal-optical ground truth), not FTIR-vs-FTIR**, and treat the
"weird" below-the-1:1-line samples as the signal rather than the noise.

## The reframe (why this week matters)
The whole hypothesis is that **FTIR is *missing* something** — it under-reports EC relative to Tor,
worst in charcoal/wood-smoke-heavy aerosol where fully-charred carbon lacks IR-active bonds. If
that's true, the samples sitting **below the 1:1 line** (FTIR < Tor) are **exactly what the
calibration should learn**, not the outliers to remove. The previous "beautiful but too perfect"
calibration removed them using **absolute** residual thresholds — which unfairly discard high-EC
smoke samples, where a big absolute residual is a small *relative* error.

## This week's direction (from the meeting)
1. **Stop over-filtering.** Rebuild keeping most samples; the removal criteria were built for
   low/uniform IMPROVE samples and don't transfer to wide-range smoke samples.
2. **Evaluate against Tor**, never FTIR-vs-FTIR. Tor is what we're trying to reproduce.
3. **Adama (5 samples), step by step:** plot the 5 spectra → Tor-vs-FTIR-EC cross-plots (general &
   biomass) → per-sample char/soot bars → identify which actually have biomass burning.
4. **Ethiopia (ETAD):** plot all spectra; split by Navid's high-biomass vs. diesel months; run the
   calibrations vs. **FABS** (MAC = 10 for now).
5. **Calibration experiments** (name each variant — see `NAMING_SCHEME.md`): no filtering; only
   below-1:1 / only removed samples; EC-threshold cutoff; **OPT vs. OPR**.
6. **Components:** pick the **first major RMSECV minimum**, and be consistent in *how*. Read Weakley.
7. **Infra:** don't recode the Shiny app — ask Sean for the comments export + the smoke-selection
   classifier; check the Davis email for Alex's Ethiopia spectra.

See `TASKS.md` for the full checklist (ordered so the unblockers come first) and
`PRESENTATION_OUTLINE.md` for the slide-by-slide storyboard. **Before building slides, read
`notes/whats_new_vs_last_week.md`** — a direct comparison against last week's deck
(`spartan_ec_2026_06_16/charcoal_spartan_kbr_presentation.pdf`) flagging what was already shown (the
906-sample calibration, the outlier removal, the jagged-K/ensemble recommendation, OPR char/soot, the
charcoal library) so this week advances instead of repeating.

## Notebooks (authored via `_build_*.py`, then `nbconvert --execute`)

| # | Notebook | Status | What it does |
|---|----------|--------|--------------|
| 01 | `01_adama_spectra` | ✅ runs | Plots the 5 Adama PTFE FTIR spectra (raw + baseline-flattened), annotates the **Teflon artifacts** (CF double peak ~1200; sloping baseline / non-zero absorbance at 4000). Sets up "does the highest-EC sample show the highest peaks?" |
| 02 | `02_adama_tor_vs_ftir` | ✅ runs (general) / ⏳ biomass | Cross-plots **Tor EC (ground truth) vs. general FTIR-EC**, joining quartz-TOR to PTFE-FTIR by sample date. Biomass-calibration EC overlays once the biomass predictions are pasted in. **Compare to Tor, not FTIR-vs-FTIR.** |
| 03 | `03_adama_char_soot` | ✅ runs | Per-sample **char (EC1 − OP) vs. soot (EC2 + EC3)** bars from the local Batch54 TOR file, computed **both ways: OP = OPTR (reflectance) and OPTT (transmittance)** — the meeting's OPT-vs-OPR request. |
| 04 | `04_ethiopia_etad_spectra` | ✅ runs | Plots **all ~319 ETAD (Addis/Ethiopia) FTIR spectra**; colors/splits by month so Navid's high-biomass vs. diesel months can be contrasted. Loads directly from the Google-Drive `ETAD FTIR` export. |
| 05 | `05_calibration_variants` | ⏳ scaffold (blocked) | Lays out every calibration variant + the naming scheme + the component-selection rule. **Blocked on** the full IMPROVE+smoke training spectra, the comments export, and Sean's smoke classifier. Ready-to-run cells are clearly marked. |
| 06 | `06_calibration_variants_components` | ✅ runs (real) | **CAL-0…CAL-5** built on the real 906-sample EC data with a **predicted-vs-measured 1:1 graph per variant** (incl. the *inverse* CAL-2 removed-only & CAL-3 below-1:1, each its own "which samples" plot). Component selection (`first_major_min` + Wold's R) and the **2nd-derivative** component-reduction test (Weakley). Shows the "too perfect" trap quantitatively: CAL-1 cleaned has the best R² but the **worst RMSECV**. See `notes/paper_slides_han_weakley.md`. |
| 07 | `07_ethiopia_variant_ec` | ✅ runs (real) | **Applies every CAL variant to the 319 ETAD (Addis/Ethiopia) spectra** (identical 2722-pt grid → direct predict). Box plot + paired-vs-CAL-0. **Result: the calibration choice swings Addis EC ~2×** — cleaned (CAL-1) ≈ 0.61× keep-everything, inverse below-1:1 (CAL-3) ≈ 1.29×; CAL-2/CAL-4 give unphysical negative EC (broken). |
| 08 | `08_adama_hips_crossplots` | ✅ runs (real) | Applies the variants to the **Adama** spectra (interpolated to the training grid) → **New EC vs TOR** crossplot (Adama has no HIPS); and the **ETAD New-EC vs HIPS** crossplot (n=259) joining SPARTAN `Fabs` (EC≈Fabs/10). AGENTS.md style (`y=mx+b` + R² + 1:1). All variants slope>1 vs HIPS/10 → **implied MAC ≈ 4–8**, and CAL-3>CAL-0>CAL-1 ordering matches `07`. |

## Data sources
- **Adama (local, from the predecessor folder):** `../spartan_ec_2026_06_16/data/adama/`
  - `adama_quartz_tor_batch54.csv` — long-format TOR fractions for 5 quartz filters (J1675, J1679,
    J1693, J1701, J1703): OC1–4, EC1–3, **OPTR & OPTT**, ECTR/ECTT, OCTR/OCTT, TCTC.
  - `adama_ptfe_spectra_batch54.csv` — the 5 Adama PTFE (Teflon) FTIR spectra.
  - `adama_ptfe_ftir_batch54.csv` — FTIR-predicted EC/OC for the PTFE filters (+ FilterId↔Barcode,
    LotId, FilterComments) — the "general" FTIR-EC.
  - `adama_quartz_ocec_batch54.csv`, `cal_lot241a_245_{EC,OC}.csv`.
- **Ethiopia / ETAD spectra (Google Drive):**
  `…/UC Davis Ann/NASA MAIA/Data/DAVIS/ETAD FTIR/` — `ETAD_FTIR_spectra.csv` (319 transmission
  spectra, 3998→ cm⁻¹) + `ETAD_metadata.csv` (SiteCode, dates, MassCollectedOnFilter_ug,
  SampleVolume_m3).
- **SPARTAN / IMPROVE** (for calibration training, blocked): see `spartan_ec_2026_06_16/README.md`.

## Literature to read this week
- **Weakley (Andrew Chad Weakley)**, EC / CSN-network paper — second-derivative spectra, ~4
  components, first components model the Teflon not the analyte. *(Add to Zotero UCDavis collection.)*
- **Han et al.** char-EC vs. soot-EC (*Chemosphere* 2007; 2010) — the char/soot definitions in
  notebook 03. *(Still not in Zotero — add it.)*

## How to (re)build
```bash
cd research/ftir_ec_calibration_2026_06_25
/opt/anaconda3/bin/python3.13 _build_01_adama_spectra.py          # writes 01_adama_spectra.ipynb
/opt/anaconda3/bin/python3.13 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 01_adama_spectra.ipynb
# …repeat for 02–05
```
