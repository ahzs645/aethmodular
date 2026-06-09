# Shiny calibration app — what we need to do with it (FTIR EC plan)

*Working document, June 2026. Context: Sean has stood up the calibration Shiny app (it can run
alternative FTIR calibrations and expose OP and EC1). It lives behind the AQRC firewall, so access
requires a campus VPN. This document expands the meeting notes into a concrete task list so the first
session with Mona is productive.*

---

## 1. Why the app matters for our problem

The working hypothesis (post-Warren) is that the **FTIR EC calibration — PLS trained on IMPROVE US
ambient samples — under-reads char/charcoal carbon**. Our own data say:

- Addis fAbs–FTIR EC has slope ≈ 4 with a ~28 Mm⁻¹ intercept (≈ 7 µg/m³ of "missing" EC); Delhi shows
  the same signature (`presentation_offset_story/figures/P1`).
- 100% of Addis filters sit inside IMPROVE's total-mass envelope, but **0% sit inside its EC-fraction
  envelope** (`follow_up_plots/tables/mass_on_filter_improve_overlap.csv`) — the calibration is being
  asked to extrapolate in composition, not in loading.
- IMPROVE itself never reaches Addis-like composition (EC share deciles top out at 0.31 vs Addis's
  0.41–0.44), and the rare samples that get close behave normally (`follow_up_plots/figures/fig14`) —
  so the failure mode, if real, is a **domain gap**, invisible from inside IMPROVE.

The app is the only tool we have that can change the calibration side of this equation. Everything else
(KBR pellets, ETBI, collocated TOR filters) changes the *evidence* side.

## 2. Background you want loaded before the Mona session

- **TOR carbon fractions**: thermal-optical reflectance reports OC1–OC4, EC1–EC3, and **OP** (pyrolyzed
  carbon — organic carbon that charred during the temperature ramp and evolved as EC; it is subtracted
  from EC1 in the reported EC).
- **Char vs soot (Han et al. 2007)**: operationally, **char ≈ EC1 − OP** (low-temperature EC) and
  **soot ≈ EC2 + EC3** (high-temperature EC). Char comes from solid-fuel smoldering (wood, charcoal);
  soot from flaming/high-T combustion (diesel). Their morphology and optics differ. This is why Satoshi
  keeps pointing at **EC1 and OP**: they are the TOR-side fingerprint of exactly the material we think
  FTIR is missing. Char isn't strictly "wood" — both char and soot can come from wood; it's morphology
  and chemistry, not source, that defines them.
- **Debus et al. (IMPROVE-wide FTIR paper)**: the published FTIR EC product already uses **two
  calibrations — a general one and a separate wildfire one** — because one calibration could not cover
  smoke-influenced composition. Precedent: composition regimes already forced a calibration split once.
  Ann has asked Mona to build a **wildfire-only calibration** as the first test.
- **Satoshi's emails** on why OP/EC1 matter ("is FTIR actually seeing the soot or not") — re-read before
  the session; bring the question list from §4.

## 3. Logistics (do these first, they gate everything)

| # | Action | Status / notes |
|---|--------|----------------|
| 1 | Email AQRC IT (help@aqrc.ucdavis.edu) requesting VPN setup, **CC Ann** (they will ask her approval) | Sent — waiting on IT; Ann has campus permission secured |
| 2 | Confirm the Tuesday session with Ann + Mona once VPN works | App is Shiny — no local setup, just firewall access |
| 3 | Before the session: Debus EC section, Satoshi's OP/EC1 email, Han et al. 2007 | Han 2007 = "we are doing the same thing they did, but for FTIR" |

## 4. The task list inside the app (in order)

### T1 — Reproduce the baseline (sanity anchor)
Run the current general IMPROVE calibration on the SPARTAN spectra and confirm we reproduce the numbers
we've been analyzing (ETAD FTIR EC ≈ 4.9 µg/m³ mean, slope-4 + 28 Mm⁻¹ intercept against HIPS).
*If this doesn't reproduce, stop and reconcile versions before anything else.*

### T2 — Wildfire calibration on Addis spectra (Mona's task, our first real test)
Apply the Debus wildfire calibration to ETAD (and INDH) spectra.
- **If EC moves up toward ~12 µg/m³** (the +7 corrected value that matches Schauer-era TOR): strong
  support for the calibration-domain story, and the practical fix may simply be "use a biomass/char
  calibration for these sites."
- **If EC barely moves**: char-specific absorption (the KBR/spectral route) gains weight; wildfire smoke
  ≠ charcoal char.

### T3 — EC1 / OP / char-soot split as calibration targets
Using the IMPROVE samples that have collocated TOR fractions:
1. Train/predict **EC1**, **OP**, **char (EC1−OP)**, **soot (EC2+EC3)** from FTIR spectra — first
   question: are these *predictable at all* from the IR (R², RMSE)?
2. Apply to Addis/Delhi spectra: does the IR say these sites are **char-dominated** (high EC1−OP share)?
3. Per-site **EC1/EC ratio** for IMPROVE slots directly into our rank-plot framework
   (`follow_up_plots` figs 02/03/08 style) — where would Addis fall?

### T4 — Domain diagnostics (quantify the extrapolation)
Ask Mona/Sean whether the app (or the underlying scripts) can output **PLS score-space diagnostics**:
Hotelling T² / leverage / PCA scores of Addis spectra against the IMPROVE calibration cloud.
This is the spectral version of our fig12/fig14 result (0% composition overlap) — it would show
*the calibration itself* flagging Addis as out-of-domain, which is publishable evidence independent of
any re-calibration.

### T5 — The collocated Ethiopia TOR set
The lab has **5 collocated quartz (TOR) + teflon (FTIR + HIPS) filters from a second Ethiopian city**.
When Alex is back: get data + spectra, have Mona compute FTIR OC/EC, then check
(a) is FTIR EC < TOR EC, by how much; (b) what does the TOR **EC-fraction profile** look like —
EC1-dominated would directly confirm char-dominance in ambient Ethiopian aerosol.
*This is the closest thing to "calibration standards in Addis" we currently have.*

### T6 — KBR charcoal spectra meet the calibration
When the KBR pellet spectra exist (Binchotan + Costco lump, replicates, blank-subtracted):
run the **pure charcoal spectra through the general calibration** — what EC does it assign per unit of
actual charcoal mass? A strong under-assignment is the smoking gun. Then compare against the Takahama
cold-start spectra and Naveed's PMF charcoal factor (the resemblance is already noted).

## 5. Decision table — what each outcome means

| Test | Outcome A | Outcome B |
|------|-----------|-----------|
| T2 wildfire cal | EC ↑ toward ~12 → calibration-domain fix plausible | EC unchanged → char ≠ smoke; spectral route |
| T3 char split | Addis predicted char-dominated → coherent story | Not predictable from IR → FTIR genuinely blind to char split |
| T4 diagnostics | Addis flagged out-of-domain → quantified extrapolation | In-domain → problem is elsewhere (back to HIPS?) |
| T5 collocated TOR | FTIR EC ≪ TOR EC, EC1-dominated → confirms under-read | FTIR ≈ TOR → Addis-specific, not Ethiopia-wide |
| T6 KBR spectra | Calibration assigns ~0 EC to charcoal → smoking gun | Assigns full EC → char story dead; HIPS back on the table |

Note the cross-check built into the data already: the tau-vs-EC-mass gap is **1.7× with FTIR EC but
still 1.4× with thermal EC** (`improve_comparison_mass/tables/relationship_fits_mass.csv`) — so even a
perfect FTIR fix should not be expected to close the entire gap. Keep that number in mind when judging T2.

## 6. What to bring back into the repo afterwards

- Wildfire-calibrated ETAD/INDH EC series → drop into `presentation_offset_story` P4 (history bar chart
  gets a sixth bar) and re-run the fAbs–EC fits (does the intercept shrink?).
- IMPROVE per-site EC1/EC + OP/EC tables → new rank-plot figures in `follow_up_plots`.
- Any T4 leverage/score outputs → a "calibration domain" figure pairing with fig12.
