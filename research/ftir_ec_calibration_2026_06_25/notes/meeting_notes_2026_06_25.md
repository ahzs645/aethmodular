# FTIR-EC Calibration Review: Rethinking Sample Filtering for Biomass-Burning Aerosols

**Date:** June 25, 2026
**Duration:** ~70 minutes
**Type:** PhD research advising / project check-in
**Source:** Auto-transcribed system audio (`transcript_2026_06_25.json`). Speaker labels are
machine-generated and imperfect, so these notes are organized by topic rather than by speaker.

---

## Attendees

- **Student / presenter** — PhD student presenting the week's calibration work (labeled "Microphone").
- **Advisors** — two research advisors / FTIR-EC collaborators (labeled "Speaker 1/2/3").

**Referenced but not present:** Mona (trained the student on the Shiny app + spectra cleanup),
Sean (maintains the Shiny app + smoke-selection code), Alex (sent the Ethiopia/SPARTAN spectra),
Navid (prior work segregating biomass-burning vs. diesel days), Hossein (student's primary PhD
advisor at UMBC), Andrew "Chad" Weakley (author of a relevant EC / CSN-network paper).

---

## Purpose

Review progress building a PLS calibration for FTIR-derived elemental carbon (EC), decide how to
handle sample filtering for smoke-heavy samples, and plan the next round of analysis on the Adama
and Ethiopia datasets. Closed with a first pass at a bigger conversation about project scope,
timeline, and funding.

---

## Key discussion points

### 1. The calibration was over-filtered
The student presented a clean, tight calibration built by removing samples using absolute residual
ranges that Mona had suggested. The advisors flagged this as **too constrained** — "beautiful, but
too perfect." A calibration that fits these specific samples precisely may not generalize.

- The removal criteria are **absolute numbers, not percentages of concentration**. For fire/smoke
  samples — which can have very high EC — a large absolute residual is a small *relative* error, so
  those high-loading samples are being discarded even though they are well-measured (the Bliss II
  sample near ~180 µg was cited).
- The criteria were "vaguely defined over years and modified by whoever's doing it," built for
  regular IMPROVE samples (lower, more uniform). Smoke samples span a much wider range.
- **Direction:** keep most samples; don't remove anywhere near this many. Rethink filtering
  specifically for smoke samples.

### 2. QC codes vs. calibration suitability
Filter comment codes: **normal**, **XX** (removed / discarded), **QD** (questionable data).

- For *reporting* IMPROVE data, if the last code is "normal" the sample is used and reported.
- **Reporting criteria ≠ calibration criteria.** A sample that flipped normal → flagged → normal
  isn't necessarily a bad calibration sample.
- The Shiny app's Excel export **does not include the comments**, so the student can't see the
  transition history. All samples currently pulled are "normal."

### 3. Shiny app access — build on top, don't recode
- Advisors wary of recoding (subtle-error risk; duplicates Sean's PLS code). Want the student's time
  on the intellectual work, not rewriting infrastructure.
- **Resolution:** don't recode — add a function to surface data the app already has (the comment
  field). Options: Sean adds a feature, Sean shares the R code / underlying data, or a separate
  "research" Shiny app is stood up.
- **Plan:** student emails Sean (CC advisor); Sean raises concerns with the advisor directly.

### 4. Key reframe — the "weird" samples may be the point
Central insight: the hypothesis is that **FTIR is *missing* something** and underestimating EC
relative to Tor (thermal/optical reference). If so, the samples *below* the 1:1 line (FTIR
under-reports vs. Tor) — the "weird," underrepresented ones — may be exactly what the calibration
should be built to capture, **not** filtered out.

Ideas to try:
- **First step: don't filter at all** — keep weird and not-weird together.
- Build a calibration using **only** the removed / below-1:1-line samples.
- Try an **EC-threshold cutoff** (e.g. ≥ ~70 µg, or Ethiopia EC range ÷ 10 as a lower bound) to
  drop very low-signal filters.
- Three notably off samples to inspect: **WICA**, **YOS** (Yosemite), and one other high sample.

### 5. Teflon filter artifacts
Several spectral features are **instrument/substrate artifacts, not aerosol**:
- The sharp double peak (CF / carbon–fluorine bonds in PTFE/Teflon) and the sloping baseline
  (non-zero absorbance at 4000) are from **Teflon scattering**, which increases with loading.
- The highest-EC sample (WICA) did **not** show the highest peaks — reinforcing "something is
  missing."

### 6. Smoke selection is automated, not manual
Correcting a Mona misunderstanding: biomass/smoke samples are **not** hand-picked. Sean runs code
(from a former master's student) that baselines each spectrum, computes peak areas (CH peaks just
below 3000, carbonyl ~1700, alcohol/OH hump 3500–3000), takes ratios, and bins the sample as
biomass if ratios cross thresholds. **Nobody looks at them individually.** Get this code from Sean.

### 7. Adama dataset (5 samples with Tor EC)
Applied biomass-only vs. general calibrations to the 5 Adama samples (Colorado State sampler; the
only 5 with Tor EC ground truth).

- **Compare against Tor, not FTIR-vs-FTIR.** Tor is the target, so the meaningful cross-plots are
  Tor vs. general FTIR-EC and Tor vs. biomass FTIR-EC.
- **Char/soot ratios** (char = biomass ≈ EC1 − OP; soot = diesel ≈ EC2 + EC3) showed **most Adama
  samples are mostly soot/diesel, little char** — except one (≈ JL1269/1270) that moves the opposite
  way under the biomass calibration.
- Contrasts with the expectation (Navid) that summer samples should be biomass-heavy — a showable
  result.

### 8. Char / soot calculation details
- From EC1 (low-temp EC), EC2 (higher-temp), EC3 (graphitic soot), OP (pyrolyzed OC).
- **Tor uses reflectance OP (OPR)**; there is also a transmittance version (OPT). Try OPT for new
  insight. Double-check equations against the source paper.

### 9. Number of PLS components
- Rule: pick the **first major minimum** of the RMSE (RMSECV) plot — and, more importantly, **be
  consistent in *how* you pick it** across all calibrations.
- The student's RMSE plot was jagged (started ~40, rose, dipped) and landed on **17 components**;
  accepted as the first major minimum. Jaggedness may relate to the large sample count (~900).
- Reference: **Weakley** EC paper on the CSN network — second-derivative spectra, only ~4
  components, and shows the first components mostly model the Teflon, not the analyte.

### 10. Literature charcoal reference — parked
Charcoal charring-temperature reference spectra (Dryad) + a spectral map. Useful for understanding
(aromatic C=C and CH evolve at highest temperatures; OH/CH/carbonyl come off earlier), but the
spectral-map analysis is **too abstract to pursue right now**.

### 11. Project scope, timeline & funding (started, to continue 1:1 next week)
- Project has expanded into a **new phase, ≥1 year, possibly two** (learning PLS, FTIR, KBr pellets).
- Conflicts with **Hossein's** expectations (concerned about time + the student's **funding**;
  reportedly wants it done in ~2–3 months).
- The advisor is **not funding the student**, wants to understand the money situation and what the
  student wants, and wants to align **privately with the student first** before Hossein.
- Options: continue the deeper (exciting, high-impact) work, **or** write up a report and return to
  the earlier "walkability" focus. The advisor is enthusiastic but stressed it's the student's call.
- **Scheduling:** meetings irregular through fall — advisor has ~6–7 trips.

---

## Decisions

1. **Stop over-filtering the calibration.** Keep most samples.
2. **Evaluate against Tor EC (ground truth), not FTIR-vs-FTIR.**
3. **Don't recode the Shiny app** — request a feature/data from Sean and build on top.
4. **Shelve the abstract charcoal spectral-map / literature analysis** for now.
5. **Work step by step and resist premature complexity** — simple plots before layered figures.
6. **Adopt a naming scheme** for calibration variants so they stay trackable. (See `NAMING_SCHEME.md`.)

---

## Open questions / parking lot
- What is FTIR actually missing that causes the underestimate vs. Tor? (Central unresolved question.)
- Right MAC value for FABS→EC conversion (6 vs. 10 vs. other) — use **10** for now.
- Whether the biomass-burning calibration is the final approach or one of several to compare.
- What happened on the anomalous Adama day (the odd, high-loading sample)?
- Whether this becomes one paper or several — undecided.
