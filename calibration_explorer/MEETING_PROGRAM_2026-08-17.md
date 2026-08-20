# The July-17 meeting program, run end-to-end through the explorer (2026-08-17)

Every number below came out of `calibration_explorer` (same shared-script math as the
notebooks), fixed 190-filter Addis cohort, MAC = 10. "s-h-o" = Site-held-out protocol
(grouped 5-fold CV, first-major-minimum, disjoint fit); "app" = Calibration-app
protocol (interleaved 10-fold, within-5%, fit on all). Deming λ\*=2.96.

## The app reproduces the meeting's own slide numbers

The intercepts quoted in the meeting were Deming at MAC 10 under the app protocol —
and the explorer lands on them:

| Meeting slide | Meeting value | Explorer (app protocol, Deming) |
|---|---|---|
| Entire IMPROVE network | −5.76 | **−5.76** (k=27) |
| Biomass smoke | "negative 10" | **−10.16** (k=19) |
| Spectral analogs (k=9) | −9.35 | **−9.37** (sweep row k=9) |
| Lowest-OC/EC | −6.74 | **−6.74** (k=17) |
| Lowest-OC/EC + AIRSpec | −2.17 | **−2.17** (k=19) |
| Ethiopia-shaped | −4.95 | −5.16 (k=18; k differs from the slide's) |

## The setup matrix, both protocols

| Setup | s-h-o k | s-h-o OLS | s-h-o Deming | held-out TOR R² | app k | app OLS | app Deming |
|---|---|---|---|---|---|---|---|
| Entire network (13,010) | 15 | 1.65x−3.44 | 1.94x−4.82 | 0.70 | 27 | 1.95x−4.05 | 2.30x−5.76 |
| Biomass-smoke (906) | 4 | 0.50x−0.99 | 0.53x−1.11 | 0.54 | 19 | 2.43x−6.35 | 3.21x−10.16 |
| Ethiopia-shaped (300) | 10 | 1.59x−3.67 | 1.85x−4.96 | **0.00** ⚠ | 18 | 1.69x−3.56 | 2.01x−5.16 |
| Spectral analogs (477) | 4 | 2.57x−6.56 | 3.22x−9.74 | 0.62 | 19 | 3.01x−7.02 | 3.81x−10.96 |
| Lowest-OC/EC (800) | 6 | 1.59x−3.22 | 1.81x−4.34 | **0.91** | 17 | 2.15x−4.59 | 2.59x−6.74 |
| Lowest-OC/EC + AIRSpec | 5 | 0.86x−1.62 | 0.95x−2.09 | 0.90 | 19 | 0.96x−1.65 | 1.07x−2.17 |

(The analog cohort resolves to 477 — see the standing 477/500 rank-vs-locked
mismatch flagged by the startup check.)

## Meeting item 1 — selection on baseline-corrected spectra (Ethiopia-shaped)

- **Does the pool change?** Barely: raw-space and corrected-space top-300 share
  **285/300 filters** (95%) — the overlap table's answer to "see whether the pool of
  selected samples actually changes".
- **Does the calibration change?** More than the membership suggests:

| Ethiopia-shaped 300 (s-h-o) | k | OLS | Deming | held-out TOR R² |
|---|---|---|---|---|
| select raw · calibrate raw | 10 | 1.59x−3.67 | 1.85x−4.96 | 0.00 |
| select corrected · calibrate raw (Satoshi's suggestion) | 20 | 1.57x−3.11 | 1.86x−4.50 | **0.63** |
| select corrected · calibrate corrected | 4 | 0.30x−0.65 | 0.30x−0.68 | 0.40 |

The corrected-space selection **rescues the cohort's held-out TOR skill** (0.00 →
0.63) with a slightly better intercept — the raw-selection Ethiopia-shaped model's
failure of the TOR test (flagged in ftir_21) appears to be a selection-space artifact,
not something intrinsic to shape-matching. Worth promoting to a notebook.

## Baselined vs non-baselined calibrations ("try a few cases")

AIRSpec-corrected calibration, s-h-o, vs its raw counterpart:

| Cohort | raw intercept (OLS) | AIRSpec intercept (OLS) | AIRSpec slope | held-out R² raw → AIRSpec |
|---|---|---|---|---|
| Biomass-smoke | −0.99 | −0.89 | 0.47 | 0.54 → 0.40 |
| Ethiopia-shaped 300 | −3.67 | −0.57 | 0.29 | 0.00 → 0.24 |
| Spectral analogs | −6.56 | −1.05 | 0.61 | 0.62 → 0.59 |
| Lowest-OC/EC 800 | −3.22 | −1.62 | 0.86 | 0.91 → 0.90 |

Baselining shrinks every intercept, but on the smoke-lineage cohorts it collapses the
slope to 0.3–0.6 (the ftir_08/ftir_13 smoke-collapse pattern) — **lowest-OC/EC +
AIRSpec remains the only setup with both a small intercept and a usable slope + TOR
skill.**

## SG second derivative (ftir_20 parameters)

| Cohort (s-h-o) | k | OLS | held-out R² | note |
|---|---|---|---|---|
| Lowest-OC/EC 800 | 9 | 1.36x−2.72 | 0.90 | Addis R² drops to 0.59 — derivative and AIRSpec are alternatives, not additive (ftir_20's finding, reproduced) |
| Biomass-smoke | 2 | 0.44x−0.61 | 0.39 | Addis R² 0.28 — derivative doesn't fix smoke |
| Ethiopia-shaped 300 | 7 | 0.59x−1.11 | 0.15 | weak |

## Selection cutoffs ("somewhat more, somewhat less")

| Cohort | less | default | more |
|---|---|---|---|
| Ethiopia-shaped | 200: −2.01, held-out 0.34 | 300: −3.67, held-out 0.00 | 450: −3.98, held-out 0.00 |
| Spectral analogs | 350: **−3.21, held-out 0.79** | 500: −6.56, held-out 0.62 | 650: −6.17, held-out 0.61 |
| Lowest-OC/EC | 500: −5.43 (k=3), held-out 0.65 | 800: −3.22, held-out **0.91** | 1200: −3.49, held-out 0.85 |

Two notable: **analogs-350 behaves far better than analogs-500** (intercept −3.2, TOR
R² 0.79 — the meeting's "technically 400 on how I chose" instinct was right, the jump
past ~400 adds harmful samples), and OC/EC-800 is confirmed as the sweet spot.

## The analog k question ("try 21 instead of 9")

k swept 4→24, both protocols: the intercept does **not** improve with more components
— OLS drifts −6.2 → −6.9 and back, Deming −9.3 → −10.9; held-out TOR R² peaks at
k=9 (0.72) and declines after. Going to 21 does not fix the spectral analogs.

## Overlaps (the "are these picking similar filters?" table)

At the default cutoffs: Ethiopia-shaped ∩ lowest-OC/EC = **1 filter**, smoke ∩
lowest-OC/EC = 3 (both match Ann's committed analysis exactly), analogs ∩ OC/EC = 14,
eth ∩ analogs = 36. Shape-based and composition-based selection pick essentially
disjoint filters; the app's Selection tab now shows this table live at any cutoffs.

## The email to-do list, item by item (second pass)

**Baseline correction before spectral matching — now complete for BOTH spectra-based
variants.** The analog machinery (ftir_09's IMPROVE-HIPS PLS score space + VIP-weighted
spectral RMSE) was re-run on the AIRSpec-corrected caches (model refit on corrected
spectra: k=17, n=916, HIPS-τ target; the raw recipe's offset-correction step dropped
since corrected spectra are already baselined). Results, calibrating on **raw** after
selection (Satoshi's instruction):

| Analogs 500 (s-h-o, calibrate raw) | k | OLS | Deming | held-out TOR R² |
|---|---|---|---|---|
| select raw (eligibility-first top 500) | 4 | 2.48x−6.35 | 3.09x−9.32 | 0.37 |
| **select corrected (eligibility-first top 500)** | **15** | **1.63x−3.17** | 1.94x−4.68 | **0.70** |

- **Does the pool change?** Completely: raw and corrected analog selections share
  **4/500 filters**. (Ethiopia-shaped shared 285/300.) The raw analog similarity was
  dominated by Teflon background — Satoshi's "the similarity is more to do with the
  Teflon" point, demonstrated. The corrected selection also lands closer to the other
  cohorts' behaviour (17 shared with lowest-OC/EC-800, now tied with raw, and intercept
  −3.17 is in lowest-OC/EC territory).
- The corrected-selection + corrected-calibration and cutoff-350 values from the
  original pass used cutoff-before-eligibility semantics and must be regenerated.
  See `ANALOG_CUTOFF_AUDIT_2026-08-18.md` for the corrected top-500 comparison.

**Number of components / CV schemes / cutoffs / spectra plots / baselined-vs-not** —
all were run in the first pass, but the analog-specific dual-CV curves and cutoff sweep
need a durable eligibility-first refresh before circulation. The spectra and non-analog
results are unaffected by this counting correction.

**Clustering "possible later step"** — implemented: the Selection tab's spectra panel
has a **sub-types (k-means 3)** mode showing cluster medians within the current cohort
against the Addis median.

**Filter lot question** — Mona's PCA plots remain a human follow-up, but the in-hand
half is now testable with the app's new **Lot** filter (from `ftir_catalog.LotNumber`),
and the first pass overturns a meeting premise:

- The 13,010-filter pool is **1,299 lot-248 + 11,415 lot-251** (+~300 other) — it is
  not "all lot 248", and the smoke-906 cohort is 98% lot 251 (887 filters; too few
  248s to even fit).
- The erratic interleaved RMSE curves are **not lot-specific between 248 and 251**:
  roughness is comparable in both single-lot pools (lot 248: 15/29 upticks; lot 251:
  12/29, same as combined). Single-lot pools under the app rule degenerate to k=1–2
  (the "average spectrum predicts best" pathology), so the jaggedness survives lot
  isolation. Testing lots that *don't* cluster with 248/251 still needs Mona's PCA +
  spectra for other lots.

**Bishoftu (ETBI)** — **blocked on data, precisely**: the local HIPS file has ETBI's
32 filters (so the x-axis exists), but there are no ETBI FTIR spectra anywhere under
the Drive data root (searched), and the local DB carries no SPARTAN spectra. The
same export used for ETAD (`DAVIS/ETAD FTIR/etad_lots_query.sql` is the template) run
for ETBI would make the crossplot a one-preset job — the app's evaluation side would
need ETBI wired as a second target once the export exists.

## Options added to the app for this pass

- **Residuals panel** (Addis tab) — predicted − Fabs/MAC vs HIPS, by season.
- **"All selection cohorts" spectra mode** (Selection tab) — Ethiopia-shaped, analogs
  and lowest-OC/EC medians overlaid on the Addis median (Ann's side-by-side ask).
- **Selection-overlap table** (Selection tab) — live pairwise shared-filter counts,
  including raw- vs corrected-space selections for both spectra-based cohorts.
- **Calibrate on SG 2nd derivative** — third spectra option, ftir_20's parameters.
- **Corrected-space spectral-analog selection** — "Select on: AIRSpec-corrected" now
  works for the analogs too (full ftir_09 recipe on the corrected caches, disk-cached).
- **Sub-types (k-means 3) spectra mode** — cluster medians within the current cohort.
- **Lot filter** — restrict any cohort to lot 248 or 251 (from `ftir_catalog`).
