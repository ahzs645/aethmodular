# Calibration transfer beyond global analogs — methods survey + first LOCAL run

Agent-assisted survey 2026-08-24 (full citations at bottom), plus the first
in-repo per-filter LOCAL results. Companion module:
`calibration_explorer/local_lab.py` (`/api/local_run`, `/api/local_cross`).

## Premise correction

There is **no Reggente/Dillner/Takahama per-sample kNN paper** — the 2016 AMT
paper is global PLSR + the score-distance diagnostic ("a crude method", their
words); 2019 (AMT 12, 2287) is peak-fitting vs PLS. The closest prior art is
Weakley 2015 (classify-then-calibrate) and Debus 2022 (GMM clustering → 21
representative sites). **True per-sample kNN-PLS appears novel in the
FTIR-aerosol space** — the template comes from soil/feed NIR (LOCAL: Shenk &
Westerhaus 1997; LWR: Næs 1990; MBL/resemble: Ramirez-Lopez 2013).

## LOCAL v1 (this repo, 2026-08-24): honest first pass

`local_lab.py`: per target filter, top-k library spectra by Pearson-on-deriv2,
small PLS, per-filter neighbor-similarity diagnostic. k=200, ncomp=8, Deming
MAC-10 all-pairs:

| target | LOCAL v1 | vs global winner | neighborhood |
|---|---|---|---|
| Addis | **4.53x−16.69, R² 0.26** | 0.93x−1.42 | Phoenix-dominated |
| Delhi | 1.37x−0.28, R² 0.71 | 2.36x−4.70 | Atlanta-dominated |
| Beijing | 1.39x−0.36, R² 0.56 | 1.48x−0.08 | LASU2 |
| Pasadena | 1.04x+0.19, R² 0.11 | 4.22x−0.70 | Yosemite |
| Bishoftu | 0.58x−0.36, R² 0.27 | 0.93x−0.56 | FRRE1 |

Read: **Delhi and Pasadena improve markedly** (slopes 1.37 / 1.04 vs 2.36 /
4.22) — per-filter neighborhoods do real work where the library has support.
**Addis collapses** — the survey's predicted failure mode verbatim: for an
off-manifold target, kNN returns the nearest library *edge* (hot, loaded US
urban sites), and a homogeneous high-similarity neighborhood (median r 0.995)
with narrow y-span amplifies noise. Local ≠ in-domain. The Delhi
reconstructed-holdout row is garbage (−237x) — that set is lot-253/high-tail
(see aethmodular-8c's warning) and unusable as a headline.

## v2 refinements (from the survey, in order)

1. **Similarity in truncated PLS-score space (Mahalanobis), not raw Pearson**
   — raw similarity is interferent-dominated; score-space tracks chemistry.
2. **Neighborhood gates on every prediction**: neighbor y-span must bracket
   the prediction; mean-kNN-distance and leverage caps. Refusing to predict
   IS the correct output at Addis.
3. **k tuned by neighborhood CV** (grid 50–250), wapls-style averaging over
   LV counts instead of one ncomp.
4. Then the DA ladder: **CORAL in score space (~20 lines, the baseline)** →
   **di-PLS** (diPLSlib; unsupervised target; λ-swept with a MAC-plausibility
   guardrail — note `scripts/domain_invariant_pls.py` + ftir_36 already exist
   in-tree from the validation-layer session; coordinate, don't duplicate).
   Importance weighting only with effective-sample-size monitoring (ESS
   collapse = "target outside library, stop").
5. **Never** use Fabs as a training signal — it is the evaluation axis
   (circularity); MAC-band plausibility checks only.

## The decisive evaluation design (runnable entirely in-hand)

**Simulated transfer at IMPROVE:** pick 2–3 held-out IMPROVE site-clusters
with collocated HIPS Fabs, *pretend they only have Fabs*, run the full
pipeline (selection, adaptation, tuning, MAC checks) exactly as at Addis,
then unblind TOR once. Measures deployment-protocol error with ground truth —
nothing at SPARTAN can substitute. Pre-registered success criteria for real
targets: AD-coverage fraction rises materially; median MAC633 in 5–13 m²/g
and season-stable; Pasadena is the bridge site (fail there = dead); Bishoftu
report-only.

## Key citations

LOCAL: Shenk & Westerhaus, JNIRS 5, 223 (1997). LWR: Næs, Isaksson &
Kowalski, Anal. Chem. 62, 664 (1990). MBL: Ramirez-Lopez et al., Geoderma
195-196, 268 (2013) + the `resemble` R package. FTIR-aerosol: Reggente et
al., AMT 9, 441 (2016); Weakley et al., AMT 8, 4013 (2015); Debus et al.,
AMT 15, 2685 (2022); Takahama et al., AMT 12, 525 (2019). di-PLS:
Nikzad-Langerodi et al., Anal. Chem. 90, 6693 (2018) + diPLSlib. CORAL: Sun,
Feng & Saenko, AAAI 2016. TCA: Pan et al., IEEE TNN 22, 199 (2011). KLIEP:
Sugiyama et al., AISM 60, 699 (2008); uLSIF: Kanamori et al., JMLR 10, 1391
(2009). AD/conformal: Jackson & Mudholkar 1979 (Q); Norinder et al., JCIM 54,
1596 (2014); Tibshirani et al., NeurIPS 2019 (weighted conformal). MAC
anchor: Bond & Bergstrom, AST 40, 27 (2006).
