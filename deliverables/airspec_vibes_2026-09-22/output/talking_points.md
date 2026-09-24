# AIRSpec and VIBES: presenter brief

**One-breath summary:** VIBES learns background patterns from independent blanks and removes far more blank residual signal. AIRSpec fits a segmented smooth baseline to each spectrum. The completed paired EC benchmark does not establish an overall VIBES accuracy gain, so AIRSpec remains our current EC default while VIBES merits further spectroscopy tests.

## Route through the deck

- **Slides 1–5:** Explain the methods and follow one actual spectrum. Correction estimates background. A separately trained PLS model predicts EC.
- **Slides 6–9:** Compare identical test identities within each cohort. Full-pool RMSE is 3.224 for AIRSpec and 3.296 for VIBES. Restricted-cohort RMSE improves with VIBES, but MAE worsens. Both pooled paired RMSE intervals include zero.
- **Slides 10–14:** Separate blank removal and synthetic recovery from EC accuracy. VIBES wins the larger synthetic additions but loses the weakest one. Q3 warrants investigation. Two severe shared prediction failures account for about 52.2% of its squared-error increase. Repeating twelve corrections changes predictions by at most 0.002 µg/filter.
- **Slides 15–18:** AIRSpec reproduction passes all 43 checks across five tables. Explain the historical claim corrections and the unresolved independent Addis reference.
- **Slide 19:** Propose training-only applicability checks, independent weak-signal standards and matched Addis thermal EC.

## Questions likely to come up

**Why keep AIRSpec?** The full-pool EC comparison offers no established VIBES gain. This is a current operating choice, not proof of universal superiority or equivalence.

**Does a cleaner blank mean better EC?** It supports background removal on those blanks. EC prediction also depends on spectral information and the fitted calibration coefficients.

**Can we delete the two failing filters?** They remain in the reported evaluation. Investigate applicability using training data and evaluate any changes on fresh data.

**Is the 1617–1620 cm⁻¹ feature a fuel marker?** Historical adjusted PMF evidence supports a regional marker. A specific charcoal/eucalyptus assignment remains unestablished.

## Caveats to say aloud

- EC errors are **µg/filter**, not concentrations. Absorbance metrics have different units.
- The full and restricted cohorts have different test populations. Historical AIRSpec reproduction has **194** test rows, versus **137** in the Colab restricted comparison.
- Subgroup intervals are exploratory, pointwise and conditional on saved fits. They omit retraining uncertainty.
- The spectral example deliberately shows a high-error case, not a typical filter or average spectrum.
- HIPS/MAC is an optical equivalent. ChemSpec does not supply independent thermal EC.
- The 25 historical sources on the priority queue remain unexecuted. A priority label is not verification.

The complete spoken script, caveats and file-level sources are in **speaker_script.md** and the PowerPoint speaker notes. Evidence was reviewed through 21 September 2026.
