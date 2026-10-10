# Slide of the week, 2 Oct 2026 (Ann + Satoshi call, 1 Oct)

Rebuild: `~/anaconda3/bin/python deliverables/slide_of_the_week_2026-10-02/build.py`
(full-range VIBES run, same inputs as `deliverables/vibes_airspec_methods_2026-10-01`; the window is
`MASKS["oct1"]` in `research/ftir_hips_chem/scripts/analog_transfer_checks.py`).

## One-breath summary

VIBES gives us the spectrum below 1500 cm⁻¹ for picking analogs. In most Addis long-rains spectra,
though, the Teflon residual is the biggest feature and would drive the similarity score. We agreed to
match on 600 to 3600 cm⁻¹ without the Teflon and CO₂ bands, and to train the VIBES background on many
more blanks: all lot 251 blanks first, then all IMPROVE blanks.

## Behind the slide (notes have the detail)

- The filter shown is ETAD-0105-1 (lot 251, 16 Aug 2023): one real filter, picked by a stated rule, not an average.
- The Teflon-band residual is the largest feature inside 600 to 3600 in 85% of Kiremt Addis filters (1.75× the next largest on the one shown).
- The current background model has rank 30 and uses 87 blanks (59 IMPROVE, 28 ETAD).

## Asks from the call

- Train the VIBES background on all labelled lot 251 blanks, then on blanks from all IMPROVE lots.
- Analog similarity: 600 to 3600 cm⁻¹, dropping CO₂ (1800 to 2500) and Teflon (1050 to 1250). PLS can keep the full spectrum.
- Sensitivity case: keep the dust horns above 3600 (Satoshi).
- Test the fingerprint region with and without; it helps for dust and sulfate but may mislead for organics.
- Add intercepts to the per-season IMPROVE slope view. Dry-matched IMPROVE analogs show the same low slope as Addis.
- Label the per-season panels as separate calibrations with separate analog sets.
- Within-season (Kiremt) correlation groupings past 1425 cm⁻¹.
- Stay on Addis; skip Delhi and Pasadena.

## Don't get caught

- The "defect-rich graphitic carbon" reading of the broad rise from ~1600 cm⁻¹ is Satoshi's hypothesis, not a result.
- Satoshi reads the Kiremt hump as seasonal composition, not a direct EC signal.
- Results using the new window and the larger blank set don't exist yet; this slide shows the plan.
