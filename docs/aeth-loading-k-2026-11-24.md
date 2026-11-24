# Aethalometer loading k: Kyan's manual loading compensation, made adjustable (24 November 2026)

Gallery: Experimental, **Aethalometer loading k** (`#tab=experimental&xp=aeth-k`, `pages/experimental/AethLoadingStudy.tsx`).
Data: `gallery/data/export_aeth_loading_k.py` → `gallery/app/public/data/experimental/aeth_loading_k.json` (rebuild step `aeth-loading-k`).

## What the correction is

The MA350 reports BC from how fast the tape darkens. On a loaded spot the same air darkens it more slowly (the loading
effect), so spot 1's BC1 reads low as ATN1 rises. The instrument corrects with a second spot at a lower flow (DualSpot,
Drinovec et al. 2015):

    k = (BC2 − BC1) / (BC2·ATN1 − BC1·ATN2),    BCc = BC1 / (1 − k·ATN1)

solved every minute, which needs Flow1:Flow2 near 3 (75:25). At Jacros (Addis, MA350-0238) the split sits near 55:45, so
the two spots load alike and the live K is noise: some minutes get a negative denominator (BC changes sign), some one
above 1 (BC shrinks).

Kyan Shlipak's fix (EC_BC deck; `src/data/processors/site_corrections.py`, `_apply_manual_loading_compensation_ethiopia`)
keeps the formula and replaces the live K with one k for the record, applied where ATN1 > 3:

| Way of choosing k | Uses FTIR? | Kyan's Jacros IR value |
|---|---|---|
| Median of the instrument's K over ATN1 > 3 | no | 0.0129 |
| k leaving corrected BC uncorrelated with ATN1 | no | 0.01185 |
| k leaving FTIR EC − BC residuals uncorrelated with ATN1 | yes | 0.01172 |

The two criteria agree within 1 %, so the corrected aethalometer is not tuned to FTIR. His sanity rules: the correction
should not more than triple BC, shrink it, or flip its sign.

## What the page adds

- Every chart follows one k (Page options): the instrument's own K, any of the three choices above, a tape-advance k,
  or a k of your own.
- **Tape-advance check** (not in the deck, and needs no other instrument): at an advance the air does not change, so the
  right k makes BC 5–15 min after equal to BC 15–5 min before.
- The three criteria as curves against k; the denominator distribution against the instrument's; FTIR EC (or HIPS BC)
  against aethalometer BC on each filter's own sampling window; the results table in the layout of Kyan's slide 9; the
  filter-by-filter series and the hour-of-day profile.
- The link to HIPS: both instruments register less of each new increment as the deposit darkens. The aethalometer's
  share is 1 − k·ATN1; HIPS's is the slope of reported against true optical depth ((1 − τ/τ∞)² for our curve). The
  aethalometer can find its own k because one spot keeps loading under the same air; HIPS reads each filter once, so its
  ceiling needs an outside reference (TOR, CSN, drain discs). ATN includes the tape's multiple scattering, so the x axes
  are not the same quantity.

## Differences from the deck

Filter values are means over each filter's own sampling window (half its minutes needed), not 9am–9am days, and the
fits in the table are OLS over the filters counted by the gallery's exclusions. Minutes whose denominator is 0.1 or
less (a factor over 10) are left out at that k and counted.
