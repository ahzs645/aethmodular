# Presentation talking points — component selection & calibration effect

One-line thesis to open and close on:
> **"The calibration you choose — not the sample — is setting the EC. So the goal isn't a
> pretty fit, it's an honest one: keep the samples, pick components the same way every time,
> and filter on *relative* error."**

Suggested order: (0) the problem → (1) fixed component graph → (2) raw vs 2nd-derivative →
(3) variants at honest k → (4) the effect. Each section below is "say this" bullets + the
punchline + likely Q&A.

---

## 0. (Optional) The "before" — last week's component graph
*Show `figures/fig06_rmsecv_by_variant.png` if you want the contrast.*
- "Last week's component-selection graph looked like this. There are two problems with it, and
  once I fixed them the story got cleaner."
- "First, everything is in absolute µg on one axis — so the high-EC variants dominate and the
  ones we care about are squished at the bottom. Second, the curves are jagged enough that
  'pick the minimum' isn't reproducible."
- **Punchline:** "So I redid the component selection properly — that's the next slide."

---

## 1. Corrected component selection — `fig09_component_selection_pct.png`
- "Same six calibration variants, but three fixes: I put the real candidates on the left and
  the small diagnostic variants on the right so they stop dominating; the y-axis is now
  **%RMSECV** — the cross-validated error as a fraction of the mean EC — so the variants are
  actually comparable; and I used 10-fold interleaved cross-validation, which is the scheme the
  production R tool uses. I checked it reproduces R's curve exactly."
- "The dashed lines are where each calibration picks its component count, using **one** rule
  applied everywhere — so it no longer flips depending on which slide you read."
- "The honest read is: **CAL-0 (keep everything) and CAL-5 are the only stable candidates.**
  The rest sit at 150–350% error — the error is as big as, or bigger than, the signal."
- **Punchline:** "Even our best full-range calibration lands near 100% %RMSECV — the
  cross-validated error is about the size of the mean EC. That's not a modeling failure, it's
  the EC distribution: a handful of extreme smoke filters carry the whole regression."
- **If asked "why is the error so high?"** — "Seven of 906 filters are above 150 µg, three above
  250. The fit tracks that high-EC spread well — that's where the R² comes from — but for a
  *typical* low-EC filter the relative error is large. That's exactly why absolute-residual
  filtering was the wrong tool."

---

## 2. Raw vs. 2nd-derivative — `fig09_raw_vs_d2.png`
- "This explains *why* the raw component curve was so jagged. Blue is the raw spectra: it
  actually gets **worse** from components one to five, because the first PLS components model
  the **Teflon baseline**, not the carbon — that's Weakley's point."
- "Red is the same data after a second-derivative — it's flat and near its floor from the very
  first component."
- "Concretely, the raw curve doesn't settle until about **19 components**; the second-derivative
  gets there at **9** — roughly half."
- **Punchline:** "Second-derivative preprocessing removes the baseline, halves the component
  count, and kills the jaggedness. This is the argument to preprocess *before* we finalize any
  calibration."
- **Honest caveat (say it — it's stronger than hiding it):** "Notice the red floor is still
  ~100%. Second-derivative fixes the *component selection*, not the underlying error — that's
  the extreme-sample problem, and preprocessing alone won't solve it."

---

## 3. Variants at their honest component count — `fig10_variants_1to1_correctedk.png`
- "Predicted vs. measured EC for each calibration, but now each one is fit at **its own
  cross-validated component count**, not a common 20 we imposed last week."
- "Look at **CAL-1, the absolute-cleaned calibration** — at its honest component count it's
  basically a flat line. It's lost the ability to tell filters apart."
- "**CAL-0 (keep everything) and CAL-6 (the new relative-clean)** spread properly along the 1:1."
- **Punchline:** "Last week the cleaned calibration looked tightest. That was an artifact of
  forcing 20 components onto a set that honestly only supports one. Give it the right component
  count and the 'beautiful' calibration collapses."

---

## 4. The effect — `fig10_effect_predicted_ec.png`  *(the money slide)*
- "Here's why all of this matters. I took the **same 906 filters** and ran them through each
  calibration. Each box is the spread of predicted EC."
- "The tiny box is **CAL-1, absolute-clean** — near-constant predictions, and a median about
  **22% higher** than keep-everything. It looks clean because it barely varies."
- "**Keep-everything and relative-clean keep the real dynamic range.** Across calibrations, the
  median EC on the identical filters swings about **1.3× here — and about 2× on the real
  Ethiopia spectra.**"
- **Punchline (land the thesis):** "So the calibration choice, not the filter, is driving the
  EC. That's the whole case for not over-filtering."
- **Then introduce the fix:** "And the new **CAL-6 — cleaning on *relative* residuals instead of
  absolute** — is the principled middle ground. It keeps **873 filters including the high-EC
  smoke** the absolute filter threw away, and it has the **best cross-validated error of any
  full-range calibration, 84% vs keep-everything's 98% and absolute-clean's 143%.** That's the
  meeting's hypothesis confirmed: a big absolute residual on a smoke filter is a small *percent*
  error, and we shouldn't be punishing it."

---

## Closing line
> "Bottom line: keep the samples, preprocess with a second-derivative, pick components one
> consistent way, and filter on relative error. When we do that, CAL-6 is the calibration to
> carry into the Ethiopia analysis."

### Numbers cheat-sheet (so you're not caught out)
| calibration | n | k* | %RMSECV | median EC (×CAL-0) |
|---|---|---|---|---|
| CAL-0 keep-everything | 906 | 19 | 98% | 11.1 (1.00) |
| CAL-1 absolute-clean | 860 | 1 | 143% | 13.5 (1.22) |
| **CAL-6 relative-clean** | **873** | 27 | **84%** | 11.1 (1.00) |
| CAL-3 below-1:1 | 469 | 4 | 206% | 12.1 (1.09) |
| CAL-5 Eth-range (placeholder) | 480 | 13 | 34% | 12.2 (1.10) |
| CAL-7 no-extremes (drop 3 EC>250) | 903 | 27 | 63% | 10.7 (0.97) |

- 2nd-derivative: raw settles at k=19, 2nd-deriv at k=9.
- Extreme samples: 7 filters > 150 µg, 3 > 250 µg — they dominate the RMSE.
- CAL-2 / CAL-4 are diagnostics only (n=46 / n=22) — do not present as calibrations.
