# Talk track — the "+ AIRSpec" half of the best calibration

Three slides explaining what AIRSpec baselining does and why it moves the Addis answer.
Companion to `filtering_by_ocec.png`, which explains the "Lowest-OC/EC" half of the same
setup name. The three PNGs are exported **without titles** so the deck sets them; suggested
titles are given per slide below. Figures live in `output/plots/deck/`; numbers come from
`scripts/build_deck_figures.py` (Addis n = 239, cohort n = 800) and ftir_13 / ftir_19.

---

## Slide 1 — `airspec_1_baseline.png`

**Suggested slide title:** Almost all of a raw spectrum is background, not chemistry

> Every faint grey line here is one Addis filter — all 239 of them. The black line is a
> single one of those, picked out so we can follow it; it is not an average, and nothing on
> these slides is an average. What you see is that almost all of the signal is a smooth
> downward slope. That's not chemistry, that's background: scattering off the filter and
> the particles themselves. The red dashed line is the baseline AIRSpec fits underneath
> that one spectrum — it fits every spectrum separately — and the shaded sliver is the only
> part that's actually absorption bands. At the CH band, ninety percent of what we measure
> is baseline. So when we train a model on raw spectra, we are handing it a predictor that
> is mostly background, and letting it decide how much to lean on that.

**If asked why this filter:** it is the one closest to the middle of the Addis set — within
10 percentile points of the median on all four labelled bands *and* on baseline height. We
did not average the spectra, because averaging would smear out band positions and hide how
much filter-to-filter spread there is; the grey lines show that spread directly. Ranking on
one band alone is not enough — the median-CH filter turns out to sit at the 92nd percentile
of O–H.

**If asked what AIRSpec is:** a smoothing-spline baseline fit segment by segment, from the
APRL group. We ported it to Python and validated it against their R code to about 1e-7, so
this is their algorithm, not our approximation.

---

## Slide 2 — `airspec_2_corrected.png`

**Suggested slide title:** What is left is the part the calibration should be using

> Same single filter, after we subtract its own baseline. Now you can see the actual
> chemistry — the
broad O–H stretch, the CH bands, carbonyl, and the 1600 band we've been
> arguing about.
> The thing to notice is the axis: this entire plot spans three hundredths of an
> absorbance unit. The raw spectrum went up to 0.34. So everything we care about chemically
> is a thin skin on top of a background ten times its size.

---

## Slide 3 — `airspec_3_background_gap.png`

**Suggested slide title:** Addis rides a higher background than the filters it was
calibrated on

> Here's why that matters for Addis specifically. This is the size of that background at
> the CH band — one value per filter, so 800 IMPROVE filters in purple and all 239 Addis
> filters in red.
> Addis sits higher — median 0.17 against 0.10. The distributions do overlap, so this
> isn't a different universe, but Addis is consistently toward the high end. That means
> whatever a raw-spectra model learned about background from IMPROVE doesn't quite apply
> here, and that error shows up as an offset in our EC predictions.

**The payoff line to land on:**

> When we baseline both sides first and retrain, the Addis intercept goes from −3.22 to
> −1.62 µg/m³, and the slope goes from 1.59 to 0.86 at MAC = 10. That's the single biggest
> improvement of anything we tried in phase 3.

---

## Caveats to have ready

- **Nothing here is an averaged spectrum.** Slides 1 and 2 show one real filter; slide 3
  shows every filter as its own value. If someone asks why not average: averaging smears
  band positions across filters and hides the spread.
- **The distributions overlap.** If someone pushes on slide 3: this is a shift in the
  centre, not a separation. It's suggestive of the mechanism, not proof of it.
- **It doesn't fix everything.** The corrected model still carries a −2 to −2.6 µg/m³
  season-stable offset (ftir_15), and it does *not* rescue the HIPS transfer (ftir_13) —
  that failure survives baselining, which is what tells us it's compositional.
- **Scatter gets worse, not better.** Corrected Addis R² is 0.66 vs 0.77 raw, RMSE 2.41 vs
  1.16. We prefer the corrected model because its residuals are a clean constant offset
  rather than extrapolation-driven, not because it fits Addis more tightly.
- **The MAC fork is untouched by this.** Baselining changes the model; it does not change
  what MAC we assume for HIPS. See `mac_slope_pivot.png` (ftir_19).
