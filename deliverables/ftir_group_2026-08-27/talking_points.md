# FTIR group talk (Thu 28 Aug) — talking points, post-run-through rebuild

**The arc (Ann's framing):** Addis is different; depending on which samples
we calibrate on, how we baseline, and how we pick the model, we get a huge
array of outcomes; a few illustrative ones, then the enormous search; so the
group discussion is: what constraints and validation make a choice
defensible?

## Deck map (26 main incl. 4 discussion + 9 backup = 36)

- **1-2**: title/arc; the deployed 1.90x-4.2 problem
- **3-5**: raw spectrum is ~90% background (split, full-res); Addis rides a
  higher background; baseline correction (using AIRSpec) explained
- **6-7**: how cohorts are picked, BEFORE any results: lowest-OC/EC cut;
  spectral-shape ranking (ranked histograms only; scatter panels removed)
- **8-9**: ONE before/after (network raw -4.3 -> corrected -0.6 but slope
  0.43) then the six baselined crossplots only
- **10**: three ways of selecting the model, no CV jargon; red circles =
  how the Shiny app / all deployed SPARTAN calibrations do it
- **11**: cohort-size sweep; intercept stable, held-out R2 bounces, 800
  somewhat arbitrary (Option-A sweep, noted)
- **12-14**: filters differ by site (R1/T1 defined in words, no fit lines,
  mislabeled blank count dropped); lot-248 timeline (kept, liked); lot
  effect removed by baselining
- **15**: five-city baseline-corrected median spectra, full-slide; CH and
  ~1700 labeled; Addis-least-organics = consistent with lowest OC/EC
  (Ann's point); 1617 panel cut (rationale lives in the working deck)
- **16**: seasonality shifted, not solved
- **17-22**: the search (capped axes, 58 outliers counted); the app (live);
  the WINNER AS A CROSSPLOT; per-city bests; cross-application matrix
  (metric stated in words); winner stability 62.5%
- **23-26**: the four discussion questions (good enough? / validation
  split-half proposal / one-vs-per-site / spectral-vs-OC/EC similarity +
  common-thread + AERONET one-liner)
- **27-36 backup**: raw grid, blank-line geometry, York x2 (with the
  plain-language "errors-in-both-variables with per-point uncertainties"
  line), carbonyl-vs-intercept (ug/m3), Mie, AERONET diurnal, lot-253 v2
  (black in-plot text), the two lab asks

## Language rules enforced (audit done)

- "baseline-corrected (using AIRSpec)" first mention (slide 5), then
  "baseline-corrected"; no "AIRSpec-ed"/"AIRSpec-corrected" anywhere.
- Calibration set vs prediction target phrasing on every result slide.
- Intercepts in ug/m3 everywhere (band scatter converted; Mm-1 gone).
- No CV jargon in the main arc (full recipes in slide-10 NOTES if pressed).
- No em dashes; verified programmatically, figures included.

## Numbers to have ready

- Deployed: 1.90x-4.2 (MAC 6: 1.14x, same intercept; Deming intercept is
  MAC-invariant).
- Before/after network: raw 1.80x-4.28 (k=15) -> corrected 0.43x-0.57 (k=7).
- Winner: lowest-OC/EC 450, baseline-corrected, k=9: 0.93x-1.4, target R2
  0.72, held-out 0.90; re-selected in 62.5% of draws (analog family ~1/3).
- Cross-application: Addis-best at Bishoftu 0.99x-0.70; Delhi-best at
  Beijing 5.36x (96% extrapolated, fails holdout).
- Lot effect at Beijing: raw -4.5 -> corrected -0.9 Mm-1 (n.s.; n=14 vs 34).
- Addis lots: 34 x 248 / 191 x 251 / 14 x 253. SPARTAN lot 248 = 332
  filters network-wide.

## Don't get caught

- Held-out R2 = IMPROVE-side TOR test; the TARGET-side validation gap is
  exactly discussion question 2 (split-half proposal from Tuesday).
- Slide-11 sweep is the site-grouped recipe; the Shiny-style version would
  differ.
- Blank-count caveat on the backup geometry slide: a lot can carry more
  than one deployed calibration line.
- The "!" cells in the matrix = >30% of that city's filters beyond the
  model's training domain.
- Q4 top-performers common-thread question: answerable live in the app
  (leaderboard + export) if the group wants it now.

## Rebuild

build_v2_figures.py (needs the explorer; EXPLORER_PORT env) then
build_deck.py (writes both the notes deck and the no-notes twin).
