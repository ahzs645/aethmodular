# Presentation script — Ann update, June 2026

Deck: `ann_update_jun2026.pptx` (speaker notes embedded too). One figure per slide; the SAY
text is written to be spoken, the NOTES are private — caveats and likely questions.

---

## Slide 1 — SPARTAN × IMPROVE follow-ups & the FTIR-EC offset — update for Ann (June 2026)

**Say:** Quick roadmap: first the data plumbing you asked about, then the mass-space remakes, the per-site ratio hunt for Addis analogs, iron, ETBI, and finally the offset story and what the Shiny app tests next.

> **Notes (private):** Everything is in the repo under research/spartan/ — three executed notebooks: follow_up_mass_and_ratio_plots, spartan_vs_improve_mass_comparison, presentation_offset_story. If short on time: slides 4, 5, 9, 16, 17 are the core.

---

## Slide 2 — Where the IMPROVE sites go — nothing is hand-picked

*Figure:* `research/spartan/follow_up_plots/figures/fig00_improve_site_funnel.png`

**Say:** Before the figures: your FED pull has 237 sites. Every figure starts from all of them and loses sites only to data screens. The one screen that matters is volume — FED only reports flow from 2015, so the mass figures were losing 30 discontinued sites, including most of the urban Addis-analog candidates. I've since recovered them with site-median-flow estimated volumes, flagged as estimates.

> **Notes (private):** If Ann asks about the fallback: IMPROVE flow is tightly controlled — 21.3 L/min median, 90% of samples within ±5%, duration almost always exactly 1440 min, so estimated volume is good to ~10%.

---

## Slide 3 — Mass on filter: the two networks genuinely overlap

*Figure:* `research/spartan/improve_comparison_mass/figures/M1_distribution_boxplots_mass.png`

**Say:** Your point from the meeting, verified: in micrograms on the filter — no volume, no log axes — the two networks overlap. SPARTAN's high concentrations and small volumes meet IMPROVE's low concentrations and big volumes in the middle. Total PM mass: medians around 90 to 125 micrograms in both. Absorption is shown as filter optical depth tau, which is fAbs times volume over area — fully volume-free, and the deposit areas are essentially identical, 3.5 versus 3.53 square centimeters.

> **Notes (private):** SPARTAN here = all 25 HIPS sites with public ChemSpec thermal carbon, date-matched to HIPS — method-matched to IMPROVE (TOR vs TOR), not FTIR. Whiskers are p05–p95.

---

## Slide 4 — Core four vs the IMPROVE envelope

*Figure:* `research/spartan/improve_comparison_mass/figures/M3_core4_vs_improve_mass.png`

**Say:** Same comparison, our four sites individually against the IMPROVE bands. Total mass: everyone's in range — even Addis is 100% inside IMPROVE's 5th-to-95th percentile envelope. EC mass and tau are where Addis and Delhi climb out the top.

> **Notes (private):** The one-row summary to say out loud: Addis filters are 100% inside IMPROVE's total-mass range, 10% inside its EC-mass range, 0% inside its EC-fraction range. Same mass, alien composition.

---

## Slide 5 — Every SPARTAN site: ordinary mass, extraordinary EC share

*Figure:* `research/spartan/follow_up_plots/figures/fig12_deposit_mass_vs_composition.png`

**Say:** This is the picture I'd keep. Site medians: total deposit mass on the x-axis, EC share of the deposit on the y. The gray cloud is IMPROVE; every SPARTAN site floats above it, and Addis is the extreme of the whole figure at about 22% EC versus IMPROVE's typical 4%. The calibration isn't being extrapolated in loading — it's being extrapolated in composition.

> **Notes (private):** Marker size = matched filter count. ETBI isn't here yet — no ChemSpec. When its carbon data arrive, its 26 filters land somewhere on this map and either confirm or break the Ethiopia pattern.

---

## Slide 6 — Volume-free bridge: same EC mass, more optical depth

*Figure:* `research/spartan/follow_up_plots/figures/fig01b_tau_vs_ec_mass.png`

**Say:** The apples-to-apples test: tau against EC mass, both volume-free, same deposit area. At the same EC mass, SPARTAN deposits produce about 1.7 times the optical depth of IMPROVE when EC is FTIR. Important check: redoing this with thermal EC on both sides still leaves a 1.4 times gap. So part of the excess is FTIR-specific — but not all of it.

> **Notes (private):** Numbers: slope 0.019 vs 0.011 tau/µg (FTIR EC); 0.0224 vs 0.0164 (thermal EC, M4 in the mass notebook). Caveat to volunteer: this is the under-predicted-EC signature itself, it doesn't by itself separate 'EC too low' from 'deposit absorbs more per µg'.

---

## Slide 7 — fAbs/EC by site: Addis's median MAC is unremarkable

*Figure:* `research/spartan/follow_up_plots/figures/fig02_fabs_ec_ratio_by_site.png`

**Say:** The per-site ratio hunt you asked for. Surprise: Addis's median fAbs over EC is about 10 — mid-pack among 211 IMPROVE sites. The Addis anomaly is not in the median ratio; it's in the slope-and-offset structure, which a single ratio can't see. So this ratio alone can't find analogs.

> **Notes (private):** Delhi 8.5, JPL 9.4, Beijing 10.3 — all unremarkable too. This is why the analog hunt needs the second dimension (next slides).

---

## Slide 8 — OC/EC by site: all four sites fall off the bottom

*Figure:* `research/spartan/follow_up_plots/figures/fig03_oc_ec_ratio_by_site.png`

**Say:** OC over EC is where SPARTAN separates. All four sites sit at or below the bottom couple percent of the IMPROVE network — Addis is about 1.6 thermal, 1.3 FTIR, against IMPROVE site medians of 3 to 9. This is Satoshi's point made visible: the functional-group calibration is being asked to work at OC-to-EC ratios it has essentially never seen.

> **Notes (private):** Stars = thermal OC/EC (method-matched to IMPROVE), diamonds = FTIR OC/EC on the HIPS-paired filters.

---

## Slide 9 — OC/fAbs by site: Addis is the lowest site anywhere

*Figure:* `research/spartan/follow_up_plots/figures/fig08_oc_fabs_ratio_by_site.png`

**Say:** And the same thing from the optical side — OC per unit absorption. Addis is the single lowest site in either network, Delhi and Beijing hug the bottom edge; only Pasadena sits inside the pack. Together with the previous slide: the problem sites are absorption-rich relative to their organics.

> **Notes (private):** Ann asked for exactly this pair — 'OC/EC by site, then the same thing, OC/fAbs' — and predicted Addis would move around in the ordering: it does, from mid-pack (fAbs/EC) to dead last (OC/fAbs).

---

## Slide 10 — The Addis-analog map — which IMPROVE sites to pull spectra for

*Figure:* `research/spartan/follow_up_plots/figures/fig04_addis_analog_map.png`

**Say:** Both ratios in one plane, site medians. The IMPROVE sites nearest Addis are almost all the urban ones — the New York IS-52 site, Washington DC, Rubidoux, Chicago, Baltimore, Detroit, plus Baengnyeong Island in Korea. Those are the candidates if we want IMPROVE FTIR spectra from Addis-like composition.

> **Notes (private):** Saved as a ranked table (addis_analog_candidates.csv). Caveat: most of these urban sites were discontinued years ago — but their spectra exist in the archive, which is what matters for the app.

---

## Slide 11 — Iron, part 1: the Abu Dhabi check across 25 sites

*Figure:* `research/spartan/follow_up_plots/figures/fig09_iron_ratios_by_site.png`

**Say:** You predicted Abu Dhabi would come up high on iron — half right. It has the second-highest absolute iron in SPARTAN, behind Delhi, and ranks fourth of 25 on iron-to-EC. But per unit of absorption the standout is actually Fajardo in Puerto Rico — Saharan dust over almost no absorption. And the key point for us: Addis is near the bottom on iron-per-absorption. Its absorption is too big for iron to be the story.

> **Notes (private):** New data here: public SPARTAN ChemSpec pulled for every HIPS site and date-matched — ~2,460 Fe×fAbs pairs across 25 sites, not just our four.

---

## Slide 12 — Iron, part 2: still no iron effect on HIPS

*Figure:* `research/spartan/follow_up_plots/figures/fig10_fabs_ec_colored_by_fe_ec.png`

**Say:** The play-around plot: fAbs against EC, colored by the iron-to-EC ratio. If iron drove absorption, the hot colors would float above the trend. They don't, in either network — residual correlations are plus 0.05 and minus 0.02. Iron is present at the dusty sites and irrelevant to the absorption signal. I'd close this question.

> **Notes (private):** Matches the conclusion from the earlier site plots; now it's one picture and two numbers.

---

## Slide 13 — ETBI: the second Ethiopian site lands in the same regime

*Figure:* `research/spartan/follow_up_plots/figures/fig06_etbi_fabs_context.png`

**Say:** Bishoftu's first 26 filters, October to December commissioning: median fAbs around 27 — lower than Addis's 47, but far above the rest of the network's typical levels, and its fAbs-versus-tau points fall exactly on Addis's curve, so HIPS is behaving identically there. No carbon data yet; when ChemSpec arrives it drops straight onto the composition map.

> **Notes (private):** If ETBI eventually shows the same fAbs–EC offset, that's a regional aerosol regime, not an ETAD-specific instrument story. Also relevant: the 5 collocated TOR+FTIR filters from the other Ethiopian city, once Alex is back.

---

## Slide 14 — The anomaly itself: slope 4, intercept 28 — and Delhi shares it

*Figure:* `research/spartan/presentation_offset_story/figures/P1_anomaly_in_context.png`

**Say:** Now the offset story. Addis is not noisy — it has the tightest fAbs-EC relation of any site, just with slope 4 instead of 10 and a 28 inverse-megameter intercept. And Delhi shows nearly the same line: slope 3.8, intercept 23, same R-squared. Whatever this is, it lives where charcoal lives. Beijing and JPL look like the calibration expects.

> **Notes (private):** R² 0.76 at both Addis and Delhi. If asked whether the tightness itself is suspicious — that's slide 18.

---

## Slide 15 — Two additive fixes the data cannot tell apart

*Figure:* `research/spartan/presentation_offset_story/figures/P2_two_additive_fixes.png`

**Say:** The algebra in one picture: add 7 micrograms to every EC value, or subtract 28 from every fAbs — both put the line through zero with the same slope and the same R-squared. Purely additive, so no regression choice can separate 'FTIR misses mass' from 'HIPS reads high'. That's exactly why the KBR pellets and the calibration tests matter — the data alone can't break the tie.

> **Notes (private):** Consistent with the Deming notebook (Δ = 7.1 there, 7.0 here — slightly different pair sets). Multiplicative fixes destroy the R²; the ×2.4 'fix' is the additive shift in disguise.

---

## Slide 16 — The seasonal test: the offset is flat — a surprise

*Figure:* `research/spartan/presentation_offset_story/figures/P3_seasonal_offset_test.png`

**Say:** Your test from last time: if the missing EC is charcoal, it should swell in the rainy season when Naveed's PMF says charcoal influence peaks. It doesn't. The implied missing EC is six to seven micrograms in every month, and independent per-season fits give intercepts of 26 to 28 in all three seasons. As it stands, the offset behaves like a constant.

> **Notes (private):** Caveats to volunteer: PMF seasonality is about relative source shares, not absolute charcoal mass — charcoal cooking is year-round in Addis; so this doesn't kill the char story, but it does constrain it: whatever FTIR misses is present on essentially every filter.

---

## Slide 17 — Reconciling with history: did Addis really get 3× cleaner?

*Figure:* `research/spartan/presentation_offset_story/figures/P4_history_reconciliation.png`

**Say:** The only independent reference is the Schauer-era TOR campaign — about 13.7 micrograms. Today's FTIR EC says 5. But both offset-corrected readings of our own data — EC plus 7, or fAbs over a MAC of 4 — land at about 12, right next to the historical value. The simpler story may be that EC never declined much and FTIR just can't see a third of it.

> **Notes (private):** Flag: the 13.7 is quoted from the meeting — confirm the citation before showing this outside the group. Campaign years and seasons differ; indicative only. The wildfire-calibration test (T2) adds a sixth bar.

---

## Slide 18 — Nothing in IMPROVE looks like Addis — and the R² is normal

*Figure:* `research/spartan/presentation_offset_story/figures/P5_improve_fit_plane.png`

**Say:** Per-site slope and intercept for 207 IMPROVE sites: nobody combines a slope of 4 with a 28 inverse-megameter intercept — IMPROVE intercepts live between 0 and 6. And to answer my own earlier worry: an R-squared of 0.76 is completely normal in this network, so the tightness of the Addis fit is not itself a red flag.

> **Notes (private):** About a third of IMPROVE sites fit tighter than Addis. The isolation of Addis/Delhi in this plane is the network-level version of the composition gap.

---

## Slide 19 — Inside IMPROVE, composition never gets there — and doesn't bend

*Figure:* `research/spartan/follow_up_plots/figures/fig14_within_improve_composition_drift.png`

**Say:** Last analysis slide, and it cuts both ways. I fit the network-wide IMPROVE fAbs-EC line and asked whether residuals drift as deposits become EC-rich. Three results: IMPROVE never reaches Addis's carbon split — deciles top out at 0.31 versus Addis at 0.41 to 0.44. The one percent of samples that get close behave completely normally. And absorption per unit EC actually falls with EC share. So the Addis signature is not reproduced inside IMPROVE by composition — which supports a domain gap the calibration has never seen, rather than a visible bending — and it keeps the char-specific FTIR explanation alive, because this test can only see thermal EC.

> **Notes (private):** If Ann pushes: yes, this weakens 'the calibration visibly degrades near Addis composition' and strengthens 'it was never trained there at all'. The discriminator remains the KBR spectra + app tests.

---

## Slide 20 — What's next: KBR pellets + the Shiny app task list

**Say:** Three tracks. One: KBR pellets — charcoal from Hossein in hand, grinding and press coordinated, open questions are the KBR-to-sample ratio, a blank pellet spectrum, and shipping containers. Two: the Shiny app once the VPN lands — reproduce the baseline, run Mona's wildfire calibration on Addis spectra, then EC1, OP and the char-soot split as targets, plus out-of-domain diagnostics on the Addis spectra; full task list and decision table are written up in the repo. Three: data arrivals — ETBI ChemSpec and the five collocated TOR-FTIR filters from the second Ethiopian city slot straight into existing figures.

> **Notes (private):** The decision table is in shiny_app_ftir_plan.md §5 — each test has a defined outcome A/B meaning. Remember Ann's advice: ask AQRC IT early, follow up if quiet; have Mona at the next Tuesday meeting if the VPN is live.
