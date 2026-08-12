# Talk track — Ann 1:1, 12 Aug 2026

Spoken notes for `deliverables/ann_briefing_2026-08-12.html`. Nine slides, ~30 minutes,
one attendee who knows the project cold — so nothing here re-explains phase 3. Every
slide either reports something that moved since 31 July or puts a decision in front of
her. Same format as `deck_notes_airspec.md`: the indented block is what to say out loud,
the notes under it are what to have ready if she pushes.

**Before you start**: everything numeric on slides 01–07 now traces to committed,
executed notebooks — ftir_26 (IMPROVE origin, curvature, loading-dependence), ftir_27
(ChemSpec circularity), ftir_28 (MA350 falsification and the revised MA350-BC(880) fit) —
pushed through e3d1f4f. Only slide 08's feasibility figures (the 55% x-range, the February
count, the pull timings) stay preliminary until ftir_24 itself is built. If she asks "is
this in the repo": yes — executed end to end, assertions on every headline number — except
slide 08's feasibility figures. The HIPS wavelength (slide 07) resolved this morning:
632.8 ≈ 633 nm, from SPARTAN's own sources.

---

## Opening — 30 seconds, before slide 01

> Three things landed since you last saw this, and I need six decisions before tomorrow
> morning. The deck for Satoshi is built and it has not gone anywhere. Nothing here has
> been said out loud outside this room.

Do not walk the scorecard line by line. Let her read it while you say the above.

---

## Slide 01 — the audit

> Before anything else: the audit found a wrong claim in the deck and it is fixed. We had
> been saying the low-OC/EC cohort beats ten random cohorts on the held-out TOR test.
> There were five, not ten. The range was three-point-one to five, not four-point-four to
> seven-point-three. And one of the five random cohorts actually beat ours. Worse than any
> of that, the comparison was structurally meaningless — each cohort was scored on its own
> disjoint-site split, so those RMSEs were never comparable in the first place.
>
> The conclusion survives, on the evidence that is like-for-like. All six models on the
> same one hundred and ninety Addis filters: RMSE 1.16 against 1.48 to 2.24. Held-out TOR
> R-squared: 0.911 against 0.594 to 0.775. That is what the deck, the summary and the email
> now say. Fixed and pushed this morning.

Frame it as the check working, not as damage. The point of an audit is that it catches
things; this one caught something before an external audience saw it, which is exactly the
outcome we want.

**If she pushes on how it got in:** the number came in through the summary and propagated
outward to the deck and the email. No executed notebook ever produced it. That is the
signature to watch for — a claim with no cell behind it.

**If she asks what else changed:** three smaller things. The VIP convergence figure is
flagged uncorroborated and pulled from the email — no executed notebook computes it and the
replication it came from left no scripts, so we do not quote the correlation values at all
any more. The ETBI contrast is now against Addis's *Dry* median of 43.2, not the all-season
47.1, because ETBI's window is October to December. And the "ninety-one percent baseline"
line is qualified as one representative filter, which is what it always was.

---

## Slide 02 — what she is approving

> This is the deck. Five parts, fourteen figures, six asks. I am not walking you through it
> — you have seen most of it. What I need is a sign-off, and specifically on two lines of
> tone.
>
> First, the component-selection slide critiques his group's Calibration app. The finding is
> real — interleaved cross-validation flatters exactly the cohorts the deployed family is
> built from — and it is framed as "different question, not wrong tool". You know him and I
> do not. Tell me if that framing holds.
>
> Second, how hard we push MAC equals ten. The deck says the corrected model "lands closest"
> at MAC ten. Deliberately not "self-consistent", because 0.86 fails ftir_19's own bar of
> slope within nought-point-one of one, while the raw models at MAC six pass it. If you want
> that stronger or weaker, now is when it costs nothing.

**If she asks about the ask order:** it changes — say "that is slide 04, hold it thirty
seconds". Do not litigate the ask ordering here.

---

## Slide 03 — the intercept, restated

> This is the answer to your July question — why is the intercept negative when nothing at
> Addis is. Set predicted EC to zero and the intercept stops being a regression artifact and
> becomes a quantity you could go and look for: about twenty-one inverse megametres of
> absorption at the HIPS wavelength with no FTIR-EC behind it. That is forty-six percent of
> a typical Addis filter.
>
> Across all six calibration setups it runs eighteen-point-eight to twenty-six, median
> twenty-one-and-a-half, while the slopes across those same six span three-and-a-half fold.
> And it is exactly MAC-invariant by construction, so the target is well posed whether or
> not the MAC fork is ever settled.
>
> Now the caveat, before you get to it. C is invariant to any multiplicative rescaling of
> the EC axis. Changing MAC is exactly that, and changing training cohort mostly is. So
> these are not six independent routes agreeing. The honest claim is that the additive
> offset is robust to every scaling choice we have made — not that six methods corroborate
> each other.
>
> One thing that did land overnight, and it is the sharpest cheap test we had. AIRSpec
> touches only the spectra, never Fabs, and it halves the intercept. That looks like a
> refutation of "the offset lives in x". It is not, because it also halves the slope — and
> under y equals a times x minus c, the intercept *is* minus a times c. Slope ratio 0.54,
> intercept ratio 0.50, and c moves seven percent where a and b each move fifty.

**If she pushes on "so what does it license":** nothing about the cause. C is a restatement
of the intercept, not new evidence about where it comes from. Brown carbon, dust, a
HIPS-generic offset and an FTIR zero error are all still live on this slide alone — slides
04 and 05 are what start eliminating them.

**If she pushes on the −7.3%:** it is systematic, not noise. Across the six setups c runs
1.885 to 2.608 and the best-performing setup has the lowest c. A single shared c fitted
jointly costs only 1.6% in total RSS. But that agreement is confounded by the same
gain-invariance, so it corroborates rather than proves.

---

## Slide 04 — IMPROVE runs through the origin

This is the slide to give the most air. It is both the strongest support for the story and
its sharpest caveat, and she should hear the caveat from you rather than infer it.

> Tier-1 item one came back. The question was whether HIPS generically reads a few inverse
> megametres at zero EC — filter scattering, the tau-to-Fabs conversion, loading correction.
> If it does, part of our offset is instrument-generic. It does not. Regress IMPROVE Fabs on
> TOR EC across the hundred-and-fifty-thousand-filter join and the intercept is one-point-three
> pooled, nought-point-two trimmed, and nought-point-one in the Addis-like subset. At EC at or
> below zero the median Fabs is plus nought-point-one-two, with twenty-four percent of filters
> reading negative — that is zero at zero with symmetric noise. No individual site reaches
> five. And the arithmetic that settles it: only a hundred and eighty-six IMPROVE filters out
> of a hundred and sixty thousand carry twenty-one megametres of *total* absorption. A generic
> offset that size is not a thing that could exist.
>
> So the offset is Addis-specific. Now the part that cuts the other way, and I want you to
> hear it today rather than tomorrow.
>
> Two explanations survive this, and neither one is an absorber. The first is curvature. Fabs
> is concave in EC across IMPROVE — it goes as EC to the nought-point-eight, passing exactly
> through zero — and fitting a straight line to a concave relationship *manufactures* a
> negative intercept with no offset present at all. Per-site intercepts run at a median
> thirty-five percent of site mean Fabs. Addis is forty-six percent. A third of IMPROVE sites
> show a fraction at or above Addis's — in data where the true offset at zero is zero.
>
> The second is a loading-dependent HIPS artifact. IMPROVE's median Fabs is one-point-three
> against Addis's forty-seven — a thirty-seven-fold gap. That gap cleanly rules out an
> additive, loading-independent offset. It says nothing about one that grows with loading.
> And per-site intercept does grow with loading; extrapolate to Addis loading and you get
> sixteen megametres, three-quarters of our C. That extrapolation is six times beyond
> IMPROVE's largest site, so it does not close the question either — but it means I cannot
> rule it out.
>
> And separating those three needs a reference we do not have. Both ChemSpec columns are
> circular. BC is Fabs over ten, rounded — circular with x. EC I wrote up in ftir_25 as the
> independent one, and this morning I disproved that: against FTIR-EC it is r-squared
> nought-point-nine-nine-nine-seven, ratio one-point-zero-zero-zero-zero, median difference
> three thousandths of a microgram, which is exactly the two-decimal rounding half-width. It
> is our own FTIR-EC coming back through the speciation table. The tell was already in the
> numbers I had quoted — its R-squared against Fabs matches FTIR-EC's own, because it *is*
> FTIR-EC — and I read it as corroboration. ftir_25 is corrected and pushed.
>
> Which means: "go find twenty-one megametres of brown carbon" is a materially weaker framing
> than it was yesterday. But what replaces it is cleaner — not a generic HIPS zero, not on
> the FTIR axis, the MA350 cannot see it, and nothing in hand can discriminate what is left.
> That is the case for quartz TOR, made from four directions.

**If she pushes on the concavity number:** the 0.796 exponent is itself attenuated by the
same errors-in-variables problem, so treat it as an upper bound on how concave the
relationship really is — which makes it an upper bound on how much of our intercept
curvature could explain.

**If she asks whether we can test curvature on Addis directly:** not right now. It needs an
EC reference that is derived from neither Fabs nor FTIR, and after this morning there is no
such column in the committed dataset. It waits on quartz TOR like everything else.

**If she asks why 170 of 176 sites are "significant":** that is n-power, not magnitude. The
sites have thousands of filters each. Magnitude is the number that matters and no site
reaches five.

**Methodological aside, only if she asks about estimator choice:** the pooled intercept is
monotone increasing in lambda, so OLS is the maximum over all errors-in-variables lambdas.
Every EIV correction moves it toward zero. That makes the OLS column a hard upper bound
rather than a point estimate, which is why the "≤ 1%" claim is safe.

---

## Slide 05 — the FTIR side is exonerated

> Item six of the attack plan, and the rule was pre-registered before the fit. Fit FTIR-EC
> against the MA350's 880-nanometre channel, where brown carbon barely absorbs. If the
> intercept there is near zero while against HIPS it sits at minus two, the additive offset
> localises to the HIPS optics and the FTIR side is clean. (Say "the HIPS
> axis". The wavelength itself is settled now — 632.8 ≈ 633 nm, slide 07 — but "the HIPS
> axis" remains the right level of claim: the localisation argument never depended on it.)
>
> That is roughly what happened, with one honest qualification. Intercept plus
> nought-point-three-two, confidence interval plus nought-point-nought-one to plus
> nought-point-six-three. That interval does *not* quite contain zero — the lower bound
> sits just above it — so I am not going to claim a formal null. What it does say is that
> the FTIR-side intercept is about thirteen times smaller than the minus-four-point-one-seven
> we see against HIPS, and it points the other way. R-squared nought-point-eight-seven, a
> better fit than the HIPS comparison. So FTIR-EC's zero is not the problem, and the additive
> offset is on the HIPS axis.
>
> **If she asks why the number moved:** an earlier exploratory run had plus
> nought-point-two-nine with an interval that did contain zero. The committed notebook swept
> plus-or-minus one and two day matching, nearest-day, smoothed data, the other instrument
> record, cohort restriction, quality filtering and Deming — none of them reproduced it. So
> we quote the committed one.

**If she pushes on the scale gap:** she will, because it looks alarming. HIPS Fabs averages
49.7 against MA350 b_ATN at 625 averaging 111.6, a factor of 2.2. That is mostly the filter
multiple-scattering C-factor — b_ATN has not been corrected for it, Fabs has. It is expected
and it is *multiplicative*. The result here is *additive*, and the scale gap does not touch
it.

**If she asks why this is preliminary:** the fit is reproduced but not yet in a committed
script. Same status as slides 04 and 06. It commits tonight.

---

## Slide 06 — the MA350 cannot answer the BrC question

> This was the single most decisive analysis available without new sampling, and it is
> closed — but not the way we wanted. Anchor AAE_BC at one on the 880 channel, attribute the
> excess at 625 to brown carbon, and you get a *negative* brown-carbon absorption on
> eighty-five percent of days. Closing our twenty-one-megametre gap would need an AAE for
> black carbon of nought-point-three-two, which no combustion aerosol has.
>
> The instrument health check says why. On this instrument the Green channel implies a
> negative AAE, which is unphysical. UV is out of range on a third of days. And Red sits on
> top of IR to within channel reproducibility. Only the IR channel is trustworthy — and IR
> is precisely the channel that carries no brown-carbon information.
>
> So the sentence is: the MA350 cannot answer this question. That is a statement about the
> instrument, not about the air.

**Say the conclusion in exactly that form.** Do not let it drift into a claim about how much
brown carbon Addis has — we have not measured that and this instrument cannot.

**If she asks what came out of it anyway:** two live traps in the repo, both now fixed and
committed. The processed-sites README documented AE33 wavelengths — 370, 520, 660 — over
what are actually five-channel MA350 exports at 375 through 880, and that is the first file
a new reader meets. And the AAE helper defaults to a BCc-derived AAE, which is offset from
the atmospheric AAE by an exact identity, about minus one. On the Addis instrument that
turned 47% biomass into 12% through the classifier. Same symptom as the inverted-AAE bug the
module was written to prevent, different cause.

---

## Slide 07 — the wavelength, resolved

This was the question only she could answer — and it answered itself before the meeting.

> One open item closed this morning. HIPS is the helium–neon line: 632.8 nanometres, which
> rounds to 633. SPARTAN's own Instruments page says the filters are illuminated with a
> HeNe laser — and notes the analysis is done by your group — and the 2025 SPARTAN methods
> paper says He–Ne at 633 with a MAC of ten at 633. The 632-versus-633 spread across the
> documents is rounding of the same laser line.
>
> Three consequences. First, the 405 in RESEARCH_PROGRESS was wrong and is corrected — and
> that means the old AERONET dust null was aimed at the right wavelength after all; dust
> absorbs less at 633 than at 405, so that null gets stronger, and it narrows "real non-EC
> absorption" toward brown carbon specifically. Second, the MA350's red channel at 625
> sits essentially at the HIPS wavelength, so yesterday's no-red-excess result bears
> directly on the band HIPS measures — with the caveat that the channel cannot resolve
> small excesses. Third, classic brown carbon absorbs weakly at 633, so of the three
> surviving explanations, "real non-EC absorption" loses weight unless the absorber is
> char- or tar-like — the loading-dependent artifact and curve geometry gain
> correspondingly. None of this changes the ask: quartz TOR separates the survivors at
> any wavelength.
>
> What I still need is the semantics question. HIPS uncertainty and MDL exist as their own
> parameter rows — one-ninety out of one-ninety, median 2.9 inverse megametres, about six
> percent of median Fabs — giving a Deming lambda of about three rather than one. What
> does SPARTAN mean by that field: counting statistics, a repeatability estimate, or a
> propagated calibration uncertainty? Only the last two make lambda-star defensible as a
> measurement error.

**If she asks who confirmed the wavelength:** Ahmad, this morning, from SPARTAN's public
Instruments page and the 2025 SPARTAN methods paper — SPARTAN-side sources, which is
exactly what the optics reference said was missing. The reference and the 405 citation
are corrected in the same commit as this deck.

**If she cannot answer the semantics question today:** fine — assign the ask: who emails
SPARTAN, and does it go before or after tomorrow's meeting.

---

## Slide 08 — ftir_24

> Your hypothesis was that the dry, diesel-leaning aerosol is roughly right and the wet,
> charcoal-dominated seasons are what break. Against the committed residual table it comes
> out inverted. For the raw model Dry is the *worst* season at minus one-point-two-five,
> Kiremt is the *best* at plus nought-point-four-nine. For the corrected model the residuals
> are season-stable, minus two-oh-four to minus two-five-nine. So season sensitivity is a
> raw-model phenomenon — which is itself the interesting result, and consistent with the
> corrected model's error being a constant rather than a domain failure.
>
> There is a confound I want to design around before building it. Dry spans only about
> fifty-five percent of the wet seasons' x-range, so a naive per-season slope is mechanically
> attenuated — a shorter segment of the same line looks like a different line. It has to be a
> pooled fit with a season-by-x interaction, not three separate crossplots.
>
> Feasibility: it is a today job. About a gigabyte off the Drive, five minutes to regenerate.
> And the forty-three undated spectra worry is retired — all forty-three fall outside the
> two-thirty-nine evaluation set, which splits one-oh-five, sixty-one, seventy-three.
>
> So: today, or after tomorrow?

**If she asks about the February convention:** run `dry_feb` as the headline with `belg_feb`
as a sensitivity row. February is 22 of the 239 and moving it cuts Dry from 105 to 83, so
the two calendars are genuinely not interchangeable here — say which one you used on every
seasonal number.

**If she wants it in tomorrow's deck:** doable if the go-ahead is now. Say plainly that a
result built this afternoon has had no audit pass, and the last unaudited number that
reached a deck is slide 01.

---

## Close — the six decisions

> Six things I need from you, and then the plan I would put on tomorrow's next-steps slide.
>
> The ask order changes. Quartz TOR moves to the top — it is no longer one of six asks, it is
> the only instrument that separates the three explanations left standing, because every EC
> reference we hold is circular with one axis or the other. The spec is unchanged: eleven to
> thirteen days per season, about thirty-six filters total, quartz only. Behind that, Tier 1:
> item one is done, the bootstrap confidence interval on c needs no refitting at all — it
> comes straight out of ftir_15's committed draws — and the Deming sweep can now run with a
> real sigma-x. Then the HIPS-side scale and wavelength work, which slide 07 either unblocks
> or redirects. Tier 2 is closed.
>
> And the standing rule holds: nothing reaches Satoshi or the co-authors until you have
> approved it. That is what today was for.

Walk the six cards quickly and get a yes/no or an owner on each. The two that cannot slip
are ① the deck approval and ⑥ Hossein, because the co-author email is blocked on it and
nothing else in the project can unblock it.

---

## Caveats to have ready across the whole deck

- **Slides 01–07 are fully committed.** ftir_26/27/28 executed end to end and pushed
  (through e3d1f4f); only slide 08's feasibility figures await ftir_24 itself. The
  reproduction gate for 04 passed against ftir_16's committed join — n = 151,843, implied
  MAC median 11.96, Addis-like subset 6,503 giving 10.05.
- **Do not quote MAC 10.05 without saying "Addis-like, OC/EC ≤ 2.27, 6,503 filters".** On its
  own it reads as an IMPROVE-wide number and it is not; the pool median is 11.96.
- **Do not say "~36 filters per season".** It is ~36 filters *total*, across 11–13 days per
  season. Getting this wrong triples the ask.
- **Do not say the MA350 result means there is no brown carbon at Addis.** The instrument
  cannot see it either way. This is the single easiest thing to get wrong in the room.
- **The HIPS wavelength is now fact: 632.8 ≈ 633 nm, MAC 10 at 633.** Quote it with its
  sources (SPARTAN Instruments page; 2025 SPARTAN methods paper). What remains open is the
  *meaning* of `HIPS_Uncertainty` — keep that as the SPARTAN ask.
- **ftir_25 briefly named ChemSpec_EC as the independent reference and that was wrong.**
  It is corrected and pushed (00ab987), but if she read the md yesterday that is the version
  she has — get ahead of it rather than letting her raise it.

---

## Appendix R1–R3 — resurfaced from July 17

Only if time allows, or as leave-behinds. These are the July-17 threads the meeting
skipped, and together they are the evidence chain under the dry/non-dry split she asked
for a better justification of.

> R1 — the uncertainty slide we skipped in July. The raw and corrected bootstrap
> distributions are fully disjoint, and the corrected one still excludes zero: the AIRSpec
> improvement is real, and so is what remains. The residual panel beside it shows Dry
> separating downward in both models — the season structure was visible before we ever
> quantified it.
>
> R2 — the burning signature. Charring destroys O–H and aliphatic C–H and grows aromatic
> C=C at exactly the 1600 band, more strongly at higher temperature. That is a
> process-based argument for the aromatic reading of the Addis band — with the honest note
> that char peaks near 1590–1600 while Addis sits at 1617–1619.
>
> R3 — the justification you asked for. A mixture component fitted with no season labels
> isolates the low-char-similarity filters, and they are almost entirely dry-season:
> share anomalous by month runs 0.82 in November, 0.61 December, ~0.5 October–February,
> and essentially zero May–September. Not the analog R-values — a season-blind structure
> that recovers the seasons on its own. ftir_24 is what puts statistics under this.
