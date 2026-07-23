# Talk track — how many components, and why the two protocols disagree

Two slides on component selection, answering the question "the site-held-out protocol uses
5–6 components and the deployed models use 21–38 — who's right?" Figures live in `output/plots/deck/`;
numbers come from `ftir_20_component_selection_across_setups.ipynb`
(`output/tables/ftir20/`). Cohorts are the same six as `calibration_setup_matrix.png`.

The honest framing: **the two protocols differ in two places at once**, and only one of
them turns out to matter. Don't lead with the component counts — lead with the error floor.

---

## Slide 1 — `component_selection_all_setups.png`

> Every calibration here is PLS, and PLS has one dial: how many components. Both protocols
> read it off a cross-validation curve, but they build that curve differently in two ways.
> The Calibration app splits the filters into ten interleaved folds — every tenth sample —
> and reads the number off the curve. The site-held-out protocol holds out whole IMPROVE
> *sites*, so no site appears in both training and testing, and takes the first real
> minimum.
>
> Each panel is one calibration setup. Blue is raw spectra, red is second-derivative.
> Solid lines are the Calibration app's interleaved folds; dashed lines hold sites out. The gap
> between a solid line and its dashed partner is the whole argument: that's how much of
> the apparent accuracy came from having already seen that site.
>
> Look at biomass-smoke, top middle. Interleaved, it looks fine. Hold the sites out and
> the error floor goes up sixty percent. Now look at lowest-OC/EC, bottom middle — the
> two lines sit on top of each other. That cohort doesn't care how you fold it.

**Why the y-axis is log:** the raw interleaved curves spike to several hundred percent on
the smoke cohorts. Linear scale would flatten the floor region, which is where the
selection rules actually act.

**Why %RMSECV rather than µg:** the cohorts have very different mean loadings (5.9 µg for
the pool, 16.9 for smoke), so absolute RMSECV isn't comparable across panels. Absolute
values are in `component_selection_summary.csv`.

---

## Slide 2 — `k_by_rule_ladder.png`

> Here are the component counts. Filled circles are the Calibration app protocol, open
> squares are site-held-out. They disagree by up to about threefold — but notice they don't
> disagree in the same direction every time. Ethiopia-shaped smoke goes *up* under the
> site-held-out rule, and the spectral analogs tie. So "the app always uses more
> components" isn't a claim I can make, and I don't want to make it.
>
> The number that does hold up is on the previous slide. Hold sites out and the error
> floor rises by sixty percent for biomass-smoke and forty for Ethiopia-shaped smoke, but
> only one or two percent for the pool and for lowest-OC/EC. Interleaved folds flatter
> exactly the cohorts the deployed calibration is built from — because those cohorts
> concentrate repeat sampling at a handful of sites, so the same site sits on both sides
> of a fold.
>
> That's the point for Addis. Addis is a site nobody has seen. The cohort whose skill
> doesn't depend on fold structure is the one I'd trust to go there.

**The mechanism, if someone pushes:**

> On second-derivative spectra that whole effect disappears — every cohort lands between
> 0.89 and 1.06, smoke-906 included, which falls from 1.59 to 1.00. Taking the derivative
> removes the smooth baseline. So what an interleaved fold is leaking is the site's
> characteristic *background*, not its chemistry. That is the same quantity in the AIRSpec
> slides, arrived at from a completely different direction.

---

## Caveats to have ready

- **The top-left panel is the entire IMPROVE network**, 13,010 filters, no selection —
  not "deployed SPARTAN." The deployed model is a network calibration trained on IMPROVE;
  the only SPARTAN spectra we hold are ETAD's, and those are the *evaluation* set, never
  training data. Calling a training cohort "SPARTAN" is a category error, so the matrix
  slide's first row should be read as "whole IMPROVE network" on the training side.
- **The analog cohort is 500, not 400.** The matrix slide says 400; that's the superseded
  ftir_09 exploratory selection. The locked cohort is the top 500. Worth fixing on that
  slide.
- **Curves are on full cohorts; the locked k values were picked on training splits.** So
  the ladder's open squares need not equal the "locked k" annotations, and don't. Both are
  shown deliberately.
- **Second-derivative is not a free win.** It rescues the baseline-dominated cohorts and
  makes lowest-OC/EC *worse* (62% → 77%). The derivative and the AIRSpec baseline fix the
  same problem, so they're alternatives, not additive. If asked about Weakley: his argument
  for choosing k on second-derivative spectra applies to the raw-spectra calibrations, not
  to the AIRSpec-corrected one.
- **If someone asks how bad the absolute numbers are:** under site-grouped CV the full-pool
  and smoke-906 raw calibrations sit at RMSECV roughly equal to the mean EC loading itself.
  That is a real statement about transfer difficulty, not a bug.
- **This is not a claim that the app is wrong as a tool.** Interleaved CV is the `pls`
  default. The issue is what it means for a *network* calibration evaluated at an unseen
  site — a question the tool wasn't built to answer.

---

## Slide 3 (optional deep-dive) — `by_protocol/<protocol>/component_selection.png`

Use when someone asks "so how did you actually pick 5?" — one slide per protocol, the
curve with the rule drawn on it.

> This is the same cross-validation curve, with the rule that reads it drawn on top. Ours
> holds out whole sites, so we get an error band — that shaded ribbon is one standard
> error across folds. We take the first real minimum that falls inside one standard error
> of the best point. The app pools its folds into a single number, so it has no band at
> all; its rule is "anything within five percent of the best is as good as the best," and
> it takes the first component count that gets there.
>
> Two things I'd point out. First, both rules stop well short of where the curve actually
> bottoms — look at the spectral analogs, the curve keeps falling to twenty-six or
> twenty-nine components and both rules refuse to follow it. That's the rule doing its job:
> that's the cohort with no held-out skill at all, so chasing the minimum would have picked
> the worst calibration we have. Second, look at how wide the error band is on the network
> and smoke panels. Saying "the minimum is at seventeen" isn't something those data
> support. The app's curve looks more decisive only because pooling threw that information
> away.

**If asked whether the protocols ever agree:** yes — biomass-smoke and lowest-OC/EC +
AIRSpec bottom early and the site-held-out rule takes their true global minimum (k = 4 and
k = 5). The disagreement is specifically about long flat tails.

---

## If asked "what about all the other figures?"

`K_SENSITIVITY_AUDIT.md` goes figure by figure. The short version:

> The band-identity work, the MAC bridge and the spectra comparisons don't involve a
> calibration at all, so the component argument can't touch them. The MAC result is
> algebra — it holds for any number of components. What is protocol-conditional is the
> intercept column of the setup matrix, which now shows both, and the uncertainty work in
> ftir_15, which is still conditional on the site-held-out component choice and would need
> re-running if we wanted it both ways.
