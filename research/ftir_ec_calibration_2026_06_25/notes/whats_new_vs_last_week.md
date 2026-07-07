# What's new this week vs. last week's deck

Guardrail so next week **advances** rather than repeats. Compared against last week's actual
presentation: `spartan_ec_2026_06_16/` (deck `charcoal_spartan_kbr_presentation.pdf`, 15 slides) and
notebooks `05_calibration_optimization`, `07_calibration_optimization_story`, `08_outlier_removal_explainer`.

## What last week's deck already covered (do NOT re-show as-is)
| Last week showed | Slide / nb | Status for next week |
|---|---|---|
| 906-sample calibration 1:1 (before cleaning), R²=0.835, RMSE=10.5 | slide 2 / nb05 | **Reframe only** — it's the un-cleaned baseline we're *returning to*, not a new result |
| Outlier removal before→after (61 removed, R² 0.835→0.942) | slides 3–5 / nb08 | **Reverse it** — the meeting says this was over-filtering. Don't re-present it as a win |
| Removed heavily-loaded vs typical IMPROVE spectra (WICA1/YOSE1/WHPA1) | slide 5 / nb08 | Already shown — only revisit via the *new* below-1:1/removed-only calibration |
| Jagged RMSEP / K-variance "K changes the Adama answer" | slide 6 / nb07 | **Already made** — don't re-derive the problem |
| Stabilization: outliers / repeated-CV / **K-ensemble (K=15–25) recommendation** | slide 7 / nb07 | **Already recommended** — the new component angle is Weakley 2nd-derivative, not the ensemble again |
| Adama general vs biomass(stabilized) vs TOR bars, char/soot annotated (0.02/0.07/0.17/0.58/0.07) | slide 8 / nb07 | **OPR already shown** — the new part is OPT vs OPR |
| Charcoal reference library / charring-temp spectra / PCA spectral map / binchotan / biochar | slides 9–14 | **Shelved** by the meeting ("too abstract") — drop |

## What is genuinely NEW (build the talk on these)
1. **The reframe / reversal** — last week's headline was the *cleaning win* (61 removed → R²=0.942).
   This week's headline is that this was **over-filtering**; keep everything, and the **below-1:1 /
   removed** samples are the signal. → the `below11` and `removed` calibrations (`05`, blocked on the
   labelled set) are the new experiments, not the `nofilt` refit alone.
2. **The 5 Adama FTIR spectra, individually** (`01`) — last week only showed *removed IMPROVE*
   spectra. The 5 Adama spectra + "does highest-EC show highest peaks?" are new.
3. **OPT vs OPR char/soot** (`03`) — last week was OPR only. The transmittance convention flips
   char-EC negative for 4/5 samples — a new, concrete result.
4. **Tor as the x-axis ground truth with a 1:1 line** (`02`) — last week showed side-by-side bars;
   the meeting's correction was to plot **Tor on x vs FTIR on y against 1:1**. Same 5 samples, new
   (and correct) framing — 4/5 below 1:1, J1269 the lone one above.
5. **All Ethiopia/ETAD spectra + the biomass-vs-diesel month split** (`04`) — **there was no Ethiopia
   in last week's deck at all.** This is wholly new, and feeds the EC-vs-FABS comparison (MAC=10).
6. **A consistent, written component rule** (`05`) — last week recommended an *ensemble*; this week's
   ask is a single reproducible "first major RMSECV minimum" rule applied to every variant, plus
   reading **Weakley** (2nd-derivative → ~4 components; first components model the Teflon).

## One-line framing for the talk
> "Last week we *cleaned* the calibration and it looked great. The feedback: it was too clean. This
> week we keep the samples we threw away, evaluate against Tor instead of FTIR-vs-FTIR, add Ethiopia,
> and test whether OPT and a second-derivative fit change the picture."
