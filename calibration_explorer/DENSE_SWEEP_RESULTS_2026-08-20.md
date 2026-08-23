# Dense cutoff sweep (every 10th) — results, 2026-08-20

The full "every 10 filters" sweep the coarse grids never ran: 981
configurations under the **site-grouped protocol** (Eth-shaped 100–600,
analogs 250–750, lowest-OC/EC 300–1500, × raw/corrected selection × raw/
AIRSpec/deriv2 calibration, ~8 k values each) → 4,936 new scored rows;
results file now holds 9,628 rows total. Frame everywhere below: Deming,
MAC 10, fixed 190 set, score = |intercept| + 5·|slope − 1|, held-out
TOR R² ≥ 0.85 floor (919 rows pass).

## Headline: the locked ocec-800 cutoff was never the optimum

The dense scan finds a better basin the 5-point ladders (600–1000) could not
see because they never looked below 600:

| rank | config | fit | held-out R² | score |
|---|---|---|---|---|
| 1 | **ocec-450 × deriv2, k=20** | 1.02x−1.57 | 0.88 | 1.68 |
| 2 | **ocec-440 × AIRSpec, k=9** | 0.98x−1.62 | **0.92** | 1.72 |
| 3 | ocec-450 × AIRSpec, k=9 | 1.00x−1.71 | 0.92 | 1.72 |
| … | (k=7/8 variants at 440–490 all ≤1.9) | | | |
| ref | ocec-800 × AIRSpec, k=9 (prev. best) | 1.01x−1.84 | 0.87 | 1.86–1.91 |

It is a **basin, not a spike**: cutoffs 440–490 × k 7–9 all beat everything
the ladder grid ever produced, and the held-out TOR R² *improves* there
(0.92–0.93 vs 0.87 at 800). Slope ≈ 1.0 at MAC 10 with intercept ≈ −1.6.
Reading: the lowest-OC/EC ordering keeps getting more Addis-like below 800,
and ftir_11's "800 is the sweet spot" was an artifact of only scanning
{600,700,800,900,1000} — the meeting instinct ("try somewhat more and
somewhat less") was right, it just needed a finer step.

## Per-family best cutoffs (passing rows only)

- **ocec × AIRSpec: best 440** (score 1.72; 60 of 121 cutoffs pass) —
  robust family.
- **ocec × deriv2: best 450** (1.68; 30 pass) — derivative is competitive
  with AIRSpec *only* in this narrow region; elsewhere it fails the floor.
- ocec × raw: flat and bad everywhere (best 770, score 8.4) — raw spectra
  remain unusable regardless of cohort size.
- Eth-shaped and analogs: almost every cutoff **fails the held-out floor**
  (1–5 passing cutoffs per family). Shape-selected cohorts do not generalize
  under the honest protocol at any size — the dense scan closes Ann's
  cutoff question for them with "no cutoff fixes it."

## Caveats before adopting ocec-450

1. Multiple comparisons: ~6,500 site-grouped rows were screened against the
   same 190 Addis pairs. The defenses: the held-out TOR (independent of the
   readout) *improves* at 450, and the basin is wide in both cutoff and k.
2. k=9 (AIRSpec) and k=20 (deriv2) are manual-k rows (rule picks lower).
   **Confirmed on locked conventions 2026-08-22**: at rule-k (the
   first-major-minimum picks k=5 for ocec-450, same as ocec-800), the two
   cohorts are roughly equivalent — 450 gives 1.02x−2.26 fixed / 0.92x−1.85
   all (held-out 0.907) vs 800's 0.95x−2.09 / 0.87x−1.71 (0.904). The basin's
   intercept advantage (→ ≈−1.6) requires k=8–9, i.e. 3–4 components past the
   rule choice. So the honest claim is two-part: (a) at matched k the cohorts
   tie, with 450 slightly better on slope and held-out; (b) the improvement
   comes from k-selection, which connects to the established ftir_23 finding
   that both stopping rules quit well short of the curve minimum — and to
   Satoshi's standing "should k be larger?" question. Note the ETBI transfer
   used k=9 and held (0.93x), so the deeper-k choice generalizes
   out-of-country; the deriv2 k=20 member is the one that does not.
3. N=450 is a smaller, compositionally more extreme cohort — check the
   Selection tab's composition ruler and site spread for concentration into
   few sites before presenting.

## Cross-readout stability (added after the full-readout backfill)

Scored the 919 passing rows under all eight readout framings
({fixed, all-pairs} × {MAC 10, 6} × {Deming, OLS}, w=5):

- **At MAC 10 the ocec-440/450 basin is rank 1 under all four framings**
  (Deming/OLS × fixed/all-pairs). Best fits: 1.02x−1.57 (fixed Deming) to
  0.97x−1.05 (all-pairs OLS). The finding is not a framing artifact.
- **At MAC 6 the known fork surfaces exactly as ftir_19 predicts**: raw-spectra
  models become slope-consistent (raw slope ×0.6 ≈ 1) and analogs-260 × raw
  tops the board — but with intercepts −2.3 to −3.2, roughly twice the MAC-10
  branch's. The basin drops to rank 3–8 there because corrected-model slopes
  fall to ~0.6.

Net: within either MAC branch the answer is stable; across branches the MAC
fork remains the single deciding unknown, and the MAC-10 branch's best
intercept (≈−1.3 to −1.6) is about half the MAC-6 branch's best (≈−2.7). If
minimizing the intercept is the goal, ocec-440/450 + baseline-treated spectra
at MAC 10 wins under every estimator and evaluation set.

## Eval-lot-251 pass (the last un-swept axis; 981 configs re-read on lot-251
## filters only, 6,492 rows — results file now 16,120 rows)

Judged on the 191 lot-251 evaluation filters alone, the basin **holds at
rank 1** and improves: ocec-450 × deriv2 k=19 reads 0.98x−1.15 (ho 0.87), and
the AIRSpec k=8–9 variants read 0.95–0.99x with intercepts −1.4 to −1.6 (ho
0.92). The lot-248/253 evaluation filters were mildly diluting the readout;
nothing about the basin depended on them.

## Where to look in the app

Optimize tab → "Load saved results": the leaderboard and Pareto view now
carry all 9,628 rows; the ocec-440/450 cluster sits at the frontier corner.
Every row is one click to open as a live run.
