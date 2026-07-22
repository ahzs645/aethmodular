# ftir_etad_char — Addis Ababa filter spectra vs the charcoal / biochar FTIR archive

Ten notebooks comparing the **ETAD (Addis Ababa) ambient filter FTIR spectra** against
the six published charcoal and biochar reference collections archived in
`research/ftir_hips_chem/charcoal_ftir_sources/` (see that folder's `README.md` for DOIs
and licenses).

Continues from `research/ftir_ec_phase3/`, whose loaders and validated AIRSpec baseline
this work depends on.

## Contents

| Notebook | Question | Headline result |
|---|---|---|
| `char_01_reference_spectra_survey.ipynb` | What is in the archive, collection by collection? | 5,889 spectra, four grids, two value scales, all showing one charring trajectory — but the aromatic C=C band **turns over near 300–500 °C**, so no single band index maps to a unique temperature |
| `char_02_addis_vs_charcoal.ipynb` | Does Addis look like charcoal, and at what temperature? | Charcoal-adjacent, not charcoal: best match **r ≈ 0.78** vs **0.99** charcoal-to-charcoal; implied **400 °C tier**, ±100 °C at best; Addis is less aromatic and more aliphatic/oxygenated than pure char |
| `char_03_burned_vs_unburned.ipynb` | Burned or unburned? | Not answerable as posed — Addis sits **off** the burned/unburned axis on all five band axes, and 59 % of filters lie beyond the training cloud's 99th percentile. The classifier's confident answer is extrapolation |
| `char_04_spectral_typology.ipynb` | What natural types exist, with no labels supplied? | The archive's structure is **charring stage, not provenance** (Cramér's V 0.61 vs 0.29). Addis forms **its own type** — 97.9 % in a single cluster essentially unshared with charcoal |
| `char_05_addis_vs_unburned_feedstock.ipynb` | Is any specific raw biomass a good Addis analogue? | *Within McCall*, raw feedstock beats char for **85 %** of filters (median mean r 0.415 vs 0.278), barley straw closest. But the full charcoal library still beats both (r ≈ 0.78), so this is a statement about McCall, not evidence Addis is unburned. Strongest in the **Dry season** (93 %) |
| `char_06_dry_season_anomaly.ipynb` | What is going on in the dry season? | A **distinct, coherent, episodic spectral type** on ~75 % of Oct–Feb filters, recurring across four years. Survives controls for **filter lot** (p = 0.89 within season), **loading** (Fabs p = 0.52, EC p = 0.72 within season) and **spectral noise** (anomalous spectra are *less* noisy and *more* coherent) |
| `char_07_char05_without_dry_season.ipynb` | Was char_05's result just the dry-season anomaly? | No — raw feedstock still beats char for **78 %** of the 134 non-dry filters (Wilcoxon p = 4e-11), but the margin was always small (median delta +0.014 → +0.009). The **feedstock ordering flips**: grasses/straw gain, woods/husk lose. Identical under both exclusion rules (Kendall τ = 1.00) |
| `char_08_fire_vs_furnace_char.ipynb` | Which *kind* of charcoal — fire-produced or furnace, woody or herbaceous? | Closes a real gap: 1,934 combustion-facility spectra were **silently excluded** from char_02/04 by the temperature filter. Tested now, **real-fire char matches worse** than furnace charcoal (0.566 vs 0.776, p = 1e-40, 0 % of filters favour it) — not a noise or library-size effect. No woody/herbaceous signal. The dry season matches **every** class worse |
| `char_09_char08_dry_vs_nondry.ipynb` | Is char_08's ranking just an average over two regimes? | No. Class ranking is **identical** dry vs non-dry (Kendall τ = 1.00) — the dry season shifts every class down together. The furnace−fire gap holds at **0.196** (non-dry) / 0.192 (anomaly rule) vs 0.211 overall, and on non-dry filters **no** filter favours fire char. Excluding the dry season raises furnace charcoal 0.776 → **0.810** |
| `char_10_seasonal_spectra.ipynb` | What do the spectra look like season by season — raw and processed? | Purely descriptive, no charcoal; shown in the **AIRSpec-baselined + area-normalized** representation rather than SNV. The structure is **one-vs-two**: dry correlates 0.94–0.955 with the rainy seasons, while Belg and Kiremt correlate **0.980** with each other — pooling them as "non-dry" was justified at shape level, though Kiremt has more aliphatic C–H and less O–H at band level. Includes the Weakley-style **raw spectra** figure showing season differences before AIRSpec are mostly baseline |
| `char_11_normalization_alternatives.ipynb` | Is there a better way to overlap them than SNV — including AIRSpec on both sides? | Mostly a non-question: under Pearson correlation SNV, vector, area, band-norm and *no normalization* are **identical to ~1e-15**. **AIRSpec cannot be applied to the charcoal side at all** — only 1 of 6 collections reaches its 3710 cm⁻¹ anchor — and forcing a truncated version halves cross-lab recovery (77 % → 34 %) and sends the temperature silhouette negative. **Keep the current asymmetric chain** |

## The baseline correction is load-bearing — read this before reusing anything

ETAD filter spectra are collected on **PTFE (Teflon)** filters, confirmed by the C–F
doublet at 1150 / 1210 cm⁻¹ in the raw data (quartz would show Si–O at ~1050 and ~800;
it does not). They carry a large sloping scattering background.

**A constant offset correction does not remove that slope.** The diagnostic used
throughout is `ramp_score` — the correlation between a mean spectrum and wavenumber, where
±1 means "featureless ramp" and 0 means "real band structure":

| Spectra | ramp score |
|---|---:|
| Addis, offset-corrected only | **+0.995** |
| Addis, AIRSpec-baselined | +0.695 |
| Addis, AIRSpec + linear detrend (**used**) | 0.000 |
| Charcoal collections, as published | −0.40 to −0.48 |
| Charcoal, + linear detrend (**used**) | 0.000 |

At +0.995 the Addis mean spectrum has no resolvable bands at all, and every "band
intensity" read off it is really a position on a slope — pointing in the *opposite*
direction to the charcoal collections' residual slope. An earlier version of `char_02`
built on offset-corrected spectra reported r = 0.60 with every filter pinned to the
library's lowest temperature; on properly baselined spectra the same code gives r = 0.78
with the estimate interior to the range. **The conclusion depended on the baseline.**

The canonical chain is therefore `prepare()` = **crop → linear detrend → SNV**, applied
identically to both sides, on **AIRSpec-baselined** Addis spectra. Cropping first keeps
excluded regions out of the normalization constants; detrending before SNV means the
normalization is computed on band structure rather than on a slope.

## Reading the figures: SNV and Δ SNV

Nearly every y-axis in this folder is **SNV absorbance** — a per-spectrum z-score along
the wavenumber axis:

```
SNV(w) = (absorbance(w) − mean of that spectrum) / (sd of that spectrum)
```

Dimensionless; the unit is "multiples of this spectrum's own sd". Zero is the spectrum's
own mean absorbance, not zero absorbance. It exists because the sources differ by ~30× in
absolute absorbance for instrumental reasons (KBr pellet vs thickness-normalized ATR vs
micrograms of aerosol on PTFE), so only band *shape* is comparable. Maezumi and Gosling
were published already in this form.

**What it costs:** absolute intensity. These axes cannot say how much material is on a
filter. And since each spectrum is scaled by its own sd, one band shrinking makes the
others rise mechanically — so multi-band shifts are described, never pinned on a single
compound.

**Δ SNV absorbance** is a different quantity: a subtraction (Addis mean − reference mean).
The zero line means the two agree; above means Addis has more of that band, below means
less. Those plots contain no spectra — every feature is a mismatch.

## Comparison window

**1430–3500 cm⁻¹**, and everything is limited by it:

- Below ~1425 the PTFE filter dominates and the AIRSpec segmented baseline does not
  extend there (`airspec_baseline.py`, SEG2 lower bound = 1425).
- Above 3500 the charcoal collections stop.

That retains O–H/N–H, aliphatic C–H, carbonyl and aromatic C=C, but **excludes the entire
C–O fingerprint**, which `char_03` shows carries much of the burned/unburned contrast.
This is the single biggest limitation of the whole comparison.

## Layout

```
ftir_etad_char/
├── scripts/
│   ├── charcoal_spectra.py   # loaders for the 6 collections + prepare/snv/detrend/ramp_score
│   ├── etad_spectra.py       # ETAD spectra via the phase-3 loader (raw or AIRSpec)
│   ├── run_char_01..10.py    # percent-format notebook sources
│   └── build_notebooks.py    # executes them into committed notebooks
├── char_01..10*.ipynb        # executed outputs
└── output/{plots,tables}/    # generated figures and CSVs (git-ignored)
```

## Reproducing

From this directory, with the repo environment (`uv sync --python 3.13` at repo root):

```bash
uv run --project ../.. python scripts/build_notebooks.py        # all ten
uv run --project ../.. python scripts/build_notebooks.py 02     # just one
```

Each notebook is generated from its percent-format script and executed top to bottom with
nbclient, so every number and figure in the committed `.ipynb` came from a real run.

Requires the Google Drive mount for the ETAD spectra (resolved through
`pls_transfer.FTIRTransferPaths.defaults()`) and the phase-3 AIRSpec cache at
`../ftir_ec_phase3/output/corrected/etad_corrected_df6.npz`.

## Figure conventions

Every figure containing Addis spectra carries a footer line naming the preprocessing
(`AIRSpec-baselined → linear detrend → SNV`), and Addis legend entries say "AIRSpec".
Given how much the baseline treatment moved the results, that provenance travels with the
figure rather than living only in the surrounding prose.

**Companion views.** From `char_02` on, every figure that overlays actual spectra
(means and spread bands — not the Δ-residual or statistic figures) is followed by a
companion version in absorbance-like units: the same chain with the final scaling
swapped from unit sd to **unit area** (`shape_norm`: detrend → shift to zero minimum →
unit area; the Addis side AIRSpec-baselined as always). Curves there read as band
intensity rather than z-scores. All statistics are computed on the SNV form only —
Pearson correlations are affine-invariant (`char_11` verifies this to ~1e-15), so the
companion view changes nothing numeric. `char_10` is the exception: it compares Addis
only with itself and uses the area-normalized representation throughout.

## The dry-season anomaly (char_06)

The clearest positive finding in this folder. Roughly three-quarters of October–February
Addis filters form a spectral type that does not match the charcoal references, recurring
every year across four sampling seasons (82 % of November filters, 0 % May–September). It
is a genuine two-population split, not a seasonal mean shift — about a quarter of
dry-season days look entirely ordinary.

Three competing explanations were tested and all fail:

| Explanation | Test | Result |
|---|---|---|
| Filter lot (248 is dry-season heavy) | within lot 251 alone; within dry season alone | season separates at **p = 9e-12**; lot does not (**p = 0.89**) |
| Low deposit loading → noisy spectra | Fabs / EC within the dry season; logistic model | **p = 0.52 / 0.72**; season absorbs loading entirely |
| Poor spectral quality | roughness and within-group coherence | anomalous spectra are **less** noisy (0.0087 vs 0.0142) and **more** coherent (r 0.954 vs 0.931) |

`output/tables/char06/per_filter_classification.csv` carries the per-filter labels — a
pre-specified target list for the dry-season quartz TOR campaign proposed in phase 3.

`char_07` then re-runs `char_05` with those filters removed. The raw-beats-char result
survives (78 % of non-dry filters, p = 4e-11) so it is not an artifact of the anomaly —
but the margin was always small, and the two regimes turn out to resemble **different
plant material**: non-dry Addis matches grasses and straw best (barley straw 0.569),
while the dry season matches rice husk and woods best (rice husk 0.483). That contrast is
the most specific compositional lead in this folder.

## Charcoal is not soot — the limit on all of this

Every reference collection here is **char**: carbonized solid fuel, formed inside the fuel
particle. Atmospheric black carbon at an urban site is largely **soot** — gas-phase
nucleated particles from incomplete combustion, structurally and chemically distinct. The
archive contains **no wood-smoke soot, no diesel soot, and no standardized black-carbon
reference material**.

That absence is the most plausible single reading of this folder. Addis matches char at
r ≈ 0.78 at best, sits at the 96th percentile of the charcoal distance distribution
(`char_02`), and forms its own cluster (`char_04`) — all consistent with the dominant
carbonaceous component simply not being in the reference set. Closing it needs soot
spectra, ideally measured on PTFE filters, which would also relieve the 1430 cm⁻¹ window
limit.

Note also `char_08`'s caution: Addis matching *furnace* charcoal better than *fire-produced*
char says more about the furnace library being a designed 10-species × 7-temperature grid
that covers composition space densely than about how the Addis material was produced.
`char_09` confirms that ordering is not a dry-season artifact — it holds, with the gap
intact, under both exclusion rules — so the caution stands on its own rather than being
explained away by the anomaly.

**Quote similarities by subset.** `char_09` shows the headline r ≈ 0.78 is an average:
it is ~**0.81** on ordinary days and ~**0.70** on dry-season days. Any future citation of
an Addis–charcoal similarity should say which subset it refers to.

## Preprocessing: what the choice does and does not affect (char_11)

`char_11` benchmarks alternatives to SNV and reaches a mostly-negative result worth
knowing before anyone proposes changing it:

- **Correlation-based results are immune to the normalization choice.** SNV, vector (L2),
  area, band normalization and even no normalization give identical Pearson correlation
  matrices (max |Δr| ≈ 8e-16), because correlation is invariant to per-spectrum offset and
  positive scaling. All the analogue-matching results are therefore not sensitive to it.
- **Distance-based results are not.** `char_02`'s Mahalanobis distance and PCA score space
  would move under a different normalization — Euclidean distances under band
  normalization correlate only 0.64 with SNV distances.
- **Derivatives are the only genuine alternative, and they fail here.** They raise the
  temperature silhouette ~8x, but degrade cross-laboratory temperature recovery
  (51 % vs 77 % within 100 °C) and send 99.6 % of Addis filters' best match to the 0 °C
  tier — a class holding 1 % of the library. EMSC is worse still on every measure.
- **The implied temperature is preprocessing-dependent** (400 °C under SNV, 150 °C under
  2nd derivative, 800 °C under EMSC), which is a further reason not to quote `char_02`'s
  400 °C on its own.
- **AIRSpec cannot be made symmetric, and should not be.** Its segment-1 fit anchors at
  3710 cm⁻¹; only McCall barley (4000) reaches it, McCall six-feedstock is marginal (3701),
  and the other four collections stop at 3500–3599. A truncated AIRSpec on the charcoal
  side flips its residual slope (−0.48 → +0.40), collapses cross-lab temperature recovery
  to 34 %, and drives the temperature silhouette **negative**. The asymmetry in the
  current chain is forced by what the published archive covers, not an oversight.
- **Beware optimizing preprocessing on the Addis answer.** Truncated AIRSpec applied to
  both sides gives the *highest* Addis–charcoal correlation of anything tested (r = 0.993)
  while scoring worst on every objective measure — it flattens both sides into a similar
  smooth residual. This is why the benchmark uses charcoal-only labels.

## Known gaps

- **No unburned reference on filters.** The only unburned material in the archive is 24
  bulk McCall spectra from one laboratory. `char_03`'s unburned recall is 41.7 % — the
  class is too small and too far from Addis in preparation to support a conclusion.
  `char_05` gets more out of it by treating the six feedstocks separately, but the
  absolute agreement there (r ≈ 0.5) is still far below what like materials achieve.
- **Temperature resolution is ~±100 °C.** Six coarse tiers (200–700 °C) hold ~15 % of the
  library each; the fine 50 °C McCall steps under 1 % combined.
- **400 °C is the least identifiable point in the range**, and it is where Addis lands —
  it is the turnover of the aromatization index (`char_01`).
- **Pure references vs an ambient mixture.** A filter that is a minority char by mass
  cannot be distinguished from one that is mostly char of a slightly different type.
