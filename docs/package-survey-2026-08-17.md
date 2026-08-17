# Package survey — existing tools for the FTIR-EC / aethalometer work (2026-08-17)

Three parallel web surveys of open-source packages that overlap what this repo
hand-rolls, run while building `calibration_explorer/`. Compact verdicts up front;
details + links below.

> **Update (same day):** the adopt-list candidates were then actually trialled against
> the repo's validated machinery — evidence, scripts, and final verdicts live in
> `research/package_trials/{baselines,chemometrics,domain}/README.md`. Where a trial
> verdict differs from the survey verdict below (e.g. pybaselines: survey said
> "replace", trial says "keep the port"; methcomp: survey said "vendor", trial found
> real bugs and rejects it), **the trial wins**.
>
> **Implemented in production code so far:** ikpls's fast CV is now the engine behind
> `calibration_modes.interleaved_cv_curve` (gated to problems ≥1e6 elements, ~5-12×
> on real cohorts, agreement ≤1e-13, silent fallback to the original loop when ikpls
> is missing or fails) — validated against the committed ftir_21 app-protocol result
> (k=17, 2.15x−4.59) on real data. Everything else on the adopt list is
> analysis-side (PyMieScatt, monetio, astartes, scipy.odr CIs): use it from
> `research/package_trials/` scripts as needed; the "keep hand-rolled" verdicts
> (AIRSpec port, selection rules, in-repo Deming closed form) are implemented by
> keeping the existing shared-script code.

## Headline verdicts

**Adopt / trial seriously**
- **pybaselines** (PyPI, active, FTIR-targeted) — could replace the hand-ported AIRSpec
  spline internals (`mixture_model`, `pspline_arpls`, `irsqr`) while keeping our PTFE
  segment logic as a wrapper.
- **ikpls** (active, JOSS) — same kernel-PLS lineage as R `pls` (our validation
  target); takes *arbitrary per-sample fold labels* (site-grouped and interleaved CV
  both drop in) and returns predictions for all 1..A components in one pass — exactly
  what our component-selection curves need, with NumPy/JAX backends.
- **chemotools** (active) — sklearn-Pipeline preprocessing (SG derivatives, SNV/MSC,
  baselines) so preprocessing hyperparameters get cross-validated jointly with PLS.
- **astartes** (± `kennard-stone`) — Kennard-Stone / SPXY / sphere-exclusion sample
  selection with sklearn-like API.
- **scipy.odr** — the maintained numerical engine for Deming-λ (ODR with fixed weight
  ratio); keep a thin wrapper exposing λ and bootstrap CIs.
- **AeroViz** (active) — readers + QC for **MA350**, AE33, OCEC, plus IMPROVE
  mass-reconstruction algorithms; the only maintained package covering both MA350 and
  IMPROVE. Verify its MA350 parser against our firmware's columns.
- **monetio** (NOAA ARL, active) — AERONET v3 web-service pulls straight into pandas,
  plus an IMPROVE FED DataWizard-export parser.
- **PyMieScatt** — standard Mie forward/inverse tool; grounds the MAC 6-vs-10 question
  theoretically (coated-BC MAC modeling).

**Borrow ideas / vendor small pieces**
- **methcomp** — Deming (with variance-ratio λ, bootstrap CIs) + Passing-Bablok +
  Bland-Altman; semi-dormant but short, readable code to cross-check our λ convention.
- **maruedt/chemometrics** — clean VIP + leverage/diagnostics code (dormant package).
- **orange-spectroscopy's EMSC with an interferent spectrum** — the one genuinely new
  idea for PTFE removal: EMSC can include a PTFE reference spectrum as an explicit
  interferent term (~30 lines of numpy to reimplement).
- **rampy's gcvspline baseline** — the closest single-function smoothing-spline
  analogue to APRLssb; worth a quick benchmark.
- **py-ona / ONA implementations** — Hagler ONA noise reduction for MA350 1-min data.
- **APRLmvr + APRLspec (Takahama GitLab, R, dormant)** — the only public
  implementation of the published FTIR-EC calibration workflow; reference to mine, not
  a dependency. The hosted **airspec.epfl.ch** app is still live and is the fastest
  ground-truth check for our AIRSpec port.

**Confirmed gaps (keep hand-rolled)**
- No package implements the component-selection rules (1-SE, first-major-minimum,
  within-5%-of-min) — sklearn's refit-callable example is the closest.
- No package does VIP-weighted Mahalanobis-in-score-space cohort selection.
- No official Python AIRSpec/APRLssb exists (R + Shiny only).
- No maintained Sandradewi/AAE two-component package; ~10 lines, keep in-repo.
- No public SPARTAN client or FED REST API; our loaders remain necessary.
- No PTFE-substrate-specific FTIR package anywhere — the two approaches this repo
  already uses (AIRSpec-style spline + SG second derivative) *are* the state of the art.

**Skip**: spectrapepper, scikit-spectra (dead), mbpls, trendfitter, pyphi (no PyPI),
pylr2, pyCompare, spacv/verde (spatial-coordinate CV — sites already define our
blocks, though blockCV literature is the right citation for the interleaved-CV-leak
argument), pyaerocom (heavier than needed), the PyPI package `aeronet` (it's a
remote-sensing raster library, not sun photometers).

Peripheral: **ESAT** (US EPA, active JOSS) is the obvious choice if PMF is ever run on
Addis speciation data.

---

## Full reports

### 1. Spectral preprocessing / baseline correction

| Package | Where | Status | Relevant | Verdict |
|---|---|---|---|---|
| pybaselines | [github.com/derb12/pybaselines](https://github.com/derb12/pybaselines) · [docs](https://pybaselines.readthedocs.io/) | 1.2.1 (Aug 2025), active | 50+ algorithms; `mixture_model` (penalized spline + EM, nearest cousin to APRLssb), `pspline_arpls`/`asls`/`aspls`, `irsqr`, Whittaker family, morphological + `snip` for broad backgrounds, anchor-region methods (analogous to fitting through PTFE no-analyte windows) | Replace AIRSpec spline internals; keep segment logic |
| APRLssb / AIRSpec | [gitlab.com/aprl/APRLssb](https://gitlab.com/aprl/APRLssb) · [airspec.epfl.ch](https://airspec.epfl.ch) · [AMT 2019](https://amt.copernicus.org/articles/12/2313/2019/) · [Kuzmiakova AMT 2016](https://amt.copernicus.org/articles/9/2615/2016/) | Dormant R; hosted app still live | The canonical method itself; R/Shiny only, no Python, no REST API | Validate our port against the live app or via rpy2 |
| chemotools | [github.com/paucablop/chemotools](https://github.com/paucablop/chemotools) | 0.4.3 (Jun 2026), active | sklearn transformers: AirPLS/ArPLS/poly/spline baselines, SNV, MSC, SG derivatives; Pipeline-native | Adopt for jointly-CV'd preprocessing |
| orange-spectroscopy | [github.com/Quasars/orange-spectroscopy](https://github.com/Quasars/orange-spectroscopy) | 0.9.3 (Jul 2026), active | EMSC **with interferent spectrum** (PTFE reference!), ALS family, rubberband; drags in Orange GUI stack | Borrow the EMSC-interferent idea |
| SpectroChemPy | [spectrochempy.fr](https://www.spectrochempy.fr/) | 0.12.3 (Aug 2026), very active | Reads OMNIC/Opus FTIR natively; interactive + auto baselines; framework-heavy | Complement (file I/O), not replacement |
| rampy | [github.com/charlesll/rampy](https://github.com/charlesll/rampy) | 0.6.4 (Mar 2026), active | `baseline()` incl. **gcvspline** (GCV-chosen smoothing spline — same family as APRLssb) | Benchmark as drop-in |
| spectrapepper | [github](https://github.com/spectrapepper/spectrapepper) | 2024, stale-ish | nothing beyond the above | Skip |
| scikit-spectra | — | dead (2015) | — | Skip |

### 2. Chemometrics / PLS / CV / EIV

| Package | Where | Status | Relevant | Verdict |
|---|---|---|---|---|
| ikpls | [github.com/Sm00thix/ikpls](https://github.com/Sm00thix/ikpls) | 6.1.2 (Jul 2026), active, JOSS | Improved-kernel PLS (Dayal & MacGregor — same family as R `pls` kernelpls); `cross_validate` takes per-sample fold labels (site-grouped / venetian-blind / custom) and returns all-components-in-one-pass predictions; JAX/GPU | Adopt as CV engine + PLS core |
| chemometrics (maruedt) | [github](https://github.com/maruedt/chemometrics) | 0.4.0 (2022), dormant | sklearn-subclassed PLS with VIP, leverage, residual diagnostics; `fit_pls` auto-LV by Q² with any CV splitter | Borrow VIP/leverage code |
| pyChemometrics | [github](https://github.com/Gscorreia89/pyChemometrics) | semi-dormant | T², DmodX outlier logic, permutation tests | Borrow ideas |
| pyphi (Garcia-Munoz) | [github](https://github.com/salvadorgarciamunoz/pyphi) | active, not on PyPI | NIPALS with missing data, T²/SPE, contribution plots | Skip as dependency |
| mbpls / trendfitter | github | unmaintained | multiblock PLS / PLS diagnostics | Skip |
| sklearn group CV | [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) etc. | core | covers site-grouped CV completely; venetian-blind is one line | Adopt (already in use) |
| spacv / spatial-kfold / verde | github | mixed | coordinate-based spatial CV; blockCV literature = the citation for the interleaved-leak argument | Skip; cite |
| scipy.odr | SciPy | maintained | Deming = ODR with weight ratio 1/λ | Adopt as engine |
| methcomp | [github.com/wptmdoorn/methcomp](https://github.com/wptmdoorn/methcomp) | 2020/2024, semi-dormant | Deming (λ, bootstrap CIs), Passing-Bablok, Bland-Altman | Vendor/cross-check λ convention |
| pyCompare / pylr2 / statsmodels | — | — | BA-only / RMA-only / no EIV | Skip |
| astartes | [github.com/JacksonBurns/astartes](https://github.com/JacksonBurns/astartes) | 1.3.3 (Sep 2025), JOSS | Kennard-Stone, SPXY, sphere exclusion; interpolation/extrapolation split framework | Adopt for KS/SPXY |
| kennard-stone | [PyPI](https://pypi.org/project/kennard-stone/) | 3.0.1 (Aug 2025) | KS-ordered `KFold`/`train_test_split`, sklearn-signature drop-in | Adopt if KS folds wanted |

No surveyed package implements the 1-SE / first-major-minimum / within-5% selection
rules or VIP-weighted score-space Mahalanobis cohort selection — those stay ours.

### 3. Aerosol / domain tools

| Package | Where | Status | Relevant | Verdict |
|---|---|---|---|---|
| APRLmvr / APRLspec / APRLmpf / AIRSpec server | [gitlab.com/aprl](https://gitlab.com/aprl) · [APRLmvr docs](https://aprl.gitlab.io/APRLmvr/) · [vignette](https://aprl.gitlab.io/spec_vignette/) | R, dormant (2019–2022) | The published FTIR-EC calibration workflow (staging, blanks, MDLs); spectra I/O conventions; the chemometrics vignette is the closest tutorial to IMPROVE-style calibrations | Borrow as reference implementation |
| UC Davis AQRC code | [aqrc.ucdavis.edu](https://aqrc.ucdavis.edu/tags/ftir) | — | No standalone public code; calibration offered as a service, defers to airspec.epfl.ch; [Reggente AMT 2019](https://amt.copernicus.org/articles/12/2287/2019/) has no code repo | — |
| AeroViz | [github.com/Alex870521/AeroViz](https://github.com/Alex870521/AeroViz) | pushed Aug 2026 | AE33/AE43/BC1054/**MA350** readers + QC, OCEC, Mie closure, IMPROVE mass reconstruction | Adopt (trial); verify MA350 parser |
| aerosol-magee-pytools | [github](https://github.com/Aerosol-Magee-Scientific/aerosol-magee-pytools) | official Magee, active | AE33/TCA parsing; **not** AethLabs MA350 | Skip unless AE33 appears |
| py-ona / ONA-in-Python | [github](https://github.com/jbandoro/py-ona) · [ncanha](https://github.com/ncanha/ONA-algorithm-in-Python-for-BC-data-noise-reduction) | new / 2020 | Hagler ONA noise reduction for 1-min BC | Borrow (~50 lines) |
| Sandradewi/AAE apportionment | — | none maintained | implement in-repo; parameter guidance in [AMT 2020](https://amt.copernicus.org/articles/13/1867/2020/); also see [micro-aeth uncertainty AMT 2026](https://amt.copernicus.org/articles/19/3123/2026/) | Keep in-repo |
| PyMieScatt | [github.com/bsumlin/PyMieScatt](https://github.com/bsumlin/PyMieScatt) | stable | Mie forward/inverse; MAC_BC for coated BC | Adopt for MAC theory |
| monetio | [github.com/noaa-oar-arl/monetio](https://github.com/noaa-oar-arl/monetio) | active | AERONET v3 web-service reader; IMPROVE FED-export parser (FED has no public REST API) | Adopt (AERONET) / borrow (IMPROVE) |
| SPARTAN tools | — | none public | CSV downloads only | Our loaders stay |
| ESAT (US EPA) | [github.com/quanted/esat](https://github.com/quanted/esat) | active, JOSS | Open PMF5 replacement (NMF, error estimation) | If PMF ever runs |
| pyaerocom / pyAERONET / aeraod.py | github/Zenodo | mixed | heavier or unmaintained AERONET options | Skip |

**Caution:** the PyPI package named `aeronet` is a remote-sensing raster library, not
the sun-photometer network.
