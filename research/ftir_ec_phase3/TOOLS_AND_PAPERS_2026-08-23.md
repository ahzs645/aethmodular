# Packages and papers for the spectral comparison — survey 2026-08-23

Agent-assisted survey (web + PyPI/CRAN), maintenance verified 2026-08-23.
Companion to `BAND1617_LEAD_2026-08-23.md`.

## The five things that change what we do next

1. **Li et al. 2024 (*AMT* 17, 2401) is the paper we needed.** PTFE-transmission
   FTIR of cookstove emissions including **56 charcoal samples**; charcoal shows
   a prominent ~1600 cm⁻¹ C=C, and the **aromatic C=C(1600)/aromatic-CH(750)
   ratio is higher for charcoal than any other fuel** ("aromatic compounds are
   poor in hydrogen"). A published, fuel-diagnostic ratio computable on our
   sites. Mean spectra are open (Zenodo `10.5281/zenodo.10970006`, CC-BY,
   Charcoal/Kerosene/Red Oak × 3 phases, 419.8–3999.4 cm⁻¹).
2. **The confirming band (750 cm⁻¹ aromatic C–H oop) is below the APRL cutoff.**
   The Takahama lineage truncates to >1500 cm⁻¹ (Ruthenburg 2014 windows
   3700–2500, 1820–1500). The char test — C=C *without* matching C–H — needs
   900–700 cm⁻¹, so extending the baseline below 1500 is a prerequisite.
3. **Our AIRSpec anchor suppresses the very band we are chasing** —
   `find_min_pos(interval=(1600,1520))`. Verified; see Correction 2 in the band
   doc. Any 1500–1650 claim needs ≥2 baselines.
4. **MAC at 633 nm differs sharply by EC type**: soot-EC **13.7 ± 3.8** vs
   char-EC **5.4 ± 2.5** m² g⁻¹ (Han group). Exactly HIPS's wavelength. Note
   this cuts *against* a naive char story (more char → lower MAC → lower slope,
   not a positive intercept), though Liu et al. 2025 (*JGR* 10.1029/2024JD043079)
   finds soot > char "except from biomass burning". We already have the
   char-EC/soot-EC machinery in
   `research/spartan_ec_2026_06_16/_build_03_adama_han_char_soot.py`
   (char-EC = EC1 − OP, soot-EC = EC2 + EC3).
5. **No one has published a bulk-filter FTIR signature for tar balls / dark
   BrC.** Adachi et al. 2024 (*ACP* 24, 10985) is TEM-only; Corbin &
   Gysel-Beer 2019 uses SP2. The only mid-IR tar-ball data is lab-generated
   (Tóth et al. 2018). **If our 1617 band holds up it would be the first
   ambient one.**

## Software — install list

```bash
pip install pybaselines lmfit chemotools spectrochempy
```

- **pybaselines** 1.2.1 (2025-08-10, healthy) — already importable in this
  environment but **not pinned in requirements**. `spline.pspline_arpls` is the
  right second opinion: same function class as APRLssb (penalized spline) so
  disagreement isolates the *anchor/weighting* logic rather than the basis.
  `custom_bc(..., regions=((1500,1800),))` is the direct "is the band being
  eaten?" test; `rubberband`/`mor` are shape-only cross-checks.
  **PTFE caveat:** zero-weight ~1300–1100 cm⁻¹ (saturated CF₂) and ~700–500
  before fitting, else every algorithm warps the neighbouring continuum. Boris
  et al. 2019 also flags substrate features at **~1780 and ~1545 cm⁻¹** —
  bracketing our band on both sides.
- **lmfit** 1.3.4 — the only maintained choice for 1500–1800 deconvolution.
  Bound centers (aromatic C=C 1595–1615; COO⁻ asym 1560–1610; H₂O bend
  1630–1650; C=O 1695–1740) or a 4-band fit silently swaps identities between
  sites. Tie widths via `expr` rather than freeing them. Use
  `conf_interval2d` on the C=C/COO⁻ amplitude pair — a thin diagonal ridge means
  the decomposition is not identifiable and must be reported as such.
  Second-derivative seeding (`savgol_filter(deriv=2)` → `find_peaks`) is
  publishable *standalone*, independent of any fit.
- **spectrochempy** 0.12.5 (2026-08-21) — best-maintained MCR-ALS in Python
  (`MCRALS`, `SIMPLISMA`, `EFA`). Do the **row-augmented** fit: stack all five
  sites, fit shared components, read concentration profiles — that directly
  answers "does Addis load on a component Beijing doesn't". `pymcr` is frozen
  (0.5.1, 2021) but fine as a cross-check.
- **chemotools** — only maintained Python **EMSC**; with a PTFE blank as
  reference it models the substrate explicitly rather than as unknown smooth
  background. Pin the version (API churns).
- R: **mdatools** (MCR + PLS with VIP/selectivity ratio), **OpenSpecy**
  (broadest format readers; `cor_spec()` to *rule out* a polymer/fiber
  contaminant — a negative result worth having before a novel-species claim).
- **Avoid**: `matchms` (peak-list data model, wrong for continuous absorbance),
  `pysptools`/`scikit-spectra`/`BaselineRemoval` (abandoned), `specutils`
  (astronomy semantics), `pyspectra` (toy).

**Notable**: the entire APRL toolchain (APRLssb, APRLspec, AIRSpec) is dormant
and R-only — **our validated Python port is currently the only maintained path
to that algorithm**, which belongs in the methods section. Takahama's active
repos are now Julia optical-constants work (`KramersKronig.jl`,
`BSplineDielectric.jl`, updated 2026-08-20) — a Kramers–Kronig bridge from
IR-derived optical constants to visible absorption is *exactly* our problem
shape, and worth an email.

## Papers — band assignment near 1600–1620

- **Bürki et al. 2020 (*AMT* 13, 1517)** — the calibrated FG set deliberately
  **excludes aromatic C=C**, justified by absent 3500/3100 cm⁻¹ features. **Their
  exclusion test (3100 cm⁻¹ aromatic C–H stretch) is directly runnable on Addis.**
- **Takahama, Ruggeri & Dillner 2016 (*AMT* 9, 3429)** — sparse-PLS finds
  ~1600 cm⁻¹ is *required* to predict **TOR EC** from PTFE spectra, attributed to
  "sp² bonds in ring-structured substances known to be emitted from combustion".
  Falsifiable signature: a broad 1800–900 continuum under the peak distinguishes
  disordered graphitic char from discrete molecular bands (Friedel & Carlson
  1971/72).
- **Debus et al. 2022 (*AMT* 15, 2685)** — "graphitic carbon displays peaks near
  1600 cm⁻¹ due to lattice defects"; predicts OC, EC, ions, metals, PM mass
  **and light absorption** from one PTFE filter across 161 IMPROVE sites. The
  natural model to port to our five sites.
- **Competitors to rule out**: ammonium oxalate at **1610** (Boris et al. 2019,
  *AMT* 12, 5391 — testable against IC oxalate); carboxylate needs a **~1400
  symmetric partner** and a counter-cation; primary amine bending is placed at
  1620 by Liu et al. 2018 (*ACP* 18, 8571) — though ftir_12 already excluded
  amine here, and Kamruzzaman et al. 2018 note amines show no strong spatial
  structure, unlike our band.
- **Rezaeian et al. 2026 (arXiv 2604.26629)** — best deconvolution template for
  our window (done in SpectroChemPy): 1580/1600 in-ring aromatic C=C,
  **1620–1640 ring-conjugated side-chain C=C**, 1640–1680 isolated C=C. **Our
  1617 sits in the partially-aromatized regime — char/tar-ball-like, not fully
  condensed soot** — a testable prediction.
- **Yazdani et al. 2021 (*ACP* 21, 10273)** — fresh wood smoke shows 1600 **plus
  a sharp 1515 lignin ring** band; coal shows 1610 **without** 1515.
  **1515-vs-1600 is our fresh-wood vs char discriminator.**
- **Tóth et al. 2018 (*ACP* 18, 10407)** — lab tar balls: strong **~1605
  aromatic C=C** and ~1700 C=O, with C=C growing relative to C=O versus wood
  tar; and the broad 3400–2400 carboxyl O–H envelope is **absent** — a built-in
  discriminator against carboxylate origin.
- **Shankar et al. 2022 (*MAPAN* 37, 529)** — Old Delhi PM₂.₅ ATR-FTIR assigns
  **1606 to aromatic C=C**. Different technique, but a precedent that ~1600–1620
  is seen and read as aromatic in Delhi aerosol.

## Papers — Ethiopian context

- **Tefera et al. 2021 (*IJERPH* 18, 11608)** — CMB in central Addis: **46% of
  OC from biomass burning** (22–74%), levoglucosan to 2234 ng m⁻³ in June–July.
  Gives a **seasonal shape (wet-season maximum) to test the 1617 band against.**
- **Tefera et al. 2020 (*IJERPH* 17, 6998)** — Addis PM₂.₅ 53.8 µg m⁻³, **EC
  25.4%** — unusually high, consistent with hydrogen-poor char-rich carbon.
- **Shlipak et al. 2026 (*ACS ES&T Air*)** — **a caution we must engage**: their
  AAE-based apportionment reportedly assigns ~94% of Addis BC to fossil fuel and
  only ~6% to biomass. Defensible reconciliation: theirs is an optical,
  BC-*mass* apportionment, which does not preclude a hydrogen-poor char
  *absorber* riding on the organic fraction — but say so explicitly.
- **Moschos et al. 2024 (*ES&T* 58(9))** — 11 sub-Saharan fuels incl. **wanza
  (*Cordia africana*)**; BrC dominated by lignin pyrolysis products.
- **Smith et al. 2020 (*ACP* 20, 10149)** — East African domestic fuel optics
  (eucalyptus, olive, acacia); **no FTIR** — the spectroscopic characterization
  of East African fuel aerosol has not been done.

**Net gap:** no FTIR/Raman/functional-group characterization of Addis Ababa or
any East African urban ambient aerosol appears to exist, and no published
PTFE-transmission FTIR comparison of megacities of this class — every network
FTIR paper is US-only (Reggente et al. 2019 explicitly dropped its one non-US
site). We have no "normal" to check against; we would be building it.

## Suggested order of operations

1. ~~Check the APRLssb Segment-2 anchor~~ — **done, confirmed suppressive.**
2. Extend the baseline below 1500 cm⁻¹ (`pspline_arpls` + PTFE masking) so
   900–700 cm⁻¹ becomes usable; validate against the APRLssb port above 1500.
3. Compute **Li et al.'s 1600/750 ratio** across all five sites vs the Zenodo
   charcoal/red-oak/kerosene means; add the **1515 lignin** and **3050/3100
   aromatic C–H** checks.
4. Rule out oxalate (IC data, ~1400 partner, 3400–2400 envelope).
5. Regress the 1617 band on **char-EC/soot-EC** and on the Tefera seasonal cycle.
6. *Then* deconvolve (lmfit, bounded centers, tied widths) and run augmented
   MCR-ALS across sites.

**Verify before citing**: Han 633 nm MAC values (paywalled *Fuel* 2023);
Shlipak et al. 2026 apportionment split. Chakrabarty et al. 2023 DOI is
`10.1038/s41561-023-01237-9`.
