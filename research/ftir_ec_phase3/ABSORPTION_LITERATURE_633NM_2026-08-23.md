# Literature: non-EC absorption at 633 nm — Delhi, Addis, and the artifact counter-case

Agent-assisted survey, 2026-08-23. Question: can real non-EC absorbers
(BrC/char/tar balls) explain a ~10–25 Mm⁻¹ excess at 633 nm at Addis and
Delhi while being absent at Beijing/Pasadena/Bishoftu — or is a filter
artifact more likely? Companion to `OFFSET_ADJUDICATION_2026-08-23.md`.

## Delhi

- Winter Delhi total b_abs(880) ≈ 190 ± 95 Mm⁻¹ (aethalometer winter study,
  [Atmos. Environ.](https://www.sciencedirect.com/science/article/abs/pii/S1352231018306277)) —
  a ~20 Mm⁻¹ non-EC residual at 633 nm is ~10% of total absorption.
- Direct BrC spectra to 660 nm: BrC declines 87 Mm⁻¹ (370 nm) → **~12 Mm⁻¹
  (660 nm)**, attributed to *primary* BrC from incomplete combustion
  ([Atmos. Pollut. Res. 2025](https://www.sciencedirect.com/science/article/pii/S1309104225003241)).
  Exactly the needed range.
- Rastogi/Satish (PRL): winter WS-BrC ~13.9, MeOH-BrC ~21.9 Mm⁻¹ at 365 nm,
  >10× summer, substantial water-insoluble darker fraction
  ([STOTEN 2021](https://www.sciencedirect.com/science/article/abs/pii/S0048969721036615)).
  Extracts understate red absorption — tar balls don't extract.
- ¹⁴C (Gustafsson group, Nat. Sustain. 2019): Delhi winter carbonaceous
  aerosol dominated by regional biomass burning — the tar-ball source class.
- Tar balls confirmed by TEM in Delhi haze
  ([ES&T 2025](https://pubs.acs.org/doi/10.1021/acs.est.5c16031)) and IGP
  outflow ([ES&T Lett. 2020](https://pubs.acs.org/doi/abs/10.1021/acs.estlett.0c00735)).
- Dark BrC (Chakrabarty): correlates with tar balls, 50–75% of shortwave
  absorption in some plumes, persists to long wavelengths
  ([Nat. Geosci. 2023](https://www.nature.com/articles/s41561-023-01237-9);
  [One Earth 2025](https://www.cell.com/one-earth/fulltext/S2590-3322(25)00031-4)).

## Addis / charcoal

- Site-specific optical-closure literature is essentially absent (a
  publishable gap). NASA MAIA: BC 4–9× US-metro, nighttime charcoal/biomass a
  major driver ([Reporter Ethiopia](https://www.thereporterethiopia.com/52371/)).
- Sub-Saharan biomass BrC characterized in
  [ES&T 2024](https://pubs.acs.org/doi/10.1021/acs.est.3c09378); charcoal
  cookstove BrC has the *lowest AAE* of the fuel set — flatter, persists into
  red ([Environ. Pollut. 2018](https://www.sciencedirect.com/science/article/abs/pii/S0269749118303476)).
- **Key physics**: lab tar balls MAC 0.8–3.0 m²/g at 550 nm, AAE 2.7–3.4 over
  467–652 nm ([Hoffer et al. ACP 2016](https://acp.copernicus.org/articles/16/239/2016/));
  tar-ball absorption at **880 nm is >10% of its 470 nm value** — "field NIR
  absorption cannot solely be due to soot"
  ([Hoffer et al. AMT 2017](https://amt.copernicus.org/articles/10/2353/2017/)).
  Char is flatter still; Asian biomass BC increasingly char-dominated
  ([Comm. Earth Environ. 2026](https://www.nature.com/articles/s43247-026-03431-0)).
  Classic water-soluble BrC fades by ~600 nm; tar/char does not.

## Why Beijing shows nothing

Beijing red-wavelength absorption is BC-dominated: bare BC AAE ≈ 0.56
(470–660), OC AAE ≈ 2.7 with absorption concentrated in UV-blue
([STOTEN 2020](https://www.sciencedirect.com/science/article/abs/pii/S0048969720361295);
[ACP 2020](https://acp.copernicus.org/articles/20/9701/2020/);
[ACP 2018](https://acp.copernicus.org/articles/18/9061/2018/)). Flaming
coal + traffic produce soot (counted as EC), not smoldering tar balls;
secondary BrC is UV-weighted and photobleaches. Same logic covers Pasadena
(traffic BC) and Bishoftu (light loading).

## The artifact counter-hypothesis

- Organic enhancement of filter photometers is real but **fiber-specific**:
  organics wick into quartz/glass mats (PSAP bias up to ~2× with high OA/BC —
  Lack et al. 2008; Cappa et al. 2008; confirmed
  [AMT 2019](https://amt.copernicus.org/articles/12/3417/2019/)). On PTFE
  (HIPS) particles sit on the surface; the analog is weaker. PTFE-specific
  loading nonlinearity/shadowing documented in
  [White et al. 2016](https://www.tandfonline.com/doi/full/10.1080/02786826.2016.1211615),
  [AMT 2019](https://amt.copernicus.org/articles/12/1365/2019/),
  [JAWMA 2024](https://www.tandfonline.com/doi/full/10.1080/10962247.2024.2442634),
  and an OC-dependent polar-photometer artifact
  ([J. Aerosol Sci. 2013](https://www.sciencedirect.com/science/article/abs/pii/S0021850213002589)).
- **Discriminating fact**: a loading/organic artifact should appear at
  Beijing's OM-rich winter haze. It doesn't. But biomass sites are also
  organic-heavy sites, so artifact and BrC are partially confounded — test
  explicitly.

## Decisive measurements, ranked

1. **Solvent extraction + HIPS re-measurement** (Kirchstetter, Novakov &
   Hobbs [JGR 2004](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2004JD004999)):
   water then methanol on archived Addis + Delhi PTFE filters, Beijing/
   Pasadena as controls. Drop only at the anomalous sites → real BrC/char;
   methanol-resistant dark residue → tar/char specifically. Uses filters we
   already have.
2. Multi-wavelength filter absorption (405/532/633/880) on the same filters:
   site-to-site AAE difference at fixed EC. MA350 caveat: 625/880 two-point
   AAE cannot separate tar balls (AAE ~2) from lensed BC (0.8–1.2), and
   loading corrections dominate at Delhi loadings.
3. Thermal-optical EC on collocated quartz (char pyrolysis EC1 fraction) —
   also resolves the offset-vs-curvature degeneracy in DUST_FE_TEST.
4. TEM tar-ball counts on a few Addis/Delhi filters (Pósfai 2004;
   [PNAS 2019](https://www.pnas.org/doi/10.1073/pnas.1900129116)) — near-zero
   marginal cost, definitive morphology.

## Strongest anchors for the paper

1. Hoffer et al. 2017 (AMT) — red/NIR absorption is not soot-only.
2. Delhi BrC ≈ 12 Mm⁻¹ at 660 nm (APR 2025) — published numbers bracket our
   intercept.
3. Kirchstetter et al. 2004 (JGR) — the extraction test; with Lack/Cappa 2008
   as the artifact hypothesis that must be (and can be) ruled out.
