# SPARTAN field availability: stand-ins and gaps in the gallery

Why some SPARTAN sites show no EC (FTIR), HIPS BC or Silicon, what the gallery
shows instead, and the documentation behind each rule. The rules live in
`gallery/app/src/lib/fieldEquivalents.ts` (tested in
`fieldEquivalents.test.mjs`); this page is their record. Audit of 2026-10-28.

## What SPARTAN documents

| Fact | Source |
|---|---|
| HIPS BC (parameter 28202) and FTIR EC/OC (28203/28204) are defined only for the AirPhoton SS5i sampler with **stretched** Teflon filters (methods 217/218). | [Sampling Parameters & Methods v2.3](https://www.spartan-network.org/_files/ugd/6321a4_219864c4f1454c58a956339bc58b9120.xlsx?dn=Sampling_Parameters_Methods_2.3.xlsx), sheet "PM2.5 BC_OC_EC" |
| Equivalent BC by smoke-stain reflectometer (28201) covers every sampler, including the older SS4i and mesh Teflon (methods 212–216). | same sheet |
| The reflectometer's equivalent BC uses σ = 0.1 cm²/µg ≡ 10 m²/g, the MAC SPARTAN uses for HIPS. | [Black Carbon SOP rev. 2.0, 23 Oct 2019](https://wustl.box.com/shared/static/nnrzksuty2v38rlssq6b3wil7nxwd9p5.pdf), §2.0; [filter-optics-reference.md](filter-optics-reference.md) §2a |
| Public BC from HIPS is Fabs / 10 (methods 219/221); method 220 is SPARTAN's estimate of it from the reflectometer through a HIPS–SSR curve. | [spartan-network-data-pulls.md](spartan-network-data-pulls.md) |
| Silicon (and the other XRF-only elements) is measured only by ED-XRF (methods 515/516), applied to filters from 11/2018; the ICP-MS before it (methods 412–416) reports no silicon. | Parameters & Methods v2.3, sheet "PM2.5 trace elements"; [XRF SOP rev. 2.3, 5 Dec 2025](https://www.spartan-network.org/_files/ugd/6321a4_28af428d5f204d53aecc5f047b6d7b37.pdf), revision history |
| FTIR EC/OC are calibrated against IMPROVE thermal-optical (TOR) carbon. | Debus et al. 2022, *AMT* 15, 2685, [doi:10.5194/amt-15-2685-2022](https://doi.org/10.5194/amt-15-2685-2022) |

SPARTAN's known-issues document covers only nitrate volatilisation; it says
nothing about these gaps.

## Stand-ins (applied automatically, labelled on the tile)

| Asked for | Stand-ins, in order | Basis |
|---|---|---|
| HIPS BC | BC (public HIPS) → BC (public SSR estimate) → Equivalent BC (SSR) | same HIPS absorption; then SPARTAN's HIPS-calibrated SSR estimate; then the reflectometer's eBC at the same 10 m²/g |
| EC (FTIR) | EC (ChemSpec FTIR) → EC (FTIR, VIBES) → EC (FTIR, spline) → EC (TOR) | the public FTIR product; the same spectra under another baseline; IMPROVE's TOR, the FTIR calibration reference |
| OC (FTIR) | OC (ChemSpec FTIR) → OC (TOR) | as above |

**No stand-in, by design:** EC at a reflectometer-only site (eBC is an optical
measurement, not EC), Silicon before XRF, and a site's public chemistry before
SPARTAN releases it. Those tiles say why, link the source, and offer a field the
site has (one click).

## Where it applies

- Dashboard starter tiles and new tiles resolve their fields per site in view.
- A tile whose field the sites never measured explains why instead of drawing
  an empty chart; one emptied only by the date, season or exclusion filters
  says that instead.
- Site cards show a site's stand-in value, marked with its method.
- The coverage tile's "not measured" tooltip gives the documented reason;
  reflectometer fields are their own group, **Optical (SSR)**, so a site with
  only those reads "Optical (HIPS) not measured".
- The sidebar's crossplot axes move from HIPS BC to the first stand-in a newly
  chosen site has (previously only to public HIPS BC).
- The composition legend lists only species with values in view.
- Sites with no filters in the public release (CODC, USBA) are greyed and listed
  last in the site picker.

## Per-site outcome

| Sites | Before | Now |
|---|---|---|
| ARCB, CAKE, PHMO, USBO, USMC (2014–2019, reflectometer only) | every EC tile, the HIPS BC tile and the scatter empty | BC tiles on the SSR estimate or eBC; EC tiles explain (no EC equivalent) and offer PM2.5 mass |
| CADO, CALE, VNHN (some FTIR, no HIPS) | HIPS BC tile and scatter empty | BC on the SSR estimate; scatter SSR BC × EC |
| NGIL, INKA, SGSU (public HIPS only) | HIPS BC tile empty | BC (public HIPS) |
| USSL (2025–26, no public chemistry yet) | almost every tile empty | EC from its own spectra (VIBES baseline); PM2.5 and ions explain they are not in the public release yet |
| CODC, USBA (0 filters) | offered in the picker, everything empty | greyed, not pickable |
| IMPROVE sites in Compare | OC (FTIR) KPI duplicated EC (TOR); EC column empty on cards | EC (TOR) / OC (TOR), labelled |
| The other 29 SPARTAN sites, focus sites | full | unchanged |

Still missing data (not fixable in the gallery): ERA5 weather for USSL
(`export_weather.py`), AERONET for CLTA, ZAJB and USSL (`export_aeronet.py`
station match), USSL's public chemistry (re-run `export_spartan_reference.py`
then `export_spartan_network.py` once SPARTAN publishes it).
