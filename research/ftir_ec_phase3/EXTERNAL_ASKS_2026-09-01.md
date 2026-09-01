# External asks — 2026-09-01

Everything on this list needs someone outside this repo (Davis lab, SPARTAN, a
collaborator, or a field team). Each entry says what we need, who plausibly has it,
which analysis is waiting on it, and what changes when it arrives. In-house follow-ups
that need no one else are tracked in `PHASE3_SUMMARY.md` and the notebook tl;drs.

## A. Adama Batch 54 (blocks ftir_41 / ftir_44 interpretation)

1. **Sampler logs for July 9 and July 30, 2024.** July 9: quartz start is 39.7 min after
   the PTFE start. July 30: PTFE sampled volume is 1.16 m³ vs 2.55 m³ quartz (0.46×).
   Ask the field team (Christian / Sina's Adama contacts) whether these are logged
   restarts, flow faults, or clock errors. *Until answered, both pairs stay flagged;
   neither is excluded or corrected.* July 9 is also the one pair where deployed FTIR EC
   reads 1.9× EC_TOT.
2. **Quartz OC sampling-artifact treatment for Batch 54.** Were backup quartz filters or
   denuders run, and was any positive-adsorption correction applied before the OCTR/OCTT
   values were exported? Ask the Davis carbon lab. *This single fact decides whether
   ftir_41's OC_ftir/OC_TOR ≈ 0.4–0.6 is FTIR under-recovery or quartz over-recovery.*
3. **Confirmation of the spectra export's id → filter map.** The AMOD spectra file keys
   rows by `SampleAnalysisId` 4744–4748 with no FilterId column; ftir_44 infers the map
   from FilterId ordering (validated by CH-band vs OC-loading rank, ρ = 1.00). Ask Davis
   to confirm, or re-export with FilterId. *A wrong map would silently scramble ftir_44.*
4. **Gravimetric PM2.5 for the Adama PTFE filters**, if the AMOD deployment weighed them.
   *Unblocks the Fabs/PM and OC/PM ratio panels the Adama summary deck marks as "not
   measurable."*

## B. ETBI / Bishoftu (blocks the wet-season replication)

5. **Status of all 21 unlinked ETBI PM2.5 catalog records** — ETBI-0049-1 (Feb 25),
   ETBI-0050-2 (Feb 27), 12 from March, 7 from May 2026. The request must distinguish
   "awaiting measurement", "measured but not exported", and "excluded (reason)". Ask
   Mona / the SPARTAN lab. *A true wet-season replication of the near-zero Bishoftu
   intercept is blocked on these.*
6. **HIPS on the 22 already-scanned ETBI filters** and **FTIR scans of the Belg-season
   batch** (from the Adama summary deck's cheap-asks slide; still open).

## C. Optical anchors (would arbitrate the HIPS-side hypotheses)

7. **UV–Vis transmittance/reflectance spectra for ETAD and ETBI filters** — exact-filter
   coverage, blank references, analysis dates, QC. Ren et al. (2025) describe UV–Vis as
   part of the SPARTAN protocol for 2019–2023; availability for later ETBI filters is not
   established. First comparison is HIPS vs UV–Vis at the same wavelength, before any
   BC conversion. Ask SPARTAN (Davis side).
8. **Kirchstetter-style solvent extraction + HIPS re-measurement** on archived Addis and
   Delhi filters, with Beijing/Pasadena controls (from OFFSET_ADJUDICATION_2026-08-23).
   Ask Ann whether the archive and the instrument time exist.

## D. The decisive measurement

9. **Co-located quartz TOR campaign at ETAD (or ETBI)** — ~36 filters, 11–13 days per
   season × 3 seasons, per `quartz_tor_campaign_onepager.md` and ftir_40. Every
   remaining fork (MAC 6 vs 10, HIPS artifact vs non-EC absorber, FTIR OC recovery)
   terminates at this. ftir_41 now adds a design input: request **both** TOR and TOT
   reporting with the full fraction set (EC1–3, OC1–4, OPTR/OPTT), because the
   convention moves EC by ~19% at Adama and ~46% across IMPROVE (ftir_42).

## E. Data we do not have locally (gates two brief directions)

10. **IMPROVE ions and XRF beyond sulfate/Fe/S/Si** for the mirror filters (nitrate,
    ammonium, Al, Si, Ca, Ti, K, Cl…) plus gravimetric mass — needed for the chemical
    mass-closure study. Extend the `pull_results.ps1` mirror on the VPN machine (in-house
    once VPN access is available) or request an export.
11. **PurpleAir sensor 93783 deployment location** and whether it is co-located with the
    BAM whose file is filed under "Jacros BAM" but names its site *Addis Ababa Central*.
    Ask Kyan Shlipak / the JPL side. *Without this, the composition-dependent sensor
    error study cannot start.*
12. **Emily Lee's charcoal FTIR spectra** (from the August Ann 1:1) — still the only route
    to a lab reference for the 1617 band beyond ATR attempts.

## F. Access (in-house once granted)

13. **Networks_1_0 / SPARTAN database read access** is granted (2026-08-22); the
    **PMF input inventory** for the Addis PMF (which optical variables, if any, went in)
    is needed before the continuous-contribution absorption model — ask Ryan / the PMF
    authors.
