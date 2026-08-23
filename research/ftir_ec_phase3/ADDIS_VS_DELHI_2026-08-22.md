# Addis vs Delhi — what we can compare today, and what needs the spectra pull

Motivated by the dust/Fe test finding that **Delhi carries a Fabs-vs-EC offset
as large as Addis's** (+35.3 vs +28.4 Mm⁻¹), while Bishoftu and Pasadena carry
none. If two heavily-loaded megacities share the anomaly, comparing them
directly is the fastest route to its cause.

## Available now: FTIR functional groups (no raw spectra needed)

`unified_filter_dataset.pkl` carries the deployed FTIR functional-group products
for all four sites. Normalizing to aliphatic CH removes loading and leaves
composition **shape**:

| ratio to alkaneCH | **ETAD Addis** | INDH Delhi | CHTS Beijing | USPA Pasadena |
|---|---|---|---|---|
| EC_ftir | **1.785** | 0.530 | 0.270 | 0.508 |
| OC_ftir | 2.322 | 1.131 | 0.761 | 2.047 |
| alcoholCOH | **0.683** | 0.276 | 0.286 | 0.639 |
| carboxylicCOOH | **0.402** | 0.523 | 0.609 | 0.372 |
| naCO (non-acid carbonyl) | **0.151** | 0.080 | 0.054 | 0.281 |

Medians (µg/m³): Addis EC 4.63 / OC 6.02 / OM 5.87; Delhi EC 3.28 / OC 6.99 /
OM 13.10.

### Two findings

**1. Addis is far more EC-rich per unit organic than Delhi — 3.4×.** EC/alkaneCH
is 1.79 at Addis vs 0.53 at Delhi and 0.27 at Beijing. Delhi's absorption
anomaly therefore arises in a *chemically very different* aerosol: Delhi is
organic-rich with moderate EC, Addis is soot-dominated with sparse organics.
A shared mechanism that is *chemical* looks unlikely; a shared mechanism that is
*instrumental and loading-driven* fits both.

**2. Addis OM/OC ≈ 0.98 — physically impossible.** OM (functional-group sum)
should exceed OC by ~1.4–2.2×; Delhi gives 1.87, Beijing 2.73, Pasadena 1.33.
Addis at 0.98 means **the measured functional groups do not account for the
carbon the same spectra imply**. This is ftir_17's "Addis is a deficit, not
exotic" appearing in a second, independent form. Either the Addis spectra are
missing functional-group absorption (baseline/background problem — which would
tie directly to why AIRSpec correction helps Addis so much) or the aerosol is
genuinely so reduced/char-like that conventional FG accounting fails on it.
Either way it is a strong, previously unquantified anomaly and it is
**Addis-specific among the four sites**.

## Needs the spectra pull: the actual band-by-band comparison

The functional-group products are *model outputs*. The raw spectra answer
questions they cannot:

- Does Addis's ~1617 cm⁻¹ band (identified in ftir_12; not amine) appear at
  Delhi? If yes it is a combustion signature; if no it is Addis-specific.
- Is the Addis baseline/background elevated relative to Delhi's (the mechanism
  behind AIRSpec's outsized benefit at Addis)?
- Do the Delhi spectra sit inside the IMPROVE calibration's score-space domain,
  or are they extrapolation like Addis? (This is the long-blocked ftir_14 work.)
- Does the ocec-450 calibration transfer to Delhi — a *third* independent site,
  chemically unlike both Addis and Bishoftu?

### How to get them (one command, Windows machine on the UCD VPN)

Paste `research/ftir_ec_phase3/scripts/get_spartan_spectra.ps1`, then:

```powershell
Get-SpartanSite "INDH"      # Delhi
Get-SpartanSite "CHTS"      # Beijing, for the third point
```

Then on this machine:

```bash
python research/ftir_ec_phase3/scripts/build_spartan_target.py INDH --name delhi
python research/ftir_ec_phase3/scripts/build_spartan_target.py CHTS --name beijing
```

Both appear in the explorer's "Evaluate on" dropdown, and the Sites tab
evaluates every calibration against all of them at once.

Note the script's `Show-SpartanSites` (runs automatically on paste) prints
per-site FTIR-scanned counts — worth reading before the pull, since the deployed
products exist for 120 INDH filters but the number with archived raw scans in
`ftir.Scan` may differ.
