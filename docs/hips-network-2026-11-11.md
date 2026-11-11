# HIPS across the SPARTAN network: where it under-reads, why, and what we can do with it

Date: 11 November 2026. Follows [addis-reference-closure-2026-11-10.md](addis-reference-closure-2026-11-10.md). The numbers are in
[`hips-network-2026-11-11.json`](hips-network-2026-11-11.json); the charts are in the gallery's Meeting tab, Nov 11. No
aethalometer data are used. FTIR EC is FTIR-derived, HIPS is optical, and no SPARTAN filter here has thermal EC.

## Summary

| Question | Answer |
|---|---|
| Does the HIPS bend apply across the network? | **Wherever filters are dark.** 12 of 30 sites lose more than a fifth on their median filter; 13 lose under a tenth. Of 3,733 filters, 59 % lose more than a tenth, 31 % more than a fifth, 16 % more than three tenths. |
| Is there a filter group we can identify? | **Yes, by optical depth.** Above 0.45 a filter loses more than a fifth; above 0.68, more than three tenths. That group is 594 filters at 17 sites, 235 of them Addis. |
| Is it seasonal? | **At most heavy sites.** Santiago, Pretoria, Johannesburg and Delhi lose at least twice as much in their polluted season as in their cleanest month. Addis does not come down: 38 % in its best month. |
| Does Delhi get affected? | **Yes, about half as much as Addis.** Its median filter loses 21 % [17 %, 26 %], and about 29 % from October to January. Its own exponent, 0.80 [0.74, 0.85], matches the prediction, 0.78. |
| Is it a volume issue? | **Volume is the lever, not the cause of most differences.** 26 of 30 sites sample about 7.2 m³, so between sites the air decides. Delhi and Jaipur have air as heavy as Addis's but sample half the volume or less. At 7.2 m³ Delhi would lose 42 %, the same as Addis. |
| Does each site's own data show it? | **In direction, not site by site.** 12 of 23 sites can tell: 6 match the curve, 3 are more compressed, 3 less. |
| What can we do? | Carry the optical depth with every HIPS value. Use HIPS as a yardstick for FTIR EC only on filters that lose a fifth or less, linearised (69 % of the network, no Addis filter). Ask for a lower sampled volume at Addis. |

## The rule applied

On IMPROVE filters, against thermal EC, HIPS optical depth follows `tau = L / (1 + L / ceiling)` with a ceiling of
2.26 [1.82, 2.70]. A filter whose optical depth is tau reports a share `tau / ceiling` less absorption than
it carries. The optical depth here is `Fabs x sampled volume / 353`, which equals the laboratory's own HIPS tau on all 546
filters where both are on file.

## 1. By site

| Site | Filters | Fabs reported (Mm⁻¹) | linearised | Volume (m³) | Optical depth | Under-read | Filters over a fifth | over three tenths |
|---|---|---|---|---|---|---|---|---|
| Addis Ababa, Ethiopia | 247 | 47 | 81 | 7.2 | 0.96 | 42 % [35 %, 52 %] | 100 % | 95 % |
| Kanpur, India | 12 | 41 | 63 | 6.9 | 0.79 | 35 % [29 %, 43 %] | 92 % | 50 % |
| Dhaka, Bangladesh | 58 | 39 | 57 | 7.2 | 0.73 | 33 % [27 %, 40 %] | 98 % | 74 % |
| Ilorin, Nigeria | 29 | 34 | 49 | 7.2 | 0.69 | 31 % [26 %, 38 %] | 72 % | 52 % |
| Bujumbura, Burundi | 56 | 35 | 50 | 7.3 | 0.66 | 29 % [25 %, 36 %] | 86 % | 45 % |
| Bandung, Indonesia | 61 | 31 | 43 | 7.2 | 0.62 | 27 % [23 %, 34 %] | 82 % | 30 % |
| Jaipur, India | 85 | 53 | 71 | 3.6 | 0.58 | 26 % [21 %, 32 %] | 71 % | 35 % |
| Singapore, Singapore | 12 | 28 | 37 | 7.2 | 0.56 | 25 % [21 %, 31 %] | 83 % | 0 % |
| Abu Dhabi, UAE | 153 | 28 | 36 | 7.2 | 0.52 | 23 % [19 %, 29 %] | 63 % | 18 % |
| Johannesburg, South Africa | 214 | 24 | 30 | 7.2 | 0.48 | 21 % [18 %, 26 %] | 54 % | 16 % |
| Delhi, India | 170 | 59 | 75 | 3.5 | 0.47 | 21 % [17 %, 26 %] | 54 % | 22 % |
| Bishoftu, Ethiopia | 60 | 24 | 30 | 7.2 | 0.47 | 21 % [17 %, 26 %] | 52 % | 7 % |
| Mexico City, Mexico | 79 | 20 | 25 | 7.3 | 0.42 | 19 % [16 %, 23 %] | 43 % | 4 % |
| Santiago, Chile | 195 | 21 | 26 | 7.1 | 0.42 | 18 % [15 %, 23 %] | 46 % | 26 % |
| Pretoria, South Africa | 280 | 20 | 25 | 7.2 | 0.41 | 18 % [15 %, 23 %] | 44 % | 19 % |

The other fifteen: Kaohsiung 12 %, Beijing 11 %, Seoul 10 %, Rehovot 9 %, Talca 9 %, St Louis 8 %, Taipei 7 %, Melbourne 6 %, Ulsan 6 %, Haifa 6 %, Sherbrooke 5 %, Pasadena 4 %, Halifax 4 %, Norman 3 %, Fajardo 1 %.

Reported Fabs mis-ranks the heavy sites. Delhi reports 59 Mm⁻¹ against Addis's 47, but once linearised they are 75 and 81.

## 2. Air or volume

The same air replayed at each of SPARTAN's three volumes (each filter's loading taken back through the curve, rescaled, and
read forward again):

| Site | Volume now (m³) | As sampled | at 7.2 m³ | at 3.6 m³ | at 1.8 m³ |
|---|---|---|---|---|---|
| Addis Ababa, Ethiopia | 7.2 | 42 % | 42 % | 27 % | 15 % |
| Delhi, India | 3.5 | 21 % | 42 % | 27 % | 15 % |
| Jaipur, India | 3.6 | 26 % | 41 % | 25 % | 15 % |
| Dhaka, Bangladesh | 7.2 | 33 % | 35 % | 21 % | 12 % |
| Bujumbura, Burundi | 7.3 | 29 % | 31 % | 18 % | 10 % |
| Ilorin, Nigeria | 7.2 | 31 % | 31 % | 18 % | 10 % |
| Bandung, Indonesia | 7.2 | 27 % | 28 % | 16 % | 9 % |
| Johannesburg, South Africa | 7.2 | 21 % | 21 % | 12 % | 6 % |
| Santiago, Chile | 7.1 | 18 % | 19 % | 11 % | 6 % |

- **Delhi's volume was cut twice:** 7 m³ in 2022–23, 3.6 m³ from March 2024, 1.8 m³ from August 2025. Addis has stayed at
  7.2 m³ since 2022. Dhaka and Bujumbura were cut in 2023–24.
- **Delhi's winter filters at 1.8 m³ lose 28 %.** The same air at 7.2 m³ would lose 61 %.
- **Long samples at clean sites carry more.** Norman's 44 m³ filters lose 11 % against 3 % at 7 m³; Melbourne's 29 m³ filters
  13 % against 4 %.

A check that uses no FTIR: in the same calendar months, Delhi filters sampled at 1.8 m³ against filters sampled at 3.6 m³,
by HIPS Fabs per unit of weighed PM2.5.

| Months | Filters at 1.8 m³ | at 3.6 m³ | Observed ratio | Predicted |
|---|---|---|---|---|
| March | 9 | 7 | 1.20 [1.06, 1.41] | 1.19 |
| April | 12 | 8 | 1.26 [1.04, 1.45] | 1.13 |
| these months together | 21 | 15 | 1.29 [1.09, 1.42] | 1.17 |

The 1.8 m³ filters are from 2026 and the 3.6 m³ filters from 2024, so a change in the aerosol between those years would
look the same.

Delhi's 2022–23 filters at 7 m³ are not evidence either way. They read low on HIPS and on FTIR EC alike (Fabs 8 to 56 Mm⁻¹
and FTIR EC 1 to 8 µg/m³ in January to March 2023), so they do not separate the two instruments.

## 3. Season

| Site | Best month | Worst month |
|---|---|---|
| Addis Ababa, Ethiopia | Jan 38 % | Jun 56 % |
| Delhi, India | Jul 3 % | Oct 31 % |
| Jaipur, India | Jun 15 % | Dec 52 % |
| Santiago, Chile | Dec 11 % | Jul 37 % |
| Pretoria, South Africa | Jan 10 % | Jul 27 % |
| Johannesburg, South Africa | Dec 11 % | Jun 28 % |
| Bujumbura, Burundi | Dec 24 % | Aug 47 % |
| Mexico City, Mexico | Sep 13 % | Feb 29 % |
| Abu Dhabi, UAE | Jan 19 % | Jun 29 % |

Months pool years and, at Delhi, Jaipur, Dhaka and Bujumbura, different volumes. Jaipur's December includes filters sampled
at 6.6 m³.

## 4. The affected filters, and how well the correction is known

| Expected under-read | Optical depth | Filters | Share | Factor | ± | Between IMPROVE half-periods | Mostly |
|---|---|---|---|---|---|---|---|
| under 5 % | 0.00 to 0.11 | 738 | 20 % | 1.03 | ±1 % | 1 % | Pasadena, United States (141), Haifa, Israel (82), Fajardo, Puerto Rico (82), Halifax, Canada (78) |
| 5 to 10 % | 0.11 to 0.23 | 801 | 21 % | 1.08 | ±2 % | 3 % | Taipei, Taiwan (126), Rehovot, Israel (103), Ulsan, Korea (83), Pasadena, United States (71) |
| 10 to 20 % | 0.23 to 0.45 | 1,021 | 27 % | 1.17 | ±3 % | 7 % | Pretoria, South Africa (112), Kaohsiung, Taiwan (102), Beijing, China (98), Santiago, Chile (87) |
| 20 to 30 % | 0.45 to 0.68 | 579 | 16 % | 1.33 | ±7 % | 15 % | Johannesburg, South Africa (80), Abu Dhabi, UAE (69), Pretoria, South Africa (69), Delhi, India (54) |
| over 30 % | 0.68 to 2.26 | 594 | 16 % | 1.60 | ±13 % | 30 % | Addis Ababa, Ethiopia (235), Pretoria, South Africa (53), Santiago, Chile (51), Dhaka, Bangladesh (43) |

± is half the spread of the correction factor across the ceiling's interval. The last-but-one column is the difference between
the ceilings of IMPROVE 2005–2014 (2.53) and 2015–2025 (1.71). IMPROVE's own strata reach an optical depth of
1.18; 12 % of the filters in the last band are darker than that, 18 % of Addis filters.

## 5. Each site's own evidence

The exponent b in `HIPS ∝ (FTIR EC)^b`, with weighed PM2.5 as the instrument, beside what the curve predicts from the site's
optical depths. Twelve of 23 sites have an interval narrower than 0.6.

| Site | Filters | Optical depth | Predicted | Observed | Reading |
|---|---|---|---|---|---|
| Addis Ababa, Ethiopia | 246 | 0.95 | 0.57 | 0.46 [0.43, 0.50] | more compressed than predicted |
| Dhaka, Bangladesh | 58 | 0.73 | 0.62 | 0.69 [0.56, 0.89] | as predicted |
| Johannesburg, South Africa | 212 | 0.48 | 0.78 | 0.88 [0.80, 0.97] | less compressed than predicted |
| Delhi, India | 133 | 0.47 | 0.78 | 0.80 [0.74, 0.85] | as predicted |
| Pretoria, South Africa | 264 | 0.43 | 0.80 | 0.90 [0.81, 0.99] | less compressed than predicted |
| Santiago, Chile | 191 | 0.41 | 0.78 | 0.56 [0.53, 0.59] | more compressed than predicted |
| Kaohsiung, Taiwan | 160 | 0.28 | 0.88 | 0.83 [0.74, 0.92] | as predicted |
| Beijing, China | 192 | 0.25 | 0.87 | 0.93 [0.81, 1.14] | as predicted |
| Melbourne, Australia | 73 | 0.20 | 0.91 | 0.80 [0.61, 1.07] | as predicted |
| Talca, Chile | 93 | 0.20 | 0.88 | 0.56 [0.52, 0.59] | more compressed than predicted |
| Taipei, Taiwan | 260 | 0.16 | 0.92 | 1.06 [0.95, 1.19] | less compressed than predicted |
| Pasadena, United States | 210 | 0.10 | 0.96 | 1.04 [0.83, 1.31] | as predicted |

Observed follows predicted with a correlation of 0.65 across the twelve. The eleven that cannot
tell are mostly light sites, and dust or sulfate sites where EC is a small share of PM2.5.

## 6. What to do

| Option | For | Against | Where it stands |
|---|---|---|---|
| Leave HIPS as reported | Fine at the 13 sites that lose under a tenth | 59 % of filters read more than a tenth low; heavy sites are mis-ranked | Not for heavy sites |
| Carry the optical depth with every HIPS value | Costs nothing; it is the laboratory's own number | Corrects nothing by itself | Do it |
| Linearise filters that lose up to a fifth; flag the rest | 69 % of filters; correction known to ±3 % | Leaves out every Addis filter and 54 % of Delhi's | Use for comparisons with FTIR EC |
| Linearise every filter with one ceiling | Removes the bend on average | ±13 % above three tenths; 6 of 12 sites match; no thermal EC to check | A stated sensitivity, not a product |
| Sample less air at the heavy sites | Addis falls from 42 % to 27 % at 3.6 m³ and 15 % at 1.8 m³ | New filters only; less mass for other analyses; a break in the record | Ask SPARTAN |
| Fit each site its own curve | Would absorb real differences between sites | Needs a reference that is not FTIR EC | Not with the data on file |

For Addis this means HIPS cannot set the FTIR EC level on the filters on file, because none loses a fifth or less. It still
shows that the published slope and intercept come from the bend. For Delhi, 46 % of filters qualify, nine in ten of them
outside October to January.

## Limits

- The curve is IMPROVE's. The same laboratory and filter type measure both networks, but SPARTAN has no thermal EC to test it.
- The exponent test uses FTIR EC, which is not thermal EC, and cannot separate a bend in HIPS from one in FTIR EC.
- Santiago and Talca are far more compressed than predicted, and Johannesburg and Pretoria less. Neither is explained.
- The volume replay assumes the curve holds at each volume. The Delhi comparison is 36 filters across two years.
- Sampled volume is the recorded value. A sampler whose flow fell on a heavy filter would have drawn less than recorded.
- Site medians pool every year on file; 7 sites have fewer than 60 filters.

## Reproduce

```bash
python research/ftir_hips_chem/workflows/run_hips_network_20261111.py --summary
python gallery/data/export_meeting_20261111.py
python -m pytest -q tests/test_hips_network.py
```

Both read committed files only: `gallery/app/public/data/spartan_reference.json` and the 10 Nov summary.
