# Bridge: SPARTAN public dataset ↔ HIPS (Drive folder)

Source: Google Drive folder `1YVmkYP_0pzs5TQwwbcTQi7gEJ-rTl9LZ` (3 files):
- `SPARTAN_HIPS_Batch1-51.csv` (with row index) — 3,963 rows
- `SPARTAN_HIPS_Batch1-51.v2.csv` (no row index) — same content, cleaner
- `SPARTAN_Site_quick_lookup.xlsx` — 37 sites with canonical lat/lon

Mirrored to `data/drive_bridge/Spartan/` (gitignored).

## HIPS in numbers

- **3,178 sample filters** + **916 blanks** (field-replicate `-7` and lab `*-LB`)
- **27 sites** (subset of public 37); top by filter count:
  ETAD 246, ZAJB 214, TWTA 205, ZAPR 202, CHTS 188, INDH 188, CLST 171,
  ILHA 174, TWKA 158, USPA 158, ILNZ 154, KRUL 122, AUMN 106, AEAZ 100,
  PRFJ 98 …
- Date range: **2022-04-26 → 2026-02-21** — much newer than the public network as a whole.
- Median MDL ≈ **1.6 Mm⁻¹**; the fraction of filters above MDL is 75-100% at most
  sites, but drops to **20-25% at PRFJ (Fajardo)** — a remote-Atlantic site
  with very low absorption.

## The key bridge finding

For every site with ≥10 filters in both datasets:

| site | n | slope (Fabs / BC) | intercept | R² |
|---|---:|---:|---:|---:|
| ETAD | 232 | 9.99 | 0.16 | 0.999 |
| ZAJB | 202 | 9.97 | 0.04 | 1.000 |
| TWTA | 193 | 10.00 | 0.01 | 1.000 |
| ZAPR | 190 | 10.04 | -0.05 | 1.000 |
| CHTS | 175 | 10.01 | 0.01 | 0.999 |
| CLST | 163 | 10.00 | 0.01 | 1.000 |
| … | … | ~10.0 | ~0 | ≥0.99 |

The slope is **10 m²/g at every single site, intercept zero, R² = 1.0**. That is
not a coincidence — the public "BC PM2.5 (μg/m³)" is computed from HIPS as
**Fabs / MAC** with **MAC = 10 m²/g** for every SPARTAN site. The two datasets
are not independent; HIPS is the source, public BC is the derived product.

Implications:
- Don't use public BC as a *validation* for HIPS Fabs. Use the more comprehensive
  HIPS file (raw Fabs, MDL, uncertainty, T1/R1, filter-specific Tau) when
  building absorption-based products.
- For network-wide consistency, the 10 m²/g MAC is a baked-in assumption —
  flag it as a sensitivity in any downstream paper.

Two slopes don't fit cleanly:
- **KRUL** R² = 0.837 (some filters with elevated Fabs vs BC — could be a units
  mismatch in a subset, or HIPS reanalysis after the public file was frozen)
- **CAHA** can't be fit due to all-zero / near-zero values

## Coverage overlap

| presence | filters |
|---|---:|
| **both** (in HIPS and public) | 2,522 |
| **right_only** (public only) | 656 |
| **left_only** (HIPS only) | 1,640 |

- The 1,640 HIPS-only filters are mostly **post-2024 measurements** not yet
  rolled into the public release (note SPARTAN public CSVs are versioned at
  most monthly).
- The 656 public-only filters are mostly **pre-2022 measurements** from
  before the HIPS workflow started.

## New sites found in the Drive lookup

Three sites appear in the Drive lookup but are not in the public FTP yet:

- **ETBI** (Bishoftu, Ethiopia — 8.76 °N, 39.00 °E): **26 real sample filters
  from 2025-10 onward** + 4 field blanks. Sister site to ETAD, just commissioned.
- **USSL** (St Louis, MO — 38.65 °N, -90.31 °W): **42 real sample filters
  from 2025-08 onward** + 6 field blanks.
- **USBA** (Baltimore, MD — 39.26 °N, -76.71 °W): in the lookup only; no data
  in either HIPS or public FTP yet.

The Drive lookup is missing **ARCB, USBO, USMC** — these are present in the
public FTP but are likely legacy or decommissioned by the time the lookup
was last updated.

## Outputs

Tables (`research/spartan/inventory/`):
- `hips_coverage_by_site.csv` — per-site filter counts, date range, Fabs
  stats, MDL, % above MDL
- `hips_vs_public_link.csv` — full FilterId join (HIPS ∪ public): site,
  base FilterId, HIPS Fabs, public BC, public PM2.5 mass, presence
- `hips_vs_bc_linear_fits.csv` — per-site OLS slope/intercept/R² of Fabs vs BC

Figures (`research/spartan/inventory/figures/`):
- `hips_coverage_by_site.png` — horizontal bar of filters per site with
  span and % above MDL annotated
- `hips_monthly_heatmap.png` — site × month HIPS filter counts
- `hips_vs_bc_scatter.png` — Fabs vs BC scatter for the 6 sites with the
  most overlap (visually confirms the 10× slope)
