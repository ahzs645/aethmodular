# FTIR group talk — 2026-08-27 — talking points (comprehensive deck)

**One breath:** The deployed calibration reads Addis at 1.90x−4.17; this talk
dissects that line completely — spectra processing (90% of a raw spectrum is
Teflon background; baselining is the lever), cohort selection (composition
wins, shape-matching was matching Teflon until selected in corrected space),
the instrument (the HIPS correction is a blank-line regression; the Addis
intercept survives it at −1.27 ± 0.17 while Pasadena's slope anomaly
dissolves), the cross-site verdict (no calibration choice reconciles Addis
and Delhi; baselining relocates seasonality onto the dry season), and the
independent witnesses (Mie physics, AERONET hardening) — ending at the two
lab measurements that settle the rest: quartz TOR and the extraction test.

## Deck map (30 main + 7 backup)

- **1–3 Framing**: roadmap · status scorecard · the deployed 1.90x−4.17 problem
- **4–5 Spectra basics**: Teflon background · AIRSpec on one real filter
- **6–9 Protocols & matrix**: A/B/B2 ladder · raw grid (A) · corrected grid (A) · OC/EC selection mechanics
- **10–15 Selection science**: analog mechanics · the −9.7→−4.3 fix · 4/477 Teflon proof · Eth-shaped rescue · k-scan (Satoshi's 21) · dense basin + 62.5%
- **16–20 The instrument**: blank-line mechanism · intercept survives · slopes resolve · lot census · lot effect (Ann's hypothesis confirmed)
- **21–23 Five sites**: cross-site spectra + 1617 lesson · the exhaustive grid · seasons
- **24–26 Independent**: Mie/MAC fork · AERONET diurnal · AERONET column
- **27–30 Close**: Adama · live app · the two asks · logistics
- **31–38 Backup**: B/B2 grids · residual-vs-D² · bootstrap CIs · lot-253 takeover · LOCAL v1

## Anchor validation (run live before building)

ocec-800 raw A: k=6, OLS 1.585x−3.221, held-out 0.911 ✓
+AIRSpec: k=5, OLS 0.86x−1.615, Deming 0.95x−2.09 ✓
Option-B historical intercepts on the ladder: −5.76 / −10.16 / −6.74 / −2.17 ✓

## Don't get caught

- "corrected" is not a valid API spectra token (silently falls back to raw) —
  everything here uses `airspec`.
- ~15% is the **blank-line share**, not the full instrument share; high-loading
  nonlinearity is invisible to blanks and needs quartz TOR.
- Screening vs quoting: 71k grid rows are descriptive; the winner takes 62.5%
  of bootstrap draws (corrected analog-440 ~⅓); Delhi's screening "winner"
  was 96% extrapolated and died on holdout — cite as validation working.
- 1617 band = Ethiopian regional marker, decoupled from the offset (Bishoftu
  has it, no offset); AIRSpec anchors at 1520–1600 → two-baseline rule.
- Season fits are per-season Deming on restricted ranges — indicative;
  ftir_24 owes the interaction fit. Season schemes differ across sites.
- Lot-effect n = 14 vs 34, loadings not fully matched (1.18 vs 1.66 µg/m³).
- Adama figure is the seed — TOR-vs-FTIR panel still to add for the 2-pager.
- Standing: analog top-500 → 477 resolved; entire-network A k=15 cell
  unexplained; Eth-shaped+AIRSpec B2 (k=21) unstable.
- H (AERONET) is comparative-only; Level 1.5; envelope partial r=+0.28
  (p≈0.002) is the honest carrier, not the discrete 1620 peak.

## The asks

1. **Quartz TOR** (~36 filters, 3 seasons; one-pager committed) — the only
   measurement that is a function of neither axis.
2. **Extraction + HIPS re-measure** (archived Addis/Delhi, Beijing/Pasadena
   controls; Kirchstetter 2004 design).
3. Lot-253 spectra pull (script staged; Windows/VPN session; lot 255 sealed).

## Personal / calendar

- AAAR: poster Thu 1–3 pm, session 9; booked. [FILL IN: forward acceptance to Ann]
- Committee: Sep 2, 3 pm. [FILL IN: confirm invites arrived]
- Adama two-pager: this week. Satoshi 1:1: ~2 weeks, app-ready.

## Rebuild

`build_deck.py` (figures are pre-staged in `figures/`, drawn from the
2026-08-18 and 2026-08-25 deliverables + committed deck plots; the two
generated-this-week figures rebuild via
`deliverables/ann_update_2026-08-25/build_figures.py`).
