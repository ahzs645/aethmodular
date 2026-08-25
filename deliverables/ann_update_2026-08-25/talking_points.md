# FTIR group talk — 2026-08-27 — talking points

**One breath:** Since Aug 19 the Addis question became a five-city
adjudication: we decoded the HIPS calibration itself (a blank-line
regression), recomputed every filter under alternative blank lines, and the
Addis intercept survives (−1.27 ± 0.17, ~15% instrument share) while
Pasadena's slope anomaly dissolves and Delhi's 1.8× hardens; an exhaustive
9,555-job grid shows no calibration choice reconciles Addis and Delhi;
baselining removes the lot effect (Ann's hypothesis, confirmed at Beijing)
but relocates seasonality onto the dry season; three lines of evidence end
at quartz TOR + the extraction test.

## Ann's Aug-19 asks — each closed

- **Select corrected × calibrate corrected** → done; found the 440–490 basin
  (held-out 0.92 vs 0.87 at the locked 800). Wins 62.5% of bootstraps —
  leading candidate, not final.
- **Evaluate on one lot only** → Eval-lot selector; winner holds on 251-only.
- **Lot-248 pool anomaly** → resolved: 1,362 analyses network-wide, a
  two-month lot (Dec 2020–Feb 2021). The 1,299 download was complete.
- **Bishoftu spectra "get them yourself"** → done via Networks_1_0 SQL;
  plus Beijing, Delhi, Pasadena while in there.
- **"Lot or aerosol?"** → both tested: Bishoftu (same lots) has no offset;
  blank-line share ~15%; baselining specifically absorbs the 248-vs-251
  difference at Beijing (−4.5 → −0.9 Mm⁻¹, n.s.) — her predicted mechanism.
- **Seasonal stability** → extended: baselining *relocates* seasonality
  (wet seasons clean up, Dry drops to 0.60x) — reverse of the dry-is-fine
  hypothesis; ftir_24 will formalize.
- **Adama 2-pager** → figure drafted (slide 13); TOR-vs-FTIR panel + text
  this week.

## Don't get caught — the caveat list

- "corrected" is NOT a valid API spectra token (silently falls back to raw);
  everything in this deck uses `airspec`. If a number looks like baselining
  did nothing, check that first.
- ~15% is the **blank-line share**, not the full instrument share —
  high-loading nonlinearity is invisible to blanks (blank τ ≈ 0) and still
  degenerate with real absorption on FTIR-EC axes. Quartz TOR terminates it.
- Screening vs quoting: 71k grid rows are descriptive; quoted results are
  held-out / out-of-country. Winner takes 62.5% of bootstrap draws;
  corrected analog-440 takes ~⅓.
- The 1617 band is an **Ethiopian regional marker, decoupled from the
  offset** (Bishoftu has it, no offset). Any 1500–1650 claim needs two
  baselines (AIRSpec anchors at 1520–1600).
- Season fits are per-season Deming on restricted ranges — indicative;
  interaction fit is the ftir_24 deliverable.
- Delhi's grid "winner" (analogs-530 × deriv2) was 96% extrapolated and
  failed fresh holdouts — cite it only as the validation layer working.
- Beijing lot-effect n = 14 vs 34; loading not fully matched (1.18 vs 1.66
  µg/m³ median predicted EC).
- Adama figure is the seed, not Ann's full ask — the HIPS + TOR-OC/EC vs
  FTIR-OC/EC comparison still needs its panel.
- Standing: analog top-500 resolves to 477; entire-network Option A k=15
  cell unexplained; Eth-shaped+AIRSpec B2 (k=21) unstable.

## Blocked items — the precise asks

1. **Quartz TOR campaign** (decisive): ~36 filters, 3 seasons — one-pager
   `research/ftir_ec_phase3/quartz_tor_campaign_onepager.md`. Ask: Ann's
   go-ahead to route via Christian/Sina.
2. **Extraction test** (cheap companion): archived Addis + Delhi PTFE,
   water→methanol, HIPS re-measure; Beijing/Pasadena controls. Ask:
   Davis lab time + permission to sacrifice ~a dozen archived filters.
3. **Lot-253 spectra pull**: script staged
   (`get_improve_lot253_spectra.ps1`, pilot mode first, lot 255 sealed).
   Needs a session on the Windows/VPN machine.

## Personal / calendar

- AAAR: poster Thu 1–3 pm, session 9; flights + conference hotel booked.
  [FILL IN: forward acceptance email to Ann — she asked again]
- Committee: Sep 2, 3 pm. [FILL IN: confirm the re-sent invites arrived]
- FTIR group talk: this deck, 2026-08-27. Satoshi invited.
- Satoshi 1:1: ~2 weeks out; the app is the venue for his k=21 question.

## Rebuild

`build_figures.py` (figures; needs explorer on :5058, anchors validated) →
`build_deck.py` (pptx). Peer figures f_lot_*, f_aeronet_diurnal courtesy of
the sibling analysis session — provenance in SPARTAN_LOT_INVENTORY /
VALIDATION_LAYER / AERONET docs.
