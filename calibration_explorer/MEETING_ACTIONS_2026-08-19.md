# Actions from the Ann 1:1 — 2026-08-19

What the meeting settled, and everything it put on the list. Items marked ✅ were
built into the explorer the same day.

## Ratified conclusions (safe to build on)

- **Baselining is the lever.** Raw spectra underperform across the board; AIRSpec
  correction roughly halves the Addis intercept and fixes the slope. Ann: "raw
  spectra don't seem to be working very well for us."
- **Protocol choice barely matters.** Interleaved vs site-grouped give similar
  answers — "comforting… if we got totally different answers, holy moly."
- **Seasonal patterns hold regardless of calibration** (higher Kiremt, lower dry
  season) — "hugely important that the overall sense of the data stays consistent."
- **If you select on corrected spectra, calibrate on corrected spectra.** Ann:
  the selected group is only similar *in corrected space*, so going back to raw
  for the calibration undoes the point — "do baseline for both."

## Explorer / interface

- ✅ **Eval lot** control (Cohort/Spectra card, next to "Evaluate on"): report the
  crossplot on ETAD filters of one lot only — judge a lot-251 calibration on
  lot-251 filters. Lots come from the SPARTAN HIPS `LotId` (ETAD: 224× lot 251,
  40× 248, 16× 253). Predictions still cover every filter; only the readout is
  masked, so curve/fit caches are untouched. Flows through Run, Sweep, presets,
  and the Optimize tab.
- ✅ **Raw vs corrected selection view** (Selection tab, third ranking-view pill):
  overlays the committed selection metric computed in raw and AIRSpec-corrected
  space for the spectra-based cohorts, with shared-membership-at-cutoff in the
  caption — the "does baselining change who gets picked" graphs, now in the app.

## Runs to do next

1. **Select-corrected × calibrate-corrected** ("double baseline") for
   Ethiopia-shaped and analogs — the variant Ann asked for. Both knobs exist
   (Select on / Calibrate on = AIRSpec); the Optimize tab covers it via
   "+ corrected-space selection".
2. **Cutoff exploration on the corrected selection.** Ann's guidance: nothing
   above metric ≈ 0.3–0.5; consider cutting at the distribution peak ("everything
   to the left of the peak"); test several — the pool-distribution view + slider
   are the tool.
3. **Lot-251-only line of evidence**: train-lot 251 × eval-lot 251, raw vs
   corrected — tests Ann's hypothesis that baselining helps partly by erasing
   248-vs-251 media differences ("is it the lot, or the aerosol?").
4. Satoshi's k question (analogs ~21 components vs 9) — k sweep already covers it.

## Data / Davis

- **Lot-248 pool anomaly — RESOLVED 2026-08-20** via the AQRC Shiny app
  (all sites, 2015→2026): the database holds only **1,445 lot-248 spectra**
  vs **12,189 lot-251** (177,690 all lots). The 1,299-filter download was
  essentially complete — nothing was missed. Lot 248 was simply used briefly,
  network-wide, with sample dates clustered **Jan–Feb 2021** (Ann's "we
  started using 248 and moved on" guess was right). Consequence: any
  lot-248-trained cohort is also a *winter-2021-restricted* cohort — the
  batch's lot-248 leaders carry a season/time confound, not a data-quality
  artifact. Note for future queries: `LotNumber` is free-text with mixed
  formats (241a, 264b, 256, B0580020-14, FH00227659…) — 248/251 have no
  suffixed variants, but never assume numeric lots. Current 2026 production
  is on lots 264b/256.
- **ETBI (Bishoftu) spectra**: not in any email because Alex can't provide
  spectra — Mona can, and they're on the same Davis server in the *SPARTAN*
  database (separate from IMPROVE). Query it directly (32 filters, Oct–Dec 2025).
  **2026-08-20 — investigated to its end, now blocked on one email.** Discovery
  ran on the VPN'd Windows machine (`get_etbi_spectra.ps1`, .NET SqlClient after
  ODBC proved absent): `AD3\ajalil` reads only `Improve_2.1`; SPARTAN/ETBI is
  definitively NOT in it (site sweep of `sampler.Samplers` finds no ET*/SPARTAN
  rows); the SPARTAN db is one of the access-denied ones (`Spada` /
  `Networks_1_0` likeliest). ACTION: send Sean the drafted email asking which
  db + a `db_datareader` grant. Full schema map in
  `research/ftir_ec_phase3/scripts/AQRC_DB_NOTES.md` — including that IMPROVE
  spectra live full-resolution in `ftir.Scan` as scaled binary blobs.
  **2026-08-21 (ticket INC2653758): ACCESS GRANTED** — Greg Philip confirmed
  SPARTAN lives in the **Networks database** (`Networks_1_0`), read access on
  the whole DB approved by Sean; supported route is SSMS on the VM
  `aqrc-vd-jalil` (server `aqrc-sql`, Windows auth), but the AD-account grant
  should also work via the direct PowerShell route from the VPN'd Windows
  machine. Next: run the Networks_1_0 discovery block → Export-Etbi → pivot
  CSVs into `calibration_explorer/targets/etbi/`. Reply to Greg's ticket after
  confirming access.
- **Lot 249 (checked 2026-08-20, Shiny app)**: exists but is a phantom — 2
  filters total in the whole network (both FRES1, sampled late Aug 2024).
  Nothing usable.

## Deliverables

- **Adama summary, 2–3 slides** for Christian & Sina (~2–3 weeks): HIPS vs TOR vs
  FTIR OC/EC at Adama vs Addis. Message: Adama is NOT Addis — no HIPS-EC gap, no
  extreme OC/EC — so collecting more Adama samples won't help this problem.
- **FTIR group talk** a week from Thursday: late-breaking state of the work is
  fine ("group meetings aren't supposed to be finished"); show the explorer.
  Satoshi invited; possible Satoshi meeting ~2 weeks after.

## Admin

- Committee meeting **Tue Sep 2, 3 pm** — the invite emails apparently never
  arrived (nobody got them); resend and get a confirmation. Deck: the story,
  timeline, impact (MAIA/NASA, air quality where measurements are sparse);
  thesis through-line Ann endorsed: *"air quality in under-measured places,
  using computational methods to enhance sparse measurements."* Send Ann the
  draft for feedback before Hossein.
- AAAR: poster **Thursday 1–3 pm, session 9**; forward-of-acceptance to Ann done
  in-meeting.
