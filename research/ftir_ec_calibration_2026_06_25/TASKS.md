# Task List — FTIR-EC Calibration (from the 2026-06-25 meeting)

Ordered so the quick unblockers come first. Notebook links point at what's already scaffolded here.

## 🔓 Do first — these unblock everything else (email, no compute)
- [ ] **Email Sean** (CC advisor) — ask to export/see **filter comments** in the Shiny app; discuss
      getting the R code / underlying data or a separate research instance.
- [ ] **Ask Sean for the smoke/non-smoke selection code** (the peak-ratio classifier: CH ~<3000,
      carbonyl ~1700, OH hump 3500–3000, ratios vs. thresholds).
- [ ] **Check the Davis email** (not just UMBC) for the **Ethiopia/SPARTAN spectra** Alex sent.
      → *Update:* the ETAD spectra are on the Drive and wired into `04_ethiopia_etad_spectra` — but
      confirm these are the same ones Alex meant.
- [ ] **Send the follow-up "list" email** with the requests above + your own thoughts.

## 🔬 Adama analysis — 5 samples, step by step  →  `01`, `02`, `03`
- [x] Plot the **FTIR spectra** of all 5 samples — `01_adama_spectra`
- [x] Cross-plot **Tor EC vs. general FTIR-EC** — `02_adama_tor_vs_ftir`
- [ ] Overlay **Tor EC vs. biomass FTIR-EC** — `02` (paste biomass predictions into the INPUT cell)
- [x] Per-sample **bar plot: char (EC1 − OP) vs. soot (EC2 + EC3)**, OPTR **and** OPTT — `03`
- [ ] Identify which samples actually have biomass burning (interpret 03 + the anomalous day)

## 🌍 Ethiopia (ETAD) analysis  →  `04`
- [x] Plot **all spectra** — `04_ethiopia_etad_spectra`
- [ ] Split by **Navid's high-biomass vs. diesel months** (fill `BIOMASS_MONTHS` / `DIESEL_MONTHS`
      in `04` with Navid's actual month list)
- [ ] Run the calibrations **vs. FABS** (MAC = 10 for now) — needs the calibration outputs (`05`)

## 🧪 Calibration experiments — name each variant  →  `05` (blocked)
- [ ] **No filtering** at all
- [ ] **Only samples below the 1:1 line** / only the removed samples
- [ ] **EC-threshold cutoff** (≥ ~70 µg, or Ethiopia EC range ÷ 10)
- [ ] **OPT (transmittance) vs. OPR (reflectance)** — Adama half already done in `03`
- [ ] Apply the **naming scheme** (`NAMING_SCHEME.md`) to every variant
> Blocked on: full IMPROVE+smoke training spectra, the comments export, and Sean's smoke classifier.

## 📐 Methods & reading
- [ ] Pick **number of components** as the **first major RMSECV minimum** — be consistent in *how*.
- [ ] Read the **Weakley EC / CSN paper** (second-derivative components; first comps = Teflon).
- [ ] Double-check the **char/soot equations** against Han et al. (source paper).

## 🎓 Career / logistics (1:1 with advisor next week)
- [ ] Prep: what **you** want, your **funding** situation, and continue-the-deep-work vs.
      write-a-report-and-return-to-walkability.
- [ ] Note: meetings **irregular through fall** (advisor traveling ~6–7 times).

---
### Legend
`[x]` = scaffolded/executed in this folder (figure/table produced) · `[ ]` = still to do / blocked.
"Scaffolded" means the plot runs on the data we have — the *scientific* follow-through (biomass
overlay, Navid's months, interpretation) is still yours.
