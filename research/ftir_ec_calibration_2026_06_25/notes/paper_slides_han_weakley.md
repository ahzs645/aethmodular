# Paper slides — Han 2007 & Weakley 2016 (2–3 slides)

Supporting-literature slides for next week. **Han confirms the char/soot equations** (verified in the
PDF: Table 1 footnote reads *"EC1 = Measured EC1 − POC"*), and Weakley supports source-specific EC
calibration + RMSECV component selection + second-derivative preprocessing.

Papers:
- **Han et al. 2007**, *Chemosphere* — "Evaluation of the thermal/optical reflectance method for
  discrimination between char- and soot-EC." (`…/NASA MAIA/Data/Han/Han_evaluation of TOR…2007.pdf`)
- **Weakley, Takahama & Dillner 2016**, *Aerosol Sci. Technol.* 50:10, 1096–1114,
  DOI 10.1080/02786826.2016.1217389 (Zotero `6BKW4NNE`). *(EC-specific "atypical/dedicated
  calibration" wording is from the companion CSN-EC Weakley paper — re-check exact wording before
  finalizing.)*

> **Caveat to state on every slide:** Han supports char-EC / soot-EC as **operational TOR-defined
> fractions**, not absolute chemical truth. Label everything **"TOR-defined char-EC / soot-EC."**

---

## Slide 1 — Why char and soot can be separated by TOR (Han 2007)

**Point:** char and soot oxidize at different thermal steps, so the TOR fractions separate them.

**Equations (show these):**
```
char-EC  = EC1 − POC          (POC = pyrolyzed OC = OP; "EC1 = Measured EC1 − POC", Han Table 1)
soot-EC  = EC2 + EC3
char/soot ratio = (EC1 − POC) / (EC2 + EC3)
```

**Figure:** Han Fig. 1 thermograms — char peak at **EC1 (550 °C)**, soot peak at **EC2 (700 °C)**,
carbon black extending into **EC3 (800 °C)** (all in 98% He / 2% O₂).

**Quotes:**
- *"…activation energy is lower for char- than soot-EC. Low-temperature EC1 (550 °C in a 98% He/2% O₂
  atmosphere) is more abundant for char samples. Diesel and n-hexane soot samples exhibit similar EC2
  (700 °C…) peaks, while carbon black samples peak at both EC2 and EC3 (800 °C…)."* — Han 2007,
  Abstract.
- *"EC1 = Measured EC1 − POC."* — Han 2007, Table 1 footnote (this **is** char-EC = EC1 − OP).

---

## Slide 2 — Why a source-specific (smoke-only) EC calibration matters (Weakley)

**Point:** a single global FTIR-EC model can fail when EC composition/source differs; separating
atypical EC into its own calibration improves prediction. → justifies our **smoke-only** calibration.

**Show:**
- single global EC model → biased for atypical samples
- PLS-DA classification → typical vs. atypical EC
- separate/dedicated calibration → improved prediction

**Quote/phrase:**
- *"…certain sites exhibiting more 'atypical' spectra may benefit from their own dedicated EC
  calibration(s)."* — Weakley (CSN EC). *(re-verify exact wording)*

**Tie-in:** this is exactly CAL-0…CAL-5 in `06_calibration_variants_components` — build the smoke-set
first, then test selection rules (incl. the inverse below-1:1 / removed-only) inside it.

---

## Slide 3 — How to choose components & why 2nd-derivative preprocessing (Weakley)

**Point:** don't hand-pick components — use RMSECV / Wold's criterion consistently; and
second-derivative preprocessing suppresses PTFE/baseline interference, cutting model complexity.

**Show:**
- Choose components at the **first major RMSECV minimum** (or first clear local minimum/plateau,
  Wold's R). Don't take a later, marginally-lower RMSE point that only adds complexity.
- Pipeline: **raw spectra → second derivative → variable selection (BMCUVE) → PLS.**
- Our own test (notebook `06`): raw vs 2nd-derivative RMSECV → does k drop? (Weakley: OC went ~35 → 3
  components after 2nd-derivative + vapor correction; first component predicts the analyte, the rest
  handle PTFE/inorganic interference.)

**Quotes/phrases:**
- *"…chosen according to a minimized root-mean-squared-error of cross-validation."* — Weakley (RMSECV
  component selection).
- Weakley OC: second-derivative + BMCUVE reduced complexity *"from 35 components to 3 after vapor
  correction, with the first component mainly predicting OC and the remaining components mostly
  handling PTFE/inorganic interference."* *(paraphrase from your notes — verify wording)*

**Caution (from the meeting):** use components to *diagnose* whether the model is learning smoke
signal vs. PTFE/water/inorganic interference; use VIP/BCI or selected wavenumbers for cautious
interpretation. **Avoid "component 1 = smoke"** unless the evidence is strong.
