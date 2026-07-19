"""Convert the percent-format run_ftir_*.py scripts into executed notebooks.

Usage: python scripts/build_notebooks.py [11 12 ...]
Run from research/ftir_ec_phase3/. Each script's tl;dr / Takeaways placeholders
are replaced with the finalized text below, then the notebook is executed
top-to-bottom with nbclient so every number in the committed notebook comes
from a real run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PLACEHOLDER = "(filled in by the finalize step after execution)"

TLDR = {
    "11": """\
Ann's lowest-OC/EC strategy is the **best-performing calibration cohort of phases 2–3, but it
does not zero the Addis intercept**. The 800 lowest-OCEC IMPROVE filters (TOR OC/EC ≤ 2.27 —
the pool median is 5.54) give the strongest site-held-out TOR test of any cohort tried
(R² **0.911**, slope **0.87**, RMSE **3.41 µg/filter**) and improve the fixed-190-filter Addis
comparison to intercept **−3.22** and RMSE **1.16 µg/m³** at MAC = 10, versus **−4.17** and
**1.49** for the deployed SPARTAN calibration. Five size-matched random cohorts show intercepts
can drift toward zero by chance (−1.8 to −3.4) — but only with collapsed slopes (0.93–1.49) and
worse RMSE; the OCEC cohort is the only one that improves intercept, RMSE, and the TOR test
simultaneously. N = 400 is too small to support a model (k = 3, held-out R² 0.19); N = 1600
dilutes the effect. The intercept is MAC-invariant: at MAC = 6 the slope becomes **0.95** with
the same −3.22 offset. An independent replication (separate implementation, same protocol and
seed; outputs in `output/tables/ftir_11/`) reproduces the headline metrics to the last digit
and adds a VIP comparison.""",
    "12": """\
The Addis ~1600 cm⁻¹ band **does not behave like amine or liquid water, and is consistent with
carboxylate / oxygenated-organic absorption**. Its peak center sits tightly at
**1616–1620 cm⁻¹** (IQR), below every IMPROVE cohort (medians ≥ **1633**), and its height
covaries strongly with carbonyl (r = **0.94**) and CH (r = **0.88**) but barely with
N–H/O–H 3100–3400 (r = **0.11**) — the amine companion that should be there if the band were
amine. The symmetric-COO⁻ partner test near 1400 cm⁻¹ is **inconclusive by construction**: that
window lies outside the baselined region, so it must be revisited with AIRSpec-corrected
spectra and, ideally, lab standards. The band also tracks deployed FTIR EC (r = **0.80**) and
HIPS τ (r = **0.66**), consistent with the phase-2 finding that the EC calibration leans on
oxygenated-organic covariates. An independent replication with different window conventions
(outputs in `output/tables/ftir_12/`) reaches the same ordering (Addis ≈ 1617 vs IMPROVE
≥ 1635).""",
}

TLDR["13"] = """\
Real AIRSpec baselines (validated Python port, worst deviation **6×10⁻⁷** vs the R run) change
the story in three ways. **(1) They do not fix the HIPS transfer**: an IMPROVE HIPS-τ model
rebuilt on EDF-6/8 baselined spectra still predicts median Addis Fabs **≈12.5 vs 47.1**
observed (bias **−35**) — the ftir_08 transfer failure is compositional, not a baseline
artifact. **(2) They produce the closest-to-zero Addis intercept of any locked-protocol
calibration yet**: the lowest-OCEC-800 cohort on df1 = 6 spectra gives intercept **−1.62** and
slope **0.86** at MAC = 10 with the project's best held-out TOR test (R² **0.904**, slope
**1.01**, k = 5) — but Addis scatter grows (R² 0.66, RMSE 2.41 vs 0.77/1.16 raw) and a
**−2.3 µg/m³** mean bias remains. **(3) The 906-sample smoke calibration collapses** on
corrected spectra (Addis slope **0.37**): its raw-spectra behavior rode on baseline structure.
df1 = 6 vs 8 is immaterial throughout. ftir_12 follow-up: the corrected 1600-band center stays
at **1617–1620 cm⁻¹**, and the apparent 1520–1560 companion (raw-height r = 0.87) vanishes
under CH normalization (Spearman **−0.29**) — a loading artifact, leaving peak position plus
the missing N–H companion as the carboxylate evidence."""

TAKEAWAYS = {
    "11": """\
- **The OCEC hypothesis has real signal.** Selecting calibration filters by low TOR OC/EC —
  with no spectral or Addis information at all — produces the only cohort that improves the
  Addis intercept, Addis RMSE, and the locked TOR test at the same time.
- **It is still ~3.2 µg/m³ from closing the gap.** If composition mismatch in OC/EC were the
  whole story, the intercept should approach zero; it does not. The remaining candidates from
  the meeting are the un-baselined spectra (ftir_13) and genuinely absent charcoal-like
  samples in IMPROVE.
- **Random-cohort intercepts are a trap.** A near-zero intercept with a collapsed slope is
  worse, not better; any cohort comparison must report slope, intercept, RMSE, and a held-out
  TOR test together.
- **Cohort size matters non-monotonically.** 400 is too few sites/filters to support the model;
  1600 pulls the OC/EC cut to 2.8 and starts re-admitting the ordinary IMPROVE mixture.
- **Next decisive test** remains an independent Addis EC reference (TOR on collocated filters
  or an agreed MAC/protocol bridge), since HIPS/MAC still sets the comparison scale.""",
    "12": """\
- **Amine and water-bend assignments are disfavored** for the Addis 1600-band: the center is
  too low and the N–H/O–H companion covariation is absent.
- **Carboxylate (or a related oxygenated-organic feature) is the leading candidate**, matching
  Satoshi's suggestion; charcoal-burning chemistry plausibly produces carboxylate-rich aerosol.
- **The definitive tests are still open**: the symmetric-COO⁻ partner near 1400 cm⁻¹ and the
  nitro symmetric band near 1340 cm⁻¹ both sit below the 1425 cm⁻¹ edge of the baselined
  region. AIRSpec-corrected spectra (ftir_13) recover 1425–1500 but not 1400; resolving this
  fully needs the sub-1500 model Satoshi's group is developing, or lab standards.
- **Calibration relevance**: the band's strong covariation with carbonyl/CH and with deployed
  EC supports the phase-2 conclusion that FTIR EC predictions ride on oxygenated-organic
  covariates — exactly the proportionality that charcoal burning could break.""",
}

TLDR["15"] = """\
Three questions, three clean answers. **(1) The corrected-model intercept is real and so is
its remaining gap**: a site-cluster bootstrap (B = 200) puts the AIRSpec OCEC-800 Addis
intercept at 95% CI **[−1.78, −1.06]** — excluding zero *and* entirely disjoint from the raw
model's **[−4.88, −2.81]**. The AIRSpec improvement is not split luck. **(2) The two models
miss differently**: the raw model's Addis residuals track extrapolation distance (D² r =
**0.71**) and swing with season (+0.49 Kiremt to −1.25 Dry), while the corrected model's
residuals are decoupled from D² (r ≈ 0.2 / Spearman −0.05) and sit at a near-constant
**−2.0 to −2.6 µg/m³ across all three seasons** — a stable offset, with |residual| growing
with loading. That is the signature of a missing constant component or a reference/MAC
mismatch, not composition-driven scatter. **(3) The hybrid cohort fails**: selecting the 800
spectrally-nearest-to-Addis filters inside the lowest-2,000-OC/EC pool collapses the held-out
TOR test (R² **0.19**, slope **0.20**) — cohort engineering has hit its ceiling, and
OCEC-800 + AIRSpec stands as the best candidate."""

TAKEAWAYS["15"] = """\
- **Stop tuning cohorts.** The hybrid negative result plus the bootstrap CIs close out the
  cohort-engineering direction: no selection strategy tested in phases 2–3 moves the intercept
  further without breaking the TOR test.
- **Prefer the corrected model for interpretation even where its RMSE is worse.** The raw
  model's Addis residuals are extrapolation-driven (D² r = 0.71) and season-dependent; the
  corrected model's are a clean, season-stable offset. A constant, explainable miss beats a
  smaller but structured one.
- **The season-independent −2 to −2.6 µg/m³ offset is the sharpest clue yet**: it behaves like
  a missing constant absorption component (Satoshi's EC sloping-absorption discussion) or a
  HIPS MAC/protocol mismatch — both resolvable only with an external anchor, not more IMPROVE
  data.
- **Decisive next data remain**: an independent Addis EC reference (TOR on archived filters or
  Teflon/quartz pairs at Adama/Bishoftu) and the INDH/CHTS/ETBI spectra pull."""

TAKEAWAYS["13"] = """\
- **The baseline caveat is discharged.** Phase-2's ">1500 cm⁻¹ offset correction is not
  AIRSpec" disclaimer no longer shields any hypothesis: with validated EDF-6/8 baselines, the
  transfer gap and most of the intercept persist. What remains points at composition — the
  charcoal-like samples IMPROVE may simply not contain.
- **OCEC-800 + AIRSpec is the best candidate calibration by the meeting's stated objective**
  (intercept toward zero without sacrificing the TOR test), at the price of Addis scatter and
  a residual negative bias. Whether slope 0.86 at MAC = 10 or 0.51 at MAC = 6 is "right"
  still hinges on the unresolved MAC/protocol bridge.
- **Baseline-sensitive skill is not skill**: the smoke model's raw-spectra Addis behavior was
  an artifact. Any future cohort comparison should run on AIRSpec-corrected spectra from the
  start (the port runs at ~15 ms/spectrum, so this is now free).
- **Band identity**: center 1617–1620 cm⁻¹ is rock-stable under real baselining; the
  1520–1560 covariation was loading-driven. Settling carboxylate vs aromatic ring modes needs
  the region below 1425 cm⁻¹ (Satoshi's sub-1500 model) or lab standards.
- **Fixed-df caveat**: this run uses fixed (df1, df2) = (6, 4) and (8, 4) as in the validated
  ETAD R run; AIRSpec's per-spectrum NAF-based EDF selection is not replicated."""

SCRIPTS = {
    "11": ("run_ftir_11.py", "ftir_11_ocec_ratio_cohort.ipynb"),
    "12": ("run_ftir_12.py", "ftir_12_band_1600_identity.ipynb"),
    "13": ("run_ftir_13.py", "ftir_13_airspec_corrected_calibrations.ipynb"),
    "15": ("run_ftir_15.py", "ftir_15_uncertainty_and_hybrid.ipynb"),
}


def script_to_cells(text: str) -> list:
    cells = []
    kind, lines = None, []

    def flush():
        nonlocal kind, lines
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if not body.strip():
            kind, lines = None, []
            return
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else ("" if line == "#" else line)
                for line in body.splitlines()
            )
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
        kind, lines = None, []

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind = "markdown"
        elif line.startswith("# %%"):
            flush()
            kind = "code"
        else:
            lines.append(line)
    flush()
    return cells


def build(number: str) -> None:
    script_name, notebook_name = SCRIPTS[number]
    text = (Path("scripts") / script_name).read_text()
    for replacement, marker_context in ((TLDR[number], "## tl;dr"),
                                        (TAKEAWAYS[number], "## Takeaways")):
        # Replace the first placeholder following each section marker.
        marker_at = text.index(marker_context)
        placeholder_at = text.index(PLACEHOLDER, marker_at)
        text = text[:placeholder_at] + "\n".join(
            "# " + line if line else "#" for line in replacement.splitlines()
        ).removeprefix("# ") + text[placeholder_at + len(PLACEHOLDER):]

    notebook = nbformat.v4.new_notebook(cells=script_to_cells(text))
    client = NotebookClient(notebook, timeout=1800,
                            resources={"metadata": {"path": "."}})
    client.execute()
    nbformat.write(notebook, notebook_name)
    print(f"executed and wrote {notebook_name}")


if __name__ == "__main__":
    for number in (sys.argv[1:] or list(SCRIPTS)):
        build(number)
