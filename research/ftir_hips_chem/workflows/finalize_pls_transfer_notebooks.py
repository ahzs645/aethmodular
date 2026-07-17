"""Replace provisional notebook summaries with findings from executed output tables.

Run after all three notebooks have executed successfully.  This keeps the summary
claims tied to saved, inspectable results rather than duplicating calculations.
"""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


SUMMARIES = {
    "ftir_07_current_pls_vip_diagnostics.ipynb": {
        "tl;dr": """## tl;dr

The current IMPROVE EC calibration shares substantial predictor structure with the current OC
calibration: VIP-profile Spearman r = **0.934**, **179 of the top 200** wavenumbers overlap, and
**58.5% of EC squared-VIP mass** lies at wavenumbers where OC VIP ≥ 1. This supports the concern
that EC prediction leans heavily on OC-associated spectral covariance, while not proving that any
individual band is uniquely causal. The Python reconstructions reproduce the exported R fitted
values to maximum absolute differences of **5.1×10⁻⁵ (EC)** and **0.0032 (OC)**.""",
        "Takeaways": """## Takeaways

- EC and OC feature rankings are strongly coupled, so an EC transfer failure can plausibly follow
  a change in OC/EC or functional-group covariance even if EC itself is present.
- The EC model places **43.1%** of squared-VIP mass below 1500 cm⁻¹, **7.4%** in the broad carbonyl
  region, and **4.7%** in the aliphatic C–H stretch. Cutting the spectrum at 1500 cm⁻¹ would discard
  a large share of the model's current information.
- VIP is a post hoc importance diagnostic. These overlaps motivate targeted transfer tests; they
  do not assign a unique molecule or establish causation.""",
    },
    "ftir_08_hips_transfer_and_vip.ipynb": {
        "tl;dr": """## tl;dr

The IMPROVE HIPS calibration does **not** transfer to Addis. After correcting the response to a
filter-loading basis (HIPS optical depth τ), the 916-sample, site-held-out IMPROVE model predicts a
median Addis Fabs of **15.0** versus **47.1 observed** (n = 239; RMSE **34.38**, mean bias **−33.01**).
The deliberately unit-mismatched direct-Fabs fit is worse (RMSE **46.52**). An Addis-only HIPS model
reaches nested-CV RMSE **3.82**, R² **0.883**, slope **0.912**, and intercept **+4.35**. Its top 200 VIP
wavenumbers have **zero exact overlap** with IMPROVE's top 200 despite a broader rank correlation of
0.797, showing that the highest-priority spectral features change materially.""",
        "Takeaways": """## Takeaways

- The requested deviation is large even after volume/deposit-area normalization, so it cannot be
  dismissed as the naive Fabs unit mismatch.
- Addis spectra contain enough repeatable information to predict Addis HIPS internally, but an
  Addis-only HIPS calibration is a mechanism probe—not a TOR EC calibration and not independent
  evidence that HIPS is absolute EC truth.
- Relative VIP emphasis shifts from the IMPROVE O–H/N–H region (**32.8%** of squared-VIP mass) toward
  the Addis fingerprint region (**32.1%**, versus **18.6%** in IMPROVE). Exact peak assignments remain
  uncertain because FTIR bands overlap.
- The newer SPARTAN HIPS source supplies 239 matched Addis filters versus 190 in the user-listed
  backup; all 190 overlapping Fabs values agree exactly, and the τ reconstruction matches the stored
  HIPS τ to numerical precision.""",
    },
    "ftir_09_improve_analog_selection.ipynb": {
        "tl;dr": """## tl;dr

The full lot-248/251 search screened **13,634 spectra** (13,632 unique filters), found **13,010** with
eligible TOR EC references, and selected **400 analogs spanning 120 sites**. The exploratory analog
TOR model improves Addis agreement with HIPS at MAC = 10: slope **1.32**, intercept **−1.64**, R²
**0.702**, and RMSE **1.02 µg m⁻³**, versus **2.30**, **−5.38**, **0.607**, and **2.70** for the current
model. Its RMSE is lower than all 10 size-matched random refits, and its R² exceeds their maximum
(0.693). The apparent improvement is MAC-sensitive: at MAC = 6, the analog model's RMSE is **3.49**
versus **3.15** for the current model.""",
        "Takeaways": """## Takeaways

- Searching the complete pool rather than smoke days alone finds a diverse analog cohort and yields
  a promising TOR recalibration at MAC = 10; the result is not robust enough to replace the deployed
  model because analog selection was informed by HIPS and evaluated against HIPS.
- Classical domain flags do not explain the transfer failure: Addis median score leverage and Q
  residual are close to the IMPROVE HIPS calibration. The robust HIPS model has only one component,
  so Q and VIP-weighted spectral mismatch remain essential complementary diagnostics.
- The strongest offset-corrected, HIPS-weighted gaps cluster near **1161, 1248, 1131, and 1463 cm⁻¹**.
  Treat these as search coordinates, not chemical identifications, until the full pool is processed
  with a validated AIRSpec EDF 6–8 workflow.
- The analog TOR CV optimum is **36 components** with a shallow neighborhood from roughly 33–37;
  component/VIP stability and a held-out TOR validation set are required next. The saved 400-sample
  list is ready for a preregistered R-tool rebuild and external test.""",
    },
}


def replace_section(source: str, heading: str, replacement: str) -> str:
    marker = f"## {heading}"
    start = source.find(marker)
    if start < 0:
        return source
    end = source.find("\n## ", start + len(marker))
    if end < 0:
        end = len(source)
    prefix = source[:start].rstrip()
    suffix = source[end:].lstrip()
    pieces = [piece for piece in (prefix, replacement.strip(), suffix) if piece]
    return "\n\n".join(pieces)


def main():
    for filename, replacements in SUMMARIES.items():
        path = ROOT / filename
        notebook = nbformat.read(path, as_version=4)
        changed = 0
        for cell in notebook.cells:
            if cell.cell_type != "markdown":
                continue
            for heading, replacement in replacements.items():
                updated = replace_section(cell.source, heading, replacement)
                if updated != cell.source:
                    cell.source = updated
                    changed += 1
        if changed != 2:
            raise RuntimeError(f"Expected two summary cells in {filename}; updated {changed}")
        nbformat.write(notebook, path)
        print(f"finalized {path}")


if __name__ == "__main__":
    main()
