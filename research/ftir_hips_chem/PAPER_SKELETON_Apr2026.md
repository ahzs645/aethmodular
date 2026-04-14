# Paper Skeleton — April 2026

## Working Title Options

1. Cross-site interpretation of MicroAeth black carbon measurements at SPARTAN sites, with Addis Ababa as a site-specific anomaly
2. Multi-site comparison of optical and thermal black carbon metrics across low- and high-concentration SPARTAN sites
3. What MicroAeth black carbon measurements can and cannot support across four SPARTAN sites

## One-Sentence Paper Claim

Across Beijing, Delhi, JPL/Pasadena, and Addis Ababa, the MicroAeth signal is broadly interpretable across sites, but Addis Ababa shows a persistent site-specific anomaly that is not removed by iron splitting or seasonal splitting and therefore requires explicit interpretation limits.

## Short Abstract Skeleton

We compared HIPS, FTIR EC, and MicroAeth black carbon measurements across four SPARTAN sites spanning low- and high-concentration regimes. The clearest cross-site pattern is that Addis Ababa shows a compressed optical relationship and a persistent intercept that is not reproduced by Delhi, Beijing, or JPL/Pasadena. Iron-based and seasonal stratifications do not remove this anomaly, while AERONET and minute-resolution MicroAeth data help define the atmospheric and instrument-context limits of interpretation. These results support a practical framing: MicroAeth data remain useful across sites, but Addis Ababa requires site-specific caution rather than a generic calibration assumption.

## Results-First Figure Order

### Figure 1
HIPS vs aethalometer across all four sites.

Core message:
Addis Ababa is the clearest anomaly, Delhi is the key high-concentration comparison, and Beijing plus JPL/Pasadena anchor the lower-concentration regime.

### Figure 2
HIPS vs FTIR EC and aethalometer vs FTIR EC.

Core message:
The Addis problem is not only an aethalometer issue because the HIPS vs FTIR EC comparison preserves the intercept and compressed range more strongly than the aethalometer vs FTIR EC comparison.

### Figure 3
All-site iron split.

Core message:
Iron does not separate the Addis anomaly in a way that explains the regression behavior.

### Figure 4
All-site seasonal split.

Core message:
Seasonality changes context but does not remove the Addis anomaly; the problem is year-round.

### Figure 5
AERONET support section.

Core message:
AERONET adds column context and surface-column coupling information, but it is supporting evidence rather than the primary proof of the anomaly.

### Figure 6
Minute-resolution aethalometer diagnostics.

Core message:
The Addis wavelength stacking and unusual smoothness define interpretation limits and motivate follow-up with instrument experts.

## Section Outline

## 1. Introduction

### Paragraph 1
Set up the measurement problem. Optical black carbon methods are widely used because they provide operationally convenient absorption-based estimates, but their interpretation can vary with aerosol regime, concentration range, and site context.

### Paragraph 2
State the specific gap. There is less cross-site evidence showing how a MicroAeth-style optical signal behaves when compared against HIPS and FTIR EC across both low- and high-concentration SPARTAN sites.

### Paragraph 3
State the study design and thesis. This study compares four sites and uses Addis Ababa as the main anomaly case, with Delhi as the critical high-concentration comparison and Beijing plus JPL/Pasadena as lower-concentration anchors.

## 2. Methods

### 2.1 Sites and Instruments
Describe the four sites and the three measurement approaches: HIPS, FTIR EC, and MicroAeth.

### 2.2 Matching Strategy
Document the matched-filter and aethalometer comparison workflow, including sample counts by site.

### 2.3 Aethalometer Processing
Describe minute-resolution data handling, negative-value removal, smoothing choice, and the reason the same smoothing logic was used across sites.

### 2.4 Stratified Tests
Briefly describe the iron split, seasonal split, and AERONET support analyses.

## 3. Results

## 3.1 Cross-Site Optical Comparison

Start with Figure 1.

Paragraph goal:
Explain that Addis Ababa retains a strong fit but with a compressed slope and large intercept, while Delhi does not reproduce this behavior despite being the closest high-concentration comparison.

## 3.2 What FTIR EC Adds

Use Figure 2.

Paragraph goal:
Show that the HIPS vs FTIR EC comparison preserves the Addis anomaly more clearly than the aethalometer vs FTIR EC comparison, which narrows the interpretation problem.

## 3.3 Iron Does Not Resolve the Addis Pattern

Use Figure 3.

Paragraph goal:
State that iron-based splitting does not materially separate the regressions at Addis or across the other sites.

## 3.4 Seasonality Does Not Remove the Addis Anomaly

Use Figure 4.

Paragraph goal:
Show that the Addis slopes remain close across seasons, supporting a year-round rather than season-specific anomaly.

## 3.5 AERONET as Supporting Context

Use Figure 5.

Paragraph goal:
Frame AERONET as a surface-column context section that helps interpret coupling and dust-related hypotheses without replacing the core method-comparison results.

## 3.6 High-Resolution Diagnostic Limits

Use Figure 6.

Paragraph goal:
Show that Addis has stronger wavelength stacking and unusual smoothness relative to Beijing and Delhi, which raises data-quality and processing questions that matter for interpretation.

## 4. Discussion

### Paragraph 1
State the main interpretive result clearly: the cross-site comparison supports general MicroAeth usefulness, but Addis Ababa cannot be treated as a routine extension of the other sites.

### Paragraph 2
Explain why Delhi matters. It prevents a simplistic high-concentration explanation because it behaves more normally despite also being a high-concentration site.

### Paragraph 3
Discuss what has been ruled out. Iron and seasonality are useful negative tests because they reduce the number of plausible explanations without resolving the anomaly.

### Paragraph 4
Discuss what remains unresolved. The Addis anomaly may reflect a combination of site-specific atmospheric context and instrument or processing behavior, especially given the wavelength stacking and smoothing questions.

## 5. Conclusion

The paper should close on interpretation boundaries, not instrument blame.

Target closing point:
MicroAeth measurements can be compared productively across sites, but Addis Ababa requires explicit site-specific caution because its optical relationships diverge from the otherwise coherent multi-site pattern.

## Items Still Outside the Repo

- Direct clarification from Sina on the Addis correction pipeline and wavelength stacking.
- Direct clarification from Warren on HIPS-specific interpretation of the persistent Addis intercept.
- Final decision on whether the paper framing should emphasize MicroAeth performance first or HIPS/FTIR disagreement first.

## Immediate Writing Order

1. Write the Results section first using Figures 1–4.
2. Add the AERONET support section after the cross-site story is stable.
3. Add the diagnostics section only after the main argument is concise.
4. Write the Introduction last, once the paper framing is fixed.
