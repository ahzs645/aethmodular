# Paper Skeleton — April 2026

## Working Title Options

1. Cross-site interpretation of MicroAeth black carbon measurements at SPARTAN sites, with Addis Ababa as a site-specific anomaly
2. Multi-site comparison of optical and thermal black carbon metrics across low- and high-concentration SPARTAN sites
3. What MicroAeth black carbon measurements can and cannot support across four SPARTAN sites

## One-Sentence Paper Claim

Across Beijing, Delhi, JPL/Pasadena, and Addis Ababa, the MicroAeth and FTIR EC signals are broadly interpretable across sites, but Addis Ababa shows a persistent HIPS-specific anomaly that is not removed by iron, dust, seasonality, blanks, or simple volume correction; the remaining explanation is an extreme HIPS/filter-loading regime that requires explicit interpretation limits.

## Short Abstract Skeleton

We compared HIPS, FTIR EC, and MicroAeth black carbon measurements across four SPARTAN sites spanning low- and high-concentration regimes. The clearest cross-site pattern is that Addis Ababa shows a compressed HIPS relationship and a persistent intercept that is not reproduced by Delhi, Beijing, or JPL/Pasadena. Iron-based, seasonal, dust, blank, and volume-related tests do not remove this anomaly. Addis instead occupies an extreme HIPS absorption and filter-loading regime. A supporting IMPROVE high-`fAbs` comparison shows that ETAD-like absorption values exist in IMPROVE but are rare, making this regime a boundary case rather than routine network behavior. These results support a practical framing: optical and thermal EC comparisons remain useful across sites, but Addis Ababa requires site-specific caution rather than a generic calibration assumption.

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
IMPROVE high-`fAbs` comparison.

Core message:
ETAD/Addis-like HIPS absorption values are uncommon but not nonexistent in IMPROVE. The current full FED pull gives 207 ETAD-range samples and only 5 strict `fAbs >= 70` samples, so this figure supports the claim that Addis sits in a rare HIPS regime while showing that strict high-tail IMPROVE data are too sparse to close the mechanism fork alone.

### Figure 7
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

### 2.5 IMPROVE High-Fabs Comparison
Describe the FED pull, parser, positive EC + fAbs filtering, ETAD-like `fAbs` range, strict high-tail threshold, OLS/through-origin MAC/bootstrap summaries, site/date counts, and loading sensitivity using IMPROVE collection-area assumptions.

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

## 3.6 IMPROVE High-Fabs Boundary Test

Use Figure 6.

Paragraph goal:
Show that IMPROVE has a small number of ETAD-like HIPS absorption samples, including a very small strict high-tail group. Emphasize that selecting rows by high `fAbs` creates a response-truncation artifact, so the figure is strongest as a regime-boundary test rather than a definitive regression comparison. Loading-matched IMPROVE samples behave more normally than the fAbs-selected tail, suggesting that high EC loading alone does not fully reproduce the Addis pattern.

## 3.7 High-Resolution Diagnostic Limits

Use Figure 7.

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
Discuss the IMPROVE fork. The full FED pull shows ETAD-like HIPS absorption values are a rare tail in IMPROVE, but the strict high-tail sample count is too small for a final slope/intercept conclusion. This strengthens the argument that Addis occupies an extreme HIPS regime while leaving open whether the failure mode is generic high-loading HIPS/MAC behavior or SPARTAN/Addis-specific filter/instrument behavior.

### Paragraph 5
Discuss what remains unresolved. The Addis anomaly may reflect a combination of site-specific atmospheric context and instrument or processing behavior, especially given the wavelength stacking, extreme HIPS loading, and unresolved SPARTAN support-screen/pixelation parameters.

## 5. Conclusion

The paper should close on interpretation boundaries, not instrument blame.

Target closing point:
MicroAeth measurements can be compared productively across sites, but Addis Ababa requires explicit site-specific caution because its optical relationships diverge from the otherwise coherent multi-site pattern.

## Items Still Outside the Repo

- Direct clarification from Sina/Cena on the Addis correction pipeline, wavelength stacking, and what calibration failure modes could produce intercept plus compression.
- Direct clarification from Warren on HIPS-specific interpretation of the persistent Addis intercept and the exact meaning of pixelation alpha.
- Direct clarification from SPARTAN / Christopher Ockfort on support-screen hole fraction H, mesh geometry, and any observed HIPS nonlinearity on heavily loaded SPARTAN PTFE filters.
- Final decision on whether the paper framing should emphasize MicroAeth performance first or HIPS/FTIR disagreement first.

## Immediate Writing Order

1. Write the Results section first using Figures 1–4.
2. Add the AERONET support section after the cross-site story is stable.
3. Add the IMPROVE high-`fAbs` boundary-test section before final diagnostics.
4. Add the diagnostics section only after the main argument is concise.
5. Write the Introduction last, once the paper framing is fixed.
