# Calibration-subset selection by spectral similarity — literature survey

Compiled 2026-08-20 (web survey) to ground the explorer's **Analogs lab** tab:
how the chemometrics/spectroscopy literature selects calibration samples by
spectral similarity, applied to our exact problem (13k IMPROVE spectra library
→ 239 Addis/ETAD target spectra).

## 1. Local calibration / library subset selection (NIR/FTIR chemometrics)

**LOCAL (Shenk & Westerhaus).** For *each* prediction sample, rank all library
spectra by **Pearson correlation** between the target spectrum and each library
spectrum (typically on derivative-preprocessed spectra); take the top *k*
(commonly 50–300) and fit a fresh (M)PLS model on them; predictions from
several factor counts are combined in a weighted average. Subset size *k* and
max factors tuned on a validation set. Consistently beats "GLOBAL" models on
large heterogeneous databases. Shenk, Westerhaus & Berzaghi, *J. Near Infrared
Spectrosc.* 5, 223 (1997). The companion **SELECT/Shenk-West algorithm** prunes
a library by standardized **Mahalanobis (H) distance** in PC space (H > 3
removed; neighborhood pruning removes near-duplicates) — `shenkWest` in R
`prospectr`.

**LWR (Næs/Isaksson).** Locally weighted regression: per target, find *k*
nearest neighbors by **Euclidean or Mahalanobis distance in PC score space**,
fit a *weighted* PCR/PLS with tricube distance weights; modern kNN-LWPLSR
weights in PLS score space. Næs, Isaksson & Kowalski, *Anal. Chem.* 62, 664
(1990); Næs & Isaksson, *Appl. Spectrosc.* 46 (1992).

**RS-LOCAL (closest published analog to our problem).** Data-driven instance
transfer: repeatedly resample small library subsets, evaluate each by how well
a model built on it predicts target-site samples *with known reference values*,
keep library samples that recur in well-performing subsets. Explicitly
target-matched. Lobsey, Viscarra Rossel et al., *Eur. J. Soil Sci.* 68, 840
(2017). Related: memory-based learning on soil MIR libraries (Dangal et al.,
*Soil Systems* 3, 11, 2019; Baumann et al., *SOIL* 7, 525, 2021).

## 2. Representative-sample selection (spanning designs — wrong objective here)

**Kennard–Stone** (greedy max–min distance; *Technometrics* 11, 137, 1969),
**DUPLEX** (Snee 1977), **SPXY** (Galvão et al., *Talanta* 67, 736, 2005).
These *span* a library rather than match a target — useful only for splitting
a selected subset into cal/val, not for the selection itself.

## 3. Similarity metrics used in practice

| Metric | Representation | Notes |
|---|---|---|
| Pearson correlation | usually 1st/2nd-derivative spectra | LOCAL's metric; offset/scale-insensitive; derivative kills the PTFE baseline |
| Spectral angle (SAM) | raw or normalized | cosine between vectors; = Pearson on mean-centered spectra (Kruse et al., *Remote Sens. Environ.* 44, 145, 1993) |
| Euclidean | raw / SNV-normalized / derivative | loading-dominated unless normalized |
| Mahalanobis in PCA/PLS score space | scores | the chemometric standard for extrapolation detection (GH distance) |
| Moving-window correlation | per-window r | localizes *where* spectra disagree (forensic IR review, *Forensic Chem.* 2020) |

Soil-library comparisons (*Geoderma* 2021) find score-space distances track
physicochemical similarity better than raw-spectrum distances.

## 4. Calibration transfer / domain adaptation (brief)

DS/PDS need shared standards → inapplicable across sites. **di-PLS**
(Nikzad-Langerodi et al., *Anal. Chem.* 90, 6693, 2018) aligns source/target
latent scores using only *unlabeled* target spectra — usable with 239 Addis
spectra, but changes the model rather than the sample set. TCA/PTCR: heavier;
noted only.

## 5. Aerosol-FTIR precedent (Dillner/Takahama lineage)

- Dillner & Takahama, *AMT* 8, 1097 & 4013 (2015): original IMPROVE PLS OC/EC.
- **Reggente, Dillner & Takahama, *AMT* 9, 441 (2016)**: transferability; squared
  **Mahalanobis distance in PLS score space** as an extrapolation diagnostic
  correlated with prediction error. 9/11 new sites transferred; 2 needed own
  models.
- **Weakley et al. 2016/2018**: QDA classification + multilevel EC models;
  Elizabeth NJ (diesel-heavy, atypical OC/EC) needed a separate calibration —
  direct precedent that atypical composition breaks pooled models and
  subsetting fixes it.
- Debus et al. 2019 (*Appl. Spectrosc.* 73, 271): instrument-to-instrument
  spectral dissimilarity degrades calibrations.
- Reggente et al. 2019 (*AMT* 12, 2287) calibration-strategy intercomparison;
  Takahama et al. 2019 (*AMT* 12, 525) statistical-calibration review;
  Ruthenburg et al. 2014 (*Atmos. Environ.* 86, 47); Debus et al. 2022
  (*AMT* 15, 2685).

**Bottom line:** the group's own tradition supplies (a) Mahalanobis
score-space distance as screening diagnostic and (b) composition-class
subsetting; nobody in aerosol FTIR has published per-target-site kNN library
selection — LOCAL-style selection on IMPROVE→Addis is novel but squarely in
the tradition.

## Cheap to try on 13k × 239 — recommendations

1. **Correlation/SAM kNN on 2nd-derivative spectra → pooled-subset PLS**
   (LOCAL-style, one model): rank by correlation to Addis, sweep N.
   *Most promising first try.* → in the explorer's Analogs lab as
   `corr_median` / `cosine_median` in `deriv2` space.
2. **Mahalanobis in PCA score space** (Reggente-2016 diagnostic as selector):
   whitened distance to the Addis centroid. → Analogs lab `mahalanobis_pca`.
3. **RS-LOCAL-style resampling** against the Addis filters that have deployed
   EC: optimizes the actual objective; run after 1–2 to check agreement.
   (Not yet implemented — candidate for a batch-mode extension.)

Skip: KS/DUPLEX/SPXY (spanning), PDS (needs standards), TCA (heavy). di-PLS is
a worthwhile fourth experiment (unlabeled-target-only) but model-side.
