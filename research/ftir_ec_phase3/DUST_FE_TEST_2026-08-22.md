# Does dust (Fe) explain the Addis absorption offset? — 2026-08-22

A **direct, AERONET-free test** of the "extra non-EC absorption" branch, prompted
by a literature finding: the canonical IMPROVE HIPS calibration is not
single-term. White et al. (2016, *Aerosol Sci. Technol.* 50, 984) fit

> **Fabs = 10.2·EC + 6.6·Fe**  (Mm⁻¹; µg m⁻³; 633 nm)

i.e. a **mineral-dust absorption term is already part of the accepted HIPS
calibration**. Since Addis is reported at 13.5% soil dust (up to 37.6% in dry
months; Tefera et al. 2020, *IJERPH* 17, 6998), dust-Fe was a strong candidate
for the offset — and we have Fe on 819 filters already.

## Method

`unified_filter_dataset.pkl`, four SPARTAN sites. Per filter, regress HIPS Fabs
on FTIR EC with and without ChemSpec Fe. Two data traps handled: ChemSpec rows
carry the **base FilterId** (`ETAD-0001`) while HIPS/FTIR carry the
cartridge-suffixed form (`ETAD-0001-1`) — join on the base id or the merge is
empty; and ChemSpec elements are **ng/m³** despite the parameter name, so Fe
must be divided by 1000 before mixing with µg/m³ EC.

## Result

| site | n | Fabs = a·EC + b·Fe + c | R² | EC-only intercept c₁ | 6.6·median(Fe) | **Fe-explained share of c₁** |
|---|---|---|---|---|---|---|
| **ETAD** Addis | 188 | 3.51·EC + 14.20·Fe + 26.10 | 0.831 | **+28.4 Mm⁻¹** | 1.92 | **7%** |
| INDH Delhi | 27 | 3.72·EC + 20.25·Fe + 24.22 | 0.599 | +35.3 | 4.36 | 12% |
| CHTS Beijing | 148 | 4.88·EC + 5.95·Fe + 4.44 | 0.641 | +6.0 | 2.21 | 37% |
| USPA Pasadena | 128 | 5.36·EC + 24.73·Fe + 0.20 | 0.661 | +0.8 | 0.49 | 59% |

Fitted Fe coefficients (5.95–24.73) bracket the literature 6.6, which is a
sanity check on the method.

## Findings

**1. Dust does NOT explain the Addis offset — 7%.** The ordering is the reverse
of what the dust hypothesis needs: Fe accounts for most of the *small* offsets
(Pasadena 59%, Beijing 37%) and almost none of the *large* ones (Delhi 12%,
Addis 7%). Adding Fe leaves a 26.1 Mm⁻¹ Addis intercept standing. Combined with
ftir_28's MA350 no-red-excess and the AERONET AAE (BC-dominated in absolute
terms), **the mineral-dust branch of "real non-EC absorption" is now largely
closed at 633 nm.**

**2. Delhi shows an offset as large as Addis (+35.3 vs +28.4 Mm⁻¹).** This
qualifies — not contradicts — the ETBI "Addis-specific" result: the offset is
not unique to Addis among *heavily loaded megacities*; it is absent at
*Bishoftu* and *Pasadena*, and small at Beijing. The pattern tracks loading, not
geography, which is the signature of a **loading-dependent HIPS artifact** —
the branch the AERONET scale-height result also favours (Addis surface
absorption too high for its column).

## Caveats — real, and they bite

- **Curvature manufactures intercepts.** ftir_26 established Fabs is *concave*
  in EC (≈7.50·EC^0.796); a straight-line fit through curved data produces a
  positive Fabs-axis intercept with no extra absorber present. Some of every c₁
  above is this artifact, and it is worse at high loading — which is exactly
  where Addis and Delhi sit. **This must be quantified (fit the power law per
  site, compare) before any of these intercepts are quoted as absorption.**
- `EC_ftir` is the *deployed* calibration's output, so it is not an independent
  EC reference; a slope of ~4 rather than 10 partly reflects that.
- INDH n = 27. Fe is a dust proxy — Al/Si/Ti are available and should be added.
- No RH/hygroscopicity treatment.

## Curvature test (run same day) — the offset may not be additive at all

Fitting `Fabs = α·EC^β` per site and asking how much intercept a **straight line
through a pure power law of that shape** would itself produce:

| site | n | linear | power law | β | curvature-implied intercept | observed | **curvature explains** |
|---|---|---|---|---|---|---|---|
| ETAD | 190 | 4.02·EC + 28.32 | 26.16·EC^0.394 | 0.394 | +29.50 | +28.32 | **104%** |
| CHTS | 160 | 6.04·EC + 5.33 | 11.14·EC^0.624 | 0.624 | +5.18 | +5.33 | **97%** |
| USPA | 130 | 8.14·EC + 0.69 | 9.99·EC^1.153 | 1.153 | −0.68 | +0.69 | −98% |
| INDH | 61 | 3.93·EC + 21.91 | 7.93·EC^1.070 | (R² < 0 — unusable) | −6.53 | +21.91 | n/a |

At **Addis and Beijing the apparent additive offset is fully accounted for by
curvature** — no residual absorber is needed (Addis residual −1.2 Mm⁻¹, Beijing
+0.15). And β falls monotonically as site loading rises (USPA 1.15 at median
Fabs 4.7 → CHTS 0.62 at 12.9 → ETAD 0.39 at 47.1), which is the shape a
loading-dependent instrument response would produce.

**However — and this is decisive for how much weight the finding can carry —
"additive offset" and "curvature" are two descriptions of the same data, not
competing hypotheses that this test can separate.** For `Fabs = a·EC + c`, the
apparent MAC is `Fabs/EC = a + c/EC`, which *must* fall with loading whenever
c > 0. So the accompanying MAC-vs-τ result (Addis apparent MAC 13.3 on the
lightest quartile → 7.7 on the darkest, r = −0.48; Beijing and Delhi flat;
Pasadena rising) is **mathematically entailed, not independent evidence**. The
power-law fits are also poor (INDH R² < 0) and `EC_ftir` is the *deployed
calibration's own output*, so none of this is an independent EC axis.

**What this means:** the Fabs–EC relationship at Addis cannot be decomposed into
"real absorber" vs "instrument curvature" using FTIR EC alone — the two are
degenerate on this data. Breaking the degeneracy needs an **EC reference that
does not come from the same optical/FTIR family**, i.e. **thermal-optical TOR
on quartz**. This is an independent argument for the quartz-TOR campaign being
the top priority, arrived at from a completely different direction than the
MAC fork.

## Next

1. Fit `Fabs = α·EC^β` per site and re-derive the offset as the *deviation from
   the site's own power law* — separates curvature from a genuine additive term.
2. Repeat with **TOR EC** where available (Adama quartz) instead of FTIR EC.
3. Add Al/Si/Ti and a K⁺ biomass tracer to the regression.
4. Test **"dark BrC"** (Chakrabarty et al. 2023, *Nat. Geosci.* 16, 683: AAE
   1.5–2.0, up to half of 664 nm absorption in biomass plumes) — the one
   non-EC absorber that survives at 633 nm and is *specifically* plausible for
   eucalyptus-fuelwood and waste burning in Addis. Currently unaddressed.
