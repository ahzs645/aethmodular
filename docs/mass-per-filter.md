# Mass per filter: sampled volumes and their limits

The gallery's **Filters → units → Mass per filter** switch (and the poster's per-filter panels)
turn every µg/m³ value into µg on the filter: concentration × the filter's sampled volume.
This is the quantity FTIR sees. A spectrum knows what sits on the filter, not how much air
went through it.

The switch matters because the two networks draw very different volumes. IMPROVE samples about
22.8 L/min for 24 h (about 33 m³). SPARTAN samples about 5 L/min (about 7 m³). The same
concentration therefore leaves about 4.5× more mass on an IMPROVE filter. Comparing
concentrations overstates how far Addis sits from the IMPROVE calibration set. Addis EC is about
32× the IMPROVE median per m³ of air, but about 9× per filter (poster build, 15 Oct 2026).

Code: `gallery/app/src/lib/viewFields.ts` does the conversion, using the field
`Sampled volume` (m³) that each network export writes.
- SPARTAN volumes: `gallery/data/export_spartan_reference.py`
- IMPROVE volumes: `gallery/data/export_improve_samples.py`

## SPARTAN

Neither the public SPARTAN files nor the public ChemSpec products carry a flow rate or a
volume. They give only `Hours_sampled` and the sampler model (`Collection_Description`). The
export takes the first of these that exists for each filter:

| Order | Source | How | Filters (of 5,064) |
|---|---|---|---|
| 1 | HIPS batches 1–51 (`SPARTAN_HIPS_Batch1-51.v2.csv`, `Volume`) | measured | 2,921 |
| 2 | Lab FTIR deliverable, batches 2–3 (`volume`) | measured; identical to HIPS on 715 shared filters (ratio 0.999–1.001) | 758 |
| 3 | Public `Hours_sampled` × 0.3 m³/h (5 L/min), SS5i samplers only | estimated | 940 |
| none | Older SS4i samplers | flow unknown | 445 |

The resulting volumes have a median of 7.2 m³ (5th–95th percentile 3.6–14.4 m³). Filters that
run 12 h or 48 h, rather than 24 h, explain the spread.

**The hours estimate.** On the 2,524 SS5i filters that also have a HIPS volume, the HIPS volume
divided by hours × 0.3 has these percentiles:
- median 1.00
- 5th–95th 0.95–1.03
- 1st–99th 0.92–1.07

160 of the 2,524 (6%) fall outside ±5%. So an estimated volume is good to a few percent for most
filters, but not for every one.

**SS4i filters (no volume).** These are 445 filters from 2013–2017, collected on the AirPhoton
SS4i sampler (cascade impactor or virtual impactor, mesh Teflon). By site: CHTS 119, IDBD 78,
BDDU 51, ARCB 43, SGSU 37, ILNZ 31, INKA 25, USMC 19, PHMO 12, USBO 11, NGIL 8, VNHN 6, ZAPR 5.
- Their `Hours_sampled` is mostly 24 h (range 3–72).
- None of them has a HIPS or lab volume, so nothing validates a flow for this sampler.
- The SS5i check below doesn't apply to them: their rows carry no `Analytical_MDL`.

In mass-per-filter views these filters drop out rather than take an assumed flow.

**A volume proxy we checked and did not use.** For SS5i filters, `Analytical_MDL / MDL` in the
public files equals about 0.12 × the HIPS volume, with an IQR of 0.120–0.123. So the detection
limits encode the volume up to a constant. The constant isn't explained, and some filters sit at
twice it, so it isn't used.

**Focus sites.** The four focus sites' rows (`filters.json`) take `Sampled volume` from their
SPARTAN network rows. This means it is present when the Filters sidebar shows the SPARTAN
network, which is the default. The 17 Sep meeting page uses its own export's `vol` (HIPS volume
for the Addis test set).

## IMPROVE

| Source | How | Samples (of 388,159) | Years |
|---|---|---|---|
| Cleaned IMPROVE table (`improve_valid_cleaned.csv`) | quartz (module C) `FlowRate_C × SampDur_C` | 162,912 | Aug 2015 on (complete from 2016) |
| FTIR local database TOR results (`results_tor.csv`) | `AverageFlowRate × ElapsedTime` | 30,567 more | 2005–2014, filters in the FTIR lots only |
| none | | 194,680 | 2003–2004 entirely; about 83% of 2005–2014; about 59% of 2015 |

Where the two sources overlap (27,189 samples), they agree to a ratio of 1.0000 (1st–99th
percentile 0.9998–1.0002). The median volume is 32.83 m³ (1st–99th percentile 29.8–34.7 m³),
which is exactly 22.8 L/min × 24 h.

**The flow is the carbon (quartz) module's.** HIPS absorption, PM2.5 mass and the elements are
collected on the Teflon module. Its median flow is 22.68 L/min, against 22.85 L/min for quartz,
so per-filter values for those fields run about 0.7% high.

**A nominal volume could fill the gaps but is not used.** All but about 2% of samples with a
known volume fall within ±10% of the 22.8 L/min × 24 h nominal. Filling the 194,680 missing
samples with that nominal value would be defensible for distributions. It would also hide real
short or low-flow runs, so the gallery leaves them out and says how many it dropped (Filters →
units).

## Per area instead of per filter

FTIR spectra respond to mass per deposit area (µg/cm²), not per filter. Per filter equals per
area only if the deposit areas match. Two things are unsettled:
- **IMPROVE:** the canonical area is an open decision, 3.5 or 3.53 cm²
  ([open-items.md](open-items.md#choose-the-canonical-improve-deposit-area)).
- **SPARTAN:** the HIPS file reports `DepositArea` per filter. It hasn't been compared with
  IMPROVE's area yet.

Until both are settled, quote µg per filter, not µg/cm².

## Open questions

See [research-questions-2026-10.md](research-questions-2026-10.md#sampled-volumes). In short:
1. What flow did the SS4i sampler run at?
2. Is there a flow source for IMPROVE 2003–2014 beyond the FTIR lots?
3. Why do the IMPROVE volumes in the poster build (median 30.5 m³, 21.2 L/min, from the FED
   export) differ from the cleaned table's (32.8 m³)?
4. Should the per-filter view use the Teflon module's flow for Teflon-borne fields?
5. What deposit areas should a µg/cm² comparison use?
