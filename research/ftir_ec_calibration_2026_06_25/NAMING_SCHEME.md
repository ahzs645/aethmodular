# Calibration variant naming scheme

Decision #6 from the 2026-06-25 meeting: adopt **one consistent name** per calibration variant so
the many experiments stay trackable across notebooks, figures, tables, and the deck. Use this string
as the `calib_id` everywhere (file names, plot legends, table columns).

## Format
```
<analyte>_<trainset>_<filter>_<op>_lot<NN>_k<comp>
```

| Field | Meaning | Allowed values (extend as needed) |
|-------|---------|-----------------------------------|
| `analyte`  | what is predicted | `EC`, `OC` |
| `trainset` | training population | `all` (every sample), `biomass` (smoke classifier = yes), `below11` (only samples below the 1:1 Tor line), `removed` (only the previously-removed samples) |
| `filter`   | sample-filtering rule | `nofilt` (keep all), `absres` (old absolute-residual removal — the over-filtered baseline), `ecthr70` (drop EC < 70 µg), `ecthrX` (Ethiopia EC range ÷ 10), `relres` (relative-residual removal) |
| `op`       | pyrolysis convention (char/soot only) | `opr` (reflectance / OPTR — the Tor default), `opt` (transmittance / OPTT) |
| `lot`      | FTIR lot | `lot251`, `lot248`, … |
| `k`        | # PLS components | `k17`, `k4`, `kmin` (first major RMSECV minimum, auto) |

Omit fields that don't apply (e.g. `op` only matters where char/soot enters).

## Examples
| `calib_id` | Reads as |
|------------|----------|
| `EC_all_nofilt_lot251_kmin`        | EC, all samples, **no filtering**, lot 251, first-min components — *the "don't over-filter" baseline* |
| `EC_all_absres_lot251_k17`         | EC, all samples, **old absolute-residual removal**, 17 comps — *the "beautiful but too perfect" cal being replaced* |
| `EC_biomass_nofilt_lot251_kmin`    | EC, **biomass-only** training, no filtering — *the "FTIR misses char" test* |
| `EC_below11_nofilt_lot251_kmin`    | EC trained **only on below-1:1 samples** — *"the weird samples may be the point"* |
| `EC_all_ecthr70_lot251_kmin`       | EC, all samples, **drop EC < 70 µg** low-signal filters |
| `charEC_opr` / `charEC_opt`        | char-EC using reflectance vs. transmittance OP (Adama, notebook 03) |

## Where it shows up
- **Figures:** `figures/fig0X_<calib_id>.png`
- **Tables:** column header `<calib_id>` (e.g. predicted-EC columns side by side)
- **Legends:** the `calib_id` verbatim, so a plot is self-documenting

Keep a one-line log below as variants are actually built:

| Built on | `calib_id` | Notebook | Notes |
|----------|-----------|----------|-------|
| 2026-06-25 | `charEC_opr`, `charEC_opt` | `03` | Adama char/soot both OP conventions |
| _tbd_ | `EC_all_nofilt_lot251_kmin` | `05` | first "keep everything" rebuild |
