# Next Tests: IMPROVE/FED vs Addis/SPARTAN

Clean notebook workspace for the narrowed analysis:

1. Test whether Addis/ETAD is outside the normal IMPROVE mass-loading and
   fAbs-derived optical-loading envelope.
2. Identify rare IMPROVE samples that occupy Addis-like fAbs, EC mass, and EC
   surface-loading regimes.
3. Treat FED `Ref*` / `Trans*` fields only as auxiliary TOR/carbon-analyzer
   laser R/T diagnostics.
4. Prepare a compact candidate table for external raw HIPS data requests.

The starter notebook is:

- `00_fed_addis_loading_envelope.ipynb`

Plotting rule: relevant scatter/envelope plots should show origin `(0, 0)`.
Use the notebook helper `set_origin_zero(ax, x, y)` for this.

