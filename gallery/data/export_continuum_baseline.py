"""The 2-3 Nov continuum / baseline study for the gallery's Experimental tab (key 'continuum').

uv run python gallery/data/export_continuum_baseline.py

Reads the tables of research/ftir_hips_chem/workflows/run_transfer_diagnosis_20261102.py (stages fit, tables, inject,
perfilter; the continuum-protected VIBES columns come from workflows/vibes_continuum_20261102.py on Colab) and writes
public/data/experimental/continuum_baseline.json: per-filter Addis predictions for the before/after crossplots, held-out
IMPROVE filters with HIPS > 10 µg, the matched-loading ratios, the continuum by loading and its noise floor, the VIBES
injection test, every site against the TOR yardstick and the window attribution. No refit here, except the saved
Addis Deming summaries (same λ as the other Addis pages).

Also reads workflows/run_intercept_fixes_20261103.py (stages attrib, fit, tables, aeth): the label-free intercept fixes by
PLS k, the window split of the Addis day-to-day gain, and the MA350 aethalometer check (Addis filters' IR BCc and the
four aethalometer sites), Deming λ 1 against the aethalometer, and stage hips: the HIPS loading-saturation test, the
saturation constant fitted without Ethiopia and Addis against HIPS linearised with it; stage stress: what that reading
survives, HIPS optical depth against aethalometer loading with Addis held out, the Addis level anchors and seasons.
And workflows/run_hips_response_20261103.py: aethalometer flow and single-spot checks, the HIPS response over SPARTAN and
IMPROVE, the Adama TOR test with the pass-rule simulation, and the baseline scoreboard.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/ftir_hips_chem/output/tables/transfer_diagnosis_20261102'
VC = ROOT / 'research/ftir_hips_chem/output/tables/vibes_continuum_20261102'
FIX = ROOT / 'research/ftir_hips_chem/output/tables/intercept_fixes_20261103'   # workflows/run_intercept_fixes_20261103.py
LAMBDA_AETH = 1.0     # aethalometer error unknown: equal error variances, as in the workflow
# label-free fixes tried on 3 Nov (trained on IMPROVE only; di-PLS and iw also see the unlabeled Addis spectra)
FIX_MODELS = ['epo k18', 'dipls_g10 k18', 'dipls_net k18', 'mask_org k18', 'cont_only k10', 'iw k18']
SWEEP_MODELS = {'vibes': 'vibes', 'vibes_anchorhp_full': 'vibes_anchorhp', 'epo': 'epo', 'dipls_g10': 'dipls_g10',
                'dipls_net': 'dipls_net', 'mask_org': 'mask_org', 'cont_only': 'cont_only', 'iw': 'iw'}
OUT = ROOT / 'gallery/app/public/data/experimental/continuum_baseline.json'
LAMBDA_ADDIS = 2.96   # σ from HIPS_Uncertainty (0.308 µg/m³) and the σy proxy 0.531, at MAC 10; as the other Addis pages
MODELS = ['published', 'raw k18', 'spline k18', 'vibes k18', 'anchored k18', 'vibes_anchorhp k18', 'vibes_deflate k18',
          'anchored_noCH k18', 'anchored_balanced k12']
# Ethiopian Meteorological Institute calendar (belg_feb): Bega Oct-Jan, Belg Feb-May, Kiremt Jun-Sep
SEASON = {**{m: 'Bega' for m in (10, 11, 12, 1)}, **{m: 'Belg' for m in (2, 3, 4, 5)}, **{m: 'Kiremt' for m in (6, 7, 8, 9)}}


def cols(df: pd.DataFrame, digits: int = 4) -> dict[str, list]:
    def clean(v):
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else round(float(v), digits)
        if isinstance(v, (np.integer, np.bool_)):
            return v.item()
        return v
    return {c: [clean(v) for v in df[c].tolist()] for c in df.columns}


def deming(x, y, lam):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy, sxy = xc @ xc / (len(x) - 1), yc @ yc / (len(x) - 1), xc @ yc / (len(x) - 1)
    d = syy - lam * sxx
    slope = (d + np.sqrt(d ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return slope, y.mean() - slope * x.mean(), np.corrcoef(x, y)[0, 1] ** 2, int(ok.sum())


def main():
    a = pd.read_csv(RUN / 'addis_predictions.csv')
    a['season'] = pd.to_datetime(a.date).dt.month.map(SEASON)
    a['hips_bc'] = a.fabs_Mm1 / 10                         # µg/m³ at MAC 10; the page rescales to the reader's MAC
    a['tor_equiv'] = a.hips_bc * a.exp_tor_hips              # what IMPROVE TOR would read at this loading and OC/BC
    a['published'] = a.ec_published_ugm3
    models = [m for m in MODELS if m in a.columns]
    fx = pd.read_csv(FIX / 'addis_predictions.csv')
    fix_models = [m for m in FIX_MODELS if m in fx.columns]
    a = a.merge(fx[['filter_id', 'aeth_ir'] + fix_models], on='filter_id', how='left')
    # HIPS on the response fitted to IMPROVE TOR alone (run_hips_response_20261103.py stage network); MAC 10 like hips_bc
    hl = pd.read_csv(FIX / 'addis_hips_response.csv').set_index('filter_id').hips_resp
    a['hips_lin'] = a.filter_id.map(hl)
    k0 = ~a.excluded & a.hips_bc.notna() & a.hips_lin.notna()
    lambda_lin = float(LAMBDA_ADDIS / np.mean((a.hips_lin / a.hips_bc)[k0] ** 4))   # HIPS error grows with the correction
    filters = a[['filter_id', 'date', 'season', 'lot', 'hips_bc', 'hips_lin', 'tor_equiv', 'exp_tor_hips', 'aeth_ir', 'excluded'] + models + fix_models]

    fits = []
    keep = ~a.excluded & a.hips_bc.notna()
    for m in models + fix_models:
        for xname in ('hips_bc', 'tor_equiv', 'aeth_ir', 'hips_lin'):
            s = keep & a[m].notna() & a[xname].notna()
            lam = {'aeth_ir': LAMBDA_AETH, 'hips_lin': lambda_lin}.get(xname, LAMBDA_ADDIS)
            sl, it, r2, n = deming(a.loc[s, xname], a.loc[s, m], lam)
            fits.append(dict(model=m, x=xname, n=n, slope=sl, intercept=it, r2=r2,
                             ratio=float(np.median(a.loc[s, m] / a.loc[s, xname]))))

    i = pd.read_csv(RUN / 'improve_heavy_predictions.csv')
    imodels = [m for m in models if m in i.columns]
    improve = i[['filter_id', 'site', 'date', 'tor_ug', 'hips_ug', 'ocec'] + imodels]

    r = pd.read_csv(RUN / 'ratios_by_loading.csv')
    mae = r[r.what == 'IMPROVE out-of-site MAE (µg)'][['model', 'k', 'median']].rename(columns={'median': 'mae'})
    curves = r[r.what.isin(['IMPROVE low OC/EC FTIR/HIPS', 'IMPROVE FTIR/HIPS', 'IMPROVE low OC/EC FTIR/TOR', 'Addis FTIR/HIPS'])
               & r.k.isin([0, 18])].copy()
    lohi = curves.hips_bin.str.strip('(]').str.split(',', expand=True).astype(float)
    curves['lo'], curves['hi'] = lohi[0], lohi[1]
    curves = curves[['model', 'k', 'what', 'lo', 'hi', 'median', 'n']]

    cb = pd.read_csv(RUN / 'continuum_by_loading.csv')
    lohi = cb.hips_bin.str.strip('(]').str.split(',', expand=True).astype(float)
    cb['lo'], cb['hi'] = lohi[0], lohi[1]
    cb = cb[['prep', 'window', 'group', 'lo', 'hi', 'median', 'n']]

    inj = pd.read_csv(RUN / 'vibes_injection.csv').groupby(['variant', 'shape']).value.median().reset_index()
    noise = pd.read_csv(RUN / 'continuum_noise.csv')
    sites = pd.read_csv(RUN / 'sites_vs_tor_yardstick.csv')
    windows = pd.read_csv(RUN / 'window_attribution.csv')
    fitrec = json.loads((RUN / 'fit_receipt.json').read_text())
    vc = json.loads((VC / 'bundle/MANIFEST.json').read_text()) if (VC / 'bundle/MANIFEST.json').exists() else {}
    diag = pd.read_csv(VC / 'fit_diagnostics.csv') if (VC / 'fit_diagnostics.csv').exists() else None

    sw = pd.read_csv(FIX / 'summary.csv')
    sw = sw[sw.model.isin(list(SWEEP_MODELS))].assign(model=lambda d: d.model.map(SWEEP_MODELS))
    sweep = sw[['model', 'k', 'improve_mae', 'improve_heavy_lowocec_ratio', 'slope_hips_bc', 'intercept_hips_bc', 'ratio_hips_bc',
                'slope_tor_equiv', 'intercept_tor_equiv', 'ratio_tor_equiv', 'slope_aeth', 'intercept_aeth', 'intercept_aeth_rel',
                'loglog_aeth']]
    gain = pd.read_csv(FIX / 'window_gain.csv')
    aeth_sites = pd.read_csv(FIX / 'aeth_sites.csv')
    fixrec = json.loads((FIX / 'fit_receipt.json').read_text())
    hipsrec = json.loads((FIX / 'hips_receipt.json').read_text())
    hips_volume = pd.read_csv(FIX / 'hips_volume_test.csv')
    hips_sat = pd.read_csv(FIX / 'hips_saturation.csv')
    hips_sites = pd.read_csv(FIX / 'hips_sites_loglog.csv')
    hips_grid = pd.read_csv(FIX / 'addis_vs_linearised_hips.csv').replace([np.inf], 1e6)
    stress = json.loads((FIX / 'stress_receipt.json').read_text())
    hb = a.loc[keep, 'hips_bc']
    ae = a.loc[keep & a.aeth_ir.notna()]
    sl, it, r2, n = deming(ae.aeth_ir, ae.hips_bc, LAMBDA_AETH)
    fits.append(dict(model='HIPS BC', x='aeth_ir', n=n, slope=sl, intercept=it, r2=r2, ratio=float(np.median(ae.hips_bc / ae.aeth_ir))))

    payload = dict(
        schema_version=1, generated=datetime.now(timezone.utc).isoformat(), source=str(RUN.relative_to(ROOT)),
        settings=dict(LAMBDA_ADDIS=LAMBDA_ADDIS, LAMBDA_AETH=LAMBDA_AETH, models=models, fix_models=fix_models, k=18,
                      hips_mean=float(hb.mean()), fix=fixrec, hips=hipsrec, stress=stress, LAMBDA_HIPS_LIN=lambda_lin, response=json.loads((FIX / 'hips_response_receipt.json').read_text()), tor=json.loads((FIX / 'tor_receipt.json').read_text()), flow_tests=json.loads((FIX / 'aeth_flow_tests.json').read_text()), fit=fitrec, calendar='belg_feb',
                      colab=dict(fits=int(len(diag)) if diag is not None else None,
                                 failed=int((~diag.success.astype(bool)).sum()) if diag is not None else None,
                                 spectra=vc.get('n_rows'), vm='Colab G4 (48 vCPUs; VIBES is CPU-bound, the GPU idles)',
                                 minutes=34)),
        addis_filters=cols(filters), addis_fits=cols(pd.DataFrame(fits)), improve_filters=cols(improve, 3),
        mae=cols(mae), curves=cols(curves), continuum=cols(cb, 5), noise=cols(noise), injection=cols(inj),
        sites=cols(sites), windows=cols(windows),
        sweep=cols(sweep), gain=cols(gain), aeth_sites=cols(aeth_sites),
        hips_volume=cols(hips_volume), hips_sat=cols(hips_sat), hips_sites=cols(hips_sites), hips_grid=cols(hips_grid),
        hips_stress=cols(pd.read_csv(FIX / 'hips_volume_stress.csv')), hips_by_site=cols(pd.read_csv(FIX / 'hips_volume_by_site.csv')),
        tau_points=cols(pd.read_csv(FIX / 'hips_tau_points.csv')), tau_holdout=cols(pd.read_csv(FIX / 'hips_tau_holdout.csv')),
        aeth_bridge=cols(pd.read_csv(FIX / 'aeth_bridge.csv')), true_over_hips=cols(pd.read_csv(FIX / 'addis_true_over_hips.csv')),
        season_aeth=cols(pd.read_csv(FIX / 'addis_season_vs_aeth.csv')),
        aeth_flow=cols(pd.read_csv(FIX / 'aeth_flow_by_site.csv')), aeth_flow_coef=cols(pd.read_csv(FIX / 'aeth_flow_coef.csv')), aeth_fix=cols(pd.read_csv(FIX / 'aeth_fix_pasadena.csv')), aeth_spots=cols(pd.read_csv(FIX / 'aeth_spot_sensitivity.csv')),
        tau_curves=cols(pd.read_csv(FIX / 'hips_tau_curves.csv')), response_fit=cols(pd.read_csv(FIX / 'hips_response_fit.csv')),
        response_addis=cols(pd.read_csv(FIX / 'addis_vs_hips_response.csv')), response_sites=cols(pd.read_csv(FIX / 'hips_site_response.csv')),
        adama=cols(pd.read_csv(FIX / 'adama_tor_test.csv')), adama_summary=cols(pd.read_csv(FIX / 'adama_tor_summary.csv')),
        rule_sim=cols(pd.read_csv(FIX / 'pass_rule_simulation.csv')), scoreboard=cols(pd.read_csv(FIX / 'baseline_scoreboard.csv')),
        spectra_story=cols(pd.read_csv(FIX / 'addis_spectra_story.csv'), 5), lot_vs_aeth=cols(pd.read_csv(FIX / 'addis_lot_vs_aeth.csv')),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    print(f'  {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
