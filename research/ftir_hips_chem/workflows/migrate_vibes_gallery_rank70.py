"""Rebuild gallery dependencies on completed rank-70 spectra, preserving archives.

Run with uv run python research/ftir_hips_chem/workflows/migrate_vibes_gallery_rank70.py STAGE.
Stages: meeting, grid, export-meeting, followup, export-followup. Long grid runs resume only on identical inputs.
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / 'scripts'))
from vibes_current import CURRENT, LABELS, MIGRATION, MODEL_NAMES, load_context, prepare_shared_run, receipt, study_inputs  # noqa: E402


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['meeting', 'grid', 'export-meeting', 'followup', 'export-followup'])
    args = ap.parse_args()
    if args.stage in ('followup', 'export-followup'):
        inputs = study_inputs()
        out = MIGRATION / 'followup'
        receipt(out, inputs['hashes'])
        studies = module(AREA / 'workflows/vibes_followup_studies.py', 'current_studies')
        if args.stage == 'followup':
            studies.write_manifest(inputs, out)
            if not (out / 'common_test_metrics.csv').exists():
                studies.common_test(inputs, out)
            if not (out / 'routing_metrics.csv').exists():
                studies.routing_study(inputs, out)
            studies.addis_gate({**inputs, 'source': prepare_shared_run()}, out)
            return
        old_export = module(ROOT / 'gallery/data/export_baseline_followup.py', 'followup_checks')
        metrics = pd.read_csv(out / 'common_test_metrics.csv')
        predictions = pd.read_csv(out / 'common_test_predictions.csv')
        test_ids = set(inputs['cases'].sample_id.iloc[inputs['test']])
        for row in metrics.to_dict('records'):
            part = predictions.loc[predictions.model.eq(row['model']) & predictions.method.eq(row['method'])]
            if set(part.sample_id) != test_ids or part.sample_id.duplicated().any():
                raise ValueError('Current common-test predictions are not paired')
            old_export.check_score(part, pd.Series(row), row['model'])
        routing = pd.read_csv(out / 'routing_metrics.csv')
        for population, name in [('training_site_nested', 'routing_nested_training_predictions.csv'), ('prior_inspected_test', 'routing_prior_test_predictions.csv')]:
            frame = pd.read_csv(out / name)
            np.testing.assert_allclose(frame.routed, np.where(frame.choice.eq('VIBES'), frame.VIBES, frame.AIRSpec))
            for method in ('AIRSpec', 'VIBES', 'routed'):
                row = routing.loc[routing.population.eq(population) & routing.method.eq(method)].iloc[0]
                old_export.check_score(frame[['y', method]].rename(columns={method: 'prediction'}), row, population)
        cases = inputs['cases']
        payload = dict(schema_version=1, background_rank=70, source_run_signature=inputs['manifest']['signature'],
                       scope='new_exploratory_followup_on_saved_corrections', common_test_n=len(test_ids),
                       common_test_sites=int(cases.Site.iloc[inputs['test']].nunique()), train_n=len(inputs['train']),
                       train_sites=int(cases.Site.iloc[inputs['train']].nunique()), selection_metrics=old_export.rows(metrics),
                       selection_overlap=json.loads((out / 'selection_overlap.json').read_text()),
                       restricted_sensitivity=old_export.rows(pd.read_csv(out / 'restricted_membership_sensitivity.csv')),
                       routing_metrics=old_export.rows(routing),
                       routing_extreme=old_export.rows(pd.read_csv(out / 'routing_largest_VIBES_errors.csv').head(1)),
                       addis_readiness=json.loads((out / 'addis_readiness.json').read_text()), source_sha256=inputs['hashes'])
        dest = ROOT / 'gallery/app/public/data'
        (dest / 'baseline_followup_r70.json').write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')))
        for name in old_export.TABLES:
            shutil.copy2(out / name, dest / f'baseline_followup_r70_{name}')
        return
    if args.stage == 'export-meeting':
        ctx, _ = load_context()
        grid = module(AREA / 'workflows/run_meeting_grid_20260917.py', 'verify_grid')
        expected = {grid.key(m, cat, cut, mode) for m, cat, cut, mode, _ in grid.configs(ctx)}
        rows = [json.loads(line) for line in (MIGRATION / 'meeting/category_grid.jsonl').read_text().splitlines()]
        actual = [row['key'] for row in rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'Grid incomplete or duplicated: {len(actual)} rows; {len(expected)} required')
        if not json.loads((MIGRATION / 'meeting/summary.json').read_text()).get('background_rank') == 70:
            raise ValueError('Seasonal analysis is not the completed rank-70 version')
        ex = module(ROOT / 'gallery/data/export_meeting_followup.py', 'current_meeting_export')
        ex.SRC = MIGRATION / 'meeting'
        ex.OUT = ROOT / 'gallery/app/public/data/meeting-r70'
        ex.OUT.mkdir(parents=True, exist_ok=True)
        ex.METHOD_NAMES.update(LABELS)
        ex.main()
        return
    ctx, data = load_context()
    out = MIGRATION / 'meeting'
    receipt(out, data['sources'])
    if args.stage == 'grid':
        grid = module(AREA / 'workflows/run_meeting_grid_20260917.py', 'current_grid')
        grid.OUT, grid.RESULTS = out, out / 'category_grid.jsonl'
        # Spline inputs are unchanged; reuse their exact saved grid rows only.
        if not grid.RESULTS.exists():
            old = AREA / 'output/tables/meeting_followup_20260917/category_grid.jsonl'
            rows = [line for line in old.read_text().splitlines() if json.loads(line)['method'] == 'AIRSpec']
            grid.RESULTS.write_text('\n'.join(rows) + '\n')
        grid.ab.load_context = lambda: ctx
        grid.main()
        return
    flow = module(AREA / 'workflows/run_meeting_followup_20260917.py', 'current_meeting')
    flow.OUT = out
    flow.ab.load_context = lambda: ctx
    original_model = flow.frozen_full_pool

    def current_model(method):
        if method not in MODEL_NAMES:
            return original_model(method)
        z = np.load(CURRENT / f'benchmark/pls_full_pool_{MODEL_NAMES[method]}.npz')
        coef, xm, intercept = z['coefficient'].ravel(), z['x_mean'], float(z['intercept'][0])
        return lambda X: (np.asarray(X, float) - xm) @ coef + intercept, int(z['components'])

    flow.frozen_full_pool = current_model
    with threadpool_limits(limits=2):
        flow.main()
    path = out / 'summary.json'
    summary = json.loads(path.read_text())
    summary.update(background_rank=70, method_labels=LABELS, source_sha256=data['sources'],
                   spec_curve_export='meeting-r70/spec_curve_20260917.json')
    path.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
