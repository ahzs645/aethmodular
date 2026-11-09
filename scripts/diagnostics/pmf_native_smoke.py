"""Public/synthetic-only Windows tests of the official EPA PMF5 runtime.

EPA_PMF_RUNTIME points to the extracted official app directory. No private
FTIR inputs, executables, sealed scripts or license keys enter CI artifacts.
"""
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pmf.io import Dataset
from src.pmf.native_backend import fit_native, ingest_case


def synthetic(n, m, k, seed):
    rng = np.random.default_rng(seed)
    g = rng.uniform(.1, 2, (n, k))
    h = rng.uniform(.1, 1, (k, m))
    x = g @ h + rng.normal(0, .05, (n, m))
    x[0, 0] = -.01
    x[min(3,n-1), min(5,m-1)] += 2.
    return x, np.full_like(x,.1)


def main():
    runtime = Path(os.environ['EPA_PMF_RUNTIME']).resolve()
    root = Path(os.environ.get('PMF_NATIVE_TEST_OUT', 'native-public-test'))
    root.mkdir(parents=True,exist_ok=False)
    cases = {'signed_synthetic': (*synthetic(30,9,3,25),3),
             'spectral_width_246x1772': (*synthetic(246,1772,5,26),5)}
    public_dir = Path(os.environ['EPA_PMF_EXAMPLE_DATA'])
    def read(name):
        with (public_dir/name).open(newline='',encoding='utf-8-sig') as f:
            rows=list(csv.reader(f))
        return np.asarray([[float(v) for v in row[1:]] for row in rows[1:] if row])
    cases['public_st_louis']=(read('Dataset-StLouis-con.csv'),read('Dataset-StLouis-unc.csv'),6)
    results=[]
    for name,(x,u,k) in cases.items():
        for robust in (False,True):
            for seed in (1,10):
                case=f'{name}_{"robust" if robust else "ordinary"}_s{seed}'
                try:
                    loss = 'native-robust' if robust else 'quadratic'
                    result = fit_native(x, u, k, seed, root/case, runtime=runtime, loss=loss, timeout=300)
                    receipt = result['provenance']['native_receipt']
                    data = Dataset(x, u, np.asarray([f'row-{i}' for i in range(len(x))]),
                        np.asarray([f'feature-{j}' for j in range(x.shape[1])]),
                        {'status': 'unknown', 'validation_case': name, 'private_measurements': False})
                    imported = ingest_case(data, root/case, root/'reports'/case,
                        dataset_id=name, loss=loss)
                    if not imported['native_EPA_PMF_or_ME2_executed'] or imported['status'] != receipt['status']:
                        raise ValueError('Common workflow lost native execution status')
                    row={'case':case,'shape':list(x.shape),'status':receipt['status'],
                         **receipt['metrics'],'elapsed_seconds':receipt['elapsed_seconds'],
                         'verified_import': True, 'report': f'reports/{case}/report/index.html'}
                except Exception as exc:
                    row={'case':case,'shape':list(x.shape),'status':'failed','error':str(exc)}
                results.append(row)
                (root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
                print(json.dumps(row),flush=True)
    return int(any(r['status']!='passed' for r in results))


if __name__=='__main__':
    raise SystemExit(main())
