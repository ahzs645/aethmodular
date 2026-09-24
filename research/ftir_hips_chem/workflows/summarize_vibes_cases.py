"""Summarize the completed fixed-parameter case repeat; no fitting or tuning."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/ftir_hips_chem'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    out=AREA/'output/tables/vibes_case_investigation';source=AREA/'output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09'
    modelpath=source/'pls_full_pool_VIBES.npz';previous=json.loads((AREA/'output/tables/vibes_subgroup_audit/audit_manifest.json').read_text());assert sha(modelpath)==previous['source_hashes'][str(modelpath.relative_to(ROOT))]
    d=pd.read_csv(out/'case_evidence.csv')
    with np.load(out/'inspection_spectra.npz') as z:
        x=z['repeated_vibes'].astype(np.float32)
        with np.load(modelpath) as model:
            x-=model['x_mean'];pred=x@model['coefficient'].reshape(-1)+model['intercept'].item()
    d['repeat_prediction']=pred;d['repeat_prediction_difference']=pred-d.VIBES;d.to_csv(out/'repeat_prediction_comparison.csv',index=False)
    limit=float(abs(d.repeat_prediction_difference).max())
    (out/'report.md').write_text(f'''# VIBES loading-range case investigation

The two largest contributors are a shared prediction failure, intensified by VIBES. For BRIS1 filter 1970697, TOR EC is 4.749 µg/filter, AIRSpec predicts −12.979 and VIBES −23.107. For CACR1 filter 1973864, TOR is 3.777, AIRSpec predicts −7.326 and VIBES −13.879. Both filters are lot 251, outer-test samples outside the locked800 cohort.

These two filters account for about 52.2% of the full-pool Q3 squared-error increase. They remain in every reported evaluation; this is not an exclusion recommendation.

## Numerical repeatability
All 12 predeclared cases (three worse and three better at each site) converged in the original run without retries. All 12 converged again using the identical saved background and original tau, loss and iteration settings. The maximum repeated corrected-spectrum difference was 1.195e-5 absorbance, and the largest prediction shift was {limit:.6f} µg/filter, far below the large discrepancies above. This is strong evidence against a gross convergence failure explaining these two cases; convergence does not establish a physically correct baseline.

## What the spectra and coefficients show
The inspected plots show broad 3000–4000 cm⁻¹ structure, baseline differences across the grid and large opposing contributions from spectral changes and fitted coefficients. For the two leading cases, the spectral-only contributions summed over all four intervals are positive, while coefficient changes contribute a larger negative shift; the result is a more negative VIBES prediction. The symmetric decomposition is conditional on both fitted models and is not a causal chemical attribution.

The controls also exhibit baseline differences, so the presence of a spectral difference alone is not a failure criterion. Neither method's negative prediction can be interpreted as valid negative aerosol mass. Retain the predictions in the performance audit rather than clipping them after viewing test errors.

[BRIS1 spectral and coefficient panels](../../plots/vibes_case_investigation/BRIS1_inspection.png) · [CACR1 panels](../../plots/vibes_case_investigation/CACR1_inspection.png).

## Decision
Do not tune VIBES on these inspected test cases or delete them to improve the score. A defensible follow-up is a training-only applicability/shape analysis and independent blank/standard evidence for the broad spectral component. Any method adjustment requires new evaluation data. This investigation narrows the numerical explanation; it does not identify a chemical species or establish a superior model.

[Case evidence](case_evidence.csv) · [Repeat prediction comparison](repeat_prediction_comparison.csv) · [Repeated solver diagnostics](repeated_solver_diagnostics.csv) · [Frozen-source manifest](manifest.json) · [Arrays for all 12 cases](inspection_spectra.npz).
''')
    (out/'summary_provenance.json').write_text(json.dumps({'workflow_sha256':sha(__file__),'model_sha256':sha(modelpath),'inspection_arrays_sha256':sha(out/'inspection_spectra.npz'),'case_evidence_sha256':sha(out/'case_evidence.csv'),'maximum_repeat_prediction_difference':limit},indent=2)+'\n')
    print('Case report complete; max repeated prediction difference',limit)

if __name__=='__main__':main()
