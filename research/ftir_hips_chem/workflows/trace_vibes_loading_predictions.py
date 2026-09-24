"""Trace frozen prediction differences without fitting, tuning or excluding cases.

The symmetric algebraic decomposition separates corrected-spectrum changes,
coefficient changes, and the intercept/centering constant. It is diagnostic,
not a causal attribution or an independent evaluation of method accuracy.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
AREA=ROOT/'research/ftir_hips_chem'
sys.path.insert(0,str(AREA/'scripts'))
from vibes_error_audit import pair_predictions, loading_bands

BANDS=((1425,1800),(1800,2500),(2500,3000),(3000,4001))


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def decompose(xa,xv,ba,bv,ca,cv):
    """Exact symmetric split; neither method is privileged as the baseline."""
    spectra=(xv-xa)*(bv+ba)/2
    coefficients=(xv+xa)*(bv-ba)/2
    return spectra,coefficients,float(cv-ca)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Verify input hashes and saved predictions only; no diagnostic outputs')
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=AREA/'output/tables/vibes_loading_trace')
    args=parser.parse_args()
    freeze=json.loads(args.freeze.read_text())
    required=freeze['inputs']
    if not required or any(not i.get('sha256') for i in required):raise ValueError('Every input must have a frozen hash')
    for item in required:
        if sha(Path(item['path']))!=item['sha256']:raise ValueError(f"Input changed: {item['path']}")
    source=Path(freeze['source_directory'])
    consumed={'case_audit.csv','heldout_predictions.csv','wn.npy','corrected_AIRSpec.npy','corrected_VIBES.npy'}
    consumed.update(f'pls_{cohort}_{method}.npz' for cohort in ['full_pool','locked800'] for method in ['AIRSpec','VIBES'])
    frozen_paths={Path(i['path']).resolve() for i in required}
    if not {(source/name).resolve() for name in consumed}.issubset(frozen_paths):
        raise ValueError('Freeze manifest omits a consumed data/model input')
    out=args.output.resolve()
    if source==out or out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('Output must be separate from frozen source')
    cases=pd.read_csv(source/'case_audit.csv')
    pred=pd.read_csv(source/'heldout_predictions.csv')
    paired=pair_predictions(pred,cases)
    wn=np.load(source/'wn.npy',allow_pickle=False)
    arrays={m:np.load(source/f'corrected_{m}.npy',mmap_mode='r',allow_pickle=False) for m in ['AIRSpec','VIBES']}
    if any(x.shape!=(len(cases),len(wn)) for x in arrays.values()):raise ValueError('Spectral shape/order contract violated')
    masks=[(wn>=low)&(wn<high) for low,high in BANDS]
    if not np.equal(np.sum(masks,axis=0),1).all():raise ValueError('Bands must partition the entire saved grid')
    index=pd.Series(np.arange(len(cases)),index=cases.sample_id)
    checks=[]; outputs=[]
    for cohort,frame in paired.groupby('cohort',sort=True):
        frame=frame.copy()
        selected=cases.kind.eq('calibration') & cases.paired_valid & cases.split.eq('train')
        if cohort=='locked800':selected &= cases.locked800
        frame['loading_band'],cuts=loading_bands(cases.loc[selected,'y'],frame.y)
        positions=index.loc[frame.sample_id].to_numpy()
        xs={m:np.asarray(x[positions],dtype=float) for m,x in arrays.items()}
        beta={};constant={};rounding={}
        for method in arrays:
            with np.load(source/f'pls_{cohort}_{method}.npz',allow_pickle=False) as model:
                beta[method]=model['coefficient'].reshape(-1)
                mean=model['x_mean']
                intercept=float(model['intercept'].item())
                constant[method]=float(intercept-mean@beta[method])
            # sklearn predict centers in the input dtype (saved arrays: float32).
            centered=np.array(arrays[method][positions],copy=True)
            centered-=mean
            reconstruction=centered@beta[method]+intercept
            ideal=xs[method]@beta[method]+constant[method]
            rounding[method]=reconstruction-ideal
            maxerr=float(np.max(np.abs(reconstruction-frame[method])))
            if maxerr>1e-8:raise ValueError(f'Saved prediction mismatch: {cohort}/{method}: {maxerr}')
            checks.append({'cohort':cohort,'method':method,'n':len(frame),'max_abs_prediction_error':maxerr,'training_quartiles':cuts.tolist(),'input_dtype':str(arrays[method].dtype),'max_abs_centering_roundoff':float(np.max(np.abs(rounding[method])))})
        if args.check:continue
        spectral,coefficient,offset=decompose(xs['AIRSpec'],xs['VIBES'],beta['AIRSpec'],beta['VIBES'],constant['AIRSpec'],constant['VIBES'])
        expected=frame.VIBES-frame.AIRSpec
        precision_delta=rounding['VIBES']-rounding['AIRSpec']
        reconstructed=spectral.sum(axis=1)+coefficient.sum(axis=1)+offset+precision_delta
        if not np.allclose(reconstructed,expected,rtol=0,atol=1e-8):raise ValueError('Decomposition does not close')
        frame['prediction_delta']=expected
        frame['centering_precision_delta']=precision_delta
        frame['delta_squared_error']=(frame.VIBES-frame.y)**2-(frame.AIRSpec-frame.y)**2
        frame['intercept_and_centering_delta']=offset
        for (low,high),mask in zip(BANDS,masks):
            frame[f'spectra_{low}_{high-1}']=spectral[:,mask].sum(axis=1)
            frame[f'coefficients_{low}_{high-1}']=coefficient[:,mask].sum(axis=1)
        outputs.append(frame)
    receipt={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'no_refit':True,
        'no_new_exclusions':True,'freeze_sha256':sha(args.freeze),'workflow_sha256':sha(Path(__file__)),
        'interpretation':'Algebraic explanation of saved predictions; post-hoc, conditional on fitted models. No independent validation or physical-causality claim.'}
    if args.check:
        print(json.dumps(receipt,indent=2));return
    if out.exists():raise ValueError('Use a new output directory; do not overwrite a prior diagnostic')
    out.mkdir(parents=True)
    all_rows=pd.concat(outputs,ignore_index=True)
    all_rows.to_csv(out/'all_heldout_contributions.csv',index=False)
    # This is a declared inspection subset, never an exclusion from evaluation.
    focus=all_rows.loc[all_rows.cohort.eq('full_pool') & all_rows.loading_band.eq('Q3')]
    selected=[]
    for site in ['BRIS1','CACR1']:
        group=focus.loc[focus.Site.eq(site)]
        for direction,part in [('worse',group.loc[group.delta_squared_error>0].sort_values(['delta_squared_error','sample_id'],ascending=[False,True])),('better',group.loc[group.delta_squared_error<0].sort_values(['delta_squared_error','sample_id'],ascending=[True,True]))]:
            selected.append(part.head(3).assign(inspection_direction=direction))
    pd.concat(selected,ignore_index=True).to_csv(out/'inspection_cases.csv',index=False)
    columns=[c for c in all_rows if c.startswith(('spectra_','coefficients_'))]+['prediction_delta','intercept_and_centering_delta','centering_precision_delta','delta_squared_error']
    all_rows.groupby(['cohort','loading_band'],sort=True)[columns].mean().to_csv(out/'mean_contributions_by_band.csv')
    # Verify source identity after processing as well as before it.
    for item in required:
        if sha(Path(item['path']))!=item['sha256']:raise ValueError('Frozen source changed during diagnostic')
    receipt['status']='completed_local_diagnostic';receipt['n_test_rows']=len(all_rows);receipt['n_full_pool_q3']=len(focus)
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'report.md').write_text('# VIBES loading trace\n\n'+receipt['interpretation']+'\n\nAll saved held-out predictions and the symmetric decomposition (including the separately exported float32 centering contribution) reconcile within 1e-8 µg/filter. All held-out cases remain in the export. See `inspection_cases.csv` for the predeclared BRIS1/CACR1 Q3 case review; these cases were selected after test inspection and cannot validate an improvement.\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
