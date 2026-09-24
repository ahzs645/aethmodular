"""Execute unchanged ftir_13 in a staged checkout and export reproduction evidence.

Run through uv in the isolated worktree, never against the historical outputs.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import runpy
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkout',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    checkout=args.checkout.resolve();original=args.original.resolve();out=args.output.resolve()
    if checkout==original or not (checkout/'.git').is_file():raise ValueError('Require an isolated Git worktree')
    if out.exists():raise ValueError('Use a new evidence output directory')
    out.mkdir(parents=True);cwd=checkout/'research/ftir_ec_phase3';os.chdir(cwd)
    target=cwd/'scripts/run_ftir_13.py';before=sha(target)
    reads={};busy=False
    def audit(event,parameters):
        nonlocal busy
        if event!='open' or busy:return
        path,mode,flags=parameters
        if not isinstance(path,(str,bytes,os.PathLike)):return
        if isinstance(path,bytes):return
        if mode and any(c in mode for c in 'wax+'):return
        if flags & (os.O_WRONLY|os.O_RDWR):return
        busy=True
        try:
            p=Path(path).resolve()
            if (p.is_relative_to(checkout) or 'GoogleDrive-' in str(p)) and p.suffix in {'.csv','.npz','.npy','.pkl','.py','.json'} and p.is_file() and not p.is_relative_to(cwd/'output/tables/ftir13'):
                reads.setdefault(str(p),sha(p))
        finally:busy=False
    sys.addaudithook(audit)
    status='failed';error=None
    try:
        ns=runpy.run_path(str(target),run_name='__reproduction__')
        status='executed'
        splits=[];metrics=[];curves=[]
        cohorts=[('lowest-OCEC 800',ns['ocec']['AnalysisId'].to_numpy(),ns['ocec_y'],ns['ocec_sites'],ns['ocec_fits'])]
        for cohort,ids,y,sites,fits in cohorts:
            train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=ns['SPLIT_SEED']).split(ids,groups=sites))
            assert set(sites[train]).isdisjoint(sites[test])
            for partition,idx in [('train',train),('test',test)]:
                splits.extend({'cohort':cohort,'AnalysisId':int(ids[i]),'Site':str(sites[i]),'partition':partition,'y':float(y[i])} for i in idx)
            for df1,(model,k,curve,heldout) in fits.items():
                metrics.append({'cohort':cohort,'df1':df1,'k':int(k),**heldout})
                curves.append(curve.assign(df1=df1))
                pred=model.predict(ns['corrected_pool_rows'](ids,df1)).ravel()
                pd.DataFrame({'AnalysisId':ids,'Site':sites,'y':y,'prediction':pred,'partition':np.where(np.isin(np.arange(len(ids)),test),'test','train')}).to_csv(out/f'ocec_df{df1}_predictions.csv',index=False)
        pd.DataFrame(splits).to_csv(out/'split_membership.csv',index=False)
        pd.DataFrame(metrics).to_csv(out/'heldout_metrics.csv',index=False)
        pd.concat(curves).to_csv(out/'component_curves.csv',index=False)
        pd.DataFrame([{'df1':d,'k':int(f[1])} for d,f in ns['smoke_fits'].items()]).to_csv(out/'smoke_components.csv',index=False)
        comparisons=[]
        expected=original/'research/ftir_ec_phase3/output/tables/ftir13'
        for p in sorted(expected.glob('*.csv')):
            a=pd.read_csv(p);b=pd.read_csv(cwd/'output/tables/ftir13'/p.name)
            same_structure=list(a.columns)==list(b.columns) and a.shape==b.shape
            for col in a.columns:
                if not same_structure:break
                if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                    av=a[col].to_numpy(float);bv=b[col].to_numpy(float);match=bool(np.allclose(av,bv,rtol=0,atol=1e-6,equal_nan=True));finite=np.isfinite(av)&np.isfinite(bv);delta=float(np.max(np.abs(av[finite]-bv[finite]))) if finite.any() else 0.
                else:match=a[col].fillna('<NA>').equals(b[col].fillna('<NA>'));delta=None
                comparisons.append({'file':p.name,'column':col,'shape_matches':same_structure,'matches_at_1e_6':match,'max_abs_difference':delta})
            if not same_structure:comparisons.append({'file':p.name,'column':'<structure>','shape_matches':False,'matches_at_1e_6':False})
        pd.DataFrame(comparisons).to_csv(out/'table_comparison.csv',index=False)
        status='reproduced' if comparisons and all(c['matches_at_1e_6'] for c in comparisons) and ns['ocec_fits'][6][1]==5 else 'executed_with_differences'
    except Exception as e:
        error=repr(e);traceback.print_exc()
    finally:
        sys.audit('reproduction.end')
        changed=[p for p,h in reads.items() if sha(p)!=h]
        modules={str(Path(m.__file__).resolve()):sha(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(checkout)) and Path(m.__file__).suffix=='.py'}
        result={'status':status,'error':error,'executed_at_utc':datetime.now(timezone.utc).isoformat(),'checkout':str(checkout),'runner_sha256':before,'runner_unchanged':sha(target)==before,'runtime_reads':reads,'changed_inputs':changed,'imported_source_hashes':modules,'original_manifest_sha256':sha(original/'docs/openresearch-retrospective/experiments/airspec-inputs.json'),'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'scope':'Unchanged historical runner and saved corrected caches; new-environment numerical reproduction, not independent Addis validation. Historical plot styling/OLS annotations preserved as reproduction artifacts.'}
        (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['runtime_reads','imported_source_hashes']},indent=2))
    if status!='reproduced' or changed:raise SystemExit(1)

if __name__=='__main__':main()
