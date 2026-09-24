"""Inspect the predeclared VIBES cases and repeat their corrections without tuning."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];AREA=ROOT/'research/ftir_hips_chem'
sys.path.insert(0,str(AREA/'scripts'))
from plotting import PlotConfig
from plotting.utils import style_axes
from vibes_baseline import VibesBackground, vibes_baseline_matrix


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    source=AREA/'output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09'
    trace=AREA/'output/tables/vibes_loading_trace';bundle=AREA/'output/tables/vibes_colab_bundle/stage'
    out=AREA/'output/tables/vibes_case_investigation';plots=AREA/'output/plots/vibes_case_investigation'
    if out.exists():raise ValueError('Preserve previous investigation; choose a new version')
    expected=json.loads((bundle/'BUNDLE_MANIFEST.json').read_text())
    run=json.loads((source/'RUN_MANIFEST.json').read_text())
    assert expected['content_hash']==run['bundle_hash']
    for name in ['data/pool_raw.npy','data/pool_metadata.csv','data/wn.npy']:
        assert sha(bundle/name)==expected['files_sha256'][name]
    previous=json.loads((AREA/'output/tables/vibes_subgroup_audit/audit_manifest.json').read_text())
    inputs=[trace/'inspection_cases.csv',trace/'all_heldout_contributions.csv',source/'case_audit.csv',source/'fit_diagnostics.csv',source/'wn.npy',source/'corrected_AIRSpec.npy',source/'corrected_VIBES.npy',source/'background_model.npz',source/'background_rank_cv.csv',source/'background_training_blanks.csv',source/'RUN_MANIFEST.json',bundle/'BUNDLE_MANIFEST.json',bundle/'data/pool_raw.npy',bundle/'data/pool_metadata.csv',bundle/'data/wn.npy']
    hashes={str(p):sha(p) for p in inputs}
    for p,h in hashes.items():
        rel=str(Path(p).relative_to(ROOT))
        if rel in previous['source_hashes']:assert previous['source_hashes'][rel]==h,p
    cases=pd.read_csv(source/'case_audit.csv');d=pd.read_csv(source/'fit_diagnostics.csv')
    chosen=pd.read_csv(trace/'inspection_cases.csv').merge(d,on='sample_id',suffixes=('','_diag'),validate='one_to_one')
    assert len(chosen)==12 and chosen.sample_id.is_unique
    meta=pd.read_csv(bundle/'data/pool_metadata.csv');raw=np.load(bundle/'data/pool_raw.npy',mmap_mode='r');wn=np.load(source/'wn.npy')
    assert np.array_equal(wn,np.load(bundle/'data/wn.npy'))
    native=cases.set_index('sample_id').loc[chosen.sample_id];positions=cases.reset_index().set_index('sample_id').loc[chosen.sample_id,'index'].to_numpy(int)
    rows=native.source_row.to_numpy(int)
    assert np.array_equal(meta.iloc[rows].FilterId.to_numpy(int),native.filter_id.to_numpy(int))
    assert np.array_equal(meta.iloc[rows].Site.to_numpy(),native.Site.to_numpy())
    assert np.allclose(meta.iloc[rows].TOR_EC_loading_ug,native.y)
    X=np.asarray(raw[rows],float)
    corrected={m:np.asarray(np.load(source/f'corrected_{m}.npy',mmap_mode='r')[positions],float) for m in ['AIRSpec','VIBES']}
    with np.load(source/'background_model.npz') as bg:
        background=VibesBackground(bg['wn'],bg['mean'],bg['components'],pd.read_csv(source/'background_rank_cv.csv'),tuple(bg['blank_ids'].astype(str)),0.)
    out.mkdir(parents=True);plots.mkdir(parents=True,exist_ok=True)
    # Same saved background and original settings. No fit to held-out labels.
    baseline,repeated,repeat_diagnostics=vibes_baseline_matrix(wn,X,background,sample_ids=chosen.sample_id.tolist(),tau=run['config']['tau'],loss=run['config']['loss'],maxiter=run['config']['maxiter'],retry_failed=True)
    repeat_diagnostics.to_csv(out/'repeated_solver_diagnostics.csv',index=False)
    chosen['repeat_success']=repeat_diagnostics.success.to_numpy()
    chosen['repeat_max_abs_corrected_difference']=np.max(np.abs(repeated.astype(np.float32).astype(float)-corrected['VIBES']),axis=1)
    chosen['rms_corrected_difference']=np.sqrt(np.mean((corrected['VIBES']-corrected['AIRSpec'])**2,axis=1))
    chosen['max_abs_corrected_difference']=np.max(np.abs(corrected['VIBES']-corrected['AIRSpec']),axis=1)
    chosen.to_csv(out/'case_evidence.csv',index=False)
    np.savez_compressed(out/'inspection_spectra.npz',sample_ids=chosen.sample_id.to_numpy(str),wn=wn,raw=X,airspec=corrected['AIRSpec'],vibes=corrected['VIBES'],repeated_vibes=repeated)
    PlotConfig.set(sites='all',layout='individual',font_size=10,title_size=12)
    for site in ['BRIS1','CACR1']:
        indices=np.flatnonzero(chosen.Site.eq(site));fig,axs=plt.subplots(3,6,figsize=(21,10),layout='constrained')
        for col,i in enumerate(indices):
            row=chosen.iloc[i]
            axs[0,col].plot(wn,X[i],label='Raw',lw=.8,color='0.35')
            axs[0,col].plot(wn,X[i]-corrected['AIRSpec'][i],label='AIRSpec baseline',lw=.9)
            axs[0,col].plot(wn,X[i]-corrected['VIBES'][i],label='VIBES baseline',lw=.9)
            axs[1,col].plot(wn,corrected['AIRSpec'][i],label='AIRSpec corrected',lw=1)
            axs[1,col].plot(wn,corrected['VIBES'][i],label='VIBES corrected',lw=1)
            spectral=[row[c] for c in ['spectra_1425_1799','spectra_1800_2499','spectra_2500_2999','spectra_3000_4000']]
            coeff=[row[c] for c in ['coefficients_1425_1799','coefficients_1800_2499','coefficients_2500_2999','coefficients_3000_4000']]
            xs=np.arange(4);axs[2,col].bar(xs-.18,spectral,.36,label='Spectral term');axs[2,col].bar(xs+.18,coeff,.36,label='Coefficient term')
            axs[2,col].set_xticks(xs,['1425–1799','1800–2499','2500–2999','3000–4000'],rotation=65,ha='right');axs[2,col].axhline(0,color='0.5',lw=.6)
            for k in [0,1]:axs[k,col].set_xlim(wn.max(),wn.min());style_axes(axs[k,col],'Wavenumber (cm⁻¹)','Absorbance',show_legend=col==0)
            style_axes(axs[2,col],'Wavenumber interval (cm⁻¹)','Prediction contribution (µg/filter)',show_legend=col==0)
            axs[0,col].set_title(f"{row.sample_id.split(':')[1]} · {row.inspection_direction}\nΔ squared error={row.delta_squared_error:.2f}")
        fig.suptitle(f'{site}: post-hoc case inspection — original fitted models and all declared controls',fontsize=16)
        fig.savefig(plots/f'{site}_inspection.png',bbox_inches='tight');plt.close(fig)
    assert all(sha(p)==h for p,h in hashes.items())
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'n_cases':len(chosen),'n_repeat_success':int(chosen.repeat_success.sum()),'original_retries':int(chosen.retry_count.sum()),'maximum_repeat_corrected_difference':float(chosen.repeat_max_abs_corrected_difference.max()),'source_hashes':hashes,'workflow_sha256':sha(__file__),'sources_unchanged':True,'no_new_exclusions':True,'no_parameter_tuning':True,'scope':'Post-hoc physical-filter case inspection and numerical repeatability. Solver convergence does not validate a physical baseline; spectral attribution is conditional on fitted coefficients.'}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':main()
