import json
import numpy as np
import pytest

from src.pmf.io import Dataset
from src.pmf.operations import compare_runs, bootstrap_case
from src.pmf.runner import run_case
from src.pmf.native_backend import result_from_native, fit_native
from src.pmf.solver import diagnostics


def tiny():
    a=np.linspace(.5,2,12)
    x=np.outer(a,[1.,2.,3.])
    return Dataset(x,np.full_like(x,.1),np.asarray([f'A{i}' for i in range(12)]),
        np.asarray(['1','2','3']),{'units':'test'},
        {'dates':np.asarray([f'2026-01-{i+1:02d}' for i in range(12)]),'lot_ids':np.repeat('L1',12)},{})


def test_comparison_and_segmented_bootstrap(tmp_path):
    d=tiny()
    run_case(d,tmp_path/'reference',dataset_id='toy',factors=[1],seeds=[1,10],loss='quadratic',max_iter=200,report=False)
    result=compare_runs(tmp_path/'reference',tmp_path/'reference',tmp_path/'comparison',1)
    assert result['common_filters']==12
    assert result['matches'][0]['profile_cosine']==pytest.approx(1)
    b=bootstrap_case(tmp_path/'reference',tmp_path/'bootstrap',1,repeats=2,starts=2,block_length=3,max_iter=200)
    assert 0 <= b['eligible_by_factor'][0] <= 2
    import csv
    with (tmp_path/'bootstrap'/'matches.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert b['eligible_by_factor']==[sum(r['eligible_for_spread']=='True' for r in rows)]
    assert all(r['converged']=='True' for r in rows if r['eligible_for_spread']=='True')
    assert len(json.loads((tmp_path/'bootstrap'/'blocks.json').read_text()))==2


def test_native_objective_is_not_independent_huber():
    x=np.asarray([[1.,2.],[3.,4.]])
    u=np.ones_like(x)
    g=np.ones((2,1))
    h=np.ones((1,2))
    q=float(np.sum((x-g@h)**2))
    receipt={'native_engine_executed':True,'script_modified':False,'status':'passed','robust':True,
        'native_convergence_observed':True,'seed':1,'version':'fixture only','runtime_sha256':{},
        'metrics':{'Q_true':q,'native_Q_main':7.,'native_iterations':32}}
    r=result_from_native(x,u,g,h,receipt,loss='native-robust',delta=4,driver_hash='fixture')
    assert r['metrics']['objective']==q
    assert r['metrics']['selection_objective']==7
    assert r['metrics']['selection_objective_name']=='native_Q_main'
    assert r['config']['requested_native_loss']=='native-robust'
    with pytest.raises(ValueError,match='distinct|quadratic or native-robust'):
        result_from_native(x,u,g,h,receipt,loss='huber',delta=4,driver_hash='fixture')


def test_native_receipt_rejects_fabricated_success():
    d=tiny()
    g=np.ones((12,1))
    h=np.ones((1,3))
    with pytest.raises(ValueError,match='verified unchanged'):
        result_from_native(d.X,d.U,g,h,{'native_engine_executed':False},loss='quadratic',delta=4,driver_hash='fixture')


def test_native_attempt_is_failed_not_python_fallback(tmp_path):
    import os
    if os.name=='nt':
        pytest.skip('Linux platform guard')
    m=run_case(tiny(),tmp_path/'native',dataset_id='synthetic',factors=[1],seeds=[1],engine='epa-pmf5',loss='quadratic',report=False)
    assert m['status']=='failed'
    assert not m['native_EPA_PMF_or_ME2_executed']
    assert not m['selected']
    reason=json.loads(next((tmp_path/'native').rglob('*.failed.json')).read_text())
    assert 'requires Windows' in reason['summary']['error']
