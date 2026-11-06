"""Real pinned ESAT code tests and synthetic EPA import-contract fixtures."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import nnls

from src.pmf.esat_backend import fit_esat,repaired_ws_update,verify_source
from src.pmf.epa import export_epa,import_epa,runtime_probe
from src.pmf.io import Dataset,write_json,write_matrix
from src.pmf.runner import run_case


def data():
    rng=np.random.default_rng(1)
    g=rng.uniform(.1,1,(12,2))
    h=rng.uniform(.1,1,(2,8))
    x=g@h
    return Dataset(x,np.full_like(x,.1),np.array([str(i) for i in range(12)]),np.array([str(i+1) for i in range(8)])),g,h


def test_upstream_pin_is_verified():
    pin=verify_source()
    assert pin['commit']=='0cc8e8d58b06a4484c8255ee77d7f5d5dc96181c'


def test_actual_esat_LS_executes():
    d,_,_=data()
    r=fit_esat(d.X,d.U,2,1,max_iter=300)
    assert r['provenance']['stock_ESAT_algorithm'] is True
    assert np.isfinite(r['G']).all() and r['metrics']['Q_true']<1


def test_actual_ESAT_stock_WS_failure_is_not_substituted(tmp_path):
    d,_,_=data()
    d.X=d.X-.7
    with pytest.raises(TypeError,match='hold_h'):
        fit_esat(d.X,d.U,2,1,method='ws-nmf',max_iter=10)
    result=run_case(d,tmp_path/'failure',dataset_id='signed',factors=[2],seeds=[1],engine='esat',
        loss='quadratic',esat_method='ws-nmf',max_iter=10,report=False)
    assert result['status']=='failed' and not result['selected']
    assert (tmp_path/'failure/k02/seed_1.failed.json').exists()


def test_ESAT_LS_rejects_signed_inputs_and_unsupported_robust():
    d,_,_=data()
    with pytest.raises(ValueError,match='signed'):
        fit_esat(d.X-1,d.U,2)
    with pytest.raises(ValueError,match='Huber'):
        fit_esat(d.X,d.U,2,loss='huber')


def test_repaired_WS_exact_signed_factorization_stays_exact():
    rng=np.random.default_rng(22)
    g=rng.normal(size=(12,2))
    h=rng.uniform(.1,1,(2,8))
    x=g@h
    w=rng.uniform(.5,2,x.shape)
    gg,hh=repaired_ws_update(x,w,g,h)
    assert np.max(np.abs(gg@hh-x))<1e-10 and (gg<0).any()


def test_repaired_WS_monotone_and_explicitly_not_stock():
    d,_,_=data()
    r=fit_esat(d.X-.7,d.U,2,1,method='ws-nmf-repaired',max_iter=500)
    assert r['provenance']['upstream_update_replaced'] is True
    assert not r['provenance']['stock_ESAT_algorithm']
    assert r['provenance']['signed_contributions_allowed']
    assert 'negative_observation_Q_lower_bound' not in r['metrics']
    assert r['metrics']['maximum_objective_relative_increase']<1e-8


def test_repaired_H_block_agrees_with_convex_problem_after_sweeps():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(10,8))
    h=rng.uniform(.2,2,(2,8))
    w=rng.uniform(.5,2,x.shape)
    g,h2=repaired_ws_update(x,w,None,h)
    # Columnwise optimal H is independently obtained for the returned signed G.
    target=np.stack([nnls(g*np.sqrt(w[:,j,None]),x[:,j]*np.sqrt(w[:,j]))[0] for j in range(8)],axis=1)
    assert np.allclose(g@h2,g@target,atol=2e-4)


def test_native_probe_does_not_claim_execution(tmp_path):
    result=runtime_probe(tmp_path/'probe',tmp_path/'absent.exe')
    assert result['status']=='blocked' and result['native_EPA_PMF_or_ME2_executed'] is False


def test_epa_export_is_not_native_run(tmp_path):
    d,_,_=data()
    r=export_epa(d,tmp_path/'handoff',dataset_id='fixture',factors=[2],seeds=[1])
    assert r['status']=='not_run' and r['numerical_readback_exact']


def _native_fixture(tmp_path):
    d,g,h=data()
    export_epa(d,tmp_path/'handoff',dataset_id='synthetic-contract-test',factors=[2],seeds=[1])
    write_matrix(tmp_path/'G.csv',d.sample_ids,['F1','F2'],g)
    write_matrix(tmp_path/'H.csv',['F1','F2'],d.features,h)
    (tmp_path/'fake_native.log').write_text('SYNTHETIC TEST FIXTURE ONLY. Not a native ME-2 execution.\n')
    evidence={'engine':'EPA PMF5','engine_version':'synthetic-fixture-not-an-execution','native_execution':True,
        'convergence_status':'converged','settings':{'seed':1},'reported_q_true':0.,'raw_files':['fake_native.log']}
    write_json(tmp_path/'evidence.json',evidence)
    return evidence


def test_native_import_recomputes_Q_and_retains_attestation_limit(tmp_path):
    _native_fixture(tmp_path)
    r=import_epa(tmp_path/'handoff',tmp_path/'G.csv',tmp_path/'H.csv',tmp_path/'evidence.json',tmp_path/'result')
    assert r['status']=='passed' and r['native_execution_independently_observed'] is False
    assert (tmp_path/'result/native_evidence/fake_native.log').exists()


def test_native_import_rejects_false_Q(tmp_path):
    e=_native_fixture(tmp_path)
    e['reported_q_true']=100.
    write_json(tmp_path/'evidence.json',e)
    with pytest.raises(ValueError,match='Q'):
        import_epa(tmp_path/'handoff',tmp_path/'G.csv',tmp_path/'H.csv',tmp_path/'evidence.json',tmp_path/'result')


def test_native_import_requires_actual_files_and_identity(tmp_path):
    e=_native_fixture(tmp_path)
    e['raw_files']=['missing.log']
    write_json(tmp_path/'evidence.json',e)
    with pytest.raises(ValueError,match='must exist'):
        import_epa(tmp_path/'handoff',tmp_path/'G.csv',tmp_path/'H.csv',tmp_path/'evidence.json',tmp_path/'result')
