"""Numerical workflow tests; native EPA fixture tests are NOT actual EPA runs."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls
from scipy.special import huber

from src.pmf.io import Dataset, from_csv, load_dataset, read_matrix, write_json, write_matrix
from src.pmf.runner import load_model, project_case, run_case, selected_model
from src.pmf.solver import _g_block, _h_block, fit, diagnostics, rho


def data():
    rng=np.random.default_rng(81)
    g=rng.uniform(.1,2,(14,2))
    h=rng.uniform(.1,2,(2,9))
    x=g@h+rng.normal(0,.04,(14,9))
    x[0,0]=-.2
    return Dataset(x,np.full_like(x,.1),np.asarray(['0001','NA',*[f'S{i:02d}' for i in range(2,14)]]),
        np.asarray([str(4000-i*100) for i in range(9)]),{'site':'TST','units':'absorbance/m3','baseline':'synthetic'},
        {'dates':np.asarray([f'2026-01-{i+1:02d}' for i in range(14)]),'lot_ids':np.repeat('L1',14)})


def test_csv_roundtrip_preserves_signed_values_and_ids(tmp_path):
    d=data()
    write_matrix(tmp_path/'x.csv',d.sample_ids,d.features,d.X)
    write_matrix(tmp_path/'u.csv',d.sample_ids,d.features,d.U)
    r=from_csv(tmp_path/'x.csv',tmp_path/'u.csv',d.protocol)
    assert r.sample_ids.tolist()[:2]==['0001','NA']
    assert np.array_equal(r.X,d.X) and np.array_equal(r.U,d.U)


@pytest.mark.parametrize('kind',['row_order','feature_order','zero','negative','nan','inf','duplicate','ragged','missing_label'])
def test_bad_csv_inputs_rejected(tmp_path,kind):
    d=data()
    x=tmp_path/'x.csv'
    u=tmp_path/'u.csv'
    write_matrix(x,d.sample_ids,d.features,d.X)
    write_matrix(u,d.sample_ids,d.features,d.U)
    rows=list(csv.reader(u.open()))
    if kind=='row_order':
        rows[1],rows[2]=rows[2],rows[1]
    elif kind=='feature_order':
        rows[0][1],rows[0][2]=rows[0][2],rows[0][1]
    elif kind in ('zero','negative','nan','inf'):
        rows[1][1]={'zero':'0','negative':'-1','nan':'NaN','inf':'inf'}[kind]
    elif kind=='duplicate':
        rows[2][0]=rows[1][0]
    elif kind=='ragged':
        rows[2].pop()
    elif kind=='missing_label':
        rows[1][0]=''
    with u.open('w',newline='') as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError):
        from_csv(x,u)


def test_snapshot_verifies_metadata_and_numerical_hashes(tmp_path):
    d=data()
    p=tmp_path/'d.npz'
    d.save(p)
    r=load_dataset(p)
    assert r.identity==d.identity
    m=json.loads(p.with_suffix('.json').read_text())
    m['protocol']['site']='CHANGED'
    write_json(p.with_suffix('.json'),m)
    with pytest.raises(ValueError,match='identity'):
        load_dataset(p)


def test_snapshot_refuses_overwrite(tmp_path):
    d=data()
    p=tmp_path/'d.npz'
    d.save(p)
    with pytest.raises(FileExistsError):
        d.save(p)


def test_quadratic_blocks_match_scipy():
    rng=np.random.default_rng(69)
    x=rng.normal(size=(9,12))
    u=rng.uniform(.3,2,x.shape)
    g=rng.random((9,3))
    h=rng.random((3,12))
    w=1/u**2
    gg=_g_block(x,w,h,g.copy(),500)
    hh=_h_block(x,w,g,h.copy(),500)
    assert np.allclose(gg,np.stack([nnls(h.T/u[i,:,None],x[i]/u[i])[0] for i in range(9)]),atol=1e-9)
    assert np.allclose(hh,np.stack([nnls(g/u[:,j,None],x[:,j]/u[:,j])[0] for j in range(12)],axis=1),atol=1e-9)


def test_huber_is_distinct_from_excluded_residual_diagnostic():
    r=np.array([[-5.,-4.,0.,4.,5.]])
    m=diagnostics(r,np.ones_like(r),np.zeros((1,1)),np.ones((1,5)),'huber')
    assert m['Q_true']==82 and m['Q_robust_excluded_4']==32 and m['Q_huber']==80
    assert np.array_equal(rho(r,'huber'),2*huber(4,r))


@pytest.mark.parametrize('loss',['quadratic','huber'])
def test_signed_fitting_monotone_reproducible_and_nonnegative(loss):
    d=data()
    r=fit(d.X,d.U,2,13,loss=loss,max_iter=120)
    s=fit(d.X,d.U,2,13,loss=loss,max_iter=120)
    assert np.min(r['G'])>=0 and np.min(r['H'])>=0
    assert np.all(np.diff(r['history'][:,1])<=1e-7)
    assert np.array_equal(r['G'],s['G']) and np.array_equal(r['history'],s['history'])
    assert d.X[0,0]==-.2


def test_multistart_checkpoint_report_resume_and_cap(tmp_path):
    d=data()
    out=tmp_path/'run'
    m=run_case(d,out,dataset_id='TST',factors=[2],seeds=[1,3],loss='huber',warmup=10,max_iter=30,report=True)
    assert m['counts']['capped']==2 and m['status']=='capped'
    assert (out/'report/index.html').exists()
    assert 'Retained samples: 14' in (out/'report/report.md').read_text()
    assert len(list(out.glob('k02/seed_*.npz')))==2
    r=run_case(d,out,dataset_id='TST',factors=[2],seeds=[1,3],loss='huber',warmup=10,max_iter=30,resume=True,report=False)
    assert r['selected']==m['selected']
    with pytest.raises(ValueError,match='Resume refused'):
        run_case(d,out,dataset_id='TST',factors=[2],seeds=[1,3],loss='huber',warmup=10,max_iter=31,resume=True,report=False)


def test_checkpoint_corruption_is_not_reused(tmp_path):
    d=data()
    out=tmp_path/'r'
    m=run_case(d,out,dataset_id='TST',factors=[2],seeds=[1],max_iter=20,warmup=0,report=False)
    p=out/m['selected']['2']['model']
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        load_model(p,d)
    from src.pmf.report import build_report
    with pytest.raises(ValueError,match='changed'):
        build_report(out)


def test_converged_candidate_preferred_to_lower_capped(monkeypatch,tmp_path):
    import src.pmf.runner as runner
    original=runner.fit
    def injected(*args,**kwargs):
        r=original(*args,**kwargs)
        # Make seed 1 a deliberately inferior but marked-converged numerical test object.
        if args[3]==1:
            r['G']*=10
            r['metrics'].update(diagnostics(args[0],args[1],r['G'],r['H'],kwargs['loss']))
            r['metrics']['converged']=True
        else:
            r['metrics']['converged']=False
        return r
    monkeypatch.setattr(runner,'fit',injected)
    result=run_case(data(),tmp_path/'r',dataset_id='fixture',factors=[2],seeds=[1,2],loss='quadratic',max_iter=10,report=False)
    assert result['selected']['2']['seed']==1 and result['selected']['2']['lowest_attained_seed']==2


def test_frozen_projection_checks_protocol_and_does_not_modify_H(tmp_path):
    d=data()
    out=tmp_path/'fit'
    run_case(d,out,dataset_id='TST',factors=[2],seeds=[1],loss='quadratic',max_iter=80,report=False)
    _,model,m=selected_model(out,2)
    p=out/m['selected']['2']['model']
    before=p.read_bytes()
    projected=project_case(d.subset(np.arange(3)),out,tmp_path/'p',2)
    assert projected['profile_unchanged'] and p.read_bytes()==before
    altered=data()
    altered.protocol['units']='other'
    with pytest.raises(ValueError,match='protocol'):
        project_case(altered,out,tmp_path/'bad',2)
    changed=data()
    changed.features=changed.features[::-1]
    with pytest.raises(ValueError,match='identities'):
        project_case(changed,out,tmp_path/'bad2',2)


def test_old_unknown_protocol_requires_acknowledgement(tmp_path):
    d=data()
    d.protocol={'status':'unknown'}
    out=tmp_path/'fit'
    run_case(d,out,dataset_id='TST',factors=[2],seeds=[1],loss='quadratic',max_iter=20,report=False)
    with pytest.raises(ValueError,match='unknown'):
        project_case(d,out,tmp_path/'p',2)
    assert project_case(d,out,tmp_path/'p2',2,allow_unknown_protocol=True)['unknown_protocol_acknowledged']


@pytest.mark.parametrize('factors,seeds', [([0],[1]),([99],[1]),([2],[1,1]),([2],[-1]),([2.0],[1]),([],[1])])
def test_invalid_run_controls(tmp_path,factors,seeds):
    with pytest.raises(ValueError):
        run_case(data(),tmp_path/'bad',dataset_id='TST',factors=factors,seeds=seeds,report=False)
