"""Guard against false evidence provenance in the retrospective import."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'research/ftir_hips_chem/workflows'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_static_reads_follow_assignment_order_and_ignore_writes(tmp_path):
    audit=module('audit_openresearch_history')
    for name in ['a.csv','b.csv']:(tmp_path/name).write_text('x\n1\n')
    source="p = 'a.csv'\npd.read_csv(p)\np = 'b.csv'\npd.read_csv(p)\nopen(p, mode='w')\n"
    calls,gaps=audit.input_calls([('test',source)],tmp_path,{})
    assert not gaps
    assert [Path(c['resolved_path']).name for c in calls]==['a.csv','b.csv']

def test_saved_error_wins_over_stale_output_and_execution_count(tmp_path):
    audit=module('audit_openresearch_history');p=tmp_path/'error.ipynb'
    p.write_text(json.dumps({'cells':[{'cell_type':'code','source':['import missing'],'execution_count':1,'outputs':[{'output_type':'error','ename':'ImportError','evalue':'missing'}]}]}))
    state,_=audit.notebook_audit(p)
    assert state['execution_state']=='saved_error'
    assert state['saved_errors'][0]['message']=='missing'

def test_symmetric_decomposition_closes_and_reverses_sign():
    trace=module('trace_vibes_loading_predictions')
    rng=np.random.default_rng(15);xa=rng.normal(size=(8,7));xv=rng.normal(size=(8,7));ba=rng.normal(size=7);bv=rng.normal(size=7)
    s,b,c=trace.decompose(xa,xv,ba,bv,2.,3.)
    np.testing.assert_allclose(s.sum(1)+b.sum(1)+c,(xv@bv+3)-(xa@ba+2),atol=1e-12)
    reverse=trace.decompose(xv,xa,bv,ba,3.,2.)
    for f,r in zip((s,b,c),reverse):np.testing.assert_allclose(f,-r)
