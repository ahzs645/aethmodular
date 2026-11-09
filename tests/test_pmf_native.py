"""Synthetic native beams and complete intake/preparation/fit/report integration."""
import base64
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from src.pmf.cli import main
from src.pmf.io import load_dataset,write_json,write_table
from src.pmf.prepare import absorbance,decode_scan,prepare_native


def beam(scan_id,y,w,kind='Sample'):
    return {'Id':str(scan_id),'DataPointFormat':'1','NumberOfDataPoints':str(len(w)),
        'FrequencyOfFirstPoint':str(w[0]),'FrequencyOfLastPoint':str(w[-1]),'YScalingFactor':'1',
        'XUnits':'WN','Type':kind,'InstrumentType':'synthetic','InstrumentSerialNumber':'S1',
        'Data':base64.b64encode(np.asarray(y,dtype='<f4').tobytes()).decode()}


def fixture(tmp_path):
    rng=np.random.default_rng(17)
    w=np.linspace(3998,1426,1100)
    scans=[beam('BG',np.ones_like(w),w,'Background')]
    analyses=[]
    filters=[]
    for i in range(10):
        blank=i<3
        y=.01*(w-w.min())/(w.max()-w.min())+rng.normal(0,.0001,len(w))
        if not blank:
            y+=(i/5)*.015*np.exp(-((w-2920)/35)**2)+.008*np.exp(-((w-1710)/22)**2)
        scans.append(beam(str(i),10**(-y),w))
        analyses.append({'Id':f'A{i}','MediaId':str(i),'AnalysisQcCode':'1','SampleScanId':str(i),'BackgroundScanId':'BG',
            'InstrumentId':'TEST','AnalysisDate':'2026-10-31','AnalysisComments':'synthetic fixture'})
        filters.append({'MediaId':str(i),'ExternalFilterId':f'TST-{i:04d}-1','ExternalFilterType':'FB' if blank else 'PM2.5',
            'ExternalLotId':'L1','SampleVolume_m3':'' if blank else '7.2','SamplingStartDate':'' if blank else f'2026-10-{i:02d}'})
    # Keep one unusable future lot and one metadata hold without blocking the good group.
    filters[8]['ExternalLotId']='L2'
    filters[9]['SampleVolume_m3']=''
    for name,rows in [('filters',filters),('analyses',analyses),('scans',scans)]:
        write_table(tmp_path/f'{name}.csv',rows)
    return filters,analyses,scans


def test_native_scan_roundtrip_and_absorbance():
    w=np.linspace(4000,1000,20)
    y=np.repeat(.5,20)
    sample=beam('S',y,w)
    background=beam('B',np.ones(20),w,'Background')
    ww,decoded=decode_scan(sample)
    assert np.array_equal(ww,w) and np.array_equal(decoded,y)
    _,a=absorbance({'SampleScanId':'S','BackgroundScanId':'B'},{'S':sample,'B':background})
    assert np.allclose(a,-np.log10(.5))


@pytest.mark.parametrize('defect',['short','zero','mismatch','range'])
def test_native_rejects_corrupt_beams_without_extrapolation(defect):
    w=np.linspace(4000,1000,20)
    sample=beam('S',np.ones(20),w)
    background=beam('B',np.ones(20),w,'Background')
    if defect=='short':
        sample['NumberOfDataPoints']='21'
    elif defect=='zero':
        sample=beam('S',np.zeros(20),w)
    elif defect=='mismatch':
        background['InstrumentSerialNumber']='OTHER'
    elif defect=='range':
        background['FrequencyOfFirstPoint']='3900'
    with pytest.raises(ValueError):
        absorbance({'SampleScanId':'S','BackgroundScanId':'B'},{'S':sample,'B':background})


def test_preparation_holds_individual_filters_and_does_not_invent_sd(tmp_path):
    fixture(tmp_path)
    m=prepare_native(tmp_path/'filters.csv',tmp_path/'analyses.csv',tmp_path/'scans.csv',tmp_path/'prepared',
        site='TST',historical_ids=['TST-0003-1'],exclusions={'TST-0004-1':'synthetic damaged filter'})
    assert m['counts']=={'eligible':4,'held':2,'excluded':1,'blank':3}
    d=load_dataset(tmp_path/'prepared/dataset.npz')
    assert len(d.X)==4 and np.all(d.U>0)
    assert list(d.metadata['historical_102']).count('True')==1
    assert 'TST-0008-1' not in d.sample_ids
    assert (tmp_path/'prepared/analysis_selection.csv').exists()
    assert m['protocol']['observations_clipped'] is False


def test_explicit_grid_out_of_bounds_holds_without_extrapolation(tmp_path):
    fixture(tmp_path)
    m=prepare_native(tmp_path/'filters.csv',tmp_path/'analyses.csv',tmp_path/'scans.csv',tmp_path/'prepared',
        site='TST',grid=np.linspace(4100,1400,1200))
    assert m['status']=='blocked' and m['counts']['eligible']==0
    assert not (tmp_path/'prepared/dataset.npz').exists()


def test_raw_pipeline_config_executes_and_documents_capped_results(tmp_path):
    fixture(tmp_path)
    write_json(tmp_path/'prepare.json',{'filters':'filters.csv','analyses':'analyses.csv','scans':'scans.csv','site':'TST'})
    write_json(tmp_path/'pipeline.json',{'output':'runs','datasets':[{'dataset_id':'TST','preparation_config':'prepare.json',
        'runs':[{'name':'huber','engine':'python-independent','factors':[2],'seeds':[1],'max_iter':30,'warmup':10},
                {'name':'epa','operation':'epa-export','engine':'epa-pmf5','factors':[2],'seeds':[1]}]}]})
    code=main(['pipeline','--config',str(tmp_path/'pipeline.json')])
    assert code==3
    m=json.loads((tmp_path/'runs/pipeline.json').read_text())
    assert m['status']=='review_required'
    assert (tmp_path/'runs/TST/huber/report/index.html').exists()
    assert json.loads((tmp_path/'runs/TST/epa/manifest.json').read_text())['status']=='not_run'


def test_dates_and_counts_are_dynamic_in_report(tmp_path):
    from src.pmf.runner import run_case
    from src.pmf.io import Dataset
    rng=np.random.default_rng(3)
    g=rng.random((7,2))
    h=rng.random((2,4))
    x=g@h
    d=Dataset(x,np.full_like(x,.2),np.asarray([f'new{i}' for i in range(7)]),np.asarray(['a','b','c','d']),
        metadata={'dates':np.repeat('2031-02-03',7)})
    run_case(d,tmp_path/'run',dataset_id='NEW',factors=[2],seeds=[1],max_iter=40,warmup=0)
    text=(tmp_path/'run/report/report.md').read_text()
    assert 'Retained samples: 7' in text and '2031-02-03 to 2031-02-03' in text
    assert '246' not in text.split('## What the numbers mean')[0]
