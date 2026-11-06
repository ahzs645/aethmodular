"""New batches cannot inherit frozen counts or unsafe missing-ID membership."""
import pytest
from src.pmf.intake import evaluate,parse_date,valid_volume


def row(fid,**kwargs):
    r={'filter_id':fid,'site':'ETBI','particle_fraction':'PM2.5','analysis_id':fid+'-scan',
       'analysis_status':'retained','date':'2026-11-01','volume_m3':'7.2','lot':'254','blank_count':'3'}
    r.update(kwargs)
    return r


def test_new_filter_count_is_unbounded():
    result=evaluate([row(f'ETBI-{i:04d}-1') for i in range(90)])
    assert len(result)==90 and all(r['status']=='eligible' for r in result)


def test_unknown_ID_is_never_historical():
    result=evaluate([row('ETAD-9999-1',site='ETAD')],['ETAD-0001-1'])
    assert result[0]['historical_102'] is False


@pytest.mark.parametrize('change,status',[({'volume_m3':''},'held'),({'volume_m3':'nan'},'held'),
    ({'volume_m3':'inf'},'held'),({'volume_m3':'-1'},'held'),({'date':'2026-99-01'},'held'),
    ({'date':''},'held'),({'blank_count':'1'},'held'),({'lot':''},'held'),
    ({'exclusion_reason':'large hole'},'excluded'),({'particle_fraction':'PM10'},'excluded'),
    ({'particle_fraction':'FB'},'blank'),({'analysis_status':'unknown'},'held')])
def test_bad_metadata_held_without_blocking_other_samples(change,status):
    result={r['filter_id']:r for r in evaluate([row('bad',**change),row('good')])}
    assert result['bad']['status']==status and result['good']['status']=='eligible'


def test_duplicate_retained_analysis_is_held():
    assert evaluate([row('repeat',analysis_id='1'),row('repeat',analysis_id='2')])[0]['status']=='held'


def test_invalidated_record_not_averaged_into_retained():
    result=evaluate([row('repeat',analysis_id='1',analysis_status='invalidated'),row('repeat',analysis_id='2')])
    assert result[0]['status']=='eligible' and result[0]['analysis_id']=='2'


@pytest.mark.parametrize('text,expected',[('12/7/2022 12:00:00 AM','2022-12-07'),('2026-01-01T12:30:00Z','2026-01-01'),('None',None)])
def test_native_date_parse(text,expected):
    assert parse_date(text)==expected
