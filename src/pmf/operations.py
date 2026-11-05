"""Cross-run factor matching and auditable block-bootstrap pilots."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .io import Dataset, fresh_directory, sha256, write_json, write_table
from .runner import run_case, selected_model
from .solver import profile_match
from .uncertainty import moving_block_indices, temporal_segments


def compare_runs(reference, candidate, output, reference_rank, candidate_rank=None, allow_protocol_difference=False):
    a,am,ar=selected_model(reference,reference_rank)
    b,bm,br=selected_model(candidate,candidate_rank or reference_rank)
    if not np.array_equal(a.features,b.features):
        raise ValueError('Different feature grids: harmonization must be explicit, not silently interpolated')
    if a.compatibility!=b.compatibility and not allow_protocol_difference:
        raise ValueError('Different preprocessing/protocols; explicit sensitivity comparison acknowledgement required')
    ii,jj,cos,_=profile_match(am['H'],bm['H'],1/np.median(a.U,axis=0))
    amap={v:i for i,v in enumerate(a.sample_ids)}
    bmap={v:i for i,v in enumerate(b.sample_ids)}
    common=[v for v in a.sample_ids if v in bmap]
    rows=[]
    for i,j,c in zip(ii,jj,cos):
        g=am['G'][[amap[v] for v in common],i]
        h=bm['G'][[bmap[v] for v in common],j]
        r=float(np.corrcoef(g,h)[0,1]) if len(g)>1 and np.std(g)>0 and np.std(h)>0 else None
        rows.append({'reference_factor':int(i+1),'candidate_factor':int(j+1),'profile_cosine':float(c),
            'common_filters':len(common),'contribution_r':r,'contribution_r_squared':r*r if r is not None else None})
    out=fresh_directory(output)
    write_table(out/'factor_matches.csv',rows)
    result={'operation':'compare','status':'passed','reference_dataset_identity':a.identity,'candidate_dataset_identity':b.identity,
        'reference_engine':ar['engine'],'candidate_engine':br['engine'],'protocols_identical':a.compatibility==b.compatibility,
        'reference_run_manifest_sha256':sha256(Path(reference)/'manifest.json'),
        'candidate_run_manifest_sha256':sha256(Path(candidate)/'manifest.json'),
        'common_filters':len(common),'matches':rows,'warnings':['Matching is descriptive, not independent source identification.',
            'Different cohorts, losses and constraints can change factor allocations.']}
    write_json(out/'manifest.json',result)
    return result


def bootstrap_case(reference, output, rank, *, repeats=20, starts=2, block_length=6, max_gap_days=14,
                   seed=64718, max_iter=2000, similarity_threshold=.90):
    data,model,manifest=selected_model(reference,rank)
    if manifest['engine']!='python-independent':
        raise ValueError('This bootstrap uses the independent solver; it cannot silently substitute an external engine')
    if 'dates' not in data.metadata:
        raise ValueError('Chronological dates are required for segmented block bootstrap')
    if repeats<1 or starts<1 or int(repeats)!=repeats or int(starts)!=starts or not 0<=similarity_threshold<=1:
        raise ValueError('Invalid bootstrap controls')
    segments=temporal_segments(data.metadata['dates'],data.metadata.get('lot_ids'),max_gap_days)
    out=fresh_directory(output)
    rng=np.random.default_rng(seed)
    rows=[]
    profiles=[]
    eligible=[]
    blocks_all=[]
    for repeat in range(repeats):
        index,blocks=moving_block_indices(len(data.X),rng,block_length,segments)
        sampled=Dataset(data.X[index],data.U[index],np.asarray([f'bootstrap_{repeat}_{i}' for i in range(len(index))]),
            data.features,data.protocol,{**{k:v[index] for k,v in data.metadata.items()},'original_sample_id':data.sample_ids[index]},data.sources)
        result=run_case(sampled,out/f'resample_{repeat:04d}',dataset_id=f'{manifest["dataset_id"]}:bootstrap:{repeat}',
            factors=[rank],seeds=list(range(100000+starts*repeat,100000+starts*(repeat+1))),
            loss=model['metrics']['loss'],delta=model['config'].get('delta',4),max_iter=max_iter,report=False)
        _,fitted,_=selected_model(out/f'resample_{repeat:04d}',rank)
        ii,jj,cos,_=profile_match(model['H'],fitted['H'],1/np.median(data.U,axis=0))
        mapped=cos>=similarity_threshold
        valid=mapped & bool(fitted['metrics']['converged'])
        profiles.append(fitted['H'][jj])
        eligible.append(valid)
        blocks_all.append({'repeat':repeat,'indices':index.tolist(),'blocks':blocks})
        for i,j,c,v in zip(ii,jj,cos,valid):
            rows.append({'repeat':repeat,'factor':int(i+1),'matched_factor':int(j+1),'cosine':float(c),
                'converged':bool(fitted['metrics']['converged']),'eligible_for_spread':bool(v)})
    profiles=np.asarray(profiles)
    eligible=np.asarray(eligible)
    pct=np.full((3,rank,data.X.shape[1]),np.nan)
    for i in range(rank):
        if eligible[:,i].any():
            pct[:,i]=np.percentile(profiles[eligible[:,i],i],[5,50,95],axis=0)
    np.savez_compressed(out/'empirical_spread.npz',profile_percentiles=pct,levels=[5,50,95],eligible=eligible,
        H_aligned=profiles,features=data.features)
    write_table(out/'matches.csv',rows)
    write_json(out/'blocks.json',blocks_all)
    result={'operation':'block-bootstrap','status':'passed' if eligible.all() else 'review_required',
        'repeats':repeats,'starts_per_resample':starts,'block_length':block_length,'segments':[s.tolist() for s in segments],
        'eligible_by_factor':eligible.sum(axis=0).tolist(),'source_manifest_sha256':sha256(Path(reference)/'manifest.json'),
        'warnings':['Central 90% empirical spread, not validated confidence coverage.',
            'Fixed preprocessing, uncertainty and factor count; not native EPA BS/DISP/BS-DISP.',
            'Each start is preserved; only converged mapped factors enter spread calculations.']}
    write_json(out/'manifest.json',result)
    return result
