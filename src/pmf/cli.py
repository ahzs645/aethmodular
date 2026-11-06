"""Unified, cross-platform FTIR PMF preparation and execution interface."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from .io import from_csv, fresh_directory, load_dataset, read_records, write_json


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _data(args):
    protocol=_json(args.protocol) if getattr(args,'protocol',None) else None
    if getattr(args,'input',None):
        if getattr(args,'x',None) or getattr(args,'u',None):
            raise ValueError('Choose NPZ or X/U CSV input, not both')
        return load_dataset(args.input,protocol)
    if not getattr(args,'x',None) or not getattr(args,'u',None):
        raise ValueError('Supply --input or both --x and --u')
    return from_csv(args.x,args.u,protocol)


def _path(value,base):
    value=os.path.expandvars(os.path.expanduser(str(value)))
    if '$' in value:
        raise ValueError(f'Unresolved environment variable in path: {value}')
    p=Path(value)
    return p if p.is_absolute() else base/p


def _historical(path):
    return [r['filter_id'] for r in read_records(path,('filter_id',),unique='filter_id')] if path else []


def prepare_config(config_path,output=None):
    from .prepare import prepare_native
    p=Path(config_path).resolve()
    spec=_json(p)
    base=p.parent
    options=dict(spec)
    dest=output or _path(options.pop('output'),base)
    options.pop('output',None)
    options['historical_ids']=_historical(_path(options.pop('historical_ids_file'),base)) if options.get('historical_ids_file') else options.get('historical_ids',[])
    options.pop('historical_ids_file',None)
    if options.get('grid_file'):
        options['grid']=np.loadtxt(_path(options.pop('grid_file'),base))
    for name in ('filters','analyses','scans'):
        options[name]=_path(options[name],base)
    return prepare_native(output=dest,**options)


def pipeline(config_path):
    from .runner import run_case
    from .epa import export_epa, runtime_probe
    p=Path(config_path).resolve()
    spec=_json(p)
    base=p.parent
    if set(spec)-{'output','datasets'}:
        raise ValueError('Unknown pipeline configuration keys')
    datasets = spec.get('datasets')
    if not isinstance(datasets, list) or not datasets:
        raise ValueError('Pipeline requires at least one dataset')
    names = set()
    for ds in datasets:
        name = ds['dataset_id']
        if not isinstance(name, str) or name in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('dataset_id must be a safe path component')
        if name in names:
            raise ValueError('Duplicate pipeline dataset_id')
        names.add(name)
        runs = ds.get('runs')
        if not isinstance(runs, list) or not runs:
            raise ValueError('Each pipeline dataset requires at least one run')
        labels = set()
        for run in runs:
            label = run['name']
            if not isinstance(label, str) or label in ('.', '..', 'prepared') or not re.fullmatch(r'[A-Za-z0-9_.-]+', label):
                raise ValueError('Run name must be a safe, non-reserved path component')
            if label in labels:
                raise ValueError('Duplicate pipeline run name')
            labels.add(label)
            if run.get('operation', 'fit') not in ('fit', 'epa-export'):
                raise ValueError('Unknown pipeline run operation')
    out=fresh_directory(_path(spec['output'],base))
    summaries=[]
    for ds in spec['datasets']:
        name=ds['dataset_id']
        prepdir=out/name/'prepared'
        prep=prepare_config(_path(ds['preparation_config'],base),prepdir)
        record={'dataset_id':name,'preparation_status':prep['status'],'counts':prep['counts'],'runs':[]}
        if prep['status']=='passed':
            data=load_dataset(prepdir/'dataset.npz')
            for run in ds['runs']:
                cfg=dict(run)
                label=cfg.pop('name')
                dest=out/name/label
                if cfg.pop('operation', 'fit')=='epa-export':
                    cfg.pop('engine',None)
                    value=export_epa(data,dest,dataset_id=name,**cfg)
                    record['epa_runtime']=runtime_probe()
                else:
                    value=run_case(data,dest,dataset_id=name,**cfg)
                record['runs'].append({'name':label,'status':value['status'],'engine':value['engine']})
        summaries.append(record)
        write_json(out/'pipeline.json',{'status':'in_progress','datasets':summaries})
    complete=all(d['preparation_status']=='passed' and all(r['status']=='passed' for r in d['runs']) for d in summaries)
    result={'schema':'aethmodular-pmf-pipeline/1','status':'passed' if complete else 'review_required','datasets':summaries,
        'warning':'Native export/not_run, failed and capped runs do not count as completed native execution.'}
    write_json(out/'pipeline.json',result)
    from .report import build_pipeline_report
    build_pipeline_report(out)
    return result


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('validate','fit','project','epa-export','esat-check','epa-ingest'):
        p=sub.add_parser(name)
        p.add_argument('--input',type=Path)
        p.add_argument('--x',type=Path)
        p.add_argument('--u',type=Path)
        p.add_argument('--protocol',type=Path)
        p.add_argument('--dataset',required=True)
        p.add_argument('--output',required=True,type=Path)
        if name in ('fit','epa-export'):
            p.add_argument('--factors',type=int,nargs='+',default=[5])
            p.add_argument('--seeds',type=int,nargs='+',default=[1,10,100])
        if name=='fit':
            p.add_argument('--engine',choices=['python-independent','esat','epa-pmf5'],default='python-independent')
            p.add_argument('--esat-method',choices=['ls-nmf','ws-nmf','ws-nmf-repaired'],default='ls-nmf')
            p.add_argument('--loss',choices=['quadratic','huber','native-robust'],default='huber')
            p.add_argument('--delta',type=float,default=4.)
            p.add_argument('--iterations',type=int,default=3000)
            p.add_argument('--tolerance',type=float,default=1e-7)
            p.add_argument('--stationarity-tolerance',type=float,default=1e-5)
            p.add_argument('--warmup',type=int,default=600)
            p.add_argument('--native-runtime',type=Path)
            p.add_argument('--native-timeout',type=float,default=600.)
            p.add_argument('--resume',action='store_true')
            p.add_argument('--no-report',action='store_true')
        if name=='project':
            p.add_argument('--reference-run',required=True,type=Path)
            p.add_argument('--rank',type=int,required=True)
            p.add_argument('--threshold',type=float,default=3.)
            p.add_argument('--allow-unknown-protocol',action='store_true')
            p.add_argument('--iterations',type=int,default=1000)
        if name=='epa-ingest':
            p.add_argument('--case',required=True,type=Path)
            p.add_argument('--loss',choices=['quadratic','native-robust'],default='quadratic')
        if name=='epa-export':
            p.add_argument('--categories',type=Path)
    p=sub.add_parser('intake')
    p.add_argument('--metadata',required=True,type=Path)
    p.add_argument('--historical-ids',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--minimum-blanks',type=int,default=2)
    p=sub.add_parser('prepare')
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',type=Path)
    p=sub.add_parser('pipeline')
    p.add_argument('--config',required=True,type=Path)
    p=sub.add_parser('report')
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--no-charts',action='store_true')
    p=sub.add_parser('compare')
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--rank',required=True,type=int)
    p.add_argument('--candidate-rank',type=int)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--allow-protocol-difference',action='store_true')
    p=sub.add_parser('bootstrap')
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--rank',required=True,type=int)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--starts',type=int,default=2)
    p.add_argument('--block-length',type=int,default=6)
    p.add_argument('--iterations',type=int,default=2000)
    p=sub.add_parser('epa-probe')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--executable',type=Path)
    p=sub.add_parser('epa-launch')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--executable',required=True,type=Path)
    p.add_argument('--timeout',type=float,default=3600)
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    p=sub.add_parser('epa-import')
    p.add_argument('--handoff',required=True,type=Path)
    p.add_argument('--contributions',required=True,type=Path)
    p.add_argument('--profiles',required=True,type=Path)
    p.add_argument('--evidence',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--effective-x',type=Path)
    p.add_argument('--effective-u',type=Path)
    return parser


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    try:
        c=args.command
        if c=='intake':
            from .intake import audit_csv
            rows=audit_csv(args.metadata,args.output,_historical(args.historical_ids),args.minimum_blanks)
            result={'status':'passed','counts':{s:sum(r['status']==s for r in rows) for s in ('eligible','held','excluded','blank')}}
        elif c=='prepare':
            result=prepare_config(args.config,args.output)
        elif c=='pipeline':
            result=pipeline(args.config)
        elif c=='report':
            from .report import build_report
            result={'status':'passed','report':str(build_report(args.run,charts=not args.no_charts))}
        elif c=='compare':
            from .operations import compare_runs
            result=compare_runs(args.reference,args.candidate,args.output,args.rank,args.candidate_rank,args.allow_protocol_difference)
        elif c=='bootstrap':
            from .operations import bootstrap_case
            result=bootstrap_case(args.reference,args.output,args.rank,repeats=args.repeats,starts=args.starts,
                block_length=args.block_length,max_iter=args.iterations)
        elif c=='epa-probe':
            from .epa import runtime_probe
            result=runtime_probe(args.output,args.executable)
        elif c=='epa-launch':
            from .epa import launch_native
            result=launch_native(args.executable,args.output,arguments=args.arguments,timeout=args.timeout)
        elif c=='epa-import':
            from .epa import import_epa
            result=import_epa(args.handoff,args.contributions,args.profiles,args.evidence,args.output,
                effective_x=args.effective_x,effective_u=args.effective_u)
        else:
            data=_data(args)
            if c=='validate':
                out=fresh_directory(args.output)
                data.save(out/'input.npz')
                result={'status':'passed','operation':'validate','samples':len(data.X),'features':data.X.shape[1],'identity':data.identity}
                write_json(out/'manifest.json',result)
            elif c=='fit':
                from .runner import run_case
                result=run_case(data,args.output,dataset_id=args.dataset,factors=args.factors,seeds=args.seeds,
                    engine=args.engine,loss=args.loss,delta=args.delta,max_iter=args.iterations,tol=args.tolerance,
                    stationarity_tol=args.stationarity_tolerance,warmup=args.warmup,esat_method=args.esat_method,
                    resume=args.resume,report=not args.no_report,native_runtime=args.native_runtime,native_timeout=args.native_timeout)
            elif c=='project':
                from .runner import project_case
                result=project_case(data,args.reference_run,args.output,args.rank,threshold=args.threshold,
                    allow_unknown_protocol=args.allow_unknown_protocol,max_iter=args.iterations)
            elif c=='epa-export':
                from .epa import export_epa
                result=export_epa(data,args.output,dataset_id=args.dataset,factors=args.factors,seeds=args.seeds,
                    categories=_json(args.categories) if args.categories else None)
            elif c=='epa-ingest':
                from .native_backend import ingest_case
                result=ingest_case(data,args.case,args.output,dataset_id=args.dataset,loss=args.loss)
            elif c=='esat-check':
                from .esat_backend import verify_source
                out=fresh_directory(args.output)
                pin=verify_source()
                result={'status':'not_run','operation':'esat-check','source_pin':pin,'samples':len(data.X),'features':data.X.shape[1],
                    'signed_observations':bool(np.any(data.X<0)),
                    'notes':'Source verification only; execute fit --engine esat with an explicit method. Stock WS currently fails; repair is opt-in.'}
                write_json(out/'manifest.json',result)
        print(json.dumps(result,indent=2,allow_nan=False))
        # A status query/export can succeed without pretending to have fitted a model.
        return 2 if result.get('status') in ('failed','blocked') else (3 if result.get('status') in ('capped','review_required') else 0)
    except (OSError,ValueError,KeyError,RuntimeError,TypeError) as exc:
        print(f'PMF ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
