"""Source-verified, data-derived scientific reports without fixed cohort text."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .io import load_dataset, read_matrix, read_records, sha256, write_json
from .runner import load_model


def verify_run(run):
    run=Path(run)
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest.get('schema')!='aethmodular-pmf-run/2':
        raise ValueError('Expected a completed fit/import run, not just a handoff')
    for name,expected in manifest.get('files',{}).items():
        if sha256(run/name)!=expected:
            raise ValueError(f'Run artifact changed: {name}')
    data=load_dataset(run/'input.npz')
    if manifest['dataset_identity']!=data.identity:
        raise ValueError('Report input identity mismatch')
    models={k:load_model(run/v['model'],data,manifest.get('request_hash')) for k,v in manifest['selected'].items()}
    return manifest,data,models


def _figures(out,data,models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[]
    try:
        x=data.features.astype(float)
        xlabel='Wavenumber (cm⁻¹)'
        spectral=True
    except ValueError:
        x=np.arange(len(data.features))
        xlabel='Feature index'
        spectral=False
    dates=data.metadata.get('dates')
    try:
        time=np.asarray(dates,dtype='datetime64[D]') if dates is not None else np.arange(len(data.X))
        if dates is not None and np.isnat(time).any():
            dates=None
            time=np.arange(len(data.X))
    except ValueError:
        dates=None
        time=np.arange(len(data.X))
    for key,result in models.items():
        rank=int(key)
        g,h=result['G'],result['H']
        r=(data.X-g@h)/data.U
        fig,ax=plt.subplots(figsize=(10,5))
        # Modeling windows are separated by real gaps. Do not draw across them.
        gap=np.r_[False,np.abs(np.diff(x))>max(10,float(np.median(np.abs(np.diff(x))))*5)] if spectral else np.zeros(len(x),bool)
        for k,row in enumerate(h):
            xx=[]
            yy=[]
            for i,value in enumerate(row):
                if gap[i]:
                    xx.append(np.nan)
                    yy.append(np.nan)
                xx.append(x[i])
                yy.append(value)
            ax.plot(xx,yy,label=f'F{k+1}')
        ax.set(xlabel=xlabel,ylabel='Factor profile (recorded scaling)',title=f'{rank} fitted components')
        if spectral:
            ax.invert_xaxis()
        ax.legend(ncol=min(rank,6))
        fig.tight_layout()
        name=f'k{rank:02d}_profiles.png'
        fig.savefig(out/name)
        plt.close(fig)
        figures.append((name,f'{rank}-factor profiles; labels are not physical source identities.'))
        fig,ax=plt.subplots(figsize=(10,5))
        for k in range(rank):
            ax.scatter(time,g[:,k],s=12,label=f'F{k+1}')
        ax.set(xlabel='Sampling date' if dates is not None else 'Sample index',ylabel='Spectral amplitude (not source mass)',title=f'{rank}-factor per-filter contributions')
        ax.legend(ncol=min(rank,6))
        fig.autofmt_xdate() if dates is not None else None
        fig.tight_layout()
        name=f'k{rank:02d}_contributions.png'
        fig.savefig(out/name)
        plt.close(fig)
        figures.append((name,'Each point is one retained filter; sampling gaps are not connected.'))
        fig,ax=plt.subplots(figsize=(10,4.6))
        ax.plot(x,np.sqrt(np.mean(r*r,axis=0)))
        ax.axhline(3,linestyle='--',label='3 supplied uncertainty units')
        ax.set(xlabel=xlabel,ylabel='RMS scaled residual',title=f'{rank}-factor residuals by feature')
        if spectral:
            ax.invert_xaxis()
        ax.legend()
        fig.tight_layout()
        name=f'k{rank:02d}_residuals.png'
        fig.savefig(out/name)
        plt.close(fig)
        figures.append((name,'Residuals use the supplied uncertainty matrix; small values do not validate its construction.'))
    if len(models)>1:
        keys=sorted(models,key=int)
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.plot([int(k) for k in keys],[models[k]['metrics']['objective']/data.X.size for k in keys],marker='o')
        ax.set(xlabel='Factor count',ylabel='Selected objective / observed cell',title='Rank screening, not automatic source-count selection')
        fig.tight_layout()
        name='rank_screening.png'
        fig.savefig(out/name)
        plt.close(fig)
        figures.append((name,'Training loss generally improves with more factors; rank selection also requires stability and interpretation.'))
    return figures


def build_report(run, *, charts=True):
    run=Path(run)
    if json.loads((run/'manifest.json').read_text()).get('schema') == 'aethmodular-pmf-projection/2':
        return build_projection_report(run, charts=charts)
    manifest,data,models=verify_run(run)
    out=run/'report'
    out.mkdir(exist_ok=True)
    figures=_figures(out,data,models) if charts and models else []
    dates=data.metadata.get('dates')
    date_range=f'{min(dates)} to {max(dates)}' if dates is not None and len(dates) else 'Not supplied'
    engine=manifest['engine']
    sections=[]
    def section(title,text):
        sections.append((title,text))
    section('Execution summary',f"Dataset: {manifest['dataset_id']}\n\nEngine: {engine}\n\nStatus: {manifest['status']}\n\n"
        f"Retained samples: {len(data.X)}; measured variables: {data.X.shape[1]}; observed cells: {data.X.size}.\n\n"
        f"Sampling period: {date_range}.\n\nStarting-model counts: {json.dumps(manifest.get('counts',{}),sort_keys=True)}.")
    section('What the numbers mean','The model approximates X = G H. Independent PMF constrains both G and H to be nonnegative while retaining signed observations. '
        'ESAT semi-NMF can allow signed G; that is a different constraint set.\n\n'
        'Q(true) = sum(((X - G H) / U)^2). The independent Huber objective is r² inside ±δ and 2δ|r| − δ² outside. '
        'Q_robust_excluded_4 omits squared residuals outside ±4; it is a displayed diagnostic, not the Huber optimization objective.\n\n'
        'The independent solver normalizes H rows with paired G rescaling to preserve G H. Imported native outputs retain their recorded scaling. G is not automatically source mass in µg/m³. '
        'No physical source names, source mass percentages or ME-2 equivalence are established by this report.')
    section('Input provenance and uncertainty',json.dumps({'dataset_identity':data.identity,'compatibility':data.compatibility,
        'protocol':data.protocol,'source_hashes':data.sources},indent=2,sort_keys=True)+'\n\n'
        'The protocol above is the actual input record. A missing/unknown protocol is not a verified laboratory method. '
        'Blank SD is an exploratory noise model where so recorded, not an independently validated full analytical uncertainty budget. '
        'Correlated FTIR channels make a naïve independent chi-square interpretation inappropriate.')
    rows=[]
    for key,result in sorted(models.items(),key=lambda kv:int(kv[0])):
        m=result['metrics']
        sel=manifest['selected'][key]
        rows.append({'factors':int(key),'seed':sel.get('seed'),'status':sel['status'],'objective':sel['objective'],'objective_name':sel.get('objective_name','recorded diagnostic'),
            'common_quadratic_or_Huber_diagnostic':m['objective'],'Q_true':m['Q_true'],
            'scaled_RMSE':m['weighted_rmse_true'],'residual_fraction_abs_gt3':m['fraction_abs_scaled_residual_gt3'],
            'gradient':m.get('relative_projected_gradient_conditioned'),'negative_G_fraction':float((result['G']<0).mean())})
    section('Selected models',json.dumps(rows,indent=2)+'\n\n'
        'A converged start is preferred when available; capped and failed starts are retained in starts.csv and per-seed receipts. '
        'Local numerical convergence does not establish a global minimum, rotational uniqueness, or an appropriate number of sources. '
        'Seed matching uses normalized profile similarity; the original labels are not transferred to newly fitted factors.')
    if models:
        first=next(iter(models.values()))['metrics']
        lower=first.get('negative_observation_Q_lower_bound')
        section('Signed observations and model adequacy',f"Negative input fraction: {float((data.X<0).mean()):.6f}. "
            f"Input fraction below −3 supplied uncertainty units: {float((data.X/data.U < -3).mean()):.6f}.\n\n"+
            (f"For nonnegative reconstruction, negative observations alone impose a scaled-RMSE lower bound of {np.sqrt(lower/data.X.size):.6g}. "
             'This is a mathematical bound, not the observed share of residual loss attributable to negative cells. ' if lower is not None else
             'The nonnegative-reconstruction bound is not applicable to signed-contribution semi-NMF. ')+
            'Systematic baseline residuals and uncertain blank coverage require scientific review; adding factors or inflating U to obtain a preferred score is not a correction.')
    provenance={k:v['provenance'] for k,v in models.items()}
    section('Engine identity and limitations',('Native EPA selection uses its reported Q_main; the common quadratic diagnostic is not its robust objective.\n\n' if engine=='epa-pmf5' else '')+json.dumps(provenance,indent=2,sort_keys=True)+'\n\n'+
        '\n'.join(manifest.get('warnings',[])))
    section('Review before scientific release','Review the preparation inventory, excluded/held filters, blank matching, baseline choice, signed residuals, '
        'seed/rank sensitivity and compatibility with prior versions. Confirm supporting chemical and meteorological evidence before naming factors. '
        'A total-OM calibration from related FTIR data is not independent validation of source-specific allocations. '
        'Native EPA results require genuine native exports and settings; ESAT repairs must remain explicitly identified. '
        'This report does not add uncertainty analyses, mass calibration or validation results that were not executed for this run.')
    md=['# PMF run report','']
    html_parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>PMF run report</title><style>body{font:16px/1.6 system-ui;margin:40px auto;max-width:1080px;padding:0 24px}h1,h2{line-height:1.2}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#f5f5f5}img{max-width:100%}section{margin:2.2em 0}figcaption{font-size:.9em}</style><h1>PMF run report</h1>']
    for title,text in sections:
        md.extend([f'## {title}',text,''])
        html_parts.append(f'<section><h2>{html.escape(title)}</h2><pre>{html.escape(text)}</pre></section>')
    for name,caption in figures:
        md.extend([f'![{caption}]({name})',''])
        html_parts.append(f'<figure><img src="{name}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    md.extend(['## Inspectable evidence','`../manifest.json`, `../request.json`, `../input.json`, and each selected model’s profile/contribution/residual tables.'])
    (out/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (out/'index.html').write_text('\n'.join(html_parts)+ '</html>',encoding='utf-8')
    write_json(out/'report_receipt.json',{'status':'passed','verified_run_manifest_sha256':sha256(run/'manifest.json'),
        'report_builder_sha256':sha256(__file__),'figures_generated':len(figures),
        'files':{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='report_receipt.json'}})
    return out/'index.html'


def build_projection_report(run, *, charts=True):
    run = Path(run)
    m = json.loads((run/'manifest.json').read_text())
    if m.get('schema') != 'aethmodular-pmf-projection/2':
        raise ValueError('Expected a fixed-profile projection')
    for name, digest in m['files'].items():
        if sha256(run/name) != digest:
            raise ValueError(f'Projection artifact changed: {name}')
    data = load_dataset(run/'input.npz')
    ids, factors, g = read_matrix(run/'contributions.csv')
    hf, features, h = read_matrix(run/'profiles.csv')
    if not np.array_equal(ids, data.sample_ids) or not np.array_equal(hf, factors) or not np.array_equal(features, data.features):
        raise ValueError('Projection identities differ')
    from .solver import array_sha256, diagnostics
    if array_sha256(h) != m['frozen_H_sha256']:
        raise ValueError('Frozen projection profiles changed')
    actual = diagnostics(data.X, data.U, g, h, m['projection_info']['loss'], m['projection_info']['delta'])
    if not np.isclose(actual['objective'], m['metrics']['objective'], rtol=1e-9, atol=1e-8):
        raise ValueError('Projection objective changed')
    out = run/'report'
    out.mkdir(exist_ok=True)
    flags = read_records(run/'review_flags.csv')
    reviews = sum(row['residual_review']=='True' for row in flags)
    extrapolated = sum(row['outside_reference_date_range']=='True' for row in flags)
    text = f"""# Fixed-profile projection report

Samples: {len(data.X)}; variables: {data.X.shape[1]}; frozen profiles: {len(h)}.

Status: {m['status']}. Profiles unchanged: {m['profile_unchanged']}.

{reviews} samples exceed the RMS standardized-residual review threshold of {m['threshold']}.
{extrapolated} samples fall outside the reference sampling date range.
These are review flags, not validated probabilities of incorrect source assignment.

The reference profiles were not refitted. This is conditional projection, not
a new independent source-apportionment solution or a mass calibration. Units,
feature order and preparation protocol must be compatible with the reference.
Unknown protocol acknowledged: {m['unknown_protocol_acknowledged']}.

Metrics:
{json.dumps(actual, indent=2)}

Reference dataset: {m['reference_dataset_identity']}
Reference model checksum: {m['reference_model_sha256']}
Frozen profile checksum: {m['frozen_H_sha256']}
"""
    figures = _figures(out, data, {str(len(h)): {'G': g, 'H': h}}) if charts else []
    (out/'report.md').write_text(text + '\n'.join(f'![{caption}]({name})' for name, caption in figures) + '\n')
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>PMF projection</title><body><pre>' + html.escape(text) + '</pre>'
    body += ''.join(f'<figure><img style="max-width:100%" src="{name}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>' for name, caption in figures)
    (out/'index.html').write_text(body+'</body></html>')
    write_json(out/'report_receipt.json', {'status':'passed','verified_run_manifest_sha256':sha256(run/'manifest.json'),
        'report_builder_sha256':sha256(__file__),'files':{p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='report_receipt.json'}})
    return out/'index.html'


def build_pipeline_report(root):
    """Cohort dispositions and real execution states, derived from this pipeline."""
    root=Path(root)
    m=json.loads((root/'pipeline.json').read_text())
    sections=['# PMF pipeline report','',f"Overall status: {m['status']}",'',m['warning'],'']
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PMF pipeline report</title><body style="font:16px/1.5 system-ui;max-width:1000px;margin:2em auto;padding:1em"><h1>PMF pipeline report</h1>',f'<p>Status: {html.escape(m["status"])}</p>']
    for d in m['datasets']:
        name=d['dataset_id']
        summary=f"{name}: preparation {d['preparation_status']}; dispositions {json.dumps(d['counts'],sort_keys=True)}"
        sections += [f'## {name}',summary,'']
        body += [f'<section><h2>{html.escape(name)}</h2><p>{html.escape(summary)}</p>',f'<p><a href="{html.escape(name)}/prepared/inventory.csv">Filter inventory</a></p>']
        for r in d['runs']:
            line=f"{r['name']}: engine {r['engine']}, status {r['status']}"
            link=f"{name}/{r['name']}/report/index.html"
            sections.append(line)
            body.append(f'<p>{html.escape(line)}'+(f' — <a href="{html.escape(link)}">Run report</a>' if (root/link).is_file() else '')+'</p>')
        body.append('</section>')
    sections += ['', 'Numerical convergence, source interpretation, and calibrated source mass are separate gates. Held filters remain in the inventory; they are not silently dropped.']
    (root/'report.md').write_text('\n'.join(sections)+'\n')
    (root/'index.html').write_text('\n'.join(body)+'</body></html>')
    return root/'index.html'
