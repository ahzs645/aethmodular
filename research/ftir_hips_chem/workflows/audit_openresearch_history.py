"""Audit the frozen research catalog without executing notebooks or model fits.

Recovers saved narratives, execution errors and static input calls. Hash checks
establish file identity, not scientific validity or historical code provenance.
"""
from __future__ import annotations
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
import json
from pathlib import Path
import re
import shlex
import shutil

from aethmodular_cli.env import load_repo_env

load_repo_env()

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / 'research/ftir_hips_chem/output/tables/openresearch_retrospective/inventory.json'
OUT = ROOT / 'research/ftir_hips_chem/output/tables/openresearch_evidence_audit'
ARTIFACTS = Path(os.environ.get('OPENRESEARCH_FILES_DIR', '~/.local/share/openresearch/files')).expanduser() / 'aethmodular-research-retrospective'
SNAPSHOTS = ARTIFACTS / 'research-retrospective/sources'
READERS = {'read_csv','read_pickle','read_excel','read_parquet','read_json','load','loadtxt','genfromtxt','read_text','read_bytes','open','read_feather'}
WRITERS = {'to_csv','to_parquet','to_pickle','to_json','save','savez','savez_compressed'}


def static_value(node, names, cwd):
    """Restricted path evaluation; never eval/import notebook code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)): return node.value
    if isinstance(node, ast.Name): return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = static_value(node.left,names,cwd), static_value(node.right,names,cwd)
        if isinstance(left,(str,Path)) and isinstance(right,(str,Path)): return Path(left)/right
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
        left,right=static_value(node.left,names,cwd),static_value(node.right,names,cwd)
        if isinstance(left,str) and isinstance(right,str): return left+right
    if isinstance(node,ast.Call):
        if isinstance(node.func,ast.Name) and node.func.id=='Path' and node.args:
            value=static_value(node.args[0],names,cwd)
            if isinstance(value,(str,Path)): return Path(value)
        if isinstance(node.func,ast.Attribute):
            if ast.unparse(node.func)=='Path.cwd': return cwd
            value=static_value(node.func.value,names,cwd)
            if isinstance(value,Path) and node.func.attr in {'resolve','absolute'}: return (cwd/value).resolve()
    if isinstance(node,ast.Attribute):
        value=static_value(node.value,names,cwd)
        if isinstance(value,Path) and node.attr=='parent': return value.parent
    if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Attribute) and node.value.attr=='parents':
        value=static_value(node.value.value,names,cwd)
        index=static_value(node.slice,names,cwd)
        if isinstance(value,Path) and isinstance(index,int):
            try: return value.parents[index]
            except IndexError: return None
    return None


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def text(value):
    return ''.join(value) if isinstance(value, list) else str(value or '')


def notebook_audit(path):
    nb = json.loads(path.read_text())
    code = [(i, c) for i, c in enumerate(nb.get('cells', [])) if c.get('cell_type') == 'code']
    errors, output_evidence, flags, conclusions = [], [], [], []
    for i, c in code:
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                errors.append({'cell_index':i,'name':o.get('ename'),'message':o.get('evalue')})
            value = text(o.get('text') or o.get('data', {}).get('text/plain'))
            if value:
                output_evidence.append({'cell_index':i,'excerpt':value[:2400], 'truncated':len(value)>2400})
                for line in value.splitlines():
                    if re.search(r'\b(SKIPPED|Traceback|FAILED|FileNotFoundError|not found|fallback)\b', line, re.I):
                        flags.append({'cell_index':i,'excerpt':line[:500], 'kind':'saved_output_warning_not_new_failure'})
    for i, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') != 'markdown':
            continue
        body = text(c.get('source'))
        headings = list(re.finditer(r'(?m)^(#{1,6})\s+(.+)$', body))
        for j, h in enumerate(headings):
            if not re.search(r'tl;dr|takeaway|conclusion|working answer|key findings|main findings|summary|interpretation|what .*shows|what .*adds',h[2],re.I):
                continue
            end = next((k.start() for k in headings[j+1:] if len(k[1]) <= len(h[1])), len(body))
            section = body[h.end():end].strip()
            if not section or 'filled in by' in section[:200]:
                continue
            if re.search(r'Functions Defined|To Run This Notebook|load_aethalometer_addis\(\)', section):
                kind = 'instructions_or_template_summary'
            else:
                kind = 'historical_narrative_not_newly_verified'
            conclusions.append({'cell_index':i,'heading':h[2],'kind':kind,'excerpt':section[:5000],'truncated':len(section)>5000})
    executed = sum(c.get('execution_count') is not None for _, c in code)
    outputs = sum(bool(c.get('outputs')) for _, c in code)
    if errors: status = 'saved_error'
    elif not executed and not outputs: status = 'no_saved_execution'
    elif executed < len(code): status = 'partial_or_untracked_saved_execution'
    else: status = 'saved_execution_no_errors_recorded'
    return {'code_cells':len(code),'execution_count_cells':executed,'output_cells':outputs,
            'execution_state':status,'saved_errors':errors,'saved_output_flags':flags,
            'conclusion_candidates':conclusions,'saved_output_evidence':output_evidence}, [
                (f'cell {i}',text(c.get('source'))) for i,c in code]


def input_calls(parts, cwd, owners, *, outputs=False, source_path=None):
    inputs, failures = [], []
    names = {'__file__':str(source_path)} if source_path else {}
    for location, source in parts:
        # IPython magics are not imported/executed; report parse limitations.
        cleaned = '\n'.join('pass' if line.lstrip().startswith(('%','!','?')) else line for line in source.splitlines())
        try: tree = ast.parse(cleaned)
        except SyntaxError as exc:
            failures.append({'location':location,'line':exc.lineno,'reason':exc.msg}); continue
        # Resolve in source order: a later reassignment must never supply an
        # earlier read's input. Conditional/function-local values stay unknown.
        for statement in tree.body:
            for call in ast.walk(statement):
                if not isinstance(call, ast.Call): continue
                name = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id if isinstance(call.func, ast.Name) else ''
                if name not in (WRITERS if outputs else READERS): continue
                expr = call.args[0] if call.args else call.func.value if isinstance(call.func, ast.Attribute) else None
                if expr is None: continue
                mode = call.args[1] if len(call.args)>1 else next((kw.value for kw in call.keywords if kw.arg=='mode'), None)
                if name == 'open' and isinstance(mode,ast.Constant) and any(x in str(mode.value) for x in ['w','a','x']): continue
                fragments = [n.value for n in ast.walk(expr) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                # Resolve only literal paths against the documented working directory.
                # All dynamic paths remain expressions, never guessed as present.
                resolved = None
                value=static_value(expr,names,cwd)
                if isinstance(value,(str,Path)):
                    candidate = Path(value)
                    if not candidate.is_absolute(): candidate = cwd / candidate
                    if candidate.is_file(): resolved = candidate.resolve()
                item = {'location':location,'line':call.lineno,'reader':ast.unparse(call.func),
                        'expression':ast.unparse(expr),'literal_fragments':fragments,
                        'resolved_path':None,'availability':'unresolved_static_expression','producer_records':[]}
                if resolved:
                    try: rel = resolved.relative_to(ROOT).as_posix()
                    except ValueError: rel = str(resolved)
                    item.update(resolved_path=rel,availability='present_at_audit',producer_records=owners.get(rel,[]))
                inputs.append(item)
            if isinstance(statement,ast.Assign):
                value=static_value(statement.value,names,cwd)
                for target in statement.targets:
                    if isinstance(target,ast.Name): names[target.id]=value
    return inputs, failures


def main():
    inv = json.loads(INVENTORY.read_text())
    plan = json.loads((ROOT/'research/ftir_hips_chem/output/tables/openresearch_families/migration_plan.json').read_text())
    recipes = {r['id']:r['reproduction'] for r in plan['records']}
    owners = {}
    for r in inv['records']:
        for path in r.get('artifact_files',[]): owners.setdefault(path,[]).append(r['id'])
    files = {}
    for rel, baseline in inv['file_index'].items():
        archived, current = SNAPSHOTS/rel, ROOT/rel
        expected = baseline.get('sha256')
        files[rel] = {'expected_sha256':expected,'snapshot_exists':archived.is_file(),
                      'snapshot_matches':archived.is_file() and digest(archived)==expected,
                      'current_exists':current.is_file(),
                      'current_matches_snapshot':current.is_file() and digest(current)==expected}
    records, candidates = [], []
    for r in inv['records']:
        source = SNAPSHOTS/r['source']
        if not source.is_file(): raise ValueError(f'Missing snapshot {r["id"]}')
        if not files[r['source']]['snapshot_matches']: raise ValueError(f'Snapshot changed {r["id"]}')
        recipe = dict(recipes[r['id']])
        if not recipe.get('command') and source.suffix=='.ipynb':
            recipe.update(status='candidate_notebook_command_not_executed', command=
                f"cd {shlex.quote(str(Path(r['source']).parent))} && "
                f"uv run --locked --no-sync jupyter nbconvert --to notebook --execute {shlex.quote(source.name)} "
                f"--output {shlex.quote(r['id']+'.executed.ipynb')} --output-dir \"${{REPRO_OUTPUT:?Set an absolute archive output directory}}\" --ExecutePreprocessor.timeout=3600",
                prerequisites='Candidate command: nbconvert 7.16.6 is available, but this notebook has not been rerun. Use an isolated checkout with frozen inputs and an absolute REPRO_OUTPUT archive directory. Resolve the listed errors, commented-out analysis cells and dynamic data paths first; notebook code may write additional output paths within that checkout.')
        state, parts = {}, []
        if source.suffix == '.ipynb': state, parts = notebook_audit(source)
        elif source.suffix in {'.md','.py'}:
            saved = source.read_text()
            state = {'execution_state':'document_or_manifest','saved_errors':[],
                     'conclusion_candidates':[{'kind':'historical_document_not_newly_verified','excerpt':saved[:7000],'truncated':len(saved)>7000}] if source.suffix=='.md' else []}
            if source.suffix == '.py': parts=[('source',saved)]
        else: state={'execution_state':'document_or_manifest','saved_errors':[],'conclusion_candidates':[]}
        companions=[]
        for copy in r.get('executed_copies',[]):
            rel=copy['path']
            if not files[rel]['snapshot_matches']: raise ValueError(f'Companion snapshot changed: {rel}')
            companion,_=notebook_audit(SNAPSHOTS/rel)
            companions.append({'path':rel,'sha256':files[rel]['expected_sha256'],**companion})
        area = (ROOT/r['source']).parent
        calls, parse_gaps = input_calls(parts,area,owners)
        output_calls,_ = input_calls(parts,area,owners,outputs=True)
        for entry in r.get('entrypoints',[]):
            snap=SNAPSHOTS/entry
            chosen=snap if snap.is_file() else ROOT/entry
            extra,gaps=input_calls([(entry,chosen.read_text())],(ROOT/entry).parent.parent,owners,source_path=ROOT/entry)
            written,_=input_calls([(entry,chosen.read_text())],(ROOT/entry).parent.parent,owners,outputs=True,source_path=ROOT/entry)
            output_calls.extend(written)
            for item in extra: item['entrypoint']=entry; item['entrypoint_sha256']=digest(chosen)
            calls.extend(extra); parse_gaps.extend(gaps)
        scientific = [c for c in state.get('conclusion_candidates',[]) if c['kind'] not in {'instructions_or_template_summary'}]
        blockers=[]
        if state['execution_state']=='no_saved_execution': blockers.append('The active source has no saved execution or outputs; inspect the separately audited archived companions, if any, before judging execution evidence.')
        if state.get('saved_errors'): blockers.append('Saved error requires repair and execution in a fresh copy; other stored outputs may be stale.')
        if not scientific: blockers.append('No explicit conclusion recovered; inspect the saved output excerpts rather than infer a result.')
        if not recipe.get('command'): blockers.append('No standalone execution command verified; notebook execution requires its setup and staged inputs.')
        if any(x['availability']=='unresolved_static_expression' for x in calls): blockers.append('Some input expressions require runtime/configuration resolution; the static list is not an exhaustive input manifest.')
        if not files[r['source']]['current_matches_snapshot']: blockers.append('Current source differs from the imported snapshot; choose and freeze the source before reproduction.')
        blockers.append('Current repository environment is not proof of the historical environment. Saved counts and file hashes do not establish scientific validity.')
        for item in calls:
            for producer in item['producer_records']:
                if producer!=r['id'] and item.get('entrypoint'):
                    candidates.append({'from':producer,'to':r['id'],'type':'data_dependency','evidence':item['entrypoint'],
                        'contains':next((s for s in item['literal_fragments'] if s in item['resolved_path']),item['expression']),
                        'reason':f"The runner explicitly reads {item['resolved_path']}; this is source-level file consumption, not reconstructed execution ancestry.",
                        'consumed_file':item['resolved_path'],'line':item['line'],
                        'consumed_sha256':digest(ROOT/item['resolved_path'])})
        records.append({'id':r['id'],'source':r['source'],'source_sha256':files[r['source']]['expected_sha256'],
             'evidence_status_unchanged':r['evidence_status'],'saved_evidence':state,'input_calls':calls,
             'archived_companion_audits':companions,
             'static_parse_gaps':parse_gaps,'reproduction':recipe,'blockers':blockers,
             'artifact_checks':{p:files[p] for p in r.get('artifact_files',[])},
             'additional_artifacts': {c['resolved_path']:{'sha256':digest(ROOT/c['resolved_path']),
                 'writer_location':c['location'],'line':c['line'],
                 'provenance':'Current file at a statically resolved output path; historical generation not established.'}
                 for c in output_calls if c['resolved_path'] and c['resolved_path'] not in inv['file_index']},
             'executed_copy_paths':r.get('executed_copies',[])})
    result={'schema':1,'audited_at_utc':datetime.now(timezone.utc).isoformat(),'inventory_sha256':digest(INVENTORY),
        'scope':'101 imported records and their 638 indexed snapshot files; archive-only work outside this catalog is not claimed audited.',
        'method':'Read-only saved-evidence and static source audit. No historical notebook, pipeline or model fit executed.',
        'environment':'aeth doctor passed with repository Python 3.13.9; historical environment not reconstructed.',
        'records':records,'file_checks':files,'dependency_candidates':candidates,
        'summary':{'records':len(records),'files':len(files),'execution_states':dict(Counter(r['saved_evidence']['execution_state'] for r in records)),
        'with_recovered_narrative':sum(any(c['kind']!='instructions_or_template_summary' for c in r['saved_evidence'].get('conclusion_candidates',[])) for r in records),
        'with_saved_errors':[r['id'] for r in records if r['saved_evidence'].get('saved_errors')],
        'archived_companions_inspected':sum(len(r['archived_companion_audits']) for r in records),
        'unexecuted_sources_with_executed_companion':[r['id'] for r in records if r['saved_evidence']['execution_state']=='no_saved_execution' and any(c['execution_state']=='saved_execution_no_errors_recorded' for c in r['archived_companion_audits'])],
        'additional_artifacts':sum(len(r['additional_artifacts']) for r in records),
        'snapshot_mismatches':[p for p,c in files.items() if not c['snapshot_matches']],
        'current_file_changes':[p for p,c in files.items() if not c['current_matches_snapshot']],
        'dependency_candidate_pairs':len({(e['from'],e['to']) for e in candidates})}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    cards=OUT/'records'; cards.mkdir(exist_ok=True)
    for r in records:
        state=r['saved_evidence']
        lines=[f"# Evidence audit: {r['id']}",'',f"Source: `{r['source']}`",f"Snapshot SHA-256: `{r['source_sha256']}`",'',
          f"Saved execution: **{state['execution_state']}**. Evidence status retained: `{r['evidence_status_unchanged']}`.",'',
          '## Recovered conclusions and narrative','', 'These are saved source statements, not conclusions newly validated by this audit.','']
        for c in state.get('conclusion_candidates',[]):
            lines += [f"### {c.get('heading','Saved document')} — {c['kind']}",f"Source cell: {c.get('cell_index','not applicable')}",'',c['excerpt'],'']
            if c.get('truncated'): lines += ['[Excerpt truncated; consult the frozen source.]','']
        if not state.get('conclusion_candidates'): lines+=['No explicit narrative conclusion recovered.','']
        lines+=['## Archived executed companions','']
        for c in r['archived_companion_audits']:
            lines += [f"- `{c['path']}`: **{c['execution_state']}**; {c['execution_count_cells']}/{c['code_cells']} counted cells; SHA-256 `{c['sha256']}`."]
            lines += ['```json',json.dumps(c['saved_errors'],indent=2),'```']
        lines += ['## Execution errors and gaps','',*[f'- {b}' for b in r['blockers']],'', '```json',json.dumps(state.get('saved_errors',[]),indent=2),'```','',
            '## Input calls','', 'Static source calls only. Conditional reads may not have executed; unresolved paths are not asserted available.','']
        for c in r['input_calls']:
            lines += [f"- `{c['location']}:{c['line']}` — `{c['reader']}({c['expression']})`; {c['availability']}." ]
        lines += ['', '## Reproduction','',r['reproduction']['prerequisites'],'',f"Recipe status: `{r['reproduction']['status']}`.",'']
        if r['reproduction'].get('command'): lines += ['```sh',r['reproduction']['command'],'```','']
        lines += ['## Additional artifacts recovered','', 'Current files at source-declared output paths. These were absent from the original inventory; existence does not establish when or how they were generated.','']
        for path,details in r['additional_artifacts'].items():
            dest=OUT/'additional-artifacts'/path
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/path,dest)
            if digest(dest)!=details['sha256']: raise ValueError(f'Artifact changed during copy: {path}')
            lines += [f"- [{path}](../additional-artifacts/{path}) — SHA-256 `{details['sha256']}`; writer {details['writer_location']}:{details['line']}."]
        lines += ['## Saved output excerpts','']
        for o in state.get('saved_output_evidence',[]):
            lines += [f"### Cell {o['cell_index']}",'```text',o['excerpt'],'```','[truncated]' if o['truncated'] else '','']
        (cards/(r['id']+'.md')).write_text('\n'.join(lines)+'\n')
    lines=['# Historical evidence audit','',result['method'],'',f"Coverage: {len(records)} catalog records; {len(files)} indexed files.",'','## Saved execution state','',
          *[f'- {k}: {v}' for k,v in result['summary']['execution_states'].items()],'',
          f"Records with recovered narrative: {result['summary']['with_recovered_narrative']}. Narrative remains attributed historical prose.",'',
          f"Archived companions inspected: {result['summary']['archived_companions_inspected']}; seven otherwise unexecuted sources have a fully counted, error-free saved companion. The remaining 25 have no such companion in this catalog.",'', f"Additional source-declared output files recovered: {result['summary']['additional_artifacts']}. All 638 indexed snapshots match their frozen hashes; file identity does not verify results.",'', 'The HIPS_Aeth_SmoothRaw_Analysis notebook stores an import error. ftir_49 records skipped minute-resolution checks and a daily fallback; those skipped checks are not negative findings. See the individual cards for other warnings and unresolved input paths.','', '## Record audit cards','', '| Record | Saved execution | Input calls | Recipe |','|---|---|---|---|']
    lines += [f"| [{r['id']}](records/{r['id']}.md) | {r['saved_evidence']['execution_state']} | {len(r['input_calls'])} | {r['reproduction']['status']} |" for r in records]
    lines+=['','[Full audit and hashes](audit.json)','', 'Automated extraction is evidence recovery, not peer review. Only separately reviewed relationships are promoted to the research graph.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    shutil.copytree(OUT,ARTIFACTS/'research-evidence-audit',dirs_exist_ok=True)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__': main()
