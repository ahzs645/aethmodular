"""Publish reviewed prospective contracts; never fabricate an execution/run row."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from migrate_openresearch_families import ARTIFACTS, ROOT, PROJECT, cli, dump, experiments, file_url, node_url, sha

SOURCE=ROOT/'docs/openresearch-retrospective/experiments'
OUT=ROOT/'research/ftir_hips_chem/output/tables/openresearch_next_experiments'
GUARD="printf '%s\\n' 'Research contract. Read its readiness gate and use a fresh output directory or isolated checkout.' >&2; exit 2"


def main():
    contracts=json.loads((SOURCE/'contracts.json').read_text())['experiments']
    historical=json.loads((ROOT/'research/ftir_hips_chem/output/tables/openresearch_families/migration_receipt.json').read_text())['ids']
    target=ARTIFACTS/'research-next-experiments'
    shutil.copytree(SOURCE,target,dirs_exist_ok=True)
    for page in target.glob('*.md'):
        page.write_text(page.read_text().replace('](../../../research/ftir_hips_chem/output/tables/vibes_loading_trace/','](vibes-results/').replace('](../../../research/ftir_hips_chem/output/tables/airspec_reproduction_preflight/','](airspec-results/'))
    additions={'airspec_locked_reproduction':'airspec-locked-results','vibes_case_investigation':'vibes-case-results','addis_validation_readiness':'addis-readiness','historical_gap_followthrough':'historical-followthrough'}
    for directory,destination in additions.items():
        shutil.copytree(ROOT/'research/ftir_hips_chem/output/tables'/directory,target/destination,dirs_exist_ok=True)
    for page in target.glob('*.md'):
        body=page.read_text()
        for directory,destination in additions.items():
            body=body.replace('](../../../research/ftir_hips_chem/output/tables/'+directory+'/',']('+destination+'/')
        page.write_text(body)
    shutil.copytree(ROOT/'research/ftir_hips_chem/output/plots/vibes_case_investigation',target/'vibes-case-results/figures',dirs_exist_ok=True)
    case_report=target/'vibes-case-results/report.md'
    case_report.write_text(case_report.read_text().replace('../../plots/vibes_case_investigation/','figures/'))
    shutil.copy2(ROOT/'research/ftir_hips_chem/notebooks/archive/executed/HIPS_Aeth_SmoothRaw_Analysis_repaired_20260921.executed.ipynb',target/'historical-followthrough/repaired.executed.ipynb')
    diagnostic=ROOT/'research/ftir_hips_chem/output/tables/vibes_loading_trace'
    shutil.copytree(diagnostic,target/'vibes-results',dirs_exist_ok=True)
    shutil.copytree(ROOT/'research/ftir_hips_chem/output/tables/airspec_reproduction_preflight',target/'airspec-results',dirs_exist_ok=True)
    shutil.copy2(ROOT/'research/ftir_hips_chem/workflows/check_airspec_reproduction_inputs.py',target/'check_airspec_reproduction_inputs.py')
    # Keep the exact new diagnostic source alongside the run evidence.
    shutil.copy2(ROOT/'research/ftir_hips_chem/workflows/trace_vibes_loading_predictions.py',target/'trace_vibes_loading_predictions.py')
    before=experiments();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'pre_registration.json').exists():dump(OUT/'pre_registration.json',before)
    branch=subprocess.check_output(['git','symbolic-ref','HEAD'],text=True).strip()
    ids={}
    for contract in contracts:
        key=contract['id'];marker=f'Import key: `aeth-plan-v1:{key}`';parent=historical['family_'+contract['family']]
        matches=[x for x in experiments() if marker in (x.get('description') or '')]
        if not matches:
            cli('create-experiment',PROJECT,'--title',contract['title'],'--description',marker,
                '--parent',parent,'--run-command',GUARD)
            matches=[x for x in experiments() if marker in (x.get('description') or '')]
        if len(matches)!=1 or matches[0]['parentExperimentId']!=parent or matches[0]['runCommand']!=GUARD:
            raise ValueError(f'Unexpected contract registration: {key}')
        ids[key]=matches[0]['id']
    for contract in contracts:
        key=contract['id']
        desc=[f'Import key: `aeth-plan-v1:{key}`','',f"**Experiment status:** `{contract['status']}`.",'',
            f"Family: [{contract['family']}]({file_url('research-families/families/'+contract['family']+'.md')})",'',
            '## Question or purpose','',contract['question'],'','## Protocol and readiness','',
            f"[Full question, fixed inputs, comparison and success criteria]({file_url('research-next-experiments/'+contract['file'])}).",'',contract['readiness'],'',
            '## Relationships','']
        for source in contract['sources']:
            if source not in historical:raise ValueError(source)
            kind='data_dependency' if source in contract.get('data_sources',[]) else 'related_research'
            reason=contract.get('dependency_reason') or 'The completed local trace reads the frozen corrected spectra, portable models, case identities and held-out predictions from this completed Colab comparison; input hashes are recorded in its manifest.' if kind=='data_dependency' else 'Historical motivation/context for this prospective protocol; not claimed execution ancestry.'
            desc += [f'- **{kind}**: [{source} → {key}]({node_url(historical[source])}). {reason}']
        if key=='next_vibes_loading_trace':
            desc += ['','## Local diagnostic evidence','',f"[Completed local diagnostic and limitations]({file_url('research-next-experiments/vibes-results/report.md')}). Executed locally against frozen saved models; no fitting and no new OpenResearch run row. Physical mechanism and future-method validation remain open.",f"[Case investigation and repeated corrections]({file_url('research-next-experiments/vibes-case-results/report.md')})."]
        if native := contract.get('native_execution'):
            desc += ['', '## Executable continuation', '',
                     f"[Open the native experiment]({native['experiment_url']}). "
                     f"Verified run `{native['run_id']}` at commit `{native['commit']}`: {native['result']}",
                     f"[Execution report and evidence]({native['report_url']}). "
                     'The completed run belongs to the linked execution project. This catalog record keeps its original evidence identity.']
            execution = 'This catalog entry has a guard command. Use the linked native experiment for new OpenResearch runs; its committed execution recipe replaces the earlier manual staging procedure.'
        else:
            execution = 'The registration command is a guard. Use the contract’s checked command and readiness gate in a separate output directory or isolated checkout. No historical or prospective result is certified by the Git parent.'
        desc += ['','## Execution','', execution]
        desired='\n'.join(desc)+'\n'
        current=next(x for x in experiments() if x['id']==ids[key])
        if current.get('description')!=desired:
            cli('exp','desc',ids[key],'--stdin',input_text=desired)
    after=experiments();old={x['id']:x for x in before};new={x['id']:x for x in after}
    for id,item in old.items():
        if id not in ids.values() and item!=new[id]:raise ValueError(f'Existing record changed during plan registration: {id}')
    if subprocess.check_output(['git','symbolic-ref','HEAD'],text=True).strip()!=branch:raise ValueError('Checkout changed')
    receipt={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'ids':ids,'native_nodes':len(after),
        'new_run_rows_created':0,'contracts_sha256':sha(SOURCE/'contracts.json'),
        'published_files':{str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file() and p.name!='registration_receipt.json'}}
    dump(OUT/'registration_receipt.json',receipt);dump(target/'registration_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='published_files'},indent=2))


if __name__=='__main__':main()
