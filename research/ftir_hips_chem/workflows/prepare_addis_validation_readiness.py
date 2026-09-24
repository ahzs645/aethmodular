"""Inventory local validation evidence and prepare an unsent identity/data request."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/ftir_ec_phase3/scripts'))
from theory_test_suite import davis_root

def main():
    data=davis_root();out=ROOT/'research/ftir_hips_chem/output/tables/addis_validation_readiness';out.mkdir(parents=True,exist_ok=True)
    inventory=subprocess.run(['rg','--files',str(data)],capture_output=True,text=True,check=True).stdout.splitlines()
    candidates=[p for p in inventory if any(w in str(Path(p).relative_to(data)).lower() for w in ['adama','quartz','thermal','crosswalk','batch54','batch_54','etad_metadata','results_tor'])]
    (out/'candidate_file_inventory.txt').write_text('\n'.join(candidates)+'\n')
    (out/'search_scope.json').write_text(json.dumps({'root':str(data),'listed_files':len(inventory),'candidate_files':len(candidates),'scope':'Local mounted Davis Data tree only; does not establish absence from lab systems, email or other locations.'},indent=2)+'\n')
    names=['DAVIS/Adama TOR/Carbon_concs_Batch54.csv','DAVIS/Adama TOR/OC_EC_concs_Batch54.csv','DAVIS/CSU_AMOD/csu_amod_FTIR_Batch_54.csv','DAVIS/CSU_AMOD/csu_amod_HIPS_Batch_54.csv','DAVIS/CSU_AMOD/csu_amod_Batch_54_ShipDate_2026-05-29_spectra.csv','DAVIS/ETAD FTIR/ETAD_metadata.csv','FTIR/local_db/tables/results_tor.csv']
    summary=[]
    for name in names:
        p=data/name;d=pd.read_csv(p,encoding='cp1252' if 'HIPS_Batch' in name or 'FTIR_Batch' in name else 'utf-8-sig',low_memory=False)
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        entry={'path':str(p),'sha256':h,'rows':len(d),'columns':d.columns.tolist()}
        if 'Site' in d:entry['ETAD_rows']=int(d.Site.astype(str).str.upper().eq('ETAD').sum());entry['sites']=d.Site.dropna().unique().tolist()
        if 'Parameter' in d:entry['parameters']=d.Parameter.dropna().unique().tolist()
        summary.append(entry)
    (out/'inspected_sources.json').write_text(json.dumps(summary,indent=2)+'\n')
    pairs=pd.read_csv(ROOT/'research/ftir_ec_phase3/output/tables/ftir41/pairing_ledger.csv')
    pairs['pairing_status']='date_candidate_only_sampling_equivalence_unconfirmed'
    pairs['spectrum_id_status']='authoritative_4744_to_4748_crosswalk_required'
    pairs['independent_addis_reference']=False
    pairs.to_csv(out/'adama_pairing_confirmation_queue.csv',index=False)
    (out/'data_request_draft.md').write_text('''# Data and identity request — draft, not sent

Please provide independently measured thermal EC for Addis/ETAD, if it exists, with physical quartz filter IDs, TOR primary and TOT sensitivity values, laboratory protocol, units, blank correction, detection limits, uncertainty, replicate information and QA flags. Include the corresponding PTFE filter IDs and sampling records; FTIR-predicted EC or HIPS/MAC cannot substitute for the thermal measurement.

For CSU AMOD Batch 54, please supply the authoritative mapping from each spectral row / SampleAnalysisId 4744–4748 to MediaId, physical PTFE FilterId and laboratory analysis record. Existing exports lack this crosswalk. Please confirm from laboratory records, not spectrum peak rank or assumed row order.

Please confirm quartz/PTFE sampling equivalence and time zone for the five July 2024 Adama candidate pairs listed in `adama_pairing_confirmation_queue.csv`. In particular, J1269/J1693 on July 9 have a 39.73-minute start offset, and J1270/J1679 on July 30 have a PTFE/quartz volume ratio of 0.456. Supply start/end times, active intervals and sampler-specific volumes; explain the flagged differences before marking the pairs eligible.

For a new Addis campaign, lock pairing eligibility and model predictions before unblinding thermal results. The existing 36-pair seasonal design remains a proposal, not observations already acquired.
''')
    (out/'report.md').write_text(f'''# Independent Addis validation readiness

Status: **blocked on independent measurements and authoritative identities**.

A filename inventory covered {len(inventory)} files under the mounted Davis Data root. Seven relevant exports were read and hashed. The IMPROVE TOR export has 1,926,865 rows and zero Site=ETAD rows. The local thermal campaign files describe five Adama filters, not an Addis reference population. This search is bounded to the local data tree and cannot establish absence from laboratory systems or correspondence.

The spectral export has five rows labelled 4744–4748; it does not contain an authoritative physical-FilterId mapping. Later repository overlay work explicitly retains those row IDs instead of guessing dates from order or peak height. The historical rank-based mapping remains provisional.

The [five-row confirmation queue](adama_pairing_confirmation_queue.csv) preserves both sampling flags. Its rows are candidate pairs, not certified matches or independent Addis observations. No new exclusion was imposed on a scientific dataset.

[Unsent data/identity request](data_request_draft.md) · [Inspected source hashes and columns](inspected_sources.json) · [Search scope](search_scope.json).

No thermal accuracy metric was computed using HIPS or FTIR predictions as a substitute reference. The next executable validation step requires the requested independent values and confirmed identities.
''')
    print('Readiness report, five-pair confirmation queue and unsent data request prepared.')

if __name__=='__main__':main()
