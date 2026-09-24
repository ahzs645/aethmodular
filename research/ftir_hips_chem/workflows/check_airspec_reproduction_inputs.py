"""Run only the bounded AIRSpec R-port check against hashed inputs, into fresh outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=json.loads(args.manifest.read_text());inputs=source['inputs']
    def check():
        for i in inputs:
            if i['status']!='present_hashed' or sha(i['path'])!=i['sha256']:raise ValueError(f"Input absent or changed: {i['path']}")
    check()
    if args.output.exists():raise ValueError('Output already exists; choose a new directory')
    raw=next(i['path'] for i in inputs if Path(i['path']).name=='ETAD_FTIR_spectra.csv')
    truth=next(i['path'] for i in inputs if Path(i['path']).name=='spectra_baselined_AIRSPEC.csv')
    args.output.mkdir(parents=True)
    command=[sys.executable,str(ROOT/'research/ftir_ec_phase3/scripts/validate_airspec_port.py'),'--raw',raw,'--truth',truth,'--output',str(args.output/'validation.csv'),'--jobs','2']
    run=subprocess.run(command,capture_output=True,text=True)
    (args.output/'stdout.txt').write_text(run.stdout);(args.output/'stderr.txt').write_text(run.stderr)
    check()
    summary={}
    for line in run.stdout.splitlines():
        key,sep,value=line.partition(': ')
        if sep:
            try:summary[key]=float(value)
            except ValueError:summary[key]=value
    numeric_complete=all(k in summary for k in ['max_abs_err','median_max_abs_err','median_ratio'])
    result={'executed_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,'returncode':run.returncode,
        'manifest_sha256':sha(args.manifest),'workflow_sha256':sha(__file__),'source_unchanged':True,
        'summary':summary,'historical_absolute_criterion_pass':numeric_complete and summary['max_abs_err']<=2e-4 and summary['median_max_abs_err']<=2e-5,
        'current_relative_criterion_pass':numeric_complete and summary['median_ratio']<=0.02,
        'scope':'R-port numerical check only. Full locked calibration and independent Addis validation are not executed.',
        'outputs':{p.name:sha(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if run.returncode or not numeric_complete:raise SystemExit(1)

if __name__=='__main__':main()
