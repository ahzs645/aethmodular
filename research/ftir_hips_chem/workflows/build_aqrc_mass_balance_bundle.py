"""Build the scoped, credential-free Davis transfer bundle for the mass-balance gaps."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from config import BASE_FILTER_ID_PATTERN
from data_matching import base_filter_id

OUT = ROOT / "research/ftir_hips_chem/output/tables/spartan_mass_balance_audit"
source = OUT / "audit.json"
audit = json.loads(source.read_text())
folder = OUT / "aqrc_mass_balance_pull"
folder.mkdir(exist_ok=True)
files = [
    ROOT / "scripts/diagnostics/inspect_sql_access.ps1",
    ROOT / "research/ftir_ec_phase3/scripts/export_spartan_mass_balance_followup.ps1",
]
for path in files:
    shutil.copyfile(path, folder / path.name)
records = audit["records"]
ids = [base_filter_id(r["id"]) for r in records]
assert len(ids) == len(set(ids))
with (folder / "requested_filters.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["FilterId", "Site", "Date", "GapReason"])
    writer.writeheader()
    writer.writerows({"FilterId": base_filter_id(r["id"]), "Site": r["site"], "Date": r["date"], "GapReason": r["reason"]} for r in records)
(folder / "request_context.json").write_text(json.dumps({
    "scope": audit["scope"], "requested_filters": len(ids),
    "source_audit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "base_filter_id_pattern": BASE_FILTER_ID_PATTERN,
    "notes": "Candidate external replicate IDs must map through the canonical physical-filter helper. Raw observations do not establish a final chemical reconstruction.",
}, indent=2) + "\n")
runner = r'''# Reviewed launcher. Does not change execution policy or connection settings.
$ErrorActionPreference='Stop'
foreach ($entry in @(Import-Csv (Join-Path $PSScriptRoot 'bundle_manifest.csv'))) {
    if ([IO.Path]::GetFileName($entry.File) -ne $entry.File) { throw 'Invalid bundle path.' }
    if ((Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $entry.File) -Algorithm SHA256).Hash -ne $entry.SHA256) { throw 'Bundle hash mismatch.' }
}
foreach ($name in @('inspect_sql_access.ps1','export_spartan_mass_balance_followup.ps1')) {
    $tokens=$null;$parseErrors=$null
    [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $name),[ref]$tokens,[ref]$parseErrors) | Out-Null
    if ($parseErrors.Count) { throw ('PowerShell parse error in '+$name) }
}
. (Join-Path $PSScriptRoot 'export_spartan_mass_balance_followup.ps1')
$request=Join-Path $PSScriptRoot 'requested_filters.csv'
$ids=@(Import-Csv $request | ForEach-Object { $_.FilterId })
$plan=@(Get-SpartanMassBalancePlan $ids)
if ($plan.Count -ne 9) { throw 'Unexpected query-plan count.' }
$rejected=$false
try { Get-SpartanMassBalancePlan @("ETAD-0001'; SELECT 1;") | Out-Null } catch { $rejected=$true }
if (-not $rejected) { throw 'Invalid filter identity was accepted.' }
$zip=Invoke-SpartanMassBalanceExport -RequestCsv $request
$runFolder=[IO.Path]::ChangeExtension($zip,$null)
$receipt=Get-Content (Join-Path $runFolder 'receipt.json') -Raw | ConvertFrom-Json
if ($receipt.Status -ne 'complete' -or $receipt.Errors -ne 0) { throw 'Partial export retained: inspect exports.csv and errors.csv.' }
Write-Host 'MASS_BALANCE_EXPORT_COMPLETE'
Write-Host ('Archive: '+$zip)
$zip
'''
(folder / "run.ps1").write_text(runner, encoding="utf-8")
with (folder / "bundle_manifest.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["File", "Bytes", "SHA256"])
    writer.writeheader()
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != "bundle_manifest.csv":
            writer.writerow({"File": path.name, "Bytes": path.stat().st_size, "SHA256": hashlib.sha256(path.read_bytes()).hexdigest().upper()})
archive = OUT / "aqrc_mass_balance_pull.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
    for path in sorted(folder.iterdir()):
        stream.write(path, f"{folder.name}/{path.name}")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
(OUT / "aqrc_mass_balance_pull.zip.sha256.txt").write_text(f"{digest}  {archive.name}\n")
print(json.dumps({"requested_filters": len(ids), "archive": str(archive.relative_to(ROOT)), "sha256": digest}, indent=2))
