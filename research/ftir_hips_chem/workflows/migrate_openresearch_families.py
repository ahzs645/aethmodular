"""Build and optionally import historical research families through the orx CLI.

Creates navigation records, never runs analyses or assigns historical commits.
Re-execution reconciles stable import keys to avoid duplicate nodes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

from aethmodular_cli.env import load_repo_env

load_repo_env()

ROOT = Path(__file__).resolve().parents[3]
PROJECT = "c8f10563-d27a-4b5a-a8fd-ad79f3d07568"
EXISTING_RELEASE = "398e3311-13fb-42e4-8919-55349e409a56"
BASE = os.environ.get("OPENRESEARCH_URL", "http://127.0.0.1:4791").rstrip("/")
PREFIX = "aeth-history-v1:"
OUT = ROOT / "research/ftir_hips_chem/output/tables/openresearch_families"
ARTIFACTS = Path(os.environ.get("OPENRESEARCH_FILES_DIR", "~/.local/share/openresearch/files")).expanduser() / "aethmodular-research-retrospective"
GUARD = "printf '%s\\n' 'Historical catalog record. See its description to stage a separate reproduction.' >&2; exit 2"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cli(*args, input_text=None):
    return subprocess.check_output(["orx", "--no-telemetry", *args], input=input_text, text=True)


def experiments():
    with urlopen(f"{BASE}/api/projects/{PROJECT}/experiments", timeout=15) as response:
        return json.load(response)["experiments"]


def pane_url(pane):
    return f"{BASE}/projects/{PROJECT}/tasks/new?pane=" + quote(json.dumps(pane, separators=(",", ":")), safe="")


def file_url(path):
    return pane_url(dict(kind="file", path=path, source="artifacts"))


def node_url(node_id):
    return pane_url(dict(kind="experiment", experimentId=node_id, view="overview"))


def section(text, pattern):
    """Extract a complete saved section, keeping its status as historical prose."""
    headings = list(re.finditer(r"(?m)^(#{1,6})\s+(.+)$", text))
    for i, match in enumerate(headings):
        if re.search(pattern, match[2], re.I):
            end = next((h.start() for h in headings[i+1:] if len(h[1]) <= len(match[1])), len(text))
            body = text[match.end():end].strip()
            if body and "filled in by" not in body[:150]:
                return body[:3000] + ("\n\n[Excerpt continues in the source.]" if len(body) > 3000 else "")
    return None


def recipe(record):
    rid = record["id"]
    workflows = {
        "filter_only_diagnostics": "analyze_filter_diagnostics.py",
        "filter_relationship_stability": "analyze_filter_relationship_stability.py",
        "filter_proportionality": "analyze_filter_proportionality.py",
        "audit_catalog": "audit_matched_samples.py",
        "active_interval": "build_active_interval_matches.py",
        "vibes_subgroup_audit": "audit_vibes_saved_predictions.py",
    }
    if rid == "september_release":
        return {"status": "verified_new_openresearch_reproduction", "command":
                "cd deliverables/filter_only_scientific_release_2026-09-11 && uv venv --python 3.13 .venv && uv pip sync --python .venv/bin/python requirements.lock && .venv/bin/python reproduce.py --no-figures",
                "prerequisites": "Run from an isolated committed checkout. Release-pinned Python 3.13 environment; see the existing verified run."}
    if rid in workflows:
        entry = "research/ftir_hips_chem/workflows/" + workflows[rid]
        return {"status": "verified_local_audit" if rid == "vibes_subgroup_audit" else "located_runner_not_rerun_in_migration",
                "command": f"uv run --locked --no-sync python {entry}", "entrypoint": entry,
                "prerequisites": "From repository root, run uv run aeth doctor first. Stage the original frozen inputs and hashes. Work in a separate checkout/output area: runners can regenerate reports and notebooks."}
    entries = record.get("entrypoints", [])
    if rid == "ftir_25":
        entries = ["research/ftir_ec_phase3/scripts/run_ftir_25_intercept_invariant.py"]
    if entries:
        entry = entries[0]
        area = Path(entry).parent.parent.as_posix()
        # These percent-cell runners explicitly resolve scripts/ and output/ from the area cwd.
        return {"status": "located_runner_not_rerun_in_migration", "entrypoint": entry,
                "command": f"cd {shlex.quote(area)} && uv run --locked --no-sync python scripts/{shlex.quote(Path(entry).name)}",
                "prerequisites": "Stage original raw/local/Drive data, phase-2 exports and preceding output/corrected caches as named by the source. Use Python 3.13 and aeth doctor. Run in a separate checkout because this runner writes historical output paths. Original run environment is not established by source presence."}
    if rid == "vibes_full_colab":
        return {"status": "completed_external_notebook_protocol", "command": None,
                "prerequisites": "The archived Colab notebook, matching bundle content hash and RUN_MANIFEST.json specify the completed run. The multi-hour notebook is the reproduction entry point; this import does not launch cloud compute."}
    if rid == "airspec_port":
        return {"status": "reference_validation_source_located", "command": None,
                "prerequisites": "Stage the original R reference exports and inspect validate_airspec_port.py arguments before reconstructing an execution command."}
    return {"status": "notebook_or_document_only", "command": None,
            "prerequisites": "No standalone reproduction command verified. Read the source setup and required data before staging a fresh execution. Saved notebook counts are not proof that today's environment can reproduce it."}


def build_plan():
    inventory_path = ROOT / "research/ftir_hips_chem/output/tables/openresearch_retrospective/inventory.json"
    inventory = json.loads(inventory_path.read_text())
    audit_path = ROOT / 'research/ftir_hips_chem/output/tables/openresearch_evidence_audit/audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if audit and audit['inventory_sha256'] != sha(inventory_path):
        raise ValueError('Evidence audit belongs to a different inventory; regenerate it first')
    audit_records = {r['id']:r for r in audit['records']} if audit else {}
    claims_path = ROOT / 'docs/openresearch-retrospective/claim_updates.json'
    claim_updates = json.loads(claims_path.read_text())['updates'] if claims_path.exists() else []
    spec = json.loads((ROOT / "docs/openresearch-retrospective/families.json").read_text())
    by_id = {r["id"]: r for r in inventory["records"]}
    assigned = [rid for family in spec["families"] for rid in family["records"]]
    if set(assigned) != set(by_id) or len(assigned) != len(set(assigned)):
        raise ValueError(f"Missing/duplicate family membership: {set(by_id)-set(assigned)} / {set(assigned)-set(by_id)}")
    membership = {rid: family["id"] for family in spec["families"] for rid in family["records"]}
    typed = {}
    for edge in spec['dependency_evidence'] + spec['revisions']:
        typed.setdefault((edge['from'],edge['to']),[]).append(edge)
    edges = []
    for edge in inventory["research_links"]:
        pair = (edge["from"], edge["to"])
        edges.extend(typed.pop(pair, [dict(edge, type="related_research", evidence=None)]))
    edges.extend(e for group in typed.values() for e in group)
    edges=list({(e['from'],e['to'],e['type']):e for e in edges}.values())
    for edge in edges:
        if edge["from"] not in by_id or edge["to"] not in by_id:
            raise ValueError("Unresolved relation")
        if edge.get("evidence"):
            evidence = ROOT / edge["evidence"]
            if not evidence.is_file() or (edge.get("contains") and edge["contains"] not in evidence.read_text()):
                raise ValueError(f"Dependency evidence missing: {edge}")
            edge["evidence_sha256"] = sha(evidence)
    records = []
    for rid, record in by_id.items():
        saved = "\n\n".join(x["text"] for x in record.get("historical_notes", []))
        if not saved and Path(record["source"]).suffix == ".md":
            snapshot = ARTIFACTS / "research-retrospective/sources" / record["source"]
            saved = snapshot.read_text()
        purpose = section(saved, r"^research question|^purpose|^what this notebook is for") or record["title"]
        finding = section(saved, r"tl;dr|takeaway|conclusion|working answer|key findings|main findings|what the audit adds")
        if rid in {"vibes_full_colab", "vibes_subgroup_audit", "september_release"}:
            finding = spec["record_notes"][rid]
        if not finding:
            finding = "No explicit scientific conclusion was extracted from the saved source. Review its outputs before claiming a result; the imported record preserves this evidence gap."
        evidence_audit = audit_records.get(rid)
        if evidence_audit and finding.startswith('No explicit scientific conclusion'):
            candidates = [c for c in evidence_audit['saved_evidence'].get('conclusion_candidates', [])
                          if c['kind']=='historical_narrative_not_newly_verified']
            if candidates:
                c=candidates[0]
                finding=f"Recovered from saved source cell {c['cell_index']} ({c['heading']}):\n\n{c['excerpt']}"
        status=record['evidence_status']
        if evidence_audit:
            states=[evidence_audit['saved_evidence'],*evidence_audit['archived_companion_audits']]
            if any(s.get('saved_errors') for s in states): status='historical_saved_error'
            elif status=='source_inventoried' and any(s['execution_state']=='saved_execution_no_errors_recorded' for s in states):
                status='historical_execution_recorded'
        caveats = [record.get("note", ""), spec["record_notes"].get(rid, "")]
        source_set = set([record["source"], *record.get("artifact_files", []), *record.get("entrypoints", [])])
        reviewed = [e for e in inventory["reviewed_evidence"] if source_set.intersection(e["sources"])]
        if membership[rid] == "relationships" and rid not in {"filter_only_diagnostics", "filter_relationship_stability", "filter_proportionality", "september_release"}:
            caveats.append("Where this record compares optical/filter dates, later sampling and provenance findings govern interpretation. FTIR EC is a prediction, not a thermal reference. Preserve canonical units, flags, wavelengths and season definitions when reproducing.")
        records.append(dict(
            id=rid, title=record["title"], family=membership[rid], kind=record["kind"],
            source=record["source"], source_sha256=inventory["file_index"][record["source"]]["sha256"],
            evidence_status=status, question=purpose, historical_finding=finding,
            reviewed_findings=reviewed, caveats=[c for c in caveats if c],
            reproduction=evidence_audit['reproduction'] if evidence_audit else recipe(record), saved_notebook_state=record.get("saved_notebook_state"),
            evidence_audit=evidence_audit, claim_updates=[c for c in claim_updates if c['from']==rid],
            artifact_files=record.get("artifact_files", []),
        ))
    for record in records:
        entry = record["reproduction"].get("entrypoint")
        if entry and not (ROOT / entry).is_file():
            raise ValueError(f"Missing runner {entry}")
    return dict(schema=1, project_id=PROJECT, inventory_sha256=sha(inventory_path),
                source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                families=spec["families"], records=records, relations=edges,
                semantics="Family edges organize the catalog. Only labelled data_dependency edges assert source consumption. Registration Git branches are current snapshots, not reconstructed historical execution commits.")


def description(record, plan, ids):
    rid = record["id"]
    family = next(f for f in plan["families"] if f["id"] == record["family"])
    card = file_url(f"research-retrospective/records/{rid}.md")
    family_url = file_url(f"research-families/families/{family['id']}.md")
    lines = [f"# {record['title']}", "", f"Import key: `{PREFIX}{rid}`", "",
             f"**Historical evidence status:** `{record['evidence_status']}`. Imported as a catalog record; no new scientific run was launched.", "",
             f"Family: [{family['title']}]({family_url}). [Full evidence card]({card}).", "",
             "## Question or purpose", "", record["question"], "", "## Findings as recorded", "",
             "The following excerpt is historical source prose, not a newly validated conclusion:", "", record["historical_finding"], ""]
    if record["reviewed_findings"]:
        lines += ["## Reviewed findings and corrections", ""]
        for item in record["reviewed_findings"]:
            lines += [f"**{item['title']}** — {item['finding']}", ""]
    if record.get('evidence_audit'):
        a=record['evidence_audit']; s=a['saved_evidence']
        lines += ['## Historical evidence audit','',
                  f"Active source saved state: **{s['execution_state']}**. {len(a['archived_companion_audits'])} archived companions inspected. {len(a['additional_artifacts'])} additional current artifacts located; their historical generation is unestablished.",'',
                  f"[Full audit: conclusions, saved errors, input calls, output excerpts and hashes]({file_url('research-evidence-audit/records/'+rid+'.md')}).",'',
                  *[f"- {b}" for b in a['blockers']], '']
        for c in a['archived_companion_audits']:
            lines += [f"- Archived `{Path(c['path']).name}`: `{c['execution_state']}` ({c['execution_count_cells']}/{c['code_cells']} counted cells)."]
    if record.get('claim_updates'):
        lines += ['', '## Claims updated by later work','']
        for c in record['claim_updates']:
            later=node_url(ids[c['to']]) if c['to'] in ids else file_url('research-retrospective/records/'+c['to']+'.md')
            lines += [f"- **{c['decision']}** — Scope: {c['claim_scope']} [Later evidence: {c['to']}]({later}). {c['current_reading']}"]
    lines += ["## Limits and status", "", *[f"- {c}" for c in record["caveats"]], "",
              "## Relationships", "", "Family membership is organizational. Research relationships below are separately typed; they do not reconstruct historical Git ancestry.", ""]
    for edge in plan["relations"]:
        if rid not in (edge["from"], edge["to"]):
            continue
        other = edge["to"] if edge["from"] == rid else edge["from"]
        label = f"{edge['from']} → {edge['to']}"
        link = node_url(ids[other]) if other in ids else file_url(f"research-retrospective/records/{other}.md")
        lines.append(f"- **{edge['type']}**: [{label}]({link}). {edge['reason']}")
        if edge.get("evidence"):
            lines.append(f"  Evidence: [{Path(edge['evidence']).name}]({file_url('research-families/evidence/'+edge['evidence'])}).")
    recipe = record["reproduction"]
    lines += ["", "## Reproduction", "", f"Recipe status: `{recipe['status']}`.", "", recipe["prerequisites"], ""]
    if recipe["command"]:
        lines += ["```sh", recipe["command"], "```", ""]
    lines += ["For a new scientific run, create a separate executable baseline with the stated inputs and environment staged. Imported catalog nodes have a non-executing guard command; their registration branches do not certify historical code provenance.", "",
              "## Supporting files", "",
              f"[Saved source]({file_url('research-retrospective/sources/'+record['source'])}); current snapshot SHA-256 `{record['source_sha256']}`.", ""]
    for path in record["artifact_files"][:8]:
        lines.append(f"- [{Path(path).name}]({file_url('research-retrospective/sources/'+path)})")
    if len(record["artifact_files"]) > 8:
        lines.append(f"- The full evidence card lists all {len(record['artifact_files'])} supporting artifacts.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create/update catalog experiment nodes and publish family navigation")
    args = parser.parse_args()
    plan = build_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "migration_plan.json", plan)
    print(json.dumps({"records": len(plan["records"]), "families": {f["title"]: len(f["records"]) for f in plan["families"]},
                      "relations": dict(Counter(e["type"] for e in plan["relations"])),
                      "recipes": dict(Counter(r["reproduction"]["status"] for r in plan["records"]))}, indent=2), flush=True)
    if not args.apply:
        return
    before = experiments()
    if not (OUT / "pre_migration_experiments.json").exists():
        dump(OUT / "pre_migration_experiments.json", before)
    ids = {"september_release": EXISTING_RELEASE}
    branch_before = subprocess.check_output(["git", "symbolic-ref", "HEAD"], text=True).strip()
    original_release = next(x for x in before if x["id"] == EXISTING_RELEASE)
    previous_path = OUT / "original_release_description.md"
    if not previous_path.exists():
        previous_path.write_text(original_release.get("description") or "")
    # Artifact files are local. No cloud service or agent is started.
    target = ARTIFACTS / "research-families"
    target.mkdir(parents=True, exist_ok=True)
    for edge in plan["relations"]:
        if edge.get("evidence"):
            dest = target / "evidence" / edge["evidence"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / edge["evidence"], dest)
            if sha(dest) != edge["evidence_sha256"]:
                raise ValueError("Evidence changed during import")

    def ensure(key, title, desc, parent=None):
        marker = f"Import key: `{PREFIX}{key}`"
        matches = [n for n in experiments() if marker in (n.get("description") or "")]
        if len(matches) > 1:
            raise ValueError(f"Duplicate import key {key}")
        if not matches:
            command = ["create-experiment", PROJECT, "--title", title, "--description", desc]
            command += ["--parent", parent] if parent else ["--baseline", "--run-command", GUARD]
            cli(*command)
            matches = [n for n in experiments() if marker in (n.get("description") or "")]
        if len(matches) != 1 or matches[0]["parentExperimentId"] != parent or matches[0]["runCommand"] != GUARD:
            raise ValueError(f"Unexpected imported node state: {key}")
        ids[key] = matches[0]["id"]
        dump(OUT / "node_ids.json", ids)
        return matches[0]["id"]

    for i, family in enumerate(plan["families"], 1):
        key = "family_" + family["id"]
        desc = f"Import key: `{PREFIX}{key}`\n\nHistorical family index: {family['purpose']}\n\nContains {len(family['records'])} analysis-level records. Organizational heading, not a measured scientific baseline.\n\n[Family guide]({file_url('research-families/families/'+family['id']+'.md')}). No historical runs or commits are fabricated."
        ensure(key, f"{i:02d} {family['title']} [historical family]", desc)
    records = {r["id"]: r for r in plan["records"]}
    # Only source-verified, same-family data dependencies become deeper branches.
    receipt_path=OUT/'migration_receipt.json'
    # Research edges can have multiple parents. Preserve registration ancestry
    # when enriching the graph instead of silently selecting a different parent.
    parents = json.loads(receipt_path.read_text())['parent_relations'] if receipt_path.exists() else {
        e['to']:e['from'] for e in plan['relations'] if e['type']=='data_dependency'
        and records[e['from']]['family']==records[e['to']]['family']}
    pending = [r for r in plan["records"] if r["id"] != "september_release"]
    while pending:
        ready = [r for r in pending if parents.get(r["id"]) is None or parents[r["id"]] in ids]
        if not ready:
            raise ValueError("Dependency cycle")
        for record in ready:
            rid = record["id"]
            parent = ids[parents[rid]] if rid in parents else ids["family_" + record["family"]]
            title = re.sub(r"^ftir_\d+\s*[—–-]\s*", "", record["title"])
            title = f"[Historical] {rid} — {title}"
            ensure(rid, title, description(record, plan, ids), parent)
            pending.remove(record)
            if (len(ids) - 7) % 20 == 0:
                print(f"Imported {len(ids)-7} historical records", flush=True)
    # Resolve native links after every id exists; preserve the release's original note.
    current_nodes = {node["id"]: node for node in experiments()}
    for record in plan["records"]:
        desc = description(record, plan, ids)
        if record["id"] == "september_release":
            desc = desc.replace("Imported as a catalog record; no new scientific run was launched.",
                                "This existing node retains its verified OpenResearch run. The migration adds navigation notes only.")
            desc = desc.replace("Imported catalog nodes have a non-executing guard command; their registration branches do not certify historical code provenance.",
                                "This existing release node retains its original command, branch and verified run; only newly imported historical nodes have guard commands.")
            desc += "\n## Original creation note\n\nThe note below predates Colab completion; current VIBES status is in its linked record.\n\n" + previous_path.read_text()
        if current_nodes[ids[record["id"]]].get("description") != desc:
            cli("exp", "desc", ids[record["id"]], "--stdin", input_text=desc)
        dest = target / "experiments" / (record["id"] + ".md")
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(desc)
    for family in plan["families"]:
        lines = [f"# {family['title']}", "", family["purpose"], "",
                 "Historical research catalog. Titles and saved findings retain their evidence status. This family includes analyses and supporting audit/report/template records where relevant.", "",
                 "| Record | Evidence status | Reproduction recipe |", "|---|---|---|"]
        for rid in family["records"]:
            r = records[rid]
            lines.append(f"| [{rid}: {r['title']}]({node_url(ids[rid])}) | {r['evidence_status']} | {r['reproduction']['status']} |")
        lines += ["", "## Relationships", "", "Only data_dependency means verified source consumption. Other links record questions or revisions. Family membership is organizational, and registration branches are not historical run commits.", ""]
        for edge in plan["relations"]:
            if edge["from"] in family["records"] or edge["to"] in family["records"]:
                lines.append(f"- **{edge['type']}**: [{edge['from']}]({node_url(ids[edge['from']])}) → [{edge['to']}]({node_url(ids[edge['to']])}): {edge['reason']}")
        dest = target / "families" / (family["id"] + ".md")
        dest.parent.mkdir(exist_ok=True)
        dest.write_text("\n".join(lines) + "\n")
    after = experiments()
    by_id = {n["id"]: n for n in after}
    for record in plan["records"]:
        node = by_id[ids[record["id"]]]
        if f"Import key: `{PREFIX}{record['id']}`" not in node["description"]:
            raise ValueError("Description persistence failure")
    for field in ("branchName", "runCommand", "parentExperimentId"):
        if by_id[EXISTING_RELEASE][field] != original_release[field]:
            raise ValueError("Existing release contract changed")
    if subprocess.check_output(["git", "symbolic-ref", "HEAD"], text=True).strip() != branch_before:
        raise ValueError("Source checkout branch changed")
    receipt = dict(status="passed", imported_at_utc=datetime.now(timezone.utc).isoformat(),
                   records=len(records), family_headings=len(plan["families"]),
                   project_experiment_nodes=len(after), ids=ids, parent_relations=parents,
                   source_head=plan["source_head"], original_release_contract_preserved=True,
                   source_checkout_branch_preserved=True, scientific_runs_launched=0,
                   plan_sha256=sha(OUT / "migration_plan.json"))
    dump(OUT / "migration_receipt.json", receipt)
    dump(target / "migration_receipt.json", receipt)
    dump(target / "migration_plan.json", plan)
    lines = ["# Research families in OpenResearch", "",
             f"{len(records)} analysis-level records are represented by native experiment entries, organized using six historical family headings. The existing verified release node is linked into the filter family without changing its branch or run contract.", "",
             "## Browse the families", ""]
    lines += [f"- [{f['title']}]({file_url('research-families/families/'+f['id']+'.md')}) — {len(f['records'])} records. [Tree heading]({node_url(ids['family_'+f['id']])})." for f in plan["families"]]
    lines += ["", "## How to use this catalog", "",
              "Open a record to read its question, saved findings, reviewed corrections, supporting files and reproduction recipe. Search the experiment table by the original notebook id, such as ftir_13 or vibes_full_colab.", "",
              "**Status:** The local interface distinguishes external completion, saved historical execution, artifacts, source-only records and saved errors. Historical evidence is not a new OpenResearch run. The September release retains its genuine Done run. Family headings are organizational containers.", "",
              "**Dependencies:** The tree uses same-family source-verified data dependencies for deeper nesting. Cross-family dependencies, corrections and merely related research are typed and linked in descriptions and family guides. Registration branches are today's catalog snapshots, not historical execution provenance.", "",
              "**Reproduction:** Located commands describe an entry point and prerequisites; they are not all verified runnable environments. Historical nodes use a guard command so the release command is not accidentally inherited. Stage inputs and create a separate executable baseline to pursue a new scientific run.", "",
              "**Coverage:** This imports the existing 101-record inventory. Archive-only notebooks, other repository subprojects and individual sweep configurations are not expanded into separate experiment nodes. The ftir_55 record links the sweep analysis rather than inventing thousands of completed runs.", "",
              "[Migration receipt](migration_receipt.json) · [Full structured plan](migration_plan.json) · [Historical evidence audit](../research-evidence-audit/report.md) · [Priority experiment contracts](../research-next-experiments/report.md)", ""]
    (target / "research-families.md").write_text("\n".join(lines))
    shutil.copy2(target / "research-families.md", OUT / "research-families.md")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"ids", "parent_relations"}}, indent=2))


if __name__ == "__main__":
    main()
