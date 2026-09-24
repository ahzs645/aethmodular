"""Inventory historical evidence without executing or modifying old analyses.

Run from the repository with uv. Writes JSON and Markdown under output/tables.
Saved notebook execution counts and existing output files are evidence locators,
not proof that the current source generated those outputs.
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
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "research/ftir_hips_chem"
P3 = ROOT / "research/ftir_ec_phase3"
CURATION = ROOT / "docs/openresearch-retrospective/curation.json"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def notebook_state(path):
    nb = json.loads(path.read_text())
    codes = [c for c in nb.get("cells", []) if c["cell_type"] == "code"]
    errors = [
        {"cell": i, "name": o.get("ename"), "message": o.get("evalue")}
        for i, c in enumerate(nb.get("cells", []))
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    title = next(
        (line.lstrip("# ") for c in nb.get("cells", []) if c["cell_type"] == "markdown"
         for line in "".join(c.get("source", [])).splitlines() if line.startswith("# ")),
        path.stem,
    )
    markdown = [(i, "".join(c.get("source", []))) for i, c in enumerate(nb.get("cells", []))
                if c["cell_type"] == "markdown"]
    selected = dict(markdown[:2])
    for i, source in markdown:
        if re.search(r"^#{1,4}\s+.*(?:takeaway|conclusion|summary|tl;dr|interpretation|verdict)", source, re.I | re.M):
            selected[i] = source
    return {
        "title": title,
        "reported_notes": [{"cell_index": i, "text": text[:6000], "truncated": len(text) > 6000}
                           for i, text in selected.items()],
        "code_cells": len(codes),
        "cells_with_execution_count": sum(c.get("execution_count") is not None for c in codes),
        "cells_with_outputs": sum(bool(c.get("outputs")) for c in codes),
        "saved_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=P2 / "output/tables/openresearch_retrospective")
    parser.add_argument("--publish-artifacts", type=Path, help="Copy a hash-checked evidence snapshot into this local artifact folder")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    curation = json.loads(CURATION.read_text())
    head = git("rev-parse", "HEAD")
    tracked = set(git("ls-files", "-z").split("\0"))
    file_cache = {}

    def file_record(path):
        path = Path(path)
        rel = path.relative_to(ROOT).as_posix()
        if rel in file_cache:
            return rel
        info = {"path": rel, "exists": path.is_file(), "tracked": rel in tracked}
        if path.is_file():
            info.update(bytes=path.stat().st_size, sha256=sha(path))
            if rel in tracked:
                history = git("log", "-1", "--format=%H%x09%cI", "--", rel).split("\t")
                info["last_source_change"] = dict(zip(["commit", "committed_at"], history))
                # This checks the current file against HEAD, not provenance of a run.
                info["matches_head"] = subprocess.run(
                    ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]
                ).returncode == 0
        file_cache[rel] = info
        return rel

    records = []
    for area in (P2, P3):
        for path in sorted(area.glob("*.ipynb")):
            match = re.match(r"ftir_(\d+)_", path.name)
            key = f"ftir_{int(match[1]):02d}" if match else path.stem
            if key == "ftir_vibes_vs_airspec":
                key = "vibes_comparison"
            state = notebook_state(path)
            record = {
                "id": key, "kind": "notebook", "area": area.name,
                "title": state.pop("title"), "source": file_record(path),
                "historical_notes": state.pop("reported_notes"),
                "saved_notebook_state": state, "executed_copies": [],
                "entrypoints": [], "artifact_files": [],
                "evidence_status": "source_inventoried", "historical_run_commit": None,
                "note": "Current file hashes and Git history do not establish the historical run environment.",
            }
            if match:
                n = int(match[1])
                runner = area / "scripts" / f"run_ftir_{n}.py"
                if runner.exists():
                    record["entrypoints"].append(file_record(runner))
                dirs = [area / "output/tables" / f"ftir{n}", area / "output/tables" / f"ftir_{n:02d}"]
            else:
                dirs = [area / "output/tables" / path.stem]
            for directory in dirs:
                if directory.is_dir():
                    record["artifact_files"].extend(
                        file_record(f) for f in sorted(directory.rglob("*"))
                        if f.is_file() and f.suffix in {".csv", ".parquet", ".json", ".md"}
                    )
            archive = area / "notebooks/archive/executed"
            for name in (path.name, path.stem + "_executed.ipynb"):
                archived = archive / name
                if archived.exists():
                    record["executed_copies"].append({"path": file_record(archived), **notebook_state(archived)})
            if record["artifact_files"]:
                record["evidence_status"] = "artifacts_located_and_hashed"
            record.update(curation["overrides"].get(key, {}))
            extra_dir = record.get("extra_artifact_directory")
            if extra_dir:
                record["artifact_files"].extend(
                    file_record(f) for f in sorted((ROOT / extra_dir).rglob("*"))
                    if f.is_file() and f.suffix in {".csv", ".parquet", ".json", ".md"}
                )
                if record["artifact_files"]:
                    record["evidence_status"] = "artifacts_located_and_hashed"
            records.append(record)

    for extra in curation["additional_records"]:
        record = dict(extra)
        record["source"] = file_record(ROOT / record["source"])
        record["artifact_files"] = [file_record(ROOT / p) for p in record.get("artifact_files", [])]
        record["historical_run_commit"] = None
        records.append(record)
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate research record ids")
    for edge in curation["links"]:
        if edge["from"] not in ids or edge["to"] not in ids:
            raise ValueError(f"Unresolved research link: {edge}")
    for item in curation["reviewed_evidence"]:
        for p in item["sources"]:
            file_record(ROOT / p)

    release = ROOT / "deliverables/filter_only_scientific_release_2026-09-11"
    manifests = {}
    for name, field in [("release_manifest.json", "immutable_files"),
                        ("release_checks/tested_numerical_files.json", "files")]:
        manifest = json.loads((release / name).read_text())
        checks = []
        for item in manifest[field]:
            path = release / item["path"]
            file_record(path)
            checks.append({"path": item["path"], "match": path.is_file() and sha(path) == item["sha256"]})
        manifests[name] = {"files": len(checks), "passed": all(c["match"] for c in checks),
                           "mismatches": [c["path"] for c in checks if not c["match"]]}

    verification = out / "reproduction/verification.json"
    reproduced = json.loads(verification.read_text()) if verification.exists() else None
    receipt_path = out / "reproduction/run_record.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if reproduced and reproduced.get("status") == "passed" and receipt and receipt.get("status") == "done":
        next(r for r in records if r["id"] == "september_release")["evidence_status"] = "reproduced_in_openresearch"
    result = {
        "schema": 1, "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT), "head": head,
        "scope": "Top-level active notebooks in ftir_hips_chem and ftir_ec_phase3 plus selected audit/release records; excludes the wider repository and archive-only analyses.",
        "limitations": ["Source revision is not historical run provenance.",
                         "Saved execution counts are not proof of scientific validity or reproducibility.",
                         "Artifact presence/hash verification is not a rerun.",
                         "Research links are curated conceptual relationships, not Git ancestry.",
                         "Only the September release has a new reproduction in this inventory."],
        "records": records, "research_links": curation["links"],
        "reviewed_evidence": curation["reviewed_evidence"],
        "file_index": file_cache, "release_integrity": manifests,
        "new_reproduction": reproduced, "openresearch_run_record": receipt,
    }
    (out / "inventory.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    def link(path, label=None):
        target = os.path.relpath(ROOT / path, out).replace(" ", "%20")
        return f"[{label or Path(path).name}]({target})"

    counts = Counter(r["evidence_status"] for r in records)
    lines = ["# Research retrospective — evidence inventory", "",
             f"Inspected {result['inspected_at_utc']}; source HEAD `{head}`.", "",
             result["scope"], "", f"**{len(records)} research records; {len(file_cache)} indexed files; {len(curation['links'])} curated links.**", "",
             "## Evidence status", "", *[f"- {k}: {v}" for k, v in sorted(counts.items())], "",
             "Only the September release was rerun. Other saved results retain their historical status.", "",
             "## New reproduction", "", "```json", json.dumps(reproduced, indent=2), "```", "",
             "## Reviewed findings and revisions", ""]
    for item in curation["reviewed_evidence"]:
        lines.extend([f"### {item['title']}", "", item["finding"], "",
                      "Evidence: " + ", ".join(link(p) for p in item["sources"]) + ".", ""])
    lines.extend(["## Research connections", "", "These links describe research reasoning, not interchangeable experiment scores or Git ancestry.", "",
                  "| From | To | Relationship |", "|---|---|---|"])
    lines.extend(f"| {e['from']} | {e['to']} | {e['reason']} |" for e in curation["links"])
    lines.extend(["", "## Catalog", "", "Notebook counts below describe saved files, not newly executed cells. See inventory.json for hashes, source revisions, saved errors, and archived copies.", "",
                  "| Record | Source | Counted code cells | Table/artifact files | Status |", "|---|---|---:|---:|---|"])
    for r in records:
        nb = r.get("saved_notebook_state", {})
        execution = f"{nb.get('cells_with_execution_count', 0)}/{nb.get('code_cells', 0)}" if nb else "—"
        lines.append(f"| [{r['id']}](records/{r['id']}.md) | {link(r['source'])} | {execution} | {len(r.get('artifact_files', []))} | {r['evidence_status']} |")
    lines.extend(["", "## Release integrity", "", "```json", json.dumps(manifests, indent=2), "```", "",
                  "## Limits", "", *[f"- {x}" for x in result["limitations"]], ""])
    (out / "report.md").write_text("\n".join(lines))
    cards = out / "records"
    cards.mkdir(exist_ok=True)
    for r in records:
        def card_link(path):
            return f"[{Path(path).name}]({os.path.relpath(ROOT / path, cards).replace(' ', '%20')})"
        info = file_cache[r["source"]]
        content = [f"# {r['title']}", "", f"Record: `{r['id']}`. Evidence status: `{r['evidence_status']}`.", "",
                   "Source: " + card_link(r["source"]), "",
                   f"Current source SHA-256: `{info.get('sha256')}`.", "",
                   "Latest source change (not historical run provenance): `" + json.dumps(info.get("last_source_change")) + "`.", "",
                   r.get("note", "Historical evidence has not been rerun in this inventory."), "",
                   "## Research connections", ""]
        for edge in curation["links"]:
            if r["id"] in (edge["from"], edge["to"]):
                content.append(f"- [{edge['from']}]({edge['from']}.md) → [{edge['to']}]({edge['to']}.md): {edge['reason']}")
        content.extend(["", "## Historical question and interpretation", "",
                        "The excerpts below are statements in the saved source, not newly validated conclusions. Consult the reviewed revisions in the inventory report.", ""])
        for excerpt in r.get("historical_notes", []):
            content.extend([f"Saved markdown cell {excerpt['cell_index']} (zero-based):", "",
                            "``````text", excerpt["text"], "``````", ""])
        if not r.get("historical_notes"):
            content.extend(["Read the linked source record for the historical question and interpretation.", ""])
        content.extend(["## Reproduction entry points", ""])
        content.extend(["- " + card_link(p) for p in r.get("entrypoints", [])] or
                       ["No standalone runner was automatically resolved; use the source and its documented workflow."])
        content.extend(["", "## Existing artifact locators", ""])
        content.extend(["- " + card_link(p) for p in r.get("artifact_files", [])] or
                       ["No artifact directory was automatically matched. This does not establish that no outputs exist."])
        content.extend(["", "## Saved execution state", "", "```json",
                        json.dumps(r.get("saved_notebook_state"), indent=2), "```", ""])
        (cards / f"{r['id']}.md").write_text("\n".join(content))
    summary = {"records": len(records), "indexed_files": len(file_cache), "research_links": len(curation["links"]),
               "statuses": dict(counts), "release_integrity": manifests,
               "missing_indexed_files": [p for p, f in file_cache.items() if not f["exists"]],
               "notebooks_with_saved_errors": [r["id"] for r in records if r.get("saved_notebook_state", {}).get("saved_errors")],
               "report": str(out / "report.md")}
    (out / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["missing_indexed_files"] or not all(m["passed"] for m in manifests.values()):
        raise SystemExit("Inventory has missing evidence or a release integrity mismatch")
    if args.publish_artifacts:
        target = args.publish_artifacts.resolve()
        if target == out or target.is_relative_to(out) or out.is_relative_to(target):
            raise ValueError("Artifact folder must be separate from the inventory output")
        target.mkdir(parents=True, exist_ok=True)
        for rel, info in file_cache.items():
            dest = target / "sources" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
            if sha(dest) != info["sha256"]:
                raise ValueError(f"Source changed during evidence export: {rel}")
        for name in ("inventory.json", "inventory_summary.json", "project.json"):
            if (out / name).exists():
                shutil.copy2(out / name, target / name)
        if (out / "reproduction").exists():
            shutil.copytree(out / "reproduction", target / "reproduction", dirs_exist_ok=True)
        def portable_link(match):
            label, href = match.groups()
            if not href.startswith("../"):
                return match[0]
            original = (out / unquote(href)).resolve()
            rel = original.relative_to(ROOT).as_posix()
            return f"[{label}](sources/{rel.replace(' ', '%20')})"
        portable = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", portable_link, (out / "report.md").read_text())
        (target / "report.md").write_text(portable)
        (target / "records").mkdir(exist_ok=True)
        for card in cards.glob("*.md"):
            def portable_card_link(match):
                label, href = match.groups()
                if not href.startswith("../"):
                    return match[0]
                rel = (cards / unquote(href)).resolve().relative_to(ROOT).as_posix()
                if rel not in file_cache:
                    return match[0]
                return f"[{label}](../sources/{rel.replace(' ', '%20')})"
            text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", portable_card_link, card.read_text())
            (target / "records" / card.name).write_text(text)
        shutil.copy2(CURATION, target / "curation.json")
        print(f"Published local evidence snapshot: {target}")


if __name__ == "__main__":
    main()
