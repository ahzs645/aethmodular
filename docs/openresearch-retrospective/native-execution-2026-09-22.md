# AIRSpec now runs through OpenResearch

The first native AIRSpec baseline passed on 2026-09-22. This closes the execution
integration gap for AIRSpec: the experiment has a real fixed command, a committed
source snapshot, a fresh environment, pinned inputs, persisted logs, and verified results.

[Open the executable experiment](http://127.0.0.1:4791/projects/203b7e92-c6ad-4d8a-9936-411a08bce8c6/tasks/new?pane=%7B%22kind%22%3A%22experiment%22%2C%22experimentId%22%3A%22d2b46613-884e-4eef-b9d8-d06d62459b63%22%2C%22view%22%3A%22overview%22%7D).

| Evidence | Result |
|---|---|
| Native run | `5e953103-e1b4-4218-8ddf-f9c67188bfd4` — done; final evidence summary passed |
| Recorded commit | `fb0abeaf5d272b76c64bf6616743e56f00800362` |
| Source archive | SHA-256 `961d30056e40a3f949f3d8048cdeb9ccfe5c5ce6c503a4d6b7cf8916e9cf479c` |
| Frozen data | 17 files, 291,137,277 bytes, verified before and after execution |
| Scientific code | Six original source files hash-verified; ftir_13 unchanged |
| Historical comparison | Five tables, all 43 column checks passed |
| Previous local comparison | Six additional tables, all 41 column checks passed |
| Maximum absolute numeric difference | 2.47e-13 (tolerance 1e-6) |
| OCEC split | 606 training / 194 test rows; sites disjoint |
| Components | DF1=6: k=5; DF1=8: k=5 |
| Held-out TOR RMSE | DF1=6: 3.858654; DF1=8: 3.971105 µg/filter |

The six input/result-gate unit tests also passed, covering input tampering,
changed identities, missing columns, finite-to-NaN changes, absolute tolerance,
and manifest path traversal. The source archive and copied result hashes were verified.

## What changed

Created **Aethmodular — AIRSpec execution**, backed by an isolated worktree of
this repository. Its baseline branch is now frozen at the recorded commit.
The existing retrospective project retains its historical records and original
September release run. The historical AIRSpec contract links to this native experiment.

The fixed run command is:

```sh
bash research/ftir_hips_chem/workflows/openresearch_airspec/run.sh
```

Launch through OpenResearch:

```sh
orx exp run d2b46613-884e-4eef-b9d8-d06d62459b63 --backend local
orx logs <new-run-id>
```

The run creates its own Python 3.13.9 environment from the committed `uv.lock`.
Large inputs are held in a content-addressed local bundle and copied into the run
snapshot after verification. Scientific data reads outside the declared inputs
and run outputs are rejected. A cloud-placeholder stall encountered during
packaging was resolved by directly downloading the file and checking its frozen
SHA-256; native execution does not depend on live Google Drive files.

## Evidence files

- [Native run log](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/runs/5e953103-e1b4-4218-8ddf-f9c67188bfd4/run.log)
- [Run receipt and source archive digest](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/runs/5e953103-e1b4-4218-8ddf-f9c67188bfd4/openresearch-receipt.json)
- [Scientific execution manifest](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/runs/5e953103-e1b4-4218-8ddf-f9c67188bfd4/manifest.json)
- [All 84 table checks](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/runs/5e953103-e1b4-4218-8ddf-f9c67188bfd4/table_comparison.csv)
- [Held-out metrics](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/runs/5e953103-e1b4-4218-8ddf-f9c67188bfd4/heldout_metrics.csv)
- [Split membership](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/runs/5e953103-e1b4-4218-8ddf-f9c67188bfd4/split_membership.csv)
- [Recipe, input staging, and continuation instructions](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/contract/README.md)
- [Execution wrapper](~/.local/share/openresearch/files/aethmodular-airspec-execution/airspec-baseline/recipe/run.py)

## Scope and next decisions

This establishes numerical reproduction of ftir_13 from previously corrected
caches. It does not rerun full-pool AIRSpec correction or establish independent
Addis accuracy. The earlier local reproduction remains separately labelled;
it was not converted into a fabricated native run.

Use this baseline as a frozen parent only when a new experiment actually builds
on this calibration. Keep one fixed command and put changes and decision rules
in committed child code/config. There is no reason to create a child just to
make the tree deeper.

The VIBES full-pool comparison uses a different cohort and split from this
800-filter calibration. Its next applicability investigation needs its own
matching input/evaluation contract; it must not silently compare against this
baseline's different population. Rules designed after inspecting its known
held-out failures remain exploratory, even if thresholds are fitted on training
data. Keep all original errors and require fresh evaluation for improvement claims.

VIBES local diagnostics are still externally executed evidence, and Addis
independent validation remains blocked on reference measurements and authoritative
identities. Historical entries retain guards intentionally. This migration makes
one active baseline executable; it does not imply all 109 catalog guards are gone.

The input bundle is local, outside Git. Transfer and verify it before running on
another host. No data or code were pushed to GitHub and no paid compute was used.
