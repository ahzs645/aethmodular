"""Export the blanks analysis for the gallery's "Blanks" tab.

Run with:  uv run --no-sync python gallery/data/export_blanks.py

Everything about the PTFE field blanks that train the VIBES background:
the inventory and which blank bank used each blank, raw blank spectra, where
blanks vary, a shared PCA of every blank, each bank's rank selection and the
per-blank leave-one-out errors behind it, each bank's background components,
held-out blanks after correction, and how much of an ambient spectrum each
background could absorb. Banks:

  VIBES-full-r70   87 blanks (capped: <=50 per lot at training sites + ETAD), rank 70
  VIBES-lot251     every labeled lot-251 IMPROVE blank at training sites + ETAD lot 251
  VIBES-allblanks  every labeled blank in the bundle (lots 248 + 251, all sites) + ETAD

Reads the Colab bundle (raw spectra), the frozen case ledger, the capped run
(repo) and the two blank-bank runs (Drive, data_paths.vibes_colab_results_dir).
Writes gallery/app/public/data/blanks/blanks_20261007.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(AREA / "scripts"))
from data_paths import vibes_colab_results_dir  # noqa: E402
from vibes_baseline import loo_pca, pca  # noqa: E402
from vibes_current import load_completed  # noqa: E402

BUNDLE = AREA / "output/tables/vibes_colab_bundle_full500/stage/data"
CAPPED = AREA / "output/tables/vibes_colab_full500_rank85_resume_20261001/persistent_results/full500-rank85-9345e818c61929fe"
DRIVE = vibes_colab_results_dir() / "blankbank_20261007"
OUT = ROOT / "gallery/app/public/data/blanks"
_RUNS = {"VIBES-lot251": "lot251_all", "VIBES-allblanks": "all_labeled",
         "VIBES-lot251-x1": "lot251_all_minus_LABE1", "VIBES-allblanks-x1": "all_labeled_minus_LABE1"}
BANKS = {"VIBES-full-r70": CAPPED, **{k: hit for k, folder in _RUNS.items()
                                      for hit in sorted((DRIVE / folder / "persistent_results").glob("full500-rank85-*"))[:1]}}
BIN = 8            # channels per displayed point (~10 cm-1), as in the similarity export
MAX_K = 85         # the rank cap every run used
N_PC = 6           # shared-PCA components exported
N_LOAD = 8         # background components shown per bank


def r(a, d=6):
    a = np.asarray(a, float)
    return np.where(np.isfinite(a), np.round(a, d), np.nan).tolist()


def clean(o):
    """NaN -> None recursively, so the JSON is strict."""
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def sig(v, n=4):
    return [None if not np.isfinite(x) else float(f"{x:.{n}g}") for x in np.asarray(v, float)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wn = np.load(BUNDLE / "wn.npy")
    bins = [np.arange(i, min(i + BIN, len(wn))) for i in range(0, len(wn), BIN)]
    bwn = np.array([wn[b].mean() for b in bins])
    binned = lambda X: np.stack([np.asarray(X)[:, b].mean(axis=1) for b in bins], axis=1)  # noqa: E731

    pool = pd.read_csv(BUNDLE / "pool_metadata.csv")
    etad = pd.read_csv(BUNDLE / "etad_metadata.csv")
    px = np.load(BUNDLE / "pool_raw.npy", mmap_mode="r")
    ex = np.load(BUNDLE / "etad_raw.npy", mmap_mode="r")

    # ---------------------------------------------------------------- inventory
    ib = pool[pool.FilterPurposeId.eq(2) & pool.finite_spectrum & pool.all_scans_have_purpose]
    eb = etad[etad.role.isin(["blank_train", "blank_test"])]
    inv = pd.concat([
        pd.DataFrame({"sample_id": [f"improve:{int(v)}" for v in ib.FilterId], "source": "IMPROVE",
                      "site": ib.Site.to_numpy(), "date": ib.date.to_numpy(), "lot": ib.LotNumber.to_numpy(),
                      "split": ib.split.to_numpy(), "role": "labeled blank"}),
        pd.DataFrame({"sample_id": [f"etad:{int(v)}" for v in eb.MediaId], "source": "ETAD",
                      "site": "ETAD", "date": None, "lot": eb.LotId.to_numpy(), "split": "external",
                      "role": eb.role.to_numpy()}),
    ], ignore_index=True)
    raw = np.concatenate([px[ib.index], ex[eb.index]], axis=0).astype(float)
    lot_label = inv.lot.map(lambda v: "lot ?" if pd.isna(v) else f"lot {int(v)}")
    inv["group"] = inv.source + " " + lot_label
    cases = load_completed()["cases"]
    case_blank = set(cases.loc[cases.kind.eq("blank"), "sample_id"])
    bank_ids = {}
    for b, run in BANKS.items():
        bm = pd.read_csv(run / "background_training_blanks.csv")
        bank_ids[b] = set(bm.sample_id)
        inv[f"in_{b}"] = inv.sample_id.isin(bank_ids[b])
    inv["evaluated_blank"] = inv.sample_id.isin(case_blank)
    # held out by every bank: the shared set for "after correction"
    inv["held_out_all"] = inv.evaluated_blank & ~inv[[f"in_{b}" for b in BANKS]].any(axis=1)

    # ---------------------------------------------------------------- raw spectra + variability
    rb = binned(raw)
    groups = sorted(inv.group.unique())
    group_stats = {}
    for g in groups:
        take = inv.group.eq(g).to_numpy()
        q = np.percentile(rb[take], [10, 25, 50, 75, 90], axis=0)
        group_stats[g] = {"n": int(take.sum()), "q": [r(v, 5) for v in q],
                          "sd": r(rb[take].std(axis=0), 6),
                          "sd_detrended": r((rb[take] - rb[take].mean(1, keepdims=True)).std(axis=0), 6)}

    # ---------------------------------------------------------------- shared PCA of every blank
    mu, lam, V, _ = pca(raw)
    scores = (raw - mu) @ V[:, :N_PC]
    var = lam ** 2 / np.sum(lam ** 2)
    shared = {"var": r(var[:N_PC], 5), "scores": [r(scores[:, i], 5) for i in range(N_PC)],
              "loadings": [r(binned(V[:, [i]].T)[0], 5) for i in range(N_PC)], "mean": r(binned(mu[None])[0], 5)}

    # ---------------------------------------------------------------- per bank
    corrected = {"VIBES-full-r70": load_completed()["arrays"]["VIBES-full-r70"]}
    held = inv.loc[inv.held_out_all, "sample_id"].tolist()
    held_rows = pd.Series(np.arange(len(cases)), index=cases.sample_id).loc[held].to_numpy()
    held_src = inv.set_index("sample_id").loc[held, "source"].tolist()
    pop_rows = {"IMPROVE samples": np.flatnonzero(cases.kind.eq("calibration")),
                "Addis samples": np.flatnonzero(cases.kind.eq("target"))}
    case_raw = None
    banks_out = {}
    for b, run in BANKS.items():
        man = json.loads((run / "RUN_MANIFEST.json").read_text())
        bm = pd.read_csv(run / "background_training_blanks.csv")
        take = inv.sample_id.isin(set(bm.sample_id)).to_numpy()
        Z = raw[take]
        ids = inv.sample_id.to_numpy()[take]
        cap = min(MAX_K, len(Z) - 2)
        _, chat, E = loo_pca(Z, max_ncomp=cap)
        mean_e, se_e = E.mean(0), E.std(0) / np.sqrt(len(Z))
        kmin = int(np.argmin(mean_e)) + 1
        rank = int(man["background_rank"])
        assert rank == chat, (b, rank, chat)
        mu_b, lam_b, V_b, _ = pca(Z)
        # per-blank LOO error at a few ranks; how much of the total the worst 5 % carry
        per_blank = {str(k): sig(E[:, k - 1]) for k in sorted({rank, 15, 30, 70, kmin} & set(range(1, cap + 1)))}
        e_r = np.sort(E[:, rank - 1])[::-1]
        # drop-one sensitivity: the one-SE rank after removing the k worst blanks at the chosen rank
        order = np.argsort(-E[:, rank - 1])
        drop = []
        for k in (1, 2, 3):
            keep = np.setdiff1d(np.arange(len(Z)), order[:k])
            _, c_k, E_k = loo_pca(Z[keep], max_ncomp=min(cap, len(keep) - 2))
            drop.append({"removed": ids[order[:k]].tolist(), "rank": int(c_k), "rank_min_loo": int(np.argmin(E_k.mean(0))) + 1})
        worst = {"id": str(ids[order[0]]), "error_x_median": float(E[order[0], rank - 1] / np.median(E[:, rank - 1])),
                 "minus_bank_median": r(binned((Z[order[0]] - np.median(Z, 0))[None])[0], 5)}
        top5 = max(1, int(round(0.05 * len(e_r))))
        # held-out blanks after correction, on the shared held-out set
        arr = corrected.get(b)
        if arr is None:
            arr = np.load(run / "corrected_VIBES_full.npy", mmap_mode="r")
        hc = binned(np.asarray(arr[held_rows], float))
        # how much of each spectrum the background subspace could explain, vs number of components
        if case_raw is None:
            src = cases.source.to_numpy()
            srow = cases.source_row.to_numpy()
            case_raw = {name: np.concatenate([px[srow[rows][src[rows] == "improve"]], ex[srow[rows][src[rows] == "etad"]]]).astype(float)
                        for name, rows in pop_rows.items()}
            case_raw["held-out blanks"] = raw[inv.held_out_all.to_numpy()]
        capture = {}
        Vk = V_b[:, :cap]
        for name, X in case_raw.items():
            Xc = X - mu_b
            tot = np.sum(Xc ** 2, axis=1)
            cum = np.cumsum((Xc @ Vk) ** 2, axis=1) / tot[:, None]
            capture[name] = {"median": r(np.median(cum, 0), 4), "p10": r(np.percentile(cum, 10, 0), 4),
                             "p90": r(np.percentile(cum, 90, 0), 4), "n": int(len(X))}
        banks_out[b] = {
            "n_blanks": int(len(Z)), "rank": rank, "rank_min_loo": kmin, "cap": cap,
            "loo_mean": sig(mean_e), "loo_se": sig(se_e),
            "one_se_threshold": float(f"{mean_e[kmin - 1] + se_e[kmin - 1]:.4g}"),
            "blank_ids": ids.tolist(), "per_blank": per_blank,
            "top5pct_share": float(e_r[:top5].sum() / e_r.sum()),
            "drop_one": drop, "worst": worst,
            "mean": r(binned(mu_b[None])[0], 5),
            "loadings": [r(binned(V_b[:, [i]].T)[0], 5) for i in range(N_LOAD)],
            "var": r((lam_b ** 2 / np.sum(lam_b ** 2))[:cap], 6),
            "held_corrected": [r(v, 6) for v in hc],
            "capture": capture,
        }
        print(f"{b}: {len(Z)} blanks, rank {rank} (LOO min at {kmin}), top 5 % carry {banks_out[b]['top5pct_share']:.0%}", flush=True)

    payload = clean({
        "schema_version": 1,
        "bin_wn": r(bwn, 2),
        "banks": banks_out,
        "bank_labels": {k: v for k, v in {"VIBES-full-r70": "87 blanks (capped, until now)", "VIBES-lot251": "every lot-251 blank",
                        "VIBES-allblanks": "every labeled blank", "VIBES-lot251-x1": "lot-251 blanks minus LABE1",
                        "VIBES-allblanks-x1": "every labeled blank minus LABE1"}.items() if k in BANKS},
        "inventory": {c: [None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in inv[c]] for c in inv.columns},
        "raw": [r(v, 5) for v in rb],
        "groups": group_stats,
        "shared_pca": shared,
        "held_ids": held, "held_source": held_src,
        "sources": {"bundle": str(BUNDLE.relative_to(ROOT)), "capped_run": str(CAPPED.relative_to(ROOT)),
                    "blankbank_runs": "My Drive/Aethmodular VIBES/results/blankbank_20261007"},
    })
    path = OUT / "blanks_20261007.json"
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
