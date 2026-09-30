import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from event_split import EMBARGO, event_level_split

SENSITIVITY = os.path.join(ROOT, "output", "val_protocol_sensitivity.json")
REFERENCE = os.path.join(ROOT, "output", "val_ablation_eventsplit_full.json")
WINDOWS = os.path.join(ROOT, "data", "windows.npz")
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s1.csv")

TARGETS = ["TeamA-2020", "TeamA-2021"]
SEEDS = [42, 123, 2026]
VARIED = {0: "event_emb0", 7: "event_emb7", 21: "event_emb21"}
REFERENCE_EMBARGO = 20
EMBARGOES = [0, 7, REFERENCE_EMBARGO, 21]
SPLIT_KEYS = ["n_events", "n_sup_events", "n_te_events", "n_sup_pos", "n_te_pos", "n_sup_neg", "n_te_neg"]


def load_runs():
    runs = {}
    for record in json.load(open(SENSITIVITY)):
        for embargo, protocol in VARIED.items():
            if record["protocol"] == protocol and record["target"] in TARGETS and record["seed"] in SEEDS:
                if record["status"] != "ok":
                    raise ValueError("failed run in sensitivity file")
                runs[(record["target"], embargo, record["seed"])] = record
    for record in json.load(open(REFERENCE)):
        if record["target"] in TARGETS and record["seed"] in SEEDS:
            if record["status"] != "ok":
                raise ValueError("failed run in reference file")
            runs[(record["target"], REFERENCE_EMBARGO, record["seed"])] = record
    expected = len(TARGETS) * len(EMBARGOES) * len(SEEDS)
    if len(runs) != expected:
        raise ValueError(f"expected {expected} runs, found {len(runs)}")
    return runs


def rebuild_splits(runs):
    if EMBARGO != REFERENCE_EMBARGO:
        raise ValueError("default embargo of the split routine is not twenty days")
    d = np.load(WINDOWS, allow_pickle=True)
    y, tasks, players = d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    for target in TARGETS:
        mask = tasks == target
        y_t, p_t, d_t = y[mask], players[mask], dates[mask].astype("datetime64[D]")
        assignments = set()
        for seed in SEEDS:
            support_positive = None
            for embargo in EMBARGOES:
                sup_p, te_p, sup_n, te_n, info = event_level_split(
                    y_t, p_t, d_t, np.random.default_rng(seed), embargo_days=embargo
                )
                archived = runs[(target, embargo, seed)]["split"]
                if any(int(info[k]) != int(archived[k]) for k in SPLIT_KEYS):
                    raise ValueError(f"rebuilt split differs from archive: {target} seed {seed} embargo {embargo}")
                if support_positive is None:
                    support_positive = frozenset(sup_p.tolist())
                elif frozenset(sup_p.tolist()) != support_positive:
                    raise ValueError(f"positive-window assignment differs across embargoes: {target} seed {seed}")
            assignments.add(support_positive)
        if len(assignments) != len(SEEDS):
            raise ValueError(f"seeds do not give distinct event assignments: {target}")


def summarise(runs):
    rows = []
    for target in TARGETS:
        for embargo in EMBARGOES:
            records = [runs[(target, embargo, seed)] for seed in SEEDS]
            auc = np.array([r["result"]["auc"] for r in records])
            ap = np.array([r["result"]["ap"] for r in records])
            split = pd.DataFrame([r["split"] for r in records])
            if split["n_sup_events"].nunique() != 1 or split["n_te_events"].nunique() != 1:
                raise ValueError("event counts vary across seeds")
            rows.append(
                dict(
                    target=target,
                    embargo_days=embargo,
                    row_type="reference, full model of Table 7" if embargo == REFERENCE_EMBARGO else "varied embargo",
                    n_seeds=len(SEEDS),
                    seeds=" ".join(str(s) for s in SEEDS),
                    n_distinct_splits=len(SEEDS),
                    support_events=int(split["n_sup_events"].iloc[0]),
                    test_events=int(split["n_te_events"].iloc[0]),
                    support_pos_min=int(split["n_sup_pos"].min()),
                    support_pos_max=int(split["n_sup_pos"].max()),
                    test_pos_min=int(split["n_te_pos"].min()),
                    test_pos_max=int(split["n_te_pos"].max()),
                    support_neg_min=int(split["n_sup_neg"].min()),
                    support_neg_max=int(split["n_sup_neg"].max()),
                    test_neg_min=int(split["n_te_neg"].min()),
                    test_neg_max=int(split["n_te_neg"].max()),
                    auc_mean=float(auc.mean()),
                    auc_sd=float(auc.std(ddof=1)),
                    auc_min=float(auc.min()),
                    auc_max=float(auc.max()),
                    ap_mean=float(ap.mean()),
                    ap_sd=float(ap.std(ddof=1)),
                    ap_min=float(ap.min()),
                    ap_max=float(ap.max()),
                )
            )
    return pd.DataFrame(rows)


def main():
    runs = load_runs()
    rebuild_splits(runs)
    table = summarise(runs)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S1 written to {os.path.relpath(OUT, ROOT)} ({len(table)} rows)")
    print("Unpurged event-grouped split, seeds 42, 123 and 2026, one split per seed; mean and s.d. (ddof = 1)")
    for r in table.itertuples():
        print(
            f"  {r.target} embargo {r.embargo_days:2d} d  events {r.support_events}/{r.test_events}  "
            f"pos {r.support_pos_min}-{r.support_pos_max}/{r.test_pos_min}-{r.test_pos_max}  "
            f"test neg {r.test_neg_min}-{r.test_neg_max}  AUC {r.auc_mean:.3f} +/- {r.auc_sd:.3f}  "
            f"AP {r.ap_mean:.3f} +/- {r.ap_sd:.3f}  [{r.row_type}]"
        )


if __name__ == "__main__":
    main()
