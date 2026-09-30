import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from event_split import _win_event_id, chronological_window_split, event_level_split

SENSITIVITY = os.path.join(ROOT, "output", "val_protocol_sensitivity.json")
WINDOWS = os.path.join(ROOT, "data", "windows.npz")
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s2.csv")

TARGETS = ["TeamA-2020", "TeamA-2021"]
SEEDS = [42, 123, 2026]
ASSIGNMENTS = {
    "chrono_window": "chronological window assignment",
    "event_emb21": "injury-event grouping, 21-day embargo",
}
EVENT_EMBARGO = 21
COUNT_KEYS = ["n_sup_pos", "n_te_pos", "n_sup_neg", "n_te_neg"]


def load_runs():
    runs = {}
    for record in json.load(open(SENSITIVITY)):
        if record["protocol"] in ASSIGNMENTS and record["target"] in TARGETS and record["seed"] in SEEDS:
            if record["status"] != "ok":
                raise ValueError("failed run in sensitivity file")
            runs[(record["target"], record["protocol"], record["seed"])] = record
    expected = len(TARGETS) * len(ASSIGNMENTS) * len(SEEDS)
    if len(runs) != expected:
        raise ValueError(f"expected {expected} runs, found {len(runs)}")
    return runs


def rebuild(protocol, y_t, p_t, d_t, seed):
    rng = np.random.default_rng(seed)
    if protocol == "chrono_window":
        return chronological_window_split(y_t, d_t, rng)
    return event_level_split(y_t, p_t, d_t, rng, embargo_days=EVENT_EMBARGO)


def event_overlap(sup_p, te_p, p_t, d_t):
    sup_events = [_win_event_id(p_t[i], d_t[i]) for i in sup_p]
    te_events = [_win_event_id(p_t[i], d_t[i]) for i in te_p]
    if any(e is None for e in sup_events + te_events):
        raise ValueError("positive window without an injury event")
    shared = set(sup_events) & set(te_events)
    return dict(
        events_total=len(set(sup_events) | set(te_events)),
        events_support=len(set(sup_events)),
        events_test=len(set(te_events)),
        events_both_sides=len(shared),
        support_pos_in_shared_events=sum(e in shared for e in sup_events),
        test_pos_in_shared_events=sum(e in shared for e in te_events),
    )


def summarise(runs):
    d = np.load(WINDOWS, allow_pickle=True)
    y, tasks, players = d["y"], d["tasks"], d["players"]
    dates = pd.to_datetime(d["dates"]).values
    rows = []
    for target in TARGETS:
        mask = tasks == target
        y_t, p_t, d_t = y[mask], players[mask], dates[mask].astype("datetime64[D]")
        for protocol, label in ASSIGNMENTS.items():
            splits, overlaps = [], []
            for seed in SEEDS:
                sup_p, te_p, sup_n, te_n, info = rebuild(protocol, y_t, p_t, d_t, seed)
                archived = runs[(target, protocol, seed)]["split"]
                if any(int(info[k]) != int(archived[k]) for k in COUNT_KEYS):
                    raise ValueError(f"rebuilt split differs from archive: {target} {protocol} seed {seed}")
                splits.append(tuple(np.sort(np.concatenate([sup_p, sup_n])).tolist()))
                overlaps.append(event_overlap(sup_p, te_p, p_t, d_t))
            records = [runs[(target, protocol, seed)] for seed in SEEDS]
            auc = np.array([r["result"]["auc"] for r in records])
            ap = np.array([r["result"]["ap"] for r in records])
            counts = pd.DataFrame([r["split"] for r in records])
            overlap = pd.DataFrame(overlaps)
            if overlap.drop(columns=["support_pos_in_shared_events", "test_pos_in_shared_events"]).nunique().max() != 1:
                raise ValueError("event counts vary across seeds")
            rows.append(
                dict(
                    target=target,
                    assignment=label,
                    n_seeds=len(SEEDS),
                    seeds=" ".join(str(s) for s in SEEDS),
                    n_distinct_splits=len(set(splits)),
                    support_pos_min=int(counts["n_sup_pos"].min()),
                    support_pos_max=int(counts["n_sup_pos"].max()),
                    test_pos_min=int(counts["n_te_pos"].min()),
                    test_pos_max=int(counts["n_te_pos"].max()),
                    support_neg_min=int(counts["n_sup_neg"].min()),
                    support_neg_max=int(counts["n_sup_neg"].max()),
                    test_neg_min=int(counts["n_te_neg"].min()),
                    test_neg_max=int(counts["n_te_neg"].max()),
                    events_total=int(overlap["events_total"].iloc[0]),
                    events_support=int(overlap["events_support"].iloc[0]),
                    events_test=int(overlap["events_test"].iloc[0]),
                    events_both_sides=int(overlap["events_both_sides"].iloc[0]),
                    support_pos_in_shared_events_min=int(overlap["support_pos_in_shared_events"].min()),
                    support_pos_in_shared_events_max=int(overlap["support_pos_in_shared_events"].max()),
                    test_pos_in_shared_events_min=int(overlap["test_pos_in_shared_events"].min()),
                    test_pos_in_shared_events_max=int(overlap["test_pos_in_shared_events"].max()),
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
    table = summarise(runs)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S2 written to {os.path.relpath(OUT, ROOT)} ({len(table)} rows)")
    print("Seeds 42, 123 and 2026; mean and s.d. (ddof = 1)")
    for r in table.itertuples():
        print(
            f"  {r.target} {r.assignment}: distinct splits {r.n_distinct_splits}  "
            f"pos {r.support_pos_min}-{r.support_pos_max}/{r.test_pos_min}-{r.test_pos_max}  "
            f"neg {r.support_neg_min}-{r.support_neg_max}/{r.test_neg_min}-{r.test_neg_max}  "
            f"events support/test/both {r.events_support}/{r.events_test}/{r.events_both_sides} of {r.events_total}  "
            f"test pos in shared events {r.test_pos_in_shared_events_min}-{r.test_pos_in_shared_events_max}  "
            f"AUC {r.auc_mean:.3f} +/- {r.auc_sd:.3f}  AP {r.ap_mean:.3f} +/- {r.ap_sd:.3f}"
        )


if __name__ == "__main__":
    main()
