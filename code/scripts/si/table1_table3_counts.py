import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline_v2"))
import run_v2 as R

SEEDS = [0, 1, 2, 3, 4, 5, 6, 42, 123, 2026]
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
PROTOCOL = "event_v2_random"


def span(values, fmt="{:,}"):
    lo, hi = min(values), max(values)
    return fmt.format(lo) if lo == hi else fmt.format(lo) + "–" + fmt.format(hi)


def support_train_size(C, task, seed):
    sup = task["sup_idx"]
    if len(task["sup_events"]) < 2:
        return int(len(sup)), False
    ep_rng = np.random.default_rng(seed + 1000)
    val_ev = ep_rng.choice(task["sup_events"])
    n_val_pos = int(sum(task["ev_of"](i) == val_ev for i in sup))
    n_sup_neg = int((C["y"][sup] == 0).sum())
    n_val_neg = min(max(4 * n_val_pos, 4), n_sup_neg)
    return int(len(sup)) - n_val_pos - n_val_neg, True


def dump_split_ids(cohort, target):
    ids = {}
    for f in glob.glob(os.path.join(ROOT, "output_v2", "dumps", cohort, PROTOCOL, f"{target}__ours__s*.npz")):
        seed = int(os.path.basename(f).split("__s")[1].split(".")[0])
        ids[seed] = json.loads(str(np.load(f, allow_pickle=True)["info"]))["split_id"]
    return ids


def cohort_rows(cohort):
    C = R.load_cohort(cohort)
    t1, t3, per_split, checks = [], [], [], []
    for t in C["tasks_all"]:
        m = C["tasks"] == t
        pos, neg = int(C["y"][m].sum()), int((C["y"][m] == 0).sum())
        shared = len(set(C["players"][m]) & set(C["players"][~m]))
        infos, sup_tr, src_n, src_pos = [], [], [], []
        known = dump_split_ids(cohort, t) if t in C["targets"] else {}
        for seed in SEEDS:
            task = R.build_task(C, PROTOCOL, t, seed)
            info = task["info"]
            infos.append(info)
            if known and seed in known:
                checks.append((t, seed, known[seed] == info.get("split_id")))
            src = task["src_mask"]
            src_n.append(int(src.sum()))
            src_pos.append(int(C["y"][src].sum()))
            sup_tr.append(support_train_size(C, task, seed))
            per_split.append(dict(cohort=cohort, task=t, seed=seed, split_id=info.get("split_id"), n_sup_events=info["n_sup_events"],
                                  n_te_events=info["n_te_events"], n_sup_pos=info["n_sup_pos"], n_sup_neg=info["n_sup_neg"],
                                  n_te_pos=info["n_te_pos"], n_te_neg=info["n_te_neg"], n_purged_pos_sup=info["n_purged_pos_sup"],
                                  n_purged_neg_te=info["n_purged_neg_te"], residual=info["residual_te_neg_obs_overlap_frac"],
                                  support_train_windows=sup_tr[-1][0], validation_subset=sup_tr[-1][1], source_windows=src_n[-1],
                                  source_positives=src_pos[-1]))
        i0 = infos[0]
        wpe = i0["windows_per_event"]
        t1.append({"Cohort": "SoccerMon" if cohort == "soccermon" else "Runners", "Task": t, "Athletes": i0["n_athletes"],
                   "Injured athletes": i0["n_injured_athletes"], "Injury events": i0["n_events"], "Positive windows": f"{pos:,}",
                   "Negative windows": f"{neg:,}", "Positive windows per event (min / median / max)": " / ".join(f"{x:g}" for x in wpe),
                   "Prevalence": f"{100 * pos / (pos + neg):.2f}%", "Athletes shared with other tasks": shared,
                   "Support / test events per split": span([x["n_sup_events"] for x in infos]) + " / " + span([x["n_te_events"] for x in infos]),
                   "Purged windows per split (positive / negative)": span([x["n_purged_pos_sup"] for x in infos]) + " / " + span([x["n_purged_neg_te"] for x in infos]),
                   "Residual negative overlap": f"{max(x['residual_te_neg_obs_overlap_frac'] for x in infos):.4f}"})
        if t in C["targets"]:
            t3.append(dict(cohort=cohort, task=t, source_windows=src_n, source_positives=src_pos,
                           sup_pos=[x["n_sup_pos"] for x in infos], sup_neg=[x["n_sup_neg"] for x in infos],
                           sup_train=[x[0] for x in sup_tr], has_val=[x[1] for x in sup_tr],
                           te_pos=[x["n_te_pos"] for x in infos], te_neg=[x["n_te_neg"] for x in infos]))
    return t1, t3, per_split, checks


def table3_row(label, recs):
    get = lambda k: [v for r in recs for v in r[k]]
    tr = get("sup_train")
    where = "a support train subset" if any(get("has_val")) else "the full support partition"
    unl = f"32 windows per episode from {where} of {span(tr)} windows"
    return {"Task": label, "Source train (windows / pos)": span(get("source_windows")) + " / " + span(get("source_positives")),
            "Target labelled support (pos / neg)": span(get("sup_pos")) + " / " + span(get("sup_neg")), "Target unlabelled": unl,
            "Target test (pos / neg)": span(get("te_pos")) + " / " + span(get("te_neg"))}


def main():
    ap = argparse.ArgumentParser(description="Counts of Table 1 and Table 3, recomputed from the partitioning code of Table 6.")
    ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t1, t3, per_split, checks = [], [], [], []
    for cohort in ("soccermon", "runners"):
        a, b, c, d = cohort_rows(cohort)
        t1 += a
        t3 += b
        per_split += c
        checks += d
    rows3 = [table3_row(r["task"], [r]) for r in t3 if r["cohort"] == "soccermon"]
    rows3.append(table3_row("Runners G1–G3", [r for r in t3 if r["cohort"] == "runners"]))
    pd.DataFrame(t1).to_csv(os.path.join(OUT_DIR, "table1_counts.csv"), index=False)
    pd.DataFrame(rows3).to_csv(os.path.join(OUT_DIR, "table3_counts.csv"), index=False)
    pd.DataFrame(per_split).to_csv(os.path.join(OUT_DIR, "table1_table3_per_split.csv"), index=False)
    ok = sum(c[2] for c in checks)
    print(f"splits reproduced against the per-window prediction files: {ok} of {len(checks)}")
    print(pd.DataFrame(t1).to_string(index=False))
    print(pd.DataFrame(rows3).to_string(index=False))
    if ok != len(checks):
        raise SystemExit("split reproduction failed")


if __name__ == "__main__":
    main()
