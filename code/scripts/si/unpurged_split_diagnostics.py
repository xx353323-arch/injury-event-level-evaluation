import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import event_split

OUT = os.path.join(ROOT, "output_v2", "si", "unpurged_split_diagnostics.csv")
FOOTBALL = ["TeamA-2020", "TeamA-2021", "TeamB-2020"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 42, 123, 2026]
WINDOW_DAYS = 14


def overlapping(a, b, players, dates):
    by_player = defaultdict(list)
    for j in b:
        by_player[players[j]].append(dates[j])
    count = 0
    for i in a:
        if any(abs(int((dates[i] - e).astype(int))) <= WINDOW_DAYS - 1 for e in by_player.get(players[i], [])):
            count += 1
    return count


def athlete_rate(sup, te, y, players):
    rates = pd.Series(y[sup]).groupby(players[sup]).mean()
    prior = float(y[sup].mean())
    return np.array([rates.get(p, prior) for p in players[te]])


def main():
    ap = argparse.ArgumentParser(description="Overlap between support and test windows and the athlete-identity baseline under the unpurged event-grouped split of Table 7.")
    ap.parse_args()
    d = np.load(os.path.join(ROOT, "data", "windows.npz"), allow_pickle=True)
    y, players, tasks = d["y"], d["players"].astype(str), d["tasks"].astype(str)
    dates = pd.to_datetime(d["dates"]).values.astype("datetime64[D]")
    recorded = {(r["target"], r["seed"]): r["split"] for r in json.load(open(os.path.join(ROOT, "output", "val_ablation_eventsplit_full.json"))) if r.get("status") == "ok"}
    rows = []
    for target in FOOTBALL:
        m = tasks == target
        ym, pm, dm = y[m], players[m], dates[m]
        for seed in SEEDS:
            sp, tp, sn, tn, info = event_split.event_level_split(ym, pm, dm, np.random.default_rng(seed))
            if info != recorded.get((target, seed)):
                raise SystemExit(f"{target} seed {seed}: regenerated split does not match val_ablation_eventsplit_full.json")
            sup = np.concatenate([sp, sn]).astype(int)
            te = np.concatenate([tp, tn]).astype(int)
            score = athlete_rate(sup, te, ym, pm)
            rows.append(dict(target=target, seed=seed, n_te_pos=len(tp), n_te_neg=len(tn),
                             te_pos_sharing_observation_days_with_support_pos=overlapping(tp, sp, pm, dm),
                             frac_te_neg_overlapping_support_neg=overlapping(tn, sn, pm, dm) / max(len(tn), 1),
                             athlete_rate_auc=float(roc_auc_score(ym[te], score)), athlete_rate_ap=float(average_precision_score(ym[te], score))))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Unpurged event-grouped split of Table 7, ten seeds per target")
    for target, g in df.groupby("target", sort=False):
        print(f"{target}: test positive windows sharing observation days with a support positive window "
              f"{g.te_pos_sharing_observation_days_with_support_pos.min()} to {g.te_pos_sharing_observation_days_with_support_pos.max()}; "
              f"test negative windows overlapping a support negative window {g.frac_te_neg_overlapping_support_neg.min():.3f} to "
              f"{g.frac_te_neg_overlapping_support_neg.max():.3f}; AthleteRate AUC {g.athlete_rate_auc.mean():.3f} "
              f"(s.d. {g.athlete_rate_auc.std(ddof=0):.3f}), AP {g.athlete_rate_ap.mean():.3f}")
    counts = os.path.join(ROOT, "output_v2", "si", "table1_table3_per_split.csv")
    table1 = os.path.join(ROOT, "output_v2", "si", "table1_counts.csv")
    if os.path.exists(counts) and os.path.exists(table1):
        c = pd.read_csv(counts)
        c = c[(c.cohort == "soccermon") & (c.task == "TeamA-2020")]
        t1 = pd.read_csv(table1)
        n_events = int(t1.loc[t1.Task == "TeamA-2020", "Injury events"].iloc[0])
        assigned = n_events - c.n_te_events
        removed = assigned - c.n_sup_events
        print(f"TeamA-2020 under the protocol of Table 6: purging removes every positive window of {removed.min()} to {removed.max()} "
              f"of the {int(assigned.iloc[0])} support events")


if __name__ == "__main__":
    main()
