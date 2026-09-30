import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
V2 = os.path.join(ROOT, "output_v2")
COHORTS = {"soccermon": ["TeamA-2020", "TeamA-2021", "TeamB-2020"], "runners": ["G1", "G2", "G3"]}
OUT_DIR = os.path.join(V2, "si")
OUT = os.path.join(OUT_DIR, "table_s8.csv")

LABELS = {
    "XGBoost": "GBT",
    "LSTM": "LSTM",
    "GAT": "GAT",
    "Transformer": "Transformer",
    "ProtoNet": "ProtoNet",
    "DANN": "DANN",
    "AthleteRate": "AthleteRate",
    "ours": "LMVG-TCA",
}
ORDER = ["GBT", "LSTM", "GAT", "Transformer", "ProtoNet", "DANN", "AthleteRate", "LMVG-TCA"]
REFERENCE = "LSTM"
MIN_EVENTS = 5
SEEDS = [42, 123, 2026]


def read(kind):
    frames = []
    for cohort in COHORTS:
        df = pd.read_csv(os.path.join(V2, f"{kind}_{cohort}_athlete_grouped5.csv"))
        df["cohort"] = cohort
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["method"] = df["method"].map(LABELS)
    if df["method"].isna().any():
        raise ValueError("unmapped method label")
    if sorted(df["seed"].unique()) != SEEDS:
        raise ValueError("unexpected seeds")
    return df


def distinct_partitions(cohort, target):
    assignments = {}
    for path in glob.glob(os.path.join(V2, "dumps", cohort, "athlete_grouped5", f"{target}_*__{REFERENCE}__s*.npz")):
        name = os.path.basename(path)[:-4]
        fold_target, _, seed = name.split("__")
        fold = int(fold_target.rsplit("_", 1)[1])
        info = json.loads(str(np.load(path, allow_pickle=True)["info"]))
        assignments.setdefault(int(seed[1:]), set()).add((fold, tuple(sorted(info["held_out_athletes"]))))
    if sorted(assignments) != SEEDS:
        return np.nan
    return len({frozenset(v) for v in assignments.values()})


def main():
    pooled = read("pooled")
    folds = read("metrics")
    parts = folds["target"].str.rsplit("_", n=1)
    folds["fold"] = parts.str[1].astype(int)
    folds["target"] = parts.str[0]
    events = pooled.groupby("target")["n_events"].agg(["min", "max"])
    if (events["min"] != events["max"]).any():
        raise ValueError("event count differs between seeds or methods")
    targets = [t for c in ("soccermon", "runners") for t in COHORTS[c] if t in events.index]
    evaluable = [t for t in targets if events.loc[t, "min"] >= MIN_EVENTS]
    excluded = [t for t in targets if t not in evaluable]
    cohort_of = {t: c for c, ts in COHORTS.items() for t in ts}
    rows = []
    for target in evaluable:
        p = pooled[pooled["target"] == target]
        f = folds[folds["target"] == target]
        reference = p[p["method"] == REFERENCE].set_index("seed")
        n_partitions = distinct_partitions(cohort_of[target], target)
        for method in ORDER:
            pm = p[p["method"] == method].set_index("seed")
            fm = f[f["method"] == method]
            if sorted(pm.index) != SEEDS:
                raise ValueError("missing seed")
            n_combinations = int(pm["n_folds"].sum())
            if n_combinations != len(fm):
                raise ValueError("fold count mismatch between pooled and per-fold files")
            diff = (pm["auc_within_athlete"] - reference.loc[pm.index, "auc_within_athlete"]).mean()
            rows.append(
                dict(
                    target=target,
                    method=method,
                    n_seeds=int(len(pm)),
                    n_distinct_partitions=n_partitions,
                    n_evaluable_fold_seed=n_combinations,
                    n_injury_events=int(pm["n_events"].iloc[0]),
                    n_athletes=int(pm["n_athletes"].iloc[0]),
                    auc=float(pm["auc"].mean()),
                    auc_within_athlete=float(pm["auc_within_athlete"].mean()),
                    auc_within_diff_vs_lstm=float(diff),
                    auc_within_fold_sd=float(fm["auc_within_athlete"].std(ddof=1)),
                )
            )
    table = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S8 written to {os.path.relpath(OUT, ROOT)} ({len(table)} rows)")
    print(f"Not evaluable (fewer than {MIN_EVENTS} injury events), reported in Supplementary Table S11: {', '.join(excluded)}")
    for target in evaluable:
        t = table[table["target"] == target].set_index("method")
        print(
            f"{target}: {int(t['n_injury_events'].iloc[0])} events, "
            f"{int(t['n_evaluable_fold_seed'].iloc[0])} evaluable fold-by-seed combinations, "
            f"{t['n_distinct_partitions'].iloc[0]} distinct athlete partitions over {int(t['n_seeds'].iloc[0])} seeds"
        )
        for method in ORDER:
            r = t.loc[method]
            print(
                f"  {method:12s} AUC {r.auc:.3f}  within {r.auc_within_athlete:.3f}  "
                f"within vs LSTM {r.auc_within_diff_vs_lstm:+.3f}  fold s.d. {r.auc_within_fold_sd:.3f}"
            )
    seven = table[table["method"] != "AthleteRate"]
    floor = table[table["method"] == "AthleteRate"]
    print(f"AthleteRate pooled overall AUC range {floor['auc'].min():.3f} to {floor['auc'].max():.3f}")
    print(f"Seven methods, within-athlete AUC range {seven['auc_within_athlete'].min():.3f} to {seven['auc_within_athlete'].max():.3f}")


if __name__ == "__main__":
    main()
