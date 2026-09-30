import glob
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import event_split
import table9_v2 as weekly_analysis

TARGET = "TeamB-2020"
FRAMEWORK = "ours"
OUTPUT = os.path.join(ROOT, "output")
OUTPUT_V2 = os.path.join(ROOT, "output_v2")
METRICS = os.path.join(OUTPUT_V2, "metrics_soccermon_event_v2_random.csv")
WINDOW_DUMPS = os.path.join(OUTPUT_V2, "dumps", "soccermon", "window_random")
WINDOW_SUMMARY = os.path.join(OUTPUT_V2, f"summary_soccermon_window_random_{TARGET}.jsonl")
POOLED_AG = os.path.join(OUTPUT_V2, "pooled_soccermon_athlete_grouped5.csv")
FOLDS_AG = os.path.join(OUTPUT_V2, "metrics_soccermon_athlete_grouped5.csv")
SENSITIVITY = os.path.join(OUTPUT, "val_protocol_sensitivity.json")
WINDOWS = os.path.join(ROOT, "data", "windows.npz")
OUT_DIR = os.path.join(OUTPUT_V2, "si")
OUT = os.path.join(OUT_DIR, "table_s11.csv")

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
SEEDS = [0, 1, 2, 3, 4, 5, 6, 42, 123, 2026]
SENSITIVITY_SEEDS = [42, 123, 2026]
MAIN_COLUMNS = ["auc", "auc_within_athlete", "auc_between_athlete", "ap", "lift", "f1_calibrated",
                "event_detection_rate"]
CALIBRATION_COLUMNS = ["brier_skill", "f1_oracle"]
VARIANTS = [
    ("full", "LMVG-TCA (full)"),
    ("no_mmd", "w/o MMD alignment"),
    ("no_focal", "w/o focal loss"),
    ("no_temporal", "w/o temporal sampling"),
    ("nvg", "w/o multi-view graph (single view)"),
]
WEEKLY_COLUMNS = ["window_auc", "daily_fpr", "daily_sens", "weekly_fpr", "weekly_sens", "weekly_ppv",
                  "healthy_weeks", "injury_weeks", "alerts_per25", "naive_per25", "brier_skill"]
FIXED_COLUMNS = ["weekly_fpr", "weekly_sens", "weekly_ppv"]
SETTINGS = [
    ("event_emb0", "Embargo 0 days"),
    ("event_emb7", "Embargo 7 days"),
    ("reference_emb20", "Embargo 20 days (full model of Table 7, same seeds)"),
    ("event_emb21", "Embargo 21 days"),
    ("chrono_window", "Chronological window assignment"),
]


def record(panel, block, row, measure, values, units, range_over):
    values = np.asarray(values, dtype=float)
    units = np.asarray(units, dtype=float)
    return dict(panel=panel, block=block, row=row, measure=measure, mean=float(values.mean()),
                range_min=float(units.min()), range_max=float(units.max()), range_over=range_over,
                n_runs=int(values.size), n_units=int(units.size))


def count(panel, block, row, measure, values, range_over="seeds"):
    values = np.asarray(values, dtype=float)
    return dict(panel=panel, block=block, row=row, measure=measure, mean=float(values.mean()),
                range_min=float(values.min()), range_max=float(values.max()), range_over=range_over,
                n_runs=int(values.size), n_units=int(values.size))


def load_event_grouped():
    df = pd.read_csv(METRICS)
    df = df[df.target == TARGET].copy()
    df["method"] = df["method"].map(LABELS)
    if df["method"].isna().any() or len(df) != len(ORDER) * len(SEEDS):
        raise ValueError("unexpected TeamB-2020 rows in the event-grouped metrics file")
    if set(df.seed) != set(SEEDS) or set(df.n_events) != {2}:
        raise ValueError("TeamB-2020 must have ten seeds with two test events per split")
    sizes = sorted(df[df.method == "LMVG-TCA"].groupby("split_id").seed.nunique().tolist())
    if sizes != [3, 3, 4] or df.groupby("seed").split_id.nunique().max() != 1:
        raise ValueError("TeamB-2020 seeds must share three distinct splits (3, 4 and 3 seeds)")
    return df


def panel_a_f(df):
    rows = []
    for panel, columns in (("a", MAIN_COLUMNS), ("f", CALIBRATION_COLUMNS)):
        for method in ORDER:
            sub = df[df.method == method]
            per_split = sub.groupby("split_id")[columns].mean()
            for column in columns:
                rows.append(record(panel, "injury-event-grouped protocol of Table 6", method, column,
                                   sub[column], per_split[column], "distinct splits"))
    return rows


def panel_a_random():
    files = sorted(glob.glob(os.path.join(WINDOW_DUMPS, f"{TARGET}__{FRAMEWORK}__*.npz")))
    runs = []
    for path in files:
        d = np.load(path, allow_pickle=True)
        info = json.loads(str(d["info"]))
        if info.get("protocol") != "window_random":
            raise ValueError(f"unexpected protocol in {path}")
        runs.append(dict(seed=int(os.path.basename(path)[:-4].split("__")[2][1:]),
                         auc=roc_auc_score(d["te_y"], d["te_prob"]),
                         ap=average_precision_score(d["te_y"], d["te_prob"]),
                         share=float(info["frac_te_pos_sharing_event_with_sup"]),
                         split_id=info["split_id"]))
    runs = pd.DataFrame(runs)
    if sorted(runs.seed) != SEEDS or runs.split_id.nunique() != len(SEEDS):
        raise ValueError("the random window-level reference must have ten seeds with ten distinct splits")
    summary = pd.DataFrame([json.loads(line) for line in open(WINDOW_SUMMARY)]).set_index("seed")
    check = runs.set_index("seed")
    if (check.auc - summary.auc.reindex(check.index)).abs().max() > 1e-12 or \
            (check.ap - summary.ap.reindex(check.index)).abs().max() > 1e-12:
        raise ValueError("per-window predictions disagree with the run summary")
    block = "random window-level split"
    return [record("a", block, "LMVG-TCA", column, runs[column], runs[column], "seeds")
            for column in ("auc", "ap", "share")]


def load_windows():
    d = np.load(WINDOWS, allow_pickle=True)
    mask = d["tasks"] == TARGET
    dates = pd.to_datetime(d["dates"]).values[mask].astype("datetime64[D]")
    return d["y"][mask], d["players"][mask], dates


def event_partitions(y, players, dates, ablation):
    partitions = {}
    for seed in SEEDS:
        support_pos, _, _, _, info = event_split.event_level_split(y, players, dates, np.random.default_rng(seed))
        for variant, _ in VARIANTS:
            if info != ablation[variant][seed]["split"]:
                raise ValueError(f"reconstructed split differs from the stored split: {variant}, seed {seed}")
        partitions[seed] = frozenset(event_split._win_event_id(players[i], dates[i]) for i in support_pos)
    labels = {}
    for seed in SEEDS:
        labels.setdefault(partitions[seed], len(labels))
    groups = {seed: labels[partitions[seed]] for seed in SEEDS}
    sizes = sorted(pd.Series(groups).value_counts().tolist())
    if sizes != [2, 4, 4]:
        raise ValueError(f"expected three event partitions shared by four, four and two seeds, found {sizes}")
    return groups


def panel_b(y, players, dates):
    ablation = {}
    for variant, _ in VARIANTS:
        path = os.path.join(OUTPUT, f"val_ablation_eventsplit_{variant}.json")
        runs = {r["seed"]: r for r in json.load(open(path)) if r["target"] == TARGET and r.get("status") == "ok"}
        if sorted(runs) != SEEDS:
            raise ValueError(f"{variant}: TeamB-2020 must have ten completed seeds")
        ablation[variant] = runs
    groups = event_partitions(y, players, dates, ablation)
    frame = pd.DataFrame([dict(variant=v, seed=s, partition=groups[s], auc=ablation[v][s]["result"]["auc"],
                               ap=ablation[v][s]["result"]["ap"]) for v, _ in VARIANTS for s in SEEDS])
    full = frame[frame.variant == "full"].set_index("seed")
    rows = []
    block = "unpurged event-grouped split"
    for variant, name in VARIANTS:
        sub = frame[frame.variant == variant].set_index("seed")
        per_partition = sub.groupby("partition")[["auc", "ap"]].mean()
        for column in ("auc", "ap"):
            rows.append(record("b", block, name, column, sub[column], per_partition[column], "distinct event partitions"))
        if variant != "full":
            delta = (sub.auc - full.auc).rename("delta")
            per_partition_delta = delta.groupby(sub.partition).mean()
            rows.append(record("b", block, name, "delta_auc_vs_full", delta, per_partition_delta,
                               "distinct event partitions"))
    return rows, groups


def panel_c():
    runs = weekly_analysis.per_run()
    runs = runs[runs.target == TARGET]
    rows = []
    for threshold, columns in (("support_rate_matched", WEEKLY_COLUMNS), (f"fixed_{weekly_analysis.FIXED}", FIXED_COLUMNS)):
        sub = runs[runs.threshold == threshold]
        if len(sub) != len(SEEDS) or sub.split_id.nunique() != 3:
            raise ValueError("weekly analysis must use the ten seeds and three distinct splits of Table 6")
        per_split = sub.groupby("split_id")[columns].mean()
        for column in columns:
            rows.append(record("c", threshold, "LMVG-TCA", column, sub[column], per_split[column], "distinct splits"))
    return rows


def panel_d(y, players, dates):
    sensitivity = [r for r in json.load(open(SENSITIVITY)) if r["target"] == TARGET]
    full = {r["seed"]: r for r in json.load(open(os.path.join(OUTPUT, "val_ablation_eventsplit_full.json")))
            if r["target"] == TARGET}
    rows = []
    for setting, name in SETTINGS:
        if setting == "reference_emb20":
            runs = [full[s] for s in SENSITIVITY_SEEDS]
        else:
            runs = sorted((r for r in sensitivity if r["protocol"] == setting), key=lambda r: r["seed"])
        if [r["seed"] for r in runs] != SENSITIVITY_SEEDS or any(r["status"] != "ok" for r in runs):
            raise ValueError(f"{setting}: expected completed runs for seeds 42, 123 and 2026")
        block = "chronological window assignment" if setting == "chrono_window" else "unpurged event-grouped split"
        for column in ("auc", "ap"):
            values = [r["result"][column] for r in runs]
            rows.append(record("d", block, name, column, values, values, "seeds"))
        rows.append(count("d", block, name, "support_positive_windows", [r["split"]["n_sup_pos"] for r in runs]))
        rows.append(count("d", block, name, "test_positive_windows", [r["split"]["n_te_pos"] for r in runs]))
        if setting == "chrono_window":
            support_pos, test_pos, support_neg, test_neg, info = event_split.chronological_window_split(
                y, dates, np.random.default_rng(0))
            if any(info != r["split"] for r in runs):
                raise ValueError("reconstructed chronological split differs from the stored split")
            support_events = {event_split._win_event_id(players[i], dates[i]) for i in support_pos}
            test_events = [event_split._win_event_id(players[i], dates[i]) for i in test_pos]
            if None in support_events or None in test_events:
                raise ValueError("a positive window of the chronological split has no injury event")
            shared = support_events & set(test_events)
            rows.append(count("d", block, name, "injury_events", [len(support_events | set(test_events))] * len(runs)))
            rows.append(count("d", block, name, "injury_events_on_both_sides", [len(shared)] * len(runs)))
            rows.append(count("d", block, name, "test_positive_windows_in_shared_events",
                              [sum(e in shared for e in test_events)] * len(runs)))
    return rows


def panel_e():
    pooled = pd.read_csv(POOLED_AG)
    pooled = pooled[pooled.target == TARGET].copy()
    pooled["method"] = pooled["method"].map(LABELS)
    if pooled["method"].isna().any() or sorted(pooled.seed.unique()) != SENSITIVITY_SEEDS or len(pooled) != 24:
        raise ValueError("unexpected TeamB-2020 rows in the athlete-grouped pooled file")
    folds = pd.read_csv(FOLDS_AG)
    folds = folds[folds.target.str.startswith(TARGET + "_") & (folds.method == FRAMEWORK)]
    evaluable = folds[["target", "seed"]].drop_duplicates()
    block = "athlete-grouped five-fold cross-validation"
    rows = [count("e", block, "all methods", "evaluable_fold_seed_combinations", [len(evaluable)], "single value"),
            count("e", block, "all methods", "injury_events", pooled.n_events.unique(), "single value")]
    for method in ORDER:
        sub = pooled[pooled.method == method]
        for column in ("auc", "auc_within_athlete"):
            rows.append(record("e", block, method, column, sub[column], sub[column], "seeds"))
    return rows


def show(table):
    for r in table.itertuples():
        print(f"  {r.panel} {r.block:42s} {r.row:52s} {r.measure:40s} {r.mean:9.4f} "
              f"({r.range_min:.4f} to {r.range_max:.4f}; over {r.n_units} {r.range_over})")


def main():
    y, players, dates = load_windows()
    event_grouped = load_event_grouped()
    rows = panel_a_f(event_grouped)
    rows += panel_a_random()
    ablation_rows, groups = panel_b(y, players, dates)
    rows += ablation_rows
    rows += panel_c()
    rows += panel_d(y, players, dates)
    rows += panel_e()
    table = pd.DataFrame(rows)
    table["panel"] = pd.Categorical(table["panel"], categories=list("abcdef"), ordered=True)
    table = table.sort_values("panel", kind="stable").reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S11 written to {os.path.relpath(OUT, ROOT)} ({len(table)} rows)")
    splits = event_grouped[event_grouped.method == "LMVG-TCA"].groupby("split_id").seed.apply(sorted).tolist()
    print(f"Distinct splits under the protocol of Table 6 and the seeds sharing them: {splits}")
    partitions = pd.Series(groups).groupby(pd.Series(groups)).apply(lambda s: sorted(s.index)).tolist()
    print(f"Distinct event partitions under the unpurged event-grouped split: {partitions}")
    print("Mean over seeds, with the range of per-split means (or of seeds where each seed has its own split)")
    show(table)


if __name__ == "__main__":
    main()
