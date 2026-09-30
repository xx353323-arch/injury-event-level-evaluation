import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SOCCER = os.path.join(ROOT, "output_v2", "metrics_soccermon_event_v2_random.csv")
RUNNERS = os.path.join(ROOT, "output_v2", "metrics_runners_event_v2_random.csv")
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
METHODS = ["GBT", "LSTM", "GAT", "Transformer", "ProtoNet", "DANN", "LMVG-TCA"]
FLOOR = "AthleteRate"
REFERENCE = "LSTM"
TARGETS = ["TeamA-2020", "TeamA-2021"]
GROUPS = ["G1", "G2", "G3"]
MARGIN = 0.02
ALPHA = 0.05


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def load(path):
    df = pd.read_csv(path)
    df["method"] = df["method"].map(LABELS)
    if df["method"].isna().any():
        raise ValueError(f"unmapped method label in {path}")
    return df


def auc_table(df, key):
    table = df.pivot_table(index=key, columns="method", values="auc")
    if table.isna().any().any():
        raise ValueError("methods are not matched over the same seed-split pairs")
    return table


def paired_family(table, reference, methods):
    rows = []
    for method in methods:
        d = (table[method] - table[reference]).values
        rows.append(dict(
            method=method,
            n=len(d),
            mean_difference=float(d.mean()),
            shapiro_p=float(stats.shapiro(d).pvalue),
            t_p=float(stats.ttest_1samp(d, 0).pvalue),
            wilcoxon_p=float(stats.wilcoxon(d).pvalue),
        ))
    out = pd.DataFrame(rows)
    out["t_p_holm"] = holm(out["t_p"])
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"])
    return out


def tost_t(d, margin=MARGIN):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    lower = stats.t.sf((d.mean() + margin) / se, n - 1)
    upper = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(lower, upper))


def tost_wilcoxon(d, margin=MARGIN):
    lower = stats.wilcoxon(d + margin, alternative="greater").pvalue
    upper = stats.wilcoxon(d - margin, alternative="less").pvalue
    return float(max(lower, upper))


def equivalence_family(table, reference, methods, margin=MARGIN):
    rows = []
    for method in methods:
        d = (table[method] - table[reference]).values
        rows.append(dict(
            method=method,
            n=len(d),
            mean_difference=float(d.mean()),
            shapiro_p=float(stats.shapiro(d).pvalue),
            t_p=tost_t(d, margin),
            wilcoxon_p=tost_wilcoxon(d, margin),
        ))
    out = pd.DataFrame(rows)
    out["t_p_holm"] = holm(out["t_p"])
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"])
    return out


def football_families():
    df = load(SOCCER)
    families = {}
    for target in TARGETS:
        sub = df[df.target == target]
        if sub.split_id.nunique() != 10 or sub.seed.nunique() != 10:
            raise ValueError(f"{target} must have ten seeds with ten distinct splits")
        table = auc_table(sub, "seed")
        families[(target, "floor")] = paired_family(table, FLOOR, METHODS)
        families[(target, "lstm")] = paired_family(table, REFERENCE, [m for m in METHODS if m != REFERENCE])
    return families


def runner_families():
    df = load(RUNNERS)
    if set(df.target) != set(GROUPS) or len(df) != len(GROUPS) * 10 * len(LABELS):
        raise ValueError("unexpected layout of the running-cohort metrics file")
    table = auc_table(df, ["target", "seed"])
    others = [m for m in METHODS if m != REFERENCE]
    return {
        ("Runners", "floor"): paired_family(table, FLOOR, METHODS),
        ("Runners", "lstm"): paired_family(table, REFERENCE, others),
        ("Runners", "tost"): equivalence_family(table, REFERENCE, others),
    }


def show(title, frame):
    print(title)
    print(frame.round(4).to_string(index=False))
    print()


def main():
    pd.set_option("display.width", 200)
    print("Football cohort, injury-event-grouped protocol of Table 6, n = 10 seed-split pairs per target")
    for (target, kind), frame in football_families().items():
        label = "against the athlete-identity floor" if kind == "floor" else "against LSTM"
        show(f"{target}, methods {label}", frame)
    print("Running cohort, three groups pooled over thirty group-by-seed splits")
    names = {"floor": "against the athlete-identity floor", "lstm": "against LSTM",
             "tost": f"equivalence with LSTM, two one-sided tests at a margin of {MARGIN} AUC"}
    for (_, kind), frame in runner_families().items():
        show(f"Runners, methods {names[kind]}", frame)


if __name__ == "__main__":
    main()
