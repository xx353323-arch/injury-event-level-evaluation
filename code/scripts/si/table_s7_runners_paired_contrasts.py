import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SOURCE = os.path.join(ROOT, "output_v2", "metrics_runners_event_v2_random.csv")
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s7.csv")

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
GROUPS = ["G1", "G2", "G3"]
FLOOR = "AthleteRate"
REFERENCE = "LSTM"
MARGIN = 0.02
KEY = ["target", "seed"]


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


def tost_p(differences, margin):
    n = len(differences)
    se = differences.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((differences.mean() + margin) / se, n - 1)
    p_upper = stats.t.cdf((differences.mean() - margin) / se, n - 1)
    return float(max(p_lower, p_upper))


def load():
    df = pd.read_csv(SOURCE)
    df["method"] = df["method"].map(LABELS)
    if df["method"].isna().any():
        raise ValueError("unmapped method label")
    if len(df) != len(GROUPS) * len(ORDER) * 10:
        raise ValueError("unexpected number of runs")
    if df.groupby(KEY)["split_id"].nunique().max() != 1:
        raise ValueError("methods do not share the split of a group-seed pair")
    if (df.groupby("target")["split_id"].nunique() != 10).any():
        raise ValueError("a group does not have ten distinct splits")
    return df


def block_rows(df, label):
    auc = df.pivot_table(index=KEY, columns="method", values="auc")
    means = df.groupby("method")[["auc", "auc_within_athlete", "auc_between_athlete"]].mean()
    rows = []
    for method in ORDER:
        rows.append(
            dict(
                block=label,
                method=method,
                n_splits=int(auc[method].notna().sum()),
                n_distinct_splits=int(df[df["method"] == method][["target", "split_id"]].drop_duplicates().shape[0]),
                n_test_events=int(df[df["method"] == method].groupby("target")["n_events"].first().sum()),
                test_prevalence=float(df[df["method"] == method]["prevalence"].mean()),
                auc=float(means.loc[method, "auc"]),
                auc_within_athlete=float(means.loc[method, "auc_within_athlete"]),
                auc_between_athlete=float(means.loc[method, "auc_between_athlete"]),
                auc_diff_vs_athleterate=float((auc[method] - auc[FLOOR]).mean()),
                auc_diff_vs_lstm=float((auc[method] - auc[REFERENCE]).mean()),
            )
        )
    return pd.DataFrame(rows), auc


def pooled_tests(auc):
    vs_floor = [m for m in ORDER if m != FLOOR]
    vs_reference = [m for m in ORDER if m not in (FLOOR, REFERENCE)]
    p_floor = [stats.ttest_rel(auc[m], auc[FLOOR]).pvalue for m in vs_floor]
    p_reference = [stats.ttest_rel(auc[m], auc[REFERENCE]).pvalue for m in vs_reference]
    p_tost = [tost_p((auc[m] - auc[REFERENCE]).values, MARGIN) for m in vs_reference]
    tests = {m: {} for m in ORDER}
    for m, p, h in zip(vs_floor, p_floor, holm(p_floor)):
        tests[m].update(p_t_vs_athleterate=float(p), p_t_vs_athleterate_holm=float(h))
    for m, p, h in zip(vs_reference, p_reference, holm(p_reference)):
        tests[m].update(p_t_vs_lstm=float(p), p_t_vs_lstm_holm=float(h))
    for m, p, h in zip(vs_reference, p_tost, holm(p_tost)):
        tests[m].update(p_tost_vs_lstm=float(p), p_tost_vs_lstm_holm=float(h))
    return tests


def main():
    df = load()
    frames = []
    for group in GROUPS:
        rows, _ = block_rows(df[df["target"] == group], group)
        frames.append(rows)
    pooled, auc = block_rows(df, "G1 to G3 pooled")
    tests = pooled_tests(auc)
    columns = [
        "p_t_vs_athleterate", "p_t_vs_athleterate_holm",
        "p_t_vs_lstm", "p_t_vs_lstm_holm",
        "p_tost_vs_lstm", "p_tost_vs_lstm_holm",
    ]
    for column in columns:
        pooled[column] = [tests[m].get(column, np.nan) for m in pooled["method"]]
    frames.append(pooled)
    table = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S7 written to {os.path.relpath(OUT, ROOT)} ({len(table)} rows)")
    print("Pooled running cohort, n = 30 matched splits, two-sided tests, alpha = 0.05, Holm within each family")
    view = table[table["block"] == "G1 to G3 pooled"].set_index("method")
    for method in ORDER:
        r = view.loc[method]
        print(
            f"  {method:12s} AUC {r.auc:.3f}  within {r.auc_within_athlete:.3f}  "
            f"vs AthleteRate {r.auc_diff_vs_athleterate:+.3f} (P {r.p_t_vs_athleterate:.2g}, Holm {r.p_t_vs_athleterate_holm:.2g})  "
            f"vs LSTM {r.auc_diff_vs_lstm:+.3f} (P {r.p_t_vs_lstm:.2g}, Holm {r.p_t_vs_lstm_holm:.2g})  "
            f"TOST P {r.p_tost_vs_lstm:.2g} (Holm {r.p_tost_vs_lstm_holm:.2g})"
        )


if __name__ == "__main__":
    main()
