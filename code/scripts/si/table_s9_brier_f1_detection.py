import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
V2 = os.path.join(ROOT, "output_v2")
SOURCES = [os.path.join(V2, f"metrics_{c}_event_v2_random.csv") for c in ("soccermon", "runners")]
OUT_DIR = os.path.join(V2, "si")
OUT = os.path.join(OUT_DIR, "table_s9.csv")

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
TARGET_ORDER = ["TeamA-2020", "TeamA-2021", "TeamB-2020", "G1", "G2", "G3"]
MIN_EVENTS = 5
Z = 1.96
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_interval(text):
    values = [float(v) for v in NUMBER.findall(re.sub(r"np\.float64", "", str(text)))]
    if len(values) != 2:
        raise ValueError(f"cannot parse interval {text!r}")
    return values


def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.maximum(0.0, centre - half), np.minimum(1.0, centre + half)


def load():
    df = pd.concat([pd.read_csv(f) for f in SOURCES], ignore_index=True)
    df["method"] = df["method"].map(LABELS)
    if df["method"].isna().any():
        raise ValueError("unmapped method label")
    bounds = np.array([parse_interval(v) for v in df["event_detection_ci"]])
    df["wilson_lower"], df["wilson_upper"] = bounds[:, 0], bounds[:, 1]
    k = np.rint(df["event_detection_rate"] * df["n_events"])
    if np.abs(df["event_detection_rate"] * df["n_events"] - k).max() > 1e-9:
        raise ValueError("detection rate is not a proportion of test events")
    lower, upper = wilson(k, df["n_events"])
    if max(np.abs(lower - df["wilson_lower"]).max(), np.abs(upper - df["wilson_upper"]).max()) > 1e-12:
        raise ValueError("stored interval is not the Wilson 95% interval")
    return df


def main():
    df = load()
    events = df.groupby("target")["n_events"].agg(["min", "max"])
    if (events["min"] != events["max"]).any():
        raise ValueError("test event count differs between runs of a target")
    targets = [t for t in TARGET_ORDER if t in events.index]
    evaluable = [t for t in targets if events.loc[t, "min"] >= MIN_EVENTS]
    excluded = [t for t in targets if t not in evaluable]
    runs = df[df["target"].isin(evaluable)]
    rows = []
    for target in evaluable:
        for method in ORDER:
            r = runs[(runs["target"] == target) & (runs["method"] == method)]
            rows.append(
                dict(
                    target=target,
                    method=method,
                    n_runs=int(len(r)),
                    n_distinct_splits=int(r["split_id"].nunique()),
                    n_test_events=int(r["n_events"].iloc[0]),
                    brier_skill_mean=float(r["brier_skill"].mean()),
                    brier_skill_sd=float(r["brier_skill"].std(ddof=1)),
                    brier_skill_max=float(r["brier_skill"].max()),
                    n_runs_negative_brier_skill=int((r["brier_skill"] < 0).sum()),
                    f1_test_optimal_mean=float(r["f1_oracle"].mean()),
                    f1_test_optimal_sd=float(r["f1_oracle"].std(ddof=1)),
                    event_detection_rate_mean=float(r["event_detection_rate"].mean()),
                    event_detection_rate_sd=float(r["event_detection_rate"].std(ddof=1)),
                    wilson_lower_min=float(r["wilson_lower"].min()),
                    wilson_upper_max=float(r["wilson_upper"].max()),
                )
            )
    table = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S9 written to {os.path.relpath(OUT, ROOT)} ({len(table)} rows)")
    print(f"Not evaluable (fewer than {MIN_EVENTS} test injury events), reported in Supplementary Table S11: {', '.join(excluded)}")
    print(
        f"Runs on evaluable targets: {len(runs)}; Brier skill negative in {int((runs['brier_skill'] < 0).sum())}; "
        f"largest value {runs['brier_skill'].max():.3f}"
    )
    for target in evaluable:
        t = table[table["target"] == target].set_index("method")
        print(f"{target}: {int(t['n_test_events'].iloc[0])} test events, {int(t['n_distinct_splits'].iloc[0])} distinct splits")
        for method in ORDER:
            r = t.loc[method]
            print(
                f"  {method:12s} Brier skill {r.brier_skill_mean:8.2f} (s.d. {r.brier_skill_sd:.2f}, max {r.brier_skill_max:.2f})  "
                f"test-optimal F1 {r.f1_test_optimal_mean:.3f}  detection {r.event_detection_rate_mean:.3f} "
                f"[{r.wilson_lower_min:.3f}, {r.wilson_upper_max:.3f}]"
            )


if __name__ == "__main__":
    main()
