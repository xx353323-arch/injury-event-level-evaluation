import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RESULTS = os.path.join(ROOT, "output")
VARIANTS = [
    ("no_mmd", "w/o MMD alignment"),
    ("no_focal", "w/o focal loss"),
    ("no_temporal", "w/o temporal sampling"),
    ("nvg", "w/o multi-view graph (single view)"),
]
TARGETS = ["TeamA-2020", "TeamA-2021"]
SEEDS = 10


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


def load(variant, metric):
    path = os.path.join(RESULTS, f"val_ablation_eventsplit_{variant}.json")
    return {(r["target"], r["seed"]): r["result"][metric] for r in json.load(open(path)) if r.get("status") == "ok"}


def ablation_family(target, metric="auc"):
    full = load("full", metric)
    rows = []
    for variant, name in VARIANTS:
        runs = load(variant, metric)
        keys = sorted(k for k in full if k[0] == target and k in runs)
        if len(keys) != SEEDS:
            raise ValueError(f"{target} {variant}: expected {SEEDS} paired seeds, found {len(keys)}")
        d = np.array([runs[k] - full[k] for k in keys])
        rows.append(dict(
            variant=name,
            n=len(keys),
            mean_difference=float(d.mean()),
            shapiro_p=float(stats.shapiro(d).pvalue),
            t_p=float(stats.ttest_1samp(d, 0).pvalue),
            wilcoxon_p=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        ))
    out = pd.DataFrame(rows)
    out["t_p_holm"] = holm(out["t_p"])
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"])
    return out


def main():
    print("Ablation under the unpurged event-grouped split, ten seeds, four components per target forming one family, "
          "two-sided tests, alpha = 0.05")
    for metric, label in (("auc", "AUC"), ("ap", "average precision")):
        for target in TARGETS:
            for r in ablation_family(target, metric).itertuples():
                print(f"{label:17s} {target} {r.variant:36s} n={r.n} difference={r.mean_difference:+.4f} "
                      f"t P={r.t_p:.4f} Holm={r.t_p_holm:.4f} Wilcoxon P={r.wilcoxon_p:.4f} "
                      f"Holm={r.wilcoxon_p_holm:.4f} Shapiro-Wilk P={r.shapiro_p:.4f}")


if __name__ == "__main__":
    main()
