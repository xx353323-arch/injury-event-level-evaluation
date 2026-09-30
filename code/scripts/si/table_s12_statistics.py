import argparse
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

import ablation_holm as ah
import stats_robust as sr

ROOT = sr.ROOT
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s12.csv")
ALPHA = sr.ALPHA
SEEDS = [0, 1, 2, 3, 4, 5, 6, 42, 123, 2026]
SETTINGS = ["full", "k10"]
PROBE_FILES = [("val_runners_probe.json", "full"), ("val_runners_probe_k10.json", "k10")]
EPISODIC_FILES = [("val_runners_probe_tp.json", "full"), ("val_runners_probe_tp_k10.json", "k10")]
TABLE6 = "injury-event-grouped protocol of Table 6"
UNPURGED = "unpurged event-grouped split"
WINDOW = "window-level split of the running cohort"
POOLED = "Running cohort, G1 to G3 pooled"


def default_runners_dir():
    if os.path.exists(os.path.join(ROOT, "runners", "val_runners_10seed_G1.json")):
        return os.path.join(ROOT, "runners")
    hits = sorted(glob.glob(os.path.join(ROOT, "..", "..", "*", "runners", "val_runners_10seed_G1.json")))
    if not hits:
        raise FileNotFoundError("running-cohort support-size result files not found; pass --runners-dir")
    return os.path.dirname(os.path.abspath(hits[0]))


def load_support_runs(runners_dir):
    runs = {}

    def put(setting, target, seed, label, result):
        if not (isinstance(result, dict) and "auc" in result):
            return
        key = (setting, target, int(seed), sr.LABELS[label])
        if key in runs:
            raise ValueError(f"duplicate result for {key}")
        runs[key] = float(result["auc"])

    for name, setting in PROBE_FILES:
        for record in json.load(open(os.path.join(runners_dir, name))):
            put(setting, record["target"], record["seed"], "ours", record.get("ours"))
            for label, result in (record.get("baselines") or {}).items():
                put(setting, record["target"], record["seed"], label, result)
    for name, setting in EPISODIC_FILES:
        for record in json.load(open(os.path.join(runners_dir, name))):
            for label in ("Transformer", "ProtoNet"):
                put(setting, record["target"], record["seed"], label, record.get(label))
    for group in sr.GROUPS:
        for record in json.load(open(os.path.join(runners_dir, f"val_runners_10seed_{group}.json"))):
            put(record["regime"], record["target"], record["seed"], record["method"], record.get("result"))
    df = pd.DataFrame([dict(setting=k[0], target=k[1], seed=k[2], method=k[3], auc=v) for k, v in runs.items()])
    expected = len(SETTINGS) * len(sr.GROUPS) * len(SEEDS) * len(sr.METHODS)
    if len(df) != expected or set(df.seed) != set(SEEDS) or set(df.target) != set(sr.GROUPS):
        raise ValueError(f"expected {expected} support-size runs over three groups and ten seeds, found {len(df)}")
    tables = {}
    for setting in SETTINGS:
        tables[setting] = df[df.setting == setting].pivot_table(index=["target", "seed"], columns="method", values="auc")
        if tables[setting].shape != (len(sr.GROUPS) * len(SEEDS), len(sr.METHODS)):
            raise ValueError("support-size runs are not matched over thirty group-by-seed pairs")
    return tables


def change_family(before, after, methods):
    rows = []
    for method in methods:
        d = (after[method] - before[method]).values
        rows.append(dict(
            method=method,
            n=len(d),
            mean_difference=float(d.mean()),
            shapiro_p=float(stats.shapiro(d).pvalue),
            t_p=float(stats.ttest_1samp(d, 0).pvalue),
            wilcoxon_p=float(stats.wilcoxon(d).pvalue),
        ))
    out = pd.DataFrame(rows)
    out["t_p_holm"] = sr.holm(out["t_p"])
    out["wilcoxon_p_holm"] = sr.holm(out["wilcoxon_p"])
    return out


def annotate(frame, order, family, target, protocol, comparison, difference, test):
    frame = frame.rename(columns={"method": "row", "variant": "row"}).copy()
    frame.insert(0, "family_order", order)
    frame.insert(1, "family", family)
    frame.insert(2, "target", target)
    frame.insert(3, "protocol", protocol)
    frame.insert(4, "comparison", comparison)
    frame.insert(5, "difference", difference)
    frame.insert(6, "test", test)
    frame["family_size"] = len(frame)
    frame["conclusions_differ"] = (frame["t_p_holm"] < ALPHA) != (frame["wilcoxon_p_holm"] < ALPHA)
    frame["shapiro_below_alpha"] = frame["shapiro_p"] < ALPHA
    return frame


def build(runners_dir):
    football = sr.football_families()
    runners = sr.runner_families()
    support = load_support_runs(runners_dir)
    others = [m for m in sr.METHODS if m != sr.REFERENCE]
    paired = "two-sided paired t-test; two-sided Wilcoxon signed-rank test"
    tost = "two one-sided paired t-tests; two one-sided Wilcoxon signed-rank tests"
    specs = []
    for target in sr.TARGETS:
        specs.append((football[(target, "floor")], f"{target}, seven methods against the athlete-identity floor",
                      target, TABLE6, "against the athlete-identity floor", "method minus AthleteRate", paired))
    for target in sr.TARGETS:
        specs.append((football[(target, "lstm")], f"{target}, six methods against LSTM",
                      target, TABLE6, "against LSTM", "method minus LSTM", paired))
    specs.append((runners[("Runners", "floor")], f"{POOLED}, seven methods against the athlete-identity floor",
                   "Runners G1 to G3 pooled", TABLE6, "against the athlete-identity floor", "method minus AthleteRate", paired))
    specs.append((runners[("Runners", "lstm")], f"{POOLED}, six methods against LSTM",
                  "Runners G1 to G3 pooled", TABLE6, "against LSTM", "method minus LSTM", paired))
    specs.append((runners[("Runners", "tost")], f"{POOLED}, six equivalence tests against LSTM at a margin of {sr.MARGIN} AUC",
                  "Runners G1 to G3 pooled", TABLE6, "equivalence with LSTM", "method minus LSTM", tost))
    for metric, label in (("auc", "AUC"), ("ap", "average precision")):
        for target in ah.TARGETS:
            specs.append((ah.ablation_family(target, metric),
                          f"{target}, four ablated components against the full model, {label}",
                          target, UNPURGED, f"ablation, {label}", "variant minus full model", paired))
    specs.append((change_family(support["full"], support["k10"], sr.METHODS),
                  f"{POOLED}, seven methods, ten-event minus full support",
                  "Runners G1 to G3 pooled", WINDOW, "support size", "ten-event minus full support", paired))
    for setting, label in (("full", "full support"), ("k10", "ten-event support")):
        specs.append((sr.paired_family(support[setting], sr.REFERENCE, others),
                      f"{POOLED}, six methods against LSTM with {label}",
                      "Runners G1 to G3 pooled", WINDOW, f"against LSTM, {label}", "method minus LSTM", paired))
    frames = [annotate(frame, i + 1, *rest) for i, (frame, *rest) in enumerate(specs)]
    columns = ["family_order", "family", "target", "protocol", "comparison", "difference", "test", "family_size", "row",
               "n", "mean_difference", "shapiro_p", "t_p", "t_p_holm", "wilcoxon_p", "wilcoxon_p_holm",
               "conclusions_differ", "shapiro_below_alpha"]
    return pd.concat(frames, ignore_index=True)[columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runners-dir", default=None)
    args = parser.parse_args()
    runners_dir = os.path.abspath(args.runners_dir) if args.runners_dir else default_runners_dir()
    table = build(runners_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S12 written to {os.path.relpath(OUT, ROOT)} "
          f"({len(table)} rows in {table.family_order.nunique()} families)")
    print(f"Holm adjustment within each family, separately for the t-tests and the Wilcoxon tests, alpha = {ALPHA}")
    for family, frame in table.groupby("family_order", sort=True):
        print(frame["family"].iloc[0])
        for r in frame.itertuples():
            flags = []
            if r.conclusions_differ:
                flags.append("tests disagree after Holm adjustment")
            if r.shapiro_below_alpha:
                flags.append("Shapiro-Wilk P < 0.05")
            print(f"  {r.row:36s} n={r.n} difference={r.mean_difference:+.4f} Shapiro-Wilk P={r.shapiro_p:.3g} "
                  f"t P={r.t_p:.3g} (Holm {r.t_p_holm:.3g}) Wilcoxon P={r.wilcoxon_p:.3g} (Holm {r.wilcoxon_p_holm:.3g})"
                  f"{'  [' + '; '.join(flags) + ']' if flags else ''}")


if __name__ == "__main__":
    main()
