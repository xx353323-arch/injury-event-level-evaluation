import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s3.csv")
OUT_TESTS = os.path.join(OUT_DIR, "table_s3_tests.csv")
OUT_DESIGN = os.path.join(OUT_DIR, "table_s3_design.csv")

LABELS = {
    "XGBoost": "GBT",
    "LSTM": "LSTM",
    "GAT": "GAT",
    "Transformer": "Transformer",
    "ProtoNet": "ProtoNet",
    "DANN": "DANN",
    "ours": "LMVG-TCA",
}
ORDER = ["GBT", "LSTM", "GAT", "Transformer", "ProtoNet", "DANN", "LMVG-TCA"]
REFERENCE = "LSTM"
GROUPS = ["G1", "G2", "G3"]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 42, 123, 2026]
SETTINGS = {"full": None, "k10": "10"}
PROBE_FILES = [("val_runners_probe.json", "full"), ("val_runners_probe_k10.json", "k10")]
EPISODIC_FILES = [("val_runners_probe_tp.json", "full"), ("val_runners_probe_tp_k10.json", "k10")]
ALPHA = 0.05
KEY = ["target", "seed"]


def default_runners_dir():
    if os.path.exists(os.path.join(ROOT, "runners", "val_runners_10seed_G1.json")):
        return os.path.join(ROOT, "runners")
    hits = sorted(glob.glob(os.path.join(ROOT, "..", "..", "*", "runners", "val_runners_10seed_G1.json")))
    if not hits:
        raise FileNotFoundError("runner result files not found; pass --runners-dir")
    return os.path.dirname(os.path.abspath(hits[0]))


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


def load_runs(runners_dir):
    entries = {}

    def put(setting, target, seed, label, result, source):
        if not (isinstance(result, dict) and "auc" in result):
            return
        key = (setting, target, int(seed), LABELS[label])
        if key in entries:
            raise ValueError(f"duplicate valid result for {key}")
        entries[key] = dict(auc=float(result["auc"]), ap=float(result["ap"]), source=source)

    for name, setting in PROBE_FILES:
        for record in json.load(open(os.path.join(runners_dir, name))):
            framework = [k for k in record if k not in ("target", "seed", "baselines")]
            if len(framework) != 1:
                raise ValueError("unexpected record layout in probe file")
            put(setting, record["target"], record["seed"], framework[0], record[framework[0]], name)
            for label, result in (record.get("baselines") or {}).items():
                put(setting, record["target"], record["seed"], label, result, name)
    for name, setting in EPISODIC_FILES:
        for record in json.load(open(os.path.join(runners_dir, name))):
            for label in ("Transformer", "ProtoNet"):
                put(setting, record["target"], record["seed"], label, record.get(label), name)
    for group in GROUPS:
        name = f"val_runners_10seed_{group}.json"
        for record in json.load(open(os.path.join(runners_dir, name))):
            put(record["regime"], record["target"], record["seed"], record["method"], record.get("result"), name)
    df = pd.DataFrame([dict(setting=k[0], target=k[1], seed=k[2], method=k[3], **v) for k, v in entries.items()])
    expected = len(SETTINGS) * len(GROUPS) * len(SEEDS) * len(ORDER)
    if len(df) != expected:
        raise ValueError(f"expected {expected} runs, found {len(df)}")
    if set(df["seed"]) != set(SEEDS) or set(df["target"]) != set(GROUPS):
        raise ValueError("unexpected seeds or groups")
    return df


def check_design(runners_dir):
    sys.path.insert(0, runners_dir)
    from run_runners_probe import split_target

    d = np.load(os.path.join(runners_dir, "windows_runners.npz"), allow_pickle=True)
    y, tasks = d["y"], d["tasks"]
    rows = []
    for group in GROUPS:
        y_t = y[tasks == group]
        full_support = {}
        for setting, ksup in SETTINGS.items():
            if ksup:
                os.environ["RUNNERS_KSUP"] = ksup
            else:
                os.environ.pop("RUNNERS_KSUP", None)
            supports = set()
            for seed in SEEDS:
                sup_p, te_p, sup_n, te_n = split_target(y_t, np.random.default_rng(seed))
                support = frozenset(np.concatenate([sup_p, sup_n]).tolist())
                supports.add(support)
                if setting == "full":
                    full_support[seed] = support
                nested = bool(support <= full_support[seed])
                rows.append(
                    dict(
                        target=group,
                        setting=setting,
                        seed=seed,
                        support_pos=len(sup_p),
                        support_neg=len(sup_n),
                        test_pos=len(te_p),
                        test_neg=len(te_n),
                        support_within_full_support=nested,
                    )
                )
            if len(supports) != len(SEEDS):
                raise ValueError(f"seeds do not give distinct splits: {group} {setting}")
        os.environ.pop("RUNNERS_KSUP", None)
    design = pd.DataFrame(rows)
    if not design["support_within_full_support"].all():
        raise ValueError("restricted support is not nested in the full support")
    return design


def paired(a, b):
    diff = (a - b).values
    return dict(
        n=len(diff),
        mean_difference=float(diff.mean()),
        shapiro_p=float(stats.shapiro(diff).pvalue),
        t_p=float(stats.ttest_rel(a, b).pvalue),
        wilcoxon_p=float(stats.wilcoxon(diff).pvalue),
        wins=int((diff > 0).sum()),
    )


def run_tests(df):
    auc = {s: df[df["setting"] == s].pivot_table(index=KEY, columns="method", values="auc") for s in SETTINGS}
    if not auc["full"].index.equals(auc["k10"].index) or auc["full"].shape != (len(GROUPS) * len(SEEDS), len(ORDER)):
        raise ValueError("settings are not matched over the same thirty group-seed pairs")
    families = []
    change = pd.DataFrame([dict(method=m, **paired(auc["k10"][m], auc["full"][m])) for m in ORDER])
    change.insert(0, "comparison", "ten-event minus full support")
    change.insert(0, "family", "support size, seven methods")
    families.append(change)
    for setting, label in [("full", "full support"), ("k10", "ten-event support")]:
        others = [m for m in ORDER if m != REFERENCE]
        vs = pd.DataFrame([dict(method=m, **paired(auc[setting][m], auc[setting][REFERENCE])) for m in others])
        vs.insert(0, "comparison", f"method minus LSTM, {label}")
        vs.insert(0, "family", f"against LSTM, {label}, six methods")
        families.append(vs)
    for frame in families:
        frame["t_p_holm"] = holm(frame["t_p"])
        frame["wilcoxon_p_holm"] = holm(frame["wilcoxon_p"])
        frame["conclusions_differ"] = (frame["t_p_holm"] < ALPHA) != (frame["wilcoxon_p_holm"] < ALPHA)
    tests = pd.concat(families, ignore_index=True)
    columns = ["family", "comparison", "method", "n", "mean_difference", "shapiro_p", "t_p", "t_p_holm",
               "wilcoxon_p", "wilcoxon_p_holm", "conclusions_differ", "wins"]
    return tests[columns], auc


def summarise(df, tests, auc):
    rows = []
    blocks = [(g, df["target"] == g) for g in GROUPS] + [("G1 to G3 pooled", df["target"].isin(GROUPS))]
    lookup = tests.set_index(["family", "method"])
    for method in ORDER:
        for block, mask in blocks:
            sub = df[mask & (df["method"] == method)]
            full = sub[sub["setting"] == "full"].set_index(KEY).sort_index()
            k10 = sub[sub["setting"] == "k10"].set_index(KEY).sort_index()
            keys = full.index
            ref_full = auc["full"].loc[keys, REFERENCE]
            ref_k10 = auc["k10"].loc[keys, REFERENCE]
            row = dict(
                method=method,
                group=block,
                n_splits=len(keys),
                auc_full_mean=float(full["auc"].mean()),
                auc_full_sd=float(full["auc"].std(ddof=1)),
                ap_full_mean=float(full["ap"].mean()),
                ap_full_sd=float(full["ap"].std(ddof=1)),
                auc_k10_mean=float(k10["auc"].mean()),
                auc_k10_sd=float(k10["auc"].std(ddof=1)),
                ap_k10_mean=float(k10["ap"].mean()),
                ap_k10_sd=float(k10["ap"].std(ddof=1)),
                auc_change_k10_minus_full=float((k10["auc"] - full["auc"]).mean()),
                auc_diff_vs_lstm_full=float((full["auc"] - ref_full).mean()),
                auc_diff_vs_lstm_k10=float((k10["auc"] - ref_k10).mean()),
                splits_above_lstm_full=int((full["auc"] > ref_full).sum()),
                splits_above_lstm_k10=int((k10["auc"] > ref_k10).sum()),
            )
            if block == "G1 to G3 pooled":
                ch = lookup.loc[("support size, seven methods", method)]
                row.update(p_change_t=ch["t_p"], p_change_t_holm=ch["t_p_holm"],
                           p_change_wilcoxon=ch["wilcoxon_p"], p_change_wilcoxon_holm=ch["wilcoxon_p_holm"],
                           shapiro_p_change=ch["shapiro_p"])
                if method != REFERENCE:
                    for setting, label in [("full", "full support"), ("k10", "ten-event support")]:
                        vs = lookup.loc[(f"against LSTM, {label}, six methods", method)]
                        row.update({f"p_vs_lstm_{setting}_t": vs["t_p"], f"p_vs_lstm_{setting}_t_holm": vs["t_p_holm"]})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runners-dir", default=None)
    args = parser.parse_args()
    runners_dir = os.path.abspath(args.runners_dir) if args.runners_dir else default_runners_dir()
    df = load_runs(runners_dir)
    design = check_design(runners_dir)
    tests, auc = run_tests(df)
    table = summarise(df, tests, auc)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    tests.to_csv(OUT_TESTS, index=False)
    design.to_csv(OUT_DESIGN, index=False)
    print(f"Supplementary Table S3 written to {os.path.relpath(OUT, ROOT)} ({len(table)} rows), "
          f"{os.path.relpath(OUT_TESTS, ROOT)} and {os.path.relpath(OUT_DESIGN, ROOT)}")
    print(f"Runs: {len(df)} ({len(SETTINGS)} settings x {len(GROUPS)} groups x {len(SEEDS)} seeds x {len(ORDER)} methods)")
    summary = design.groupby(["target", "setting"])[["support_pos", "support_neg", "test_pos", "test_neg"]].agg(["min", "max"])
    print(summary.to_string())
    print("Two-sided paired tests over thirty group-seed pairs, alpha = 0.05, Holm within each family")
    for r in tests.itertuples():
        print(f"  {r.family:40s} {r.method:12s} diff {r.mean_difference:+.4f}  t P {r.t_p:.2g} (Holm {r.t_p_holm:.2g})  "
              f"Wilcoxon P {r.wilcoxon_p:.2g} (Holm {r.wilcoxon_p_holm:.2g})  Shapiro-Wilk P {r.shapiro_p:.2g}  "
              f"wins {r.wins}/{r.n}{'  tests disagree' if r.conclusions_differ else ''}")


if __name__ == "__main__":
    main()
