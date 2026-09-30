import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s10.csv")
HEADER = ["Item", "Cohort", "Partitioning implementation", "Seeds", "Distinct splits", "Inputs", "Result files", "Script"]
TEN = [0, 1, 2, 3, 4, 5, 6, 42, 123, 2026]
THREE = [42, 123, 2026]
S_TEN = "0–6, 42, 123, 2026"
S_THREE = "42, 123, 2026"
NA = "Not applicable"
OBS = "Observed"
FOOTBALL = "SoccerMon (TeamA-2020, TeamA-2021)"
RUNNERS = "Running cohort (G1 to G3)"
RUNNERS_LC = "running cohort (G1 to G3)"
T6 = "Injury-event-grouped protocol of Table 6"
UNPURGED = "Unpurged event-grouped split"
EV_DUMPS = ["output_v2/dumps/soccermon/event_v2_random/ (split metadata)", "output_v2/dumps/runners/event_v2_random/ (split metadata)"]
METRICS = ["output_v2/metrics_soccermon_event_v2_random.csv", "output_v2/metrics_runners_event_v2_random.csv"]
ABLATION = "output/val_ablation_eventsplit_{full,no_mmd,no_focal,no_temporal,nvg}.json"
RUNNER_FILES = ["runners/val_runners_probe.json", "runners/val_runners_probe_k10.json", "runners/val_runners_probe_tp.json",
                "runners/val_runners_probe_tp_k10.json", "runners/val_runners_10seed_{G1,G2,G3}.json"]

ROWS = [
    dict(item="Table 1", cohort="SoccerMon; running cohort",
         impl="Injury events reconstructed with g = 7 days; per-split columns from the injury-event-grouped protocol of Table 6",
         seeds=S_TEN, splits="10 each for TeamA-2020 and TeamA-2021, 3 for TeamB-2020 and 10 per running group; TeamB-2021 is a source task only",
         inputs=OBS, results=["data/windows.npz", "runners/windows_runners.npz", "injury/injury.csv (SoccerMon release)"] + EV_DUMPS
         + ["output_v2/si/table1_counts.csv", "output_v2/si/table1_table3_per_split.csv"],
         scripts=["scripts/si/table1_table3_counts.py", "pipeline_v2/events.py", "pipeline_v2/splits.py", "pipeline_v2/run_v2.py"]),
    dict(item="Table 2 (Algorithm 1)", cohort="Both cohorts", impl="Training procedure of the pipeline of Table 6",
         seeds=NA, splits=NA, inputs="Not applicable (the table describes the released code)", results=[],
         scripts=["pipeline_v2/run_v2.py"]),
    dict(item="Table 3", cohort="SoccerMon; running cohort", impl=T6, seeds=S_TEN,
         splits="10 each for TeamA-2020 and TeamA-2021, 3 for TeamB-2020 and 10 per running group",
         inputs=OBS, results=list(EV_DUMPS) + ["output_v2/si/table3_counts.csv", "output_v2/si/table1_table3_per_split.csv"],
         scripts=["scripts/si/table1_table3_counts.py", "pipeline_v2/splits.py", "pipeline_v2/run_v2.py"]),
    dict(item="Table 4", cohort="Both cohorts", impl="Access of each method to the partitions in the pipeline of Table 6",
         seeds=NA, splits=NA, inputs="Not applicable (the table describes the released code)", results=[],
         scripts=["pipeline_v2/run_v2.py", "scripts/baselines_comparison.py", "scripts/new_baselines.py", "scripts/run_dann_10seed.py"]),
    dict(item="Table 5", cohort="Both cohorts", impl="Settings of the evaluated framework in the released code",
         seeds=NA, splits=NA, inputs="Not applicable (the table describes the released code)", results=[],
         scripts=["pipeline_v2/run_v2.py", "src/lmvg_v2.py", "src/tca_gnn.py"]),
    dict(item="Table 6", cohort=f"{FOOTBALL}; running cohort (G1 to G3 pooled)", impl=T6, seeds=S_TEN,
         splits="10 per football target; 10 per running group, 30 pooled", inputs=OBS,
         results=METRICS + [d.replace("split metadata", "per-window predictions and split metadata") for d in EV_DUMPS],
         scripts=["pipeline_v2/run_v2.py", "pipeline_v2/analyze_v2.py", "scripts/si/stats_robust.py (paired tests and Holm adjustment)",
                  "scripts/si/stats_exact.py (pooled running-cohort tests and equivalence tests)"]),
    dict(item="Section 4.1, random window-level reference and proportion of positive test windows sharing an injury event with the support partition",
         cohort=FOOTBALL, impl="Window-level random split in the pipeline of Table 6 (protocol window_random), LMVG-TCA only",
         seeds=S_TEN, splits="10 per target, one per seed", inputs=OBS,
         results=["output_v2/summary_soccermon_window_random_*.jsonl",
                  "output_v2/dumps/soccermon/window_random/ (per-window predictions, split indices and shared-event proportions)"],
         scripts=["pipeline_v2/run_v2.py", "pipeline_v2/splits.py"]),
    dict(item="Section 4.1, athlete-grouped cross-validation, and Supplementary Table S8", cohort=f"{RUNNERS}; {FOOTBALL}",
         impl="Athlete-grouped five-fold cross-validation in the pipeline of Table 6 (protocol athlete_grouped5)", seeds=S_THREE,
         splits="3 assignments of athletes to five folds per target; 15 evaluable fold-by-seed combinations per running group and for TeamA-2020, 12 for TeamA-2021",
         inputs=OBS,
         results=["output_v2/pooled_runners_athlete_grouped5.csv", "output_v2/pooled_soccermon_athlete_grouped5.csv",
                  "output_v2/metrics_runners_athlete_grouped5.csv", "output_v2/metrics_soccermon_athlete_grouped5.csv",
                  "output_v2/dumps/runners/athlete_grouped5/", "output_v2/dumps/soccermon/athlete_grouped5/", "output_v2/si/table_s8.csv"],
         scripts=["pipeline_v2/run_v2.py", "pipeline_v2/analyze_v2.py", "scripts/si/table_s8_athlete_grouped.py"]),
    dict(item="Table 7", cohort=FOOTBALL, impl=UNPURGED, seeds=S_TEN, splits="10 event assignments per target", inputs=OBS,
         results=[ABLATION, "output_v2/si/table7.csv", "output/split_indices/unpurged_event_grouped_ablation.npz", "output/split_indices/split_indices_counts.csv"],
         scripts=["scripts/run_ablation_eventsplit.py", "scripts/event_split.py", "scripts/si/table7_ablation.py",
                  "scripts/si/ablation_holm.py (Holm-adjusted P values)", "scripts/si/split_indices.py (split indices)"]),
    dict(item="Table 8", cohort=NA,
         impl="Timing of one supervised optimisation step and one inference pass per method; no partition",
         seeds="42 (input generation and initialisation); four repeated runs", splits=NA,
         inputs="Randomly generated windows of shape 14 × 10; 500 synthetic windows for GBT",
         results=["output_v2/cost_table8/computational_cost_table.csv", "output_v2/cost_table8/computational_cost_results.json",
                  "output_v2/cost_table8_run.log (first run)", "output_v2/cost_table8/rep{1,2,3}/ (three further runs)",
                  "output_v2/cost_table8/rep{1,2,3}.log", "output_v2/cost_table8/computational_cost_median.csv (medians over the four runs)"],
         scripts=["scripts/compute_cost_table.py", "scripts/si/table8_cost_median.py (medians over the four runs)"]),
    dict(item="Table 9", cohort=FOOTBALL,
         impl="Per-window predictions of LMVG-TCA from the runs of Table 6, support-rate-matched threshold", seeds=S_TEN,
         splits="10 per target", inputs=OBS,
         results=["output_v2/dumps/soccermon/event_v2_random/ (per-window predictions)", "output_v2/si/table9_v2_per_run.csv",
                  "output_v2/si/table9.csv"],
         scripts=["scripts/si/table9_v2.py"]),
    dict(item="Figure 1", cohort=NA, impl="Schematic", seeds=NA, splits=NA, inputs="Not applicable (schematic)", results=[],
         scripts=["figures/fig1_framework.tex"]),
    dict(item="Figure 2", cohort="SoccerMon", impl="All 35,550 windows pooled; no partition", seeds=NA, splits=NA, inputs=OBS,
         results=["data/windows.npz", "figures/fig2_correlation_matrix.csv"], scripts=["scripts/plot_fig2_channel_correlation.py"]),
    dict(item="Figure 3", cohort=f"{FOOTBALL}; running cohort (G1 to G3 pooled)", impl=T6, seeds=S_TEN,
         splits="10 per football target; 10 per running group, 30 pooled", inputs=OBS,
         results=METRICS + ["figures/fig3_source_data.csv"], scripts=["scripts/plot_fig3_main_comparison.py"]),
    dict(item="Figure 4", cohort=FOOTBALL, impl=UNPURGED, seeds=S_TEN, splits="10 event assignments per target", inputs=OBS,
         results=[ABLATION, "figures/fig4_source_data.csv", "figures/fig4_ablation_tests.csv", "output/split_indices/unpurged_event_grouped_ablation.npz"],
         scripts=["scripts/plot_fig4_ablation.py"]),
    dict(item="Supplementary Table S1", cohort=FOOTBALL,
         impl="Unpurged event-grouped split with embargoes of 0, 7 and 21 days; the 20-day row is the full model of Table 7",
         seeds=S_THREE, splits="3 per target, one per seed; the event assignment of each seed is that of Table 7", inputs=OBS,
         results=["output/val_protocol_sensitivity.json", "output/val_ablation_eventsplit_full.json", "output_v2/si/table_s1.csv",
                  "output/split_indices/unpurged_event_grouped_embargo.npz", "output/split_indices/unpurged_event_grouped_ablation.npz (20-day rows)"],
         scripts=["scripts/run_protocol_sensitivity.py", "scripts/run_ablation_eventsplit.py", "scripts/si/table_s1_embargo.py",
                  "scripts/si/split_indices.py (split indices)"]),
    dict(item="Supplementary Table S2", cohort=FOOTBALL,
         impl="Chronological window assignment, compared with the unpurged event-grouped split with a 21-day embargo",
         seeds=S_THREE, splits="1 chronological split per target, shared by the three seeds; 3 event assignments per target for the event-grouped rows",
         inputs=OBS, results=["output/val_protocol_sensitivity.json", "data/windows.npz", "output_v2/si/table_s2.csv",
                              "output/split_indices/chronological.npz", "output/split_indices/unpurged_event_grouped_embargo.npz (event-grouped rows)"],
         scripts=["scripts/run_protocol_sensitivity.py", "scripts/event_split.py", "scripts/si/table_s2_chronological.py",
                  "scripts/si/split_indices.py (split indices)"]),
    dict(item="Supplementary Table S3", cohort=RUNNERS,
         impl="Window-level split of the running cohort, with full and ten-event support partitions", seeds=S_TEN,
         splits="10 per group in each support setting, one per seed; 30 pooled", inputs=OBS,
         results=RUNNER_FILES + ["output_v2/si/table_s3.csv", "output_v2/si/table_s3_tests.csv", "output_v2/si/table_s3_design.csv",
                                 "output/split_indices/runners_window_level_support_size.npz"],
         scripts=["runners/run_runners_probe.py", "runners/run_runners_probe_tp.py", "runners/run_runners_10seed.py",
                  "scripts/si/table_s3_support_size.py", "scripts/si/split_indices.py (split indices)"]),
    dict(item="Supplementary Table S4", cohort="SoccerMon",
         impl="Injury events reconstructed with merge intervals g from 0 to 28 days; no partition", seeds=NA, splits=NA, inputs=OBS,
         results=["injury/injury.csv (SoccerMon release)", "data/windows.npz", "output_v2/si/table_s4.csv", "output_v2/si/table_s4_gaps.csv"],
         scripts=["pipeline_v2/events.py", "scripts/si/table_s4_merge_interval.py"]),
    dict(item="Supplementary Table S5", cohort="Thirty published studies", impl="Literature coding; no data partition",
         seeds=NA, splits=NA, inputs="Published studies coded by a single coder",
         results=["data/literature_audit_coding_sheet.csv", "output_v2/si/table_s5.csv"],
         scripts=["scripts/si/table_s5_literature_audit.py"]),
    dict(item="Supplementary Table S6", cohort="SoccerMon (TeamA-2020 as target)",
         impl="One meta-iteration of the pipeline of Table 6 from initialisation; no partition", seeds=S_THREE, splits="No partition",
         inputs="Observed windows; framework at initialisation", results=["output_v2/si/table_s6.csv"],
         scripts=["scripts/trace_lmvg_gradients.py"]),
    dict(item="Supplementary Table S7", cohort=RUNNERS, impl=T6, seeds=S_TEN, splits="10 per group, 30 pooled", inputs=OBS,
         results=["output_v2/metrics_runners_event_v2_random.csv", "output_v2/si/table_s7.csv"],
         scripts=["scripts/si/table_s7_runners_paired_contrasts.py"]),
    dict(item="Supplementary Table S9", cohort=f"{FOOTBALL}; {RUNNERS_LC}", impl=T6, seeds=S_TEN, splits="10 per target", inputs=OBS,
         results=METRICS + ["output_v2/si/table_s9.csv"],
         scripts=["pipeline_v2/analyze_v2.py", "scripts/si/table_s9_brier_f1_detection.py"]),
    dict(item="Supplementary Table S11", cohort="SoccerMon (TeamB-2020)",
         impl="Injury-event-grouped protocol of Table 6 (panels a, c and f), window-level random split (panel a), unpurged event-grouped split and chronological assignment (panels b and d) and athlete-grouped cross-validation (panel e)",
         seeds=f"{S_TEN} (panels a, b, c and f); {S_THREE} (panels d and e)",
         splits="3 under event grouping, shared by 3, 4 and 3 seeds; 10 for the random window-level reference, one per seed; 3 event assignments in the ablation, shared by 4, 4 and 2 seeds; 3 event assignments for the embargo variants, one per seed; 1 chronological split; 3 assignments of athletes to folds, with 9 evaluable fold-by-seed combinations",
         inputs=OBS,
         results=["output_v2/metrics_soccermon_event_v2_random.csv", "output_v2/summary_soccermon_window_random_TeamB-2020.jsonl",
                  "output_v2/dumps/soccermon/window_random/", ABLATION, "output_v2/si/table9_v2_per_run.csv",
                  "output/val_protocol_sensitivity.json", "output_v2/pooled_soccermon_athlete_grouped5.csv",
                  "output_v2/metrics_soccermon_athlete_grouped5.csv", "output_v2/si/table_s11.csv"],
         scripts=["scripts/si/table_s11_teamb2020.py"]),
    dict(item="Supplementary Table S12", cohort=f"{FOOTBALL}; {RUNNERS_LC}",
         impl="Families from Table 6 (injury-event-grouped protocol), Table 7 (unpurged event-grouped split) and Supplementary Table S3 (window-level split of the running cohort)",
         seeds=S_TEN, splits="As in the source tables: 10 per football target and 30 pooled running splits; 10 event assignments per target; 30 group-by-seed splits",
         inputs=OBS, results=METRICS + [ABLATION] + RUNNER_FILES + ["output_v2/si/table_s12.csv"],
         scripts=["scripts/si/table_s12_statistics.py", "scripts/si/stats_robust.py", "scripts/si/ablation_holm.py"]),
    dict(item="Supplementary Table S13", cohort=FOOTBALL,
         impl="Per-window predictions of LMVG-TCA from the runs of Table 6, fixed threshold of 0.5", seeds=S_TEN,
         splits="10 per target", inputs=OBS,
         results=["output_v2/si/table9_v2_per_run.csv", "output_v2/si/table_s13.csv"],
         scripts=["scripts/si/table_s13_fixed_threshold.py", "scripts/si/table9_v2.py"]),
]


def expand(pattern):
    m = re.search(r"\{([^}]*)\}", pattern)
    if not m:
        return [pattern]
    return list(itertools.chain.from_iterable(expand(pattern[:m.start()] + alt + pattern[m.end():]) for alt in m.group(1).split(",")))


def default_dir(pattern, anchor, first=None):
    if first and os.path.exists(os.path.join(ROOT, *first)):
        return os.path.dirname(os.path.abspath(os.path.join(ROOT, *first)))
    hits = sorted(glob.glob(os.path.join(ROOT, *pattern)))
    return os.path.dirname(os.path.abspath(hits[0])) if hits else os.path.join(ROOT, anchor)


def resolve(path, dirs):
    path = path.split(" (")[0]
    if path.startswith("runners/") and not os.path.exists(os.path.join(ROOT, "runners")):
        return os.path.join(dirs["runners"], path[len("runners/"):])
    if path.startswith("injury/") and not os.path.exists(os.path.join(ROOT, "injury", path[len("injury/"):].split(" (")[0])):
        return os.path.join(dirs["injury"], path[len("injury/"):])
    return os.path.join(ROOT, path)


def missing_files(dirs):
    missing = []
    for row in ROWS:
        for entry in row["results"] + row["scripts"]:
            for p in expand(entry.split(" (")[0]):
                full = resolve(p, dirs)
                if not glob.glob(full) and not os.path.exists(full):
                    missing.append((row["item"], p))
    return missing


def event_assignments(seeds, reference, embargo_days=None):
    import event_split
    d = np.load(os.path.join(ROOT, "data", "windows.npz"), allow_pickle=True)
    y, players, tasks = d["y"], d["players"], d["tasks"].astype(str)
    dates = pd.to_datetime(d["dates"]).values
    out = {}
    for target in ["TeamA-2020", "TeamA-2021", "TeamB-2020"]:
        m = tasks == target
        ym, pm, dm = y[m], players[m], dates[m].astype("datetime64[D]")
        sets = {}
        for s in seeds:
            sp, tp, sn, tn, info = event_split.event_level_split(ym, pm, dm, np.random.default_rng(s), embargo_days=embargo_days)
            if info != reference[(target, s)]:
                raise ValueError(f"{target} seed {s}: rebuilt split does not match the result file")
            sets[s] = frozenset(event_split._win_event_id(pm[i], dm[i]) for i in sp)
        out[target] = sets
    return out


def checks(dirs):
    rep = []

    def check(name, observed, expected):
        rep.append((name, observed, expected, observed == expected))

    ev = pd.read_csv(os.path.join(ROOT, METRICS[0]))
    for t, n in [("TeamA-2020", 10), ("TeamA-2021", 10), ("TeamB-2020", 3)]:
        s = ev[ev.target == t]
        check(f"Table 6 seeds, {t}", sorted(s.seed.unique().tolist()), TEN)
        check(f"Table 6 distinct splits, {t}", int(s.split_id.nunique()), n)
    rn = pd.read_csv(os.path.join(ROOT, METRICS[1]))
    for g in ["G1", "G2", "G3"]:
        s = rn[rn.target == g]
        check(f"Table 6 seeds, {g}", sorted(s.seed.unique().tolist()), TEN)
        check(f"Table 6 distinct splits, {g}", int(s.split_id.nunique()), 10)
    for t in ["TeamA-2020", "TeamA-2021", "TeamB-2020"]:
        recs = [json.loads(l) for l in open(os.path.join(ROOT, "output_v2", f"summary_soccermon_window_random_{t}.jsonl")) if l.strip()]
        check(f"random window-level reference seeds, {t}", sorted(r["seed"] for r in recs), TEN)
        check(f"random window-level reference distinct splits, {t}", len({r["split_id"] for r in recs}), 10)
        check(f"random window-level reference methods, {t}", sorted({r["method"] for r in recs}), ["ours"])
        dumps = glob.glob(os.path.join(ROOT, "output_v2", "dumps", "soccermon", "window_random", f"{t}__ours__s*.npz"))
        check(f"random window-level reference dumps, {t}", len(dumps), 10)
    for cohort, expected in [("runners", {"G1": 15, "G2": 15, "G3": 15}), ("soccermon", {"TeamA-2020": 15, "TeamA-2021": 12, "TeamB-2020": 9})]:
        ag = pd.read_csv(os.path.join(ROOT, "output_v2", f"metrics_{cohort}_athlete_grouped5.csv"))
        ag["base"] = ag.target.str.rsplit("_", n=1).str[0]
        ref = ag[ag.method == ag.method.iloc[0]]
        for t, n in expected.items():
            s = ref[ref.base == t]
            check(f"athlete-grouped seeds, {t}", sorted(s.seed.unique().tolist()), THREE)
            check(f"athlete-grouped evaluable fold-by-seed combinations, {t}", int(len(s)), n)
            per_seed = {}
            for seed in THREE:
                folds = {}
                for f in glob.glob(os.path.join(ROOT, "output_v2", "dumps", cohort, "athlete_grouped5", f"{t}_*__*__s{seed}.npz")):
                    info = json.loads(str(np.load(f, allow_pickle=True)["info"]))
                    folds[info["fold"]] = frozenset(info["held_out_athletes"])
                per_seed[seed] = frozenset(folds.items())
            check(f"athlete-grouped evaluable folds with dumps, {t}", sum(len(v) for v in per_seed.values()), n)
            check(f"athlete-grouped distinct fold assignments, {t}", len(set(per_seed.values())), 3)
    variants = ["full", "no_mmd", "no_focal", "no_temporal", "nvg"]
    for v in variants:
        recs = json.load(open(os.path.join(ROOT, "output", f"val_ablation_eventsplit_{v}.json")))
        for t in ["TeamA-2020", "TeamA-2021", "TeamB-2020"]:
            check(f"ablation seeds, {v}, {t}", sorted(r["seed"] for r in recs if r["target"] == t and r.get("status") == "ok"), TEN)
    full = {(r["target"], r["seed"]): r["split"] for r in json.load(open(os.path.join(ROOT, "output", "val_ablation_eventsplit_full.json")))}
    sets = event_assignments(TEN, full)
    for t, n in [("TeamA-2020", 10), ("TeamA-2021", 10), ("TeamB-2020", 3)]:
        check(f"ablation distinct event assignments, {t}", len(set(sets[t].values())), n)
    check("ablation event-assignment frequencies, TeamB-2020", sorted(Counter(sets["TeamB-2020"].values()).values(), reverse=True), [4, 4, 2])
    ps = json.load(open(os.path.join(ROOT, "output", "val_protocol_sensitivity.json")))
    for e in (0, 7, 21):
        ref = {(r["target"], r["seed"]): r["split"] for r in ps if r["protocol"] == f"event_emb{e}"}
        emb = event_assignments(THREE, ref, embargo_days=e)
        for t in ["TeamA-2020", "TeamA-2021", "TeamB-2020"]:
            check(f"embargo {e} days shares the event assignment of Table 7 for each seed, {t}", all(emb[t][s] == sets[t][s] for s in THREE), True)
            check(f"embargo {e} days distinct event assignments over seeds 42, 123, 2026, {t}", len(set(emb[t].values())), 3)
    b20 = ev[(ev.target == "TeamB-2020") & (ev.method == ev.method.iloc[0])]
    check("event-grouped splits of TeamB-2020, seeds per split", sorted(b20.groupby("split_id").seed.nunique().tolist()), [3, 3, 4])
    by = defaultdict(list)
    for r in ps:
        by[(r["protocol"], r["target"])].append(r)
    for (protocol, t), recs in sorted(by.items()):
        check(f"protocol sensitivity seeds, {protocol}, {t}", sorted(r["seed"] for r in recs if r.get("status") == "ok"), THREE)
        if protocol == "chrono_window":
            check(f"chronological split shared by the three seeds, {t}", len({json.dumps(r["split"], sort_keys=True) for r in recs}), 1)
    seeds_by = defaultdict(set)
    for f in expand(RUNNER_FILES[-1]) + RUNNER_FILES[:-1]:
        regime_of_file = "k10" if "k10" in f else "full"
        for r in json.load(open(resolve(f, dirs))):
            seeds_by[(r["target"], r.get("regime", regime_of_file))].add(r["seed"])
    for g in ["G1", "G2", "G3"]:
        for regime in ["full", "k10"]:
            check(f"support-size seeds, {g}, {regime}", sorted(seeds_by[(g, regime)]), TEN)
    t9 = pd.read_csv(os.path.join(ROOT, "output_v2", "si", "table9_v2_per_run.csv"))
    for t, n in [("TeamA-2020", 10), ("TeamA-2021", 10), ("TeamB-2020", 3)]:
        for thr in ["support_rate_matched", "fixed_0.5"]:
            s = t9[(t9.target == t) & (t9.threshold == thr)]
            check(f"weekly analysis seeds, {t}, {thr}", sorted(s.seed.unique().tolist()), TEN)
            check(f"weekly analysis distinct splits, {t}, {thr}", int(s.split_id.nunique()), n)
    g6 = pd.read_csv(os.path.join(ROOT, "output_v2", "si", "table_s6.csv"))
    check("gradient trace seeds", sorted(g6.seed.unique().tolist()), THREE)
    check("gradient trace runs", int(len(g6)), 12)
    runs = [os.path.join(ROOT, "output_v2", "cost_table8", "computational_cost_table.csv")] + \
           [os.path.join(ROOT, "output_v2", "cost_table8", f"rep{i}", "computational_cost_table.csv") for i in (1, 2, 3)]
    cost = pd.concat([pd.read_csv(p).assign(run=i) for i, p in enumerate(runs)])
    cost["train"] = cost.train_ms_per_step.fillna(cost.train_full_ms)
    med = cost.groupby("method").agg(train=("train", "median"), infer=("infer_ms", "median"), n=("run", "nunique"))
    ref = pd.read_csv(os.path.join(ROOT, "output_v2", "cost_table8", "computational_cost_median.csv")).set_index("method")
    check("computational cost runs per method", sorted(set(med.n.tolist())), [4])
    check("computational cost medians reproduced", bool(np.allclose(med.loc[ref.index, "train"], ref.train_ms_median)
                                                       and np.allclose(med.loc[ref.index, "infer"], ref.infer_ms_median)), True)
    sheet = pd.read_csv(os.path.join(ROOT, "data", "literature_audit_coding_sheet.csv"))
    check("literature audit records", int(len(sheet)), 30)
    return rep


def main():
    ap = argparse.ArgumentParser(description="Provenance table of every reported result, with file and count checks.")
    ap.add_argument("--runners-dir", default=None)
    ap.add_argument("--injury-dir", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    dirs = dict(runners=a.runners_dir or default_dir(["..", "..", "*", "runners", "val_runners_10seed_G1.json"], "runners", ["runners", "val_runners_10seed_G1.json"]),
                injury=a.injury_dir or default_dir(["..", "*", "injury", "injury.csv"], "injury", ["..", "data", "soccermon", "injury", "injury.csv"]))
    table = pd.DataFrame([[r["item"], r["cohort"], r["impl"], r["seeds"], r["splits"], r["inputs"],
                           "; ".join(r["results"]) if r["results"] else NA, "; ".join(r["scripts"])] for r in ROWS], columns=HEADER)
    rep = checks(dirs)
    failed = [x for x in rep if not x[3]]
    for name, observed, expected, ok in rep:
        print(("ok      " if ok else "FAILED  ") + f"{name}: {observed}" + ("" if ok else f" (expected {expected})"))
    miss = missing_files(dirs)
    for item, p in miss:
        print(f"missing {item}: {p}")
    print(f"{len(rep) - len(failed)} of {len(rep)} count checks passed; {len(miss)} listed files not found")
    if failed or (a.strict and miss):
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
