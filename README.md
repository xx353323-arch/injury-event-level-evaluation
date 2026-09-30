# Reproduction package — BZLR-090-1

Manuscript: *Injury-event-level evaluation of machine learning models for athlete injury risk prediction: a benchmark across two cohorts* (Scientific Reports, under revision).

This package contains the code, both public datasets and the result files from which every table and data figure of the manuscript and of its Supplementary Information is regenerated. This version contains no manuscript text, review correspondence or internal notes.

## Layout

```
README.md
LICENSE                      MIT licence of the code
DATA_SOURCES.md              provenance and licences of both datasets
code/
├── pipeline_v2/             partitioning protocol and main pipeline: events.py, splits.py, run_v2.py, analyze_v2.py
├── src/                     model implementations and prepare_data.py (see "Development and legacy code")
├── scripts/                 baselines, drivers of the ablation and sensitivity runs, computational cost,
│   │                        gradient trace (trace_lmvg_gradients.py) and figure scripts
│   └── si/                  scripts for Tables 1, 3, 6 (statistics), 7, 8 and 9, the split indices,
│                            the Section 4.2 diagnostics and Supplementary Tables S1 to S5 and S7 to S13
├── runners/                 window construction, pipeline and result files of the support-size analysis
│                            (running cohort), with a copy of the raw daily CSV
├── data/                    windows.npz and windows_runners.npz (derived arrays), literature_audit_coding_sheet.csv
├── figures/                 source and PDF of Figure 1, renderings and source data of Figures 2 to 4
├── output/                  results of the unpurged event-grouped split (ablation, embargo and chronological
│                            analyses), their split indices (split_indices/) and outputs of earlier analyses
└── output_v2/               results of the main pipeline
    ├── metrics_*_event_v2_random.csv             per-run metrics of Table 6
    ├── metrics_*_athlete_grouped5.csv, pooled_*_athlete_grouped5.csv   athlete-grouped cross-validation
    ├── summary_soccermon_window_random_*.jsonl   random window-level reference of Section 4.1
    ├── dumps/               per-window predicted probabilities and split metadata of every main-pipeline run
    ├── cost_table8/, cost_table8_run.log         the four timing runs of Table 8 and their medians
    └── si/                  CSV outputs of the scripts in scripts/si/ and of the gradient trace
data/
├── soccermon/               football cohort, raw CSVs (CC BY 4.0)
└── runners/                 running cohort, raw CSVs (CC0 1.0)
```

Paths given in Supplementary Table S10 are relative to `code/`; the SoccerMon injury table `injury/injury.csv` is part of the SoccerMon release in `data/soccermon/`.

## Names used in the code

| Manuscript | Code |
|---|---|
| LMVG-TCA, the evaluated framework | method key `ours` in `run_v2.py` and its result files; `LMVG` in the timing files of Table 8; key `paml` in the result records of the earlier scripts (`src/paml_full*.py`, `scripts/run_main_eventsplit.py`, `output/exp1_main_eventsplit_results.json`) |
| GBT, the gradient-boosting baseline | method key `XGBoost`; it is implemented with the `GradientBoostingClassifier` of scikit-learn and does not use the XGBoost library |
| NVG-TCA, the single-view variant | `NVG`, `nvg` |
| AthleteRate, the athlete-identity baseline | `AthleteRate` |
| F1 at the support-rate-matched threshold; test-optimal F1 | `f1_calibrated`; `f1_oracle` |
| injury-event-grouped protocol of Table 6 | protocol `event_v2_random` |
| athlete-grouped five-fold cross-validation | protocol `athlete_grouped5` |
| random window-level reference (Section 4.1) | protocol `window_random` |
| unpurged event-grouped split (Table 7, Supplementary Tables S1 and S2) | `scripts/run_ablation_eventsplit.py`, `scripts/run_protocol_sensitivity.py`, `scripts/event_split.py` |
| window-level split of the running cohort (Supplementary Table S3) | `runners/run_runners_probe.py`, `runners/run_runners_probe_tp.py`, `runners/run_runners_10seed.py` |

Source files whose names begin with paml (`src/paml.py` and `src/paml_*.py`) implement the LMVG-TCA framework and its earlier prototypes; the prefix is retained from an earlier designation to keep the commit history intact.

## Development and legacy code

The pipeline of the manuscript (`pipeline_v2/run_v2.py`) imports `src/paml_full_lmvg_v2.py`, `src/graph_backend_v2.py`, `src/lmvg_v2.py` and `src/tca_gnn.py`, and from `paml_full_lmvg_v2.py` it uses only the training utilities: normalisation, episode sampling, the meta-update with its focal loss and discrepancy term, and fine-tuning. The functions `run_one` and `main` of that file, and the modules `src/paml.py`, `src/paml_full.py`, `src/paml_full_lmvg.py`, `src/paml_full_lmvg_tuned.py`, `src/graph_backend.py`, `src/graph_backend_tuned.py`, `src/lmvg.py`, `src/lmvg_tuned.py` and `src/train_v2.py`, are development code written before the evaluation protocol of the manuscript; no command in this README calls them.

The function `main` of `paml_full_lmvg_v2.py` is the development comparison in which the outer-loop learning rates used in all reported runs (2 × 10⁻³ for the encoder and classification head, 0.5 for the graph parameters) were chosen. It compares the framework at these rates with its initial settings and with the single-view variant by test-partition AUC under an earlier random window-level split of the three football targets, and its training loop stops a run when the test AUC stays below a reference value. The pipeline of the manuscript selects models on a support-validation subset and never reads test labels during training. The limitations in Section 5 of the manuscript describe this development choice.

The hyperparameters of Table 5 are set in `pipeline_v2/run_v2.py` (inner and outer learning rates, inner steps, episode budget, patience and fine-tuning steps) and in `src/paml_full_lmvg_v2.py` (focal-loss parameters, discrepancy kernel widths, structure-regulariser weight, Gumbel schedule and gradient clipping).

## Quick start

```bash
cd code/pipeline_v2

# football cohort, injury-event-grouped protocol, one target, one method
python3 run_v2.py --cohort soccermon --protocol event_v2_random \
                  --targets TeamA-2021 --methods XGBoost --seeds 42
```

`run_v2.py` writes to `code/output_v2/` and skips every run whose per-window file already exists in `code/output_v2/dumps/`, so in the shipped package this command returns at once and the shipped results are not overwritten. To repeat a run, work on a copy of the package and remove the corresponding file from `dumps/` first.

Key arguments of `run_v2.py`: `--cohort` (`soccermon` or `runners`), `--protocol` (`event_v2_random`, `athlete_grouped5` or `window_random`), `--targets` (target names, or `<target>:<fold>` with folds 0 to 4 for `athlete_grouped5`), `--methods` (default: ours, LSTM, XGBoost, GAT, Transformer, ProtoNet, DANN, AthleteRate), `--seeds` (default: 42, 123, 2026, 0 to 6), `--episodes` (default 120), `--embargo` (default 20), `--threads` (default 2).

The main-pipeline runs reported in the manuscript are:

```bash
cd code/pipeline_v2
python3 run_v2.py --cohort soccermon --protocol event_v2_random                          # Table 6, football cohort
python3 run_v2.py --cohort runners --protocol event_v2_random                            # Table 6, running cohort
python3 run_v2.py --cohort soccermon --protocol athlete_grouped5 --seeds 42,123,2026     # Supplementary Table S8
python3 run_v2.py --cohort runners --protocol athlete_grouped5 --seeds 42,123,2026       # Supplementary Table S8
python3 run_v2.py --cohort soccermon --protocol window_random --methods ours             # random window-level reference
python3 analyze_v2.py --cohort soccermon --protocol event_v2_random                      # per-run metrics, Table 6
python3 analyze_v2.py --cohort runners --protocol event_v2_random
python3 analyze_v2.py --cohort soccermon --protocol athlete_grouped5                     # per-run and pooled metrics, Supplementary Table S8
python3 analyze_v2.py --cohort runners --protocol athlete_grouped5
```

Rebuilding the derived arrays from the raw CSVs is optional. From the package root, `python3 code/src/prepare_data.py` writes `code/src/windows.npz`, and `python3 code/runners/build_windows_runners.py` writes `code/runners/windows_runners.npz` from the copy of the daily CSV in `code/runners/`. The support-size scripts read `code/runners/windows_runners.npz`; the main pipeline reads the copies in `code/data/`, so copy a rebuilt array there to use it. All shipped copies of each array are identical.

## Regenerating the tables and figures

Run from `code/`. Most scripts write a CSV to `output_v2/si/`, `output/split_indices/` or `figures/`; `stats_robust.py`, `stats_exact.py` and `ablation_holm.py` print their results.

| Item | Command |
|---|---|
| Tables 1 and 3 | `python3 scripts/si/table1_table3_counts.py` (recomputes every split with the partitioning code and checks it against the split identifiers in `output_v2/dumps/`) |
| Table 6 | `python3 pipeline_v2/analyze_v2.py --cohort soccermon --protocol event_v2_random` and the same for `runners`; `python3 scripts/si/stats_robust.py`; `python3 scripts/si/stats_exact.py` |
| Table 7 | `python3 scripts/si/table7_ablation.py`; the Holm-adjusted values are also printed by `python3 scripts/si/ablation_holm.py` |
| Table 8 | `PYTHONPATH=src COST_OUT=output_v2/cost_table8 python3 scripts/compute_cost_table.py`, repeated with `COST_OUT=output_v2/cost_table8/rep1` to `rep3`, then `python3 scripts/si/table8_cost_median.py`; timings depend on the hardware |
| Table 9 | `python3 scripts/si/table9_v2.py` |
| Figure 1 | `figures/fig1_framework.tex`, a schematic compiled with XeLaTeX; it uses the Helvetica and Arial fonts |
| Figure 2 | `python3 scripts/plot_fig2_channel_correlation.py . figures` |
| Figure 3 | `python3 scripts/plot_fig3_main_comparison.py . figures` |
| Figure 4 | `python3 scripts/plot_fig4_ablation.py . figures` |
| Supplementary Tables S1 to S5 | `python3 scripts/si/table_s1_embargo.py`, `table_s2_chronological.py`, `table_s3_support_size.py`, `table_s4_merge_interval.py`, `table_s5_literature_audit.py` |
| Supplementary Table S6 | `python3 scripts/trace_lmvg_gradients.py` |
| Supplementary Tables S7 to S9 | `python3 scripts/si/table_s7_runners_paired_contrasts.py`, `table_s8_athlete_grouped.py`, `table_s9_brier_f1_detection.py` |
| Supplementary Table S10 | `python3 scripts/si/table_s10_provenance.py --strict` (checks that every listed file is present and recomputes the seeds and the numbers of distinct splits from the result files) |
| Supplementary Tables S11 to S13 | `python3 scripts/si/table_s11_teamb2020.py`, `table_s12_statistics.py`, `table_s13_fixed_threshold.py` |
| Split indices of the ablation, sensitivity and support-size runs | `python3 scripts/si/split_indices.py` (regenerates every split from its seed, checks it against the counts recorded with the run and writes the indices to `output/split_indices/`) |
| Overlap and athlete-identity baseline under the unpurged split (Section 4.2) | `python3 scripts/si/unpurged_split_diagnostics.py` |

The indices in `output/split_indices/*.npz` refer to the rows of `data/windows.npz` (football cohort) and `runners/windows_runners.npz` (running cohort); each key has the form `<target>__[e<embargo>__|<setting>__]s<seed>__<part>`, where the part is `sup_pos`, `sup_neg`, `te_pos` or `te_neg`, and `split_indices_counts.csv` lists every split with its counts.

Each per-window file in `output_v2/dumps/` holds the indices, athletes, dates, event identifiers, labels and predicted probabilities of the test windows, the indices, labels, athletes and dates of the support windows, and an `info` record. For `event_v2_random` the record gives the split identifier and the counts of events, positive and negative windows and purged windows; for `athlete_grouped5` it gives the fold, the held-out athletes and the window counts; for `window_random` it gives the split identifier, the window counts and the share of positive test windows that belong to an injury event represented in the support partition.

## Outputs of earlier analyses

`code/output/` also holds results of earlier analyses that the manuscript does not report, kept with the scripts that produced them: `exp1_main_eventsplit_results.json` (the framework under the unpurged split with 200 meta-training episodes, `scripts/run_main_eventsplit.py`), `val_baselines_eventsplit_*.json` (the comparison methods under the unpurged split, `scripts/run_baselines_eventsplit.py`), `val_proto_feasibility.json` (a prototype-classifier variant on two splits, `scripts/proto_feasibility.py`), `val_weekly_fpr.json` (weekly operating characteristics under the unpurged split with three seeds at a fixed threshold of 0.5, `scripts/run_weekly_fpr.py`, superseded by Table 9), `val_athlete_rate_baseline.json` (the athlete-identity baseline under the unpurged split and the support-size settings; the football values are regenerated by `scripts/si/unpurged_split_diagnostics.py`), and `val_data_access_table.md` and `val_v2_split_counts.txt` (descriptive tables of data flow and split counts, superseded by Tables 1, 3 and 4 and by `scripts/si/table1_table3_counts.py`).

## Protocol v2 in brief

Event reconstruction uses three independent symbols: `g`, the merge gap for injury reports (default 7 days); `h`, the label horizon (7 days for football, 1 day for runners); and `e`, the embargo (default 20 days). Splitting fixes the test side first, then applies a one-directional purge of support-side positive windows overlapping the observation interval of any test event. Negative windows are drawn as whole blocks of consecutive dates within an athlete, with a ±13-day buffer on the test side. Every split emits purge counts, residual overlap rates and cross-side assertions.

The athlete-grouped protocol (`athlete_grouped5`) divides the athletes of a target into five folds, separately for injured and uninjured athletes, and removes the held-out athletes from the source side as well. Predictions of the evaluable folds of a seed are pooled before the pooled metrics are computed.

The window-level random protocol (`window_random`) assigns windows to the support and test partitions at random, with the model, training settings and seeds of the event-grouped protocol.

## Metrics

Besides overall AUC and average precision, the package computes within-athlete AUC (positive versus negative windows compared only inside the same athlete) and between-athlete AUC (each athlete aggregated to one point). Threshold-dependent metrics use the support-rate-matched threshold, set on the unlabelled test score distribution without reading any test label; the test-optimal threshold is reported separately as an optimistic upper bound.

## Licence

The code is released under the MIT License (`LICENSE`). The bundled datasets keep their original licences, CC BY 4.0 for SoccerMon and CC0 1.0 for the running cohort (`DATA_SOURCES.md`).

## Environment

Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, scikit-learn 1.6.1, pandas, SciPy, matplotlib and Pillow. CPU only; no GPU required.

## Note on file paths

Data paths in six scripts were rewritten for this package so that everything resolves inside the package directory (`run_v2.py`, `event_split.py`, `prepare_data.py`, `run_runners_probe.py`, `run_runners_probe_tp.py`, `run_runners_10seed.py`). No analysis logic was changed. Every command listed under "Regenerating the tables and figures", except the timing runs of Table 8 and the compilation of Figure 1, was run inside this package and reproduced the shipped results; the rows of the CSV files written by `analyze_v2.py` follow the order in which the file system lists the per-window files.
