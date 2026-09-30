import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import table9_v2 as weekly_analysis

OUT = os.path.join(weekly_analysis.OUT_DIR, "table_s13.csv")
TARGETS = weekly_analysis.EVALUABLE
THRESHOLD = f"fixed_{weekly_analysis.FIXED}"
MATCHED = "support_rate_matched"
SEEDS = 10


def build(runs):
    tables = {}
    for name in (THRESHOLD, MATCHED):
        sub = runs[(runs.threshold == name) & runs.target.isin(TARGETS)]
        tables[name] = sub.groupby("target").agg(
            n_seeds=("seed", "nunique"),
            n_distinct_splits=("split_id", "nunique"),
            weekly_fpr=("weekly_fpr", "mean"),
            weekly_sensitivity=("weekly_sens", "mean"),
            weekly_ppv=("weekly_ppv", "mean"),
        ).reindex(TARGETS)
    fixed = tables[THRESHOLD]
    if (fixed.n_seeds != SEEDS).any() or (fixed.n_distinct_splits != SEEDS).any():
        raise ValueError("each evaluable target must contribute ten seeds with ten distinct splits")
    matched = tables[MATCHED].add_suffix("_support_rate_matched")
    table = fixed.join(matched[["weekly_fpr_support_rate_matched", "weekly_sensitivity_support_rate_matched",
                                "weekly_ppv_support_rate_matched"]]).reset_index()
    table.insert(1, "threshold", weekly_analysis.FIXED)
    return table


def main():
    runs = weekly_analysis.per_run()
    table = build(runs)
    os.makedirs(weekly_analysis.OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Supplementary Table S13 written to {os.path.relpath(OUT, weekly_analysis.ROOT)} ({len(table)} rows)")
    print("Weekly operating characteristics at a fixed threshold of 0.5, means over ten seeds (ten distinct splits per target)")
    for r in table.itertuples():
        print(f"  {r.target}: weekly FPR {r.weekly_fpr:.3f}, weekly sensitivity {r.weekly_sensitivity:.3f}, "
              f"weekly PPV {r.weekly_ppv:.3f} (support-rate-matched threshold: {r.weekly_fpr_support_rate_matched:.3f}, "
              f"{r.weekly_sensitivity_support_rate_matched:.3f}, {r.weekly_ppv_support_rate_matched:.3f})")


if __name__ == "__main__":
    main()
