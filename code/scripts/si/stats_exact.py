import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stats_robust as sr


def fmt(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def main():
    families = sr.runner_families()
    floor = families[("Runners", "floor")]
    lstm = families[("Runners", "lstm")]
    tost = families[("Runners", "tost")].set_index("method")
    print("Running cohort pooled over thirty group-by-seed splits: two-sided paired t-tests against the athlete-identity floor, "
          "exact P and Holm-adjusted P (family of seven, alpha = 0.05)")
    for r in floor.itertuples():
        print(f"  {r.method:12s} difference {r.mean_difference:+.3f}  t P={fmt(r.t_p)} Holm={fmt(r.t_p_holm)} | "
              f"Wilcoxon P={fmt(r.wilcoxon_p)} Holm={fmt(r.wilcoxon_p_holm)}")
    print("Running cohort pooled over thirty group-by-seed splits: paired t-tests against LSTM and two one-sided tests "
          f"at a margin of {sr.MARGIN} AUC, exact P and Holm-adjusted P (family of six, alpha = 0.05)")
    for r in lstm.itertuples():
        e = tost.loc[r.method]
        print(f"  {r.method:12s} difference {r.mean_difference:+.3f}  t P={fmt(r.t_p)} Holm={fmt(r.t_p_holm)} | "
              f"TOST P={fmt(e.t_p)} Holm={fmt(e.t_p_holm)} | Wilcoxon P={fmt(r.wilcoxon_p)} Holm={fmt(r.wilcoxon_p_holm)}")


if __name__ == "__main__":
    main()
