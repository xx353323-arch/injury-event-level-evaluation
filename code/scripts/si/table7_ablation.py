import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_holm import ROOT, RESULTS, VARIANTS, TARGETS, SEEDS, holm, load

OUT = os.path.join(ROOT, "output_v2", "si", "table7.csv")


def signed(x, digits):
    s = f"{x:+.{digits}f}"
    if float(s) == 0:
        return f"{0:.{digits}f}"
    return s.replace("-", "−")


def test_events(target):
    recs = [r for r in json.load(open(os.path.join(RESULTS, "val_ablation_eventsplit_full.json"))) if r.get("status") == "ok" and r["target"] == target]
    vals = sorted(set(r["split"]["n_te_events"] for r in recs))
    return vals[0] if len(vals) == 1 else f"{vals[0]}–{vals[-1]}"


def main():
    rows = []
    for target in TARGETS:
        label = f"{target} ({test_events(target)} test events)"
        full_auc, full_ap = load("full", "auc"), load("full", "ap")
        keys = sorted(k for k in full_auc if k[0] == target)
        if len(keys) != SEEDS:
            raise SystemExit(f"{target}: expected {SEEDS} seeds, found {len(keys)}")
        fa = np.array([full_auc[k] for k in keys])
        fp = np.array([full_ap[k] for k in keys])
        rows.append(dict(target=label, variant="LMVG-TCA (full)", n=len(keys), auc_mean=fa.mean(), auc_sd=fa.std(ddof=1), ap_mean=fp.mean(), ap_sd=fp.std(ddof=1),
                         delta_auc=np.nan, t_p=np.nan, t_p_holm=np.nan, hedges_g=np.nan))
        block = []
        for variant, name in VARIANTS:
            va, vp = load(variant, "auc"), load(variant, "ap")
            if any(k not in va for k in keys):
                raise SystemExit(f"{target} {variant}: missing paired seeds")
            a = np.array([va[k] for k in keys])
            p = np.array([vp[k] for k in keys])
            d = a - fa
            n = len(d)
            j = 1 - 3 / (4 * n - 5)
            block.append(dict(target=label, variant=name, n=n, auc_mean=a.mean(), auc_sd=a.std(ddof=1), ap_mean=p.mean(), ap_sd=p.std(ddof=1),
                              delta_auc=d.mean(), t_p=float(stats.ttest_1samp(d, 0).pvalue), hedges_g=j * d.mean() / d.std(ddof=1)))
        adj = holm([b["t_p"] for b in block])
        for b, h in zip(block, adj):
            b["t_p_holm"] = float(h)
        rows += block
    df = pd.DataFrame(rows)
    ref = df["delta_auc"].isna()
    df["AUC"] = [f"{m:.3f}±{s:.3f}" for m, s in zip(df.auc_mean, df.auc_sd)]
    df["AP"] = [f"{m:.3f}±{s:.3f}" for m, s in zip(df.ap_mean, df.ap_sd)]
    df["Delta AUC vs full"] = ["reference" if r else signed(x, 3) for r, x in zip(ref, df.delta_auc)]
    df["Paired P"] = ["reference" if r else f"{x:.3f}" for r, x in zip(ref, df.t_p)]
    df["Holm-adjusted P"] = ["reference" if r else f"{x:.3f}" for r, x in zip(ref, df.t_p_holm)]
    df["Hedges g"] = ["reference" if r else signed(x, 2) for r, x in zip(ref, df.hedges_g)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Table 7, ablation under the unpurged event-grouped split, ten seeds, two-sided paired t-tests, Holm adjustment across the four components within each target, alpha = 0.05")
    print(df[["target", "variant", "AUC", "AP", "Delta AUC vs full", "Paired P", "Holm-adjusted P", "Hedges g"]].to_string(index=False))


if __name__ == "__main__":
    main()
