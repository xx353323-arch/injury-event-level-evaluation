import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
COST = os.path.join(ROOT, "output_v2", "cost_table8")
ORDER = ["XGBoost", "LSTM", "GAT", "Transformer", "DANN", "ProtoNet", "NVG", "LMVG"]
NAMES = {"XGBoost": "GBT", "NVG": "NVG-TCA (single view)", "LMVG": "LMVG-TCA (this work)"}


def main():
    ap = argparse.ArgumentParser(description="Medians of Table 8 over the four timing runs in output_v2/cost_table8.")
    ap.add_argument("--out", default=os.path.join(COST, "computational_cost_median.csv"))
    a = ap.parse_args()
    runs = [os.path.join(COST, "computational_cost_table.csv")] + [os.path.join(COST, f"rep{i}", "computational_cost_table.csv") for i in (1, 2, 3)]
    frames = [pd.read_csv(p).set_index("method") for p in runs]
    rows = []
    for m in ORDER:
        col = "train_full_ms" if m == "XGBoost" else "train_ms_per_step"
        tr = [float(f.loc[m, col]) for f in frames]
        inf = [float(f.loc[m, "infer_ms"]) for f in frames]
        params = {int(f.loc[m, "params"]) for f in frames}
        if len(params) != 1:
            raise SystemExit(f"{m}: parameter counts differ between runs")
        rows.append(dict(method=m, params=params.pop(), train_ms_median=float(np.median(tr)), train_ms_min=min(tr), train_ms_max=max(tr),
                         infer_ms_median=float(np.median(inf)), infer_ms_min=min(inf), infer_ms_max=max(inf), n_runs=len(frames)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    for r in df.itertuples():
        train = f"{r.train_ms_median:,.0f}" if r.method == "XGBoost" else f"{r.train_ms_median:.2f}"
        print(f"{NAMES.get(r.method, r.method):24s} {r.params:>7,} {train:>8} {r.infer_ms_median:6.2f}")


if __name__ == "__main__":
    main()
