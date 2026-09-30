import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DUMPS = os.path.join(ROOT, "output_v2", "dumps", "soccermon", "event_v2_random")
METRICS = os.path.join(ROOT, "output_v2", "metrics_soccermon_event_v2_random.csv")
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
PER_RUN = os.path.join(OUT_DIR, "table9_v2_per_run.csv")
SUMMARY = os.path.join(OUT_DIR, "table9.csv")
FRAMEWORK = "ours"
SQUAD = 25
FIXED = 0.5
EVALUABLE = ["TeamA-2020", "TeamA-2021"]
COLUMNS = [
    "window_auc", "daily_fpr", "daily_sens", "weekly_fpr", "weekly_sens", "weekly_ppv",
    "healthy_weeks", "injury_weeks", "alerts_per25", "naive_per25", "brier_skill", "thr_value",
    "mean_days_per_week",
]


def weekly(prob, y, player, date, thr):
    df = pd.DataFrame(dict(prob=prob, y=y, player=player, date=pd.to_datetime(date)))
    iso = df.date.dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    weeks = df.groupby(["player", "week"]).agg(score=("prob", "max"), inj=("y", "max"), n=("y", "size")).reset_index()
    flagged = weeks.score >= thr
    healthy = weeks.inj == 0
    injured = weeks.inj == 1
    daily_fpr = ((df.prob >= thr) & (df.y == 0)).sum() / max((df.y == 0).sum(), 1)
    weekly_fpr = (flagged & healthy).sum() / max(healthy.sum(), 1)
    return dict(
        daily_fpr=daily_fpr,
        daily_sens=((df.prob >= thr) & (df.y == 1)).sum() / max((df.y == 1).sum(), 1),
        weekly_fpr=weekly_fpr,
        weekly_sens=(flagged & injured).sum() / max(injured.sum(), 1),
        weekly_ppv=(flagged & injured).sum() / max(flagged.sum(), 1),
        healthy_weeks=int(healthy.sum()),
        injury_weeks=int(injured.sum()),
        alerts_per25=SQUAD * weekly_fpr,
        naive_per25=SQUAD * daily_fpr,
        mean_days_per_week=float(weeks.n.mean()),
    )


def per_run():
    files = sorted(glob.glob(os.path.join(DUMPS, f"*__{FRAMEWORK}__*.npz")))
    if not files:
        raise FileNotFoundError(f"no per-window predictions found in {DUMPS}")
    rows = []
    for path in files:
        target, _, seed = os.path.basename(path)[:-4].split("__")
        d = np.load(path, allow_pickle=True)
        y = d["te_y"].astype(int)
        p = d["te_prob"].astype(float)
        support_rate = float(d["sup_y"].mean())
        matched = float(np.quantile(p, 1 - support_rate))
        for name, thr in (("support_rate_matched", matched), (f"fixed_{FIXED}", FIXED)):
            r = weekly(p, y, d["te_player"], d["te_date"], thr)
            r.update(target=target, seed=seed[1:], threshold=name, thr_value=thr, window_auc=roc_auc_score(y, p))
            rows.append(r)
    df = pd.DataFrame(rows)
    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics.method == FRAMEWORK][["target", "seed", "split_id", "brier_skill", "auc"]]
    metrics["seed"] = metrics["seed"].astype(str)
    df = df.merge(metrics, on=["target", "seed"], how="left")
    if df["split_id"].isna().any():
        raise ValueError("a run has no matching row in the metrics file")
    return df


def summarise(df):
    means = df.groupby(["threshold", "target"])[COLUMNS].mean()
    counts = df.groupby(["threshold", "target"]).agg(runs=("seed", "nunique"), distinct_splits=("split_id", "nunique"))
    return means.join(counts).reset_index()


def table9_rows(summary):
    view = summary[summary.threshold == "support_rate_matched"].set_index("target")
    lines = []
    for target in EVALUABLE:
        r = view.loc[target]
        lines.append(
            f"{target} | {r.window_auc:.3f} | {r.daily_fpr:.3f} | {r.daily_sens:.3f} | {r.weekly_fpr:.3f} | "
            f"{r.weekly_sens:.3f} | {r.weekly_ppv:.3f} | {r.healthy_weeks:.0f} / {r.injury_weeks:.0f} | "
            f"{r.alerts_per25:.1f} / {r.naive_per25:.1f} | {r.brier_skill:.2f}"
        )
    return lines


def main():
    df = per_run()
    summary = summarise(df)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(PER_RUN, index=False)
    summary.to_csv(SUMMARY, index=False)
    print(f"Per-run weekly operating characteristics written to {os.path.relpath(PER_RUN, ROOT)} ({len(df)} rows)")
    print(f"Summary by threshold and target written to {os.path.relpath(SUMMARY, ROOT)}")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print()
    print("Table 9, support-rate-matched threshold, means over ten seeds")
    print("Target | Window AUC | Daily FPR | Daily sensitivity | Weekly FPR | Weekly sensitivity | Weekly PPV | "
          "Healthy / injury athlete-weeks | Weekly false alerts per 25 athletes (measured / daily extrapolation) | Brier skill")
    for line in table9_rows(summary):
        print(line)
    print()
    teamb = df[df.target == "TeamB-2020"].groupby(["threshold", "split_id"])[["weekly_fpr", "weekly_sens", "weekly_ppv"]].mean()
    print("TeamB-2020 by distinct split (descriptive)")
    print(teamb.round(3).to_string())
    print()
    print("Maximum absolute difference between window AUC recomputed from the dumps and the metrics file:",
          float((df.window_auc - df.auc).abs().max()))


if __name__ == "__main__":
    main()
