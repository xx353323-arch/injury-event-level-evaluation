import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline_v2"))
from events import load_injury_table, reconstruct_events, window_event_id

WINDOWS = os.path.join(ROOT, "data", "windows.npz")
OUT_DIR = os.path.join(ROOT, "output_v2", "si")
OUT = os.path.join(OUT_DIR, "table_s4.csv")
OUT_GAPS = os.path.join(OUT_DIR, "table_s4_gaps.csv")

INTERVALS = [0, 1, 3, 5, 7, 10, 14, 21, 28]
MAIN_INTERVAL = 7
HORIZON = 7
TASKS = ["TeamA-2020", "TeamA-2021", "TeamB-2020", "TeamB-2021"]
GAP_CLASSES = [("1", 1, 1), ("2-7", 2, 7), ("8-13", 8, 13), ("14-28", 14, 28), (">28", 29, None)]


def default_injury_table():
    for cand in (os.path.join(ROOT, "injury", "injury.csv"), os.path.join(ROOT, "..", "data", "soccermon", "injury", "injury.csv")):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    hits = sorted(glob.glob(os.path.join(ROOT, "..", "*", "injury", "injury.csv")))
    if not hits:
        raise FileNotFoundError("injury table not found; pass --injury")
    return os.path.abspath(hits[0])


def team_season(player, start):
    return f"{player.split('-')[0]}-{str(start)[:4]}"


def event_counts(injury, players, dates, labels, tasks):
    positive = np.where(labels == 1)[0]
    rows = []
    for gap in INTERVALS:
        events = reconstruct_events(injury, gap_days=gap)
        by_season = Counter(team_season(p, e["start"]) for p, v in events.items() for e in v)
        if set(by_season) - set(TASKS):
            raise ValueError("event outside the four team-seasons")
        per_event = defaultdict(Counter)
        for i in positive:
            eid = window_event_id(events, players[i], dates[i], HORIZON)
            if eid is None:
                raise ValueError("positive window without an injury event")
            per_event[tasks[i]][eid] += 1
        sizes = np.array([c for t in TASKS for c in per_event[t].values()])
        with_positive = {t: len(per_event[t]) for t in TASKS}
        total = sum(len(v) for v in events.values())
        if sum(by_season.values()) != total:
            raise ValueError("team-season counts do not sum to the total")
        if sizes.sum() != len(positive):
            raise ValueError("positive windows are not all assigned to events")
        row = dict(merge_interval_days=gap, main_protocol=gap == MAIN_INTERVAL, events_total=total)
        for t in TASKS:
            row[f"events_{t}"] = by_season[t]
        for t in TASKS:
            row[f"events_with_positive_window_{t}"] = with_positive[t]
        row.update(
            events_without_positive_window=total - sum(with_positive.values()),
            positive_windows=int(sizes.sum()),
            positive_windows_per_event_min=int(sizes.min()),
            positive_windows_per_event_median=float(np.median(sizes)),
            positive_windows_per_event_max=int(sizes.max()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def gap_distribution(injury):
    unique = injury.drop_duplicates(["player", "date"]).sort_values(["player", "date"])
    gaps = np.concatenate([np.diff(g["date"].values.astype("datetime64[D]")).astype(int) for _, g in unique.groupby("player")])
    rows = []
    for label, low, high in GAP_CLASSES:
        mask = gaps >= low if high is None else (gaps >= low) & (gaps <= high)
        rows.append(dict(gap_days=label, n_gaps=int(mask.sum())))
    table = pd.DataFrame(rows)
    if table["n_gaps"].sum() != len(gaps) or (gaps < 1).any():
        raise ValueError("gap classes do not partition the gaps")
    return table, gaps, len(unique), unique["player"].nunique()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--injury", default=None)
    args = parser.parse_args()
    injury_path = os.path.abspath(args.injury) if args.injury else default_injury_table()
    injury = load_injury_table(injury_path)
    d = np.load(WINDOWS, allow_pickle=True)
    players = d["players"].astype(str)
    dates = pd.to_datetime(d["dates"]).values.astype("datetime64[D]")
    table = event_counts(injury, players, dates, d["y"], d["tasks"].astype(str))
    gaps_table, gaps, n_unique, n_athletes = gap_distribution(injury)
    for r in table.itertuples():
        if r.events_total != n_unique - int((gaps <= r.merge_interval_days).sum()):
            raise ValueError("event totals are inconsistent with the gap distribution")
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(OUT, index=False)
    gaps_table.to_csv(OUT_GAPS, index=False)
    print(f"Supplementary Table S4 written to {os.path.relpath(OUT, ROOT)} and {os.path.relpath(OUT_GAPS, ROOT)}")
    print(f"Injury table: {len(injury)} records, {n_unique} distinct athlete-days, {n_athletes} athletes, {len(gaps)} gaps")
    for _, r in table.iterrows():
        seasons = " ".join(str(int(r[f"events_{t}"])) for t in TASKS)
        print(f"  g = {int(r['merge_interval_days']):2d}  events {int(r['events_total']):3d}  "
              f"by team-season {seasons}  without positive window {int(r['events_without_positive_window'])}  "
              f"positive windows per event {int(r['positive_windows_per_event_min'])}/"
              f"{float(r['positive_windows_per_event_median']):g}/{int(r['positive_windows_per_event_max'])}")
    for _, r in gaps_table.iterrows():
        print(f"  gap {r['gap_days']:>5s} days: {int(r['n_gaps'])}")


if __name__ == "__main__":
    main()
