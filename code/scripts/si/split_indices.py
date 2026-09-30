import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import event_split

OUT_DIR = os.path.join(ROOT, "output", "split_indices")
FOOTBALL = ["TeamA-2020", "TeamA-2021", "TeamB-2020"]
TEN = [0, 1, 2, 3, 4, 5, 6, 42, 123, 2026]
THREE = [42, 123, 2026]
EMBARGO = {"event_emb0": 0, "event_emb7": 7, "event_emb21": 21}
GROUPS = ["G1", "G2", "G3"]
SETTINGS = {"full": None, "ten_event": "10"}
PARTS = ("sup_pos", "sup_neg", "te_pos", "te_neg")


def default_runners_dir():
    if os.path.exists(os.path.join(ROOT, "runners", "val_runners_10seed_G1.json")):
        return os.path.join(ROOT, "runners")
    hits = sorted(glob.glob(os.path.join(ROOT, "..", "..", "*", "runners", "val_runners_10seed_G1.json")))
    if not hits:
        raise FileNotFoundError("running-cohort files not found; pass --runners-dir")
    return os.path.dirname(os.path.abspath(hits[0]))


def recorded(path, key=None):
    out = {}
    for r in json.load(open(path)):
        if r.get("status") != "ok":
            continue
        if key is not None and r.get("protocol") != key:
            continue
        out[(r["target"], r["seed"])] = r["split"]
    return out


def football():
    d = np.load(os.path.join(ROOT, "data", "windows.npz"), allow_pickle=True)
    y, players, tasks = d["y"], d["players"], d["tasks"].astype(str)
    dates = pd.to_datetime(d["dates"]).values
    ablation = recorded(os.path.join(ROOT, "output", "val_ablation_eventsplit_full.json"))
    variants = {v: recorded(os.path.join(ROOT, "output", f"val_ablation_eventsplit_{v}.json")) for v in ("no_mmd", "no_focal", "no_temporal", "nvg")}
    sens = os.path.join(ROOT, "output", "val_protocol_sensitivity.json")
    files = {"unpurged_event_grouped_ablation": {}, "unpurged_event_grouped_embargo": {}, "chronological": {}}
    rows = []
    for target in FOOTBALL:
        m = tasks == target
        g = np.where(m)[0]
        ym, pm, dm = y[m], players[m], dates[m].astype("datetime64[D]")
        for seed in TEN:
            sp, tp, sn, tn, info = event_split.event_level_split(ym, pm, dm, np.random.default_rng(seed))
            ok = info == ablation.get((target, seed)) and all(v.get((target, seed)) == info for v in variants.values())
            for part, idx in zip(PARTS, (sp, sn, tp, tn)):
                files["unpurged_event_grouped_ablation"][f"{target}__s{seed}__{part}"] = g[np.asarray(idx, dtype=int)]
            rows.append(dict(analysis="ablation (Table 7, Figure 4)", split="unpurged event-grouped, 20-day embargo", target=target, seed=seed,
                             n_sup_pos=len(sp), n_sup_neg=len(sn), n_te_pos=len(tp), n_te_neg=len(tn), matches_result_file=bool(ok)))
        for key, emb in EMBARGO.items():
            ref = recorded(sens, key)
            for seed in THREE:
                sp, tp, sn, tn, info = event_split.event_level_split(ym, pm, dm, np.random.default_rng(seed), embargo_days=emb)
                ok = info == ref.get((target, seed))
                for part, idx in zip(PARTS, (sp, sn, tp, tn)):
                    files["unpurged_event_grouped_embargo"][f"{target}__e{emb}__s{seed}__{part}"] = g[np.asarray(idx, dtype=int)]
                rows.append(dict(analysis="embargo length (Supplementary Table S1)", split=f"unpurged event-grouped, {emb}-day embargo", target=target, seed=seed,
                                 n_sup_pos=len(sp), n_sup_neg=len(sn), n_te_pos=len(tp), n_te_neg=len(tn), matches_result_file=bool(ok)))
        ref = recorded(sens, "chrono_window")
        for seed in THREE:
            sp, tp, sn, tn, info = event_split.chronological_window_split(ym, dm, np.random.default_rng(seed))
            ok = info == ref.get((target, seed))
            for part, idx in zip(PARTS, (sp, sn, tp, tn)):
                files["chronological"][f"{target}__s{seed}__{part}"] = g[np.asarray(idx, dtype=int)]
            rows.append(dict(analysis="chronological variant (Supplementary Table S2)", split="chronological window assignment", target=target, seed=seed,
                             n_sup_pos=len(sp), n_sup_neg=len(sn), n_te_pos=len(tp), n_te_neg=len(tn), matches_result_file=bool(ok)))
    return files, rows


def runners(runners_dir):
    sys.path.insert(0, runners_dir)
    from run_runners_probe import split_target
    d = np.load(os.path.join(runners_dir, "windows_runners.npz"), allow_pickle=True)
    y, tasks = d["y"], d["tasks"].astype(str)
    design = pd.read_csv(os.path.join(ROOT, "output_v2", "si", "table_s3_design.csv"))
    files, rows = {}, []
    for group in GROUPS:
        g = np.where(tasks == group)[0]
        yg = y[g]
        for setting, ksup in SETTINGS.items():
            if ksup:
                os.environ["RUNNERS_KSUP"] = ksup
            else:
                os.environ.pop("RUNNERS_KSUP", None)
            for seed in TEN:
                sp, tp, sn, tn = split_target(yg, np.random.default_rng(seed))
                ref = design[(design.target == group) & (design.setting == ("full" if ksup is None else "k10")) & (design.seed == seed)]
                ok = len(ref) == 1 and [int(ref.iloc[0][c]) for c in ("support_pos", "support_neg", "test_pos", "test_neg")] == [len(sp), len(sn), len(tp), len(tn)]
                for part, idx in zip(PARTS, (sp, sn, tp, tn)):
                    files[f"{group}__{setting}__s{seed}__{part}"] = g[np.asarray(idx, dtype=int)]
                rows.append(dict(analysis="support size (Supplementary Table S3)", split=f"window-level split of the running cohort, {setting.replace('_', '-')} support",
                                 target=group, seed=seed, n_sup_pos=len(sp), n_sup_neg=len(sn), n_te_pos=len(tp), n_te_neg=len(tn), matches_result_file=bool(ok)))
        os.environ.pop("RUNNERS_KSUP", None)
    return {"runners_window_level_support_size": files}, rows


def main():
    ap = argparse.ArgumentParser(description="Split indices of the ablation, sensitivity and support-size runs, regenerated from their seeds and checked against the result files.")
    ap.add_argument("--runners-dir", default=None)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    files, rows = football()
    rfiles, rrows = runners(os.path.abspath(a.runners_dir) if a.runners_dir else default_runners_dir())
    files.update(rfiles)
    rows += rrows
    table = pd.DataFrame(rows)
    bad = table[~table.matches_result_file]
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit(f"{len(bad)} regenerated splits do not match the result files")
    for name, arrays in files.items():
        np.savez_compressed(os.path.join(OUT_DIR, f"{name}.npz"), **arrays)
    table.to_csv(os.path.join(OUT_DIR, "split_indices_counts.csv"), index=False)
    print(f"{len(table)} splits regenerated; all match the counts recorded with the runs")
    print("indices refer to the rows of data/windows.npz (football cohort) and runners/windows_runners.npz (running cohort)")
    for name, arrays in files.items():
        print(f"  output/split_indices/{name}.npz: {len(arrays) // 4} splits")


if __name__ == "__main__":
    main()
