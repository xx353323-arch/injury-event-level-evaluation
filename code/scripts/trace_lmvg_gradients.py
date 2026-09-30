import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("src", "scripts", "pipeline_v2"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import lmvg_v2
import run_v2
from graph_backend_v2 import make_model_v2

FRAMEWORK = run_v2.P
TASKS = ["TeamA-2020", "TeamA-2021", "TeamB-2020", "TeamB-2021"]
GRAPH_PARAMS = ("lmvg.view_logits", "lmvg.tau")
ORIGINAL_STRUCTURE_LOSS = lmvg_v2.LMVGv2.structure_loss
OBJECTIVES = {
    "with_structure_regulariser": ORIGINAL_STRUCTURE_LOSS,
    "without_structure_regulariser": lambda self, target_sparsity=0.25: torch.zeros((), device=self.tau.device),
}


def episode_sizes(n_pos):
    if n_pos >= 10:
        return 4, 16, 4, 16
    if n_pos >= 4:
        return 2, 8, 2, 8
    return 1, 8, max(1, n_pos - 1), 8


def load_windows():
    d = np.load(os.path.join(ROOT, "data", "windows.npz"), allow_pickle=True)
    return d["X"], d["y"], d["tasks"].astype(str), pd.to_datetime(d["dates"]).values


def trace_once(X, y, tasks, dates, target, seed, hard, temperature, batch_size):
    sources = [t for t in TASKS if t != target]
    mu, sd = FRAMEWORK.normalize_fit(X[np.isin(tasks, sources)])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model_v2(backend="lmvg")
    model.set_hard(hard)
    model.set_gumbel_temp(temperature)
    episodes = []
    for t in sources:
        m = tasks == t
        episodes.append(FRAMEWORK.sample_task_temporal(FRAMEWORK.apply_norm(X[m], mu, sd), y[m], dates[m], rng, *episode_sizes(int(y[m].sum()))))
    target_idx = rng.choice(np.where(tasks == target)[0], batch_size, replace=False)
    target_batch = (FRAMEWORK.apply_norm(X[target_idx], mu, sd), y[target_idx])
    grads, _, _ = FRAMEWORK.meta_train_step(model, "lmvg", episodes, target_batch, 5e-3, 2, 0.1)
    named = dict(zip([n for n, _ in model.named_parameters()], grads))
    sizes = {n: p.numel() for n, p in model.named_parameters()}
    rest = [n for n in named if n not in GRAPH_PARAMS and named[n] is not None]
    missing = [n for n in named if n not in GRAPH_PARAMS and named[n] is None]
    g_view = float(named["lmvg.view_logits"].norm())
    g_tau = float(named["lmvg.tau"].norm())
    g_rest = float(torch.sqrt(sum((named[n] ** 2).sum() for n in rest)))
    return dict(
        target=target,
        seed=seed,
        gumbel_sampling="hard" if hard else "soft",
        temperature=temperature,
        n_view_logits=sizes["lmvg.view_logits"],
        n_thresholds=sizes["lmvg.tau"],
        n_remaining=sum(sizes[n] for n in rest),
        n_remaining_without_gradient=sum(sizes[n] for n in missing),
        grad_norm_view_logits=g_view,
        grad_norm_thresholds=g_tau,
        grad_norm_remaining=g_rest,
        ratio_remaining_to_thresholds=g_rest / g_tau,
        ratio_remaining_to_view_logits=g_rest / g_view,
        nonzero_view_logits=bool(g_view > 0),
        nonzero_thresholds=bool(g_tau > 0),
    )


def main():
    ap = argparse.ArgumentParser(description="Gradient norms at the LMVG parameters after one meta-iteration from initialisation.")
    ap.add_argument("--objective", choices=["with_structure_regulariser", "without_structure_regulariser", "both"], default="both")
    ap.add_argument("--sampling", choices=["soft", "hard", "both"], default="both")
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--target", default="TeamA-2020")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--target-batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "output_v2", "si", "table_s6.csv"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    X, y, tasks, dates = load_windows()
    objectives = list(OBJECTIVES) if a.objective == "both" else [a.objective]
    samplings = [False, True] if a.sampling == "both" else [a.sampling == "hard"]
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    for objective in objectives:
        lmvg_v2.LMVGv2.structure_loss = OBJECTIVES[objective]
        try:
            for hard in samplings:
                for seed in seeds:
                    r = trace_once(X, y, tasks, dates, a.target, seed, hard, a.temperature, a.target_batch)
                    r = dict(objective=objective, **r)
                    rows.append(r)
                    print(f"{objective:30s} {r['gumbel_sampling']:4s} seed {seed:5d}  |g| view logits {r['grad_norm_view_logits']:.3e}  thresholds {r['grad_norm_thresholds']:.3e}  remaining {r['grad_norm_remaining']:.3e}  remaining/thresholds {r['ratio_remaining_to_thresholds']:.1f}  remaining/view logits {r['ratio_remaining_to_view_logits']:.1f}", flush=True)
        finally:
            lmvg_v2.LMVGv2.structure_loss = ORIGINAL_STRUCTURE_LOSS
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if k.startswith("grad_norm") else (f"{v:.4f}" if k.startswith("ratio") else v)) for k, v in r.items()})
    for objective in objectives:
        sel = [r for r in rows if r["objective"] == objective]
        rt = [r["ratio_remaining_to_thresholds"] for r in sel]
        rv = [r["ratio_remaining_to_view_logits"] for r in sel]
        nz = all(r["nonzero_view_logits"] and r["nonzero_thresholds"] for r in sel)
        print(f"{objective}: remaining/thresholds {min(rt):.1f} to {max(rt):.1f}; remaining/view logits {min(rv):.1f} to {max(rv):.1f}; non-zero at both graph parameters in every run: {nz}")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
