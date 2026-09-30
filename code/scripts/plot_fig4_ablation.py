import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
V = [("full", "LMVG-TCA (full)"), ("no_mmd", "w/o MMD alignment"), ("no_focal", "w/o focal loss"),
     ("no_temporal", "w/o temporal sampling"), ("nvg", "w/o multi-view graph (single view)")]


def load(v):
    path = os.path.join(ROOT, "output", f"val_ablation_eventsplit_{v}.json")
    return {(r["target"], r["seed"]): r["result"] for r in json.load(open(path)) if r.get("status") == "ok"}


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


data = {v: load(v) for v, _ in V}
targets = [("TeamA-2020", "TeamA-2020 (19 test events, 10 seeds)"), ("TeamA-2021", "TeamA-2021 (6 test events, 10 seeds)")]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.6), sharey=True)
rows, tests = [], []
for r_, metric in enumerate(["auc", "ap"]):
    for c_, (tgt, title) in enumerate(targets):
        ax = axes[r_, c_]
        full = {k: v[metric] for k, v in data["full"].items() if k[0] == tgt}
        seeds = sorted(s for (_, s) in full)
        ypos = np.arange(len(V))[::-1]
        pv = []
        for i, (v, lab) in enumerate(V):
            vals = np.array([data[v][(tgt, s)][metric] for s in seeds])
            mu, sd = vals.mean(), vals.std(ddof=1)
            y = ypos[i]
            jit = (np.arange(len(vals)) - (len(vals) - 1) / 2) / max(len(vals) - 1, 1) * 0.36
            ax.scatter(vals, y + jit, s=6, color="#a8a8a8", linewidth=0, zorder=2)
            ax.errorbar(mu, y, xerr=sd, fmt="D" if v == "full" else "o", color="black", markersize=4.0, capsize=2.2, elinewidth=0.9, zorder=4)
            rows.append(dict(metric=metric, target=tgt, variant=lab, n=len(vals), mean=round(mu, 3), sd=round(sd, 3),
                             lower=round(mu - sd, 3), upper=round(mu + sd, 3)))
            if v != "full" and metric == "auc":
                diff = vals - np.array([full[(tgt, s)] for s in seeds])
                pv.append((lab, diff.mean(), stats.ttest_1samp(diff, 0).pvalue))
        fm = np.mean(list(full.values()))
        ax.axvline(fm, color="black", linestyle="--", linewidth=0.9, zorder=1)
        if metric == "auc":
            adj = holm([p for _, _, p in pv])
            for j, ((lab, dmean, p), pa) in enumerate(zip(pv, adj)):
                tests.append(dict(target=tgt, variant=lab, delta_auc=round(dmean, 4), p_unadjusted=round(p, 4), p_holm=round(pa, 4)))
                if pa < 0.05 and dmean != 0:
                    y = ypos[j + 1]
                    vals = np.array([data[V[j + 1][0]][(tgt, s)][metric] for s in seeds])
                    ax.text(vals.max() + 0.015, y, "*", ha="left", va="center", fontsize=11)
            ax.set_xlim(0.3, 1.0)
            ax.set_xlabel("AUC")
        else:
            lo = min(0.0, min(r["lower"] for r in rows if r["metric"] == metric and r["target"] == tgt)) - 0.01
            ax.set_xlim(lo, 0.37)
            ax.axvline(0, color="#999999", linewidth=0.5, zorder=0)
            ax.set_xlabel("Average precision")
        ax.set_yticks(ypos)
        ax.set_yticklabels([l for _, l in V], fontsize=7.5)
        if r_ == 0:
            ax.set_title(title, fontsize=8)
        ax.text(-0.02 if c_ else -0.62, 1.04, "abcd"[r_ * 2 + c_], transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
h = [Line2D([], [], marker="D", color="black", linestyle="none", markersize=4, label="Full model, mean ± s.d."),
     Line2D([], [], marker="o", color="black", linestyle="none", markersize=4, label="Ablated variant, mean ± s.d."),
     Line2D([], [], marker="o", color="#a8a8a8", linestyle="none", markersize=3, label="Individual seeds"),
     Line2D([], [], color="black", linestyle="--", linewidth=0.9, label="Full-model mean")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=1.0)
fig.savefig(os.path.join(OUT, "fig4_ablation.pdf"))
raw = os.path.join(OUT, "fig4_raw.png")
fig.savefig(raw, dpi=600)
im = Image.open(raw).convert("RGB")
W, H = 4034, 2021
s = min(W / im.width, H / im.height)
nw, nh = int(im.width * s), int(im.height * s)
c = Image.new("RGB", (W, H), "white")
c.paste(im.resize((nw, nh), Image.LANCZOS), ((W - nw) // 2, (H - nh) // 2))
c.save(os.path.join(OUT, "fig4_ablation.png"), dpi=(600, 600))
os.remove(raw)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig4_source_data.csv"), index=False)
pd.DataFrame(tests).to_csv(os.path.join(OUT, "fig4_ablation_tests.csv"), index=False)
print(pd.DataFrame(tests).to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
