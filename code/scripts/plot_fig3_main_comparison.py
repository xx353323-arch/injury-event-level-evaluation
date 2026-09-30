import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 8.5, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
fb = pd.read_csv(os.path.join(ROOT, "output_v2", "metrics_soccermon_event_v2_random.csv"))
fr = pd.read_csv(os.path.join(ROOT, "output_v2", "metrics_runners_event_v2_random.csv"))
fb = fb[fb.target != "TeamB-2020"]
order = ["XGBoost", "LSTM", "GAT", "Transformer", "ProtoNet", "DANN", "AthleteRate", "ours"]
label = {"XGBoost": "GBT", "LSTM": "LSTM", "GAT": "GAT", "Transformer": "Transformer", "ProtoNet": "ProtoNet",
         "DANN": "DANN", "AthleteRate": "AthleteRate", "ours": "LMVG-TCA"}
panels = [("TeamA-2020 (19 test events, 10 splits)", fb[fb.target == "TeamA-2020"], False),
          ("TeamA-2021 (6 test events, 10 splits)", fb[fb.target == "TeamA-2021"], False),
          ("Competitive runners, G1 to G3 (30 splits)", fr, True)]
gmark = {"G1": "o", "G2": "^", "G3": "D"}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
rows = []
for k, (ax, (name, d, runners)) in enumerate(zip(axes, panels)):
    fl = d[d.method == "AthleteRate"].auc.mean()
    ax.hlines(fl, -0.6, len(order) - 0.25, color="black", linestyle="--", linewidth=0.9, zorder=1)
    ax.hlines(0.5, -0.6, len(order) - 0.25, color="black", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(len(order) - 0.15, fl, f"{fl:.3f}", fontsize=7, ha="left", va="center")
    for i, m in enumerate(order):
        x = d[d.method == m]
        n = len(x)
        mu, sd, wmu = x.auc.mean(), x.auc.std(ddof=1), x.auc_within_athlete.mean()
        jit = (np.arange(n) - (n - 1) / 2) / max(n - 1, 1) * 0.28
        if runners:
            for g, mk in gmark.items():
                sel = (x.target == g).values
                ax.scatter(i - 0.12 + jit[sel], x.auc.values[sel], s=6, marker=mk, color="#a8a8a8", linewidth=0, zorder=2)
        else:
            ax.scatter(i - 0.12 + jit, x.auc.values, s=6, color="#a8a8a8", linewidth=0, zorder=2)
        ax.errorbar(i - 0.12, mu, yerr=sd, fmt="o", color="black", markersize=4.2, capsize=2.3, elinewidth=0.9, zorder=4)
        ax.scatter(i + 0.22, wmu, marker="s", s=20, facecolor="white", edgecolor="black", linewidth=0.9, zorder=4)
        rows.append(dict(panel=name, method=label[m], n=n, auc_mean=round(mu, 3), auc_sd=round(sd, 3),
                         within_mean=round(wmu, 3), floor=round(fl, 3)))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label[m] for m in order], rotation=45, ha="right")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(-0.6, len(order) + 0.45)
    ax.set_ylim(0.3, 1.0)
    ax.set_title(name, fontsize=7.6, pad=4)
    ax.text(-0.16 if k == 0 else -0.12, 1.035, "abc"[k], transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("AUC")
h = [Line2D([], [], marker="o", color="black", linestyle="none", markersize=4.2, label="Overall AUC, mean ± s.d."),
     Line2D([], [], marker="o", color="#a8a8a8", linestyle="none", markersize=3, label="Individual splits (c: G1)"),
     Line2D([], [], marker="^", color="#a8a8a8", linestyle="none", markersize=3.3, label="G2"),
     Line2D([], [], marker="D", color="#a8a8a8", linestyle="none", markersize=2.8, label="G3"),
     Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor="black", linestyle="none", markersize=5, label="Within-athlete AUC, mean"),
     Line2D([], [], color="black", linestyle="--", linewidth=0.9, label="Athlete-identity floor"),
     Line2D([], [], color="black", linestyle=":", linewidth=0.8, label="Chance")]
fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=6.8, columnspacing=1.0, handletextpad=0.3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "fig3_main_comparison.pdf"))
raw = os.path.join(OUT, "fig3_raw.png")
fig.savefig(raw, dpi=600)
im = Image.open(raw).convert("RGB")
W, H = 4034, 2021
s = min(W / im.width, H / im.height)
nw, nh = int(im.width * s), int(im.height * s)
c = Image.new("RGB", (W, H), "white")
c.paste(im.resize((nw, nh), Image.LANCZOS), ((W - nw) // 2, (H - nh) // 2))
c.save(os.path.join(OUT, "fig3_main_comparison.png"), dpi=(600, 600))
os.remove(raw)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig3_source_data.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
