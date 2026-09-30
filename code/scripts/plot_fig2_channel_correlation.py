import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
d = np.load(os.path.join(ROOT, "data", "windows.npz"), allow_pickle=True)
X = d["X"]
flat = X.reshape(-1, X.shape[-1]).astype(float)
names = ["daily_load", "atl", "ctl28", "acwr", "fatigue", "mood", "readiness", "sleep_duration", "soreness", "stress"]
if "channels" in d.files:
    names = [str(c) for c in d["channels"]]
C = np.corrcoef(flat, rowvar=False)
fig = plt.figure(figsize=(6.72, 3.37))
ax = fig.add_axes([0.255, 0.15, 0.39, 0.70])
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=4.6, color="white" if abs(C[i, j]) > 0.6 else "black")
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5.8)
ax.set_yticklabels(names, fontsize=5.8)
ax.add_patch(Rectangle((-0.5, -0.5), 4, 4, fill=False, lw=1.0, ec="black"))
ax.add_patch(Rectangle((3.5, 3.5), 6, 6, fill=False, lw=1.0, ec="black"))
ax.text(1.5, -0.72, "training load", ha="center", va="bottom", fontsize=6.2, style="italic")
ax.text(6.5, -0.72, "subjective wellness", ha="center", va="bottom", fontsize=6.2, style="italic")
ax.set_title(f"Inter-channel correlation, SoccerMon ({X.shape[0]:,} windows × {X.shape[1]} days)", fontsize=7, pad=19)
cax = fig.add_axes([0.665, 0.15, 0.018, 0.70])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Pearson r", fontsize=6.2)
cb.ax.tick_params(labelsize=5.8)
fig.savefig(os.path.join(OUT, "fig2_channel_correlation.pdf"))
raw = os.path.join(OUT, "fig2_raw.png")
fig.savefig(raw, dpi=600)
img = Image.open(raw).convert("RGB")
W, H = 4034, 2021
s = min(W / img.width, H / img.height)
nw, nh = int(img.width * s), int(img.height * s)
c = Image.new("RGB", (W, H), "white")
c.paste(img.resize((nw, nh), Image.LANCZOS), ((W - nw) // 2, (H - nh) // 2))
c.save(os.path.join(OUT, "fig2_channel_correlation.png"), dpi=(600, 600))
os.remove(raw)
np.savetxt(os.path.join(OUT, "fig2_correlation_matrix.csv"), C, delimiter=",", fmt="%.4f", header=",".join(names), comments="")
print(X.shape, names)
print(np.round(C, 2))
