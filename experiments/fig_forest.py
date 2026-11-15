"""Fig. 5 (FINAL, English): difference of balanced accuracy against Full, mean with 95% CI.

Every value is read from outputs/numbers.tex (keys Rr@<model>|<comparison>|delta/lo/hi); nothing typed by hand.
Writes outputs/fig_forest.pdf.
Run: python experiments/fig_forest.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "outputs" / "numbers.tex").read_text(encoding="utf-8")
RAW = dict(re.findall(r"\\csname Rr@(.+?)\\endcsname\{([^}]*)\}", TEX))


def rgb(r, g, b):
    return (r / 255, g / 255, b / 255)


MODELS = [("qwen2.5-0.5b", "Qwen2.5\n0.5B"), ("qwen2.5-1.5b", "Qwen2.5\n1.5B"), ("llama-3.2-1b", "Llama-3.2\n1B"),
          ("qwen2.5-3b", "Qwen2.5\n3B"), ("llama-3.2-3b", "Llama-3.2\n3B"), ("phi-3-mini", "Phi-3\nmini")]
# comparison keys (without the model; the first one present is used), legend name, colour, marker. Phi-3-mini's
# Compressed + code is compared with Full trained again in the same batch of runs (full_rerun).
CONFIGS = [(("compressed - full",), "Compressed", rgb(31, 96, 156), "o"),
           (("progressive - full",), "Progressive", rgb(46, 125, 50), "s"),
           (("compressed_code - full_rerun", "compressed_code - full"), "Compressed + code", rgb(196, 110, 0), "D")]
MARGIN = -2.0   # non-inferiority margin used in the paper
RED = rgb(183, 50, 50)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.45, 2.6))
ax.yaxis.grid(True, linestyle="--", color="0.82", linewidth=0.6)
ax.set_axisbelow(True)
ax.axhline(0, color="black", linewidth=0.7)
ax.axhline(MARGIN, color=RED, linestyle="--", linewidth=0.9)

step = 0.27
for i, (m, _) in enumerate(MODELS):
    present = [(next(x for x in c[0] if f"{m}|{x}|delta" in RAW),) + c[1:] for c in CONFIGS
               if any(f"{m}|{x}|delta" in RAW for x in c[0])]
    for k, (cmp_, _, col, mk) in enumerate(present):
        base = f"{m}|{cmp_}|"
        x = i + (k - (len(present) - 1) / 2) * step
        d, lo, hi = (float(RAW[base + f]) for f in ("delta", "lo", "hi"))
        ax.plot([x, x], [lo, hi], color=col, linewidth=1.3, solid_capstyle="butt", zorder=2)
        for e in (lo, hi):
            ax.plot([x - 0.05, x + 0.05], [e, e], color=col, linewidth=1.0, zorder=2)
        ax.scatter([x], [d], s=20 if mk != "D" else 16, marker=mk, color=col, edgecolors="black", linewidths=0.5, zorder=4)

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([n for _, n in MODELS], fontsize=7.5)
ax.set_xlim(-0.55, len(MODELS) - 0.45)
ax.set_ylim(-16, 20)
ax.set_yticks(range(-15, 21, 5))
ax.set_ylabel("Difference from Full (points)")
ax.tick_params(axis="x", length=0)

handles = [Line2D([], [], color=col, marker=mk, markeredgecolor="black", markeredgewidth=0.5,
                  markersize=4.5 if mk != "D" else 3.8, linewidth=1.3, label=name) for _, name, col, mk in CONFIGS]
handles.append(Line2D([], [], color=RED, linestyle="--", linewidth=0.9, label="non-inferiority margin (-2)"))
ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=True, framealpha=1, edgecolor="0.7",
          handlelength=1.6, borderpad=0.5, labelspacing=0.3, fancybox=False)

fig.tight_layout(pad=0.2)
out = ROOT / "outputs" / "fig_forest.pdf"
fig.savefig(out)
print("wrote", out)
