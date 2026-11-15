"""Fig. 6 (FINAL, English): violation detection rate (%) per rule, mean over the seeds.

Every value is read from outputs/numbers.tex (R@ keys for the printed text, Rr@ keys for the shading); nothing typed by hand.
Writes outputs/fig_map.pdf.
Run: python experiments/fig_map.py
The layout is in points (1 data unit = 1 pt), so spacing does not depend on the figure height.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "outputs" / "numbers.tex").read_text(encoding="utf-8")
RAW = dict(re.findall(r"\\csname Rr@(.+?)\\endcsname\{([^}]*)\}", TEX))
TXT = dict(re.findall(r"\\csname R@(.+?)\\endcsname\{([^}]*)\}", TEX))

RULES = ["payload", "reach_pick", "reach_place", "height", "clearance", "battery"]
RULE_NAMES = [("payload", ""), ("reach", "at pick"), ("reach", "at place"), ("place", "height"),
              ("clearance", ""), ("battery", "")]
GROUPS = [("lookup", 0, 0), ("compare 2 numbers", 1, 3), ("combine numbers", 4, 5)]
# labels "first line|second line": headers bold, rows with a smaller grey second line
BLOCKS = [
    ("SLM alone (Full)", [("qwen2.5-0.5b", "full", "Qwen2.5-0.5B"), ("qwen2.5-1.5b", "full", "Qwen2.5-1.5B"),
                         ("llama-3.2-1b", "full", "Llama-3.2-1B"), ("qwen2.5-3b", "full", "Qwen2.5-3B"),
                         ("llama-3.2-3b", "full", "Llama-3.2-3B"), ("phi-3-mini", "full", "Phi-3-mini"),
                         ("phi-3-mini", "full_4x_data", "Phi-3-mini|4 times the data"),
                         ("qwen2.5-7b", "full", "Qwen2.5-7B (4-bit)")]),
    ("Combinations precomputed by code", [("phi-3-mini", "full_code", "Phi-3-mini|Full + code"),
                                         ("phi-3-mini", "compressed_code", "Phi-3-mini|Compressed + code"),
                                         ("llama-3.2-3b", "compressed_code", "Llama-3.2-3B|Compressed + code")]),
    ("SLM reasons step by step", [("qwen2.5-1.5b", "step_by_step", "Qwen2.5-1.5B"),
                                 ("llama-3.2-3b", "step_by_step", "Llama-3.2-3B"),
                                 ("phi-3-mini", "step_by_step", "Phi-3-mini")]),
]

BLUE = (31 / 255, 96 / 255, 156 / 255)
CMAP = LinearSegmentedColormap.from_list("w2b", [(1, 1, 1), BLUE])
FS, FS2 = 7.5, 6.4                     # main and second-line font sizes (pt)

# geometry in points
WIDTH = 248.0                          # 3.45 in
LABEL_W = 72.0                         # label column
CW = (WIDTH - LABEL_W - 2) / 6         # cell width
X0, XL = 1.0, 6.0                      # header and row label left edges
H1, H2 = 15.5, 20.5                    # row height with one and two label lines
HALF = 4.4                              # half distance between the two lines of a label
HEAD, LH = 15.0, 4.8                  # header height and its half line distance
GAP = 7.0                              # space before a block
TOP = 46.0                             # column headers above the first block

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": FS, "pdf.fonttype": 42})

rows, headers = [], []
y = TOP
for bi, (head, items) in enumerate(BLOCKS):
    y += GAP if bi else 0
    headers.append((head, y + HEAD / 2))
    y += HEAD
    for m, c, text in items:
        h = H2 if "|" in text else H1
        rows.append((y + h / 2, h, m, c, text))
        y += h
height = y + 2.0

fig = plt.figure(figsize=(WIDTH / 72, height / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WIDTH)
ax.set_ylim(height, 0)
ax.axis("off")


def label(x, yc, text, bold=False, half=HALF):
    first, _, second = text.partition("|")
    w = "bold" if bold else "normal"
    if not second:
        ax.text(x, yc, first, ha="left", va="center", fontweight=w)
        return
    ax.text(x, yc - half, first, ha="left", va="center", fontweight=w)
    ax.text(x, yc + half, second, ha="left", va="center", fontweight=w,
            **({} if bold else {"fontsize": FS2, "color": "0.3"}))


x_cells = LABEL_W
for yc, h, m, c, text in rows:
    label(XL, yc, text)
    for j, rule in enumerate(RULES):
        key = f"{m}|{c}|{rule}"
        fc = CMAP(float(RAW[key]) / 100)
        lum = 0.2126 * fc[0] + 0.7152 * fc[1] + 0.0722 * fc[2]
        ax.add_patch(Rectangle((x_cells + j * CW, yc - h / 2), CW, h, facecolor=fc, edgecolor="white", linewidth=0.8))
        ax.text(x_cells + (j + 0.5) * CW, yc, "%.1f" % float(RAW[key]), ha="center", va="center", color="white" if lum < 0.5 else "black")
for head, yc in headers:
    label(X0, yc, head, bold=True, half=LH)

# column headers: rule groups, then the rule names in two lines
for name, a, b in GROUPS:
    xa, xb = x_cells + a * CW + 2, x_cells + (b + 1) * CW - 2
    ax.text((xa + xb) / 2, 9, name, ha="center", va="center", style="italic")
    ax.plot([xa, xb], [15.5, 15.5], color="0.45", linewidth=0.5)
for j, (l1, l2) in enumerate(RULE_NAMES):
    xc = x_cells + (j + 0.5) * CW
    if l2:
        ax.text(xc, 27, l1, ha="center", va="center", fontsize=6.8)
        ax.text(xc, 36, l2, ha="center", va="center", fontsize=6.8)
    else:
        ax.text(xc, 36, l1, ha="center", va="center", fontsize=6.8)

# colour scale in the empty top-left corner
lx, lw, ly = 13.0, 46.0, 33.0
for i in range(60):
    ax.add_patch(Rectangle((lx + i * lw / 60, ly - 3), lw / 60, 6, facecolor=CMAP(i / 59), edgecolor="none"))
ax.add_patch(Rectangle((lx, ly - 3), lw, 6, facecolor="none", edgecolor="0.6", linewidth=0.4))
ax.text(lx, ly - 5, "0%", ha="center", va="bottom", fontsize=6.4)
ax.text(lx + lw, ly - 5, "100%", ha="center", va="bottom", fontsize=6.4)

out = ROOT / "outputs" / "fig_map.pdf"
fig.savefig(out)
print("wrote", out, "height %.0f pt" % height)
