"""Figure 1 — cohort and episode formation with exclusion counts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

panels = [
    ("MOVER (perioperative)", [
        "20,274 surgical cases\nwith postoperative pain scores",
        "7,238 cases with an eligible\nindex dose (6,231 patients)",
        "17,797 candidate subsequent doses",
        "17,452 after removing 345 records\n<15 min after a preceding record",
        "9,908 with a valid pre-dose score\nand a reassessment 15–120 min later",
        "9,765 episodes in 3,938 cases\n(3,483 patients) after excluding 143\nwith an intervening opioid",
    ]),
    ("MIMIC-IV (intensive care)", [
        "1,624 ICU stays with an\neligible index dose",
        "4,918 candidate subsequent doses",
        "4,834 after removing 84 records\n<15 min after a preceding record",
        "1,496 with a valid pre-dose score\nand a reassessment 15–120 min later",
        "1,312 episodes in 707 stays (683 patients)\nafter excluding 184 with an intervening opioid",
    ]),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, (title, boxes) in zip(axes, panels):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    n = len(boxes)
    ys = [9.0 - i * (8.4 / n) for i in range(n)]
    for txt, y in zip(boxes, ys):
        ax.add_patch(FancyBboxPatch((0.8, y - 1.0), 8.4, 1.05,
                                    boxstyle="round,pad=0.10,rounding_size=0.12",
                                    linewidth=1.0, edgecolor="#1f3b73", facecolor="#eef1f7"))
        ax.text(5.0, y - 0.47, txt, ha="center", va="center", fontsize=8.2)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((5.0, y1 - 1.02), (5.0, y2),
                                     arrowstyle="-|>", mutation_scale=11, linewidth=0.9, color="#1f3b73"))
fig.tight_layout()
for ext, kw in [("png", dict(dpi=300)), ("pdf", {}), ("tiff", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}))]:
    fig.savefig(OUT / f"Figure_1_Flow.{ext}", **kw)
print("saved", OUT)
