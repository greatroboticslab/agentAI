"""Journal-figure conventions, and the poster's own palette.

Matched to PPTFinal112.pptx: Arial throughout, the slate-blue system, 48 x 36 in.

What separates a figure that reads as scientific from one that reads as generated:
  - the figure carries no title. The claim lives in the caption, set in the poster.
  - hairline spines, ticks pointing out, no box. Grid only where a reader must
    compare across a long distance, and then at the lightest weight that works.
  - direct labels on the marks instead of a legend, wherever the marks are few.
  - one accent colour against greys; categorical colour only when the categories
    are the subject.
  - panel letters lower-case bold outside the axes, which is what a caption
    references.
  - error bars, n, and the test wherever a difference is claimed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# straight from the reference deck
INK     = "#1E293B"
NAVY    = "#14314E"
BLUE    = "#1C6FB5"
PALE    = "#F4F7FA"
PALEBLU = "#EAF2F9"
RULE    = "#C2CEDA"
MUTE    = "#556575"
WHITE   = "#FFFFFF"
# two more, chosen to sit in the same family rather than fight it
GOOD    = "#1B7A55"
WARN    = "#B5502A"
GREY    = "#8D99A6"

SERIES = [BLUE, INK, GOOD, WARN, MUTE, GREY]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "normal",
    "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "legend.frameon": False, "legend.fontsize": 10.5,
    "grid.color": RULE, "grid.linewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "figure.facecolor": WHITE, "savefig.facecolor": WHITE,
    "lines.linewidth": 1.4, "lines.markersize": 5,
    "errorbar.capsize": 3,
})


def panel(ax, letter, dx=-0.085, dy=1.045):
    """Lower-case bold panel letter, outside the axes, where a caption points."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left", color=INK)


def despine(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def hline_grid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=RULE, lw=0.6)
    ax.xaxis.grid(False)


def nlabel(ax, x, y, n, dy=-13):
    ax.annotate("n = %d" % n, (x, y), textcoords="offset points",
                xytext=(0, dy), ha="center", fontsize=9, color=MUTE)
