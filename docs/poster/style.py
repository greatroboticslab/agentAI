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

# Two looks, one set of names. `POSTER_LOOK=journal` gives the figures the
# palette an offset-printed journal plate had: black ink, three greys, and one
# muted spot colour that the press ran as a second pass. Colour there is
# expensive, so it is spent once. The default look is the reference deck's
# slate-blue system.
#
# The switch is an environment variable rather than an argument because every
# figure function imports these names at module scope; threading a palette
# through seventeen signatures would be the same change written seventeen times.
import os as _os

LOOK = _os.environ.get("POSTER_LOOK", "modern")

if LOOK == "mtsu":
    # Read out of the lab's own poster (MTSU_LaserCar_Poster.pptx) rather than
    # chosen: #1C6FB5 and #C2CEDA and #14314E are already this file's BLUE, RULE
    # and NAVY, because both came from the same house deck. What the template
    # adds is the near-black navy it sets text and the header band in, and the
    # rule that colour appears once -- in the band and on the section rules --
    # and never inside a plate except as the single series that carries the
    # claim.
    INK     = "#16202B"
    NAVY    = "#14314E"
    BLUE    = "#1C6FB5"   # the one spot colour, and it is the university's
    PALE    = "#EDF3F9"
    PALEBLU = "#D6E3F0"
    RULE    = "#C2CEDA"
    MUTE    = "#4C5A69"
    WHITE   = "#FFFFFF"
    GOOD    = "#7E8794"
    WARN    = "#5A6470"
    GREY    = "#A7AEB8"
    SERIES  = [INK, BLUE, MUTE, GREY, GOOD, WARN]
    FAMILY  = "Arial"
    FIGDIR  = "fig_mtsu"
elif LOOK == "journal":
    INK     = "#111111"   # plate black
    NAVY    = "#111111"
    BLUE    = "#4A4A4A"   # the first grey does the work colour used to do
    PALE    = "#F2F1EE"
    PALEBLU = "#E8E6E1"
    RULE    = "#BFBDB8"
    MUTE    = "#5C5A55"
    WHITE   = "#FFFFFF"
    GOOD    = "#8A8A8A"
    WARN    = "#8C3A1E"   # the one spot colour, a dull brick
    GREY    = "#A8A6A1"
    SERIES  = [INK, BLUE, WARN, GREY, MUTE, GOOD]
    FAMILY  = "Times New Roman"
    FIGDIR  = "fig_journal"
else:
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
    SERIES  = [BLUE, INK, GOOD, WARN, MUTE, GREY]
    FAMILY  = "Arial"
    FIGDIR  = "fig"

# In the journal look every series colour is a grey except one, so a reader
# separating two lines is reading shape and dash, not hue -- which is what the
# note below has always asked for and what colour figures rarely deliver.
MARKERS = ["o", "s", "^", "D", "v", "P"]
DASHES  = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]

# BLUE 0.150 / WARN 0.157 / GOOD 0.148 relative luminance -- within 0.01 of one
# another. Any two of them are ONE mark in a photocopy or a phone photo, so
# colour is never the only thing separating two series here: pair it with a
# marker shape, a dash pattern, a position, or the word itself.

# The width build.py actually places each figure at, in inches. Authoring a
# figure at any other width means the layout scales it, and scaling a figure
# scales its type: before this table existed the same declared 10.5 pt tick
# printed at 32.5 pt in p_field (x3.09) and 12.5 pt in c_ladder (x1.19) -- a
# 2.6x typographic spread across one sheet, invisible in the source because
# every figure said font.size=11. `save()` asserts against it.
PLACED = {
    "g_router": 11.6,
    "l_robots": 11.6, "h_drive": 11.6, "k_funnel": 11.6, "s_sources": 11.6,
    "d_wall": 11.6, "p_field": 11.6, "j_species": 11.6,
    # There is no 10.925 width any more. It existed so a four-column grid could
    # squeeze two narrower middle columns in, and the cost was that a plate
    # authored at 11.6 landed in a 10.925 column and was silently not drawn --
    # which is how the rover's and the cart's own frame mosaics went missing
    # from a sheet whose text still described them. Four columns of 11.6 fit a
    # 48 in sheet with 0.30 margins and 0.333 gutters.
    "a_families": 11.6, "b_zeroshot": 11.6, "c_ladder": 11.6,
    "e_supervision": 11.6, "r_tta": 11.6,
    "f_rounds": 22.4,
    "v_r241": 11.6, "v_cart": 11.6,
    "u_detect": 46.4,
    "m_ledger": 22.4, "n_system": 22.4, "t_journey": 46.4, "w_robots": 46.4, "q_projects": 46.4,
    "x_algorithm": 46.4,
}

# Three sizes, named, so a figure cannot drift into seven of them 1.5 pt apart.
# These are PRINTED points, because every figure is authored at its placed
# width. Matched to the poster's own reading tier -- 20 pt body, 16 pt captions
# -- so a figure label sits just under its caption instead of shouting over the
# body text or disappearing under it.
TICK, AXIS, ANNOT, LETTER = 14.0, 16.0, 13.5, 20.0

plt.rcParams.update({
    "font.family": FAMILY,
    "font.size": TICK,
    "axes.titlesize": TICK,
    "axes.labelsize": AXIS,
    # Without these, mathtext falls back to italic DejaVu Sans: every
    # `mAP$_{50-95}$` printed Arial letters with a DejaVu subscript, two
    # typefaces inside one label.
    "mathtext.fontset": "custom",
    "mathtext.rm": FAMILY,
    "mathtext.it": FAMILY + ":italic",
    "mathtext.bf": FAMILY + ":bold",
    "mathtext.default": "regular",
    # Type 3 is a named preflight failure wherever a PDF poster is submitted.
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "normal",
    "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4.5, "ytick.major.size": 4.5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "legend.frameon": False, "legend.fontsize": TICK,
    "grid.color": RULE, "grid.linewidth": 0.9,
    "figure.dpi": 300, "savefig.dpi": 300,
    # NOT "tight": trimming the canvas means figsize is no longer the saved
    # width, and then a figure cannot be authored at the width it is placed at.
    "savefig.bbox": None, "savefig.pad_inches": 0.06,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.04,
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.wspace": 0.06,
    "figure.facecolor": WHITE, "savefig.facecolor": WHITE,
    # Data heaviest, annotation middle, grid and spines lightest -- which is
    # the layering this module's docstring claims and did not have.
    "lines.linewidth": 2.0, "lines.markersize": 7,
    "errorbar.capsize": 3,
})


def panel(ax, letter, dx=None, dy=None, inside=False, fig=None, figx=None):
    """Lower-case bold panel letter where a caption can point at it.

    Anchored in FIGURE coordinates off the axes box, so every letter sits at the
    same offset from its own panel across all thirteen figures. Anchoring in axes
    coordinates made the offset depend on how wide that panel's y-label happened
    to be, and in one figure the letter landed on the label's closing bracket.

    `inside` puts it in the top-left of the plotting area instead, which is the
    right call when the panel is a photograph or when the axes reach the canvas.
    `figx` overrides the horizontal anchor in figure coordinates, which is what a
    horizontal bar chart needs: its category labels sit outside the axes box, so
    an offset measured from that box lands on top of them.

    `dx`/`dy` are accepted and ignored; call sites still pass them.
    """
    if inside:
        ax.text(0.015, 0.985, letter, transform=ax.transAxes, fontsize=LETTER,
                fontweight="bold", va="top", ha="left", color=INK, zorder=8)
        return
    f = fig or ax.figure
    bb = ax.get_position()
    f.text(figx if figx is not None else max(0.004, bb.x0 - 0.055),
           min(0.995, bb.y1 + 0.030), letter,
           fontsize=LETTER, fontweight="bold", va="top", ha="left", color=INK)


def despine(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def hline_grid(ax):
    """A horizontal rule only where a reader must carry a value across distance.

    Not on any figure whose marks already print their own number -- grid plus
    ticks plus length plus digits is the same fact four times.
    """
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=RULE, lw=0.7)
    ax.xaxis.grid(False)


def nlabel(ax, x, y, n, dy=-13):
    ax.annotate("n = %d" % n, (x, y), textcoords="offset points",
                xytext=(0, dy), ha="center", fontsize=9, color=MUTE)


def sig(v, err, signed=False):
    """Quote a value to the precision its own uncertainty supports.

    Four decimals against a seed spread of 0.0040 asserts a precision the
    measurement does not have, and in one figure it printed +0.0174 and +0.0172
    as two different numbers whose bars differ by a millimetre while the caption
    rounded both to +0.017. The minus is U+2212 so a negative datum is set the
    same way the axis sets it.
    """
    import math
    d = max(0, -int(math.floor(math.log10(abs(err))))) if err else 3
    s = ("%+." + str(d) + "f") if signed else ("%." + str(d) + "f")
    return (s % v).replace("-", "\u2212")


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion. No new dependency.

    The normal approximation is wrong at the ends and these arms sit near them:
    one arm is 40 of 57 and another 32 of 39. Wilson is the interval that stays
    inside [0, 1] and does not collapse when k reaches n.
    """
    if not n:
        return (0.0, 0.0)
    p = float(k) / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4.0 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))
