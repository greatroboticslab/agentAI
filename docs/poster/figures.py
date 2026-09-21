#!/usr/bin/env python3
"""Poster figures, drawn to journal convention. One command:

    python3 docs/poster/figures.py

No figure carries a title — the claim belongs in the caption, which is set in the
poster beside it. Each function returns (filename, caption) and the captions are
printed so the poster text cannot drift from the figure.
"""
import os, sys, json, csv, re, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style import (INK, NAVY, BLUE, PALE, PALEBLU, RULE, MUTE, WHITE, GOOD, WARN,
                   GREY, panel, despine, hline_grid, PLACED, sig, wilson,
                   TICK, AXIS, ANNOT, LETTER, FIGDIR, LOOK, SERIES)
import poster_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, FIGDIR)
os.makedirs(OUT, exist_ok=True)
CAPS = {}


def save(fig, name, caption):
    p = os.path.join(OUT, name + ".png")
    w = fig.get_size_inches()[0]
    placed = PLACED.get(name)
    assert placed and abs(w / placed - 1.0) < 0.02, (
        "%s is authored %.3f in wide but build.py places it at %.3f in (x%.2f): "
        "the layout would scale its type" % (name, w, placed or 0, (placed or w) / w))
    fig.savefig(p)
    plt.close(fig)
    # A caption is printed twice: here, and by python-pptx into the poster. PowerPoint
    # has no mathtext, so a caption carrying $...$ prints the dollars and the braces.
    assert "$" not in caption, "%s: caption carries mathtext, which PowerPoint prints raw" % name
    CAPS[name] = caption
    print("%-16s %s" % (name, caption[:118]))
    return p


# ---------------------------------------------------------------- a. families
def fam():
    """Three detector families, as positions rather than as bar lengths.

    This was a bar chart on a baseline of 0.78. Bars encode by LENGTH, so a
    cropped baseline makes the length a lie: the three heights above 0.78 were
    0.0955 / 0.0465 / 0.0241 for data 0.8755 / 0.8266 / 0.8041, drawing the first
    family at 3.96x the third for a difference of 1.089x. A dot encodes by
    POSITION, which a cropped scale does not distort, and the crop is what makes
    the effect visible at all. The bottom spine goes with the bars: at 0.78 it
    read as a zero it is not.

    The three per-seed runs are drawn behind each mean. The Mamba family's spread
    is twice the others' and it is a low outlier at 0.8203 -- a reader who is
    shown only the mean and one bar cannot see that, and it is the arm the
    architecture claim rests on.
    """
    S = json.load(open(os.path.join(HERE, "figures_data.json")))["s3_families_2026_08_24"]
    # Colour is bound to the claim, which is about initialisation, not to the
    # row index: one hue for the pretrained arm, one grey for both random-init
    # arms, so the figure says what it is about before a label is read.
    rows = [("YOLO11n, COCO-pretrained", "yolo11n_sealed", BLUE),
            ("Mamba-YOLO-T, random init", "mamba_yolo_t", GREY),
            ("YOLO11n, random init", "yolo11n_scratch_control", GREY)]
    fig, ax = plt.subplots(figsize=(11.600, 3.77))
    ys = np.arange(len(rows))[::-1]        # first row on top
    for y, (lab, key, col) in zip(ys, rows):
        d = S[key]
        seeds = d["per_seed"]
        ax.plot(seeds, np.full(len(seeds), y) + np.linspace(-0.16, 0.16, len(seeds)),
                "o", ms=4.0, mfc="none", mec=GREY, mew=0.9, zorder=2)
        ax.errorbar(d["mean"], y, xerr=d["std"], fmt="o", ms=8, color=col,
                    elinewidth=1.3, capsize=0, zorder=4)
        ax.text(d["mean"] + 0.0055, y, sig(d["mean"], d["std"]), ha="left",
                va="center", fontsize=ANNOT, color=INK, zorder=5)

    # The two legitimate comparisons, set in the right margin the way a forest
    # plot sets a contrast. The pretrained-versus-Mamba pair is deliberately
    # absent: figures_data.json flags it CONFOUND, because one side started from
    # COCO weights and the other from scratch, so it is not an architecture
    # result and nothing in this figure should invite reading it as one.
    def span(ya, yb, x, txt, sub):
        ax.plot([x, x], [ya, yb], lw=0.8, color=MUTE, zorder=3, clip_on=False)
        for yy in (ya, yb):
            ax.plot([x - 0.0022, x], [yy, yy], lw=0.8, color=MUTE, clip_on=False)
        ax.text(x + 0.0035, (ya + yb) / 2, txt, ha="left", va="bottom",
                fontsize=ANNOT, color=INK, clip_on=False)
        ax.text(x + 0.0035, (ya + yb) / 2, sub, ha="left", va="top",
                fontsize=ANNOT, color=MUTE, clip_on=False)
    span(ys[0], ys[2], 0.9125, "+0.0714", "pretraining")
    span(ys[1], ys[2], 0.8865, "+0.0225", "architecture")

    ax.set_yticks(ys)
    ax.set_yticklabels(["%s   n = %d" % (r[0], S[r[1]]["n_seeds"]) for r in rows],
                       fontsize=TICK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.set_xlim(0.792, 0.955)
    # Four ticks, not eight: every mean prints its own value beside it, so a
    # gridline at every 0.01 would be the same fact a third time.
    ax.set_xticks([0.80, 0.84, 0.88])
    ax.set_xlabel("mAP$_{50\\mathdefault{-}95}$   on the sealed 1,977-image holdout")
    despine(ax, left=False)
    ax.spines["bottom"].set_bounds(0.80, 0.88)
    return save(fig, "a_families",
                "Three detector families on the sealed 1,977-image CottonWeedDet12 holdout. Filled "
                "mark: the mean over seeds 101/102/103, with 1 s.d.; open marks behind it: the three "
                "runs themselves. How the network is initialised is worth three times what the "
                "architecture is worth. The pretrained-versus-Mamba pair carries no bracket on "
                "purpose -- one side started from COCO weights and the other from scratch, so it is "
                "not an architecture comparison.")


# ---------------------------------------------------------------- b. zero-shot
def vlm():
    import json as _j
    fd = _j.load(open(os.path.join(HERE, "figures_data.json")))
    rows = [r for r in fd["benchmark_cwd12_map50"] if r.get("map50") is not None]
    rows = sorted(rows, key=lambda r: r["map50"])
    fig, ax = plt.subplots(figsize=(11.600, 5.63))
    y = np.arange(len(rows))
    cols = [BLUE if "fine-tuned" in r["model"] else GREY for r in rows]
    ax.barh(y, [r["map50"] for r in rows], color=cols, height=0.62)
    for i, r in enumerate(rows):
        ax.text(r["map50"] + 0.012, i, "%.3f" % r["map50"], va="center",
                fontsize=10, color=INK if cols[i] == BLUE else MUTE)
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"].replace(" / ", "/").replace(
        "G-DINO/Molmo/Llama-Vision/Moondream/LLaVA", "5 models with no grounding")
        for r in rows], fontsize=9.5)
    ax.set_xlim(0, 1.04); ax.set_xlabel("mAP$_{50}$, 848-image CottonWeedDet12 test split")
    despine(ax, left=False)
    ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    return save(fig, "b_zeroshot",
                "One deterministic pass per model, no repeats. Fifteen zero-shot vision-language "
                "models against the fine-tuned detector: the best of them reaches 0.434 against "
                "0.929, and five produce no usable grounding at all.")


# ---------------------------------------------------------------- c. ladder
def ladder():
    """Two exams, the same eight checkpoints, side by side.

    Drawing only the CottonWeedDet12 panel invites the obvious objection -- that
    metric rewards training data resembling CottonWeedDet12, so a greenhouse and
    aerial corpus can only cost. The second panel answers it in the figure."""
    L = D.LADDER; X = D.SECOND_EXAM
    seeds = L.get("seeds") or {}
    rungs = L["rungs"]; xs = np.arange(len(rungs))
    A, Ae, An = [], [], []
    for k, v in zip(rungs, L["seed101"]):
        sv = seeds.get(k)
        if isinstance(sv, list) and len(sv) >= 2:
            A.append(st.mean(sv)); Ae.append(st.stdev(sv)); An.append(len(sv))
        else:
            A.append(v); Ae.append(0.0); An.append(1)
    B = L.get("armB")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.600, 3.60))
    ax.errorbar(xs, A, yerr=Ae, fmt="o-", color=BLUE, capsize=4, elinewidth=1.0, zorder=3)
    if B:
        ax.plot(xs, B, "s--", color=WARN, markerfacecolor=WHITE, zorder=3)
    ax.text(xs[1] + 0.08, A[1] + 0.0022, "class per source dataset", ha="left",
            va="bottom", fontsize=TICK - 2, color=BLUE)
    # between two rules, not on one. Above the series it sat on the line it
    # names; below it, it sat on the 0.820 gridline.
    ax.text(2.20, 0.8355, "one shared class", ha="right", va="center",
            fontsize=TICK - 2, color=WARN)
    ax.annotate("", xy=(3.28, A[0]), xytext=(3.28, A[-1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.text(3.36, (A[0] + A[-1]) / 2, "−0.0189\n8.2 σ", fontsize=9.5, color=INK,
            va="center")
    ax.set_xlim(-0.35, 4.15)
    ax.set_ylabel("mAP$_{50\\mathdefault{-}95}$")
    ax.text(0.0, 1.035, "CottonWeedDet12 holdout", transform=ax.transAxes,
            fontsize=TICK - 1.5, color=MUTE, va="bottom", ha="left")
    ax.set_ylim(0.818, 0.872)

    a2 = [X["armA"][k] for k in rungs]; b2 = [X["armB"][k] for k in rungs]
    ax2.plot(xs, a2, "o-", color=BLUE, zorder=3)
    ax2.plot(xs, b2, "s--", color=WARN, markerfacecolor=WHITE, zorder=3)
    ax2.set_ylabel("mAP$_{50\\mathdefault{-}95}$")
    ax2.text(0.0, 1.035, "ImageWeeds, 3,208 images, class-agnostic",
             transform=ax2.transAxes, fontsize=TICK - 1.5, color=MUTE,
             va="bottom", ha="left")
    ax2.set_ylim(0.055, 0.118)
    ax2.set_xlim(-0.35, 3.35)

    for a_, lets in ((ax, "c"), (ax2, "d")):
        a_.set_xticks(xs)
        a_.set_xticklabels(["core\nalone", "+5k", "+15k", "+40k"], fontsize=10)
        despine(a_); hline_grid(a_); panel(a_, lets, inside=True)
    # Below the n row, not through it. At labelpad 14 the axis label sat at the
    # same height as the per-rung n annotations and the "+40k" one crossed it.
    ax.set_xlabel("harvested images added", labelpad=36)
    ax2.set_xlabel("harvested images added", labelpad=36)
    for i, n in enumerate(An):
        ax.annotate("n = %d" % n, (xs[i], 0), xytext=(0, -42),
                    textcoords="offset points", xycoords=("data", "axes fraction"),
                    ha="center", fontsize=TICK - 3, color=MUTE,
                    annotation_clip=False)
    for i in range(4):
        ax2.annotate("n = 1", (i, 0), xytext=(0, -42),
                     textcoords="offset points", xycoords=("data", "axes fraction"),
                     ha="center", fontsize=TICK - 3, color=MUTE,
                     annotation_clip=False)
    return save(fig, "c_ladder",
                
                
                "(c) Three seeds at each level, mean and 1 s.d., on the 1,977-image holdout, which is also each run's validation set, so every mark is that run's best epoch. (d) The same eight checkpoints on an unseen weed dataset, one seed per level. Both panels decline.")


# ---------------------------------------------------------------- d. the wall
def wall():
    W = D.WALL
    a = float(W["in_domain"].split("±")[0]); ae = float(W["in_domain"].split("±")[1])
    b = float(W["out_domain"].split("±")[0]); be = float(W["out_domain"].split("±")[1])
    fig, ax = plt.subplots(figsize=(11.600, 5.30))
    ax.errorbar([0, 1], [a, b], yerr=[ae, be], fmt="o-", color=INK,
                capsize=4, elinewidth=1.0, zorder=3)
    ax.annotate("%.4f" % a, (0, a), xytext=(8, 4), textcoords="offset points",
                fontsize=11.5, color=INK)
    # Below-left of the endpoint: above it the label sat on its own marker.
    # Right of the endpoint, where the x limit leaves a gutter: above the point
    # the label sat on its own marker and below it on the tick label.
    ax.annotate("%.4f" % b, (1, b), xytext=(12, 0), textcoords="offset points",
                ha="left", va="center", fontsize=11.5, color=WARN)
    ax.annotate("", xy=(0.5, a), xytext=(0.5, b),
                arrowprops=dict(arrowstyle="<->", color=MUTE, lw=0.9))
    ax.text(0.545, (a + b) / 2, "−0.773", fontsize=12, color=INK, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CottonWeedDet12\nheld out", "ImageWeeds\n3,208 images"],
                       fontsize=10)
    ax.set_xlim(-0.42, 1.42); ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP$_{50\\mathdefault{-}95}$, class-agnostic")
    despine(ax); hline_grid(ax); panel(ax, "d", dx=-0.25)
    ax.annotate("n = 3 both ends", (0.5, 0.045), ha="center", fontsize=9, color=MUTE)
    return save(fig, "d_wall",
                "The same three checkpoints and the same matcher on both sides, so the evaluator "
                "offset cancels. A detector trained on one field does not transfer to another.")


# ---------------------------------------------------------------- e. supervision
def supervision():
    """What each tier of supervision catches, and how much of it is evidenced.

    Panel (e) is the contest the benchmark was built for: detection against false
    alarms. Panel (f) exists because (e) alone would mislead -- the 7 B model has
    the highest recall on the page and two thirds of what it "detects" quotes a
    line that does not resolve in the artifact.

    Three things this figure has to get right and an earlier version did not.
    The DENOMINATOR is per arm, not per figure: two arms are still running and
    stand on 57 and 39 incidents against the others' 112-114, so a single
    "n = 116" on the axis was wrong for every mark on it. An ARROW asserts a
    paired comparison, so it is drawn only where both ends were scored on the
    same number of incidents. And every mark is a proportion from a few dozen
    cases, so it carries a Wilson interval: without one, the 14 B's tier gain of
    a single incident is drawn at the same weight as the 27 B's fourteen.
    """
    import matplotlib.patheffects as pe
    S = D.SUPERVISION
    A = S["table"]["arms"]

    def pt(key):
        a = A[key]
        r = a["detection_recall"]
        return {"fa": a["false_alarm_rate"]["v"], "r": r["v"],
                "g": a["detection_grounded"]["v"], "k": r["k"], "n": r["n"],
                "fa_k": a["false_alarm_rate"]["k"], "fa_n": a["false_alarm_rate"]["n"],
                "cases": a["counts"]["cases"]}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.600, 5.00), sharey=True)

    # ---- (e) detection against false alarms ---------------------------------
    # Square, so the chance diagonal is a true 45 degrees and vertical distance
    # above it -- the only thing that line exists to show -- can be read. At the
    # earlier 2.9x compression it sat at 19 degrees and the 14 B looked far above
    # a line it is barely above.
    ax.set_box_aspect(0.88)
    ax.plot([0, 0.95], [0, 0.95], color=MUTE, lw=0.9, zorder=1)
    ax.text(0.885, 0.835, "chance", fontsize=ANNOT, color=MUTE, rotation=45,
            ha="center", va="top", rotation_mode="anchor")
    ax.axhline(S["ceiling"], color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    # the dashed rule is named in the caption, not on top of the data

    def mark(a, x, y, colour, ms=8, filled=True):
        lo, hi = wilson(a["k"], a["n"])
        flo, fhi = wilson(a["fa_k"], a["fa_n"])
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], xerr=[[x - flo], [fhi - x]],
                    fmt="none", elinewidth=0.8, capsize=0, color=colour,
                    alpha=0.40, zorder=3)
        ax.plot(x, y, "o", color=colour, ms=ms,
                markerfacecolor=colour if filled else WHITE,
                markeredgewidth=0 if filled else 1.6, zorder=5)

    a0p = pt("A0p")
    mark(a0p, a0p["fa"], a0p["r"], MUTE, ms=7)
    ax.annotate("12 deterministic rules, n = %d" % a0p["n"], (a0p["fa"], a0p["r"]),
                textcoords="offset points", xytext=(13, -6), ha="left",
                va="top", fontsize=ANNOT, color=MUTE)
    # The watchdog returned no decision on any of 149 cases, so it has no
    # coordinate on either axis. It sits in the margin rather than at an
    # arithmetic (0, 0) that would read as a measurement.
    ax.plot(-0.045, -0.045, "s", color=GREY, ms=7, clip_on=False, zorder=5)
    ax.annotate("scripted watchdog",
                (-0.045, -0.045), textcoords="offset points", xytext=(13, 1),
                ha="left", va="center", fontsize=ANNOT, color=MUTE,
                annotation_clip=False,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=WHITE)])

    # Full model names, not parameter counts. "27 B" is not something a reader
    # can check against anything, and a figure in a journal names the thing it
    # measured. The name goes on its own line so the label stays narrow: three
    # of the four arms sit inside a false-alarm band 0.12 wide.
    style = {"GLM-4.7-Flash": (GOOD, (0.010, 0.995), "left", "top"),
             "Qwen2.5-7B":    (WARN, (0.345, 0.995), "left", "top"),
             "Qwen3.8-27B":   (NAVY, (0.010, 0.430), "left", "top"),
             "Qwen3-14B":     (BLUE, (0.560, 0.545), "left", "top")}
    for m in S["models"]:
        c, (lx, ly), ha, va = style[m["name"]]
        p2 = pt(m["l2"])
        mark(p2, p2["fa"], p2["r"], c)
        lab = "%s\nn = %d" % (m["name"], p2["n"])
        if m["l3"]:
            p3 = pt(m["l3"])
            mark(p3, p3["fa"], p3["r"], c)
            # "Same cases" within the handful that one side failed to score:
            # 112 against 113 is the same experiment, 114 against 57 is not.
            paired = abs(p2["n"] - p3["n"]) <= 3
            if paired:
                # An arrow asserts the same cases at both ends. Only then.
                ax.annotate("", (p3["fa"], p3["r"]), xytext=(p2["fa"], p2["r"]),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2,
                                            shrinkA=8, shrinkB=8,
                                            connectionstyle="arc3,rad=0.18"),
                            zorder=4)
                lab = "%s\nn = %d" % (m["name"], p2["n"])
            else:
                ax.plot([p2["fa"], p3["fa"]], [p2["r"], p3["r"]], ls=(0, (2, 2)),
                        lw=1.0, color=c, zorder=4)
                lab = "%s\nn = %d" % (m["name"], p2["n"])
                ax.annotate("n = %d" % p3["n"], (p3["fa"], p3["r"]),
                            textcoords="offset points", xytext=(13, 4),
                            ha="left", fontsize=ANNOT, color=c,
                            path_effects=[pe.withStroke(linewidth=2.2,
                                                        foreground=WHITE)])
        ax.text(lx, ly, lab, ha=ha, va=va, fontsize=ANNOT, color=c,
                linespacing=1.25,
                path_effects=[pe.withStroke(linewidth=2.2, foreground=WHITE)])

    ax.set_xlim(-0.06, 0.97); ax.set_ylim(-0.06, 0.97)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("detection recall")
    despine(ax); hline_grid(ax); panel(ax, "e")
    for sp in ax.spines.values():
        sp.set_zorder(8)

    # ---- (f) how much of that detection is evidenced ------------------------
    ax2.set_box_aspect(0.88)
    xs = np.arange(len(S["models"]))
    for i, m in enumerate(S["models"]):
        c = style[m["name"]][0]
        key = m["l3"] or m["l2"]
        a = pt(key)
        gap = a["r"] - a["g"]
        if gap < 0.02:
            # One mark, not two on top of each other: at this marker size a gap
            # of zero would show a filled dot with the open one hidden beneath,
            # which reads against the key as "quoted a line without flagging".
            ax2.plot(i, a["r"], "o", color=c, ms=9, markerfacecolor=WHITE,
                     markeredgewidth=2.2, zorder=4)
            ax2.plot(i, a["r"], "o", color=c, ms=4.2, zorder=5)
            ax2.text(i + 0.17, a["r"], "no gap", fontsize=ANNOT, color=c,
                     va="center", ha="left")
        else:
            ax2.plot([i, i], [a["g"], a["r"]], color=c, lw=2.6,
                     solid_capstyle="butt", zorder=3)
            ax2.plot(i, a["r"], "o", color=c, ms=8, markerfacecolor=WHITE,
                     markeredgewidth=1.6, zorder=4)
            ax2.plot(i, a["g"], "o", color=c, ms=8, zorder=4)
            ax2.text(i + 0.17, (a["r"] + a["g"]) / 2,
                     "%.2f\n%d of %d" % (gap, round(gap * a["n"]), a["n"]),
                     fontsize=ANNOT, color=c, va="center", ha="left",
                     linespacing=1.2)
    ax2.text(-0.42, 0.135, "open   flagged an incident", fontsize=ANNOT, color=MUTE)
    ax2.text(-0.42, 0.055, "solid   and quoted a line that resolves",
             fontsize=ANNOT, color=MUTE)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([m["name"] for m in S["models"]], fontsize=TICK - 1.5,
                        rotation=18, ha="right")
    ax2.set_xlim(-0.5, len(xs) - 0.10)
    ax2.set_xlabel("reviewer")
    ax2.tick_params(labelleft=True)
    despine(ax2); hline_grid(ax2); panel(ax2, "f")

    g7 = pt("L3@qwen2.5:7b"); g27 = pt("L3@qwen3.8:27b")
    d27 = pt("L3@qwen3.8:27b")["r"] - pt("L2@qwen3.8:27b")["r"]
    d14 = pt("L3@qwen3:14b")["r"] - pt("L2@qwen3:14b")["r"]
    return save(fig, "e_supervision",
                
                "(e) Recall and (f) grounded recall on 149 frozen failures taken from this project's own record, scored by the project's own scorer. Each arrow runs from the model reading the raw files to the same model handed the twelve checks' findings. Grounded counts only the cases where the model also quoted a line that resolves the failure. The dotted arrow carries no head: its two ends were scored on different cases, 57 against 114.")


# --------------------------------------------------- photographic plates
def _plate(name, path, width, caption, aspect=None):
    """A photograph placed at a column width, with nothing drawn over it.

    Authored at the width the layout places it at, like every other plate here,
    so the assertion in save() applies to photographs too. `aspect` crops the
    picture to width/height; without it the picture keeps its own.
    """
    im = plt.imread(os.path.join(HERE, "photos", path))
    h, w = im.shape[0], im.shape[1]
    if aspect:
        keep = int(round(w / aspect))
        if keep < h:
            top = (h - keep) // 2
            im = im[top:top + keep]
        h = im.shape[0]
    fig = plt.figure(figsize=(width, width * h / float(w)))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return save(fig, name, caption)


def _cells(path, cols, cell_ar=16.0 / 9.0):
    """How many cells a mosaic has, read off the file rather than remembered.

    The cells are camera frames at 16:9, not squares, so a row is
    (width / cols) / 16 * 9 tall. Assuming square cells read a 3 x 2 mosaic as
    3 x 1 and printed "3 frames" under six pictures.
    """
    from PIL import Image
    w, h = Image.open(os.path.join(HERE, "photos", path)).size
    rows = int(round(h / ((w / float(cols)) / cell_ar)))
    return cols * max(1, rows)


def detect_grid():
    # Both numbers, and what separates them. A reader who checks one will find
    # the other, and a caption that quotes only the higher one is the kind of
    # thing this poster's footer exists to apologise for.
    P = D.DEPLOYED
    return _plate(
        "u_detect", "u_detect.jpg", 46.4,
        "Twelve images from the sealed %s-image holdout, one per species, run through the "
        "deployed checkpoint at the confidence the laser robot fires at. Every box and every label is "
        "the model's output; no ground truth is drawn. Seed %d scores %.4f mAP50-95 and %.4f "
        "mAP50 over all %s images on one plain inference pass; the ledger's %s is the three-seed "
        "mean at each run's best epoch under Ultralytics, on a holdout that doubled as the "
        "validation set, and the %.4f between them is evaluator and epoch choice." % ("{:,}".format(P["n_images"]), P["seed"], P["map50_95"], P["map50"],
                     "{:,}".format(P["n_images"]), P["ledger_mean"], P["gap"]))


def r241_grid():
    n = _cells("m_r241.jpg", 3)
    return _plate(
        "v_r241", "m_r241.jpg", 11.6,
        "The Rover, %d frames from %d drives. Each is the most vegetated frame in its drive, which "
        "is a selection rule and is stated as one: the rover also records paved and indoor runs. "
        "2,238 frames in all, none labelled." % (n, n))


def cart_grid():
    n = _cells("m_cart.jpg", 3)
    return _plate(
        "v_cart", "m_cart.jpg", 11.6,
        "The laser robot, %d frames from six drives. The rectangle is the laser robot's fixed work-zone "
        "overlay: it sits at the same pixels whether the frame is grass or an indoor floor, so it "
        "is not a detection. 427 frames in all." % n)



def _router_roles():
    """model_router.ROLES, or a recorded copy if the framework is not importable.

    The poster is built on a laptop that does not always have the framework's
    dependencies. Falling back silently to a stale copy is exactly the drift
    this change exists to stop, so the fallback carries the commit it was taken
    from and the test compares the two.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "weed_llm_benchmark"))
    sys.path.insert(0, os.path.join(root, "weed_optimizer_framework", "tools"))
    import model_router
    return model_router.ROLES


# ----------------------------------------------------------- the algorithm
def router():
    """Where each role runs, and whether its answer counts.

    Every row is read out of `weed_optimizer_framework/tools/model_router.py`:
    the eight roles, their place, their model, and whether the role exercises
    judgement. The rule the figure exists to show is one line of that file --
    `"authoritative": bool(place == "cluster") if judgement else True` -- so a
    judgement produced on the lab box returns marked as a draft. The flag
    travels with the answer instead of sitting in a configuration block, which
    is the difference between a rule and a note.

    "Read out of" used to be a claim the code did not honour: the two lists
    below were hand-typed literals that happened to match. A role added to the
    router, or a model swapped in it, would not have reached this figure and
    nothing would have said so. They are now built from ROLES itself, and every
    role must carry a printed label here or the render stops -- so the only way
    to desync the figure is to delete the assertion.
    """
    from matplotlib.patches import Rectangle
    W, H = 11.6, 5.72
    fig = plt.figure(figsize=(W, H)); fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ROLES = _router_roles()
    # role key -> the phrase a visitor reads. The router's own keys are for
    # code; "labeling_vlm" tells a reader at a poster nothing.
    SAYS = {"planner":          "plan the next step",
            "deep_review":      "review in depth",
            "hard_reasoning":   "hard one-off reasoning",
            "harvest_brain":    "decide what to collect",
            "curation":         "judge dataset quality",
            "labeling_vlm":     "label assistance",
            "interactive_plan": "turn intent into a project",
            "analysis_summary": "summarise a dataset"}
    missing = [r for r in ROLES if r not in SAYS]
    assert not missing, ("model_router.ROLES carries a role this figure has no "
                         "label for, so the table would print short: %s" % missing)
    ORDER = ["planner", "deep_review", "hard_reasoning", "harvest_brain",
             "curation", "labeling_vlm", "interactive_plan", "analysis_summary"]

    def _model_name(spec):
        """ollama:qwen3.8:27b -> qwen3.8 27B. The prefix is how it is served."""
        m = spec["model"].split(":", 1)[-1]
        fam, _, size = m.partition(":")
        fam = fam.replace("gemma4", "gemma 4")
        return ("%s %s" % (fam, size.upper())) if size else fam

    rows = {"cluster": [], "lab": []}
    for r in ORDER:
        if r not in ROLES:
            continue
        spec = ROLES[r]
        rows[spec["place"]].append(
            (SAYS[r], _model_name(spec), bool(spec.get("judgement", False))))
    CLUSTER, LAB = rows["cluster"], rows["lab"]
    assert len(CLUSTER) + len(LAB) == len(ROLES)

    ROW, GAP, HEAD = 0.445, 0.30, 0.60
    x0, xm = 0.30, 6.65          # role column, model column
    y = H - 0.30

    def band(title, note, rows, y, tint):
        ax.text(x0, y, title, fontsize=22.0, fontweight="bold",
                color=INK, va="top", ha="left")
        ax.text(W - 0.30, y, note, fontsize=17.5, color=MUTE,
                va="top", ha="right", style="italic")
        y -= HEAD
        top = y + 0.10
        for name, model, judge in rows:
            ax.add_patch(Rectangle((x0 - 0.16, y - ROW + 0.10), W - 2 * x0 + 0.32,
                                   ROW - 0.08, facecolor=tint, edgecolor="none",
                                   zorder=1))
            if judge:
                ax.add_patch(Rectangle((x0 - 0.02, y - 0.265), 0.135, 0.135,
                                       facecolor=BLUE, edgecolor="none", zorder=3))
            ax.text(x0 + 0.26, y - 0.20, name, fontsize=19.5, color=INK,
                    va="center", ha="left", zorder=3)
            ax.text(xm, y - 0.20, model, fontsize=18.0, color=MUTE,
                    va="center", ha="left", zorder=3)
            y -= ROW
        ax.plot([x0 - 0.16, W - x0 + 0.16], [top, top], color=RULE, lw=1.4)
        return y

    y = band("On the cluster", "the answer is authoritative", CLUSTER, y, PALE)
    y -= GAP
    y = band("On the lab box", "the answer returns marked a draft", LAB, y, "#F3E9E2")
    ax.text(x0, y - 0.04, "filled square: the role exercises judgement",
            fontsize=16.0, color=MUTE, va="top", ha="left")
    assert y - 0.22 >= 0, ("the router table runs off its canvas by %.2f in -- "
                           "raise H or lower ROW" % (0.22 - y))
    return save(fig, "g_router",
                "The eight jobs that need a model, and the model each one was given, read from the router's own table. A filled square marks a job that exercises judgement. The six on the cluster come back authoritative; the two in the lab come back marked a draft.")


def _glyph(ax, kind, cx, cy, s, color=INK, lw=2.4, accent=None):
    """One line-drawn icon, centred on (cx, cy) inside a square of side s.

    Hongbo Zhang, on the diagram: "Within diagram, 需要用图标加上文字."
    Drawn rather than imported: an icon font would not survive the PNG round
    trip at 300 dpi, and these are five shapes each. Everything is stroked at
    one weight so the row reads as one set.
    """
    from matplotlib.patches import Circle, Polygon
    def L(pts, w=lw, col=None):
        xs = [cx + px * s for px, _py in pts]
        ys = [cy + py * s for _px, py in pts]
        ax.plot(xs, ys, color=col or color, lw=w, solid_capstyle="round",
                solid_joinstyle="round", zorder=6)
    def C(px, py, r, fill=False, col=None):
        ax.add_patch(Circle((cx + px * s, cy + py * s), r * s,
                            facecolor=(col or color) if fill else "none",
                            edgecolor=col or color, lw=lw, zorder=6))

    if kind == "need":                      # a need arrives -- a target
        C(0, 0, 0.46); C(0, 0, 0.26); C(0, 0, 0.09, fill=True)
    elif kind == "plan":                    # work out what would answer it
        C(-0.10, 0.10, 0.30)
        L([(0.12, -0.12), (0.44, -0.44)], w=lw * 1.15)
        L([(-0.44, -0.30), (-0.30, -0.30)])
        L([(-0.44, -0.46), (-0.02, -0.46)])
    elif kind == "fetch":                   # go and get it -- into the tray
        L([(0, 0.48), (0, -0.04)])
        L([(-0.20, 0.16), (0, -0.06), (0.20, 0.16)])
        L([(-0.44, -0.20), (-0.44, -0.44), (0.44, -0.44), (0.44, -0.20)])
    elif kind == "train":                   # train on what survived
        L([(-0.46, -0.44), (0.46, -0.44)])
        L([(-0.28, -0.44), (-0.28, -0.08)])
        L([(0.00, -0.44), (0.00, 0.16)])
        L([(0.28, -0.44), (0.28, 0.44)])
    elif kind == "decide":                  # read the result and decide
        L([(-0.34, 0.46), (0.34, 0.46), (0.34, -0.46), (-0.34, -0.46), (-0.34, 0.46)])
        L([(-0.18, 0.04), (-0.04, -0.14), (0.20, 0.22)], w=lw * 1.15,
          col=accent or color)
    elif kind == "shield":                  # a scripted watchdog
        L([(-0.32, 0.40), (0.32, 0.40), (0.32, -0.04), (0, -0.44),
           (-0.32, -0.04), (-0.32, 0.40)])
    elif kind == "checks":                  # deterministic checks
        for k, y in enumerate((0.30, 0.00, -0.30)):
            L([(-0.44, y + 0.04), (-0.32, y - 0.08), (-0.10, y + 0.18)])
            L([(0.06, y), (0.44, y)])
    elif kind == "read":                    # a model reads the artifacts
        L([(-0.30, 0.44), (0.14, 0.44), (0.30, 0.26), (0.30, -0.44),
           (-0.30, -0.44), (-0.30, 0.44)])
        L([(-0.14, 0.12), (0.14, 0.12)])
        L([(-0.14, -0.10), (0.14, -0.10)])
        L([(-0.14, -0.30), (0.02, -0.30)])
    elif kind == "retrieve":                # the same model, given retrieval
        L([(-0.46, -0.42), (-0.46, 0.34)])
        L([(-0.26, -0.42), (-0.26, 0.34)])
        L([(-0.06, -0.42), (-0.06, 0.34)])
        L([(-0.46, -0.42), (0.04, -0.42)])
        C(0.22, 0.14, 0.22)
        L([(0.38, -0.02), (0.48, -0.16)])
    else:
        raise ValueError("no glyph named %r" % kind)


def algorithm():
    """The loop the agent runs to answer a need, drawn as a loop.

    A schematic that is read in four seconds cannot carry counts and dates --
    Bang Wong's rule is that an overview figure conveys concepts, and that
    research data inside one is disconcerting -- and a figure called an
    algorithm has to be shaped like the algorithm. So: a racetrack. Five
    stations on the outward run, the way a person works a problem. A need
    arrives, work out what data would answer it, go and get it, train on what
    survives, read the result and decide what is next. The return run carries
    the decision back to the need, and it is DASHED, because that is the edge
    the platform has not closed. The escalation ladder climbs inside the loop,
    left to right, in the negative space the outward run leaves behind.

    Sized for a reader four feet away. The first version spent 5.1 in of a
    36 in sheet, put its content in the top half, and set its stations at
    24 pt against 22 pt body text -- a full-width figure whose type was the
    size of the paragraphs beside it. This one is a full inch shorter, the
    stations are half again as large, and the ladder runs across the empty
    middle instead of stacking into it. The inch goes to the columns, which
    were shedding sections for want of it.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
    W, H = 46.4, 5.10
    STN, SUB, LAD, EDGE = 34.0, 24.0, 29.0, 25.0
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    TOPY, BOTY = 3.85, 0.38
    L, R = 0.85, W - 0.85
    bw, bh = 8.05, 1.95

    def arrowhead(x, y, dx, color=INK, lw=1.9, ls="-"):
        ax.annotate("", xy=(x + dx, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    linestyle=ls, shrinkA=0, shrinkB=0))

    # Phase grounds. The stations were five identical boxes on an empty field,
    # which says nothing about which of them belong together. A tinted ground
    # per phase with the phase named on it is the device the reference figures
    # in this literature use, and it costs no height and no type size: the
    # boxes, their font and their spacing are untouched, the ground goes
    # behind them.
    PHASES = [("COLLECTION", 0, 2), ("TRAINING AND EVALUATION", 3, 4)]
    ax.add_patch(FancyBboxPatch((L, BOTY), R - L, TOPY - BOTY,
                                boxstyle="round,pad=0,rounding_size=0.55",
                                facecolor="none", edgecolor=RULE, lw=1.3,
                                zorder=1))
    ax.plot([L + 0.55, R - 0.55], [BOTY, BOTY], color=INK, lw=2.2,
            linestyle=(0, (8, 5)), zorder=2)
    arrowhead(L + 3.2, BOTY, -1.0, lw=2.2, ls=(0, (8, 5)))

    # Each station names what that step DOES. Every one of them used to be
    # captioned with its own defect -- "no new dataset in the last eight
    # rounds", "the next round submits identical commands" -- so the diagram
    # read as five confessions and a reader could not tell what the loop was
    # for. The defects are not dropped: they are stated once in the caption,
    # and the one with a shape in the drawing is stated on the edge it explains.
    stations = [
        ("Requirement definition", "a researcher's goal, planned out by a model", "need"),
        ("Source search", "public repositories are searched by keyword", "plan"),
        ("Acquisition and audit", "what arrives is audited before it is kept", "fetch"),
        ("Training on retained data", "trained on what survives the audit", "train"),
        ("Evaluation and reporting", "scored; a plateau test ends the run", "decide"),
    ]
    xs = [L + 0.30 + i * ((R - L - 0.60 - bw) / (len(stations) - 1))
          for i in range(len(stations))]
    # one number for the sub-line measure, used by the wrap AND by the
    # assertion that guards it. They were 40 and 44, so a 41-character
    # sub-line passed the check and printed on two lines, the second of
    # them below its own box.
    SUB_WRAP = 46
    fit = int(bw * 72 / (STN * 0.55))
    over = [t for t, _s, _i in stations if len(t) > fit]
    assert not over, ("station title wider than its box at %g pt (max %d chars): %s"
                      % (STN, fit, over))
    # Two sub-lines is what the box holds; a third prints below its own rule.
    deep = [t for t, sb, _i in stations if len(_wrap(sb, SUB_WRAP)) > 1]
    assert not deep, "station sub-line no longer fits one line: %s" % deep
    # drawn before the stations so the boxes sit ON the ground, not under it
    # The plate is 5.10 in and the boxes top out at 4.83, so there is no room
    # above them for a phase name. It goes in the empty band BETWEEN the station
    # row and the ladder, which the drawing was wasting anyway.
    PH_Y0, PH_Y1 = TOPY - bh / 2 - 0.52, min(H - 0.14, TOPY + bh / 2 + 0.16)
    for pi, (pname, a, b) in enumerate(PHASES):
        x0 = xs[a] - 0.34
        x1 = xs[b] + bw + 0.34
        ax.add_patch(FancyBboxPatch((x0, PH_Y0), x1 - x0, PH_Y1 - PH_Y0,
                                    boxstyle="round,pad=0,rounding_size=0.22",
                                    facecolor=PALEBLU if pi % 2 == 0 else "#C3D6EA",
                                    edgecolor="none", zorder=1))
        # A solid tab rather than type on the tint. The reference figures in
        # this literature set a stage's name in a filled bar, which is what
        # makes the stage read as a stage and not as a caption that happens to
        # be sitting there.
        tw = 0.46 + len(pname) * (SUB - 1) * 0.56 / 72.0
        ax.add_patch(FancyBboxPatch((x0 + 0.16, PH_Y0 + 0.08), tw, 0.46,
                                    boxstyle="round,pad=0,rounding_size=0.10",
                                    facecolor=NAVY, edgecolor="none", zorder=2))
        ax.text(x0 + 0.16 + tw / 2, PH_Y0 + 0.31, pname, fontsize=SUB - 1,
                fontweight="bold", color=WHITE, ha="center", va="center",
                zorder=3)

    top = TOPY + bh / 2
    for i, ((title, sub, icon), x) in enumerate(zip(stations, xs)):
        ax.add_patch(Rectangle((x, TOPY - bh / 2), bw, bh, facecolor=WHITE,
                               edgecolor=INK, lw=1.7, zorder=3))
        # A numbered marker instead of a line glyph. A magnifier over "Source
        # search" carries nothing the two words do not; a number carries the
        # one thing the row does not otherwise state, which is its order. Thin
        # generic line icons are also the clearest tell of a slide template.
        ax.add_patch(Circle((x + bw / 2, top - 0.52), 0.30,
                            facecolor=NAVY, edgecolor="none", zorder=4))
        ax.text(x + bw / 2, top - 0.52, str(i + 1), fontsize=SUB + 1,
                fontweight="bold", color=WHITE, ha="center", va="center",
                zorder=5)
        ax.text(x + bw / 2, top - 1.22, title, ha="center", va="center",
                fontsize=STN, fontweight="bold", color=INK, zorder=4)
        for k, line in enumerate(_wrap(sub, SUB_WRAP)):
            ax.text(x + bw / 2, top - 1.63 - k * 0.38, line, ha="center",
                    va="center", fontsize=SUB, color=MUTE, zorder=4)
        if i:
            mid = (xs[i - 1] + bw + x) / 2
            arrowhead(mid - 0.40, TOPY, 0.80, color=BLUE, lw=2.4)

    # A third phase ground, under the station row, for the review tiers. The
    # reference figures in this literature stack their stages as tinted bands;
    # this is the third band, and it is the one that runs across every step
    # rather than being a step itself. It stops short of the dashed return
    # edge, which is not part of it.
    RV_X0, RV_X1, RV_Y0, RV_Y1 = 11.30, R - 0.25, 0.42, TOPY - bh / 2 - 0.58
    ax.add_patch(FancyBboxPatch((RV_X0, RV_Y0), RV_X1 - RV_X0, RV_Y1 - RV_Y0,
                                boxstyle="round,pad=0,rounding_size=0.22",
                                facecolor="#F7FAFD", edgecolor="none", zorder=1))
    # The band name goes at its RIGHT edge: the left is where the entry label
    # sits, and at the left the two printed over one another.
    _rw = 0.46 + len("REVIEW") * (SUB - 1) * 0.56 / 72.0
    ax.add_patch(FancyBboxPatch((RV_X1 - 0.16 - _rw, RV_Y0 + 0.08), _rw, 0.46,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor=NAVY, edgecolor="none", zorder=2))
    ax.text(RV_X1 - 0.16 - _rw / 2, RV_Y0 + 0.31, "REVIEW", fontsize=SUB - 1,
            fontweight="bold", color=WHITE, ha="center", va="center", zorder=3)

    # the ladder, climbing left to right under the outward run --------------
    # Entered at the cheap end and read upward: the escalation goes to the
    # scripted watchdog first, each rung reads the one below it, and only the
    # verdict that survives the climb returns to the station that got stuck.
    # An arrow from the station into the TOP rung, which is what the first
    # version drew, says the opposite -- that being stuck summons the most
    # expensive reviewer.
    rungs = [("scripted watchdog", "shield"),
             ("deterministic checks", "checks"),
             ("the language model reads artifacts", "read"),
             ("the same model, given check findings", "retrieve")]
    rw, rh, rise, stepx = 8.70, 1.00, 0.28, 8.05
    rx0, ry0 = 12.10, 0.52
    for k, (name, icon) in enumerate(rungs):
        x, y = rx0 + k * stepx, ry0 + k * rise
        top = k == len(rungs) - 1
        ink = WHITE if top else INK
        # Cost rises with the rung, and the fill rises with it: the reference
        # figures grade a sequence by tone rather than leaving it to the reader
        # to infer from position alone. White, then two tints, then the solid
        # blue that the supervision benchmark is actually about.
        FILLS = [WHITE, PALE, PALEBLU, BLUE]   # band is lighter than all four
        ax.add_patch(Rectangle((x, y), rw, rh,
                               facecolor=FILLS[k],
                               edgecolor=BLUE if top else INK,
                               lw=2.2 if top else 1.5, zorder=3))
        # No glyph. The stations carry numbers now and the tiers carry tone;
        # a shield and a checklist beside them would be a third alphabet, and
        # a generic line icon is the tell this plate is being kept clear of.
        ax.text(x + 0.42, y + rh / 2, name, fontsize=LAD,
                va="center", ha="left", color=ink,
                fontweight="bold" if top else "normal", zorder=4)
    # a rung whose label is wider than its box is clipped, and the clip is
    # invisible at thumbnail size -- the top rung printed "handed those chec".
    # Two ways a rung label can be lost, and both are invisible at thumbnail
    # size. It can be wider than its own box, and -- for every rung but the
    # last -- it can run under the rung drawn on top of it, which is what
    # covered the tail of "reads artifacts".
    _w = lambda n: 0.42 + len(n) * LAD * 0.50 / 72.0
    wide = [n for n, _i in rungs if _w(n) > rw]
    assert not wide, "ladder rung label wider than its box: %s" % wide
    under = [n for n, _i in rungs[:-1] if _w(n) + 0.12 > stepx]
    assert not under, ("ladder rung label runs under the rung above it "
                       "(needs stepx >= %.2f): %s"
                       % (max(_w(n) for n, _i in rungs[:-1]) + 0.12, under))
    last = rx0 + (len(rungs) - 1) * stepx + rw
    assert last <= R - 0.40, "the ladder runs past the loop: %.2f > %.2f" % (last, R - 0.40)
    arrowhead(rx0 - 0.95, ry0 + rh / 2, 0.80)
    ax.text(rx0 - 1.20, ry0 + rh / 2, "after every step,\ncheapest first",
            fontsize=LAD, color=INK, ha="right", va="center",
            fontweight="bold", linespacing=1.22)
    # and the verdict that survives the climb comes back up to the station
    tx = rx0 + 3 * stepx + rw * 0.50
    ty = ry0 + 3 * rise + rh
    ax.annotate("", xy=(tx, TOPY - bh / 2 - 0.03), xytext=(tx, ty + 0.04),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.9,
                                shrinkA=0, shrinkB=0))

    # what the dashed run means, inside the loop where the space already was.
    # The explanation of WHY it is dashed belongs in the caption, not in the
    # drawing: four lines of prose inside a schematic is a paragraph wearing a
    # figure's clothes.
    lx = L + 3.60
    ax.text(lx, BOTY + 0.42, "the loop reads its result",
            fontsize=EDGE, color=INK, ha="center", va="center")

    return save(fig, "x_algorithm",
                
                
                "The harvest-and-train loop, as it runs unattended. A researcher's goal is planned into keyword queries, what they return is audited, what survives is trained on, and the round is scored. The stopping rule is a plateau test today and a reviewer's judgement by design. Review follows every step.")

# " -- " is how a sentence break is typed in the sources, which are kept ASCII.
# deck.py sets it as an em dash on the way into PowerPoint, but the text drawn
# INSIDE a plate never goes that way -- matplotlib draws it -- so the laser
# panel printed two hyphens on the sheet. Same substitution, same place in the
# pipeline: the last moment before the text is measured and set.
_DASH = re.compile(r"\s+--\s+")


def _wrap(text, n):
    import textwrap
    return textwrap.wrap(_DASH.sub("\u2014", text), n)


# ------------------------------------------------------- the four projects
def _pic(ax, path, box, ax_w_in, ax_h_in, crop=0.5, fallback=None,
         note="no photograph yet"):
    """Draw one picture into a box in axes coordinates, cropped to the box.

    The crop aspect is the box's own, so nothing is stretched -- an earlier
    version cropped to 1.62 and drew at 2.42 and smeared every horizon by half.
    A missing file draws a ruled empty frame that says so, which is the honest
    thing for a project nothing has been uploaded to.
    """
    x, y, w, h = box
    p = os.path.join(HERE, "photos", path) if path else ""
    if (not p or not os.path.exists(p)) and fallback:
        p = os.path.join(HERE, "photos", fallback)
    if p and os.path.exists(p):
        im = plt.imread(p)
        ih, iw = im.shape[0], im.shape[1]
        # The aspect the box is DRAWN at, measured from the axes it sits in --
        # not from the whole figure. Using the figure's 46.4 x 5.6 here cropped
        # a square box's picture to 3.7:1 and then stretched it back to square.
        want = (w * ax_w_in) / (h * ax_h_in)
        keep = int(round(iw / want))
        if keep < ih:
            top = int(round((ih - keep) * crop))
            im = im[top:top + keep]
        else:
            keep_w = int(round(ih * want))
            left = int(round((iw - keep_w) * 0.5))
            im = im[:, left:left + keep_w]
        ax.imshow(im, extent=(x, x + w, y, y + h), aspect="auto", zorder=1)
    else:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=PALE, edgecolor="none",
                                   zorder=1))
        ax.text(x + w / 2, y + h / 2, note, fontsize=ANNOT + 1, color=MUTE,
                ha="center", va="center", style="italic", zorder=2)
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="none", edgecolor=RULE,
                               lw=1.2, zorder=3))


FIG_W, FIG_H = 46.4, 3.90


def projects():
    """Four projects, each drawn the shape it needs.

    Four identical cards in a row is the layout primitive the poster-design
    literature names as the generated-deck signature, and forcing it here meant
    cropping the rover to a letterbox strip to match a detection grid. So: the
    detection grid stays wide because it is a strip of twelve, the rover gets a
    square with its text beside it because it is a machine and wants to be seen
    whole, and the humanoid project gets two pictures because it has two.
    """
    import textwrap
    P = D.PROJECTS
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.set_layout_engine("none")
    L, R, GAP = 0.004, 0.996, 0.009
    total = sum(p["w"] for p in P)
    span = (R - L) - GAP * (len(P) - 1)
    x = L
    # The introductions are the first prose a visitor reads, and at ANNOT+1 they
    # were the smallest type on the sheet -- under a professor's note that the
    # font is too small. They are set larger than the column body now, and the
    # plate pays for it by being shorter: the picture takes whatever height the
    # text leaves, instead of a fixed height the text then overruns.
    BLURB, NAME = AXIS + 9, AXIS + 12
    step_in = BLURB * 1.25 / 72.0
    for pr in P:
        w = span * pr["w"] / total
        ax = fig.add_axes([x, 0.010, w, 0.980])
        x += w + GAP
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        cell_w_in = w * FIG_W
        ax_h_in = 0.980 * FIG_H
        step = step_in / ax_h_in
        lay = pr.get("layout", "top")
        name_in = NAME * 1.30 / 72.0

        ax.text(0, 0.985, pr["name"], fontsize=NAME, fontweight="bold",
                color=INK, va="top")
        ytop = 1.0 - (name_in / ax_h_in)

        def wrapped(width_in):
            n = max(24, int((width_in - 0.55) * 72 / (BLURB * 0.50)))
            return textwrap.wrap(_DASH.sub("\u2014", pr["blurb"]), n)

        if lay == "side":
            # a square, so the machine is not cropped to fit a neighbour's shape
            side_in = min(cell_w_in * 0.40, 2.75, ax_h_in - name_in - 0.18)
            pw, ph = side_in / cell_w_in, side_in / ax_h_in
            _pic(ax, pr["photo"], (0.0, ytop - ph, pw, ph), cell_w_in, ax_h_in,
                 pr.get("crop", 0.5), pr.get("fallback"))
            if pr.get("photo_note"):
                ax.text(0.0, ytop - ph - 0.035, pr["photo_note"], fontsize=ANNOT - 2,
                        color=MUTE, style="italic", va="top")
            tw_in = cell_w_in * (1 - pw - 0.045)
            lines = wrapped(tw_in)
            tx = pw + 0.045
            y = ytop - 0.02
        else:
            lines = wrapped(cell_w_in if lay == "top" else cell_w_in)
            # the picture takes what the text leaves, within reason
            free = ax_h_in - name_in - 0.16 - len(lines) * step_in
            lo, hi = (1.15, 2.20) if lay == "top" else (1.30, 2.55)
            pic_in = max(lo, min(hi, free))
            ph = pic_in / ax_h_in
            if lay == "pair":
                half = (1.0 - 0.012) / 2.0
                _pic(ax, pr["photo"], (0.0, ytop - ph, half, ph), cell_w_in,
                     ax_h_in, pr.get("crop", 0.5))
                _pic(ax, pr.get("photo2", ""), (half + 0.012, ytop - ph, half, ph),
                     cell_w_in, ax_h_in, pr.get("crop", 0.5))
            else:
                _pic(ax, pr["photo"], (0.0, ytop - ph, 1.0, ph), cell_w_in,
                     ax_h_in, pr.get("crop", 0.5), pr.get("fallback"))
            tx = 0.0
            y = ytop - ph - 0.045

        for ln in lines:
            ax.text(tx, y, ln, fontsize=BLURB, color=INK, va="top")
            y -= step
        # Text past the bottom of the plate is invisible in the PNG thumbnail
        # and obvious at 48 inches. The blurb budget is what keeps this true,
        # so the plate refuses to be built when a blurb outgrows it.
        if y < -0.035:
            raise AssertionError(
                "%s: blurb overruns the plate by %.2f in -- shorten it or raise FIG_H"
                % (pr["name"], -y * ax_h_in))
    return save(fig, "q_projects",
                
                
                "Four projects, one platform. Each was opened by the same procedure, and its data is registered the same way whether it was harvested, driven in by a robot, or uploaded.")


# ------------------------------------------------------------- the model ledger
def ledger():
    """Every model this project trained, ran in the loop, or rejected.

    Drawn as the list it counts, so the headline number on the poster is whatever
    the list is. Four families in four columns, ordered by what each family was
    for rather than by score: a reader should be able to see the shape of the
    search before reading a single number.
    """
    import textwrap
    L = D.LEDGER
    fig = plt.figure(figsize=(22.4, 6.60))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(L["groups"])
    gw = 1.0 / n
    # One pitch for every column, set by the fullest one, so the four families
    # share a baseline grid instead of each finding its own.
    # Count the tag line the way it is DRAWN. The tag prints when either the
    # outcome or the n is set, but this counted only the outcome, so the
    # zero-shot column reserved ten lines for twenty and its last row -- five
    # models that returned nothing -- ran off the bottom and was never seen.
    tallest = max(sum(1 + (1 if (r[3] or r[2]) else 0) + r[0].count("\n")
                      for r in g["rows"]) for g in L["groups"])
    top, bottom = 0.735, 0.035
    pitch = (top - bottom) / tallest
    for gi, g in enumerate(L["groups"]):
        x0 = gi * gw + 0.008
        w = gw - 0.024
        ax.plot([x0, x0 + w], [0.975, 0.975], color=INK, lw=1.8,
                transform=ax.transAxes, clip_on=False)
        for k, ln in enumerate(textwrap.wrap(g["group"], 34)[:2]):
            ax.text(x0, 0.945 - k * 0.048, ln, fontsize=14, color=INK,
                    fontweight="bold", va="top")
        for k, ln in enumerate(textwrap.wrap(g["note"], 44)[:2]):
            ax.text(x0, 0.845 - k * 0.040, ln, fontsize=11.5, color=MUTE, va="top")
        y = top
        for name, score, n_, outcome in g["rows"]:
            nl = name.count("\n") + 1
            ax.text(x0, y, name, fontsize=12.5, color=INK, va="top")
            ax.text(x0 + w, y, score, fontsize=12.5, color=BLUE, va="top", ha="right")
            y -= pitch * nl
            tag = "  \u00b7  ".join(x for x in (outcome, n_) if x)
            if tag:
                ax.text(x0, y + pitch * 0.10, tag, fontsize=11, color=MUTE, va="top")
                y -= pitch
            ax.plot([x0, x0 + w], [y + pitch * 0.18, y + pitch * 0.18], color=RULE,
                    lw=0.7, transform=ax.transAxes, clip_on=False)
    return save(fig, "m_ledger",
                "%d rows in four families, which is %d distinct models: OWLv2-large and Qwen2.5-7B "
                "each appear twice because each does two jobs here. One model is deployed. The "
                "zero-shot column is a single deterministic pass per model with no repeats, scored "
                "as mAP50 on the 848-image split; the reviewer column is the 149-case dev split, "
                "and two of its arms are still running. RF-DETR's score is on a third scale and "
                "is never differenced against the rows above it: the one measured offset between "
                "two of our evaluators, 0.0239, is larger than the gap on display."
                % (L["n_rows"] + 4, L["n_models"]))


# -------------------------------------------------------- the platform, drawn
def system():
    """What the platform is, as five stages a reader can follow left to right.

    Annotated with what runs unattended and what does not, because a system
    diagram that does not say which boxes are switched off is a wish.
    """
    C = D.CENSUS
    stages = [
        ("Two ground robots", [
            "Rover  %s frames, %d drives" % ("{:,}".format(C["r241_frames"]),
                                                 C["r241_sessions"]),
            "laser robot  %d frames, %d drives" % (C["cart_frames"], C["cart_sessions"]),
            "uplink test  %d frames, %d drive" % (C["smoke_frames"], C["smoke_sessions"]),
            "%s telemetry rows, 9 streams" % "{:,}".format(C["sensor_rows"]),
            "%.0f m of GPS track" % C["gps_m"]], BLUE, "runs"),
        ("Live uplink", [
            "one API key per project",
            "JPEG frames and sensor batches",
            "a drive becomes a dataset as it",
            "opens, not after it ends"], BLUE, "runs"),
        ("Locked registry", [
            "every dataset versioned and owned",
            "holdout stems excluded at merge",
            "content-hash duplicate guard",
            "%s images, %s unique" % ("{:,}".format(D.FUNNEL["registry_labelled"]),
                                      "{:,}".format(D.FUNNEL["unique"]))], BLUE, "runs"),
        ("Collector and trainer", [
            "collect \u2192 filter \u2192 merge \u2192 train \u2192 evaluate",
            "two SLURM jobs, no person in the loop",
            "15 rounds run this way",
            "paused since 2026-08-29"], WARN, "paused"),
        ("Supervision", [
            "a model reads the run's own artifacts",
            "162 real incidents, frozen as a corpus",
            "%d models scored against it" % len(D.SUPERVISION["models"]),
            "advisory only, nothing is applied"], MUTE, "shadow"),
    ]
    fig = plt.figure(figsize=(22.4, 3.70))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(stages)
    gap = 0.022
    w = (1.0 - gap * (n - 1)) / n
    for i, (title, lines, col, state) in enumerate(stages):
        x0 = i * (w + gap)
        ax.add_patch(plt.Rectangle((x0, 0.06), w, 0.86, transform=ax.transAxes,
                                   facecolor=WHITE, edgecolor=RULE, lw=1.2, zorder=1))
        ax.add_patch(plt.Rectangle((x0, 0.875), w, 0.045, transform=ax.transAxes,
                                   facecolor=col, edgecolor="none", zorder=2))
        ax.text(x0 + 0.014, 0.835, title, fontsize=15, color=INK, fontweight="bold",
                va="top", zorder=3)
        ax.text(x0 + w - 0.014, 0.835, state, fontsize=12, color=col, va="top",
                ha="right", zorder=3)
        y = 0.700
        for ln in lines:
            ax.text(x0 + 0.014, y, ln, fontsize=12.5, color=MUTE, va="top", zorder=3)
            y -= 0.132
        if i < n - 1:
            ax.annotate("", (x0 + w + gap - 0.004, 0.50), xytext=(x0 + w + 0.004, 0.50),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=4)
    return save(fig, "n_system",
                "The path as built, from a camera in a crop row to a checkpoint on the cluster. "
                "Every stage has run. Three qualifications ride on it. The collector and trainer "
                "are paused after a stop-loss on 2026-08-29. The supervisor is advisory and "
                "nothing it says is applied. And the last link is not closed: no robot frame has "
                "entered training, because none of them is labelled yet.")


# ---------------------------------------------------------- six months, in order
def journey():
    """Eight turns, each dated, each with what it changed.

    The wrong turns are the point. Four of these eight are things that did not
    work, and two are numbers this project withdrew after checking them.
    """
    import textwrap
    J = D.JOURNEY
    fig = plt.figure(figsize=(46.4, 3.30))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(J)
    gap = 0.011
    w = (1.0 - gap * (n - 1)) / n
    ax.plot([0, 1], [0.845, 0.845], color=RULE, lw=1.4, transform=ax.transAxes,
            clip_on=False)
    for i, (month, title, what, learned) in enumerate(J):
        x0 = i * (w + gap)
        ax.plot(x0 + 0.004, 0.845, "o", color=BLUE, ms=9, transform=ax.transAxes,
                clip_on=False, zorder=3)
        ax.text(x0 + 0.018, 0.860, month, fontsize=13, color=BLUE, fontweight="bold",
                va="center")
        # matplotlib does not wrap text inside an axes the way a text box does,
        # so the wrapping is done here against the real column width: 46.4 in
        # over eight stations is 5.8 in, which is about 46 characters at 12.5 pt.
        ax.text(x0, 0.760, title, fontsize=15.5, color=INK, fontweight="bold",
                va="top")
        y = 0.640
        for ln in textwrap.wrap(what, 46):
            ax.text(x0, y, ln, fontsize=12.5, color=MUTE, va="top")
            y -= 0.094
        y = min(y - 0.050, 0.215)
        for ln in textwrap.wrap(learned, 44):
            ax.text(x0, y, ln, fontsize=12.5, color=INK, va="top", style="italic")
            y -= 0.094
    return save(fig, "t_journey",
                "Six months. Four of these eight turns are things that did not work, and two are "
                "numbers this project measured, published internally and then withdrew after "
                "checking them.")


# --------------------------------------------- s. the audit, source by source
def sources():
    """Label precision per harvested source, against the bar the gate sets.

    The funnel in (k) compresses this to "one of six". Drawn out, the six are not
    close to the bar and not close to each other, which is the part that makes
    the gate an audit result rather than a threshold chosen after the fact.
    """
    S = D.SOURCES
    labs = [a for a, _ in S["rows"]]
    vals = [v for _, v in S["rows"]]
    ys = np.arange(len(vals))[::-1]
    fig, ax = plt.subplots(figsize=(11.600, 4.30))
    ax.barh(ys, vals, height=0.56, zorder=3,
            color=[GOOD if v >= S["bar"] else WARN for v in vals])
    ax.axvline(S["bar"], color=INK, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.text(S["bar"] - 0.012, ys[0] + 0.62, "audit bar", fontsize=9.5, color=INK,
            ha="right", va="center")
    for y, v in zip(ys, vals):
        ax.text(v + 0.012, y, "%.3f" % v, va="center", fontsize=10,
                color=GOOD if v >= S["bar"] else WARN)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.12); ax.set_ylim(-0.62, len(vals) - 0.15)
    ax.set_xlabel("label precision against the audit probe")
    despine(ax, left=False)
    fig.text(0.004, 0.985, "s", fontsize=13, fontweight="bold", color=INK, va="top")
    return save(fig, "s_sources",
                "The six audited harvested sources. The same probe reads %.3f on human-labelled "
                "CottonWeedDet12, so a source at %.3f is the data and not the instrument. Only the "
                "source at the top clears the bar, and it supplies %s of the %s audited images."
                % (S["calibration"], vals[-1], "{:,}".format(D.FUNNEL["passing_images"]),
                   "{:,}".format(D.FUNNEL["audited_images"])))


# ------------------------------------------------- r. what inference can buy
def tta():
    """What test-time compute buys, against the noise bar and against its cost.

    Every arm here is scored by the same matcher as the plain baseline, so these
    are differences inside one instrument. Drawn against the seed-noise band
    because the first arm -- fusion with no extra views -- sits inside it, and a
    bar chart without that band would show it as a small gain rather than as
    nothing.
    """
    T = D.TTA
    labs = [a for a, _ in T["arms"]]
    d = [v - T["baseline"] for _, v in T["arms"]]
    ys = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(11.600, 3.93))
    ax.axvspan(-T["seed_noise"], T["seed_noise"], color=PALEBLU, zorder=0)
    ax.barh(ys, d, height=0.52, zorder=3,
            color=[WARN if v < T["seed_noise"] else BLUE for v in d])
    for y, v in zip(ys, d):
        ax.text(v + (0.0013 if v > 0 else -0.0013), y, "%+.4f" % v, va="center",
                ha="left" if v > 0 else "right", fontsize=10,
                color=BLUE if v >= T["seed_noise"] else WARN)
    ax.axvline(0, color=INK, lw=0.9, zorder=4)
    # The arm names are the y axis, not floating text: an earlier version drew
    # them inside and the negative bar ran straight through its own label.
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.text(T["seed_noise"] + 0.0013, ys[0] + 0.60, "seed noise", fontsize=9,
            color=MUTE, ha="left", va="center")
    ax.set_xlim(-0.0125, 0.0365); ax.set_ylim(-0.60, len(d) - 0.15)
    ax.set_xlabel("change in mAP$_{50\\mathdefault{-}95}$, one matcher throughout")
    despine(ax, left=False)
    # The y labels own the left third of the canvas, so a letter placed against
    # the axes reads as centred. Anchor it to the figure instead.
    fig.text(0.004, 0.985, "r", fontsize=13, fontweight="bold", color=INK, va="top")
    return save(fig, "r_tta",
                "Test-time compute on the deployable checkpoint. Fusion with no extra views is "
                "nothing; multi-scale with h-flip and a 3-seed ensemble are each worth about "
                "+0.017 and stack to +0.028. At %.2f s per image against the deployed %.1f ms "
                "this is a ceiling on what inference-time compute can buy, not a deployment "
                "option -- and it is still smaller than the +0.0714 that initialisation is worth."
                % (T["s_per_image"], T["deployed_ms"]))


# ------------------------------------------------------- p. the field fire rate
def field():
    """How often the deployed detector fires on the frames our own robots recorded.

    Figure d measures the wall against another labelled corpus. This measures it
    against the corpus the platform actually holds, which has no labels -- so the
    quantity is the fire rate, not recall, and the caption says so twice. The
    vegetation bar is what stops "there was nothing to detect" from explaining the
    answer: 1,157 of these frames are more than a third green, and the detector
    produces a box on 21 of them.
    """
    F = D.FIELD
    stages = [("frames the robots recorded", F["frames"], BLUE),
              ("more than a third vegetation", F["vegetated"], BLUE),
              ("detector fires, conf 0.25", F["fired_25"], WARN),
              ("detector fires, conf 0.40", F["fired_40"], WARN)]
    fig, ax = plt.subplots(figsize=(11.600, 2.70))
    tot = stages[0][1]
    for i, (lab, v, c) in enumerate(stages):
        ax.barh(-i, max(v / tot, 0.0), height=0.42, color=c)
        ax.text(0.0, -i + 0.44, lab, fontsize=18.5, color=INK, va="center")
        ax.text(max(v / tot, 0.0) + 0.012, -i, "{:,}".format(v), va="center",
                fontsize=11, color=c)
    ax.set_xlim(0, 1.26); ax.set_ylim(-len(stages) + 0.42, 0.78)
    ax.axis("off")
    return save(fig, "p_field",
                "The deployable checkpoint over every frame the two robots recorded, at the "
                "confidence the laser robot deploys at. There is no ground truth for these frames, so "
                "this is the rate at which the detector produces a box at all -- not recall and "
                "not precision. Vegetation is excess green over a thumbnail, so \"nothing to "
                "detect\" does not account for it: %d frames clear that bar and %d of them draw a "
                "box. %d of the %d boxes it does draw are the same class."
                % (F["vegetated"], F["veg_fired_25"], F["species"]["Purslane"], F["fired_25"]))


# ------------------------------------------------------------- l. the robots
def robots_wide():
    """The same four frames, printed full-width as an opening image.

    A data-led layout wants the field to be the first thing a reader meets, at a
    size where the crop rows and the mulch film are legible from two metres.
    """
    # Cropped to 2.35:1 so four frames fill the full width without the bands of
    # white that a 16:9 frame leaves when it is laid out four across 46 inches.
    fig, axes = plt.subplots(1, 4, figsize=(46.4, 4.40))
    shots = [("r241_row.jpg", "Rover", "along a mulched crop row"),
             ("r241_weeds.jpg", "Rover", "weeds between the rows"),
             ("lc_down.jpg", "laser robot", "down camera, work-zone overlay"),
             ("lc_front.jpg", "laser robot", "forward camera")]
    for ax, (fn, who, what), let in zip(axes, shots, "abcd"):
        im = plt.imread(os.path.join(HERE, "photos", fn))
        h, wpx = im.shape[0], im.shape[1]
        keep = int(round(wpx / 2.35))
        if keep < h:
            top = (h - keep) // 2
            im = im[top:top + keep]
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(RULE); sp.set_linewidth(1.2)
        ax.text(0.022, 0.950, let, transform=ax.transAxes, fontsize=22,
                fontweight="bold", color=INK, va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.24", fc=WHITE, ec="none", alpha=0.88))
        ax.set_xlabel("%s   %s" % (who, what), fontsize=17, color=MUTE, labelpad=8)
    return save(fig, "w_robots",
                "Four frames as recorded, unenhanced and unlabelled. The rectangle in (c) is the "
                "cart's fixed work-zone overlay. It is drawn on every frame, including the ones "
                "with no vegetation in them.")


def robots():
    """Four frames off the platform, as recorded.

    A poster about a field platform has to show that the field half is real
    equipment on real ground, so these are frames straight out of the archive.
    Nothing is enhanced and nothing is labelled.

    The rectangle in the laser robot's down camera is a FIXED WORK-ZONE OVERLAY, not a
    detection. An earlier version of this caption called it "the machine's own
    detector drawing on its own video", which a reader could falsify in one
    click: it sits at the same pixels, at the same size, on frames with no
    vegetation in them at all. Checked 2026-09-12 by pulling the lowest-
    vegetation down frame on the platform (excess green 0.018, an indoor floor)
    beside the highest (0.263, grass) -- same rectangle on both.
    """
    P = D.CENSUS
    shots = [("r241_row.jpg", "Rover", "along a mulched crop row"),
             ("r241_weeds.jpg", "Rover", "weeds between the rows"),
             ("lc_down.jpg", "laser robot", "down camera, work-zone overlay"),
             ("lc_front.jpg", "laser robot", "forward camera")]
    fig, axes = plt.subplots(1, 4, figsize=(11.600, 2.60))
    for ax, (fn, who, what), let in zip(axes, shots, "lmno"):
        img = plt.imread(os.path.join(HERE, "photos", fn))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(RULE); sp.set_linewidth(0.8)
        # On a light plate: a bare letter disappears into a sunlit frame.
        ax.text(0.028, 0.945, let, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK, va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.22", fc=WHITE, ec="none", alpha=0.86))
        ax.set_xlabel("%s   %s" % (who, what), fontsize=9, color=MUTE, labelpad=5)
    return save(fig, "l_robots",
                "Frames as recorded, unenhanced. The Rover has sent %s frames at %s and %g Hz, "
                "the laser robot %d, and %d of its were taken in a field. None of the %s is "
                "labelled. The rectangle in the down camera is the laser robot's fixed work-zone overlay. "
                "It is drawn on every frame, including the ones with no vegetation in them."
                % ("{:,}".format(P["r241_frames"]), P["r241_res"], P["r241_hz"],
                   P["cart_frames"], P["cart_field_frames"],
                   "{:,}".format(P["frames"])))


# ---------------------------------------------------------------- f. rounds
def rounds():
    R = D.ROUNDS
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(22.400, 5.10),
                                  gridspec_kw={"width_ratios": [1.75, 1]})
    xs = np.arange(1, len(R["map"]) + 1)
    ax.axvspan(R["frozen_from"] - 0.4, len(xs) + 0.4, color=PALEBLU, zorder=0)
    ax.plot(xs, R["map"], "o-", color=BLUE, ms=4, zorder=3)
    ax.plot(xs, R["last"], "s--", color=MUTE, ms=3.4, zorder=3)
    ax.text(len(xs) + 0.1, R["map"][-1], " reported\n (best epoch)", fontsize=9.5,
            color=BLUE, va="center")
    ax.text(len(xs) + 0.1, R["last"][-1] - 0.004, " last epoch", fontsize=9.5,
            color=MUTE, va="center")
    ax.text(R["frozen_from"] + 0.25, 0.6065,
            "training corpus identical from here on", fontsize=9.5, color=MUTE,
            va="top")
    ax.set_xlabel("unattended round"); ax.set_ylabel("mAP$_{50\\mathdefault{-}95}$")
    ax.set_xlim(0.3, len(xs) + 4.6); ax.set_ylim(0.542, 0.616)
    # Integer rounds only: the automatic locator labelled this 2.5 / 5.0 / ... /
    # 17.5, and round 2.5 does not exist while round 17.5 is outside the
    # experiment. The spine is bounded to the data for the same reason -- the
    # right-hand gutter exists to hold the two series labels, not to assert that
    # rounds 16 to 19 were run.
    ax.set_xticks([1, 5, 10, 15])
    ax.spines["bottom"].set_bounds(1, len(xs))
    despine(ax); hline_grid(ax); panel(ax, "f")

    # Three bars, not two: bar 0 vs bar 1 is the matched pair that isolates the
    # warm start (same data, same 30 epochs, same complete cosine, same seed), and
    # bar 2 is the recipe the campaign actually ran, which is the only arm this
    # project has repeated over seeds. Quoting the gap against a bar drawn from a
    # different recipe is what the earlier draft of this panel did wrong.
    # Positions, not bar lengths. Two of these three arms are single runs, and a
    # bar drawn from a baseline of 0.540 gave the fresh-start arm 3.4x the length
    # of the warm-start arm for a difference of 1.05x, with a zero-height error
    # bar sitting on top of it. A dot carries the same value honestly on a
    # cropped scale, and the crop is what makes a 0.029 effect visible at all.
    vals = [R["cold"], R["warm"], R["recipe_mean"]]
    errs = [0.0, 0.0, R["recipe_sd"]]
    ax2.plot(np.full(len(R["recipe_seeds"]), 2) + np.linspace(-0.10, 0.10, 3),
             R["recipe_seeds"], "o", ms=4.0, mfc="none", mec=GREY, mew=0.9,
             zorder=2)
    ax2.errorbar([0, 1, 2], vals, yerr=errs, fmt="o", ls="none", ms=8, color=INK,
                 elinewidth=1.3, capsize=0, zorder=4)
    for i, v in enumerate(vals):
        ax2.text(i + 0.13, v, sig(v, R["recipe_sd"]), ha="left", va="center",
                 fontsize=ANNOT, color=INK)
    ax2.plot([0, 0, 1, 1], [0.5905, 0.5918, 0.5918, 0.5905], lw=0.7, color=MUTE)
    ax2.text(0.5, 0.5930, "%s,  %.1f \u03c3" % (sig(R["chain_effect"], R["recipe_sd"],
             signed=True), R["chain_sigma"]), ha="center", fontsize=ANNOT, color=INK)
    ax2.plot([1, 1, 2, 2], [0.5726, 0.5739, 0.5739, 0.5726], lw=0.7, color=MUTE)
    ax2.text(1.5, 0.5751, "%s,  %.1f \u03c3" % (sig(R["sched_effect"], R["recipe_sd"],
             signed=True), abs(R["sched_sigma"])), ha="center", fontsize=ANNOT,
             color=MUTE)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["fresh\nstart", "warm\nstart", "warm start\n+ clock cap"],
                        fontsize=TICK)
    ax2.set_xlim(-0.55, 2.55)
    ax2.set_ylim(0.540, 0.600); ax2.set_ylabel("mAP$_{50\\mathdefault{-}95}$")
    # Under the tick labels, not inside the bars, where the fill fought the text.
    for i, n in enumerate((1, 1, 3)):
        ax2.annotate("n = %d" % n, (i, 0), xytext=(0, -48),
                     textcoords="offset points", xycoords=("data", "axes fraction"),
                     ha="center", fontsize=ANNOT, color=MUTE, annotation_clip=False)
    despine(ax2, bottom=False); hline_grid(ax2); panel(ax2, "g", inside=True)
    return save(fig, "f_rounds",
                "(f) Fifteen unattended rounds on a fixed holdout. (g) Round 15's exact dataset, "
                "three ways. The first two marks differ only in the starting weights, so their gap "
                "is the warm-start chain; the third is the recipe the campaign ran, whose "
                "wall-clock cap truncates the cosine, with its three seeds drawn behind it. Sigma throughout is that recipe's own seed spread, "
                "0.0040 over three seeds, propagated to the difference being quoted. Completing "
                "the schedule is not the fix -- the second mark sits below the third.")


# ---------------------------------------------------------------- h. field drive
def drive():
    trk = json.load(open(os.path.join(HERE, "hero_track.json")))
    pts = trk["board"]
    lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
    mx = 111320.0 * np.cos(np.radians(sum(lats) / len(lats)))
    xs = [(lo - lons[0]) * mx for lo in lons]
    ys = [(la - lats[0]) * 110540.0 for la in lats]
    rows = list(csv.DictReader(open(os.path.join(HERE, "hero_imu.csv"))))
    t0 = float(rows[0]["timestamp"])
    t = [float(r["timestamp"]) - t0 for r in rows]
    hd = np.degrees(np.unwrap(np.radians([float(r["heading"]) for r in rows])))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.600, 4.40),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    ax.plot(xs, ys, "-", color=BLUE, lw=1.2, zorder=3)
    ax.plot(xs, ys, ".", color=BLUE, ms=2.4, alpha=0.5, zorder=4)
    ax.plot(xs[0], ys[0], "o", color=GOOD, ms=7, zorder=5)
    ax.plot(xs[-1], ys[-1], "s", color=WARN, ms=7, zorder=5)
    ax.annotate("start", (xs[0], ys[0]), xytext=(6, 6),
                textcoords="offset points", fontsize=9.5, color=GOOD)
    ax.annotate("end", (xs[-1], ys[-1]), xytext=(6, -12),
                textcoords="offset points", fontsize=9.5, color=WARN)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("metres east"); ax.set_ylabel("metres north")
    despine(ax); panel(ax, "h")
    ax.text(0.02, 0.04, "%.0f m in %.0f s" % (trk["path_m"], trk["seconds"]),
            transform=ax.transAxes, fontsize=10, color=INK)

    ax2.plot(t, hd, "-", color=INK, lw=0.8)
    ax2.set_xlabel("seconds"); ax2.set_ylabel("heading, degrees (unwrapped)")
    despine(ax2); hline_grid(ax2); panel(ax2, "i", dx=-0.22)
    ax2.text(0.97, 0.06, "%s samples, %.0f Hz" % ("{:,}".format(len(rows)),
                                                  len(rows) / (t[-1] or 1)),
             transform=ax2.transAxes, fontsize=10, color=MUTE, ha="right")
    return save(fig, "h_drive",
                "The Rover, 2026-08-29: one 213 s pass through a crop plot. (h) GPS track from the "
                "board fix. (i) IMU heading over the same pass.")


# ---------------------------------------------------------------- j. per species
def species():
    """Twelve classes, ordered, with the count each score rests on.

    Grey bars with the two ends in the spot colour: twelve saturated bars all
    the same hue spend colour on nothing, and a reader cannot tell from them
    which class is the problem. The n beside each bar is the point -- the two
    weakest classes carry 349 and 320 instances, so neither is a small-sample
    artefact, and the strongest carries 91, so it partly is.
    """
    rows = sorted(D.PER_SPECIES, key=lambda r: r[1])
    lo, hi = rows[0][1], rows[-1][1]
    fig, ax = plt.subplots(figsize=(11.600, 6.60))
    y = np.arange(len(rows))
    cols = [BLUE if (r[1] == lo or r[1] == hi) else GREY for r in rows]
    ax.barh(y, [r[1] for r in rows], color=cols, height=0.62)
    for i, (cls, v, n) in enumerate(rows):
        ax.text(v - 0.012, i, "%.3f" % v, va="center", ha="right",
                fontsize=ANNOT - 1.5, color=WHITE)
        ax.text(v + 0.014, i, "n = %d" % n, va="center", fontsize=ANNOT - 2.5,
                color=MUTE)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=ANNOT - 1)
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("mAP$_{50\\mathdefault{-}95}$")
    despine(ax, left=False); ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.annotate("", xy=(1.075, 0), xytext=(1.075, len(rows) - 1),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(1.092, (len(rows) - 1) / 2.0, "%.3f" % (hi - lo), rotation=90,
            va="center", ha="left", fontsize=ANNOT - 1, color=INK)
    return save(fig, "j_species",
                "Per-species mAP50-95 of the deployed checkpoint over the sealed holdout, from the "
                "same inference pass as the detection plate. n is ground-truth instances and sums "
                "to %s. The spread between the best and worst class is %.3f, about sixty times the "
                "0.0040 spread between training seeds; the two weakest classes carry %d and %d "
                "instances, so neither is a small-sample artefact."
                % ("{:,}".format(sum(r[2] for r in rows)), hi - lo, rows[0][2], rows[1][2]))


# ---------------------------------------------------------------- k. funnel
def funnel():
    F = D.FUNNEL
    fig, ax = plt.subplots(figsize=(11.600, 4.15))
    stages = [("harvested and labelled", F["registry_labelled"], BLUE),
              ("unique after dedup", F["unique"], BLUE),
              ("audited for label quality", F["audited_images"], MUTE),
              ("clears the 0.90 bar", F["passing_images"], WARN)]
    tot = stages[0][1]
    for i, (lab, v, c) in enumerate(stages):
        w = v / tot
        ax.barh(-i, w, height=0.40, color=c)
        ax.text(0.0, -i + 0.44, lab, fontsize=18.5, color=INK, va="center")
        ax.text(w + 0.015, -i, "{:,}".format(v), va="center", fontsize=19.5, color=c)
    ax.set_xlim(0, 1.26); ax.set_ylim(-len(stages) + 0.45, 0.75)
    ax.axis("off")
    return save(fig, "k_funnel",
                "One of six audited harvested sources clears the label-precision bar. The audit "
                "probe reads 1.000 on human-labelled CottonWeedDet12, so the low scores are the data.")


if __name__ == "__main__":
    print("rendering into", OUT)
    for fn in (fam, vlm, ladder, wall, supervision, rounds, robots, drive,
               species, funnel, field, tta, sources, ledger, system,
               journey, robots_wide, projects, detect_grid, r241_grid, cart_grid, algorithm,
               router):
        fn()
    json.dump(CAPS, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    print("\n%d figures" % len(CAPS))
