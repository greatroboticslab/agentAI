#!/usr/bin/env python3
"""Poster figures, drawn to journal convention. One command:

    python3 docs/poster/figures.py

No figure carries a title — the claim belongs in the caption, which is set in the
poster beside it. Each function returns (filename, caption) and the captions are
printed so the poster text cannot drift from the figure.
"""
import os, sys, json, csv, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style import (INK, NAVY, BLUE, PALE, PALEBLU, RULE, MUTE, WHITE, GOOD, WARN,
                   GREY, panel, despine, hline_grid, PLACED, sig, wilson,
                   TICK, AXIS, ANNOT, LETTER)
import poster_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig")
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
    fig, ax = plt.subplots(figsize=(10.925, 2.80))
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
    panel(ax, "a", figx=0.004)
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
    fig, ax = plt.subplots(figsize=(10.925, 4.20))
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
    ax.set_xlim(0, 1.04); ax.set_xlabel("mAP$_{50}$, 848-image cwd12 test split")
    despine(ax, left=False)
    ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    panel(ax, "b", dx=-0.46)
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.925, 3.30))
    ax.errorbar(xs, A, yerr=Ae, fmt="o-", color=BLUE, capsize=4, elinewidth=1.0, zorder=3)
    if B:
        ax.plot(xs, B, "s--", color=WARN, markerfacecolor=WHITE, zorder=3)
    ax.text(xs[1] + 0.08, A[1] + 0.0022, "class per source dataset", ha="left",
            va="bottom", fontsize=9.5, color=BLUE)
    ax.text(xs[-1] - 0.05, B[-1] + 0.0026, "one shared class", ha="right",
            va="bottom", fontsize=9.5, color=WARN)
    ax.annotate("", xy=(3.28, A[0]), xytext=(3.28, A[-1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.text(3.36, (A[0] + A[-1]) / 2, "−0.0189\n8.2 σ", fontsize=9.5, color=INK,
            va="center")
    ax.set_xlim(-0.35, 4.15)
    ax.set_ylabel("mAP$_{50\\mathdefault{-}95}$")
    ax.text(0.055, 0.985, "CottonWeedDet12 holdout", transform=ax.transAxes,
            fontsize=10, color=MUTE, va="top")
    ax.set_ylim(0.818, 0.872)

    a2 = [X["armA"][k] for k in rungs]; b2 = [X["armB"][k] for k in rungs]
    ax2.plot(xs, a2, "o-", color=BLUE, zorder=3)
    ax2.plot(xs, b2, "s--", color=WARN, markerfacecolor=WHITE, zorder=3)
    ax2.set_ylabel("mAP$_{50\\mathdefault{-}95}$")
    ax2.text(0.055, 0.985, "ImageWeeds, class-agnostic", transform=ax2.transAxes,
             fontsize=10, color=MUTE, va="top")
    ax2.set_ylim(0.055, 0.118)
    ax2.set_xlim(-0.35, 3.35)

    for a_, lets in ((ax, "c"), (ax2, "d")):
        a_.set_xticks(xs)
        a_.set_xticklabels(["core\nalone", "+5k", "+15k", "+40k"], fontsize=10)
        despine(a_); hline_grid(a_); panel(a_, lets, inside=True)
    ax.set_xlabel("harvested images added", labelpad=14)
    ax2.set_xlabel("harvested images added", labelpad=14)
    for i, n in enumerate(An):
        ax.annotate("n=%d" % n, (xs[i], 0), xytext=(0, -30),
                    textcoords="offset points", xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8.5, color=MUTE, annotation_clip=False)
    return save(fig, "c_ladder",
                "(c) Twelve times the training data costs 0.0189, three seeds at every rung. "
                "(d) The same eight checkpoints on a second dataset, class-agnostic, nothing "
                "of which it saw in training and 0 images excluded by the leak check; the "
                "series labels in (c) apply to (d). Both exams fall, so the narrow metric is "
                "not what makes the ladder drop.")


# ---------------------------------------------------------------- d. the wall
def wall():
    W = D.WALL
    a = float(W["in_domain"].split("±")[0]); ae = float(W["in_domain"].split("±")[1])
    b = float(W["out_domain"].split("±")[0]); be = float(W["out_domain"].split("±")[1])
    fig, ax = plt.subplots(figsize=(11.600, 4.20))
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.925, 5.40), sharey=True)

    # ---- (e) detection against false alarms ---------------------------------
    # Square, so the chance diagonal is a true 45 degrees and vertical distance
    # above it -- the only thing that line exists to show -- can be read. At the
    # earlier 2.9x compression it sat at 19 degrees and the 14 B looked far above
    # a line it is barely above.
    ax.set_box_aspect(0.88)
    ax.plot([0, 0.95], [0, 0.95], color=MUTE, lw=0.9, zorder=1)
    ax.text(0.855, 0.865, "chance", fontsize=ANNOT, color=MUTE, rotation=45,
            ha="center", va="center", rotation_mode="anchor")
    ax.axhline(S["ceiling"], color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.955, S["ceiling"] - 0.024, "ceiling for a rules-only arm",
            fontsize=ANNOT, color=MUTE, ha="right", va="top")

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
    ax.annotate("12 deterministic rules", (a0p["fa"], a0p["r"]),
                textcoords="offset points", xytext=(12, 6), ha="left",
                fontsize=ANNOT, color=MUTE)
    # The watchdog returned no decision on any of 149 cases, so it has no
    # coordinate on either axis. It sits in the margin rather than at an
    # arithmetic (0, 0) that would read as a measurement.
    ax.plot(-0.045, -0.045, "s", color=GREY, ms=7, clip_on=False, zorder=5)
    ax.annotate("scripted watchdog\nno decision on any of %d" % S["a0_cases"],
                (-0.045, -0.045), textcoords="offset points", xytext=(13, 0),
                ha="left", va="center", fontsize=ANNOT, color=GREY,
                annotation_clip=False,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=WHITE)])

    # Short labels, placed tight to their own marks. Three of the four arms sit
    # inside a false-alarm band 0.12 wide, so anything longer than a size and a
    # denominator collides with a neighbour's interval.
    style = {"Qwen2.5-7B":    (WARN, (0.430, 0.875), "left", "bottom"),
             "Qwen3-14B":     (BLUE, (0.660, 0.760), "left", "bottom"),
             "Qwen3.8-27B":   (NAVY, (0.305, 0.440), "left", "center"),
             "GLM-4.7-Flash": (GOOD, (0.010, 0.880), "left", "bottom")}
    for m in S["models"]:
        c, (lx, ly), ha, va = style[m["name"]]
        p2 = pt(m["l2"])
        mark(p2, p2["fa"], p2["r"], c)
        lab = "%s   n = %d" % (m["size"], p2["n"])
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
                                            connectionstyle="arc3,rad=-0.18"),
                            zorder=4)
                lab = "%s   n = %d" % (m["size"], p3["n"])
            else:
                ax.plot([p2["fa"], p3["fa"]], [p2["r"], p3["r"]], ls=(0, (2, 2)),
                        lw=1.0, color=c, zorder=4)
                lab = "%s   n = %d" % (m["size"], p3["n"])
        ax.text(lx, ly, lab, ha=ha, va=va, fontsize=ANNOT, color=c,
                linespacing=1.25,
                path_effects=[pe.withStroke(linewidth=2.2, foreground=WHITE)])

    ax.text(0.955, 0.045, "dashed: the two ends were scored on different cases",
            fontsize=ANNOT, color=MUTE, ha="right", va="bottom")
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
    ax2.set_xticklabels([m["size"] for m in S["models"]], fontsize=TICK)
    ax2.set_xlim(-0.5, len(xs) - 0.10)
    ax2.set_xlabel("reviewer size")
    despine(ax2); hline_grid(ax2); panel(ax2, "f")

    g7 = pt("L3@qwen2.5:7b"); g27 = pt("L3@qwen3.8:27b")
    d27 = pt("L3@qwen3.8:27b")["r"] - pt("L2@qwen3.8:27b")["r"]
    d14 = pt("L3@qwen3:14b")["r"] - pt("L2@qwen3:14b")["r"]
    return save(fig, "e_supervision",
                "(e) Every arm on one frozen corpus of real incidents from this project's own "
                "history, scored by the project's own scorer. Each mark carries its own "
                "denominator: two arms are still running and stand on 57 and 39 incidents against "
                "the others' 112 to 114, and bars are 95%% Wilson intervals. Open circle: the model "
                "reads raw artifact excerpts; the arrow runs to the same model given a retrieval "
                "round over them, and is drawn solid only where both ends were scored on the same "
                "cases. (f) The same arms, asking how much of that detection quotes a line that "
                "resolves in the artifact. The 7 B has the highest recall on the page and %.2f of "
                "it is unevidenced; the 27 B has no gap at all. Recall alone would have cleared the "
                "size class this campaign's own reviewer ran on for a week. The intervals are "
                "wide and they overlap: on this corpus the ordering among the model arms is not "
                "established, and the grounded gap is."
                % (g7["r"] - g7["g"]))


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
    fig, ax = plt.subplots(figsize=(11.600, 3.40))
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
    fig, ax = plt.subplots(figsize=(10.925, 2.90))
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
    fig, ax = plt.subplots(figsize=(11.600, 3.00))
    tot = stages[0][1]
    for i, (lab, v, c) in enumerate(stages):
        ax.barh(-i, max(v / tot, 0.0), height=0.42, color=c)
        ax.text(0.0, -i + 0.40, lab, fontsize=10, color=INK, va="center")
        ax.text(max(v / tot, 0.0) + 0.012, -i, "{:,}".format(v), va="center",
                fontsize=11, color=c)
    ax.set_xlim(0, 1.26); ax.set_ylim(-len(stages) + 0.42, 0.78)
    ax.axis("off"); panel(ax, "p", dx=-0.02, dy=1.10)
    return save(fig, "p_field",
                "The deployable checkpoint over every frame the two robots recorded, at the "
                "confidence the cart deploys at. There is no ground truth for these frames, so "
                "this is the rate at which the detector produces a box at all -- not recall and "
                "not precision. Vegetation is excess green over a thumbnail, so \"nothing to "
                "detect\" does not account for it: %d frames clear that bar and %d of them draw a "
                "box. %d of the %d boxes it does draw are the same class."
                % (F["vegetated"], F["veg_fired_25"], F["species"]["Purslane"], F["fired_25"]))


# ------------------------------------------------------------- l. the robots
def robots():
    """Four frames off the platform, as recorded.

    A poster about a field platform has to show that the field half is real
    equipment on real ground, so these are frames straight out of the archive.
    Nothing is enhanced and nothing is labelled.

    The rectangle in the cart's down camera is a FIXED WORK-ZONE OVERLAY, not a
    detection. An earlier version of this caption called it "the cart's own
    detector drawing on its own video", which a reader could falsify in one
    click: it sits at the same pixels, at the same size, on frames with no
    vegetation in them at all. Checked 2026-09-12 by pulling the lowest-
    vegetation down frame on the platform (excess green 0.018, an indoor floor)
    beside the highest (0.263, grass) -- same rectangle on both.
    """
    P = D.PLATFORM
    shots = [("r241_row.jpg", "robot 241", "along a mulched crop row"),
             ("r241_weeds.jpg", "robot 241", "weeds between the rows"),
             ("lc_down.jpg", "laser cart", "down camera, work-zone overlay"),
             ("lc_front.jpg", "laser cart", "forward camera")]
    fig, axes = plt.subplots(1, 4, figsize=(11.600, 2.30))
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
                "Frames as recorded, unenhanced. Robot 241 has sent %s frames at %s and %g Hz, "
                "the laser cart %d, and %d of the cart's were taken in a field. None of the %s is "
                "labelled. The rectangle in the down camera is the cart's fixed work-zone overlay. "
                "It is drawn on every frame, including the ones with no vegetation in them."
                % ("{:,}".format(P["r241_frames"]), P["r241_res"], P["r241_hz"],
                   P["lasercar_frames"], P["lasercar_field_frames"],
                   "{:,}".format(P["total_frames"])))


# ---------------------------------------------------------------- f. rounds
def rounds():
    R = D.ROUNDS
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(22.400, 4.00),
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.600, 4.20),
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
                "Robot 241, 2026-08-29: one 213 s pass through a crop plot. (h) GPS track from the "
                "board fix. (i) IMU heading over the same pass.")


# ---------------------------------------------------------------- j. per species
def species():
    fd = json.load(open(os.path.join(HERE, "figures_data.json")))
    rows = sorted(fd["per_species_yolo11n_val"]["rows"], key=lambda r: r["map50_95"])
    fig, ax = plt.subplots(figsize=(11.600, 5.20))
    y = np.arange(len(rows))
    ax.barh(y, [r["map50_95"] for r in rows], color=BLUE, height=0.62)
    for i, r in enumerate(rows):
        ax.text(r["map50_95"] - 0.012, i, "%.3f" % r["map50_95"], va="center",
                ha="right", fontsize=9.5, color=WHITE)
    ax.set_yticks(y); ax.set_yticklabels([r["cls"] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 1.0); ax.set_xlabel("mAP$_{50\\mathdefault{-}95}$")
    despine(ax, left=False); ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    panel(ax, "j", dx=-0.40)
    return save(fig, "j_species",
                "Per-species performance of the deployable checkpoint. The spread across the twelve "
                "classes is 0.214, roughly seventy times the seed noise.")


# ---------------------------------------------------------------- k. funnel
def funnel():
    F = D.FUNNEL
    fig, ax = plt.subplots(figsize=(11.600, 3.00))
    stages = [("harvested and labelled", F["registry_labelled"], BLUE),
              ("unique after dedup", F["unique"], BLUE),
              ("audited for label quality", F["audited_images"], MUTE),
              ("clears the 0.90 bar", F["passing_images"], WARN)]
    tot = stages[0][1]
    for i, (lab, v, c) in enumerate(stages):
        w = v / tot
        ax.barh(-i, w, height=0.42, color=c)
        ax.text(0.0, -i + 0.40, lab, fontsize=10, color=INK, va="center")
        ax.text(w + 0.012, -i, "{:,}".format(v), va="center", fontsize=11, color=c)
    ax.set_xlim(0, 1.26); ax.set_ylim(-len(stages) + 0.45, 0.75)
    ax.axis("off"); panel(ax, "k", dx=-0.02, dy=1.10)
    return save(fig, "k_funnel",
                "One of six audited harvested sources clears the label-precision bar. The audit "
                "probe reads 1.000 on human-labelled cwd12, so the low scores are the data.")


if __name__ == "__main__":
    print("rendering into", OUT)
    for fn in (fam, vlm, ladder, wall, supervision, rounds, robots, drive,
               species, funnel, field, tta, sources):
        fn()
    json.dump(CAPS, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    print("\n%d figures" % len(CAPS))
