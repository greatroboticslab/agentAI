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
                   GREY, panel, despine, hline_grid)
import poster_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)
CAPS = {}


def save(fig, name, caption):
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p)
    plt.close(fig)
    CAPS[name] = caption
    print("%-16s %s" % (name, caption[:118]))
    return p


# ---------------------------------------------------------------- a. families
def fam():
    rows = [("YOLO11n\nCOCO-pretrained", 0.8755, 0.0029, BLUE),
            ("Mamba-YOLO-T\nrandom init", 0.8266, 0.0064, MUTE),
            ("YOLO11n\nrandom init", 0.8041, 0.0028, GREY)]
    fig, ax = plt.subplots(figsize=(5.2, 2.45))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], width=0.56,
           color=[r[3] for r in rows], capsize=4, error_kw=dict(lw=1.0, ecolor=INK))
    for i, r in enumerate(rows):
        ax.text(i, r[1] + r[2] + 0.004, "%.4f" % r[1], ha="center",
                fontsize=11, color=INK)
    # the two differences, as brackets
    def bracket(i, j, y, txt):
        ax.plot([i, i, j, j], [y - 0.004, y, y, y - 0.004], lw=0.9, color=INK)
        ax.text((i + j) / 2, y + 0.002, txt, ha="center", fontsize=10.5, color=INK)
    bracket(0, 2, 0.895, "+0.0714")
    bracket(1, 2, 0.871, "+0.0225")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(0.78, 0.915); ax.set_ylabel("mAP$_{50-95}$")
    despine(ax); hline_grid(ax); panel(ax, "a")
    for i in x:
        ax.annotate("n = 3", (i, 0), xytext=(0, -34), textcoords="offset points",
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=9, color=MUTE, annotation_clip=False)
    return save(fig, "a_families",
                "Three detector families on the CottonWeedDet12 holdout, three seeds each. How the "
                "network is initialised is worth three times what the architecture is worth.")


# ---------------------------------------------------------------- b. zero-shot
def vlm():
    import json as _j
    fd = _j.load(open(os.path.join(HERE, "figures_data.json")))
    rows = [r for r in fd["benchmark_cwd12_map50"] if r.get("map50") is not None]
    rows = sorted(rows, key=lambda r: r["map50"])
    fig, ax = plt.subplots(figsize=(4.8, 2.62))
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 2.55))
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
    ax.set_ylabel("mAP$_{50-95}$")
    ax.text(0.055, 0.985, "CottonWeedDet12 holdout", transform=ax.transAxes,
            fontsize=10, color=MUTE, va="top")
    ax.set_ylim(0.818, 0.872)

    a2 = [X["armA"][k] for k in rungs]; b2 = [X["armB"][k] for k in rungs]
    ax2.plot(xs, a2, "o-", color=BLUE, zorder=3)
    ax2.plot(xs, b2, "s--", color=WARN, markerfacecolor=WHITE, zorder=3)
    ax2.set_ylabel("mAP$_{50-95}$")
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
    fig.tight_layout(w_pad=3.0)
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
    fig, ax = plt.subplots(figsize=(4.6, 2.80))
    ax.errorbar([0, 1], [a, b], yerr=[ae, be], fmt="o-", color=INK,
                capsize=4, elinewidth=1.0, zorder=3)
    ax.annotate("%.4f" % a, (0, a), xytext=(8, 4), textcoords="offset points",
                fontsize=11.5, color=INK)
    # Below-left of the endpoint: above it the label sat on its own marker.
    ax.annotate("%.4f" % b, (1, b), xytext=(-11, -12), textcoords="offset points",
                ha="right", va="top", fontsize=11.5, color=WARN)
    ax.annotate("", xy=(0.5, a), xytext=(0.5, b),
                arrowprops=dict(arrowstyle="<->", color=MUTE, lw=0.9))
    ax.text(0.545, (a + b) / 2, "−0.773", fontsize=12, color=INK, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CottonWeedDet12\nheld out", "ImageWeeds\n3,208 images"],
                       fontsize=10)
    ax.set_xlim(-0.42, 1.42); ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP$_{50-95}$, class-agnostic")
    despine(ax); hline_grid(ax); panel(ax, "d", dx=-0.25)
    ax.annotate("n = 3 both ends", (0.5, 0.045), ha="center", fontsize=9, color=MUTE)
    return save(fig, "d_wall",
                "The same three checkpoints and the same matcher on both sides, so the evaluator "
                "offset cancels. A detector trained on one field does not transfer to another.")


# ---------------------------------------------------------------- e. supervision
def supervision():
    """What each tier of supervision catches, and what the retrieval tier buys.

    Every point is the project's own scorer re-reading committed verdicts, so the
    arms are comparable to each other and to nothing else. Each model is drawn as
    an arrow, not a dot: the tail is the tier that reads raw artifact excerpts and
    the head is the same model given a retrieval round over the same artifacts.
    That arrow is the only thing on this poster that measures the tiering itself.
    """
    S = D.SUPERVISION
    A = S["table"]["arms"]

    def pt(key):
        a = A[key]
        return (a["false_alarm_rate"]["v"], a["detection_recall"]["v"],
                a["detection_recall"]["n"], a["counts"]["cases"])

    fig, ax = plt.subplots(figsize=(5.2, 2.85))
    ax.plot([0, 0.70], [0, 0.70], color=RULE, lw=0.8, zorder=0)
    ax.text(0.455, 0.425, "chance", fontsize=9, color=GREY, rotation=33, ha="right")
    ax.axhline(S["ceiling"], color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.020, S["ceiling"] - 0.028, "ceiling for any rules-only arm",
            fontsize=9.5, color=MUTE, ha="left", va="top")

    fa, rec, _, _ = pt("A0p")
    ax.plot(fa, rec, "^", color=MUTE, ms=9, zorder=4)
    ax.annotate("Deterministic signals", (fa, rec), textcoords="offset points",
                xytext=(11, 6), ha="left", fontsize=10.5, color=MUTE)
    ax.plot(0, 0, "s", color=GREY, ms=9, zorder=4)
    ax.annotate("Scripted watchdog\n%d of %d undecidable"
                % (S["a0_undecidable"], S["a0_cases"]), (0, 0),
                textcoords="offset points", xytext=(11, -1), ha="left", va="center",
                fontsize=10.5, color=GREY)

    # Labels are placed in data coordinates, not as offsets from the marker: the
    # three models sit close enough that offset labels stacked on one another.
    style = {"Qwen3-14B":     (WARN, (0.621, 0.700), "center", "bottom"),
             "Qwen3.8-27B":   (BLUE, (0.272, 0.702), "left", "center"),
             "GLM-4.7-Flash": (GOOD, (0.183, 0.821), "right", "center")}
    for m in S["models"]:
        c, (lx, ly), ha, va = style[m["name"]]
        x2, y2, n2, cases2 = pt(m["l2"])
        ax.plot(x2, y2, "o", color=c, ms=9, markerfacecolor=WHITE,
                markeredgewidth=1.7, zorder=4)
        if m["l3"]:
            x3, y3, n3, cases3 = pt(m["l3"])
            ax.annotate("", (x3, y3), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3,
                                        shrinkA=7, shrinkB=7), zorder=3)
            ax.plot(x3, y3, "o", color=c, ms=9, zorder=4)
            partial = min(cases2, cases3)
        else:
            partial = cases2
        note = "" if partial >= 149 else "\n%d of 149 so far" % partial
        ax.text(lx, ly, m["name"] + note, ha=ha, va=va, fontsize=10.5, color=c)

    ax.set_xlim(-0.035, 0.72); ax.set_ylim(-0.06, 0.95)
    ax.set_xlabel("false-alarm rate  (33 control cases)")
    ax.set_ylabel("detection recall  (n = 116)")
    despine(ax); hline_grid(ax); panel(ax, "e", inside=True)
    d14 = pt("L3@qwen3:14b")[1] - pt("L2@qwen3:14b")[1]
    d27 = pt("L3@qwen3.8:27b")[1] - pt("L2@qwen3.8:27b")[1]
    return save(fig, "e_supervision",
                "Every arm on one frozen corpus of real incidents from this project's own history "
                "-- 116 incidents and 33 controls -- scored by the project's own scorer, which "
                "requires the finding to be about the incident and not merely that the arm raised "
                "something. Up and to the left is better. Open circle: the model reads raw "
                "artifact excerpts. Filled: the same model with a retrieval round over the same "
                "artifacts. The retrieval tier is worth %+.3f recall to the 27B and %+.3f to the "
                "14B, so tiering pays where the model can use it and not otherwise."
                % (d27, d14))


# ------------------------------------------------------------- l. the robots
def robots():
    """Four frames off the platform, as recorded.

    A poster about an autonomous pipeline has to show that the field half is real
    equipment on real ground, and the honest version of that is frames straight
    out of the archive -- not a staged photograph and not a rendering. Nothing is
    enhanced here and nothing is labelled; the boxes in the laser cart's down
    camera are its own detector drawing on its own video, which is what the cart
    records while it drives, and they are not ground truth.
    """
    P = D.PLATFORM
    shots = [("r241_row.jpg", "robot 241", "along a mulched crop row"),
             ("r241_weeds.jpg", "robot 241", "weeds between the rows"),
             ("lc_down.jpg", "laser cart", "down camera, onboard detector"),
             ("lc_front.jpg", "laser cart", "forward camera")]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 1.95))
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
    fig.tight_layout(w_pad=1.1)
    return save(fig, "l_robots",
                "Frames as recorded, unenhanced. %d frames across %d sessions and %d robots sit "
                "on the platform and %d of them are labelled: robot 241 contributes %d at "
                "%s and %g Hz, the laser cart %d of which %d are in a field. The boxes in (n) "
                "are the cart's own detector drawing on its own video, not ground truth."
                % (P["total_frames"], P["sessions"], P["robots"], P["labelled"],
                   P["r241_frames"], P["r241_res"], P["r241_hz"],
                   P["lasercar_frames"], P["lasercar_field_frames"]))


# ---------------------------------------------------------------- f. rounds
def rounds():
    R = D.ROUNDS
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 2.05),
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
    ax.set_xlabel("unattended round"); ax.set_ylabel("mAP$_{50-95}$")
    ax.set_xlim(0.3, len(xs) + 4.6); ax.set_ylim(0.542, 0.616)
    despine(ax); hline_grid(ax); panel(ax, "f")

    # Three bars, not two: bar 0 vs bar 1 is the matched pair that isolates the
    # warm start (same data, same 30 epochs, same complete cosine, same seed), and
    # bar 2 is the recipe the campaign actually ran, which is the only arm this
    # project has repeated over seeds. Quoting the gap against a bar drawn from a
    # different recipe is what the earlier draft of this panel did wrong.
    vals = [R["cold"], R["warm"], R["recipe_mean"]]
    errs = [0.0, 0.0, R["recipe_sd"]]
    ax2.bar([0, 1, 2], vals, yerr=errs, width=0.56,
            color=[GOOD, WARN, MUTE], capsize=4, error_kw=dict(lw=1.0, ecolor=INK))
    for i, v in enumerate(vals):
        ax2.text(i, v + errs[i] + 0.0030, "%.4f" % v, ha="center", fontsize=10.5,
                 color=INK)
    ax2.plot([0, 0, 1, 1], [0.5905, 0.5918, 0.5918, 0.5905], lw=0.9, color=INK)
    ax2.text(0.5, 0.5928, "%+.4f,  %.1f $\\sigma$" % (R["chain_effect"], R["chain_sigma"]),
             ha="center", fontsize=10, color=INK)
    ax2.plot([1, 1, 2, 2], [0.5726, 0.5739, 0.5739, 0.5726], lw=0.9, color=MUTE)
    ax2.text(1.5, 0.5749, "%+.4f,  %.1f $\\sigma$" % (R["sched_effect"], abs(R["sched_sigma"])),
             ha="center", fontsize=10, color=MUTE)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["fresh\nstart", "warm\nstart", "warm start\n+ clock cap"],
                        fontsize=10)
    ax2.set_ylim(0.540, 0.600); ax2.set_ylabel("mAP$_{50-95}$")
    # Under the tick labels, not inside the bars, where the fill fought the text.
    for i, n in enumerate((1, 1, 3)):
        ax2.annotate("n = %d" % n, (i, 0), xytext=(0, -34),
                     textcoords="offset points", xycoords=("data", "axes fraction"),
                     ha="center", fontsize=9, color=MUTE, annotation_clip=False)
    despine(ax2); hline_grid(ax2); panel(ax2, "g", dx=-0.30)
    fig.tight_layout(w_pad=2.4)
    return save(fig, "f_rounds",
                "(f) Fifteen unattended rounds on a fixed holdout. (g) Round 15's exact dataset, "
                "three ways. Bars 1 and 2 differ only in the starting weights, so their gap is "
                "the warm-start chain; bar 3 is the recipe the campaign ran, whose wall-clock "
                "cap truncates the cosine. Sigma throughout is that recipe's own seed spread, "
                "0.0040 over three seeds, propagated to the difference being quoted. Completing "
                "the schedule is not the fix -- bar 2 sits below bar 3.")


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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.05),
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
    fig.tight_layout(w_pad=2.2)
    return save(fig, "h_drive",
                "Robot 241, 2026-08-29: one 213 s pass through a crop plot. (h) GPS track from the "
                "board fix. (i) IMU heading over the same pass.")


# ---------------------------------------------------------------- j. per species
def species():
    fd = json.load(open(os.path.join(HERE, "figures_data.json")))
    rows = sorted(fd["per_species_yolo11n_val"]["rows"], key=lambda r: r["map50_95"])
    fig, ax = plt.subplots(figsize=(4.6, 2.62))
    y = np.arange(len(rows))
    ax.barh(y, [r["map50_95"] for r in rows], color=BLUE, height=0.62)
    for i, r in enumerate(rows):
        ax.text(r["map50_95"] - 0.012, i, "%.3f" % r["map50_95"], va="center",
                ha="right", fontsize=9.5, color=WHITE)
    ax.set_yticks(y); ax.set_yticklabels([r["cls"] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 1.0); ax.set_xlabel("mAP$_{50-95}$")
    despine(ax, left=False); ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    panel(ax, "j", dx=-0.40)
    return save(fig, "j_species",
                "Per-species performance of the deployable checkpoint. The spread across the twelve "
                "classes is 0.214, roughly seventy times the seed noise.")


# ---------------------------------------------------------------- k. funnel
def funnel():
    F = D.FUNNEL
    fig, ax = plt.subplots(figsize=(5.6, 2.15))
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
               species, funnel):
        fn()
    json.dump(CAPS, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    print("\n%d figures" % len(CAPS))
