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
    L = D.LADDER
    seeds = L.get("seeds") or {}
    x = L["rungs"]; xs = np.arange(len(x))
    A, Ae, An = [], [], []
    for k, v in zip(x, L["seed101"]):
        sv = seeds.get(k)
        if isinstance(sv, list) and len(sv) >= 2:
            A.append(st.mean(sv)); Ae.append(st.stdev(sv)); An.append(len(sv))
        else:
            A.append(v); Ae.append(0.0); An.append(1)
    B = L.get("armB")
    fig, ax = plt.subplots(figsize=(5.2, 2.50))
    ax.errorbar(xs, A, yerr=Ae, fmt="o-", color=BLUE, capsize=4,
                elinewidth=1.0, markerfacecolor=BLUE, zorder=3)
    if B:
        ax.plot(xs, B, "s--", color=WARN, markerfacecolor=WHITE, zorder=3)
        ax.text(xs[1] + 0.08, B[1] - 0.0016, "one shared class", ha="left",
                va="top", fontsize=10, color=WARN)
    ax.text(xs[1] + 0.08, A[1] + 0.0022, "class per source dataset", ha="left",
            va="bottom", fontsize=10, color=BLUE)
    for i, (m, n) in enumerate(zip(A, An)):
        ax.annotate("n=%d" % n, (xs[i], 0), xytext=(0, -32),
                    textcoords="offset points", xycoords=("data", "axes fraction"),
                    ha="center", fontsize=9, color=MUTE if n > 1 else WARN,
                    annotation_clip=False)
    ax.set_xticks(xs)
    ax.set_xticklabels(["core\nalone", "+5k", "+15k", "+40k"], fontsize=10.5)
    ax.set_xlabel("harvested images added to a 3,671-image core", labelpad=16)
    ax.set_ylabel("mAP$_{50-95}$")
    ax.set_ylim(0.820, 0.872)
    despine(ax); hline_grid(ax); panel(ax, "c")
    return save(fig, "c_ladder",
                "Twelve times the training data. Error bars are the seed spread where three seeds "
                "exist; rungs still at one run are marked. The two class spaces differ only in the "
                "label a harvested box carries.")


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
    ax.annotate("%.4f" % b, (1, b), xytext=(-8, 4), textcoords="offset points",
                ha="right", fontsize=11.5, color=WARN)
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
    S = D.SUPERVISION
    fig, ax = plt.subplots(figsize=(5.2, 2.55))
    ax.axhline(S["ceiling"], color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.020, S["ceiling"] + 0.022, "ceiling for any rules-only arm",
            fontsize=9.5, color=MUTE, ha="left")
    ax.plot([0, 0.78], [0, 0.78], color=RULE, lw=0.8, zorder=0)
    ax.text(0.735, 0.700, "chance", fontsize=9, color=GREY, rotation=34, ha="right")
    # label offsets chosen per point so no two collide
    place = {
        "Scripted watchdog":     (GREY, "s", (10, -4), "left"),
        "Deterministic signals": (MUTE, "^", (10, 10), "left"),
        "Qwen3-14B":             (WARN, "o", (-11, -8), "right"),
        "Qwen3.8-27B":           (BLUE, "o", (10, 8), "left"),
    }
    for r in S["rows"]:
        c, mk, off, ha = place.get(r["arm"], (INK, "o", (8, 8), "left"))
        ax.plot(r["fa"], r["recall"], mk, color=c, ms=9, markerfacecolor=c, zorder=4)
        ax.annotate(r["arm"], (r["fa"], r["recall"]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=10.5, color=c)
    ex = S.get("in_flight")
    if ex:
        ax.plot(ex["fa"], ex["recall"], "o", color=GOOD, ms=9,
                markerfacecolor=WHITE, markeredgewidth=1.6, zorder=4)
        # Below the point: to its left is the y-axis label, to its right is the 27B.
        ax.annotate("%s\n%d of 149" % (ex["model"].split(" (")[0], ex["cases"]),
                    (ex["fa"], ex["recall"]), textcoords="offset points",
                    xytext=(-4, -13), ha="right", va="top", fontsize=10, color=GOOD)
    ax.set_xlim(-0.035, 0.80); ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("false-alarm rate  (33 control cases)")
    ax.set_ylabel("detection recall  (116 incidents)")
    despine(ax); hline_grid(ax); panel(ax, "e", dx=-0.13, dy=1.03)
    return save(fig, "e_supervision",
                "Every arm on one frozen corpus of 162 real incidents from this project's own "
                "history. Up and to the left is better. The scripted watchdog sits at the origin: "
                "it flagged nothing, 149 times out of 149.")


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

    ax2.bar([0, 1], [R["cold"], 0.5577], yerr=[0, 0.0040], width=0.5,
            color=[GOOD, WARN], capsize=4, error_kw=dict(lw=1.0, ecolor=INK))
    for i, v in enumerate([R["cold"], 0.5577]):
        ax2.text(i, v + 0.0032, "%.4f" % v, ha="center", fontsize=11, color=INK)
    ax2.plot([0, 0, 1, 1], [0.5895, 0.5905, 0.5905, 0.5895], lw=0.9, color=INK)
    ax2.text(0.5, 0.5912, "+0.0287,  5.0σ", ha="center", fontsize=10.5, color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["fresh\nstart", "warm\nstart"], fontsize=10.5)
    ax2.set_ylim(0.540, 0.598); ax2.set_ylabel("mAP$_{50-95}$")
    # Under the tick labels, not inside the bars, where the fill fought the text.
    for i, (n, c) in enumerate(((1, WARN), (3, MUTE))):
        ax2.annotate("n = %d" % n, (i, 0), xytext=(0, -34),
                     textcoords="offset points", xycoords=("data", "axes fraction"),
                     ha="center", fontsize=9, color=c, annotation_clip=False)
    despine(ax2); hline_grid(ax2); panel(ax2, "g", dx=-0.30)
    fig.tight_layout(w_pad=2.4)
    return save(fig, "f_rounds",
                "(f) Fifteen unattended rounds on a fixed holdout. (g) On round 15's exact dataset "
                "and schedule, starting fresh instead of from the previous round's weights.")


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
    for fn in (fam, vlm, ladder, wall, supervision, rounds, drive, species, funnel):
        fn()
    json.dump(CAPS, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    print("\n%d figures" % len(CAPS))
