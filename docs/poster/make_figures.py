#!/usr/bin/env python3
"""Render the poster's figures from poster_data.py. One command, no hand-editing.

    python3 docs/poster/make_figures.py

Every figure prints its own caption to stdout with the source file, so the poster
text and the figure cannot drift apart.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poster_data as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#12243a"; MUTE = "#5d6b7a"; RULE = "#c9d2dc"
BLUE = "#0a4d8c"; GOOD = "#1f6f4a"; WARN = "#b4511f"; FADE = "#dbe3ea"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlesize": 19,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})

def save(fig, name, caption):
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    print("%-22s %s" % (name + ".png", caption))
    return p


# ---------------------------------------------------------------- F1 levers
def f1():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    rows = D.LEVERS[::-1]
    y = np.arange(len(rows))
    vals = [r["delta"] for r in rows]
    cols = [GOOD if v > 0 else WARN for v in vals]
    ax.barh(y, vals, color=cols, height=0.52, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["lever"] for r in rows], fontsize=16)
    for i, r in enumerate(rows):
        v = r["delta"]
        lbl = "%+.4f" % v
        ax.text(v + (0.0035 if v > 0 else -0.0035), i, lbl, va="center",
                ha="left" if v > 0 else "right", fontsize=17, fontweight="bold",
                color=GOOD if v > 0 else WARN, zorder=4)
        ax.text(0.0, i - 0.34, "  %s   ·   n = %d" % (r["hi"], r["n"]),
                va="top", ha="left", fontsize=11.5, color=MUTE)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_xlim(-0.032, 0.098)
    ax.set_xlabel("change in mAP50-95 on the 1,977-image holdout", fontsize=14)
    ax.set_title("What actually moves the detector", pad=14)
    ax.grid(axis="x", color=RULE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    return save(fig, "f1_levers",
                "Initialisation beats architecture beats data. n=3 seeds on the top two rows. "
                "Source: " + D.LEVERS[0]["src"])


# ---------------------------------------------------------------- F2 ladder
def f2():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = D.LADDER["rungs"]; y = D.LADDER["seed101"]
    xs = np.arange(len(x))
    band = 0.0029
    ax.fill_between(xs, [v - band for v in y], [v + band for v in y],
                    color=FADE, zorder=1, label="± measured seed std (0.0029)")
    ax.plot(xs, y, "-o", color=BLUE, lw=2.6, ms=10, zorder=3, label="seed 101")
    for i, v in enumerate(y):
        ax.annotate("%.4f" % v, (xs[i], v), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=14, fontweight="bold", color=INK)
    if D.LADDER["seed102"] == D.PENDING:
        ax.text(0.0, 1.015, "seeds 102 and 103 in flight",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5,
                color=WARN, style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels(["core\n+0", "+5,000", "+15,000", "+40,000"], fontsize=14)
    ax.set_ylabel("mAP50-95", fontsize=14)
    ax.set_ylim(0.835, 0.872)
    ax.set_title("Twelve times the harvested data buys nothing", pad=14)
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    return save(fig, "f2_ladder",
                D.LADDER["reading"] + " Source: " + D.LADDER["src"])


# ---------------------------------------------------------------- F3 the wall
def f3():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    a = float(D.WALL["in_domain"].split("±")[0]); ae = float(D.WALL["in_domain"].split("±")[1])
    b = float(D.WALL["out_domain"].split("±")[0]); be = float(D.WALL["out_domain"].split("±")[1])
    ax.errorbar([0], [a], yerr=[ae], fmt="o", ms=16, color=GOOD, capsize=8, lw=2.5, zorder=3)
    ax.errorbar([1], [b], yerr=[be], fmt="o", ms=16, color=WARN, capsize=8, lw=2.5, zorder=3)
    ax.plot([0, 1], [a, b], "--", color=MUTE, lw=2, zorder=2)
    ax.annotate(D.WALL["in_domain"], (0, a), xytext=(0, 22), textcoords="offset points",
                ha="center", fontsize=17, fontweight="bold", color=GOOD)
    ax.annotate(D.WALL["out_domain"], (1, b), xytext=(0, 22), textcoords="offset points",
                ha="center", fontsize=17, fontweight="bold", color=WARN)
    ax.text(0.5, (a + b) / 2 + 0.06, "-%.4f" % (a - b), ha="center", fontsize=20,
            fontweight="bold", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["same field, held out\n(CottonWeedDet12)",
                        "a different weed dataset\n(ImageWeeds, 3,208 imgs)"], fontsize=13)
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP50-95, class-agnostic", fontsize=14)
    ax.set_title("The detector does not travel", pad=14)
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    return save(fig, "f3_wall",
                "n=3 on both ends, %s. Source: %s" % (D.WALL["note"], D.WALL["src"]))


# ---------------------------------------------------------------- F4 funnel
def f4():
    """Stage labels sit ABOVE each bar, numbers to the right of it. The first
    draft put the label inside the bar, where the two short stages had no room."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    F = D.FUNNEL
    stages = [("harvested and labelled", F["registry_labelled"], BLUE),
              ("unique after dedup", F["unique"], BLUE),
              ("audited for label quality", F["audited_images"], MUTE),
              ("clears the 0.90 precision bar", F["passing_images"], WARN)]
    tot = stages[0][1]
    for i, (lab, v, c) in enumerate(stages):
        w = v / tot
        yb = -i * 1.0
        ax.text(0.0, yb + 0.50, lab, va="center", fontsize=13.5, color=INK)
        ax.add_patch(Rectangle((0, yb - 0.02), w, 0.40, color=c, zorder=3))
        ax.text(w + 0.014, yb + 0.18, "{:,}".format(v), va="center", fontsize=17,
                fontweight="bold", color=c)
    ax.text(0.0, -len(stages) + 0.42,
            "%d of %d audited sources pass   ·   the rest score %s\n%s"
            % (F["sources_passing"], F["audited_sources"], F["failing_range"],
               F["probe_calibration"]),
            fontsize=12, color=MUTE, va="top", linespacing=1.5)
    ax.set_xlim(-0.01, 1.26); ax.set_ylim(-len(stages) + 0.05, 0.95)
    ax.axis("off")
    ax.set_title("Where 156,521 harvested images go", pad=10, loc="left")
    return save(fig, "f4_funnel",
                "Volume is not supervision. Source: " + F["src"])


# ---------------------------------------------------------------- F5 headline
def f5():
    """Left gutter for the labels, bars to the right of it. The first draft put
    the arm names at x=-0.30 and the false-alarm bars grew leftward into them."""
    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    S = D.SUPERVISION
    rows = S["rows"]
    y = np.arange(len(rows))[::-1]
    GUT = -0.34          # everything left of this is text
    for i, r in enumerate(rows):
        yy = y[i]
        ax.barh(yy, r["recall"], height=0.44, color=(WARN if r["recall"] < 0.2 else GOOD),
                alpha=0.45 if r["partial"] else 1.0, zorder=3)
        ax.barh(yy, -r["fa"] * 0.85, height=0.44, color=MUTE, alpha=0.5, zorder=3)
        lbl = "%.3f" % r["recall"]
        ax.text(r["recall"] + 0.012, yy, lbl, va="center", fontsize=18,
                fontweight="bold", color=INK, zorder=4)
        if r["partial"]:
            ax.text(r["recall"] + 0.012, yy - 0.235,
                    "%d of 149 cases, still running" % r["cases"],
                    va="center", fontsize=10.5, color=WARN, style="italic", zorder=4)
        ax.text(-r["fa"] * 0.85 - 0.008, yy, "%.3f" % r["fa"], va="center", ha="right",
                fontsize=12, color=MUTE, zorder=4)
        ax.text(GUT - 0.02, yy + 0.26, r["arm"], va="center", ha="right",
                fontsize=15.5, fontweight="bold", color=INK)
        ax.text(GUT - 0.02, yy + 0.02, r["reads"], va="center", ha="right",
                fontsize=11.5, color=MUTE)
        ax.text(GUT - 0.02, yy - 0.24, r["note"], va="center", ha="right",
                fontsize=10.5, color=MUTE, style="italic")
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.axvline(S["ceiling"], color=BLUE, lw=1.6, ls=":", zorder=2)
    ax.text(S["ceiling"] + 0.008, y[0] + 0.62,
            "ceiling for any rules-only arm (%.3f)" % S["ceiling"],
            fontsize=12, color=BLUE)
    ax.set_xlim(-1.05, 1.04); ax.set_ylim(y[-1] - 0.58, y[0] + 0.92)
    ax.set_xticks([-0.17, 0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.2\nfalse alarms", "0", "0.25", "0.50", "0.75", "1.0\nrecall"],
                       fontsize=12)
    for t, xx in zip(ax.get_xticklabels(), [-0.17, 0, 0.25, 0.5, 0.75, 1.0]):
        pass
    ax.set_yticks([])
    ax.set_title("What each kind of supervision actually catches   ·   %d real incidents, %d controls"
                 % (S["incidents"], S["controls"]), pad=18, loc="left", x=0.30)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.22, 1.0)
    return save(fig, "f5_supervision",
                "%s Source: %s" % (S["ceiling_why"], S["src"]))


# ---------------------------------------------------------------- F6 rounds
def f6():
    R = D.ROUNDS
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                  gridspec_kw={"width_ratios": [1.65, 1]})
    xs = np.arange(1, len(R["map"]) + 1)
    ax.axvspan(R["frozen_from"] - 0.4, len(xs) + 0.4, color=FADE, zorder=0)
    ax.plot(xs, R["map"], "-o", color=BLUE, lw=2.4, ms=7, zorder=3, label="reported (best epoch)")
    ax.plot(xs, R["last"], "-s", color=MUTE, lw=1.8, ms=5, zorder=3, label="last epoch")
    ax.text(R["frozen_from"] + 0.2, 0.605,
            "training corpus identical from here on\n%s" % R["corpus"],
            fontsize=11, color=MUTE, va="top")
    ax.set_xlabel("unattended round", fontsize=13)
    ax.set_ylabel("mAP50-95", fontsize=13)
    ax.set_title("Fifteen rounds of \"compounding\"", pad=12, loc="left")
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

    bars = [("cold start", R["cold"], GOOD), ("chained (measured)", 0.5577, WARN)]
    ax2.bar([0, 1], [b[1] for b in bars], color=[b[2] for b in bars], width=0.5, zorder=3)
    ax2.errorbar([1], [0.5577], yerr=[0.0040], fmt="none", ecolor=INK, capsize=7, lw=2, zorder=4)
    for i, b in enumerate(bars):
        ax2.text(i, b[1] + 0.0025, "%.4f" % b[1], ha="center", fontsize=16,
                 fontweight="bold", color=INK)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["start fresh\neach round", "warm-start\nfrom the last"],
                                                fontsize=12)
    ax2.set_ylim(0.54, 0.595); ax2.set_ylabel("mAP50-95", fontsize=13)
    ax2.set_title("%+0.4f, %.1f sigma  ·  same data" % (R["chain_effect"], R["chain_sigma"]),
                  pad=12, loc="left")
    ax2.grid(axis="y", color=RULE, lw=0.7, zorder=0); ax2.set_axisbelow(True)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    fig.tight_layout()
    return save(fig, "f6_rounds",
                "The corpus never changed; the decline is the warm-start chain. Source: " + R["src"])


# ------------------------------------------------------------- F7 field drive
def f7():
    """The one unambiguous field drive on the platform: robot 241 through a crop
    plot. GPS track and IMU heading from the same 213 s recording.

    The track is drawn from the BOARD fix. The platform's own derived gps.csv
    preferred the Pi fix, which is frozen to a single coordinate, so every
    trajectory the platform exposed was a motionless point -- 17 sessions,
    1,281 rows, 0.0 m between them (robot_ingest.py:264, fixed 2026-09-11)."""
    import json, csv
    here = os.path.dirname(os.path.abspath(__file__))
    trk = json.load(open(os.path.join(here, "hero_track.json")))
    pts = trk["board"]
    lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
    lat0 = sum(lats) / len(lats)
    mx = 111320.0 * np.cos(np.radians(lat0))
    xs = [(lo - lons[0]) * mx for lo in lons]
    ys = [(la - lats[0]) * 110540.0 for la in lats]

    rows = list(csv.DictReader(open(os.path.join(here, "hero_imu.csv"))))
    t0 = float(rows[0]["timestamp"])
    t = [float(r["timestamp"]) - t0 for r in rows]
    # Unwrap the compass: a 359 -> 1 step is one degree of turn, not 358, and
    # drawn raw it reads as vertical noise spikes across the whole trace.
    raw = [float(r["heading"]) for r in rows]
    hd = np.degrees(np.unwrap(np.radians(raw)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                                  gridspec_kw={"width_ratios": [1.05, 1]})
    ax.plot(xs, ys, "-", color=BLUE, lw=2.0, zorder=3)
    ax.scatter(xs, ys, s=9, color=BLUE, alpha=0.45, zorder=4)
    ax.scatter([xs[0]], [ys[0]], s=150, color=GOOD, zorder=5, label="start")
    ax.scatter([xs[-1]], [ys[-1]], s=150, color=WARN, marker="s", zorder=5, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("metres east", fontsize=13); ax.set_ylabel("metres north", fontsize=13)
    ax.set_title("GPS track  ·  %.1f m over %.0f s" % (trk["path_m"], trk["seconds"]),
                 pad=12, loc="left")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.grid(color=RULE, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    ax2.plot(t, hd, "-", color=INK, lw=1.1)
    ax2.set_xlabel("seconds into the drive", fontsize=13)
    ax2.set_ylabel("heading, degrees (unwrapped)", fontsize=13)
    ax2.set_title("IMU heading  ·  %d samples at %.1f Hz" % (len(rows), len(rows) / (t[-1] or 1)),
                  pad=12, loc="left")
    ax2.grid(color=RULE, lw=0.7, zorder=0); ax2.set_axisbelow(True)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    fig.tight_layout()
    return save(fig, "f7_fielddrive",
                "Robot 241, 2026-08-29, one 213 s pass through a crop plot: 1,013 camera frames "
                "at 640x360, 211 GPS fixes, 3,154 IMU samples. Source: "
                "uploads/ul_4_09test_49ea7a2a/files/")


if __name__ == "__main__":
    print("rendering into", OUT)
    for fn in (f1, f2, f3, f4, f5, f6, f7):
        fn()
    print("\ndone")
