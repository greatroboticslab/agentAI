#!/usr/bin/env python3
"""Draft B — the road.

Leads with the six months: what we believed, what broke it, and what we built in
response. Four of the eight turns are things that did not work. The platform is
what the road produced and the measurements are the turns themselves.

    python3 docs/poster/figures.py && python3 docs/poster/poster_b_journey.py
"""
import os
from deck import Deck, D, COL, FULL, CAPTION, HERE, MUTE, RULE

C, R = D.CENSUS, D.ROUNDS
L = D.LEDGER

d = Deck("How Much of This Pipeline Can the Agents Run by Themselves?",
         "We spent six months finding out, on our own robots and our own cluster. The answer is "
         "that collection and training run alone, and judging does not.")

TOP = 6.15

# -------------------------------------------------------------- the road first
y = d.head(FULL[0], TOP, FULL[1], "The route we actually took")
y = d.figure(FULL[0], y, FULL[1], "t_journey", "Figure 1.")
y += 0.34

# --------------------------------------------------------------- left column
x, w = COL[0]
y0 = d.head(x, y, w, "What we collect now")
y0 = d.figure(x, y0, w, "l_robots", "Figure 2.")
y0 = d.body(x, y0, w,
    "Two ground vehicles drive our own cotton plots. Robot 241 streams frames, position and "
    "attitude; the laser cart carries the weeder and two cameras. %s frames in %d drives, "
    "%s telemetry rows across nine streams, %.0f m of GPS track. None of it is labelled, and "
    "the cart writes no position at all."
    % ("{:,}".format(C["frames"]), C["sessions"], "{:,}".format(C["sensor_rows"]), C["gps_m"]),
    after=24)
y0 = d.figure(x, y0, w, "h_drive", "Figure 3.")

# ------------------------------------------------------------- centre column
x, w = COL[1]
y1 = d.head(x, y, w, "The part that runs alone")
y1 = d.figure(x, y1, w, "n_system", "Figure 4.")
y1 = d.body(x, y1, w,
    "Collection and training run without a person. A round searches public sources, filters what "
    "it finds against a DINOv2 gate, merges into a registry that excludes the holdout by content "
    "hash, trains on an H100 allocation and evaluates. Fifteen rounds went through that path and "
    "every one reported success.")
y1 = d.sub(x, y1, w, "What fifteen unattended rounds actually did")
y1 = d.figure(x, y1, w, "f_rounds", "Figure 5.")
y1 = d.body(x, y1, w,
    "Accuracy fell the whole time. The corpus stopped growing at round three and eight rounds "
    "collected nothing while reporting success, so what was left was the chain: each round warm-"
    "started from the last one's weights. A control on round 15's own data prices that at "
    "%s, 5.1 times this recipe's own seed spread. Completing the truncated schedule does not "
    "recover it, which was the obvious fix and the wrong one."
    % ("+%.4f" % R["chain_effect"]))

# -------------------------------------------------------------- right column
x, w = COL[2]
y2 = d.head(x, y, w, "The part that does not")
y2 = d.body(x, y2, w,
    "Judging does not run alone yet. We froze 162 real failures out of this project's own "
    "engineering record and scored every kind of supervision we had. The watchdog the loop was "
    "running returned no decision on any of the 149 dev cases.", after=24)
y2 = d.table(x, y2, w, ["reviewer", "recall", "grounded"],
             [["scripted watchdog", "no decision", "—"],
              ["12 rules", "0.095", "0.095"],
              ["Qwen2.5-7B", "0.823", "0.274"],
              ["Qwen3-14B", "0.675", "0.614"],
              ["Qwen3.8-27B", "0.702", "0.702"]], [0.46, 0.27, 0.27], hi=4)
y2 = d.body(x, y2, w,
    "Recall does not order the reviewers by size. Evidence does. The 7 B has the highest recall "
    "on the table and two thirds of what it reports quotes a line that does not exist. That is "
    "the size class we had been running.", size=CAPTION, color=MUTE)
y2 = d.sub(x, y2 + 0.06, w, "Does more data help?")
y2 = d.table(x, y2, w, ["harvested added", "mAP50-95", "n"],
             [["none, the core", "0.8637 ± 0.0027", "3"],
              ["+5,000", "0.8609 ± 0.0036", "3"],
              ["+15,000", "0.8580 ± 0.0047", "3"],
              ["+40,000", "0.8448 ± 0.0018", "3"]], [0.44, 0.40, 0.16], hi=3)
y2 = d.body(x, y2, w,
    "Twelve times the data costs 0.0189, which is 8.2 times the seed spread, and the same eight "
    "checkpoints fall on a second weed dataset too. One of six audited web sources clears a 0.90 "
    "label-precision bar.", size=CAPTION, color=MUTE)

# --------------------------------------------------- what the road ran through
y = max(y0, y1, y2) + 0.50
yc = d.head(COL[1][0], y, COL[1][1], "Everything we trained, ran, or rejected")
yc = d.figure(COL[1][0], yc, COL[1][1], "m_ledger", "Figure 6.")

yl = d.head(COL[0][0], y, COL[0][1], "Six audited web sources")
yl = d.figure(COL[0][0], yl, COL[0][1], "s_sources", "Figure 7.")

yr = d.head(COL[2][0], y, COL[2][1], "Back on our own rows")
yr = d.figure(COL[2][0], yr, COL[2][1], "p_field", "Figure 8.")

# -------------------------------------------------------------------- footer
y = max(yl, yc, yr) + 0.55
d.rect(FULL[0], y - 0.28, FULL[1], 0.022, fill=RULE)
cw = (FULL[1] - 1.0) / 3.0
d.body(FULL[0], y, cw,
    "Where this stands today. The collector and trainer are paused after a stop-loss on "
    "2026-08-29, the supervisor is advisory, the laser has never fired at a weed under our "
    "control, and none of the %s frames is labelled." % "{:,}".format(C["frames"]),
    size=CAPTION, color=MUTE)
d.body(FULL[0] + cw + 0.5, y, cw,
    "What we ran. %d distinct models in four families: five detectors trained on our own holdout, "
    "two vision models inside the loop, fourteen vision-language models zero-shot, and nine "
    "language models as reviewers. One is deployed." % L["n_models"], size=CAPTION, color=MUTE)
d.body(FULL[0] + 2 * (cw + 0.5), y, cw,
    "Numbers we withdrew. 0.910 was a data leak, found by a check we built after publishing it. "
    "0.9033 is the best of four unseeded runs. An earlier reading of the ladder as a flat curve "
    "is retracted.", size=CAPTION, color=MUTE)

print("  row1 %.2f | row2 %.2f | end %.2f" % (max(y0,y1,y2), max(yl,yc,yr), y + 1.6))
d.save(os.path.join(HERE, "draft_B_journey.pptx"))
