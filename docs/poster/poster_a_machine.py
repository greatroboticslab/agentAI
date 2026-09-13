#!/usr/bin/env python3
"""Draft A — the machine.

Leads with the system as built: two robots in a Tennessee crop plot, a live
uplink, a governed registry, two agents on an HPC allocation, and a supervisor
reading the run's own artifacts. The measurements are evidence that the shape is
right; they are not the argument.

    python3 docs/poster/figures.py && python3 docs/poster/poster_a_machine.py
"""
import os
from deck import (Deck, D, COL, FULL, MARG, W, INK, NAVY, BLUE, MUTE, WARN, RULE,
                  PALEB, WHITE, BODY, CAPTION, HERE)

C, L, S, R, F, X = D.CENSUS, D.LEDGER, D.SUPERVISION, D.ROUNDS, D.FIELD, D.SECOND_EXAM
A = S["table"]["arms"]
LD = D.LADDER

d = Deck("Two Field Robots and an Agent Platform for Laser Weeding",
         "Anyone can download a weed dataset this afternoon. Nobody can download the path from a "
         "robot in a Tennessee crop row to a trained checkpoint. We built that path, ran it "
         "unattended for fifteen rounds, and measured every stage of it, including the last one "
         "we have not closed.")

TOP = 6.15

# ---------------------------------------------------------------- hero band
y = d.figure(COL[1][0], TOP, COL[1][1], "n_system", "Figure 1.")
hero_bottom = y

yl = d.head(COL[0][0], TOP, COL[0][1], "The two vehicles")
yl = d.figure(COL[0][0], yl, COL[0][1], "l_robots", "Figure 2.")
yl = d.body(COL[0][0], yl, COL[0][1],
    "Robot 241 is a tracked rover that drives crop rows and streams frames, position and "
    "attitude over a cellular link. The laser cart carries the weeder itself: a galvanometer "
    "laser, a down camera over the work zone and a forward camera. Between them they have "
    "recorded %s frames in %d drives. None is labelled."
    % ("{:,}".format(C["frames"]), C["sessions"]))

yr = d.head(COL[2][0], TOP, COL[2][1], "What is on disk")
for val, lab, note in (
        ("{:,}".format(C["frames"]), "camera frames", "%d drives, 2 robots, %d labelled"
         % (C["sessions"], C["labelled"])),
        ("{:,}".format(C["sensor_rows"]), "telemetry rows",
         "nine streams: GPS, two IMUs, motor, laser, control"),
        ("%.0f m" % C["gps_m"], "of GPS track",
         "%s fixes over %d drives; the longest single pass is %.0f m"
         % ("{:,}".format(C["gps_fixes"]), C["gps_sessions"], C["hero_m"]))):
    yr = d.bignum(COL[2][0], yr, COL[2][1], val, lab, note)
    yr += 0.16
yr = d.note(COL[2][0], yr, COL[2][1], "The cart has no position sensor.",
    "Its archive declares four telemetry streams and neither of them is GPS or IMU, so the "
    "sensor-fusion work on this platform is robot 241 only.")

y = max(hero_bottom, yl, yr) + 0.42

# ---------------------------------------------------------------- left column
x, w = COL[0]
y0 = y
y0 = d.head(x, y0, w, "Why build the path")
y0 = d.body(x, y0, w,
    "The end product burns weeds between crop rows instead of spraying them, so the detector has "
    "to work on the rows this vehicle drives and has to fit on the Jetson the cart carries. The "
    "deployed checkpoint is 2.6 M parameters. No public dataset contains our rows, and a "
    "five-person lab does not hand-label a hundred thousand field images.")
y0 = d.sub(x, y0, w, "One drive, in full")
y0 = d.figure(x, y0, w, "h_drive", "Figure 3.")

# ---------------------------------------------------------------- centre column
x, w = COL[1]
y1 = y
y1 = d.head(x, y1, w, "What the agents did on their own")
y1 = d.body(x, y1, w,
    "Two SLURM jobs, one collecting and one training, ran fifteen rounds against the locked "
    "registry with nobody in the room. Each round searched public sources, filtered what it "
    "found, merged it into the corpus, trained, and evaluated on a sealed holdout. Every round "
    "reported success. Accuracy fell for fifteen rounds.")
y1 = d.figure(x, y1, w, "f_rounds", "Figure 4.")
y1 = d.body(x, y1, w,
    "Reading the loop's own artifacts instead of its dashboard explained it. The training corpus "
    "stopped growing at round three: the same 24 datasets and the same 48,752 unique images went "
    "in every time, and the apparent growth was one staging directory nobody cleared. Eight "
    "rounds collected nothing at all and reported success. What was left was the chain itself, "
    "and a control on round 15's own data priced it: starting fresh instead of from the previous "
    "round is worth %s, which is 5.1 times the seed spread this recipe actually has."
    % ("+%.4f" % R["chain_effect"]))

# ---------------------------------------------------------------- right column
x, w = COL[2]
y2 = y
y2 = d.head(x, y2, w, "Does more data help?")
y2 = d.table(x, y2, w, ["harvested added", "mAP50-95", "n"],
             [["none, the core", "0.8637 ± 0.0027", "3"],
              ["+5,000", "0.8609 ± 0.0036", "3"],
              ["+15,000", "0.8580 ± 0.0047", "3"],
              ["+40,000", "0.8448 ± 0.0018", "3"]], [0.44, 0.40, 0.16], hi=3)
y2 = d.body(x, y2, w,
    "Twelve times the training data costs 0.0189 on our own holdout, which is 8.2 times the seed "
    "spread. The same eight checkpoints scored on a second weed dataset fall too, by 0.0299, so "
    "this is the data and not the metric. One of six audited web sources reaches a 0.90 "
    "label-precision bar, and 44,750 of 156,521 harvested images are copies of each other.",
    size=CAPTION, color=MUTE)

# ------------------------------------------------------ ledger and supervision
y = max(y0, y1, y2) + 0.46
yc = d.head(COL[1][0], y, COL[1][1], "Everything we trained, ran, or rejected")
yc = d.figure(COL[1][0], yc, COL[1][1], "m_ledger", "Figure 5.")

yl = d.head(COL[0][0], y, COL[0][1], "Back on our own rows")
yl = d.figure(COL[0][0], yl, COL[0][1], "p_field", "Figure 6.")

yr = d.head(COL[2][0], y, COL[2][1], "Who notices when it breaks")
yr = d.body(COL[2][0], yr, COL[2][1],
    "We froze 162 real failures out of this project's own engineering record and scored every "
    "kind of supervision we had against them. The watchdog the loop actually ran returned no "
    "decision on any of the 149 dev cases: \u201cno signal fired\u201d is not a judgement. Twelve "
    "deterministic checks reach 0.095. A model reading the raw artifacts reaches 0.70, and what "
    "separates the models is not size but evidence. The 7 B leaves 55% of its findings "
    "unevidenced and the 27 B leaves none.", after=22)
yr += 0.34
yr = d.table(COL[2][0], yr, COL[2][1],
             ["reviewer", "recall", "grounded", "incidents"],
             [["scripted watchdog", "no decision", "\u2014", "\u2014"],
              ["12 rules", "0.095", "0.095", "116"],
              ["Qwen2.5-7B", "0.823", "0.274", "113"],
              ["Qwen3-14B", "0.675", "0.614", "114"],
              ["Qwen3.8-27B", "0.702", "0.702", "57"]], [0.38, 0.21, 0.21, 0.20])

# -------------------------------------------------------------------- footer
# No journey strip here on purpose. This draft argues that the machine is the
# contribution, so the sheet ends on what the machine is and what it cannot do
# yet, not on how we got to it. Draft B is the one that leads with the road.
y = max(yl, yc, yr) + 0.60
d.rect(FULL[0], y - 0.28, FULL[1], 0.022, fill=RULE)
cw = (FULL[1] - 1.0) / 3.0
d.body(FULL[0], y, cw,
    "Where this stands today. The collector and trainer are paused after a stop-loss on "
    "2026-08-29. The last field drive was the same day. The supervisor is advisory and nothing it "
    "says is applied. The laser has never fired at a weed under our control. None of the %s "
    "frames is labelled."
    % "{:,}".format(C["frames"]), size=CAPTION, color=MUTE)
d.body(FULL[0] + cw + 0.5, y, cw,
    "How the numbers were made. Most in-domain numbers are mAP50-95 on the same 1,977 images "
    "under Ultralytics. RF-DETR is pycocotools and the zero-shot column is mAP50 on the "
    "848-image test split, and neither is ever differenced against the rest. That holdout is "
    "also the validation set during training, so every mAP here is that run's best epoch, a "
    "maximum over 21 to 100 evaluations on the set it is reported on. We measured that optimism "
    "at +0.002 to +0.017 rather than hiding it.", size=CAPTION, color=MUTE)
d.body(FULL[0] + 2 * (cw + 0.5), y, cw,
    "Numbers we withdrew. 0.910 was a data leak, found by a content-level check we built after "
    "publishing it. 0.9033 is the best of four unseeded runs and only one of the four crossed "
    "0.90. An earlier reading of the ladder as a flat curve is retracted: with three seeds at "
    "every rung the top rung is down, not flat.", size=CAPTION, color=MUTE)

print("  ROW1 %.2f | ROW2 %.2f | ROW3 %.2f | journey %.2f | footer end %.2f"
      % (hero_bottom, max(y0,y1,y2), max(yl,yc,yr), y, y + 1.6))
d.save(os.path.join(HERE, "draft_A_machine.pptx"))
