#!/usr/bin/env python3
"""Draft C — the field.

Leads with what the machines actually saw, at a size a reader meets from two
metres away. The argument is that we own a real pipeline from soil to
checkpoint, which is the one thing a public dataset cannot supply.

    python3 docs/poster/figures.py && python3 docs/poster/poster_c_field.py
"""
import os
from deck import Deck, D, COL, FULL, CAPTION, HERE, MUTE, RULE, BLUE

C, R, L = D.CENSUS, D.ROUNDS, D.LEDGER

d = Deck("Robot-Collected Field Data and Agent-Run Training for Laser Weeding",
         "Two ground vehicles drive our own cotton plots and a pair of agents turn what they "
         "record into trained models. This is what the machines saw, what the agents did with it, "
         "and where the path breaks.")

TOP = 6.15

# --------------------------------------------------------------- the field
y = d.figure(FULL[0], TOP, FULL[1], "w_robots", "Figure 1.")
y += 0.30

# the census, as one rule-bounded row
cw = (FULL[1] - 4 * 0.55) / 5.0
cells = [("{:,}".format(C["frames"]), "camera frames", "%d drives, 2 robots" % C["sessions"]),
         ("{:,}".format(C["sensor_rows"]), "telemetry rows", "nine streams, counted line by line"),
         ("%.0f m" % C["gps_m"], "of GPS track", "%s fixes over %d drives"
          % ("{:,}".format(C["gps_fixes"]), C["gps_sessions"])),
         ("%d" % C["cart_field_frames"], "cart frames in a field",
          "of %d the cart has recorded" % C["cart_frames"]),
         ("%d" % C["labelled"], "of them labelled", "the whole archive, so far")]
d.rect(FULL[0], y, FULL[1], 0.026, fill=RULE)
ybot = y + 0.24
for i, (v, lab, note) in enumerate(cells):
    cx = FULL[0] + i * (cw + 0.55)
    ybot = max(ybot, d.bignum(cx, y + 0.20, cw, v, lab, note, size=56))
y = ybot + 0.20
d.rect(FULL[0], y - 0.10, FULL[1], 0.026, fill=RULE)
y += 0.28

# --------------------------------------------------------------- left column
x, w = COL[0]
y0 = d.head(x, y, w, "How a drive becomes a dataset")
y0 = d.body(x, y0, w,
    "A vehicle opens a session, the platform registers it as a governed dataset, and frames and "
    "sensor batches arrive over one API key while the vehicle is still driving. Robot 241 sends "
    "camera, GPS and two IMUs. The laser cart sends camera, laser state, vehicle telemetry and "
    "its own detections, and no position at all.", after=22)
y0 = d.sub(x, y0, w, "213 seconds in one crop plot")
y0 = d.figure(x, y0, w, "h_drive", "Figure 2.")


# ------------------------------------------------------------- centre column
x, w = COL[1]
y1 = d.head(x, y, w, "What the agents do with it")
y1 = d.figure(x, y1, w, "n_system", "Figure 3.")
y1 = d.sub(x, y1, w, "We let it run itself for fifteen rounds")
y1 = d.figure(x, y1, w, "f_rounds", "Figure 4.")
y1 = d.body(x, y1, w,
    "Every round reported success and accuracy fell the whole time. The corpus stopped growing at "
    "round three and eight rounds collected nothing. What was left was the chain: starting fresh "
    "instead of from the previous round's weights is worth %s, 5.1 times this recipe's own seed "
    "spread." % ("+%.4f" % R["chain_effect"]))

# -------------------------------------------------------------- right column
x, w = COL[2]
y2 = d.head(x, y, w, "Then we pointed it back at our own rows")
y2 = d.figure(x, y2, w, "p_field", "Figure 5.")
y2 = d.body(x, y2, w,
    "The deployable checkpoint draws a box on %d of the %s frames our robots recorded, and %d of "
    "those %d boxes are the same class. These frames carry no labels, so that is a firing rate, "
    "not recall and not precision. On a second labelled weed dataset the same checkpoints fall "
    "from 0.873 to 0.100."
    % (F["fired_25"] if False else D.FIELD["fired_25"], "{:,}".format(C["frames"]),
       D.FIELD["species"]["Purslane"], D.FIELD["fired_25"]), after=22)
y2 = d.sub(x, y2, w, "Does more web data help?")
y2 = d.table(x, y2, w, ["harvested added", "mAP50-95", "n"],
             [["none, the core", "0.8637 ± 0.0027", "3"],
              ["+5,000", "0.8609 ± 0.0036", "3"],
              ["+15,000", "0.8580 ± 0.0047", "3"],
              ["+40,000", "0.8448 ± 0.0018", "3"]], [0.44, 0.40, 0.16], hi=3)
y2 = d.body(x, y2, w,
    "Twelve times the data costs 0.0189, which is 8.2 times the seed spread, and the same eight "
    "checkpoints fall on a second dataset too. One of six audited web sources clears a 0.90 "
    "label-precision bar. This is why the robots exist.", size=CAPTION, color=MUTE)


# ------------------------------------------------------- ledger and supervision
y = max(y0, y1, y2) + 0.50
yc = d.head(COL[1][0], y, COL[1][1], "Everything we trained, ran, or rejected")
yc = d.figure(COL[1][0], yc, COL[1][1], "m_ledger", "Figure 6.")

yl = d.head(COL[0][0], y, COL[0][1], "Six audited web sources")
yl = d.body(COL[0][0], yl, COL[0][1],
    "Of 13,527 labelled images from six audited harvested sources, only 3,208 come from a source "
    "that clears a 0.90 label-precision bar. The other five score between 0.18 and 0.74. The same "
    "probe reads 1.000 on human-labelled CottonWeedDet12, so those are the data and not the "
    "instrument.", size=CAPTION, color=MUTE)

yr = d.head(COL[2][0], y, COL[2][1], "Who notices when it breaks")
yr = d.table(COL[2][0], yr, COL[2][1], ["reviewer", "recall", "grounded"],
             [["scripted watchdog", "no decision", "—"],
              ["12 rules", "0.095", "0.095"],
              ["Qwen2.5-7B", "0.823", "0.274"],
              ["Qwen3-14B", "0.675", "0.614"],
              ["Qwen3.8-27B", "0.702", "0.702"]], [0.46, 0.27, 0.27], hi=4)
yr = d.body(COL[2][0], yr, COL[2][1],
    "162 real failures out of this project's own record, frozen as a corpus. The watchdog the "
    "loop was running returned no decision on any of the 149 dev cases. Recall does not order "
    "the reviewers by size; evidence does.", size=CAPTION, color=MUTE)

# -------------------------------------------------------------------- footer
y = max(yl, yc, yr) + 0.55
d.rect(FULL[0], y - 0.28, FULL[1], 0.022, fill=RULE)
fw = (FULL[1] - 1.0) / 3.0
d.body(FULL[0], y, fw,
    "Where this stands today. The collector and trainer are paused after a stop-loss on "
    "2026-08-29 and the last field drive was the same day. The supervisor is advisory. The laser "
    "has never fired at a weed under our control.", size=CAPTION, color=MUTE)
d.body(FULL[0] + fw + 0.5, y, fw,
    "How the numbers were made. Every in-domain number is mAP50-95 on the same 1,977 images under "
    "one evaluator, and that set is also the validation set during training, so each figure is a "
    "maximum over many evaluations. We measured that optimism at +0.002 to +0.017.",
    size=CAPTION, color=MUTE)
d.body(FULL[0] + 2 * (fw + 0.5), y, fw,
    "Numbers we withdrew. 0.910 was a data leak, found by a check we built after publishing it. "
    "0.9033 is the best of four unseeded runs. An earlier reading of the ladder as a flat curve "
    "is retracted.", size=CAPTION, color=MUTE)

print("  row1 %.2f | row2 %.2f | end %.2f" % (max(y0, y1, y2), max(yl, yc, yr), y + 1.6))
d.save(os.path.join(HERE, "draft_C_field.pptx"))
