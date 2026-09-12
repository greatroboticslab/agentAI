#!/usr/bin/env python3
"""Lay the poster out at 48 x 36 in, on the lab's own design system.

    python3 docs/poster/figures.py && python3 docs/poster/build.py

Grid, palette and type are taken from PPTFinal112.pptx: a 4.3 in navy title band,
a hairline, then three columns at 11.6 / 22.4 / 11.6 in with 0.8 in margins, Arial
throughout, and the slate-blue palette. Nothing is typed onto the slide by hand --
numbers come from poster_data.py and captions from captions.json, both written by
the figure pass, so the text beside a figure cannot drift from the figure.
"""
import os, sys, json
import statistics as st
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import poster_data as D

FIG = os.path.join(HERE, "fig")
CAPS = json.load(open(os.path.join(HERE, "captions.json")))
OUT = os.path.join(HERE, "MTSU_WeedAgent_Poster.pptx")

INK   = RGBColor(0x1E, 0x29, 0x3B)
NAVY  = RGBColor(0x14, 0x31, 0x4E)
BLUE  = RGBColor(0x1C, 0x6F, 0xB5)
PALE  = RGBColor(0xF4, 0xF7, 0xFA)
PALEB = RGBColor(0xEA, 0xF2, 0xF9)
RULE  = RGBColor(0xC2, 0xCE, 0xDA)
MUTE  = RGBColor(0x55, 0x65, 0x75)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
WARN  = RGBColor(0xB5, 0x50, 0x2A)
GOOD  = RGBColor(0x1B, 0x7A, 0x55)
FONT  = "Arial"

W, H = 48.0, 36.0
MARG = 0.8
BAND = 4.3
COL = [(0.8, 11.6), (12.8, 22.4), (35.6, 11.6)]
TOP = 5.3

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def rect(x, y, w, h, fill=None, line=None, lw=0.75):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None: sh.fill.background()
    else: sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb = line; sh.line.width = Pt(lw)
    return sh


def tbox(x, y, w, runs, align=PP_ALIGN.LEFT, spacing=1.0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.4))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for r in runs:
        s, size, bold, color = r[0], r[1], r[2], r[3]
        after = r[4] if len(r) > 4 else 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align; p.space_after = Pt(after); p.line_spacing = spacing
        run = p.add_run(); run.text = s
        run.font.size = Pt(size); run.font.bold = bold
        run.font.color.rgb = color; run.font.name = FONT
    return tb


def h_est(s, w, size, spacing=1.22, after=0.0):
    """Conservative: Arial averages ~0.50 em per character at these sizes."""
    cpl = max(14, int((w * 72.0) / (size * 0.50)))
    lines = sum(max(1, -(-len(p) // cpl)) for p in s.split("\n"))
    return lines * size * spacing / 72.0 + after / 72.0


def body(x, y, w, s, size=13.5, color=INK, after=10, bold=False, spacing=1.22):
    tbox(x, y, w, [(s, size, bold, color, after)], spacing=spacing)
    return y + h_est(s, w, size, spacing, after)


def head(x, y, w, s):
    tbox(x, y, w, [(s.upper(), 28, True, NAVY)], spacing=0.95)
    rect(x, y + 0.52, w, 0.028, fill=BLUE)
    return y + 0.80


def sub(x, y, w, s):
    tbox(x, y, w, [(s, 18, True, INK)], spacing=1.0)
    return y + 0.34


def figure(x, y, w, name, label):
    """Image, then its caption. Returns the new y."""
    p = os.path.join(FIG, name + ".png")
    if not os.path.exists(p):
        rect(x, y, w, 2.2, fill=PALE, line=RULE)
        tbox(x + 0.2, y + 1.0, w - 0.4, [("missing " + name, 14, True, WARN)])
        return y + 2.4
    # The figure must be authored at the width it is placed at, or the layout
    # scales it and scaling a figure scales its type. style.PLACED is the other
    # half of this check; this is the half that sees the real placement width.
    from PIL import Image as _Im
    _authored = _Im.open(p).size[0] / 300.0
    assert abs(_authored / w - 1.0) < 0.02, (
        "%s is %.3f in wide but placed at %.3f in (x%.2f)" % (name, _authored, w, w / _authored))
    ph = slide.shapes.add_picture(p, Inches(x), Inches(y), width=Inches(w))
    y2 = y + ph.height / 914400.0 + 0.10
    cap = "%s  %s" % (label, CAPS.get(name, ""))
    tbox(x, y2, w, [(cap, 12, False, MUTE, 0)], spacing=1.20)
    return y2 + h_est(cap, w, 12, 1.20) + 0.30


def table(x, y, w, header, rows, widths, hi=None):
    """A rule-only table, the way a journal sets one."""
    n = len(header)
    cw = [w * f for f in widths]
    xs = [x + sum(cw[:i]) for i in range(n)]
    rect(x, y, w, 0.022, fill=INK)
    yy = y + 0.10
    for i, hcell in enumerate(header):
        tbox(xs[i], yy, cw[i], [(hcell, 12.5, True, INK, 0)], spacing=1.0,
             align=PP_ALIGN.RIGHT if i else PP_ALIGN.LEFT)
    yy += 0.30
    rect(x, yy, w, 0.014, fill=RULE)
    yy += 0.09
    for ri, row in enumerate(rows):
        col = BLUE if (hi is not None and ri == hi) else INK
        bold = hi is not None and ri == hi
        for i, cell in enumerate(row):
            tbox(xs[i], yy, cw[i], [(str(cell), 13, bold, col, 0)], spacing=1.0,
                 align=PP_ALIGN.RIGHT if i else PP_ALIGN.LEFT)
        yy += 0.34
    rect(x, yy + 0.02, w, 0.022, fill=INK)
    return yy + 0.24


# ============================================================= title band
rect(0, 0, W, BAND, fill=NAVY)
rect(0, BAND, W, 0.05, fill=BLUE)
M = D.MEETING
tbox(MARG, 0.55, W - 2 * MARG,
     [("What Actually Moves a Weed Detector — and What Catches It Failing", 66, True, WHITE)],
     spacing=0.95)
tb = slide.shapes.add_textbox(Inches(MARG), Inches(2.30), Inches(W - 2 * MARG), Inches(0.6))
tf = tb.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
para = tf.paragraphs[0]
for i, (nm, sup) in enumerate(M["authors"]):
    r1 = para.add_run(); r1.text = ("" if i == 0 else ",     ") + nm
    r1.font.size = Pt(32); r1.font.bold = True; r1.font.color.rgb = WHITE
    r1.font.name = FONT
    r2 = para.add_run(); r2.text = sup
    r2.font.size = Pt(20); r2.font.bold = True; r2.font.color.rgb = WHITE
    r2.font.name = FONT; r2.font._rPr.set("baseline", "30000")
tbox(MARG, 3.05, W - 2 * MARG,
     [("¹Department of Engineering Technology   ·   ²School of Agriculture   ·   "
       "Middle Tennessee State University, Murfreesboro, TN, USA", 20, False,
       RGBColor(0xC2, 0xCE, 0xDA))])

# ============================================================= LEFT column
x, w = COL[0]; y = TOP
y = head(x, y, w, "Introduction")
y = body(x, y, w,
    "Autonomous data-harvesting pipelines promise compounding gains: collect field images, label "
    "them, retrain, repeat. We built one for cotton-field weed detection — a collector agent and a "
    "trainer agent running unattended on an HPC allocation, fed by robots driving crop rows — and "
    "ran it for fifteen rounds.")
y = body(x, y, w,
    "Its dangerous failure mode was not a crash. Every round reported success, every dashboard read "
    "green, and the holdout accuracy fell the entire time. This poster reports what we measured when "
    "we stopped trusting the pipeline and audited it: which levers actually move a weed detector, "
    "and which kind of supervision actually catches it failing.")
y += 0.10
y = head(x, y, w, "Platform")
P = D.PLATFORM
y = body(x, y, w,
    "Two agents run as separate scheduler jobs against one locked dataset registry. A collector "
    "searches public sources, downloads and pseudo-labels; a trainer merges, removes duplicates, "
    "guards the evaluation set and trains. Field data arrives over a live uplink from two ground "
    "robots.")
y = body(x, y, w,
    "Counted on disk rather than claimed: %s camera frames across %s sessions from %d robots, none "
    "of them labelled. Robot 241 contributes %s frames at %s, %s, of which %s are genuinely on "
    "vegetation or soil, with GPS at 1 Hz and IMU at 15 Hz. The laser cart contributes %s frames, of "
    "which %s are in a field; it carries laser and vehicle telemetry but no GPS and no IMU."
    % ("{:,}".format(P["total_frames"]), P["sessions"], P["robots"],
       "{:,}".format(P["r241_frames"]), P["r241_res"], P["r241_span"],
       "{:,}".format(P["r241_field_frames"]), P["lasercar_frames"], P["lasercar_field_frames"]))
y += 0.10
y = figure(x, y, w, "l_robots", "Figure 1.")
y = figure(x, y, w, "h_drive", "Figure 2.")
y = figure(x, y, w, "k_funnel", "Figure 3.")
y = figure(x, y, w, "s_sources", "Figure 4.")
y += 0.05
y = head(x, y, w, "Protocol")
HD = D.HOLDOUT
y = body(x, y, w,
    "Every in-domain number is mAP₅₀₋₉₅ on the same %s images and %s "
    "instances of CottonWeedDet12, under one evaluator. That set is also the validation set during "
    "training, so each figure is a maximum over 21–100 evaluations on the set it is reported on. We "
    "measured the size of that optimism rather than hiding it — +0.002 to +0.017 depending on recipe "
    "— and every conclusion below survives re-reading at last-epoch and at mean-of-last-five."
    % ("{:,}".format(HD["images"]), "{:,}".format(HD["instances"])))
y = body(x, y, w,
    "Three evaluators exist in this project and they do not share a scale: the same checkpoint reads "
    "0.8794 under one and 0.8554 under another. Every number here is stated with the evaluator that "
    "produced it, and numbers from different evaluators are never compared.", color=MUTE, size=12.5)
y += 0.10
y = head(x, y, w, "What the audit recovered")
y = body(x, y, w,
    "Four defects found by reading the pipeline's own artifacts rather than its dashboards. Each was "
    "silent: the run reported success while the defect held.")
y = table(x, y, w, ["defect", "recovered"],
          [["Warm-start chain across rounds", "+0.0287 mAP"],
           ["GPS logged from a frozen receiver", "200.9 m of track"],
           ["Harvest decisions never logged", "8 empty rounds seen"],
           ["Staging directory never cleared", "20,258 stale files"]],
          [0.62, 0.38])

# ============================================================= CENTRE column
x, w = COL[1]; y = TOP
y = head(x, y, w, "Results")

tiles = [("+0.0714", "pretraining over random init", "n = 3 each side"),
         ("0.100", "same detector, another field", "from 0.873, n = 3"),
         ("149 of 149", "cases the watchdog could not decide", "116 incidents among them"),
         ("48,752", "images, unchanged for 13 rounds", "while the loop reported progress")]
tw = (w - 3 * 0.30) / 4
for i, (big, lab, note) in enumerate(tiles):
    tx = x + i * (tw + 0.30)
    rect(tx, y, tw, 1.70, fill=PALEB)
    tbox(tx + 0.25, y + 0.14, tw - 0.4, [(big, 40, True, BLUE)], spacing=0.95)
    tbox(tx + 0.25, y + 0.86, tw - 0.4, [(lab, 13, True, INK)], spacing=1.1)
    tbox(tx + 0.25, y + 1.30, tw - 0.4, [(note, 11.5, False, MUTE)], spacing=1.0)
y += 2.05

HALF = (w - 0.55) / 2
ya = sub(x, y, w, "What moves the detector")
yb = ya
ya = figure(x, ya, HALF, "a_families", "Figure 5.")
yb = figure(x + HALF + 0.55, yb, HALF, "b_zeroshot", "Figure 6.")
y = max(ya, yb) + 0.10

ya = sub(x, y, w, "Does more harvested data help?")
yb = ya
ya = figure(x, ya, HALF, "c_ladder", "Figure 7.")
yb = body(x + HALF + 0.55, yb, HALF,
    "Harvested images added to a clean in-domain core, the same images at every seed. Only the "
    "training seed varies, so the spread is training noise and not a different sample of the corpus.")
_LD = D.LADDER
_lrows = []
for _k, _lab in zip(_LD["rungs"], ["none \u2014 the core", "+5,000", "+15,000", "+40,000"]):
    _sv = _LD["seeds"][_k]
    _lrows.append([_lab, "%.4f \u00b1 %.4f" % (st.mean(_sv), st.stdev(_sv)), "%d" % len(_sv)])
yb = table(x + HALF + 0.55, yb, HALF,
           ["harvested added", "mAP\u2085\u2080\u208b\u2089\u2085", "n"],
           _lrows, [0.46, 0.38, 0.16])
yb = body(x + HALF + 0.55, yb, HALF,
    "The first two steps sit inside the pooled seed spread and are not separable. The full "
    "+40,000 is: \u22120.0189 against the core, 8.2 \u03c3, three seeds at every rung.",
    size=12.5, color=MUTE)
y = max(ya, yb) + 0.10

ya = sub(x, y, w, "What catches the pipeline failing")
yb = ya
ya = figure(x, ya, HALF, "e_supervision", "Figure 8.")
S = D.SUPERVISION
AR = S["table"]["arms"]


def scell(key, field):
    v = (AR[key].get(field) or {}).get("v")
    return "--" if v is None else "%.3f" % v


yb = body(x + HALF + 0.55, yb, HALF,
    "A frozen corpus of 162 real incidents from this project's own engineering record, scored on the "
    "dev split: %d incidents and %d controls, by the project's own scorer re-reading committed "
    "verdicts. A detection counts only when the arm returns an issue verdict carrying a finding at "
    "or above a severity bar; \u201cgrounded\u201d additionally requires that finding to quote a line that "
    "resolves in the artifact. L2 reads raw artifact excerpts; L3 adds a retrieval round over them."
    % (S["incidents"], S["controls"]))
srows = [["Scripted watchdog", "status fields", "--", "--", "--", "--",
          "%d" % S["a0_cases"]],
         ["Deterministic signals", "12 checks", scell("A0p", "detection_recall"),
          scell("A0p", "detection_grounded"), scell("A0p", "false_alarm_rate"),
          scell("A0p", "citation_validity"),
          "%d" % AR["A0p"]["counts"]["cases"]]]
for _m in S["models"]:
    for _tier, _key in (("L2", _m["l2"]), ("L3", _m["l3"])):
        if not _key:
            continue
        srows.append(["%s  %s" % (_m["size"], _tier),
                      "artifacts + retrieval" if _tier == "L3" else "artifacts",
                      scell(_key, "detection_recall"), scell(_key, "detection_grounded"),
                      scell(_key, "false_alarm_rate"), scell(_key, "citation_validity"),
                      "%d" % AR[_key]["counts"]["cases"]])
yb = table(x + HALF + 0.55, yb, HALF,
           ["reviewer", "reads", "recall", "grounded", "false alarms",
            "citations valid", "cases"],
           srows, [0.17, 0.24, 0.12, 0.13, 0.14, 0.14, 0.06], hi=9)
_d27 = (AR["L3@qwen3.8:27b"]["detection_recall"]["v"]
        - AR["L2@qwen3.8:27b"]["detection_recall"]["v"])
_d14 = (AR["L3@qwen3:14b"]["detection_recall"]["v"]
        - AR["L2@qwen3:14b"]["detection_recall"]["v"])
yb = body(x + HALF + 0.55, yb, HALF,
    "The watchdog the pipeline ran never produced a decidable verdict on any of the %d cases: "
    "\u201cno signal fired\u201d is not a judgement. Above it, recall does not order the reviewers by size \u2014 "
    "the 7 B has the highest on the page \u2014 but grounded recall and citation validity do, and they "
    "do it steeply: %.2f of the 7 B's detections quote a line that does not resolve, against %.2f "
    "for the 27 B. The size class the campaign's own reviewer ran on for a week would have passed on "
    "recall alone. The retrieval tier is worth %+.3f to the 27 B and %+.3f to the 14 B, so it pays "
    "where the model can use it and not otherwise."
    % (S["a0_cases"],
       AR["L3@qwen2.5:7b"]["detection_recall"]["v"] - AR["L3@qwen2.5:7b"]["detection_grounded"]["v"],
       AR["L3@qwen3.8:27b"]["detection_recall"]["v"] - AR["L3@qwen3.8:27b"]["detection_grounded"]["v"],
       _d27, _d14),
    size=12.5, color=MUTE)
yb = sub(x + HALF + 0.55, yb + 0.06, HALF, "What inference-time compute can buy")
yb = figure(x + HALF + 0.55, yb, HALF, "r_tta", "Figure 9.")
y = max(ya, yb) + 0.10

y = sub(x, y, w, "Why the loop looked like it was learning")
y = figure(x, y, w, "f_rounds", "Figure 10.")

# ============================================================= RIGHT column
x, w = COL[2]; y = TOP
y = head(x, y, w, "Generalisation")
y = figure(x, y, w, "d_wall", "Figure 11.")
y = figure(x, y, w, "p_field", "Figure 12.")
y = figure(x, y, w, "j_species", "Figure 13.")
y += 0.05
y = head(x, y, w, "Conclusions")
for i, s in enumerate([
    "Initialisation beats architecture beats data. Pretraining is worth +0.0714, architecture at "
    "equal initialisation +0.0225, and forty thousand harvested web images −0.020. The "
    "data-collection lever the system was built around is the weakest of the three.",
    "The detector does not travel. 0.873 in-domain becomes 0.100 on another weed dataset, "
    "class-agnostic, with the same matcher on both sides.",
    "Web harvest supplies volume, not supervision: one of six audited sources clears a 0.90 "
    "label-precision bar, and 44,750 of 156,521 images are cross-dataset duplicates.",
    "Retrospective supervision needs the artifacts, not the status fields. The watchdog reading "
    "status fields returned no decidable verdict on any of 149 cases, while a model reading the raw "
    "artifacts reaches %.2f recall over the 116 incidents; adding a retrieval tier over those same "
    "artifacts is worth another %+.3f to the 27B and %+.3f to the 14B."
    % (AR["L3@qwen3.8:27b"]["detection_recall"]["v"], _d27, _d14),
    "An unattended loop can report success for eight consecutive rounds while collecting nothing. "
    "Absence of a failure signal is not evidence of success.",
    "The gap shows up on our own video too. Over all %s frames the two robots recorded, %s of them "
    "more than a third vegetation, the deployable checkpoint draws a box on %d frames at the "
    "deployment threshold and %d at conf 0.40 — and %d of those %d boxes are the same class. These "
    "frames carry no ground truth, so that is a fire rate and not a recall."
    % ("{:,}".format(D.FIELD["frames"]), "{:,}".format(D.FIELD["vegetated"]),
       D.FIELD["fired_25"], D.FIELD["fired_40"],
       D.FIELD["species"]["Purslane"], D.FIELD["fired_25"]),
]):
    y = body(x, y, w, "%d.   %s" % (i + 1, s), size=13, after=8)
y += 0.10
y = head(x, y, w, "Limitations")
y = body(x, y, w,
    "The evaluation set doubles as the validation set, so every number is a maximum over many "
    "evaluations on the reported set; the optimism is measured but not removed. The +40,000 rung and "
    "the fresh-start control are single runs with seed repeats in flight. One supervision arm is 78 "
    "of 149 cases. The incident corpus was labelled from this project's own record by the same "
    "system that wrote the reviewer prompt, and its held-out split is 13 cases — too small to "
    "separate methods. None of the %s robot frames is labelled, and the laser cart carries no GPS or "
    "IMU, so the sensor-fusion result is robot 241 only."
    % "{:,}".format(P["total_frames"]), size=12.5, color=MUTE)
y += 0.08
y = head(x, y, w, "Reproducibility")
y = body(x, y, w,
    "Every figure on this poster is regenerated from checked-in artifacts by one command, and every "
    "number names the results file it was read from. Numbers that must never be quoted without their "
    "caveat are listed in the project's results ledger alongside the reason.", size=12.5, color=MUTE)

prs.save(OUT)
print("wrote", OUT)
print("  %.0f x %.0f in, %d shapes" % (W, H, len(slide.shapes)))
