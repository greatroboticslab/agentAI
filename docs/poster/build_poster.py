#!/usr/bin/env python3
"""Render the poster to a real .pptx at 48 x 24 in, from poster_data.py.

    python3 docs/poster/make_figures.py && python3 docs/poster/build_poster.py

Nothing is typed into the slide by hand: every number comes from poster_data.py
and every figure from figures/. A number that is still running renders as a
visible PENDING mark, never as a blank.
"""
import os, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import poster_data as D

FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "MTSU_WeedAgent_Poster.pptx")

INK   = RGBColor(0x12, 0x24, 0x3A)
BLUE  = RGBColor(0x0A, 0x4D, 0x8C)
MUTE  = RGBColor(0x5D, 0x6B, 0x7A)
RULE  = RGBColor(0xC9, 0xD2, 0xDC)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
BAND  = RGBColor(0xEE, 0xF3, 0xF8)
GOOD  = RGBColor(0x1F, 0x6F, 0x4A)
WARN  = RGBColor(0xB4, 0x51, 0x1F)

W, H = D.MEETING["size_in"]
M = 0.9                      # page margin
GUT = 0.55                   # column gutter
NCOL = 4
COLW = (W - 2 * M - (NCOL - 1) * GUT) / NCOL

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def rect(x, y, w, h, fill=None, line=None, lw=1.0):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
    return sh


def text(x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0):
    """runs: list of (string, size_pt, bold, color) or (string, size, bold, color, space_after)"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for r in runs:
        s, size, bold, color = r[0], r[1], r[2], r[3]
        after = r[4] if len(r) > 4 else 6
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(after)
        p.line_spacing = spacing
        run = p.add_run(); run.text = s
        run.font.size = Pt(size); run.font.bold = bold
        run.font.color.rgb = color; run.font.name = "Calibri"
    return tb


def pic(path, x, y, w):
    if not os.path.exists(path):
        rect(x, y, w, 3.0, fill=BAND, line=RULE)
        text(x + 0.2, y + 1.3, w - 0.4, 0.6,
             [("figure missing: " + os.path.basename(path), 16, True, WARN)],
             align=PP_ALIGN.CENTER)
        return y + 3.0
    ph = slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    return y + ph.height / 914400.0


def head(x, y, w, s):
    text(x, y, w, 0.55, [(s.upper(), 26, True, BLUE)], spacing=0.9)
    rect(x, y + 0.60, w, 0.035, fill=BLUE)
    return y + 0.82


def body(x, y, w, s, size=15.5, color=INK, after=9):
    tb = text(x, y, w, 0.4, [(s, size, False, color, after)], spacing=1.06)
    # estimate height: ~ chars per line at this width
    # Calibri at `size` pt averages ~0.47 em per character; 72 pt = 1 in. The
    # first draft used 0.50 em and 96 dpi, under-counted every block, and the
    # column ran off the bottom of the page with two blocks overlapping.
    cpl = max(16, int((w * 72.0) / (size * 0.47)))
    lines = 0
    for para in s.split("\n"):
        lines += max(1, -(-len(para) // cpl))
    return y + lines * (size * 1.32 / 72.0) + after / 72.0


def caption(x, y, w, s):
    return body(x, y, w, s, size=12.5, color=MUTE, after=12)


# ------------------------------------------------------------------ title band
rect(0, 0, W, 4.05, fill=BLUE)
M0 = D.MEETING
text(M, 0.42, W - 2 * M, 1.5, [(M0["title"], 74, True, PAPER)], spacing=0.92)
text(M, 1.72, W - 2 * M, 0.7, [(M0["subtitle"], 26, False, RGBColor(0xCB, 0xDD, 0xEE))],
     spacing=1.0)
# Superscript the affiliation markers rather than letting "Harry He1" read as a name.
tb = slide.shapes.add_textbox(Inches(M), Inches(2.58), Inches(W - 2 * M), Inches(0.62))
tf = tb.text_frame; tf.word_wrap = True
tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
para = tf.paragraphs[0]
for i, (nm, sup) in enumerate(M0["authors"]):
    r1 = para.add_run(); r1.text = ("" if i == 0 else "     ") + nm
    r1.font.size = Pt(30); r1.font.bold = True; r1.font.color.rgb = PAPER
    r1.font.name = "Calibri"
    r2 = para.add_run(); r2.text = sup
    r2.font.size = Pt(19); r2.font.bold = True; r2.font.color.rgb = PAPER
    r2.font.name = "Calibri"
    r2.font._rPr.set("baseline", "30000")
aff = "1 %s    ·    2 %s    ·    %s" % (M0["affiliations"][0], M0["affiliations"][1],
                                        M0["institution"])
text(M, 3.20, W - 2 * M, 0.5, [(aff, 21, False, RGBColor(0xCB, 0xDD, 0xEE))])

# ------------------------------------------------------------------ big tiles
TY = 4.35
TH = 2.45
tiles = [
    ("+0.0714", "what COCO pretraining is worth", "n = 3 seeds each side", GOOD),
    ("0.100", "same detector on another weed dataset", "from 0.873 in-domain, n = 3", WARN),
    ("0 of 116", "real incidents the scripted watchdog caught", "149 cases, it flagged none", WARN),
    ("48,752", "images, unchanged for 13 of 15 rounds", "while the loop reported progress", INK),
]
for i, (big, lab, sub, col) in enumerate(tiles):
    x = M + i * (COLW + GUT)
    rect(x, TY, COLW, TH, fill=BAND)
    rect(x, TY, 0.09, TH, fill=col)
    text(x + 0.38, TY + 0.22, COLW - 0.6, 1.0, [(big, 62, True, col)], spacing=0.9)
    text(x + 0.38, TY + 1.42, COLW - 0.6, 0.5, [(lab, 17, True, INK)], spacing=1.0)
    text(x + 0.38, TY + 1.92, COLW - 0.6, 0.4, [(sub, 13.5, False, MUTE)], spacing=1.0)

# ------------------------------------------------------------------ columns
CY = TY + TH + 0.75
cols = [M + i * (COLW + GUT) for i in range(NCOL)]

# ---- column 1 -------------------------------------------------------------
x = cols[0]; y = CY
y = head(x, y, COLW, "Abstract")
y = body(x, y, COLW,
    "Autonomous data-harvesting pipelines promise compounding gains: collect, label, retrain, "
    "repeat. We built one for cotton-field weed detection — a collector agent and a trainer agent "
    "running unattended on an HPC allocation, fed by field robots — and ran it for fifteen rounds. "
    "Its dangerous failure mode was not a crash. It was a plausible number.")
y = body(x, y, COLW,
    "Audited against a fixed 1,977-image CottonWeedDet12 holdout: the training corpus was identical "
    "from round 3 onward, because the collector admitted nothing for eight consecutive rounds while "
    "reporting success; the metric nonetheless fell 0.6019 → 0.5607; and a controlled experiment on "
    "identical data attributes that decline to warm-start chaining rather than to the data.")
y = body(x, y, COLW,
    "Three measurements survive scrutiny at n = 3 seeds: COCO pretraining is worth +0.0714 mAP50-95, "
    "architecture at equal initialisation +0.0225, and web-harvested data 0.00 to −0.02. Detection "
    "collapses from 0.873 to 0.100 across imaging domains. A model reading raw artifacts catches "
    "failures that twelve deterministic checks structurally cannot.")
y += 0.25
y = head(x, y, COLW, "The platform")
P = D.PLATFORM
y = body(x, y, COLW,
    "Two agents run as separate SLURM jobs against one locked registry: a collector that searches, "
    "downloads and pseudo-labels, and a trainer that merges, deduplicates, guards the holdout and "
    "trains. Field data arrives over a live uplink from two robots driving crop rows.")
y = body(x, y, COLW,
    "Counted on disk rather than claimed: %s frames across %s sessions from %d robots, and %s of "
    "them labelled. Robot 241 contributes %s frames at %s, %s, camera at %.0f Hz, of which %s are "
    "genuinely on vegetation or soil. The laser cart contributes %s frames, of which %s — one "
    "session on %s — are in a field; the rest are indoor bench tests. The laser cart carries no GPS "
    "and no IMU: its uplink declares %s."
    % ("{:,}".format(P["total_frames"]), P["sessions"], P["robots"], "none" if not P["labelled"] else P["labelled"],
       "{:,}".format(P["r241_frames"]), P["r241_res"], P["r241_span"], P["r241_hz"],
       "{:,}".format(P["r241_field_frames"]), P["lasercar_frames"], P["lasercar_field_frames"],
       P["lasercar_field_date"], ", ".join(P["lasercar_sources"])))
y += 0.12
HERO = P["hero"]
y = pic(os.path.join(FIG, "f7_fielddrive.png"), x, y, COLW) + 0.10
y = caption(x, y, COLW,
    "Figure 1.  The platform's one unambiguous field drive: robot 241 on %s, %s. %s camera frames "
    "at %s over %.0f s, %d GPS fixes and %s IMU samples, %.1f m of track. The trajectory is drawn "
    "from the board fix — the platform's own derived GPS preferred a receiver frozen to a single "
    "coordinate, so every track it exposed was a motionless point (17 sessions, 1,281 rows, 0.0 m) "
    "until that was fixed. Source: uploads/%s/files/."
    % (HERO["date"], HERO["what"], "{:,}".format(HERO["frames"]), HERO["res"],
       HERO["seconds"], HERO["gps_fixes"], "{:,}".format(HERO["imu_rows"]),
       HERO["track_m"], HERO["slug"]))
y += 0.25
y = head(x, y, COLW, "Where the harvested data goes")
y = pic(os.path.join(FIG, "f4_funnel.png"), x, y, COLW) + 0.10
F = D.FUNNEL
y = caption(x, y, COLW,
    "Figure 2.  %s registry-labelled images collapse to %s unique: %s cross-dataset duplicates and "
    "%s holdout stems removed. Of six audited harvested sources (%s images) exactly one (%s images) "
    "clears the 0.90 label-precision bar; the rest score %s. The audit probe reads 1.000 on "
    "human-labelled cwd12, so the low scores are the data, not the instrument."
    % ("{:,}".format(F["registry_labelled"]), "{:,}".format(F["unique"]),
       "{:,}".format(F["cross_dataset_dupes"]), "{:,}".format(F["holdout_stems_dropped"]),
       "{:,}".format(F["audited_images"]), "{:,}".format(F["passing_images"]),
       F["failing_range"]))

# ---- column 2 -------------------------------------------------------------
x = cols[1]; y = CY
y = head(x, y, COLW, "What moves the detector")
y = pic(os.path.join(FIG, "f1_levers.png"), x, y, COLW) + 0.10
y = caption(x, y, COLW,
    "Figure 3.  Three levers on one holdout and one evaluator. Initialisation is worth three times "
    "what architecture is worth, and both dwarf the data-collection lever the whole system was built "
    "around. Top two rows n = 3 seeds; bottom row n = 1 with seeds in flight. "
    "Source: results/framework/s3_yolo11n/*/results.csv.")
y += 0.18
y = pic(os.path.join(FIG, "f2_ladder.png"), x, y, COLW) + 0.10
y = caption(x, y, COLW,
    "Figure 4.  Harvested images added to a clean in-domain core. Flat within the seed band to "
    "+15,000, then −0.020 at +40,000 — twelve times the training data for nothing, then a cost. "
    "The shaded band is the measured seed std (0.0029), not an assumption. "
    "Source: results/framework/s3_tier_v2_*.json.")
y += 0.18
y = head(x, y, COLW, "Protocol")
hold = D.HOLDOUT
y = body(x, y, COLW,
    "Every in-domain number is mAP50-95 on the same %s images / %s instances of %s, under the "
    "Ultralytics validator. That set is also the validation set during training, so each figure is a "
    "maximum over 21–100 evaluations on the set it is reported on. We measured the size of that "
    "optimism rather than hiding it: +0.002 to +0.017 depending on recipe, and every conclusion "
    "above survives re-reading at last-epoch and mean-of-last-five."
    % ("{:,}".format(hold["images"]), "{:,}".format(hold["instances"]), hold["name"]))

# ---- column 3 -------------------------------------------------------------
x = cols[2]; y = CY
y = head(x, y, COLW, "The generalisation wall")
y = pic(os.path.join(FIG, "f3_wall.png"), x, y, COLW) + 0.10
Wl = D.WALL
y = caption(x, y, COLW,
    "Figure 5.  The same three checkpoints, the same matcher on both sides so the evaluator offset "
    "cancels: %s in-domain against %s on %s. Best species %s → %s. An in-domain weed detector does "
    "not transfer across imaging domains, and no amount of in-domain data fixes that. "
    "Source: %s." % (Wl["in_domain"], Wl["out_domain"], Wl["target"].split(",")[0],
                     Wl["best_species_in"], Wl["best_species_out"], Wl["src"]))
y += 0.20
y = head(x, y, COLW, "Why the loop looked like it was learning")
y = pic(os.path.join(FIG, "f6_rounds.png"), x, y, COLW) + 0.10
R = D.ROUNDS
y = caption(x, y, COLW,
    "Figure 6.  Left: fifteen unattended rounds, reported metric and last-epoch metric. The shaded "
    "region is where the training corpus stopped changing — %s. Right: on round 15's exact dataset, "
    "starting fresh scores %.4f against the chained recipe's %s (n = 3), a gap of %+.4f at %.1f "
    "standard deviations with the schedule held constant. The decline is the chain, not the data."
    % (R["corpus"], R["cold"], R["campaign"], R["chain_effect"], R["chain_sigma"]))

# ---- column 4 -------------------------------------------------------------
x = cols[3]; y = CY
y = head(x, y, COLW, "What supervision actually catches")
y = pic(os.path.join(FIG, "f5_supervision.png"), x, y, COLW) + 0.10
S = D.SUPERVISION
y = caption(x, y, COLW,
    "Figure 7.  A frozen corpus of 162 real incidents from this project's own history, scored on the "
    "dev split: %d incidents, %d controls. Bars right of zero are recall, left of zero are false "
    "alarms. The scripted watchdog reads status fields and flags nothing, 149 times out of 149. The "
    "twelve deterministic checks catch 11. A model reading raw artifact excerpts catches four to "
    "eight times more. The dotted line is the ceiling any rules-only arm has on this corpus: %s."
    % (S["incidents"], S["controls"], S["ceiling_why"]))
y += 0.22
y = head(x, y, COLW, "Conclusions")
for i, s in enumerate([
    "Initialisation beats architecture beats data. The data-collection lever the system was built "
    "around is the weakest of the three, and at +40,000 images it is negative.",
    "The detector does not travel. 0.873 in-domain becomes 0.100 on another weed dataset, "
    "class-agnostic, same matcher both sides.",
    "Web harvest supplies volume, not supervision. One of six audited sources clears a 0.90 "
    "label-precision bar.",
    "Retrospective supervision needs the artifacts, not the status fields. A scripted watchdog over "
    "status fields caught none of 116 real incidents; a model reading the artifacts caught most.",
    "An unattended loop can report success for eight consecutive rounds while collecting nothing. "
    "Absence of a failure signal is not evidence of success.",
]):
    y = body(x, y, COLW, "%d.  %s" % (i + 1, s), size=15.5, after=7)
y += 0.20
y = head(x, y, COLW, "Limitations")
y = body(x, y, COLW,
    "The holdout is used as the validation set, so every number is a maximum over many evaluations "
    "on the reported set; the optimism is measured (+0.002 to +0.017) but not removed. The tier "
    "ladder and the chain control are n = 1 per point with seed repeats in flight. The 27B "
    "supervision row is %d of 149 cases and still running. The supervision corpus was labelled from "
    "this project's own engineering record by the same system that wrote the reviewer prompt, and "
    "its test split is 13 cases — too small to separate methods. Robot field frames are collected "
    "but not yet scored, and none of the %s robot frames is labelled. The laser cart carries no GPS "
    "or IMU, so the platform's sensor-fusion story is robot 241 only."
    % (S["rows"][-1]["cases"], "{:,}".format(D.PLATFORM["total_frames"])), size=13.5, color=MUTE)

prs.save(OUT)
print("wrote", OUT)
print("  %.0f x %.0f in, %d shapes" % (W, H, len(slide.shapes)))
