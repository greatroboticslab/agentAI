#!/usr/bin/env python3
"""One poster, many looks.

    python3 docs/poster/gen.py sections.json --n 36 --out variants/ --render

The CONTENT is fixed and is the platform's own story: agentAI, the MTSU Great
Robotics Lab's research-data platform for physical agents; the robots on it
today and how a new one is added; live collection; remote control; the agent
that analyses what a collected dataset can be used for; the agents that iterate
and the brain that watches them; what six months of running it produced. Every
number is `poster_data`, every sentence is the section library, every figure is
`fig/`. What varies is the STYLE only: grid, palette, display face, title
treatment, opening image, density (styles.py).

Every figure is placed only in a slot matching the width it was authored at,
so a two-column look is 22.4 + 22.4 and a four-column look is four of 11.6.
If a column overflows the sheet its last section is dropped and
the drop is recorded in the manifest. --render tiles every PNG into a contact
sheet so forty looks can be compared on one screen.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deck import Deck, D, W, H, MARG, CAPS                 # noqa: E402
from pptx.enum.text import PP_ALIGN                        # noqa: E402
import styles                                              # noqa: E402

C, L2 = D.CENSUS, D.LEDGER
L = L2
FOOT_H = 1.45          # only a floor now; foot_h() measures the real thing.
                       # It was 2.55, which is what the footer needed when the
                       # three caveats ran to five lines each. A floor that high
                       # kept charging the columns for footer inches after the
                       # footer gave them back.

SLOT_OF_WIDTH = {11.6: "column", 22.4: "centre", 46.4: "full"}
FIGW = {}
FIGW_IN = {}
try:
    import style as _style
    FIGW = {k: SLOT_OF_WIDTH.get(v, "column") for k, v in _style.PLACED.items()}
    FIGW_IN = dict(_style.PLACED)
except Exception:
    pass

TITLE = "agentAI: A Research-Data Platform for Physical and Embodied Agents"
# The stand-first used to open with "ran for fifteen unattended rounds", which
# framed the whole sheet as a fifteen-round experiment. Fifteen is what has run
# so far, not what the loop is for. Checked in round_scheduler.py: the only
# bounds on the loop are max_rounds_per_day (a RATE, clamped 1..6 at :1642) and
# the stop-loss, which pauses a domain after two consecutive failed rounds
# (:1376) -- a brake, not a finish line. There is no total round cap anywhere,
# and no implemented stop condition at all, so the sentence says the loop is
# MEANT to stop on a reviewer's judgement rather than claiming it does. The
# count keeps its place in the loop block, next to what those rounds did.
STAND = ("On one platform, agents collect their own data, review one another's work and train "
         "without an operator. Nothing sets the number of rounds: the loop is meant to stop when "
         "the reviewer above it judges there is nothing further to be gained. This sheet reports "
         "how the agents coordinate and what the collection produced.")

# The one argument, in Harry's order. Section ids come from the library; a
# missing id is skipped, so this order can name sections before they exist.
SPINE = [
    ("One platform supports any robot or dataset",           ["platform_idea", "system", "platform_domains"]),
    ("robots-connected",        ["robots_today", "shared_models", "future_robots"]),
    ("live-collection",         ["uplink", "r241_frames", "cart_frames", "census",
                                 "drive", "auto_diag"]),
    ("remote-control",          ["remote_control", "drive_button", "advice", "laser_control"]),
    ("analysis-agent",          ["analysis_agent", "analysis_example", "analysis_sandbox",
                                 "analysis_gap"]),
    ("Agents ran the loop with nobody in the room",  ["loop", "brain", "dispatch", "watch", "supervisor"]),
    ("results",                 ["detect_grid", "ladder", "field", "families", "species",
                                 "zeroshot", "tta", "sources", "ledger", "journey"]),
]

# Drawn full width, straight under the opening, because four photographs in a
# row is the one thing a visitor reads before any prose.
BAND = "projects"
# The full-width plates that open the sheet, in order: the four projects, then
# the algorithm. "We do not have algorithm diagrams for the agent poster. One
# good looking diagram is fine." -- Hongbo Zhang.
BANDS = ["projects", "algorithm"]

# No rider line. It caveated numbers the sheet does not print: after the census
# strip and the ledger came off, no absolute mAP is printed anywhere in the
# prose, and it retracted 0.910 and 0.9033, which appear nowhere either -- a
# retraction of numbers the reader never saw is noise. The one clause that is
# still load-bearing went where the numbers actually are, into Figure 4's
# caption, which is the only place an mAP can be read off this sheet.
RIDER = ""

# A second full-width strip, drawn just above the closing when its plate has no
# column of its width in this grid -- which is every grid, because it is 46.4 in
# wide. Six months in order is an article's timeline, and it was the one plate
# the layout could never place.
TAIL_BANDS = ["detect_grid", "journey"]

# Drawn last, across the sheet, above the footer. The vision belongs at the end
# of the argument and not at the top of it: a reader should meet what the
# platform is and what it has done before being told where it goes.
# One closing block, not two. Two half-width blocks cost two 30 pt headings
# and twice the lines, for 2.6 in of a 36 in sheet the columns were
# shedding sections for want of. The second block's claim is inside the
# first one's paragraph.
# No closing band. It restated the title, the stand-first and the first
# column's heading a fourth time, in the future tense, for 2.5 in of a 36 in
# sheet. Its one idea is now the opening band's lede, where it is read.
CLOSING = []

# No footer band. Three columns of caveat type across the foot of the sheet is
# where a reader arrives last and squints; the live ones are the opening band's
# lede now, at body size, above the first plate.
FOOTER = []

# The house look does not run twenty numbered sections down a sheet. The lab's
# own poster puts ONE all-caps heading over ONE white panel per column and
# separates the blocks inside it with small accent-square sub-heads -- which is
# why it reads as an article and not as a list. These are the columns.
# Headings that assert rather than label. A reader walking the row of four
# should get the abstract: what it is, what it holds, what runs on it, what it
# measured. "The platform / Collecting in the field / What the agents do with it
# / What six months measured" labelled four containers and told them nothing,
# and two of the four opened with the same word.
# (heading, sections in shed order, how many of them the sheet may not give up)
#
# The last number is the editorial decision the packer cannot make. Without it
# a theme's share of the sheet is whatever is left when the themes before it
# have taken theirs, and the label campaign -- the longest theme and the reason
# this poster exists -- came off the sheet with one block standing. The counts
# say: two blocks for what the platform is, one for the reviewer, one for the
# unattended loop, five for the campaign -- nine in all. Nine blocks at 26 pt
# is a poster; fourteen at 19 pt is a page pinned to a board. Everything past the count is slack
# that a short column may take and a crowded one gives up first.
MTSU_COLUMNS = [
    # (heading, sections in shed order, how many the sheet may not give up)
    #
    # No accuracy plate. The presenting author's instruction is that the
    # detector's accuracy is not shown for now, so the tier ladder and the
    # round curve are off the sheet and the argument is carried by the two
    # things that are not accuracy: how the agents coordinate, and what the
    # collection actually produced.
    ("One platform supports any robot or dataset",
     ["platform_idea", "robots_today", "uplink", "platform_domains"], 2),
    ("Open-weight models review in tiers",
     ["supervisor", "escalate"], 1),
    # Renamed from "A judgement counts only where it runs", which was true of
    # dispatch and of nothing else in the column. When the theme spilled, the
    # continuation heading sat over the unattended-loop block and described
    # something the block does not say. The heading now covers all three.
    ("What the agents decided on their own",
     ["dispatch", "label_agent", "loop", "diagnosis"], 3),
    ("What the collection produced",
     ["funnel", "label_curator", "label_unit", "label_ceiling", "sources"], 3),
]

# The two plates that ARE the evidence for the two headline claims: what more
# data cost, with error bars and three seeds at every rung, and what separates
# a cheap reviewer from an expensive one. A research poster whose columns carry
# no plot has asserted its findings and shown none of them.
MTSU_PINNED = {"supervisor", "dispatch"}
MTSU_KEEP_FIGURE = {"supervisor", "dispatch"}

MTSU_PAD = 0.24            # the template's own panel padding, 0.26, less a hair


class Poster(object):
    def __init__(self, st, lib):
        self.st, self.lib = st, {s["id"]: s for s in lib}
        self.dropped = []
        # Sections whose figure has no column of its width in this grid: the
        # text runs, the plate does not, and the manifest says so.
        self.figless = []
        # Full-width plates, in the order they earn their inches. A sheet that
        # cannot hold them all keeps the ones nearer the front: the detection
        # strip is the evidence the models work, the timeline is context.
        # Decided before the layout runs, not during it. Changing the number of
        # full-width bands inside the drop loop makes the two fight: the loop
        # gives up blocks to fit two bands, then removes a band, and either the
        # blocks stay lost or the loop starts over and gives up the same ones
        # again. One band costs about 5.4 in of sheet, which is 21 in of column.
        # One full-width diagram, not three full-width bands. "We do not have
        # algorithm diagrams for the agent poster. One good looking diagram is
        # fine." The twelve-species strip cost 5.6 in and showed that the
        # detector fires, which the weed cell of the project plate already
        # shows; the algorithm is what this poster is about.
        self.tail_ids = ([] if getattr(st, "look", "") == "mtsu"
                         else list(TAIL_BANDS))
        self.d = Deck(TITLE, STAND, style=st)

    def slot_of(self, sid):
        s = self.lib[sid]
        f = s.get("figure") or "none"
        if f != "none" and not f.startswith("NEW") and f in FIGW:
            return FIGW[f]
        return None

    def draws_figure(self, sid, slot):
        f = self.lib[sid].get("figure") or "none"
        return f != "none" and not f.startswith("NEW") and FIGW.get(f) == slot

    def est_h(self, sid, w, slot):
        """About how many inches this section will take in a column of width w."""
        s, d = self.lib[sid], self.d
        h = d.dz["heading"] / 72.0 * 1.34 + 0.30 + d.dz["sec_gap"]
        if s.get("big"):
            h += 1.05
        if self.draws_figure(sid, slot):
            from PIL import Image
            fp = os.path.join(d.FIG, s["figure"] + ".png")
            if os.path.exists(fp):
                im = Image.open(fp)
                h += w * im.size[1] / float(im.size[0]) + 0.80
        h += d.h_est(s["body"], w, d.dz["body"], 1.22, 14)
        return h

    def section(self, x, y, w, sid, slot, fig_no):
        s, d = self.lib[sid], self.d
        # A section may carry one number set to be read from across a room. The
        # two themes Harry named first -- remote control and the analysis agent --
        # are where these sit, so a reader who only scans the sheet still meets
        # them.
        if s.get("big"):
            # An article sets a headline quantity as a small ruled entry above
            # the section, not as a 62 pt coloured numeral in the middle of a
            # column. The poster look keeps the numeral, because a poster is
            # read from four feet away and an article is not.
            if d.look == "journal":
                d.rect(x, y, w * 0.30, 0.020, fill=d.c["rule"])
                d.tbox(x, y + 0.16, w, [(s["big"], 36, True, d.c["ink"])],
                       spacing=0.95, face=d.f["display"])
                y = d.body(x, y + 0.16 + 36 / 72.0 * 1.04, w, s["big_label"],
                           size=d.dz["caption"], color=d.c["mute"], after=6,
                           justify=False) + 0.14
            else:
                d.tbox(x, y, w, [(s["big"], 62, True, d.c["accent"])], spacing=0.95,
                       face=d.f["display"])
                y += 62 / 72.0 * 1.02
                y = d.body(x, y, w, s["big_label"], size=d.dz["caption"],
                           color=d.c["mute"], after=6) + 0.16
        y = d.head(x, y, w, s["heading"])
        f = s.get("figure") or "none"
        if f != "none" and not f.startswith("NEW") and FIGW.get(f) == slot:
            y = d.figure(x, y, w, f, "Figure %d." % fig_no[0]); fig_no[0] += 1
        body = s["body"]
        if s.get("status") == "designed_only":
            body += " This part is a design and has not been built."
        elif s.get("status") == "built_but_off":
            body += " This part is built and is switched off today."
        y = d.body(x, y, w, body, after=14)
        return y + d.dz["sec_gap"]

    def census_cells(self):
        """The five quantities set large enough to read from across a room.

        Frames, telemetry rows and metres of track answer "how much did you
        collect", which anyone with a robot and an afternoon can answer, and
        every one of them appeared again in the body anyway. These five are
        findings: what the loop did on its own, what it cost, what caused it,
        what separates a cheap reviewer from an expensive one, and how much of
        the harvested web survives an audit.
        """
        R, L, S, F = D.ROUNDS, D.LADDER, D.SUPERVISION, D.FUNNEL
        if self.d.look != "mtsu":
            return [("{:,}".format(C["frames"]), "camera frames", "%d drives, 2 robots, %d labelled"
                     % (C["sessions"], C["labelled"])),
                    ("{:,}".format(C["sensor_rows"]), "telemetry rows", "nine streams, counted line by line"),
                    ("%.0f m" % C["gps_m"], "of GPS track", "%s fixes over %d drives"
                     % ("{:,}".format(C["gps_fixes"]), C["gps_sessions"])),
                    ("%d" % L2["n_models"], "distinct models run", "four families, one deployed"),
                    ("%.4f" % D.DEPLOYED["map50_95"], "mAP50-95, deployed model",
                     "all %s sealed holdout images" % "{:,}".format(D.DEPLOYED["n_images"]))]
        arms = S["table"]["arms"]
        g7 = arms["L3@qwen2.5:7b"]["detection_grounded"]["v"]
        g27 = arms["L3@qwen3.8:27b"]["detection_grounded"]["v"]
        best = next(r for grp in D.LEDGER["groups"] for r in grp["rows"]
                    if len(r) > 3 and r[3] == "deployed")
        # Five lines a visitor reads in four seconds. The professor could not
        # read the previous five: they opened on a negative, quoted sigmas in
        # the sub-line, and named an arm by its parameter count. Lead with what
        # was built and what it scores; say the loop's finding as a cause that
        # has a fix, because it has one.
        return [
            ("%d" % len(D.PROJECTS), "projects on one platform",
             "one ingest contract, one dataset registry"),
            (best[1].split()[0], "our weed detector, mAP50-95",
             "%s, on a sealed %s-image test set"
             % (best[2], "{:,}".format(D.HOLDOUT["images"]))),
            ("15", "rounds the agents ran alone",
             "nobody in the room, on the cluster"),
            ("%.2f vs %.2f" % (g27, g7), "which reviewer cites real evidence",
             "27 B against 7 B, on %d frozen cases" % S["table"]["n_cases"]),
            ("+%.4f" % R["chain_effect"], "the cause we found, and tested",
             "and the change the next campaign makes"),
        ]

    def census_row(self, y):
        d = self.d
        FULL = d.FULL
        cells = self.census_cells()
        cw = (FULL[1] - 4 * 0.55) / 5.0
        if d.look == "journal":
            # A ruled summary row, the way a paper prints its headline figures:
            # value and label on one line, the note under it, no colour. Set as
            # five giant coloured numerals it cost 2.35 in and was the loudest
            # thing on a sheet that is meant to read as an article.
            d.rect(FULL[0], y, FULL[1], 0.045, fill=d.c["ink"])
            ybot = y + 0.20
            for i, (v, lab, note) in enumerate(cells):
                x = FULL[0] + i * (cw + 0.55)
                d.tbox(x, y + 0.20, cw, [(v, 40, True, d.c["ink"])], spacing=1.0,
                       face=d.f["display"])
                yy = d.body(x, y + 0.20 + 40 / 72.0 * 1.06, cw, lab,
                            size=d.dz["caption"], after=0, justify=False)
                yy = d.body(x, yy, cw, note, size=d.dz["caption"] - 2,
                            color=d.c["mute"], after=0, justify=False)
                ybot = max(ybot, yy)
            d.rect(FULL[0], ybot + 0.10, FULL[1], 0.020, fill=d.c["rule"])
            return ybot + 0.34
        d.rect(FULL[0], y, FULL[1], 0.026, fill=d.c["rule"])
        ybot = y + 0.22
        for i, (v, lab, note) in enumerate(cells):
            ybot = max(ybot, d.bignum(FULL[0] + i * (cw + 0.55), y + 0.22, cw, v, lab, note, size=56))
        d.rect(FULL[0], ybot + 0.12, FULL[1], 0.026, fill=d.c["rule"])
        return ybot + 0.30

    def hero(self, fig_no):
        d, kind = self.d, self.st.hero
        # The four-project band carries photographs of three of the four
        # projects. Opening with the four-frame strip as well prints the 241 row
        # and the cart's forward camera twice on one sheet and costs 4.4 in of
        # height for the privilege.
        if kind == "photos" and BAND in self.lib:
            kind = "census"
        FULL, COL = d.FULL, d.COL
        y = d.top
        if kind == "photos":
            y = d.figure(FULL[0], y, FULL[1], "w_robots", "Figure %d." % fig_no[0]); fig_no[0] += 1
            y = self.census_row(y + 0.2)
        elif kind == "system":
            # the diagram needs a 22.4 slot; find one in this grid, else full-width census
            cen = [c for c in COL if abs(c[1] - 22.4) < 0.05]
            if cen:
                x, w = cen[0]
                y = d.figure(x, y, w, "n_system", "Figure %d." % fig_no[0]); fig_no[0] += 1
            else:
                y = self.census_row(y)
        elif kind == "census":
            y = self.census_row(y)
        return y + 0.36

    def band(self, y, fig_no):
        """The full-width plates that open the sheet, in order."""
        d = self.d
        x, w = d.FULL
        for sid in BANDS:
            s = self.lib.get(sid)
            if not s:
                continue
            f = s.get("figure") or "none"
            if abs(FIGW_IN.get(f, 0) - w) > 0.02:
                continue
            # A band either carries a section heading above its plate, or it is
            # a figure with its title underneath, which is what a diagram wants.
            if not s.get("title_below"):
                y = d.head(x, y, w, s["heading"])
            if s.get("lede"):
                y = d.body(x, y, w, s["lede"], after=10)
            y = d.figure(x, y, w, f, "Figure %d." % fig_no[0],
                         title=(s["heading"] if s.get("title_below") else None),
                         cap_size=d.dz["caption"] + 5)
            fig_no[0] += 1
            if s.get("body"):
                y = d.body(x, y, w, s["body"], after=10)
            y += d.dz["sec_gap"] * 0.6
        if RIDER:
            y = d.body(x, y, w, RIDER, size=d.dz["caption"] + 3,
                       color=d.c["mute"], after=8)
            y += 0.10
        return y

    def _full_bands(self):
        """Sections whose plate is 46.4 in wide: they can only run full width."""
        out = []
        for sid in self.tail_ids:
            s = self.lib.get(sid)
            if not s:
                continue
            f = s.get("figure") or "none"
            if abs(FIGW_IN.get(f, 0) - self.d.FULL[1]) < 0.02:
                out.append((sid, f))
        return out

    def tail_band_h(self):
        d = self.d
        from PIL import Image
        h = 0.0
        for sid, f in self._full_bands():
            fp = os.path.join(d.FIG, f + ".png")
            if not os.path.exists(fp):
                continue
            im = Image.open(fp)
            h += (self._head_h(d, d.FULL[1], self.lib[sid]["heading"])
                  + d.FULL[1] * im.size[1] / float(im.size[0]) + 0.95)
        return h

    def tail_band(self, y, fig_no):
        d = self.d
        x, w = d.FULL
        for sid, f in self._full_bands():
            s = self.lib[sid]
            y = d.head(x, y, w, s["heading"])
            y = d.figure(x, y, w, f, "Figure %d." % fig_no[0]); fig_no[0] += 1
            y += d.dz["sec_gap"] * 0.5
        return y

    def foot_h(self):
        """How tall the footer actually is at this type tier.

        It was a constant, and the type tier is not: at body 25.6 pt the three
        caveat blocks needed more than the 2.55 in reserved for them and ran
        off the bottom of the sheet, with only their first line printed.
        """
        d = self.d
        ids = [f for f in FOOTER if f in self.lib][:3]
        if not ids:
            return 0.0
        fw = (d.FULL[1] - 1.0) / 3.0
        size = d.dz["caption"]
        h = max(d.h_est("%s. %s" % (self.lib[s]["heading"], self.lib[s]["body"]),
                        fw, size, 1.22) for s in ids)
        return max(FOOT_H, h + 0.70)

    def closing_h(self):
        """How tall the closing block will be, so the columns can stop above it."""
        d = self.d
        ids = [i for i in CLOSING if i in self.lib]
        if not ids:
            return 0.0
        cw = (d.FULL[1] - 1.2 * (len(ids) - 1)) / len(ids)
        h = 0.0
        for sid in ids:
            s = self.lib[sid]
            body = s["body"]
            if s.get("status") == "designed_only":
                body += " This is the direction, not a description of what runs today."
            hh = (self._head_h(d, cw, s["heading"])
                  + d.h_est(body, cw, d.dz["caption"] + 1, 1.22, 12))
            h = max(h, hh)
        # the footer rule sits 0.25 above the footer text and the closing needs
        # air under it; an earlier estimate left the last two lines off the sheet
        return h + 0.72

    def closing(self, y):
        d = self.d
        ids = [i for i in CLOSING if i in self.lib]
        if not ids:
            return y
        x0, W0 = d.FULL
        d.rect(x0, y, W0, 0.045, fill=d.c["ink"])
        y += 0.26
        cw = (W0 - 1.2 * (len(ids) - 1)) / len(ids)
        ends = []
        for i, sid in enumerate(ids):
            s = self.lib[sid]
            x = x0 + i * (cw + 1.2)
            yy = d.head(x, y, cw, s["heading"])
            body = s["body"]
            if s.get("status") == "designed_only":
                body += " This is the direction, not a description of what runs today."
            ends.append(d.body(x, yy, cw, body, size=d.dz["caption"] + 1, after=10))
        return max(ends) + 0.30

    def mtsu_kpi(self, y):
        # The strip costs 2.07 in of sheet, which is 8 in of column -- three
        # blocks of argument. It earns that only where a reader is meant to take
        # five numbers away from across the room, so it rides on the `census`
        # opening rather than on every sheet.
        if self.st.hero != "census":
            return y
        """The template's summary strip, rebuilt from its own measurements.

        slide1.xml: a #C2CEDA hairline 0.014 in tall, the numbers in Arial bold
        accent centred in equal cells with no gaps, 0.012 in dividers on the
        cell boundaries, the labels in Arial 13 #4C5A69, a second hairline
        underneath, 1.294 in overall. Scaled here for a 36 in sheet.

        Every cell carries its denominator. A row of big percentages with no n
        is the most-named tell in generated decks, and it is unfalsifiable --
        which is why it is easy to write and useless to read.
        """
        d = self.d
        x0, W0 = d.FULL
        cells = self.census_cells()
        n = len(cells)
        cw = W0 / float(n)
        NUM, LAB, NOTE = 52, 17, 15
        d.rect(x0, y, W0, 0.020, fill=d.c["rule"])
        ytop = y + 0.22
        bottom = ytop
        for i, (v, lab, note) in enumerate(cells):
            cx = x0 + i * cw
            d.tbox(cx, ytop, cw, [(v, NUM, True, d.c["accent"])], spacing=0.98,
                   align=PP_ALIGN.CENTER, face="Arial")
            yy = ytop + NUM / 72.0 * 1.06
            d.tbox(cx, yy, cw, [(lab, LAB, False, d.c["ink"])], spacing=1.0,
                   align=PP_ALIGN.CENTER, face="Arial")
            yy += LAB / 72.0 * 1.42
            d.tbox(cx, yy, cw, [(note, NOTE, False, d.c["mute"])], spacing=1.10,
                   align=PP_ALIGN.CENTER, face="Arial")
            bottom = max(bottom, yy + d.h_est(note, cw, NOTE, 1.10))
            if i:
                d.rect(cx, ytop + 0.04, 0.014, bottom - ytop - 0.02, fill=d.c["rule"])
        d.rect(x0, bottom + 0.14, W0, 0.020, fill=d.c["rule"])
        return bottom + 0.52

    def mtsu_has_plate(self, sid, w):
        if sid in getattr(self, "_fig_off", ()):
            return False
        f = self.lib[sid].get("figure") or "none"
        return f != "none" and abs(FIGW_IN.get(f, 0) - w) < 0.02

    # The platform is reachable from outside the lab -- Tailscale Funnel is on
    # for lab-b660m-c, verified with `tailscale funnel status` and by resolving
    # the name against a public resolver. The code points at /login rather than
    # at / because / answers an anonymous request with a JSON 401, and a visitor
    # who scans a poster and gets `{"error":"unauthorized"}` is worse served
    # than one who scans nothing. /login is a real page: it names the platform,
    # says what it does, and says what signing in needs.
    QR_URL  = "lab-b660m-c.tailfa6424.ts.net"
    QR_FILE = "qr_platform.png"

    def qr_card(self, x, y, w, free=3.2):
        """The code, and what a stranger gets for scanning it. Returns the height."""
        import os
        from pptx.util import Inches
        d = self.d
        p = os.path.join(HERE, "photos", self.QR_FILE)
        if not os.path.exists(p):
            self.dropped.append("qr_card:missing-" + self.QR_FILE)
            return 0.0
        pad = MTSU_PAD
        # the code sizes itself to the corner it is given. Decoded from the
        # built PDF at 2.6, 2.2, 1.8 and 1.4 in, so 1.55 is still legible.
        side = max(1.55, min(2.45, free - 0.62))
        d.slide.shapes.add_picture(p, Inches(x + pad), Inches(y),
                                   width=Inches(side), height=Inches(side))
        tx = x + pad + side + 0.34
        tw = w - pad - (tx - x)
        yy = d.sub(tx, y + 0.08, tw, "The platform is live; the QR code links to it")
        yy = d.body(tx, yy, tw, self.QR_URL, after=4)
        d.body(tx, yy, tw, "Authentication requires an institutional Google account.",
               size=d.dz["caption"], color=d.c["mute"], after=0)
        return side

    def mtsu_block(self, x, y, w, sid, fig_no):
        """One sub-headed block inside a column panel.

        The plate runs the full panel width and the prose is inset, which is how
        the template sets it: a figure touching the hairline reads as a plate in
        a box, and prose touching it reads as a mistake.
        """
        s, d = self.lib[sid], self.d
        pad = MTSU_PAD
        tx, tw = x + pad, w - 2 * pad
        y = d.sub(tx, y, tw, s["heading"])
        f = s.get("figure") or "none"
        if sid in getattr(self, "_fig_off", ()):
            f = "none"
        if f != "none" and not f.startswith("NEW") and abs(FIGW_IN.get(f, 0) - w) < 0.02:
            y = d.figure(x, y + 0.04, w, f, "Figure %d." % fig_no[0]); fig_no[0] += 1
        # No status stamp. It appended the SAME eleven words to every block
        # marked built_but_off, so the sheet carried one sentence twice in one
        # reading -- a template, not prose -- and on a block that had just shown
        # a measured result it read as though the result no longer counted. It
        # was also invisible to mtsu_block_h, which measured the body without
        # it and under-counted the block by up to a line. What is true about a
        # block is written into that block, once, in its own words.
        y = d.body(tx, y, tw, s["body"], after=10)
        return y + 0.26

    def widow(self, w, sid):
        """A last line carrying one or two words, which is a typographic defect.

        Not an assertion: a widow is a quality fault, not a broken build, and
        the type tier moves under it. Reported per build instead, so it cannot
        sit on the sheet unnoticed the way the justified rivers did.
        """
        d = self.d
        tw = w - 2 * MTSU_PAD
        em = {"Times New Roman": 0.442, "Georgia": 0.478,
              "Arial Narrow": 0.425}.get(d.f["body"], 0.50)
        cpl = max(14, int((tw * 72.0) / (d.dz["body"] * em)))
        lines, cur = [], ""
        for word in self.lib[sid]["body"].split():
            t = word if not cur else cur + " " + word
            if len(t) <= cpl:
                cur = t
            else:
                lines.append(cur); cur = word
        if cur:
            lines.append(cur)
        if len(lines) < 2:
            return None
        last = lines[-1]
        if len(last.split()) <= 2 or len(last) / float(cpl) < 0.22:
            return "%s: last line is %r" % (sid, last)
        return None

    def mtsu_block_h(self, w, sid):
        s, d = self.lib[sid], self.d
        pad = MTSU_PAD
        tw = w - 2 * pad
        # the sub-head is Arial bold at the sub size, not Georgia at body size
        h = d.h_est(s["heading"], tw - 0.26, d.dz.get("sub", 21), 1.0,
                    em=d.SUB_EM) + 0.10 + 0.26
        f = s.get("figure") or "none"
        if sid in getattr(self, "_fig_off", ()):
            f = "none"
        if f != "none" and abs(FIGW_IN.get(f, 0) - w) < 0.02:
            from PIL import Image
            fp = os.path.join(d.FIG, f + ".png")
            if os.path.exists(fp):
                im = Image.open(fp)
                # Measure the caption. A flat 0.76 in allowance was two and a
                # half inches short once e_supervision's caption grew to five
                # lines, so the column was estimated at 28.8 and drawn at 31.2
                # and the footer went off the bottom of the sheet.
                cap = CAPS.get(f, "")
                cs = d.dz["caption"]
                cap_h = d.h_est("Figure 9.  " + cap, w, cs, 1.18) + 0.40
                h += w * im.size[1] / float(im.size[0]) + max(0.76, cap_h)
        return h + d.h_est(s["body"], tw, d.dz["body"], 1.22, 10)

    def _mtsu_layout(self, flow, n, COL, draw, demand=None):
        """Lay the flow out once. Returns (fits, ends, fig_no, spilled).

        With draw=False nothing is added to the slide, so a trial fit costs
        arithmetic instead of a rebuilt deck.
        """
        d = self.d
        fig_no = [1]
        # The same y0 either way. A trial that used an ESTIMATE of where the
        # columns start while the draw used the real position thought it had
        # room it did not, and a pinned plate that fitted in every trial fell
        # off the finished sheet.
        y0 = (self.band(self.mtsu_kpi(d.top), fig_no) if draw
              else self._mtsu_top_real())
        # 0.55 of clearance under the last column, not 0.95. closing_h() and
        # foot_h() both measure their own blocks now, so the constant is
        # clearance and nothing else, and every tenth of it is four tenths
        # of column across the sheet.
        limit = H - self.foot_h() - 0.55 - self.closing_h() - self.tail_band_h()
        avail = limit - y0

        heights = []
        for head, sid in flow:
            hh = self.mtsu_block_h(COL[0][1], sid)
            if head:
                # heading, plus the panel it opens: MTSU_PAD above the first
                # block and below the last, plus the 0.24 between two panels in
                # one column. Leaving these out let the packer fill a column to
                # 24.6 in against a 23.8 in page and report that it fitted.
                hh += self._head_h(d, COL[0][1], head) + 2 * MTSU_PAD + 0.24
            heights.append(hh)
        # Fill each column, then balance: an even share is only worth having
        # when there is enough content to go round, and capping at one left
        # every column three inches short while blocks spilled off the sheet.
        #
        # Balance against what the LIBRARY wants, not against what is left after
        # shedding. Measuring the share off the shed flow is circular: shed
        # enough and the share drops below the column, the 0.82 cap engages, and
        # every column stops two inches short -- eight inches of sheet held back
        # from sections that were dropped for want of room. Demand is taken once,
        # from the flow before anything is given up.
        share = ((sum(heights) if demand is None else demand) + 0.5 * n) / float(n)
        target = avail if share > avail else max(share, 0.82 * avail)

        # Which themes are still to come, so a theme can claim a fresh column
        # when there are exactly as many columns left as themes left. Without
        # this, filling an early column pushed a later theme into a column that
        # already held something, and a block that needs almost a whole column
        # -- the results plate is 9.1 in against a 9.5 in column -- fell off the
        # sheet. That is why backfilling any block anywhere cost the results.
        theme_seq = []
        for _h, sid in flow:
            th = self._theme_of(sid)
            if th not in theme_seq:
                theme_seq.append(th)
        cols = [[] for _ in range(n)]
        ci, y, carried, spilled = 0, y0, None, []
        for head, sid in flow:
            x, w = COL[ci]
            hh = self.mtsu_block_h(w, sid)
            if head:
                hh += self._head_h(d, w, head) + 2 * MTSU_PAD + 0.24
            if head and cols[ci] and ci < n - 1:
                th = self._theme_of(sid)
                left = len(theme_seq) - theme_seq.index(th)
                if left >= n - ci:
                    ci += 1
                    x, w = COL[ci]
                    y = y0
                    hh = (self.mtsu_block_h(w, sid) + self._head_h(d, w, head)
                          + 2 * MTSU_PAD + 0.24)
            if cols[ci] and ci < n - 1 and (y - y0 + hh > target or y + hh > limit):
                ci += 1
                x, w = COL[ci]
                y = y0
                if not head:
                    head = (carried or "") + " (cont.)"
                hh = (self.mtsu_block_h(w, sid) + self._head_h(d, w, head)
                      + 2 * MTSU_PAD + 0.24)
            # A block taller than a whole column used to be placed anyway,
            # because the guard required the column to be non-empty. That is how
            # the last column ran three inches off the bottom of the sheet while
            # the layout reported that it fitted.
            if y + hh > limit and ci == n - 1:
                spilled.append(sid)
                continue
            if head and not head.endswith("(cont.)"):
                carried = head
            cols[ci].append((head, sid))
            y += hh

        ends = []
        last_panels = []
        for k in range(n):
            x, w = COL[k]
            yy = y0
            panel = None
            for head, sid in cols[k]:
                if head:
                    if panel is not None and draw:
                        d.panel_close(panel, yy - 0.10)
                    if panel is not None:
                        yy += 0.24
                    yy = (d.head(x, yy, w, head) if draw
                          else yy + self._head_h(d, w, head))
                    panel = d.panel_open(x, yy, w) if draw else True
                    yy += MTSU_PAD
                elif panel is None:
                    panel = d.panel_open(x, yy, w) if draw else True
                    yy += MTSU_PAD
                yy = (self.mtsu_block(x, yy, w, sid, fig_no) if draw
                      else yy + self.mtsu_block_h(w, sid))
            if panel is not None and draw:
                d.panel_close(panel, yy + MTSU_PAD - 0.10)
                last_panels.append(panel)
            ends.append(yy + (MTSU_PAD if cols[k] else 0))
        # A column fits when nothing spilled AND nothing ran past the page. The
        # spill guard only fires in the last column, so a block taller than a
        # column placed in an earlier one overflowed while this reported that
        # the sheet fitted -- once by 3.4 in.
        # Run every column's last panel to the same baseline. Columns of
        # different length are honest -- content does not come in equal amounts
        # -- but three panels stopping at three different heights above a
        # full-width strip reads as unfinished rather than as considered.
        if draw and last_panels:
            foot = max(ends) - 0.10
            for sh in last_panels:
                d.panel_close(sh, foot)
        return (not spilled and max(ends) <= limit + 0.05), ends, fig_no, spilled

    def _mtsu_top_real(self):
        """Where the columns start, measured by drawing into a scratch deck.

        Cached per type tier, because it only depends on the tier and on the
        two opening plates.
        """
        key = round(self.st.d["body"], 2)
        cache = getattr(self, "_y0_cache", None)
        if cache is None:
            cache = self._y0_cache = {}
        if key not in cache:
            real = self.d
            try:
                self.d = Deck(TITLE, STAND, style=self.st)
                cache[key] = self.band(self.mtsu_kpi(self.d.top), [1])
            finally:
                self.d = real
        return cache[key]

    def _mtsu_top(self):
        """Where the columns start, without drawing anything."""
        d = self.d
        from PIL import Image
        y = d.top
        if self.st.hero == "census":
            cells = self.census_cells()
            cw = d.FULL[1] / float(len(cells))
            note_h = max(d.h_est(c[2], cw, 15, 1.10) for c in cells)
            y += 0.22 + 52 / 72.0 * 1.06 + 17 / 72.0 * 1.42 + note_h + 0.52
        for sid in BANDS:
            s = self.lib.get(sid)
            if not s:
                continue
            f = s.get("figure") or "none"
            fp = os.path.join(d.FIG, f + ".png")
            if abs(FIGW_IN.get(f, 0) - d.FULL[1]) > 0.02 or not os.path.exists(fp):
                continue
            if not s.get("title_below"):
                y += self._head_h(d, d.FULL[1], s["heading"])
            if s.get("lede"):
                y += d.h_est(s["lede"], d.FULL[1], d.dz["body"], 1.22, 10)
            im = Image.open(fp)
            y += d.FULL[1] * im.size[1] / float(im.size[0]) + 0.70
            if s.get("title_below"):
                y += (d.dz["caption"] + 13) / 72.0 * 1.10 + 0.44
            if s.get("body"):
                y += d.h_est(s["body"], d.FULL[1], d.dz["body"], 1.22, 10)
            y += d.dz["sec_gap"] * 0.6
        return y

    _TYPE_KEYS = ("body", "caption", "heading", "sub", "stand")

    def _set_scale(self, base, scale):
        """Set the whole type tier at once and rebuild the deck at it."""
        self.st.d = dict(base)
        for k in self._TYPE_KEYS:
            if k in base:
                self.st.d[k] = round(base[k] * scale, 1)
        self.st.d["sec_gap"] = base["sec_gap"] * scale
        self._y0_cache = {}
        self.d = Deck(TITLE, STAND, style=self.st)
        return self.d

    def _mtsu_fit(self, n, COL, scale, base):
        """Shed and backfill at one type scale. Returns (kept_flow, dropped)."""
        self._set_scale(base, scale)
        self._fig_off = set()
        d = self.d
        skip = set(TAIL_BANDS) | set(BANDS) | set(CLOSING) | set(self.tail_ids)
        flow = []
        for head, ids, _keep in MTSU_COLUMNS:
            first = True
            for sid in ids:
                if sid in self.lib and sid not in skip:
                    flow.append((head if first else None, sid))
                    first = False
        order = {s: i for _, ids, _k in MTSU_COLUMNS for i, s in enumerate(ids)}
        # What the sheet gives up first, across all four themes at once. Ranking
        # by position within a theme alone amputates the LONGEST theme: with
        # seven blocks in one theme and ten in another, every one of the ten
        # ranks above the seven and the long theme is stripped to its first
        # block before the short one loses anything. The label campaign, which
        # is the longest theme and the reason for the poster, came off the sheet
        # entirely that way. Shedding goes round the themes instead: every
        # theme's seventh block, then every theme's sixth, and so on.
        shed = {s: i * 100 + t
                for t, (_h, ids, _k) in enumerate(MTSU_COLUMNS)
                for i, s in enumerate(ids)}
        # The blocks a theme may not be stripped of, by the count it declares,
        # and -- when even those will not pack -- the order in which the counts
        # themselves give way. That order is PROPORTIONAL: a theme that claims
        # six blocks gives up its sixth before a theme that claims three gives
        # up its third, so a short theme is not quietly held whole while the
        # long one is cut to the bone.
        keep, hard_rank = set(), {}
        for t, (_h, ids, k) in enumerate(MTSU_COLUMNS):
            keep |= set(ids[:k])
            for i, sid in enumerate(ids[:k]):
                hard_rank[sid] = (i / float(max(1, k))) * 1000 + t

        def reheaded(seq):
            out_, seen = [], set()
            for _h, sid in seq:
                th = self._theme_of(sid)
                out_.append((self._head_of(th), sid) if th not in seen else (None, sid))
                seen.add(th)
            return out_

        def demand_of(seq):
            tot = 0.0
            for head, sid in seq:
                tot += self.mtsu_block_h(COL[0][1], sid)
                if head:
                    tot += self._head_h(d, COL[0][1], head) + 2 * MTSU_PAD + 0.24
            return tot

        self._demand = demand_of(flow)
        dem = self._demand

        dropped = []
        while True:
            if self._mtsu_layout(flow, n, COL, draw=False, demand=dem)[0]:
                break
            counts = {}
            for _h, sid in flow:
                counts.setdefault(self._theme_of(sid), []).append(sid)
            cands = [s for pool in counts.values() for s in pool[1:]
                     if s not in MTSU_PINNED and s not in keep]
            if not cands:
                # Nothing left to give up but blocks a theme may not lose. If
                # one of them is spilling because of the plate it carries, give
                # up the PLATE and keep the claim: a theme reduced to a heading
                # over nothing is worse than a finding stated without its chart.
                _f, _e, _n, sp = self._mtsu_layout(flow, n, COL, draw=False,
                                                   demand=dem)
                off = [s for s in sp if self.mtsu_has_plate(s, COL[0][1])
                       and s not in MTSU_KEEP_FIGURE]
                if off:
                    self._fig_off = set(getattr(self, "_fig_off", set())) | {off[0]}
                    continue
                # Nothing left but the counts themselves. Give one up rather
                # than spill: a block that runs off the bottom of the last
                # column is not on the poster either, and it leaves the sheet
                # thinking it is. Fourteen protected blocks would not pack into
                # four columns even when their inches fitted -- a column breaks
                # early when the next block is a plate taller than its
                # remainder -- and the sheet was printing that as success.
                hard = [sid for _h, sid in flow
                        if sid in keep and sid not in MTSU_PINNED]
                if hard:
                    victim = max(hard, key=lambda s: (
                        hard_rank.get(s, 0), self.mtsu_block_h(COL[0][1], s)))
                    dropped.append(victim)
                    flow = reheaded([(h, s) for h, s in flow if s != victim])
                    continue
                break
            victim = max(cands, key=lambda s: (shed.get(s, 9999),
                                               self.mtsu_block_h(COL[0][1], s)))
            dropped.append(victim)
            flow = reheaded([(h, s) for h, s in flow if s != victim])


        for _ in range(12):
            placed = False
            # A block a theme declared it may not lose comes back before any
            # block that is slack, whatever theme the slack belongs to.
            # Sorting by shed rank alone let three spare paragraphs from the
            # first theme take the gaps that the campaign's own blocks were
            # waiting for.
            for take in sorted(dropped, key=lambda s: (s not in keep,
                                                       shed.get(s, 9999))):
                th = self._theme_of(take)
                same = [k for k, (_h, sid) in enumerate(flow)
                        if self._theme_of(sid) == th]
                if not same:
                    continue
                after = [k for k in same
                         if order.get(flow[k][1], 99) > order.get(take, 99)]
                idx = after[0] if after else same[-1] + 1
                trial = reheaded(flow[:idx] + [(None, take)] + flow[idx:])
                if self._mtsu_layout(trial, n, COL, draw=False, demand=dem)[0]:
                    flow = trial
                    dropped.remove(take)
                    placed = True
                    break
            if not placed:
                break
        return flow, dropped, set(self._fig_off)

    def build_mtsu(self, out):
        """Four themes flowed across the columns, not bolted one per column.

        Binding a theme to a column makes the sheet as tall as its longest theme
        and as empty as its shortest. Flowing fills every column and lets a
        theme's heading appear where that theme begins, marked "(cont.)" when it
        carries over.
        """
        FULL, COL = self.d.FULL, self.d.COL
        n = len(COL)
        base = dict(self.st.d)

        # A section in the library but in no theme never enters the flow, so it
        # is neither drawn nor counted as dropped -- it just is not there, and
        # nothing said so. Record it: leaving a section off the sheet is a
        # decision, and a decision that leaves no trace is indistinguishable
        # from a bug.
        themed = set(s for _h, ids, _k in MTSU_COLUMNS for s in ids)
        reserved = (set(BANDS) | set(TAIL_BANDS) | set(CLOSING)
                    | set(FOOTER) | set(self.tail_ids))
        self.offsheet = sorted(s for s in self.lib
                               if s not in themed and s not in reserved)

        # Fill the sheet with type rather than with air. Nine inches of column
        # were sitting empty under the last block of three of the four columns,
        # which is a poster asking to be read from four feet and leaving a third
        # of its measure blank. Try the type tier large first and take the
        # largest that keeps as much of the argument as the smallest does.
        # Try every tier and take the one that carries the most argument; where
        # two carry the same, take the larger type. Nine inches of column were
        # sitting empty under three of the four columns -- a poster meant to be
        # read from four feet, leaving a third of its measure blank.
        keepset = set()
        for _h, ids, k in MTSU_COLUMNS:
            keepset |= set(ids[:k])
        trials = []
        for scale in (1.70, 1.62, 1.52, 1.46, 1.40, 1.34, 1.28, 1.22, 1.16,
                      1.10, 1.05, 1.00, 0.96, 0.92, 0.88, 0.84):
            flow, dropped, figoff = self._mtsu_fit(n, COL, scale, base)
            fits = self._mtsu_layout(flow, n, COL, draw=False,
                                     demand=self._demand)[0]
            demand = self._demand
            n_keep = len([sid for _h, sid in flow if sid in keepset])
            trials.append((len(flow), scale, flow, dropped, fits, figoff, demand,
                           n_keep))
        ok = [r for r in trials if r[4]] or trials
        # Bigger type beats more paragraphs. Hongbo Zhang, on the sheet:
        # "The font needs to be bigger. You can reduce the amount of text."
        # So: of the tiers that fit, take the LARGEST that still carries within
        # two blocks of the most any tier carries -- not the one that carries
        # the most, which is always the smallest type.
        # Carry the argument first, then set it as large as will fit. Choosing
        # by block COUNT picks whichever tier fits the most paragraphs, and
        # paragraphs are not equal: the tier that carried the most blocks
        # carried them by filling the gaps with spare notes from the first
        # theme while the label campaign, which is what the poster is for, sat
        # in the dropped list. So: maximise the blocks a theme declared it may
        # not lose; among tiers that carry the same number of those, take the
        # largest type; and only then prefer more blocks.
        most_keep = max(r[7] for r in ok)
        good = [r for r in ok if r[7] >= most_keep] or ok
        best = max(good, key=lambda r: (r[1], r[0]))
        # Restore the suppressed-figure set that BELONGS to the chosen tier.
        # It is per-trial state, and carrying the last trial's set into the
        # final draw silently dropped a plate the chosen tier had room for.
        self._fig_off = set(best[5])
        self._demand = best[6]
        best = best[:4]
        kept, scale, flow, dropped = best
        self.type_scale = scale
        self.dropped = list(dropped)
        self._set_scale(base, scale)
        d = self.d

        self.d = Deck(TITLE, STAND, style=self.st); d = self.d
        _fits, ends, fig_no, spilled = self._mtsu_layout(flow, n, COL, draw=True,
                                                         demand=self._demand)
        # Whatever still runs off the last column is dropped, and saying so is
        # the whole point of the manifest. The shed loop can exit with sections
        # still spilling -- when every theme is down to the one block it may not
        # lose -- and those were vanishing from the sheet with nothing recorded.
        for sid in spilled:
            if sid not in self.dropped:
                self.dropped.append(sid)
        self.widows = [msg for _h, sid in flow
                       for msg in [self.widow(COL[0][1], sid)] if msg]

        # The corner the columns leave empty is where the code goes -- it is the
        # one place on the sheet a reader can take the platform away with them.
        foot = max(ends)
        free = foot - ends[0]
        if free >= 1.95:
            self.qr_card(COL[0][0], ends[0] + 0.30, COL[0][1], free)
        elif free > 0:
            self.dropped.append("qr_card:only-%.2f-in-free" % free)

        y = self.closing(self.tail_band(max(ends) + 0.50, fig_no))
        foot_ids = [f for f in FOOTER if f in self.lib][:3]
        if not foot_ids:
            d.save(out)
            return self.dropped
        y = max(y + 0.40, H - self.foot_h() - 0.35)
        d.rect(FULL[0], y - 0.25, FULL[1], 0.022, fill=d.c["rule"])
        fw = (FULL[1] - 1.0) / 3.0
        for i, sid in enumerate(foot_ids):
            s = self.lib[sid]
            d.body(FULL[0] + i * (fw + 0.5), y, fw,
                   "%s. %s" % (s["heading"], s["body"]),
                   size=d.dz["caption"], color=d.c["mute"], justify=False)
        d.save(out)
        return self.dropped

    def _head_h(self, d, w, head):
        em = {"Georgia": 0.88, "Times New Roman": 0.80}.get(d.f["display"], 0.81)
        cpl = max(8, int((w * 72.0) / (d.dz["heading"] * em)))
        lines = max(1, -(-len(head.upper()) // cpl))
        return max(0.500, lines * d.dz["heading"] / 72.0 * 1.12 + 0.12) + 0.255

    def _theme_of(self, sid):
        for head, ids, _keep in MTSU_COLUMNS:
            if sid in ids:
                return head
        return ""

    def _head_of(self, theme):
        return theme

    def build(self, out):
        if getattr(self.st, "look", "") == "mtsu":
            return self.build_mtsu(out)
        d = self.d
        FULL, COL = d.FULL, d.COL
        n = len(COL)
        slots = [SLOT_OF_WIDTH.get(round(w, 3), "column") for _, w in COL]

        # Interleave by round, not by theme: the first section of every theme,
        # then the second of every theme, and so on. Overflow drops from the end,
        # so what a crowded sheet loses is each theme's third and fourth section
        # rather than the whole of the last theme. Theme-order dropping cost the
        # results block on every sheet.
        rounds = max(len(ids) for _, ids in SPINE)
        skip = {BAND} | set(CLOSING) | set(TAIL_BANDS)
        # The opening already shows four frames from both vehicles; the same four
        # printed again two columns later reads as a mistake, and on the first
        # journal proof it appeared as Fig. 1 and Fig. 4 on one sheet.
        if self.st.hero == "photos":
            skip |= {"vehicles", "vehicles_wide"}
        ordered, rank = [], {}
        for r in range(rounds):
            for _theme, ids in SPINE:
                if r < len(ids) and ids[r] in self.lib and ids[r] not in skip:
                    ordered.append(ids[r])
                    rank[ids[r]] = r
        def assign(live):
            """Deal the surviving sections into columns by estimated inches.

            Re-dealt after every drop. Dealing once and then popping from a
            fixed assignment left one column of the four-column proof empty
            below the halfway rule while another ran to the footer: the drops
            all came out of the same column, and nothing moved up to fill it.
            """
            cols = [[] for _ in range(n)]
            load = [0.0] * n
            figless = []
            for sid in live:
                want = self.slot_of(sid)
                # A section carrying a figure goes to a column of exactly that
                # width or its figure is not drawn at all. The near-miss rule
                # that used to let an 11.6 in figure into a 10.925 in column did
                # not scale the plate -- `section()` simply skipped it -- so the
                # four-column sheet printed twenty-two sections and one figure
                # and nothing said a plate had gone missing.
                exact = [i for i, sl in enumerate(slots) if sl == want]
                fit = exact if (want and exact) else list(range(n))
                if want and not exact:
                    figless.append(sid)
                i = min(fit, key=lambda k: load[k])
                cols[i].append(sid)
                load[i] += self.est_h(sid, COL[i][1], slots[i])
            return cols, figless

        live = list(ordered)
        fig_no = [1]
        while True:
            cols, self.figless = assign(live)
            fig_no[0] = 1
            y0 = self.band(self.hero(fig_no), fig_no)
            limit = H - self.foot_h() - 0.95 - self.closing_h() - self.tail_band_h()
            ends = []
            for ci, ids in enumerate(cols):
                x, w = COL[ci]
                y = y0
                for sid in ids:
                    y = self.section(x, y, w, sid, slots[ci], fig_no)
                ends.append(y)
            if max(ends) <= limit or not live:
                break
            # What to give up first. Each section knows which round of the spine
            # it came from: round 0 is a theme's opening claim, round 4 its
            # fourth elaboration. A plate counts as two rounds earlier than its
            # prose, because a sheet that keeps the sentence and drops the
            # figure has kept the claim and thrown away the proof.
            ci = max(range(n), key=lambda i: ends[i])
            col = cols[ci]
            worst = max(col, key=lambda s: (rank.get(s, 9)
                                            - (2 if self.draws_figure(s, slots[ci]) else 0),
                                            0 if self.draws_figure(s, slots[ci]) else 1,
                                            col.index(s)))
            self.dropped.append(worst)
            live.remove(worst)
            self.d = Deck(TITLE, STAND, style=self.st); d = self.d

        # Backfill. One drop can be worth eight inches -- a plate and its caption
        # -- so the loop above routinely overshoots and leaves three or four
        # inches of paper empty above the closing rule. Try the best of what was
        # dropped, cheapest first, and keep whatever still fits. Each attempt is
        # a real render, so this is bounded rather than exhaustive.
        for _ in range(6):
            spare = limit - max(ends)
            cands = sorted(self.dropped,
                           key=lambda s: (rank.get(s, 9),
                                          self.est_h(s, COL[0][1], slots[0])))
            cands = [s for s in cands
                     if self.est_h(s, COL[0][1], slots[0]) <= spare * 0.92]
            if not cands:
                break
            take = cands[0]
            trial = list(live) + [take]
            trial.sort(key=lambda s: ordered.index(s))
            self.d = Deck(TITLE, STAND, style=self.st); d = self.d
            cols2, figless2 = assign(trial)
            fig_no[0] = 1
            y0 = self.band(self.hero(fig_no), fig_no)
            ends2 = []
            for ci, ids in enumerate(cols2):
                x, w = COL[ci]
                y = y0
                for sid in ids:
                    y = self.section(x, y, w, sid, slots[ci], fig_no)
                ends2.append(y)
            if max(ends2) <= limit:
                live, cols, ends, self.figless = trial, cols2, ends2, figless2
                self.dropped.remove(take)
            else:
                # put the sheet back the way it was before the attempt
                self.d = Deck(TITLE, STAND, style=self.st); d = self.d
                fig_no[0] = 1
                y0 = self.band(self.hero(fig_no), fig_no)
                ends = []
                for ci, ids in enumerate(cols):
                    x, w = COL[ci]
                    y = y0
                    for sid in ids:
                        y = self.section(x, y, w, sid, slots[ci], fig_no)
                    ends.append(y)
                break

        y = self.closing(self.tail_band(max(ends) + 0.55, fig_no))
        y = max(y + 0.45, H - self.foot_h() - 0.35)
        d.rect(FULL[0], y - 0.25, FULL[1], 0.022, fill=d.c["rule"])
        fw = (FULL[1] - 1.0) / 3.0
        for i, sid in enumerate([f for f in FOOTER if f in self.lib][:3]):
            s = self.lib[sid]
            d.body(FULL[0] + i * (fw + 0.5), y, fw, "%s. %s" % (s["heading"], s["body"]),
                   size=d.dz["caption"], color=d.c["mute"])
        d.save(out)
        return self.dropped


def contact_sheet(items, out, cols=4, thumb_w=880):
    from PIL import Image, ImageDraw
    ims = []
    for name, p in items:
        im = Image.open(p)
        r = thumb_w / im.width
        ims.append((name, im.resize((thumb_w, int(im.height * r)))))
    if not ims:
        return None
    tw, th = ims[0][1].size
    rows = -(-len(ims) // cols)
    sheet = Image.new("RGB", (cols * (tw + 24) + 24, rows * (th + 64) + 24), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(ims):
        x = 24 + (i % cols) * (tw + 24)
        y = 24 + (i // cols) * (th + 64)
        sheet.paste(im, (x, y + 40))
        dr.text((x, y + 10), name, fill=(30, 41, 59))
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--out", default=os.path.join(HERE, "variants"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    lib = json.load(open(a.library))
    lib = lib.get("sections", lib) if isinstance(lib, dict) else lib
    os.makedirs(a.out, exist_ok=True)
    manifest = []
    for k, st in enumerate(styles.sample(a.n, a.seed)):
        name = "v%02d" % (k + 1)
        out = os.path.join(a.out, name + ".pptx")
        p = Poster(st, lib)
        dropped = p.build(out)
        manifest.append({"name": name, "style": st.name, "describe": st.describe(),
                         "dropped": dropped, "figless": p.figless, "pptx": out})
        print("  %s  %-52s dropped=%s" % (name, st.name, dropped or "-"))
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    if a.render:
        items = []
        for m in manifest:
            subprocess.run(["soffice", "--headless", "--convert-to", "png", "--outdir", a.out, m["pptx"]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            png = m["pptx"][:-5] + ".png"
            if os.path.exists(png):
                items.append(("%s  %s" % (m["name"], m["style"]), png))
        cs = contact_sheet(items, os.path.join(a.out, "contact_sheet.png"))
        print("contact sheet:", cs, "(%d posters)" % len(items))


if __name__ == "__main__":
    main()
