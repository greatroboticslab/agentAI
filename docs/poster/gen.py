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
from deck import Deck, D, W, H, MARG                       # noqa: E402
from pptx.enum.text import PP_ALIGN                        # noqa: E402
import styles                                              # noqa: E402

C, L = D.CENSUS, D.LEDGER
FOOT_H = 2.20

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
STAND = ("Built at the MTSU Great Robotics Lab. Every robot the lab owns reports into one platform; "
         "agents on it harvest, filter, label and train; a third agent watches the other two. This is "
         "what runs today, what the robots have recorded, and what six months of running it taught us.")

# The one argument, in Harry's order. Section ids come from the library; a
# missing id is skipped, so this order can name sections before they exist.
SPINE = [
    ("platform-idea",           ["platform_idea", "system", "platform_domains"]),
    ("robots-connected",        ["robots_today", "shared_models", "future_robots"]),
    ("live-collection",         ["uplink", "r241_frames", "cart_frames", "census",
                                 "drive", "auto_diag"]),
    ("remote-control",          ["remote_control", "drive_button", "advice", "laser_control"]),
    ("analysis-agent",          ["analysis_agent", "analysis_example", "analysis_sandbox",
                                 "analysis_gap"]),
    ("iterating-agents-brain",  ["loop", "brain", "dispatch", "watch", "supervisor"]),
    ("results",                 ["detect_grid", "ladder", "field", "families", "species",
                                 "zeroshot", "tta", "sources", "ledger", "journey"]),
]

# Drawn full width, straight under the opening, because four photographs in a
# row is the one thing a visitor reads before any prose.
BAND = "projects"

# A second full-width strip, drawn just above the closing when its plate has no
# column of its width in this grid -- which is every grid, because it is 46.4 in
# wide. Six months in order is an article's timeline, and it was the one plate
# the layout could never place.
TAIL_BANDS = ["detect_grid", "journey"]

# Drawn last, across the sheet, above the footer. The vision belongs at the end
# of the argument and not at the top of it: a reader should meet what the
# platform is and what it has done before being told where it goes.
CLOSING = ["vision_platform", "vision_permanent"]

FOOTER = ["stands", "made", "withdrew"]

# The house look does not run twenty numbered sections down a sheet. The lab's
# own poster puts ONE all-caps heading over ONE white panel per column and
# separates the blocks inside it with small accent-square sub-heads -- which is
# why it reads as an article and not as a list. These are the columns.
# Headings that assert rather than label. A reader walking the row of four
# should get the abstract: what it is, what it holds, what runs on it, what it
# measured. "The platform / Collecting in the field / What the agents do with it
# / What six months measured" labelled four containers and told them nothing,
# and two of the four opened with the same word.
MTSU_COLUMNS = [
    # Order inside a theme IS its priority: the sheet gives up the last block of
    # a theme first. Sorting by theme size instead dropped remote control, the
    # fifteen-round loop and the agent-dispatch block -- three of the things
    # this poster exists to show -- because the agent theme happened to be the
    # longest list.
    ("Every robot joins one platform",
     ["platform_idea", "platform_domains", "robots_today", "shared_models",
      "future_robots"]),
    ("Two vehicles, 2,686 frames",
     ["uplink", "r241_frames", "cart_frames", "remote_control", "drive",
      "drive_button", "census", "auto_diag", "advice", "laser_control"]),
    ("Agents plan, code and review",
     ["analysis_agent", "loop", "dispatch", "brain", "supervisor",
      "analysis_example", "watch", "analysis_sandbox"]),
    ("More data made accuracy worse",
     ["ladder", "field", "species", "families", "ledger", "sources", "tta",
      "zeroshot"]),
]
# Blocks that are never shed, whatever the fit costs elsewhere. These are the
# ones the poster was asked for by name: a single camera frame says nothing
# about a robot, so each vehicle shows a grid of its own frames. Without a pin
# the shed loop reaches them every time, because they are expensive and sit
# second and third in their theme.
MTSU_PINNED = {"ladder"}

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
        self.tail_ids = (["r241_frames", "detect_grid"] if getattr(st, "look", "") == "mtsu"
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
        return [("{:,}".format(C["frames"]), "camera frames", "%d drives, 2 robots, %d labelled"
                 % (C["sessions"], C["labelled"])),
                ("{:,}".format(C["sensor_rows"]), "telemetry rows", "nine streams, counted line by line"),
                ("%.0f m" % C["gps_m"], "of GPS track", "%s fixes over %d drives"
                 % ("{:,}".format(C["gps_fixes"]), C["gps_sessions"])),
                ("%d" % L["n_models"], "distinct models run", "four families, one deployed"),
                ("%.4f" % D.DEPLOYED["map50_95"], "mAP50-95, deployed model",
                 "all %s sealed holdout images, %s instances"
                 % ("{:,}".format(D.DEPLOYED["n_images"]),
                    "{:,}".format(D.HOLDOUT["instances"])))]

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
        """The four-project strip, full width."""
        d = self.d
        sid = BAND
        if sid not in self.lib or "q_projects" not in FIGW:
            return y
        s = self.lib[sid]
        x, w = d.FULL
        y = d.head(x, y, w, s["heading"])
        y = d.figure(x, y, w, "q_projects", "Figure %d." % fig_no[0]); fig_no[0] += 1
        # No body text under this one: the caption and the four columns of the
        # plate itself already carry every sentence the section had, and printing
        # both put the same claim on the sheet twice, four inches apart.
        return y + d.dz["sec_gap"] * 0.6

    def _full_bands(self):
        """The sections whose plate is 46.4 in wide: they can only run full width."""
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
            h += (d.dz["heading"] / 72.0 * 1.34 + 0.30
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
        return h + 1.05

    def closing(self, y):
        d = self.d
        ids = [i for i in CLOSING if i in self.lib]
        if not ids:
            return y
        x0, W0 = d.FULL
        d.rect(x0, y, W0, 0.045, fill=d.c["ink"])
        y += 0.34
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
        f = self.lib[sid].get("figure") or "none"
        return f != "none" and abs(FIGW_IN.get(f, 0) - w) < 0.02

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
        if f != "none" and not f.startswith("NEW") and abs(FIGW_IN.get(f, 0) - w) < 0.02:
            y = d.figure(x, y + 0.04, w, f, "Figure %d." % fig_no[0]); fig_no[0] += 1
        body = s["body"]
        if s.get("status") == "designed_only":
            body += " This part is a design and has not been built."
        elif s.get("status") == "built_but_off":
            body += " This part is built and is switched off today."
        y = d.body(tx, y, tw, body, after=10)
        return y + 0.26

    def mtsu_block_h(self, w, sid):
        s, d = self.lib[sid], self.d
        pad = MTSU_PAD
        tw = w - 2 * pad
        h = d.h_est(s["heading"], tw, d.dz["body"], 1.0) + 0.10 + 0.26
        f = s.get("figure") or "none"
        if f != "none" and abs(FIGW_IN.get(f, 0) - w) < 0.02:
            from PIL import Image
            fp = os.path.join(d.FIG, f + ".png")
            if os.path.exists(fp):
                im = Image.open(fp)
                h += w * im.size[1] / float(im.size[0]) + 0.76
        return h + d.h_est(s["body"], tw, d.dz["body"], 1.22, 10)

    def _mtsu_layout(self, flow, n, COL, draw):
        """Lay the flow out once. Returns (fits, ends, fig_no, spilled).

        With draw=False nothing is added to the slide, so a trial fit costs
        arithmetic instead of a rebuilt deck.
        """
        d = self.d
        fig_no = [1]
        y0 = self.band(self.mtsu_kpi(d.top), fig_no) if draw else self._mtsu_top()
        limit = H - FOOT_H - 0.95 - self.closing_h() - self.tail_band_h()
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
        share = (sum(heights) + 0.5 * n) / float(n)
        target = avail if share > avail else max(share, 0.82 * avail)

        cols = [[] for _ in range(n)]
        ci, y, carried, spilled = 0, y0, None, []
        for head, sid in flow:
            x, w = COL[ci]
            hh = self.mtsu_block_h(w, sid)
            if head:
                hh += self._head_h(d, w, head) + 2 * MTSU_PAD + 0.24
            if cols[ci] and ci < n - 1 and (y - y0 + hh > target or y + hh > limit):
                ci += 1
                x, w = COL[ci]
                y = y0
                if not head:
                    head = (carried or "") + " (cont.)"
                hh = (self.mtsu_block_h(w, sid) + self._head_h(d, w, head)
                      + 2 * MTSU_PAD + 0.24)
            if y + hh > limit and ci == n - 1 and cols[ci]:
                spilled.append(sid)
                continue
            if head and not head.endswith("(cont.)"):
                carried = head
            cols[ci].append((head, sid))
            y += hh

        ends = []
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
            ends.append(yy + (MTSU_PAD if cols[k] else 0))
        return (not spilled), ends, fig_no, spilled

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
        s = self.lib.get(BAND)
        if s:
            fp = os.path.join(d.FIG, "q_projects.png")
            y += self._head_h(d, d.FULL[1], s["heading"])
            if os.path.exists(fp):
                im = Image.open(fp)
                y += d.FULL[1] * im.size[1] / float(im.size[0]) + 0.70
            y += d.dz["sec_gap"] * 0.6
        return y

    def build_mtsu(self, out):
        """Four themes flowed across the columns, not bolted one per column.

        Binding a theme to a column makes the sheet as tall as its longest theme
        and as empty as its shortest. Flowing fills every column and lets a
        theme's heading appear where that theme begins, marked "(cont.)" when it
        carries over.
        """
        d = self.d
        FULL, COL = d.FULL, d.COL
        n = len(COL)
        skip = set(TAIL_BANDS) | {BAND} | set(CLOSING)
        flow = []
        for head, ids in MTSU_COLUMNS:
            first = True
            for sid in ids:
                if sid in self.lib and sid not in skip:
                    flow.append((head if first else None, sid))
                    first = False

        order = {s: i for _, ids in MTSU_COLUMNS for i, s in enumerate(ids)}
        widths = {w for _, w in COL}

        def plate(sid):
            return any(self.mtsu_has_plate(sid, w) for w in widths)

        def reheaded(seq):
            out_, seen = [], set()
            for _h, sid in seq:
                th = self._theme_of(sid)
                out_.append((self._head_of(th), sid) if th not in seen else (None, sid))
                seen.add(th)
            return out_

        # Shed until it fits. Priority inside a theme decides what goes; a plate
        # only breaks a tie, because a seven-inch plate is not worth three
        # blocks carrying the claims this poster exists to make. A theme never
        # loses its first block, which is what its heading asserts.
        while True:
            fits, _ends, _fn, _sp = self._mtsu_layout(flow, n, COL, draw=False)
            if fits:
                break
            counts = {}
            for _h, sid in flow:
                counts.setdefault(self._theme_of(sid), []).append(sid)
            cands = [s for pool in counts.values() for s in pool[1:]
                     if s not in MTSU_PINNED]
            if not cands:
                break
            victim = max(cands, key=lambda s: (order.get(s, 99), 0 if plate(s) else 1))
            self.dropped.append(victim)
            flow = reheaded([(h, s) for h, s in flow if s != victim])

        # Backfill. Shedding stops the moment the sheet fits, which leaves the
        # slack the last drop opened -- 2.8 in at the foot of every column on
        # one proof, about four blocks. Put back the best of what went, in
        # priority order, while it still fits.
        for _ in range(10):
            placed = False
            for take in sorted(self.dropped, key=lambda s: order.get(s, 99)):
                # Put it back INSIDE its own theme. Falling through to the end
                # of the flow put a platform block in the last column under the
                # results heading, because every later block of its theme had
                # already been dropped.
                th = self._theme_of(take)
                same = [k for k, (_h, sid) in enumerate(flow)
                        if self._theme_of(sid) == th]
                if not same:
                    continue
                after = [k for k in same
                         if order.get(flow[k][1], 99) > order.get(take, 99)]
                idx = after[0] if after else same[-1] + 1
                trial = reheaded(flow[:idx] + [(None, take)] + flow[idx:])
                if self._mtsu_layout(trial, n, COL, draw=False)[0]:
                    flow = trial
                    self.dropped.remove(take)
                    placed = True
                    break
            if not placed:
                break

        self.d = Deck(TITLE, STAND, style=self.st); d = self.d
        _fits, ends, fig_no, spilled = self._mtsu_layout(flow, n, COL, draw=True)
        # Whatever still runs off the last column is dropped, and saying so is
        # the whole point of the manifest. The shed loop can exit with sections
        # still spilling -- when every theme is down to the one block it may not
        # lose -- and those were vanishing from the sheet with nothing recorded.
        for sid in spilled:
            if sid not in self.dropped:
                self.dropped.append(sid)

        y = self.closing(self.tail_band(max(ends) + 0.50, fig_no))
        y = max(y + 0.40, H - FOOT_H - 0.35)
        d.rect(FULL[0], y - 0.25, FULL[1], 0.022, fill=d.c["rule"])
        fw = (FULL[1] - 1.0) / 3.0
        for i, sid in enumerate([f for f in FOOTER if f in self.lib][:3]):
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
        for head, ids in MTSU_COLUMNS:
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
            limit = H - FOOT_H - 0.95 - self.closing_h() - self.tail_band_h()
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
        y = max(y + 0.45, H - FOOT_H - 0.35)
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
