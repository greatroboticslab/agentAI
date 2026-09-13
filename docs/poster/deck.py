#!/usr/bin/env python3
"""Shared poster furniture, so three layouts can differ in argument and in
nothing else.

Every variant imports this, gets the same palette, the same grid, the same type
scale and the same numbers out of `poster_data`, and then decides only what goes
where. That is deliberate: if the three drafts disagreed about a number, the
comparison between them would be a comparison of bugs.

Type scale. This poster reads at about a metre, so the body is 20 pt and the
captions 16 pt, with a 34 pt pull-in tier for headings and a 120 pt title. The
figures are authored at their placed width with their own labels set at 14 pt,
just under the caption, so nothing on the sheet is scaled after it is drawn.

Word budget. At 20 pt one inch of an 11.6 in column carries about 37 words and
one inch of the 22.4 in centre column about 72. A 48 x 36 sheet laid out this way
holds roughly 950 words of prose. A sentence that does not fit is cut, not shrunk.
"""
import json
import os
import sys

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import poster_data as D                                        # noqa: E402

FIG = os.path.join(HERE, "fig")
CAPS = json.load(open(os.path.join(HERE, "captions.json")))

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
GREY  = RGBColor(0x8D, 0x99, 0xA6)
FONT  = "Arial"

W, H = 48.0, 36.0
MARG = 0.8
COL = [(0.8, 11.6), (12.8, 22.4), (35.6, 11.6)]
FULL = (0.8, 46.4)

BODY, CAPTION, HEADING, SUBHEAD, TITLE, STAND = 20, 16, 34, 24, 120, 34
TABLE_TXT = 17


class Deck(object):
    """One slide, and the handful of marks anything on it is made of.

    A `Style` (styles.py) decides the grid, the palette, the display face, the
    title treatment and the density. With no style the deck is the navy
    three-column look the hand-built drafts used, so those still build.
    """

    def __init__(self, title, standfirst=None, band=4.3, style=None):
        if style is None:
            from styles import Style
            style = Style()
        self.style = style
        self.c, self.f, self.dz = style.c, style.f, style.d
        self.COL, self.FULL = style.COL, style.FULL
        self.look = getattr(style, "look", "modern")
        self.FIG = os.path.join(HERE, getattr(style, "fig_dir", "fig"))
        self.sec_no = 0
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.band = band
        # paper colour under everything, so a warm palette is not white
        self.rect(0, 0, W, H, fill=self.c["paper"])
        self._titleblock(title, standfirst)

    # ---- marks ----------------------------------------------------------
    def rect(self, x, y, w, h, fill=None, line=None, lw=0.75):
        sh = self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
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

    def tbox(self, x, y, w, runs, align=PP_ALIGN.LEFT, spacing=1.0, face=None):
        tb = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.4))
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
            run.font.color.rgb = color; run.font.name = face or self.f["body"]
        return tb

    # ---- measurement ----------------------------------------------------
    def h_est(self, s, w, size, spacing=1.22, after=0.0):
        """Characters per line, from the face actually being set.

        Arial runs about 0.50 em per character at these sizes; Times runs 0.44
        and Georgia 0.48. Estimating a Times column at Arial's width predicts
        12 per cent more lines than the render has, which on a 36 in sheet is
        two inches of phantom height and a section dropped that would have fit.
        """
        em = {"Times New Roman": 0.442, "Georgia": 0.478,
              "Arial Narrow": 0.425}.get(self.f["body"], 0.50)
        cpl = max(14, int((w * 72.0) / (size * em)))
        lines = sum(max(1, -(-len(p) // cpl)) for p in s.split("\n"))
        return lines * size * spacing / 72.0 + after / 72.0

    @staticmethod
    def words(*strings):
        return sum(len(s.split()) for s in strings)

    # ---- blocks ---------------------------------------------------------
    def _titleblock(self, title, standfirst):
        M = D.MEETING
        c, f, dz = self.c, self.f, self.dz
        kind = self.style.title_kind
        # One line, always: a title that wraps eats the byline. The display face
        # is measured at ~0.55 em per character for Arial and ~0.50 for the serifs.
        # Measured against the render, not guessed: Georgia bold runs wide.
        em = {"Georgia": 0.62, "Times New Roman": 0.56, "Arial Narrow": 0.47}.get(f["display"], 0.55)
        size = TITLE
        while size > 56 and len(title) * em * size / 72.0 > (W - 2 * MARG - (2.2 if kind == "block" else 0)):
            size -= 2
        self.title_pt = size
        names = "   \u00b7   ".join(n for n, _ in M["authors"])
        affil = "%s   \u00b7   %s" % (M["affiliations"][0], M["institution"])
        if self.look == "journal":
            y = self._journal_title(title, names, affil, size, kind)
        elif kind == "band":
            self.rect(0, 0, W, self.band, fill=c["band"])
            self.rect(0, self.band, W, 0.05, fill=c["accent"])
            self.tbox(MARG, 0.42, W - 2 * MARG, [(title, size, f["display_bold"], c["band_ink"])],
                      spacing=0.95, face=f["display"])
            self.tbox(MARG, 2.32, W - 2 * MARG, [(names, 40, True, c["band_ink"])], spacing=1.0)
            self.tbox(MARG, 3.02, W - 2 * MARG, [(affil, 24, False, c["rule"])], spacing=1.0)
            y = self.band + 0.42
        elif kind == "rule":
            self.tbox(MARG, 0.55, W - 2 * MARG, [(title, size, f["display_bold"], c["ink"])],
                      spacing=0.95, face=f["display"])
            self.tbox(MARG, 2.45, W - 2 * MARG, [(names, 40, True, c["ink"])], spacing=1.0)
            self.tbox(MARG, 3.15, W - 2 * MARG, [(affil, 24, False, c["mute"])], spacing=1.0)
            self.rect(MARG, self.band - 0.20, W - 2 * MARG, 0.045, fill=c["ink"])
            y = self.band + 0.42
        elif kind == "block":
            # a vertical accent bar and the title set beside it
            self.rect(MARG, 0.45, 0.42, self.band - 0.9, fill=c["accent"])
            self.tbox(MARG + 1.1, 0.42, W - 2 * MARG - 1.1, [(title, size, f["display_bold"], c["ink"])],
                      spacing=0.95, face=f["display"])
            self.tbox(MARG + 1.1, 2.32, W - 2 * MARG - 1.1, [(names, 40, True, c["ink"])], spacing=1.0)
            self.tbox(MARG + 1.1, 3.02, W - 2 * MARG - 1.1, [(affil, 24, False, c["mute"])], spacing=1.0)
            y = self.band + 0.42
        else:  # underline
            self.tbox(MARG, 0.55, W - 2 * MARG, [(title, size, f["display_bold"], c["ink"])],
                      spacing=0.95, face=f["display"])
            self.rect(MARG, 2.30, min(W - 2 * MARG, len(title) * em * size / 72.0), 0.16,
                      fill=c["accent"])
            self.tbox(MARG, 2.70, W - 2 * MARG, [(names, 40, True, c["ink"])], spacing=1.0)
            self.tbox(MARG, 3.40, W - 2 * MARG, [(affil, 24, False, c["mute"])], spacing=1.0)
            y = self.band + 0.42
        if standfirst and self.look == "journal":
            # An abstract: one measure narrower than the sheet, justified, ruled
            # top and bottom, set a size down from the title's byline. This is
            # the single most recognisable thing about an article's first page.
            aw = (W - 2 * MARG) * 0.86
            ax = MARG + (W - 2 * MARG - aw) / 2.0
            self.rect(ax, y, aw, 0.022, fill=c["rule"])
            self.tbox(ax, y + 0.26, aw, [(standfirst, dz["stand"] - 4, False, c["ink"])],
                      spacing=1.20, face=f["body"], align=PP_ALIGN.JUSTIFY)
            hh = self.h_est(standfirst, aw, dz["stand"] - 4, 1.20)
            self.rect(ax, y + 0.26 + hh + 0.18, aw, 0.022, fill=c["rule"])
            self.top = y + 0.26 + hh + 0.18 + 0.62
        elif standfirst:
            self.tbox(MARG, y, W - 2 * MARG, [(standfirst, dz["stand"], False, c["ink"])],
                      spacing=1.16, face=f["display"] if f["display"] != "Arial Narrow" else "Arial")
            self.top = y + self.h_est(standfirst, W - 2 * MARG, dz["stand"], 1.16) + 0.55
        else:
            self.top = y + 0.3

    def _journal_title(self, title, names, affil, size, kind):
        """The first page of a paper: rules, centred type, no coloured ground.

        The three treatments differ in where the rules go, not in whether there
        is a band -- a filled colour band across the head of a sheet is the one
        mark that says poster-template loudest, and none of the three has one.
        """
        c, f = self.c, self.f
        tw = W - 2 * MARG
        size = min(size, 96)                 # a serif at 120 pt over 48 in shouts
        if kind == "masthead":
            eyebrow = "%s   \u00b7   %s" % (D.MEETING["institution"], D.MEETING["venue"]) \
                if D.MEETING.get("venue") else D.MEETING["institution"]
            self.rect(MARG, 0.55, tw, 0.020, fill=c["ink"])
            self.tbox(MARG, 0.70, tw, [(eyebrow.upper(), 22, False, c["mute"])],
                      spacing=1.0, align=PP_ALIGN.CENTER, face=f["body"])
            self.rect(MARG, 1.16, tw, 0.020, fill=c["ink"])
            self.tbox(MARG, 1.40, tw, [(title, size, True, c["ink"])], spacing=0.98,
                      align=PP_ALIGN.CENTER, face=f["display"])
            yy = 1.40 + size / 72.0 * 1.14
            self.tbox(MARG, yy, tw, [(names, 36, False, c["ink"])], spacing=1.0,
                      align=PP_ALIGN.CENTER, face=f["body"])
            self.tbox(MARG, yy + 0.58, tw, [(affil, 22, False, c["mute"])], spacing=1.0,
                      align=PP_ALIGN.CENTER, face=f["body"])
            return yy + 1.28
        if kind == "hairline":
            self.rect(MARG, 0.62, tw, 0.020, fill=c["rule"])
            self.tbox(MARG, 0.86, tw, [(title, size, True, c["ink"])], spacing=0.98,
                      face=f["display"])
            yy = 0.86 + size / 72.0 * 1.14
            self.tbox(MARG, yy, tw * 0.62, [(names, 34, False, c["ink"])], spacing=1.0,
                      face=f["body"])
            self.tbox(MARG + tw * 0.62, yy, tw * 0.38, [(affil, 22, False, c["mute"])],
                      spacing=1.0, align=PP_ALIGN.RIGHT, face=f["body"])
            self.rect(MARG, yy + 0.72, tw, 0.045, fill=c["ink"])
            return yy + 1.16
        # classic: centred between a thick rule and a thin one
        self.rect(MARG, 0.58, tw, 0.048, fill=c["ink"])
        self.tbox(MARG, 0.86, tw, [(title, size, True, c["ink"])], spacing=0.98,
                  align=PP_ALIGN.CENTER, face=f["display"])
        yy = 0.86 + size / 72.0 * 1.14
        self.tbox(MARG, yy, tw, [(names, 36, False, c["ink"])], spacing=1.0,
                  align=PP_ALIGN.CENTER, face=f["body"])
        self.tbox(MARG, yy + 0.58, tw, [(affil, 22, False, c["mute"])], spacing=1.0,
                  align=PP_ALIGN.CENTER, face=f["body"])
        self.rect(MARG, yy + 1.16, tw, 0.020, fill=c["ink"])
        return yy + 1.44

    def head(self, x, y, w, s):
        c, f, dz = self.c, self.f, self.dz
        if self.look == "journal":
            # Numbered, ruled above, set in the body serif at a size the eye
            # reads as a heading and not as a banner. An article numbers its
            # sections because the argument has an order; so does this sheet.
            self.sec_no += 1
            self.rect(x, y, w, 0.020, fill=c["rule"])
            self.tbox(x, y + 0.20, w, [("%d.  %s" % (self.sec_no, s),
                                        dz["heading"], True, c["ink"])],
                      spacing=1.0, face=f["display"])
            return y + 0.20 + self.h_est("%d.  %s" % (self.sec_no, s), w,
                                         dz["heading"], 1.0) + 0.22
        self.tbox(x, y, w, [(s, dz["heading"], f["display_bold"], c["band"] if self.style.palette_name != "sand" else c["ink"])],
                  spacing=0.95, face=f["display"])
        self.rect(x, y + dz["heading"] / 72.0 * 1.30, w, 0.030, fill=c["accent"])
        return y + dz["heading"] / 72.0 * 1.30 + 0.36

    def sub(self, x, y, w, s):
        self.tbox(x, y, w, [(s, SUBHEAD, True, self.c["ink"])], spacing=1.0, face=self.f["display"])
        return y + 0.46

    def body(self, x, y, w, s, size=None, color=None, after=12, bold=False,
             spacing=1.22, justify=None):
        size = size or self.dz["body"]
        color = color or self.c["ink"]
        # Justified only for running prose, and only in the journal look: a
        # justified two-word label is a line of holes.
        if justify is None:
            justify = self.look == "journal" and len(s) > 160
        self.tbox(x, y, w, [(s, size, bold, color, after)], spacing=spacing,
                  align=PP_ALIGN.JUSTIFY if justify else PP_ALIGN.LEFT)
        return y + self.h_est(s, w, size, spacing, after)

    def figure(self, x, y, w, name, label, caption=None):
        """Image, then its caption. Returns the new y."""
        p = os.path.join(self.FIG, name + ".png")
        if not os.path.exists(p):
            self.rect(x, y, w, 2.2, fill=PALE, line=RULE)
            self.tbox(x + 0.2, y + 1.0, w - 0.4, [("missing " + name, 18, True, WARN)])
            return y + 2.4
        authored = Image.open(p).size[0] / 300.0
        assert abs(authored / w - 1.0) < 0.02, (
            "%s is %.3f in wide but placed at %.3f in (x%.2f)"
            % (name, authored, w, w / authored))
        ph = self.slide.shapes.add_picture(p, Inches(x), Inches(y), width=Inches(w))
        y2 = y + ph.height / 914400.0 + 0.12
        text = caption if caption is not None else CAPS.get(name, "")
        cs = self.dz["caption"]
        if self.look == "journal":
            # "Fig. 4." bold, the caption in the body serif at caption size, both
            # in one paragraph so the lead-in sits on the same line as the text.
            lead = label.replace("Figure", "Fig.").replace("..", ".")
            tb = self.slide.shapes.add_textbox(Inches(x), Inches(y2), Inches(w), Inches(0.4))
            tf = tb.text_frame; tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            par = tf.paragraphs[0]; par.line_spacing = 1.16
            for txt, bold, col in ((lead + " ", True, self.c["ink"]),
                                   (text, False, self.c["ink"])):
                r = par.add_run(); r.text = txt
                r.font.size = Pt(cs); r.font.bold = bold
                r.font.color.rgb = col; r.font.name = self.f["body"]
            cap = lead + " " + text
            return y2 + self.h_est(cap, w, cs, 1.16) + 0.34
        cap = "%s  %s" % (label, text)
        self.tbox(x, y2, w, [(cap, cs, False, self.c["mute"], 0)], spacing=1.18)
        return y2 + self.h_est(cap, w, cs, 1.18) + 0.34

    def table(self, x, y, w, header, rows, widths, hi=None, size=TABLE_TXT):
        n = len(header)
        cw = [w * f for f in widths]
        xs = [x + sum(cw[:i]) for i in range(n)]
        ink, acc, rule = self.c["ink"], self.c["accent"], self.c["rule"]
        self.rect(x, y, w, 0.026, fill=ink)
        yy = y + 0.13
        for i, hcell in enumerate(header):
            self.tbox(xs[i], yy, cw[i], [(hcell, size - 1.5, True, ink, 0)], spacing=1.0,
                      align=PP_ALIGN.RIGHT if i else PP_ALIGN.LEFT)
        yy += 0.40
        self.rect(x, yy, w, 0.016, fill=rule)
        yy += 0.12
        for ri, row in enumerate(rows):
            col = acc if (hi is not None and ri == hi) else ink
            bold = hi is not None and ri == hi
            for i, cell in enumerate(row):
                self.tbox(xs[i], yy, cw[i], [(str(cell), size, bold, col, 0)], spacing=1.0,
                          align=PP_ALIGN.RIGHT if i else PP_ALIGN.LEFT)
            yy += 0.44
        self.rect(x, yy + 0.03, w, 0.026, fill=ink)
        return yy + 0.32

    def bignum(self, x, y, w, value, label, note="", size=60):
        """One measured quantity, set to be read from across a room.

        In the journal look it is set in the text ink, not the spot colour, and
        a size down. One colour ran in a second pass on those presses, so it
        appears once on a page and never on a numeral -- a 62 pt brick-red
        figure floating in a column is the single most template-looking mark a
        sheet like this can carry.
        """
        if self.look == "journal":
            size = min(size, 38)
        col = self.c["ink"] if self.look == "journal" else self.c["accent"]
        self.tbox(x, y, w, [(value, size, True, col)], spacing=0.95, face=self.f["display"])
        yy = y + size / 72.0 * 1.02
        yy = self.body(x, yy, w, label, after=2)
        if note:
            yy = self.body(x, yy, w, note, size=self.dz["caption"], color=self.c["mute"], after=0)
        return yy

    def steps(self, x, y, w, items, gap=0.35):
        """A numbered horizontal strip. Four to six short stages, no arrows."""
        n = len(items)
        cw = (w - gap * (n - 1)) / n
        bottom = y
        for i, (t, d) in enumerate(items):
            cx = x + i * (cw + gap)
            self.tbox(cx, y, cw, [("%d" % (i + 1), 30, True, BLUE)], spacing=0.95)
            yy = y + 0.50
            self.tbox(cx, yy, cw, [(t, BODY, True, INK)], spacing=1.12)
            yy += self.h_est(t, cw, BODY, 1.12)
            yy = self.body(cx, yy + 0.04, cw, d, size=CAPTION, color=MUTE, after=0)
            bottom = max(bottom, yy)
            self.rect(cx, y - 0.14, cw, 0.018, fill=RULE)
        return bottom + 0.22

    def note(self, x, y, w, title, s):
        """A quiet ruled block for a caveat that has to ride on the sheet."""
        yy = self.body(x + 0.26, y, w - 0.26, title, bold=True, after=3)
        yy = self.body(x + 0.26, yy, w - 0.26, s, size=self.dz["caption"], color=self.c["mute"], after=0)
        self.rect(x, y, 0.030, yy - y, fill=self.c["rule"])
        return yy + 0.22

    def save(self, path):
        self.prs.core_properties.author = "Harry He"
        self.prs.core_properties.last_modified_by = "Harry He"
        self.prs.core_properties.title = D.MEETING["institution"]
        self.prs.save(path)
        print("wrote %s\n  %g x %g in, %d shapes"
              % (path, W, H, len(self.slide.shapes)))
        return path
