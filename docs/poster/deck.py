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
    """One slide, and the handful of marks anything on it is made of."""

    def __init__(self, title, standfirst=None, band=4.3):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.band = band
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

    def tbox(self, x, y, w, runs, align=PP_ALIGN.LEFT, spacing=1.0):
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
            run.font.color.rgb = color; run.font.name = FONT
        return tb

    # ---- measurement ----------------------------------------------------
    @staticmethod
    def h_est(s, w, size, spacing=1.22, after=0.0):
        """Conservative: Arial averages about 0.50 em per character here."""
        cpl = max(14, int((w * 72.0) / (size * 0.50)))
        lines = sum(max(1, -(-len(p) // cpl)) for p in s.split("\n"))
        return lines * size * spacing / 72.0 + after / 72.0

    @staticmethod
    def words(*strings):
        return sum(len(s.split()) for s in strings)

    # ---- blocks ---------------------------------------------------------
    def _titleblock(self, title, standfirst):
        self.rect(0, 0, W, self.band, fill=NAVY)
        self.rect(0, self.band, W, 0.05, fill=BLUE)
        M = D.MEETING
        # One line, always. A title that wraps eats the byline, and a poster
        # whose first act is to collide with itself has lost the reader before
        # the first number. Arial bold runs about 0.55 em per character.
        size = TITLE
        while size > 60 and len(title) * 0.55 * size / 72.0 > (W - 2 * MARG):
            size -= 2
        self.title_pt = size
        self.tbox(MARG, 0.42, W - 2 * MARG, [(title, size, True, WHITE)], spacing=0.95)
        names = "   ·   ".join(n for n, _ in M["authors"])
        self.tbox(MARG, 2.32, W - 2 * MARG, [(names, 40, True, WHITE)], spacing=1.0)
        self.tbox(MARG, 3.02, W - 2 * MARG,
                  [("%s   ·   %s" % (M["affiliations"][0], M["institution"]),
                    24, False, RULE)], spacing=1.0)
        if standfirst:
            self.tbox(MARG, self.band + 0.42, W - 2 * MARG,
                      [(standfirst, STAND, False, INK)], spacing=1.16)

    def head(self, x, y, w, s):
        self.tbox(x, y, w, [(s, HEADING, True, NAVY)], spacing=0.95)
        self.rect(x, y + 0.62, w, 0.030, fill=BLUE)
        return y + 0.98

    def sub(self, x, y, w, s):
        self.tbox(x, y, w, [(s, SUBHEAD, True, INK)], spacing=1.0)
        return y + 0.46

    def body(self, x, y, w, s, size=BODY, color=INK, after=12, bold=False,
             spacing=1.22):
        self.tbox(x, y, w, [(s, size, bold, color, after)], spacing=spacing)
        return y + self.h_est(s, w, size, spacing, after)

    def figure(self, x, y, w, name, label, caption=None):
        """Image, then its caption. Returns the new y."""
        p = os.path.join(FIG, name + ".png")
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
        cap = "%s  %s" % (label, caption if caption is not None else CAPS.get(name, ""))
        self.tbox(x, y2, w, [(cap, CAPTION, False, MUTE, 0)], spacing=1.18)
        return y2 + self.h_est(cap, w, CAPTION, 1.18) + 0.34

    def table(self, x, y, w, header, rows, widths, hi=None, size=TABLE_TXT):
        n = len(header)
        cw = [w * f for f in widths]
        xs = [x + sum(cw[:i]) for i in range(n)]
        self.rect(x, y, w, 0.026, fill=INK)
        yy = y + 0.13
        for i, hcell in enumerate(header):
            self.tbox(xs[i], yy, cw[i], [(hcell, size - 1.5, True, INK, 0)], spacing=1.0,
                      align=PP_ALIGN.RIGHT if i else PP_ALIGN.LEFT)
        yy += 0.40
        self.rect(x, yy, w, 0.016, fill=RULE)
        yy += 0.12
        for ri, row in enumerate(rows):
            col = BLUE if (hi is not None and ri == hi) else INK
            bold = hi is not None and ri == hi
            for i, cell in enumerate(row):
                self.tbox(xs[i], yy, cw[i], [(str(cell), size, bold, col, 0)], spacing=1.0,
                          align=PP_ALIGN.RIGHT if i else PP_ALIGN.LEFT)
            yy += 0.44
        self.rect(x, yy + 0.03, w, 0.026, fill=INK)
        return yy + 0.32

    def bignum(self, x, y, w, value, label, note="", size=60):
        """One measured quantity, set to be read from across a room."""
        self.tbox(x, y, w, [(value, size, True, BLUE)], spacing=0.95)
        yy = y + size / 72.0 * 1.02
        yy = self.body(x, yy, w, label, size=BODY, color=INK, after=2)
        if note:
            yy = self.body(x, yy, w, note, size=CAPTION, color=MUTE, after=0)
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
        self.rect(x, y, 0.030, 0.0, fill=RULE)
        yy = self.body(x + 0.26, y, w - 0.26, title, size=BODY, bold=True, after=3)
        yy = self.body(x + 0.26, yy, w - 0.26, s, size=CAPTION, color=MUTE, after=0)
        self.rect(x, y, 0.030, yy - y, fill=RULE)
        return yy + 0.22

    def save(self, path):
        self.prs.core_properties.author = "Harry He"
        self.prs.core_properties.last_modified_by = "Harry He"
        self.prs.core_properties.title = D.MEETING["institution"]
        self.prs.save(path)
        print("wrote %s\n  %g x %g in, %d shapes"
              % (path, W, H, len(self.slide.shapes)))
        return path
