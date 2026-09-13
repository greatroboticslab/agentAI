#!/usr/bin/env python3
"""Visual styles for one poster.

The content of the poster is fixed: agentAI, the MTSU Great Robotics Lab's
research-data platform for physical agents, the robots on it, what the agents
do, what six months of running it produced. What varies here is only how that
one argument LOOKS: the grid, the palette, the type, the title treatment, the
opening image, and how densely it is set.

Every grid is built from the four widths the figures were authored at, 11.6,
22.4 and 46.4 inches, so a figure never lands in a slot it was not drawn
for. A two-column look is 22.4 + 22.4 with a 1.6 in gutter, not two 22.9 columns.
"""
import itertools
import random

from pptx.dml.color import RGBColor


def _rgb(h):
    return RGBColor(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


# ---------------------------------------------------------------- palettes
# Each is six roles. `band` is the title band, `accent` the one colour that is
# allowed to shout, `ink` the text, `mute` the second text, `rule` the hairline.
PALETTES = {
    "navy":   dict(band="#14314E", band_ink="#FFFFFF", accent="#1C6FB5", ink="#1E293B",
                   mute="#556575", rule="#C2CEDA", pale="#F4F7FA", paper="#FFFFFF"),
    "ink":    dict(band="#FFFFFF", band_ink="#111111", accent="#B5502A", ink="#111111",
                   mute="#5B5B5B", rule="#D0D0D0", pale="#F6F6F4", paper="#FFFFFF"),
    "forest": dict(band="#1F3D2B", band_ink="#F4F1E8", accent="#1B7A55", ink="#1C2A22",
                   mute="#5C6B62", rule="#CBD3CC", pale="#F3F5F1", paper="#FFFFFF"),
    "slate":  dict(band="#2F3A45", band_ink="#FFFFFF", accent="#B5502A", ink="#24303B",
                   mute="#66727E", rule="#CAD2DA", pale="#F5F6F8", paper="#FFFFFF"),
    "mono":   dict(band="#1A1A1A", band_ink="#FFFFFF", accent="#1C6FB5", ink="#1A1A1A",
                   mute="#6A6A6A", rule="#D5D5D5", pale="#F5F5F5", paper="#FFFFFF"),
    "sand":   dict(band="#F1EBDD", band_ink="#2B2A26", accent="#8A4B1E", ink="#2B2A26",
                   mute="#6E6759", rule="#D8D0BF", pale="#F7F3EA", paper="#FFFDF8"),
}

# ---------------------------------------------------------------- grids
# name -> list of (x, w) columns; FULL is always (0.8, 46.4).
GRIDS = {
    "three":      [(0.8, 11.6), (12.8, 22.4), (35.6, 11.6)],
    "wide-left":  [(0.8, 22.4), (23.6, 11.6), (35.6, 11.6)],
    "wide-right": [(0.8, 11.6), (12.8, 11.6), (24.8, 22.4)],
    "two":        [(0.8, 22.4), (24.8, 22.4)],
    "four":       [(0.30, 11.6), (12.233, 11.6), (24.167, 11.6), (36.10, 11.6)],
}

# ---------------------------------------------------------------- type
TYPES = {
    "arial":   dict(display="Arial", body="Arial", display_bold=True),
    "serif":   dict(display="Georgia", body="Arial", display_bold=True),
    "times":   dict(display="Times New Roman", body="Arial", display_bold=True),
    "narrow":  dict(display="Arial Narrow", body="Arial", display_bold=True),
}

# ------------------------------------------------- the old-journal look
# A second visual world, not a sixth palette. An offset-printed article from
# before colour separations were cheap: paper, black ink, and one spot colour
# that the press ran as a second pass, so colour is spent once and never on
# decoration. The typographic consequences are what actually remove the
# generated-poster flavour -- serif body, justified measure, hairline column
# rules, sections numbered like an article, captions that open "Fig. 4." --
# and those live in deck.py behind `style.look`.
JOURNAL_PALETTES = {
    "plate":  dict(band="#FFFFFF", band_ink="#111111", accent="#8C3A1E", ink="#111111",
                   mute="#5C5A55", rule="#BFBDB8", pale="#F2F1EE", paper="#FFFFFF"),
    "offset": dict(band="#FFFFFF", band_ink="#1A1A1A", accent="#1F3A5F", ink="#1A1A1A",
                   mute="#5E5E5E", rule="#C4C4C4", pale="#F4F4F2", paper="#FFFFFF"),
    "laid":   dict(band="#FDFBF6", band_ink="#20201C", accent="#7A3B12", ink="#20201C",
                   mute="#5F5B51", rule="#C9C3B4", pale="#F4F0E6", paper="#FDFBF6"),
    "proof":  dict(band="#FFFFFF", band_ink="#111111", accent="#3F5E43", ink="#111111",
                   mute="#585856", rule="#C6C6C2", pale="#F3F3F0", paper="#FFFFFF"),
}

JOURNAL_TYPES = {
    "times":   dict(display="Times New Roman", body="Times New Roman", display_bold=True),
    "georgia": dict(display="Georgia", body="Georgia", display_bold=True),
    "mixed":   dict(display="Arial Narrow", body="Times New Roman", display_bold=True),
}

# classic: centred title between two rules, the way a paper opens.
# masthead: a ruled eyebrow above, title and byline centred under it.
# hairline: title flush left under a single hairline, byline on the same line.
JOURNAL_TITLES = ("classic", "masthead", "hairline")

# ------------------------------------------------------------ the house look
# Lifted, not invented: every value here was read out of the lab's own
# MTSU_LaserCar_Poster.pptx. #1C6FB5, #C2CEDA and #14314E were already in
# style.py because both files descend from the same house deck; what the
# template adds is the near-black navy of the header band, the very pale blue
# the page sits on, and white content panels with a 0.75 pt #C2CEDA border.
#
# The template's own geometry, in inches on a 48 in sheet: band 3.05 high with a
# 0.10 accent rule under it, logo in a 3.94 x 2.66 white box at (0.60, 0.20),
# page margin 0.42, gutter 0.34, section heading 0.62 high over a 0.06 rule,
# panels starting 0.26 below that with 0.26 of padding, sub-heads marked by a
# 0.14 square.
MTSU_PALETTES = {
    "house":  dict(band="#14314E", band_ink="#FFFFFF", accent="#1C6FB5", ink="#16202B",
                   mute="#4C5A69", rule="#C2CEDA", pale="#EDF3F9", paper="#EDF1F6",
                   panel="#FFFFFF"),
    "navy":   dict(band="#14314E", band_ink="#FFFFFF", accent="#1C6FB5", ink="#16202B",
                   mute="#4C5A69", rule="#C2CEDA", pale="#EDF3F9", paper="#FFFFFF",
                   panel="#FFFFFF"),
    "trueblue": dict(band="#1C6FB5", band_ink="#FFFFFF", accent="#14314E", ink="#16202B",
                   mute="#4C5A69", rule="#C2CEDA", pale="#EDF3F9", paper="#F7F9FC",
                   panel="#FFFFFF"),
    "paper":  dict(band="#14314E", band_ink="#FFFFFF", accent="#1C6FB5", ink="#16202B",
                   mute="#4C5A69", rule="#C2CEDA", pale="#EDF3F9", paper="#FFFFFF",
                   panel="#FFFFFF"),
}

MTSU_TYPES = {
    # The template sets headings in Arial and body in Georgia: 123 Arial runs
    # against 120 Georgia runs in slide1.xml.
    "house":   dict(display="Arial", body="Georgia", display_bold=True),
    "serif":   dict(display="Georgia", body="Georgia", display_bold=True),
    "sans":    dict(display="Arial", body="Arial", display_bold=True),
}

# Where the logo sits and what runs beside it.
# The full-width dark band with reversed white type is the one element the
# poster-design literature names as the signature of a template: Faulkes lists
# "large coloured title bars" as the single visual marker of Canva/PowerPoint
# output, and CMU's own Field Robotics Center poster uses a PALE band with the
# title in dark type instead. Both are offered here so the choice is made on a
# comparison rather than by default.
MTSU_TITLES = ("logo-left", "pale-band", "logo-corner")

# title treatment: how the top of the sheet is set
TITLE_KINDS = ("band", "rule", "block", "underline")

# opening image under the title
HEROES = ("photos", "system", "census", "none")

DENSITY = {
    "dense": dict(body=19, caption=15, heading=32, stand=30, sec_gap=0.34),
    "normal": dict(body=20, caption=16, heading=34, stand=34, sec_gap=0.42),
    "airy":  dict(body=22, caption=17, heading=36, stand=36, sec_gap=0.62),
}


class Style(object):
    def __init__(self, grid="three", palette="navy", type_="arial", title="band",
                 hero="photos", density="normal", look="modern"):
        self.look = look
        pals = {"journal": JOURNAL_PALETTES, "mtsu": MTSU_PALETTES}.get(look, PALETTES)
        typs = {"journal": JOURNAL_TYPES, "mtsu": MTSU_TYPES}.get(look, TYPES)
        if palette not in pals:
            palette = sorted(pals)[0]
        if type_ not in typs:
            type_ = sorted(typs)[0]
        if look == "journal" and title not in JOURNAL_TITLES:
            title = JOURNAL_TITLES[0]
        if look == "mtsu" and title not in MTSU_TITLES:
            title = MTSU_TITLES[0]
        self.name = "%s-%s-%s-%s-%s-%s-%s" % (look, grid, palette, type_, title, hero, density)
        self.grid_name, self.palette_name, self.type_name = grid, palette, type_
        self.title_kind, self.hero, self.density_name = title, hero, density
        self.COL = GRIDS[grid]
        self.FULL = (0.8, 46.4)
        self.c = {k: _rgb(v) for k, v in pals[palette].items()}
        self.f = typs[type_]
        self.d = dict(DENSITY[density])
        # A serif at the same nominal size reads smaller than Arial -- Times has
        # an x-height of 0.448 em against Arial's 0.519. Matching the apparent
        # size means adding a point, not keeping the number. The heading comes
        # down instead: an article head is a label, not a banner.
        if look == "journal":
            self.d["body"] += 1
            self.d["caption"] += 1
            self.d["heading"] -= 6
        elif look == "mtsu":
            # Georgia body, so the same point size reads a size smaller than
            # Arial. The heading comes down to the template's own 30 pt.
            self.d["body"] += 1
            self.d["caption"] += 1
            self.d["heading"] = 30
        # Which plates this look draws from: the slate-blue set in fig/, or the
        # near-monochrome set rendered with POSTER_LOOK=journal.
        self.fig_dir = {"journal": "fig_journal", "mtsu": "fig_mtsu"}.get(look, "fig")

    def describe(self):
        face = self.f["body"] if self.f["body"] == self.f["display"] else (
            "%s / %s" % (self.f["display"], self.f["body"]))
        bits = [self.look, "grid " + self.grid_name, "palette " + self.palette_name,
                face, "title " + self.title_kind, "opens with " + self.hero,
                self.density_name]
        return (u" \u00b7 ").join(bits)

def sample(n, seed=11):
    """N styles, every axis value used about equally, no two alike.

    Each axis is a shuffled deck that is dealt in order and reshuffled when it
    runs out, so a value cannot dominate and cannot be skipped. (The first
    version rotated each axis by k*(i+1), which for an axis of length 4 at
    position 3 advanced by 4k -- i.e. never -- and dealt the same title
    treatment to every sample.)
    """
    rnd = random.Random(seed)
    out, seen = [], set()
    decks = {}

    def deal(key, values):
        if not decks.get(key):
            decks[key] = values[:]
            rnd.shuffle(decks[key])
        return decks[key].pop()

    # Three journal looks for every modern one. Harry asked for the old-journal
    # feel to be the poster's default, not one option among six, so the deck is
    # weighted rather than the modern look being deleted -- a sheet of thirty-six
    # that shows no alternative is not a comparison.
    looks = (["mtsu"] * 5 + ["journal"] * 2 + ["modern"]) * (n // 8 + 2)
    rnd.shuffle(looks)
    tries = 0
    while len(out) < n and tries < n * 30:
        tries += 1
        look = looks[len(out) % len(looks)]
        if look == "mtsu":
            pick = (deal("g", list(GRIDS)), deal("mp", list(MTSU_PALETTES)),
                    deal("mt", list(MTSU_TYPES)), deal("mk", list(MTSU_TITLES)),
                    deal("h", list(HEROES)), deal("d", list(DENSITY)), "mtsu")
        elif look == "journal":
            pick = (deal("g", list(GRIDS)), deal("jp", list(JOURNAL_PALETTES)),
                    deal("jt", list(JOURNAL_TYPES)), deal("jk", list(JOURNAL_TITLES)),
                    deal("h", list(HEROES)), deal("d", list(DENSITY)), "journal")
        else:
            pick = (deal("g", list(GRIDS)), deal("p", list(PALETTES)),
                    deal("t", list(TYPES)), deal("k", list(TITLE_KINDS)),
                    deal("h", list(HEROES)), deal("d", list(DENSITY)), "modern")
        if pick in seen:
            continue
        seen.add(pick)
        out.append(Style(*pick))
    return out


if __name__ == "__main__":
    for s in sample(12):
        print(s.name, "|", s.describe())
