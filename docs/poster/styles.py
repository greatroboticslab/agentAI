#!/usr/bin/env python3
"""Visual styles for one poster.

The content of the poster is fixed: agentAI, the MTSU Great Robotics Lab's
research-data platform for physical agents, the robots on it, what the agents
do, what six months of running it produced. What varies here is only how that
one argument LOOKS: the grid, the palette, the type, the title treatment, the
opening image, and how densely it is set.

Every grid is built from the four widths the figures were authored at, 11.6,
22.4, 10.925 and 46.4 inches, so a figure never lands in a slot it was not drawn
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
    "four":       [(0.8, 11.6), (12.8, 10.925), (24.125, 10.925), (35.6, 11.6)],
}

# ---------------------------------------------------------------- type
TYPES = {
    "arial":   dict(display="Arial", body="Arial", display_bold=True),
    "serif":   dict(display="Georgia", body="Arial", display_bold=True),
    "times":   dict(display="Times New Roman", body="Arial", display_bold=True),
    "narrow":  dict(display="Arial Narrow", body="Arial", display_bold=True),
}

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
                 hero="photos", density="normal"):
        self.name = "%s-%s-%s-%s-%s-%s" % (grid, palette, type_, title, hero, density)
        self.grid_name, self.palette_name, self.type_name = grid, palette, type_
        self.title_kind, self.hero, self.density_name = title, hero, density
        self.COL = GRIDS[grid]
        self.FULL = (0.8, 46.4)
        p = PALETTES[palette]
        self.c = {k: _rgb(v) for k, v in p.items()}
        self.f = TYPES[type_]
        self.d = DENSITY[density]

    def describe(self):
        return ("grid %s · palette %s · display %s · title %s · opens with %s · %s"
                % (self.grid_name, self.palette_name, self.f["display"], self.title_kind,
                   self.hero, self.density_name))


def sample(n, seed=11):
    """N styles, every axis value used about equally, no two alike.

    Each axis is a shuffled deck that is dealt in order and reshuffled when it
    runs out, so a value cannot dominate and cannot be skipped. (The first
    version rotated each axis by k*(i+1), which for an axis of length 4 at
    position 3 advanced by 4k -- i.e. never -- and dealt the same title
    treatment to every sample.)
    """
    rnd = random.Random(seed)
    axes = [list(GRIDS), list(PALETTES), list(TYPES), list(TITLE_KINDS), list(HEROES), list(DENSITY)]
    decks = [[] for _ in axes]

    def deal(i):
        if not decks[i]:
            decks[i] = axes[i][:]
            rnd.shuffle(decks[i])
        return decks[i].pop()

    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        pick = tuple(deal(i) for i in range(len(axes)))
        if pick in seen:
            continue
        seen.add(pick)
        out.append(Style(*pick))
    return out


if __name__ == "__main__":
    for s in sample(12):
        print(s.name, "|", s.describe())
