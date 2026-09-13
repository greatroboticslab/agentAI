#!/usr/bin/env python3
"""Compose many posters from one section library and one set of numbers.

    python3 docs/poster/gen.py sections.json [--n 24] [--out variants/]

A poster is a SPEC: a title, a standfirst, a hero, three columns of section ids
and a footer. The engine flows each column, places every figure at the width it
was authored at (an 11.6 in figure only ever lands in an 11.6 in slot), and if a
column overflows the sheet it drops that column's last section and records it.
Nothing is typed by hand: every number is `poster_data`, every sentence is the
library, every figure is `fig/`.

Then it renders every spec to PPTX and PNG and tiles the PNGs into a contact
sheet, so forty drafts can be compared on one screen before any one is opened.
"""
import argparse
import itertools
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deck import (Deck, D, COL, FULL, CAPTION, BODY, HERE as DECK_HERE, MUTE, RULE,   # noqa: E402
                  INK, BLUE)

C, L = D.CENSUS, D.LEDGER

# figure name -> the slot width it was drawn for
SLOT = {11.6: "column", 22.4: "centre", 10.925: "half", 46.4: "full"}
FIGW = {}
try:
    import style
    FIGW = {k: SLOT.get(v, "column") for k, v in style.PLACED.items()}
except Exception:
    pass

TOP = 6.15
FOOT_H = 1.9


class Poster(object):
    def __init__(self, spec, lib):
        self.spec, self.lib = spec, {s["id"]: s for s in lib}
        self.dropped = []
        self.d = Deck(spec["title"], spec.get("standfirst"))

    # ---- one section into a slot ------------------------------------------
    def section(self, x, y, w, sid, slot, fig_no):
        s = self.lib[sid]
        d = self.d
        y = d.head(x, y, w, s["heading"])
        fig = s.get("figure") or "none"
        placed = False
        if fig != "none" and not fig.startswith("NEW") and FIGW.get(fig) == slot:
            y = d.figure(x, y, w, fig, "Figure %d." % fig_no[0])
            fig_no[0] += 1
            placed = True
        body = s["body"]
        if s.get("status") == "designed_only":
            body += " This part is a design and has not been built."
        elif s.get("status") == "built_but_off":
            body += " This part is built and is switched off today."
        y = d.body(x, y, w, body, size=BODY, after=14)
        return y + 0.42, placed

    # ---- the hero band ------------------------------------------------------
    def hero(self, kind, fig_no):
        d = self.d
        y = TOP
        if kind == "photos":
            y = d.figure(FULL[0], y, FULL[1], "w_robots", "Figure 1."); fig_no[0] += 1
            y = self.census_row(y + 0.2)
        elif kind == "journey":
            y = d.head(FULL[0], y, FULL[1], "Six months, in order")
            y = d.figure(FULL[0], y, FULL[1], "t_journey", "Figure 1."); fig_no[0] += 1
        elif kind == "system":
            y1 = d.figure(COL[1][0], y, COL[1][1], "n_system", "Figure 1."); fig_no[0] += 1
            yl = d.head(COL[0][0], y, COL[0][1], "The two vehicles")
            yl = d.figure(COL[0][0], yl, COL[0][1], "l_robots", "Figure 2."); fig_no[0] += 1
            yr = d.head(COL[2][0], y, COL[2][1], "On disk today")
            for val, lab, note in self.census_cells()[:3]:
                yr = d.bignum(COL[2][0], yr, COL[2][1], val, lab, note, size=54) + 0.12
            y = max(y1, yl, yr)
        elif kind == "census":
            y = self.census_row(y)
        return y + 0.36

    def census_cells(self):
        return [("{:,}".format(C["frames"]), "camera frames", "%d drives, 2 robots, %d labelled"
                 % (C["sessions"], C["labelled"])),
                ("{:,}".format(C["sensor_rows"]), "telemetry rows", "nine streams, counted line by line"),
                ("%.0f m" % C["gps_m"], "of GPS track", "%s fixes over %d drives"
                 % ("{:,}".format(C["gps_fixes"]), C["gps_sessions"])),
                ("%d" % L["n_models"], "distinct models run", "four families, one deployed"),
                ("%d" % C["labelled"], "robot frames labelled", "the whole archive, so far")]

    def census_row(self, y):
        d = self.d
        cells = self.census_cells()
        cw = (FULL[1] - 4 * 0.55) / 5.0
        d.rect(FULL[0], y, FULL[1], 0.026, fill=RULE)
        ybot = y + 0.22
        for i, (v, lab, note) in enumerate(cells):
            ybot = max(ybot, d.bignum(FULL[0] + i * (cw + 0.55), y + 0.22, cw, v, lab, note, size=56))
        d.rect(FULL[0], ybot + 0.12, FULL[1], 0.026, fill=RULE)
        return ybot + 0.30

    # ---- flow the columns, dropping from the longest on overflow -----------
    def build(self, out):
        d = self.d
        fig_no = [1]
        y0 = self.hero(self.spec.get("hero", "census"), fig_no)
        cols = [list(c) for c in self.spec["columns"]]
        limit = 36.0 - FOOT_H - 0.95
        while True:
            # dry run is expensive with pptx, so flow for real and rebuild on overflow
            ends = []
            for ci, ids in enumerate(cols):
                x, w = COL[ci]
                slot = "column" if ci != 1 else "centre"
                y = y0
                for sid in ids:
                    y, _ = self.section(x, y, w, sid, slot, fig_no)
                ends.append(y)
            if max(ends) <= limit or not any(cols):
                break
            # overflow: drop the last section of the longest column and start over
            ci = max(range(3), key=lambda i: ends[i])
            if not cols[ci]:
                break
            self.dropped.append(cols[ci].pop())
            self.d = Deck(self.spec["title"], self.spec.get("standfirst"))
            d = self.d
            fig_no = [1]
            y0 = self.hero(self.spec.get("hero", "census"), fig_no)
        y = max(max(ends) + 0.45, 36.0 - FOOT_H - 0.35)
        d.rect(FULL[0], y - 0.25, FULL[1], 0.022, fill=RULE)
        fw = (FULL[1] - 1.0) / 3.0
        for i, sid in enumerate(self.spec.get("footer", [])[:3]):
            s = self.lib.get(sid)
            if s:
                d.body(FULL[0] + i * (fw + 0.5), y, fw, "%s. %s" % (s["heading"], s["body"]),
                       size=CAPTION, color=MUTE)
        d.save(out)
        return self.dropped


# ---------------------------------------------------------------- generator
TITLES = [
    "Two Field Robots and an Agent Platform for Laser Weeding",
    "Robots in the Field, Agents on the Cluster",
    "An Agent Platform That Collects, Trains and Watches Itself",
    "Field Robots, Live Data, and the Agents That Use It",
    "From a Crop Row to a Checkpoint Without a Person in the Loop",
    "Six Months of an Agent Platform for Field Weed Detection",
]
STANDS = [
    "Anyone can download a weed dataset this afternoon. Nobody can download the path from a robot in a Tennessee crop row to a trained checkpoint. We built that path, ran it unattended, and measured every stage of it.",
    "Two ground vehicles drive our own crop plots while a pair of agents collect, filter and train on the cluster with nobody in the room. This is what the machines recorded, what the agents did, and where the path still breaks.",
    "We connected every robot we own to one platform, let agents decide what to collect and train, and put a third agent in charge of watching the other two. Here is what that bought and what it cost.",
]
HEROES = ["photos", "system", "journey", "census"]


def specs_from(lib, n, seed=7):
    """N distinct specs.

    Every spec starts with the WHOLE library, ordered by a shuffled theme
    priority, each section sent to a column whose slot matches its figure's
    authored width. The overflow loop in `Poster.build` then trims from the
    longest column, so what survives on a given poster is what its theme order
    put first. Variation comes from title x standfirst x hero x theme order;
    fullness comes from starting with everything.
    """
    rnd = random.Random(seed)
    by_theme = {}
    for s in lib:
        by_theme.setdefault(s["theme"], []).append(s)
    themes = ["platform-idea", "robots-connected", "live-collection", "remote-control",
              "analysis-agent", "iterating-agents-brain", "models", "results"]
    honesty = [s["id"] for s in by_theme.get("honesty", [])]

    def slot_of(s):
        f = s.get("figure") or "none"
        if f != "none" and not f.startswith("NEW") and f in FIGW:
            return {"column": "side", "centre": "centre", "half": "centre", "full": "any"}[FIGW[f]]
        return {"centre": "centre", "column": "side", "full": "any"}.get(s.get("width"), "any")

    combos = list(itertools.product(range(len(TITLES)), range(len(STANDS)), HEROES))
    rnd.shuffle(combos)
    out = []
    for k, (ti, si, hero) in enumerate(combos[:n]):
        order = themes[:]
        rnd.shuffle(order)
        # the hero already carries some themes; push those to the back
        covered = {"photos": ["robots-connected"], "system": ["platform-idea", "robots-connected"],
                   "journey": ["iterating-agents-brain"], "census": []}[hero]
        order = [t for t in order if t not in covered] + [t for t in order if t in covered]
        cols, side_turn = [[], [], []], 0
        for th in order:
            pool = by_theme.get(th, [])[:]
            rnd.shuffle(pool)
            for s in pool:
                where = slot_of(s)
                if where == "centre":
                    cols[1].append(s["id"])
                elif where == "side":
                    cols[[0, 2][side_turn % 2]].append(s["id"]); side_turn += 1
                else:
                    # a figure-less section goes to whichever column is shortest
                    ci = min(range(3), key=lambda i: len(cols[i]))
                    cols[ci].append(s["id"])
        out.append({"name": "v%02d" % (k + 1), "title": TITLES[ti], "standfirst": STANDS[si],
                    "hero": hero, "columns": cols, "footer": honesty[:3]})
    return out


def contact_sheet(pngs, out, cols=4, thumb_w=900):
    from PIL import Image, ImageDraw
    ims = []
    for p in pngs:
        im = Image.open(p)
        r = thumb_w / im.width
        ims.append((os.path.basename(p), im.resize((thumb_w, int(im.height * r)))))
    if not ims:
        return None
    tw, th = ims[0][1].size
    rows = -(-len(ims) // cols)
    sheet = Image.new("RGB", (cols * (tw + 24) + 24, rows * (th + 60) + 24), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(ims):
        x = 24 + (i % cols) * (tw + 24)
        y = 24 + (i // cols) * (th + 60)
        sheet.paste(im, (x, y + 36))
        dr.text((x, y + 8), name, fill=(30, 41, 59))
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "variants"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    lib = json.load(open(a.library))
    lib = lib.get("sections", lib) if isinstance(lib, dict) else lib
    os.makedirs(a.out, exist_ok=True)
    specs = specs_from(lib, a.n, a.seed)
    manifest = []
    for sp in specs:
        out = os.path.join(a.out, sp["name"] + ".pptx")
        dropped = Poster(sp, lib).build(out)
        manifest.append({**sp, "dropped": dropped, "pptx": out})
        print("  %s  hero=%-7s dropped=%s" % (sp["name"], sp["hero"], dropped or "-"))
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    if a.render:
        pngs = []
        for m in manifest:
            subprocess.run(["soffice", "--headless", "--convert-to", "png", "--outdir", a.out, m["pptx"]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            png = m["pptx"][:-5] + ".png"
            if os.path.exists(png):
                pngs.append(png)
        cs = contact_sheet(pngs, os.path.join(a.out, "contact_sheet.png"))
        print("contact sheet:", cs, "(%d posters)" % len(pngs))


if __name__ == "__main__":
    main()
