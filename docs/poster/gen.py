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
so a two-column look is 22.4 + 22.4 and a four-column look is 11.6 / 10.925 /
10.925 / 11.6. If a column overflows the sheet its last section is dropped and
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
import styles                                              # noqa: E402

C, L = D.CENSUS, D.LEDGER
FOOT_H = 1.9

SLOT_OF_WIDTH = {11.6: "column", 22.4: "centre", 10.925: "half", 46.4: "full"}
FIGW = {}
try:
    import style as _style
    FIGW = {k: SLOT_OF_WIDTH.get(v, "column") for k, v in _style.PLACED.items()}
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
    ("robots-connected",        ["robots_today", "vehicles", "future_robots"]),
    ("live-collection",         ["uplink", "census", "drive", "auto_diag"]),
    ("remote-control",          ["remote_control", "drive_button", "advice", "laser_control"]),
    ("analysis-agent",          ["analysis_agent", "analysis_example", "analysis_sandbox",
                                 "analysis_gap"]),
    ("iterating-agents-brain",  ["loop", "brain", "diagnosis", "watch", "supervisor"]),
    ("results",                 ["ledger", "ladder", "field", "sources", "journey"]),
]
FOOTER = ["stands", "made", "withdrew"]


class Poster(object):
    def __init__(self, st, lib):
        self.st, self.lib = st, {s["id"]: s for s in lib}
        self.dropped = []
        self.d = Deck(TITLE, STAND, style=st)

    def slot_of(self, sid):
        s = self.lib[sid]
        f = s.get("figure") or "none"
        if f != "none" and not f.startswith("NEW") and f in FIGW:
            return FIGW[f]
        return None

    def section(self, x, y, w, sid, slot, fig_no):
        s, d = self.lib[sid], self.d
        # A section may carry one number set to be read from across a room. The
        # two themes Harry named first -- remote control and the analysis agent --
        # are where these sit, so a reader who only scans the sheet still meets
        # them.
        if s.get("big"):
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
                ("%d" % C["labelled"], "robot frames labelled", "the whole archive, so far")]

    def census_row(self, y):
        d = self.d
        FULL = d.FULL
        cells = self.census_cells()
        cw = (FULL[1] - 4 * 0.55) / 5.0
        d.rect(FULL[0], y, FULL[1], 0.026, fill=d.c["rule"])
        ybot = y + 0.22
        for i, (v, lab, note) in enumerate(cells):
            ybot = max(ybot, d.bignum(FULL[0] + i * (cw + 0.55), y + 0.22, cw, v, lab, note, size=56))
        d.rect(FULL[0], ybot + 0.12, FULL[1], 0.026, fill=d.c["rule"])
        return ybot + 0.30

    def hero(self, fig_no):
        d, kind = self.d, self.st.hero
        FULL, COL = d.FULL, d.COL
        y = d.top
        if kind == "photos":
            y = d.figure(FULL[0], y, FULL[1], "w_robots", "Figure 1."); fig_no[0] += 1
            y = self.census_row(y + 0.2)
        elif kind == "system":
            # the diagram needs a 22.4 slot; find one in this grid, else full-width census
            cen = [c for c in COL if abs(c[1] - 22.4) < 0.05]
            if cen:
                x, w = cen[0]
                y = d.figure(x, y, w, "n_system", "Figure 1."); fig_no[0] += 1
            else:
                y = self.census_row(y)
        elif kind == "census":
            y = self.census_row(y)
        return y + 0.36

    def build(self, out):
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
        ordered = []
        for r in range(rounds):
            for _theme, ids in SPINE:
                if r < len(ids) and ids[r] in self.lib:
                    ordered.append(ids[r])
        cols = [[] for _ in range(n)]
        for sid in ordered:
            want = self.slot_of(sid)
            fit = [i for i, sl in enumerate(slots) if want is None or sl == want
                   or (want == "half" and sl == "centre") or (want == "column" and sl == "half")]
            if not fit:
                continue
            i = min(fit, key=lambda k: len(cols[k]))
            cols[i].append(sid)

        limit = H - FOOT_H - 0.95
        fig_no = [1]
        while True:
            fig_no[0] = 1 + (1 if self.st.hero in ("photos", "system") else 0)
            y0 = self.hero(fig_no)
            ends = []
            for ci, ids in enumerate(cols):
                x, w = COL[ci]
                y = y0
                for sid in ids:
                    y = self.section(x, y, w, sid, slots[ci], fig_no)
                ends.append(y)
            if max(ends) <= limit or not any(cols):
                break
            ci = max(range(n), key=lambda i: ends[i])
            self.dropped.append(cols[ci].pop())
            self.d = Deck(TITLE, STAND, style=self.st); d = self.d

        y = max(max(ends) + 0.45, H - FOOT_H - 0.35)
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
        dropped = Poster(st, lib).build(out)
        manifest.append({"name": name, "style": st.name, "describe": st.describe(),
                         "dropped": dropped, "pptx": out})
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
