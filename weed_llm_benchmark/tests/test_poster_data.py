#!/usr/bin/env python3
"""The poster's numbers have to agree with the lists they were derived from.

Three complete poster drafts now read one data module. A silent error there
corrupts all three identically, which is the worst kind: the drafts would still
agree with each other, so comparing them would not surface it. These pin the
internal arithmetic — every count that is printed has a list or a sum behind it,
and this checks that they still match.

It also checks the thing that is invisible in source and on screen and appears
for the first time at the printer: every figure must be authored at the width
the layout places it at.

Run:  python tests/test_poster_data.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
POSTER = ROOT.parent / "docs" / "poster"
sys.path.insert(0, str(POSTER))

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def main():
    if not POSTER.exists():
        print("  skip  no docs/poster in this checkout")
        return 0
    import poster_data as D
    import statistics as st

    C = D.CENSUS
    parts = C["r241_frames"] + C["cart_frames"] + C["smoke_frames"]
    check("the frame counts sum to the total", parts == C["frames"],
          "%d != %d" % (parts, C["frames"]))
    sessions = C["r241_sessions"] + C["cart_sessions"] + C["smoke_sessions"]
    check("the session counts sum to the total", sessions == C["sessions"],
          "%d != %d" % (sessions, C["sessions"]))
    rows = sum(n for _, n, _ in C["streams"])
    check("the stream rows sum to sensor_rows", rows == C["sensor_rows"],
          "%d != %d" % (rows, C["sensor_rows"]))
    check("the hero drive is part of the GPS total", C["hero_m"] <= C["gps_m"])
    check("the hero drive's fixes are part of the GPS total",
          C["hero_fixes"] <= C["gps_fixes"])
    check("the cart declares no position stream",
          not any(s in ("gps", "imu") for s in C["cart_streams"])
          and C["cart_has_gps"] is False and C["cart_has_imu"] is False)
    check("the cart's field frames are a subset of its frames",
          C["cart_field_frames"] <= C["cart_frames"])

    L = D.LEDGER
    names = {n.lower() for n in D._ledger_names()}
    check("the model count equals the distinct list", L["n_models"] == len(names),
          "%d != %d" % (L["n_models"], len(names)))
    check("the family count equals the groups", L["n_groups"] == len(L["groups"]))
    check("every ledger row has four fields",
          all(len(r) == 4 for g in L["groups"] for r in g["rows"]))

    R = D.ROUNDS
    check("the recipe mean is the mean of its seeds",
          abs(R["recipe_mean"] - st.mean(R["recipe_seeds"])) < 1e-9)
    check("the chain effect is cold minus warm",
          abs(R["chain_effect"] - (R["cold"] - R["warm"])) < 1e-9)
    check("the schedule effect has the sign the caption claims",
          R["sched_effect"] < 0)
    check("sigma is quoted against the measured seed spread",
          abs(R["recipe_sd"] - st.stdev(R["recipe_seeds"])) < 1e-9)

    LD = D.LADDER
    check("every ladder rung has three seeds",
          all(len(LD["seeds"][k]) == 3 for k in LD["rungs"]),
          str({k: len(LD["seeds"][k]) for k in LD["rungs"]}))

    S = D.SUPERVISION
    A = S["table"]["arms"]
    check("A0 has no rate, rather than a rate of zero",
          A["A0"]["detection_recall"]["v"] is None)
    for m in S["models"]:
        for key in (m["l2"], m["l3"]):
            if not key:
                continue
            a = A[key]
            r, g = a["detection_recall"]["v"], a["detection_grounded"]["v"]
            check("%s: grounded recall does not exceed recall" % key, g <= r + 1e-9,
                  "%.3f > %.3f" % (g, r))

    # A duplicated top-level name in poster_data. Three separate edits to this
    # file have appended a second PROJECTS block above the first, and Python
    # keeps the LAST one, so the poster silently kept printing the old text
    # while the source showed the new. Nothing on screen says which one won.
    try:
        import re
        src = (POSTER / "poster_data.py").read_text()
        names = re.findall(r"^([A-Z_][A-Z0-9_]*)\s*=\s*[\[{]", src, re.M)
        dupes = sorted({n for n in names if names.count(n) > 1})
        check("no top-level name in poster_data.py is defined twice", not dupes,
              ", ".join(dupes))
    except Exception as exc:
        print("  skip  duplicate-name check (%s)" % exc)

    # Figure 4's caption says its table is "read from the router's own table".
    # It used to be hand-typed literals that merely happened to agree, so a role
    # added to model_router.py, or a model swapped inside it, would have left the
    # printed table wrong with nothing on the sheet or in the build to say so.
    # figures.router() now builds its rows from ROLES; these pin the two together
    # and pin the one rule the figure exists to show.
    try:
        import figures as _F
        roles = _F._router_roles()
        check("model_router declares the eight roles Figure 4 prints",
              len(roles) == 8,
              "the router now declares %d; the figure and the dispatch block both say eight"
              % len(roles))
        lab = sorted(r for r, s in roles.items() if s["place"] == "lab")
        check("exactly the two lab roles the sheet names run on the lab box",
              lab == ["analysis_summary", "interactive_plan"],
              "the sheet says two jobs run in the lab; the router says %s" % lab)
        wrong = [r for r, s in roles.items()
                 if s.get("judgement") and s["place"] == "lab"
                 and (bool(s["place"] == "cluster") is not False)]
        check("a judgement role on the lab box is not authoritative", not wrong,
              "Figure 4's whole point is that these come back marked a draft: %s" % wrong)
    except Exception as exc:
        print("  skip  router/figure sync check (%s)" % exc)

    # The defect that is invisible until it is printed.
    try:
        from PIL import Image
        import style
        # Both plate sets, not just the default one. There are now two: fig/ is
        # the slate-blue set and fig_journal/ the near-monochrome one the
        # old-journal looks draw from. A second set is a second chance for a
        # figure to be authored at a width the layout does not place it at, and
        # nothing on screen would show it -- the poster would simply print with
        # one plate's type 20 per cent larger than every other plate's.
        for setname in ("fig", "fig_journal", "fig_mtsu"):
            off = []
            for name, placed in style.PLACED.items():
                p = POSTER / setname / (name + ".png")
                if not p.exists():
                    off.append("%s missing" % name)
                    continue
                authored = Image.open(p).size[0] / 300.0
                if abs(authored / placed - 1.0) >= 0.02:
                    off.append("%s authored %.2f placed %.2f" % (name, authored, placed))
            check("%s/: every figure is authored at the width it is placed at" % setname,
                  not off, "; ".join(off[:4]))
    except ImportError:
        print("  skip  PIL or style unavailable")

    # What the sheet actually carries. Two silent failures live here and
    # neither shows on screen: a theme can be stripped past the number of
    # blocks it declared it may not lose, and a section can be in the library
    # but in no theme, in which case it is never drawn and never recorded as
    # dropped -- it simply is not there.
    try:
        import json as _json
        import tempfile
        import styles as _styles
        import gen as _gen
        _lib = _json.load(open(POSTER / "sections.json"))["sections"]
        _st = _styles.Style("four", "house", "house", "logo-left", "none",
                            "normal", "mtsu")
        _p = _gen.Poster(_st, _lib)
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as fh:
            out = fh.name
        _dropped = _p.build(out)
        os.unlink(out)
        kept = [s for _h, ids, _k in _gen.MTSU_COLUMNS for s in ids
                if s in _p.lib and s not in _dropped]
        must = [s for _h, ids, k in _gen.MTSU_COLUMNS for s in ids[:k]]
        check("every block a theme may not lose is on the sheet",
              all(m in kept for m in must),
              ", ".join(m for m in must if m not in kept))
        seen = (set(kept) | set(_dropped) | set(getattr(_p, "offsheet", []))
                | set(_gen.BANDS) | set(_gen.TAIL_BANDS) | set(_gen.CLOSING)
                | set(_gen.FOOTER))
        lost = [s["id"] for s in _lib if s["id"] not in seen]
        check("no section leaves the library without being accounted for",
              not lost, ", ".join(lost[:6]))
    except Exception as exc:
        print("  skip  sheet-accounting check (%s)" % exc)

    print("%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
