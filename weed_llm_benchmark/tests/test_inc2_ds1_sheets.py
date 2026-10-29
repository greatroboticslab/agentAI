#!/usr/bin/env python3
"""DS1's sheets (inc2/ds1_sheets.py: lever L31's render on the lab, the `record` step; docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-29): DS1", DS.0 items 4 and 20, DS.8.2, DS.9.1, DS.10).

- the seeded samples: a source sheet's 24 admitted, 12 removed and 12 per class unit; the owner's audit sample (the
  5 lowest stable_int("inc2/ds1/audit/<source>")); LN's 550 images (200, 100, 250) mixed by
  stable_int("inc2/ds1/ln/<sha256>"); the MT-W check's min(60, all) plus the flagged images;
- a bundle written once with every file's sha256; a tampered file refuses the render;
- the rendered page: one self-contained HTML file (no external URL), every image inline with every box drawn and
  its class written, headings in Chinese and English, one decision box per source, the sheet's sha256 inside, a
  save button; its render record (the spec's and the file's sha256);
- `record`: a saved decisions file round-trips (each decision with decided_by, the sheet's sha256, the decider, the
  time); a file naming no rendered sheet, a decision outside the three, a source the sheet does not ask, an audit
  file decided by the delegate, and the same file twice are refused;
- the fold: the delegate's decision governs until the owner's audit; a reversal is recorded as his, governs, and
  widens the audit to every decided sheet; names-round sources and SIU's table always need his audit; the
  MT-W checks (pass, fail, whole-source) and LN's marks.

Run:  python3 tests/test_inc2_ds1_sheets.py
"""
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import traceback

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_sheets_"))
os.environ["INC_DIR"] = str(TMP / "inc")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_data as DD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_sheets as SH  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except D.Ds1Refused as e:
        return e
    return None


def img(name, seed, wh=(320, 240)):
    p = TMP / "src" / ("%s.png" % name)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (wh[1], wh[0], 3), dtype=np.uint8)).save(p)
    return str(p)


def rows(source, n, seed0, units=(12,)):
    return [{"key": "%s__%03d" % (source, i), "image": img("%s_%d" % (source, i), seed0 + i), "rel": "f_%03d.jpg" % i,
             "boxes": [(units[i % len(units)], 0.5, 0.5, 0.3, 0.3), (units[0], 0.2, 0.25, 0.1, 0.1)], "source": source}
            for i in range(n)]


def t_samples():
    print("the seeded samples")
    adm = rows("src_a", 40, 100, units=(12, 3))
    rem = [dict(r, reason="duplicate") for r in rows("src_a_rm", 20, 300)]
    items = SH.source_sample("src_a", adm, rem, DD.unit_of)
    groups = {}
    for g, _r in items:
        groups[g] = groups.get(g, 0) + 1
    check("24 admitted, 12 removed, 12 per class unit", groups.get("admitted") == 24 and groups.get("removed") == 12
          and groups.get("unit:inc:12") == 12 and groups.get("unit:inc:3") == 12, groups)
    check("  the same draw again (seeded)", [r["key"] for _g, r in SH.source_sample("src_a", adm, rem, DD.unit_of)] ==
          [r["key"] for _g, r in items])
    srcs = ["src_%02d" % i for i in range(12)]
    a = SH.audit_sample(srcs)
    check("the owner's audit sample: the 5 lowest stable_int(inc2/ds1/audit/<source>)", len(a) == 5 and a ==
          sorted(srcs, key=lambda s: (C.stable_int("inc2/ds1/audit/%s" % s), s))[:5])
    groups_ln = {"siu_increments": [{"key": "i%d" % i, "sha256": "s%04d" % i} for i in range(300)],
                 "inc0005": [{"key": "j%d" % i, "sha256": "t%04d" % i} for i in range(150)],
                 "train_core": [{"key": "k%d" % i, "sha256": "u%04d" % i} for i in range(400)]}
    ln = SH.ln_draw(groups_ln)
    by = {}
    for r in ln:
        by[r["ln_group"]] = by.get(r["ln_group"], 0) + 1
    check("LN: 550 images, 200 / 100 / 250, mixed by stable_int(inc2/ds1/ln/<sha256>)", len(ln) == 550 and by ==
          {"siu_increments": 200, "inc0005": 100, "train_core": 250} and [r["sha256"] for r in ln] == sorted(
              (r["sha256"] for r in ln), key=lambda s: (C.stable_int("inc2/ds1/ln/%s" % s))))
    check("  a group short of its size refuses", refused(SH.ln_draw, dict(groups_ln, inc0005=groups_ln["inc0005"][:9])))
    mrows = [{"key": "m%03d" % i} for i in range(100)]
    pick, whole = SH.mtw_sample("src_m", mrows, flagged=["m099", "m098"])
    check("MT-W: 60 seeded images plus every flagged one", len(pick) >= 60 and {"m099", "m098"} <= {r["key"] for r in
                                                                                                   pick} and not whole)
    pick2, whole2 = SH.mtw_sample("src_n", mrows[:35])
    check("  a source of 35 MT-W images is checked whole", len(pick2) == 35 and whole2)


def t_render_record():
    print("render (one self-contained HTML per batch) and record")
    srcs = {"src_a": rows("src_a", 30, 1000, units=(12, 3)), "src_b": rows("src_b", 30, 2000)}
    rec = {s: {"rows": 30, "live": 30, "units": {"inc:12": {"boxes": 50, "suggested": "unresolved"},
                                                "inc:3": {"boxes": 10, "suggested": "weed"}} if s == "src_a" else
               {"inc:12": {"boxes": 60, "suggested": "unresolved"}}, "unit_names": {"inc:12": ["weed", "other"]},
               "capture_names": {"hits": 0, "examples": []}, "convention": {"passes": True}, "cwd12_hits": 0}
           for s in srcs}
    made = SH.bundle_sources(1, 1, srcs, rec, out_rows={"src_a": [dict(r, why="duplicate") for r in rows(
        "src_a_x", 15, 3000)]})
    check("one bundle per source, bundle.json listing every file with its sha256", len(made) == 2 and all(
        (SH.bundle_dir(n) / "bundle.json").is_file() for n in made))
    b = SH.verify_bundle(made[0])
    check("  every bundled file hashes as recorded; written once", all(C.sha256_file(SH.bundle_dir(made[0]) / it[
        "file"]) == it["sha256"] for it in b["items"] if it.get("file")) and SH.write_bundle(made[0], [], {}) == b)
    r = SH.render("source", 1)
    html = pathlib.Path(r["html"]).read_text(encoding="utf-8")
    check("the page is self-contained: images inline, no external URL", "data:image/jpeg;base64," in html and
          not re.search(r"(src|href)=\"https?://", html))
    check("  headings in Chinese and English, a decision box per source, the save button",
          "来源审核表 Source sheets" in html and html.count('class="decide"') == 2 and 'id="save"' in html and
          "接受 accept" in html and "拒绝 reject" in html)
    check("  every image drawn with every box and its class written", html.count('<rect class="box"') >= 2 * (
        24 + 12 + 12) and "inc:12" in html)
    check("  the sheet's sha256 inside the page, the render record names it and the file's",
          r["spec_sha256"] in html and C.sha256_file(r["html"]) == r["html_sha256"] and
          SH.rendered_path("source", 1).is_file())
    dec = {"format": SH.FMT_DECISIONS, "sheet": {"kind": "source", "batch": 1, "spec_sha256": r["spec_sha256"]},
           "decider": "Delegate", "saved_utc": "2026-11-09T10:00:00Z",
           "sources": {"src_a": {"decision": "accept", "setting": "Field", "roles": {"inc:12": "weed", "inc:3": "weed"},
                                 "family": None, "series": False, "note": ""},
                       "src_b": {"decision": "reject", "setting": "Turf", "roles": {"inc:12": "crop"}}}}
    f = TMP / "dec1.json"
    f.write_text(json.dumps(dec))
    got = SH.record(str(f), "delegate")
    check("a saved decisions file round-trips through record (decided_by, the sheet's sha256, the decider, the time)",
          len(got) == 2 and all(x["decided_by"] == "delegate" and x["sheet_sha256"] == r["spec_sha256"] and
                                x["decider"] == "Delegate" for x in got))
    check("  the same file twice is refused", refused(SH.record, str(f), "delegate"))
    bad = dict(dec, sheet=dict(dec["sheet"], spec_sha256="0" * 64))
    (TMP / "bad1.json").write_text(json.dumps(bad))
    check("  a file naming no rendered sheet is refused", refused(SH.record, str(TMP / "bad1.json"), "delegate"))
    bad2 = json.loads(json.dumps(dec))
    bad2["sources"]["src_a"]["decision"] = "maybe"
    bad2["saved_utc"] = "x"
    (TMP / "bad2.json").write_text(json.dumps(bad2))
    check("  a decision outside accept / accept for training only / reject is refused", refused(
        SH.record, str(TMP / "bad2.json"), "delegate"))
    bad3 = json.loads(json.dumps(dec))
    bad3["sources"]["src_c"] = bad3["sources"]["src_b"]
    (TMP / "bad3.json").write_text(json.dumps(bad3))
    check("  a source the sheet does not ask is refused", refused(SH.record, str(TMP / "bad3.json"), "delegate"))
    check("  decided_by is human:<id> or delegate", refused(SH.record, str(f), "someone"))
    fd = SH.fold()
    check("the fold: the delegate's decision governs before any audit", fd["src_a"]["decision"] == "accept" and
          fd["src_b"]["decision"] == "reject" and not fd["src_a"]["audited"])
    return r


def t_audit():
    print("the owner's audit: the seeded 5, a reversal widening it, names-round sources and SIU's table")
    srcs = {"src_%02d" % i: rows("src_%02d" % i, 26, 5000 + 100 * i) for i in range(7)}
    rec = {s: {"rows": 26, "live": 26, "units": {"inc:12": {"boxes": 52, "suggested": "unresolved"}}} for s in srcs}
    made = SH.bundle_sources(1, 2, srcs, rec)
    batches = len(SH.source_chunks(SH.bundles("source")))
    check("a later screen revision's bundles form new batches; batch 1 is unchanged",
          SH.batch_bundles("source", 1) == sorted(n for n in SH.bundles("source") if "_r001_" in n) and batches == 3)
    for b in range(2, batches + 1):
        r = SH.render("source", b)
        decs = {s: {"decision": "accept", "setting": "Field", "roles": {"inc:12": "weed"}} for s in r["sources"]}
        f = TMP / ("dec_b%d.json" % b)
        f.write_text(json.dumps({"format": SH.FMT_DECISIONS, "sheet": {"kind": "source", "batch": b,
                                                                       "spec_sha256": r["spec_sha256"]},
                                 "decider": "Delegate", "saved_utc": "2026-11-10T00:00:00Z", "sources": decs}))
        SH.record(str(f), "delegate")
    fd = SH.fold()
    phase_srcs = sorted(fd)
    want = set(SH.audit_sample(phase_srcs))
    req = {s for s, v in fd.items() if v["audit_required"]}
    check("before any reversal the audit applies to the seeded 5 (of the phase's decided sheets)", req == want,
          (sorted(req), sorted(want)))
    pend = SH.audit_sources(fd)
    check("  the audit batch shows them", set(pend) == want)
    r = SH.render("audit", 1)
    check("  the audit sheet shows the delegate's decision beside the owner's box", "The delegate decided" in
          pathlib.Path(r["html"]).read_text(encoding="utf-8"))
    decs = {s: {"decision": "accept", "setting": "Field", "roles": {"inc:12": "weed"}} for s in r["sources"]}
    first = r["sources"][0]
    decs[first] = {"decision": "reject", "setting": "Field", "roles": {"inc:12": "crop"}}
    f = TMP / "audit1.json"
    f.write_text(json.dumps({"format": SH.FMT_DECISIONS, "sheet": {"kind": "audit", "batch": 1,
                                                                   "spec_sha256": r["spec_sha256"]},
                             "decider": "Harry", "saved_utc": "2026-11-11T00:00:00Z", "sources": decs}))
    check("an audit file decided by the delegate is refused", refused(SH.record, str(f), "delegate"))
    SH.record(str(f), "human:harry567566")
    fd2 = SH.fold()
    check("a reversal is recorded as the owner's and governs", fd2[first]["decision"] == "reject" and
          fd2[first]["reversed"] and fd2[first]["decided_by"] == "human:harry567566")
    check("  and widens the audit to every decided sheet", all(v["audit_required"] for v in fd2.values() if
                                                               v.get("delegate_decision")))
    check("  the audited sources are done; the others now wait for him", all(fd2[s]["audited"] for s in r["sources"])
          and set(SH.audit_sources(fd2)) == {s for s in fd2 if not fd2[s]["audited"]})
    names = C.INC_DIR / "intake" / "names"
    names.mkdir(parents=True, exist_ok=True)
    (names / "names_src_x.json").write_text(json.dumps({"source": "src_00"}))
    check("a names-round source always needs his audit", "names round" in SH.fold()["src_00"]["audit_reason"])
    # without a reversal: a names-round source outside the seeded 5 still needs his audit, another one does not
    inc2 = TMP / "inc_names_round"
    dp = SH.decisions_path(inc2)
    dp.parent.mkdir(parents=True, exist_ok=True)
    srcs = ["nr_%02d" % i for i in range(9)]
    dp.write_text("".join(json.dumps({"kind": "source", "source": s_, "decision": "accept", "setting": "Field",
                                      "roles": {"inc:12": "weed"}, "decided_by": "delegate", "sheet_sha256": "x"})
                          + "\n" for s_ in srcs))
    f0 = SH.fold(inc2)
    outside = [s_ for s_ in srcs if not f0[s_]["audit_required"]]
    (inc2 / "intake" / "names").mkdir(parents=True, exist_ok=True)
    (inc2 / "intake" / "names" / "names_nr.json").write_text(json.dumps({"source": outside[0]}))
    f1 = SH.fold(inc2)
    check("  without a reversal: a names-round source outside the seeded 5 needs his audit; another source outside "
          "it does not", len(outside) >= 2 and f1[outside[0]]["audit_required"] and not f1[outside[1]][
              "audit_required"] and sum(1 for s_ in srcs if f0[s_]["audit_required"]) <= 5,
          (outside, {s_: f1[s_]["audit_reason"] for s_ in srcs}))


def t_folds():
    print("the MT-W checks and LN's marks")
    p = SH.decisions_path()
    with open(p, "a", encoding="utf-8") as fh:
        for i in range(60):
            fh.write(json.dumps({"kind": "mtw", "source": "src_m", "item": "mtw|src_m|k%d" % i, "error": i == 3,
                                 "removed": [], "added": [], "total": 400}) + "\n")
        for i in range(60):
            fh.write(json.dumps({"kind": "mtw", "source": "src_f", "item": "mtw|src_f|k%d" % i, "error": i in (1, 2),
                                 "removed": [], "added": [], "total": 400}) + "\n")
        for i in range(20):
            fh.write(json.dumps({"kind": "mtw", "source": "src_w", "item": "mtw|src_w|k%d" % i, "error": i == 0,
                                 "removed": [0] if i == 0 else [], "added": [[0.5, 0.5, 0.1, 0.1]] if i == 0 else [],
                                 "total": 20}) + "\n")
        for g, n in (("siu_increments", 3), ("train_core", 2)):
            for i in range(n):
                fh.write(json.dumps({"kind": "ln", "group": g, "marks": ["correct", "loose"], "missing": 1}) + "\n")
    m = SH.mtw_checks()
    check("MT-W: 1 error in 60 passes, 2 fail", m["src_m"]["verdict"] == "pass" and m["src_f"]["verdict"] == "fail")
    check("  a source checked whole passes once its errors are corrected (the correction logged)",
          m["src_w"]["verdict"] == "pass" and m["src_w"]["corrections"] == 1 and m["src_w"]["corrected_labels"])
    ln = SH.ln_marks()
    check("LN's marks folded per group", len(ln["siu_increments"]) == 3 and len(ln["train_core"]) == 2)
    st = SH.status()
    check("status: what waits for whom", "src_m" in st["mtw_checks"] and st["ln_recorded"] == 5)


def t_tamper(r):
    print("a tampered bundle file refuses the render")
    name = r["bundles"][0]["name"]
    d = SH.bundle_dir(name)
    f = next(p for p in d.iterdir() if p.suffix == ".jpg")
    raw = f.read_bytes()
    f.write_bytes(raw[:-10] + b"0123456789")
    check("refused (no file rendered unverified)", refused(SH.render, "source", 1))
    f.write_bytes(raw)


def main():
    r = None
    for fn in (t_samples, t_render_record, t_audit, t_folds):
        try:
            got = fn()
            r = got or r
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2000:]))
    if r:
        try:
            t_tamper(r)
        except Exception as e:                   # noqa: BLE001
            check("t_tamper ran", False, "%s: %s" % (type(e).__name__, e))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
