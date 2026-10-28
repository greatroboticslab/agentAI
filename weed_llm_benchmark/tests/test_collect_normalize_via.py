#!/usr/bin/env python3
"""VIA layouts (collect.normalize.read_via, via_regions, via_label).

Intake job 49959269 (zenodo_10762138, 2SeasonWeedDet8) crashed in read_via:
the dataset's per-image VIA files hold their regions as a columnar object,
{"shape_attributes": [...], "region_attributes": [...]} (one object each for a
single region), which the reader took for VIA 1.x's indexed object
(AttributeError: 'list' object has no attribute 'get'). Pinned here:

  * the real files (tests/fixtures/via_2swd8, byte copies whose sha256
    FIXTURE.json records) read as VIA, every box equal to the box of the
    same image's Pascal VOC twin (the dataset ships both; the survey of all
    6,664 files found them equal on every image), and the class names
    exactly the dataset's, numeric and "Unknown" included;
  * every other layout VIA 1.x and 2.x write: a project and an export,
    regions as a list, an indexed object, a columnar object (one region or
    parallel lists), lists nested once; checkbox, dropdown, text and number
    attributes; a checkbox with several names checked keeps its box as
    NO_CLASS; the string-first rule of before is unchanged;
  * a project's region attribute definitions: a typed choice (dropdown,
    radio, checkbox) names the class before a text note, an option id reads
    as its description; a region naming a class in two attributes is
    recorded; a rotated ellipse (theta) is boxed by its rotated extent, a
    polyline is no box; an image described by a file beside it and by one
    that found it by name elsewhere keeps the description beside it
    (recorded), one found by name with no file beside it keeps its entry;
  * anything else refuses with a NormaliseError naming the file, the entry
    and what was found (a VIA 3.x project, regions of another type, parallel
    lists of different lengths), never a crash;
  * intake holds such a source normalise_failed (R3) with that message.

Run:  python3 tests/test_collect_normalize_via.py
"""
import hashlib
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_via_")
check, raises = W.check, W.raises
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "via_2swd8"


def stand_in(path, w, h):
    """A picture of the recorded size (a 1-bit PNG body: PIL reads the size
    from the header, the file stays small); the reader never looks at pixels."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("1", (w, h)).save(path, format="PNG")


def real_tree():
    doc = json.loads((FIX / "FIXTURE.json").read_text())
    root = TMP / "real"
    bad = []
    for f in doc["files"]:
        for ext, key in ((".json", "json_sha256"), (".xml", "xml_sha256")):
            p = FIX / (f["fixture"] + ext)
            if hashlib.sha256(p.read_bytes()).hexdigest() != f[key]:
                bad.append(str(p))
            dst = root / (f["fixture"] + ext)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        stand_in(root / (f["fixture"] + ".jpg"), f["width"], f["height"])
    return doc, root, bad


def test_real_files():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("the real 2SeasonWeedDet8 files")
    doc, root, bad = real_tree()
    check("fixture: every file hashes as FIXTURE.json records", not bad and len(doc["files"]) == 15, bad)
    tree, res = N.read(root)
    check("detected as VIA (the VOC twins noted, not read)", res.format == "via" and res.ignored_formats == ["voc"],
          (res.format, res.ignored_formats))
    _t, voc = N.read(root, {"format": "voc"})
    vn = {c["id"]: c["name"] for c in voc.classes}
    an = {c["id"]: c["name"] for c in res.classes}
    check("the class names are the dataset's own, exactly (those of its VOC twins)",
          sorted(an.values()) == sorted(vn.values()) and {"10", "11", "12", "15", "21", "Unknown",
                                                          "CutleafGroundcherry", "PurpleAmmania"} <= set(an.values()),
          sorted(an.values()))
    vb = {it["rel"]: sorted((vn[b[0]],) + tuple(round(x, 6) for x in b[1:]) for b in it["boxes"]) for it in voc.items}
    ab = {it["rel"]: sorted((an[b[0]],) + tuple(round(x, 6) for x in b[1:]) for b in it["boxes"]) for it in res.items}
    diff = sorted(r for r in set(vb) | set(ab) if vb.get(r) != ab.get(r))
    check("every image's VIA boxes equal its VOC boxes (15 images, single, multi and 32 regions)",
          not diff and len(ab) == 15, diff[:3])
    n = sum(len(v) for v in ab.values())
    big = ab.get("Year2021_Part2/20210903_iPhone11Pro_BT_140.jpg") or []
    check("all regions read (none bad), the 32-region file whole",
          len(big) == 32 and sum(i["bad"] for i in res.items) == 0 and n == sum(len(v) for v in vb.values()),
          (len(big), n))
    check("no problem recorded on the real files", res.problems == [], res.problems)


def via_tree(name, meta, n=1, project=True):
    root = TMP / name
    for i in range(n):
        W.grid_img(root / ("p%d.jpg" % i), 40 + i)
    body = {"_via_img_metadata": meta} if project else meta
    (root / "via.json").write_text(json.dumps(body))
    return root


def entry(i, regions):
    return {"filename": "p%d.jpg" % i, "size": 1, "regions": regions, "file_attributes": {}}


RA = {"name": "rect", "x": 10, "y": 20, "width": 40, "height": 40}
RB = {"name": "polygon", "all_points_x": [100, 160, 130], "all_points_y": [100, 100, 140]}


def names_of(res):
    an = {c["id"]: c["name"] for c in res.classes}
    return [sorted(an.get(b[0], b[0]) for b in it["boxes"]) for it in sorted(res.items, key=lambda x: x["rel"])]


def test_layouts():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("VIA 1.x and 2.x layouts")
    lay = {
        "list (VIA 2.x)": [{"shape_attributes": RA, "region_attributes": {"sp": "a"}},
                           {"shape_attributes": RB, "region_attributes": {"sp": "b"}}],
        "indexed object (VIA 1.x)": {"1": {"shape_attributes": RB, "region_attributes": {"sp": "b"}},
                                     "0": {"shape_attributes": RA, "region_attributes": {"sp": "a"}}},
        "columnar lists": {"shape_attributes": [RA, RB], "region_attributes": [{"sp": {"a": True}},
                                                                                {"sp": {"b": True, "c": False}}]},
        "nested list": [[{"shape_attributes": RA, "region_attributes": {"sp": "a"}},
                         {"shape_attributes": RB, "region_attributes": {"sp": "b"}}]],
        "columnar nested lists": {"shape_attributes": [[RA, RB]], "region_attributes": [[{"sp": "a"}, {"sp": "b"}]]},
    }
    for i, (what, regs) in enumerate(sorted(lay.items())):
        for project in (True, False):
            _t, res = N.read(via_tree("lay%d_%d" % (i, project), {"p0.jpg1": entry(0, regs)}, project=project))
            check("regions as %s, %s: both boxes with their classes" % (what, "project" if project else "export"),
                  res.format == "via" and names_of(res) == [["a", "b"]] and res.items[0]["bad"] == 0,
                  (names_of(res), res.problems))
    _t, res = N.read(via_tree("col1", {"k": entry(0, {"shape_attributes": RA, "region_attributes": {"Weed": {"x": True}}})}))
    check("a columnar object with one region", names_of(res) == [["x"]])
    _t, res = N.read(via_tree("empty", {"k0": entry(0, []), "k1": entry(1, {})}, n=2))
    check("empty regions: the images are listed with no box", [len(i["boxes"]) for i in res.items] == [0, 0])
    regs = [{"shape_attributes": RA, "region_attributes": {"sp": {"a": True, "b": True}}},
            {"shape_attributes": RB, "region_attributes": {"sp": 7}}]
    _t, res = N.read(via_tree("several", {"k": entry(0, regs)}))
    nb = sorted(b[0] for b in res.items[0]["boxes"])
    check("a checkbox with several names checked keeps its box as NO_CLASS (recorded); a number names a class",
          [c["name"] for c in res.classes] == ["7"] and nb == [N.NO_CLASS, 0]
          and any("several classes" in p for p in res.problems), (res.classes, nb, res.problems))
    regs = [{"shape_attributes": RA, "region_attributes": {"stage": {"seedling": True}, "species": "amaranth"}}]
    _t, res = N.read(via_tree("strfirst", {"k": entry(0, regs)}))
    check("a string attribute still wins over a checkbox (the rule before is unchanged)",
          [c["name"] for c in res.classes] == ["amaranth"])
    regs = [{"shape_attributes": RA, "region_attributes": {"stage": {"seedling": True}, "species": "amaranth"}}]
    _t, res = N.read(via_tree("attr", {"k": entry(0, regs)}), {"via_class_attribute": "stage"})
    check("the options' attribute decides", [c["name"] for c in res.classes] == ["seedling"])
    meta = {"k": entry(0, [{"shape_attributes": RA, "region_attributes": {"sp": "a"}}])}
    root = via_tree("twice", meta)
    (root / "via_export.json").write_text(json.dumps(meta))
    _t, res = N.read(root)
    check("an image described by two VIA files keeps the first file's regions once (recorded)",
          names_of(res) == [["a"]] and any("more than one VIA file" in p for p in res.problems), res.problems)


def test_definitions_shapes_elsewhere():
    import math
    from weed_optimizer_framework.tools.collect import normalize as N
    print("a project's attribute definitions, rotated ellipses and polylines, an image found elsewhere")
    defs = {"species": {"type": "dropdown", "description": "", "default_options": {},
                        "options": {"1": "Waterhemp", "2": "PalmerAmaranth", "3": ""}},
            "kind": {"type": "checkbox", "options": {"a": "Amaranth"}}, "note": {"type": "text"}}

    def project(name, meta):
        root = via_tree(name, meta)
        (root / "via.json").write_text(json.dumps({"_via_img_metadata": meta,
                                                   "_via_attributes": {"region": defs, "file": {}}}))
        return root
    regs = [{"shape_attributes": RA, "region_attributes": {"note": "half occluded", "species": "1"}},
            {"shape_attributes": RB, "region_attributes": {"species": "3"}},
            {"shape_attributes": RA, "region_attributes": {"kind": {"a": True}}}]
    _t, res = N.read(project("defs", {"k": entry(0, regs)}))
    check("definitions: a dropdown option id reads as its description (one with none keeps its id), a typed choice "
          "wins over a text note, a checkbox option reads as its description",
          names_of(res) == [sorted(["Waterhemp", "3", "Amaranth"])], (names_of(res), res.problems))
    check("  and the region that names a class in two attributes is recorded",
          any("more than one attribute" in p and "note/species" in p for p in res.problems), res.problems)
    regs = [{"shape_attributes": RA, "region_attributes": {"note": "half occluded", "species": {"Waterhemp": True}}}]
    _t, res = N.read(via_tree("nodefs", {"k": entry(0, regs)}, project=False))
    check("no definitions (an export): the rule before is unchanged (the text first), and the region is recorded",
          names_of(res) == [["half occluded"]]
          and any("more than one attribute" in p and "note/species" in p for p in res.problems),
          (names_of(res), res.problems))

    shapes = [{"shape_attributes": {"name": "ellipse", "cx": 60, "cy": 60, "rx": 20, "ry": 10, "theta": math.pi / 2},
               "region_attributes": {"sp": "e"}},
              {"shape_attributes": {"name": "ellipse", "cx": 60, "cy": 60, "rx": 20, "ry": 10},
               "region_attributes": {"sp": "f"}},
              {"shape_attributes": {"name": "polyline", "all_points_x": [10, 50], "all_points_y": [10, 60]},
               "region_attributes": {"sp": "g"}},
              {"shape_attributes": RB, "region_attributes": {"sp": "h"}}]
    _t, res = N.read(via_tree("shapes", {"k": entry(0, shapes)}))
    an = {c["id"]: c["name"] for c in res.classes}
    wh = {an[b[0]]: (round(b[3] * 144, 3), round(b[4] * 128, 3)) for b in res.items[0]["boxes"]}
    check("an ellipse rotated a quarter turn (theta) is boxed 20 x 40, unrotated 40 x 20; a polyline is no box "
          "(bad), a polygon its extent (clipped)", wh == {"e": (20.0, 40.0), "f": (40.0, 20.0), "h": (44.0, 28.0)}
          and res.items[0]["bad"] == 1, (wh, res.items[0]["bad"]))

    # an entry whose image is absent beside its file, and a same-named photograph in another folder with its own file
    root = TMP / "elsewhere"
    W.grid_img(root / "b" / "p0.jpg", 70)
    (root / "a").mkdir(parents=True)
    reg_a = [{"shape_attributes": RA, "region_attributes": {"sp": "wrong"}}]
    reg_b = [{"shape_attributes": RB, "region_attributes": {"sp": "right"}}]
    (root / "a" / "via.json").write_text(json.dumps({"p0.jpg1": {
        "filename": "p0.jpg", "size": 1, "regions": reg_a, "file_attributes": {}}}))
    (root / "b" / "via.json").write_text(json.dumps({"p0.jpg1": {
        "filename": "p0.jpg", "size": 1, "regions": reg_b, "file_attributes": {}}}))
    _t, res = N.read(root)
    check("an image described beside it (b/) and by an earlier file that found it by name elsewhere (a/): the "
          "description beside it is kept, the other not given to it (recorded)",
          names_of(res) == [["right"]] and "p0.jpg" in res.labels_without_images
          and any("described beside it by b/via.json" in p for p in res.problems), (names_of(res),
                                                                                   res.labels_without_images,
                                                                                   res.problems))
    (root / "b" / "via.json").unlink()
    _t, res = N.read(root)
    check("  with no file beside it, the image found by name keeps the entry's boxes (a project whose images sit in "
          "another folder)", names_of(res) == [["wrong"]] and res.problems == [], (names_of(res), res.problems))


def test_refusals():
    from weed_optimizer_framework.tools.collect import normalize as N
    from weed_optimizer_framework.tools.collect import NormaliseError
    print("anything else refuses, clearly")
    cases = {
        "regions a string": ("str", entry(0, "rect 10 20 40 40"), "regions is a str"),
        "parallel lists of different lengths": ("len", entry(0, {"shape_attributes": [RA, RB],
                                                                 "region_attributes": [{"sp": "a"}]}),
                                                "cannot be paired"),
        "one shape and a list of attribute sets": ("one", entry(0, {"shape_attributes": RA,
                                                                    "region_attributes": [{"sp": "a"}, {"sp": "b"}]}),
                                                   "one shape and a list"),
        "a region element that is a number": ("num", entry(0, [3]), "element 0 is a int"),
        "columnar regions with another key": ("key", entry(0, {"shape_attributes": RA, "region_attributes": {},
                                                               "id": 4}), "other keys"),
    }
    for what, (tag, e, msg) in sorted(cases.items()):
        root = via_tree("bad_" + tag, {"p0.jpg1": e})
        try:
            N.read(root)
            got = None
        except NormaliseError as ex:
            got = str(ex)
        except Exception as ex:  # noqa: BLE001 - a crash is the failure this pins
            got = "CRASH %r" % ex
        check("%s: a NormaliseError naming the file and entry (%s)" % (what, msg),
              got is not None and msg in got and "via.json" in got and "p0.jpg1" in got, got)
    root = TMP / "via3"
    W.grid_img(root / "p0.jpg", 3)
    (root / "project.json").write_text(json.dumps({"project": {"pname": "x"}, "config": {}, "attribute": {},
                                                   "file": {"1": {"fname": "p0.jpg"}}, "metadata": {}, "view": {}}))
    got = None
    try:
        N.read(root)
    except NormaliseError as ex:
        got = str(ex)
    check("a VIA 3.x project refuses, named", got is not None and "VIA 3.x" in got, got)


def test_intake_holds():
    """The intake turns the refusal into its held normalise_failed (R3) with
    the reader's message, as for any NormaliseError (intake.tree_and_read)."""
    import inspect
    from weed_optimizer_framework.tools.collect import intake as I
    src = inspect.getsource(I.intake)
    check("intake: a NormaliseError is held normalise_failed with its message",
          "except NormaliseError as e:" in src and '"normalise_failed"' in src and "str(e)" in src)


def main():
    try:
        test_real_files()
        test_layouts()
        test_definitions_shapes_elsewhere()
        test_refusals()
        test_intake_holds()
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
