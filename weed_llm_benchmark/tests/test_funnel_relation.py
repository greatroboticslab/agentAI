#!/usr/bin/env python3
"""The relation module: geometry match, id alignments, relation audit, class-map
proposals (docs/FUNNEL_AUDIT.md §6 H0b, H0c, H1-pre, H3a; §8.5 L11; runner
§4.14, §5.2.4).

What is pinned, and why:
  * read_upstream reads VOC (names, pixel boxes), YOLO with a classes file
    (ids, normalised boxes) and VOC+YOLO (pixel boxes, YOLO ids) from a zip or
    a directory, and counts capture stems when the config gives a regex;
  * geometry match on a synthetic upstream with a planted "drop class 5"
    export: the chosen alignment is drop5, identity and +-1 score lower, and
    the choice is confirmed on the other half; a 2 px shift still matches and
    a 5 px shift does not; a YOLO upstream rounded to 2 decimals matches
    within its own rounding step;
  * relation audit (H0b rule) on synthetic copies with a planted old join:
    the old join is flagged, the current join is not, every unit maps to its
    truth; run_relation writes relation_audit_v1.json with H0b and H0c;
  * a card class map whose kept class count disagrees with the source's goes
    to the person queue (to_L14);
  * run_geometry on the synthetic world (fetched-card layout written as fetch
    writes it): H3a passes, the card maps id 12 to Sicklepod, H1-pre passes
    for a card whose names agree and fails, naming the swapped pair, for one
    with an off-by-one class order; class_maps.json is immutable (a rerun with
    other proposals refuses without --force); KT6 appears once H3a passes.
  * fail-closed rules: an untested H0(b) or H0(c) check is not a pass; H0(c)
    fails on any resolved named non-target mapped to a target; a card whose
    names differ from the upstream's goes to L14 and gives no KT6 truth;
    run_geometry needs the census, refuses before writing anything, and a
    rerun with the same inputs rewrites nothing; classes files that number
    the classes differently refuse; an unconfirmed alignment is reported.

Run:  python3 tests/test_funnel_relation.py
"""
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import time
import zipfile
import funnel_prereg as FPR  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_relation_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))


class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise AssertionError("the test touched the network")


socket.socket = _NoNet

import numpy as np  # noqa: E402

import funnel_world as FW  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.funnel import RelationError, read_json  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import relation as R  # noqa: E402

FAILURES, SKIPS = [], []
RNG = np.random.default_rng(11)


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return None


def voc(fname, W, H, objs):
    o = "".join("<object><name>%s</name><bndbox><xmin>%g</xmin><ymin>%g</ymin><xmax>%g</xmax><ymax>%g</ymax>"
                "</bndbox></object>" % tuple(x) for x in objs)
    return ("<annotation><filename>%s</filename><size><width>%d</width><height>%d</height><depth>3</depth></size>%s"
            "</annotation>" % (fname, W, H, o))


def synthetic_upstream(n_files=80, n_classes=16, W=640, H=480):
    """{stem: [(id, x0, y0, x1, y1)]}: 2-6 non-overlapping boxes per file, every class present."""
    ups = {}
    for f in range(n_files):
        boxes = []
        k = 2 + int(RNG.integers(0, 5))
        for b in range(k):
            cid = (f * 3 + b * 5) % n_classes
            x0 = 20 + 110 * b + float(RNG.uniform(0, 10))
            y0 = float(RNG.uniform(10, 300))
            boxes.append((cid, round(x0), round(y0), min(W - 1, round(x0 + 60 + RNG.uniform(0, 30))),
                          min(H - 1, round(y0 + 70 + RNG.uniform(0, 60)))))
        ups["vid%03d.mp4_%d" % (f // 4, f)] = boxes
    return ups, W, H


def write_zip(path, ups, W, H, kind):
    with zipfile.ZipFile(path, "w") as z:
        for i, (stem, boxes) in enumerate(sorted(ups.items())):
            member = "set/%s_%d" % ("f%05d" % i, i)
            if "voc" in kind:
                z.writestr("set/PASCAL_VOC/%s.xml" % member.split("/")[-1],
                           voc(stem + ".png", W, H, [("name%d" % b[0],) + tuple(b[1:]) for b in boxes]))
            if "yolo" in kind:
                z.writestr("set/YOLO_darknet/%s.txt" % member.split("/")[-1], "".join(
                    "%d %.6f %.6f %.6f %.6f\n" % (b[0], (b[1] + b[3]) / 2 / W, (b[2] + b[4]) / 2 / H,
                                                  (b[3] - b[1]) / W, (b[4] - b[2]) / H) for b in boxes))


def pool_from(up, amap, W, H, shift_px=0.0):
    pool = {}
    for f, boxes in up.items():
        rows = []
        for (cid, x0, y0, x1, y1, _W, _H) in boxes:
            a = amap.get(cid)
            if a is None:
                continue
            rows.append((a, (x0 + x1) / 2 / W + shift_px / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H, W, H))
        if rows:
            pool["pool_" + f] = rows
    return pool


def test_upstream_and_alignment():
    print("read_upstream")
    ups, W, H = synthetic_upstream()
    z = TMP / "ann.zip"
    write_zip(z, ups, W, H, "voc+yolo")
    up = R.read_upstream(z, "voc+yolo", voc_dir="PASCAL_VOC", yolo_dir="YOLO_darknet",
                         frame_name_regex=r"^(?P<stem>.+?\.mp4)_(?P<frame>\d+)\.png$")
    check("VOC+YOLO: every file read, pixel boxes with YOLO ids", len(up) == len(ups)
          and all(isinstance(b[0], int) and b[5] == W for bs in up.values() for b in bs))
    check("the id -> name table comes from the paired VOC names",
          all(up.names[i] == "name%d" % i for i in up.names), up.names)
    check("capture stems are counted from the VOC file names", len({s for s in up.stems.values() if s}) == 20,
          len(set(up.stems.values())))
    upv = R.read_upstream(z, "voc", voc_dir="PASCAL_VOC")
    check("VOC alone: names and pixel boxes", all(isinstance(b[0], str) for bs in upv.values() for b in bs))
    d = TMP / "yolo_dir" / "folder"
    d.mkdir(parents=True)
    (d / "classes.txt").write_text("alpha\nbeta\n")
    (d / "img1.txt").write_text("1 0.47 0.2 0.06 0.08\n0 0.2 0.59 0.14 0.11\n")
    upy = R.read_upstream(TMP / "yolo_dir", "yolo", classes_file="classes.txt")
    check("YOLO with a classes file, from a directory: ids, names, rounding step",
          list(upy) == ["img1"] and upy.names == {0: "alpha", 1: "beta"} and abs(upy.quantum["img1"] - 0.005) < 1e-12,
          (list(upy), upy.names, upy.quantum))
    check("an unknown kind refuses", raises(lambda: R.read_upstream(z, "coco"), RelationError))
    d2 = TMP / "yolo_dir" / "folder2"
    d2.mkdir(parents=True)
    (d2 / "classes.txt").write_text("alpha\nbeta\n")
    (d2 / "img2.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    upy2 = R.read_upstream(TMP / "yolo_dir", "yolo", classes_file="classes.txt")
    check("two folders with the same classes file read as one table", upy2.names == {0: "alpha", 1: "beta"}
          and set(upy2) == {"img1", "img2"})
    (d2 / "classes.txt").write_text("beta\nalpha\n")
    msg = raises(lambda: R.read_upstream(TMP / "yolo_dir", "yolo", classes_file="classes.txt"), RelationError)
    check("folders whose classes files number the classes differently refuse (one table would misname one of "
          "them)", msg is not None and "disagree" in msg, msg)
    shutil.rmtree(d2)

    print("alignments")
    al = R.alignments(16)
    check("identity, plus1, minus1 and one drop per class", len(al) == 3 + 16 and al["identity"][7] == 7
          and al["plus1"][7] == 8 and al["minus1"][0] is None and al["drop5"][5] is None and al["drop5"][6] == 5
          and al["drop5"][4] == 4)

    print("a planted 'drop class 5' export")
    pool = pool_from(up, al["drop5"], W, H)
    gm = R.geometry_match(pool, up)
    check("every pool image pairs with its upstream file", len(gm["image_pairs"]) == len(pool)
          and all(v == k[5:] for k, v in gm["image_pairs"].items()) and not gm["unmatched"], gm["unmatched"][:3])
    ch = R.choose_alignment(gm["box_pairs"], al, "funnel/v1/h3a/half", n_pool_classes=15)
    ag = {a["name"]: a for a in ch["alignments"]}
    check("the chosen alignment is drop5", ch["chosen"] == "drop5", ch["chosen"])
    check("identity and +-1 score lower on both halves", all(
        ag[n]["agreement_a"] < ag["drop5"]["agreement_a"] and ag[n]["agreement_b"] < ag["drop5"]["agreement_b"]
        for n in ("identity", "plus1", "minus1")), {n: (ag[n]["agreement_a"], ag[n]["agreement_b"])
                                                    for n in ("identity", "plus1", "minus1", "drop5")})
    check("the choice is confirmed on the other half (agreement 1.0)", ch["confirmed"]
          and ag["drop5"]["agreement_b"] == 1.0)
    check("the halves are seeded and recorded", ch["halves"]["seed_text"] == "funnel/v1/h3a/half"
          and ch["halves"]["n_a"] + ch["halves"]["n_b"] == len(pool)
          and R.choose_alignment(gm["box_pairs"], al, "funnel/v1/h3a/half")["halves"] == ch["halves"])
    print("an alignment the second half does not confirm")
    keys = sorted({bp["pool_key"] for bp in gm["box_pairs"]})
    perm = np.random.default_rng(C.stable_int("funnel/v1/h3a/half")).permutation(len(keys))
    half_a = {keys[i] for i in perm[:(len(keys) + 1) // 2]}
    mixed = [dict(bp, pool_id=(al["drop5"] if bp["pool_key"] in half_a else al["identity"])[bp["up_id"]])
             for bp in gm["box_pairs"] if bp["up_id"] != 5]
    ch2 = R.choose_alignment(mixed, al, "funnel/v1/h3a/half")
    check("half a picks drop5, half b prefers identity: the choice is not confirmed",
          ch2["chosen"] == "drop5" and ch2["confirmed"] is False, (ch2["chosen"], ch2["confirmed"]))
    ch3 = R.choose_alignment(gm["box_pairs"], al, "funnel/v1/h3a/half", n_pool_classes=15, min_agreement=1.01)
    check("a confirmed choice below the minimum agreement is not confirmed", ch3["confirmed"] is False)
    print("tolerance")
    ok2 = R.geometry_match(pool_from(up, al["identity"], W, H, shift_px=2.0), up)
    bad5 = R.geometry_match(pool_from(up, al["identity"], W, H, shift_px=5.0), up)
    check("a 2 px shift still matches", len(ok2["image_pairs"]) == len(up), len(ok2["image_pairs"]))
    check("a 5 px shift does not", len(bad5["image_pairs"]) == 0, len(bad5["image_pairs"]))
    yd = TMP / "yolo_round"
    yd.mkdir()
    for stem, boxes in ups.items():
        (yd / (stem.replace(".", "_") + ".txt")).write_text("".join(
            "%d %.2f %.2f %.2f %.2f\n" % (b[0], (b[1] + b[3]) / 2 / W, (b[2] + b[4]) / 2 / H, (b[3] - b[1]) / W,
                                         (b[4] - b[2]) / H) for b in boxes))
    upr = R.read_upstream(yd, "yolo")
    pr = {("pool_" + s.replace(".", "_")): [(b[0], (b[1] + b[3]) / 2 / W, (b[2] + b[4]) / 2 / H, (b[3] - b[1]) / W,
                                             (b[4] - b[2]) / H, W, H) for b in boxes] for s, boxes in ups.items()}
    gr = R.geometry_match(pr, upr)
    check("an upstream rounded to 2 decimals matches within its own rounding step",
          len(gr["image_pairs"]) == len(pr), len(gr["image_pairs"]))


def test_relation_audit():
    print("relation audit (H0b rule) on synthetic copies with a planted old join")
    units = {"c:hold|0": 40, "c:hold|1": 35, "c:small|0": 5}
    scores = {"c:hold|0": {"Waterhemp": 0.91, "Purslane": 0.05}, "c:hold|1": {"Purslane": 0.8, "Waterhemp": 0.1},
              "c:small|0": {"Purslane": 0.9}}
    cur = R.relation_audit(units, scores, {"c:hold|0": "Waterhemp", "c:hold|1": "Purslane", "c:small|0": "Purslane"},
                           {"c:hold|0": "Waterhemp", "c:hold|1": "Purslane"})
    old = R.relation_audit(units, scores, {"c:hold|0": "PalmerAmaranth", "c:hold|1": "Purslane"})
    by = {r["unit"]: r for r in cur}
    check("every unit with enough boxes maps to its truth", by["c:hold|0"]["mapped_to"] == "Waterhemp"
          and by["c:hold|1"]["mapped_to"] == "Purslane")
    check("a unit under min_boxes is not mapped", by["c:small|0"]["mapped_to"] is None)
    check("the current join is not flagged", not any(r["join_flagged"] for r in cur))
    ob = {r["unit"]: r for r in old}
    check("the planted old join is flagged", ob["c:hold|0"]["join_flagged"] and not ob["c:hold|1"]["join_flagged"])
    sc = R.relation_scores({"u": [3, 5]}, np.array([[0, 0], [0, 0], [0, 0], [0.2, 0.8], [0, 0], [0.4, 0.6]]),
                           ["a", "b"])
    check("relation scores are the mean judge probability over the unit's crops", sc == {"u": {"a": 0.3, "b": 0.7}}, sc)

    print("run_relation with a fake adapter")
    fd = TMP / "relfun"
    (fd / "judges").mkdir(parents=True)
    (TMP / "repo" / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT.parent / "docs" / "FUNNEL_AUDIT.md", TMP / "repo" / "docs" / "FUNNEL_AUDIT.md")
    FPR.write_pre_draw(fd / "prereg_v1.json", ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json")
    dom = D.load("weed")
    rc = dom.raw["sources"]["relation_checks"]
    hold, three, keep = rc["h0b_flag_old_join"][0], rc["h0b_map_sources"][1], rc["h0b_keep_current_join"][0]
    labels = dom.target_names
    crop = 0
    P1, P2, u_copy, u_pool = [], [], {}, {}

    def onehot(names_weights, labs):
        v = np.zeros(len(labs))
        for n, wgt in names_weights.items():
            v[labs.index(n)] = wgt
        return v
    for src, sid, truth, cur_join, old_join in ((hold, "0", "Waterhemp", "Waterhemp", "Purslane"),
                                                (hold, "2", "Purslane", "Purslane", "Carpetweed"),
                                                (three, "5", "Ragweed", "Ragweed", "Sicklepod"),
                                                (keep, "1", "Goosegrass", "Goosegrass", "Goosegrass")):
        ids = list(range(crop, crop + 25))
        crop += 25
        for _ in ids:
            P1.append(onehot({truth: 0.9, "Eclipta": 0.1}, labels))
        u_copy["c:%s|%s" % (src, sid)] = {"source": src, "src_id": sid, "boxes": 25, "crop_ids": ids, "truth": truth,
                                          "join": cur_join, "old_join": old_join}
    lab2 = labels + ["other"]
    for sid, status, join, dist in (("rel", "target_related", "OtherPlant", {"Waterhemp": 0.8, "other": 0.2}),
                                    ("tg", "target", "Ragweed", {"Ragweed": 0.9, "other": 0.1}),
                                    ("oth", "taxon_resolved", "OtherPlant", {"other": 0.95, "Ragweed": 0.05})):
        ids = list(range(crop, crop + 30))
        crop += 30
        u_pool["c:%s|%s" % (rc["h0c_source"], sid)] = {"source": rc["h0c_source"], "src_id": sid, "boxes": 30,
                                                        "crop_ids": ids, "join": join, "status_v2": status}
    ids = list(range(crop, crop + 30))
    u_pool["c:noname|*"] = {"source": "noname", "src_id": "*", "boxes": 30, "crop_ids": ids, "join": "OtherPlant",
                            "status_v2": "no_name"}
    crop += 30
    P2 = np.zeros((crop, len(lab2)))
    for u in u_pool.values():
        dist = {"c:%s|rel" % rc["h0c_source"]: {"Waterhemp": 0.8, "other": 0.2},
                "c:%s|tg" % rc["h0c_source"]: {"Ragweed": 0.9, "other": 0.1},
                "c:%s|oth" % rc["h0c_source"]: {"other": 0.95, "Ragweed": 0.05},
                "c:noname|*": {"Sicklepod": 0.7, "other": 0.3}}["c:%s|%s" % (u["source"], u["src_id"])]
        for c in u["crop_ids"]:
            P2[c] = onehot(dist, lab2)
    P1 = np.array(P1)

    def save(name, P, labs):
        np.savez(fd / "judges" / name, unit_index=np.arange(len(P), dtype=np.int64), P=P.astype(np.float16),
                 top=P.argmax(1).astype(np.int16), meta=np.array(json.dumps({"labels": labs})))
    save("J-knn1__crops.npz", P1, labels)
    save("J-knn2__crops.npz", P2, lab2)
    (fd / "judge_qualification.json").write_text(json.dumps({"judges": {
        "J-knn2": {"by_type": {"other_noinfo": {"qualified": True, "precision_at_half": {"lb": 0.9}}}},
        "J-zs": {"by_type": {"other_noinfo": {"qualified": False, "precision_at_half": {"lb": 0.95}}}}}}))

    class FakeAdapter(object):
        def relation_units(self, domain, sources):
            return {u: v for u, v in u_copy.items() if v["source"] in sources}

        def pool_class_units(self, domain, funnel_dir):
            return u_pool
    doc = R.run_relation(fd / "prereg_v1.json", "weed", fd, FakeAdapter(), testing=True)
    check("relation_audit_v1.json is written with its header", (fd / "relation_audit_v1.json").exists()
          and doc["format"] == "funnel-relation-audit/1" and doc["judge"] == "J-knn1")
    check("H0(b) passes: maps every class, flags the old joins, keeps the current join",
          doc["h0b"]["pass"] is True and all(c["pass"] for c in doc["h0b"]["checks"]), doc["h0b"]["checks"])
    check("H0(c) uses the best qualified judge for no-information strata", doc["h0c"]["judge"] == "J-knn2")
    check("relation_audit_v1.json records the sha256 of both judges' score files",
          doc["inputs"]["h0b_scores"]["sha256"] == C.sha256_file(fd / "judges" / "J-knn1__crops.npz")
          and doc["inputs"]["h0c_scores"]["sha256"] == C.sha256_file(fd / "judges" / "J-knn2__crops.npz"),
          sorted(doc["inputs"]))
    check("H0(c): a named relative mapped to a target is reported and fails the check",
          doc["h0c"]["maps_relative_to_target"] == ["c:%s|rel" % rc["h0c_source"]] and doc["h0c"]["pass"] is False,
          doc["h0c"])
    check("H0(c): the class accuracy is recorded", doc["h0c"]["class_accuracy"] == round(2 / 3.0, 6),
          doc["h0c"]["class_accuracy"])
    check("a no-name class the judge calls a target becomes a person-queue proposal",
          [p["unit"] for p in doc["visual_proposals"]] == ["c:noname|*"]
          and doc["visual_proposals"][0]["status"] == "proposal_L14")
    good_ids = list(u_copy["c:%s|0" % hold]["crop_ids"])
    u_copy["c:%s|0" % hold]["crop_ids"] = u_copy["c:%s|2" % hold]["crop_ids"]
    doc2 = R.run_relation(fd / "prereg_v1.json", "weed", fd, FakeAdapter(), testing=True)
    check("a unit that maps to the wrong species fails H0(b)", doc2["h0b"]["pass"] is False)
    u_copy["c:%s|0" % hold]["crop_ids"] = good_ids

    print("H0(b) and H0(c) fail closed: an untested check is not a pass")
    for gone, check_name in (("c:%s|1" % keep, "does not flag the current join"),
                             ("c:%s|5" % three, "maps every class to its twin's species")):
        saved = u_copy.pop(gone)
        d3 = R.run_relation(fd / "prereg_v1.json", "weed", fd, FakeAdapter(), testing=True)
        ck = {c["name"]: c for c in d3["h0b"]["checks"]}
        check("with no unit of %s to judge, H0(b) fails and names the untested source" % gone.split("|")[0][2:],
              d3["h0b"]["pass"] is False and ck[check_name]["pass"] is False
              and ck[check_name]["untested_sources"] == [saved["source"]], ck[check_name])
        u_copy[gone] = saved
    d4 = R.run_relation(fd / "prereg_v1.json", "weed", fd, FakeAdapter(), testing=True)
    check("with every unit back, H0(b) passes again", d4["h0b"]["pass"] is True, d4["h0b"]["checks"])
    oth = "c:%s|oth" % rc["h0c_source"]
    P2b = P2.copy()
    for c in u_pool[oth]["crop_ids"]:
        P2b[c] = onehot({"Goosegrass": 0.9, "other": 0.1}, lab2)
    save("J-knn2__crops.npz", P2b, lab2)
    d5 = R.run_relation(fd / "prereg_v1.json", "weed", fd, FakeAdapter(), testing=True)
    check("H0(c): a taxon-resolved named class (a crabgrass) mapped to a target fails the check, not only a "
          "same-genus relative", oth in d5["h0c"]["maps_relative_to_target"] and d5["h0c"]["pass"] is False,
          d5["h0c"]["maps_relative_to_target"])
    save("J-knn2__crops.npz", P2, lab2)
    keep_pool = dict(u_pool)
    u_pool.clear()
    u_pool["c:%s|nn" % rc["h0c_source"]] = dict(keep_pool["c:noname|*"], source=rc["h0c_source"], src_id="nn")
    d6 = R.run_relation(fd / "prereg_v1.json", "weed", fd, FakeAdapter(), testing=True)
    check("H0(c) on a source with no named class is untested: not a pass, no accuracy",
          d6["h0c"]["pass"] is False and d6["h0c"]["class_accuracy"] is None
          and d6["h0c"]["units_without_name_truth"] == ["c:%s|nn" % rc["h0c_source"]], d6["h0c"])
    u_pool.clear()
    u_pool.update(keep_pool)
    os.unlink(fd / "judge_qualification.json")
    check("run_relation refuses without judge_qualification.json",
          raises(lambda: R.run_relation(fd / "prereg_v1.json", "weed", fd, FakeAdapter()), RelationError))


def test_card_proposals():
    print("card proposals")
    dom = D.load("weed")
    slug = "project_agml__mh_weed16_weed_detection"
    ps = {"per_slug": {slug: {"join": {str(i): [str(i), "OtherPlant"] for i in range(15)}}}}
    table = dom.raw["sources"]["card_resolvers"][slug]["class_table"]
    up_names = {i: table[str(i)]["name"] for i in range(15)}
    geo = {"matches": {slug: {"mode": "ids", "chosen": "drop15", "confirmed": True, "agreement_b": 1.0,
                              "card_path": "x", "card_sha256": "0" * 64,
                              "card_names": R.card_name_check(table, up_names)}}}
    check("the card check: every upstream name equals the card table's under the same id",
          geo["matches"][slug]["card_names"]["checked"] and not geo["matches"][slug]["card_names"]["disagree"])
    props = [p for p in R.card_proposals(dom, geo, ps) if p["source"] == slug]
    by = {p["src_id"]: p for p in props}
    check("with 15 classes kept by drop15, the card maps id 12 to Sicklepod and id 5 to MorningGlory",
          by["12"]["map_to"] == "Sicklepod" and by["5"]["map_to"] == "MorningGlory" and by["12"]["status"] == "proposed",
          (by["12"], by["5"]))
    check("the other ids map to no target", all(by[k]["map_to"] is None for k in by if k not in ("5", "12")))
    geo["matches"][slug]["chosen"] = "identity"
    props = [p for p in R.card_proposals(dom, geo, ps) if p["source"] == slug]
    check("a class count that disagrees with the card (16 kept, 15 in the source) goes to L14",
          props and all(p["status"] == "to_L14" and "16" in p["reason"] for p in props), props[:1])
    geo["matches"][slug].update(chosen="drop15", confirmed=False)
    props = [p for p in R.card_proposals(dom, geo, ps) if p["source"] == slug]
    check("an unconfirmed alignment goes to L14", all(p["status"] == "to_L14" for p in props))
    shifted = {i: table[str(i + 1)]["name"] for i in range(15)}        # upstream ids one off the card's
    geo["matches"][slug].update(confirmed=True, card_names=R.card_name_check(table, shifted))
    props = [p for p in R.card_proposals(dom, geo, ps) if p["source"] == slug]
    check("a card whose names disagree with the upstream's under the same ids goes to L14 (no map is read "
          "from it)", props and all(p["status"] == "to_L14" and "disagree" in p["reason"] for p in props),
          props[:1])
    geo["matches"][slug]["card_names"] = R.card_name_check(table, {})
    props = [p for p in R.card_proposals(dom, geo, ps) if p["source"] == slug]
    check("an upstream without class names cannot be compared with the card: L14",
          props and all(p["status"] == "to_L14" for p in props), props[:1])
    geo["matches"][slug].pop("card_names")
    props = [p for p in R.card_proposals(dom, geo, ps) if p["source"] == slug]
    check("a geometry record without the card check proposes nothing", all(p["status"] == "to_L14" for p in props))


def test_world():
    try:
        w = FW.build_world(TMP)
    except FW.WorldUnavailable as e:
        print("SKIP: %s" % e)
        SKIPS.append("world (%s)" % e)
        return
    from weed_optimizer_framework.tools.funnel.adapters import inc_step1 as A
    fd = w.funnel_dir
    ps = json.load(open(w.step1 / "pool_summary.json"))
    names = sorted({v[0] for st in ps["per_slug"].values() for v in st["join"].values() if v[0]})
    tax = FW.build_taxonomy_cache(w, names)
    A.census(fd / "prereg_v1.json", "weed", fd, tax, testing=True)
    meta = [json.loads(ln) for ln in open(w.step1 / "pool_meta.jsonl")]
    cards = {}
    # MH-Weed16: VOC names + YOLO ids = the pool's ids (the AgML export kept ids 0-14)
    z = fd / "cards" / FW.MH / "annotations.zip"
    z.parent.mkdir(parents=True)
    mh_table = D.load("weed").raw["sources"]["card_resolvers"][FW.MH]["class_table"]
    with zipfile.ZipFile(z, "w") as zf:
        for m in (m for m in meta if m["source"] == FW.MH):
            stem = m["key"].split("__")[-1]
            W, H = m["W"], m["H"]
            objs, lines = [], []
            for b, (_c, cx, cy, bw, bh) in enumerate(m["boxes"]):
                sid = int(m["src"][b][0])
                objs.append((mh_table[str(sid)]["name"], (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W,
                             (cy + bh / 2) * H))
                lines.append("%d %.6f %.6f %.6f %.6f\n" % (sid, cx, cy, bw, bh))
            zf.writestr("a/PASCAL_VOC/%s.xml" % stem, voc("VID_1.mp4_%d.png" % len(objs), W, H, objs))
            zf.writestr("a/YOLO_darknet/%s.txt" % stem, "".join(lines))
    cards[FW.MH] = [{"what": "annotations", "file": "%s/annotations.zip" % FW.MH, "sha256": C.sha256_file(z)}]
    # the two authoritative sources: YOLO label files and classes.txt per folder
    for slug, names_order in ((FW.WEEDCROP, FW.SOURCE_NAMES[FW.WEEDCROP]),
                              (FW.GREENHOUSE, FW.SOURCE_NAMES[FW.GREENHOUSE][1:] + FW.SOURCE_NAMES[FW.GREENHOUSE][:1])):
        ents = []
        base = fd / "cards" / slug / "annotations" / "Folder_A"
        base.mkdir(parents=True)
        (base / "classes.txt").write_text("\n".join(names_order) + "\n")
        ents.append({"what": "annotations", "file": "%s/annotations/Folder_A/classes.txt" % slug,
                     "sha256": C.sha256_file(base / "classes.txt")})
        for m in (m for m in meta if m["source"] == slug):
            stem = m["key"].split("__")[-1]
            p = base / (stem + ".txt")
            p.write_text("".join("%d %.6f %.6f %.6f %.6f\n" % (int(m["src"][b][0]), cx, cy, bw, bh)
                                 for b, (_c, cx, cy, bw, bh) in enumerate(m["boxes"])))
            ents.append({"what": "annotations", "file": "%s/annotations/Folder_A/%s.txt" % (slug, stem),
                         "sha256": C.sha256_file(p)})
        cards[slug] = ents
    (fd / "cards" / "index.json").write_text(json.dumps({"format": "funnel-cards/1", "cards": cards}))
    geo = R.run_geometry(fd / "prereg_v1.json", "weed", fd, A, testing=True)
    mh = geo["matches"][FW.MH]
    check("MH: every pool box matched, ids mode", mh["matched_share"] == 1.0 and mh["mode"] == "ids", mh)
    check("MH: ids 13-15 are unseen, so the tie-break keeps 15 classes (as the source has) and moves no id: "
          "drop15, confirmed",
          mh["chosen"] == "drop15" and mh["confirmed"], (mh["chosen"], mh["confirmed"]))
    check("H3a's exact part passes", geo["h3a_exact"]["pass"] is True, geo["h3a_exact"])
    check("capture stems are counted", mh["stems"] and mh["stems"]["distinct"] == 1)
    check("H1-pre passes where the card's class order is the pool's", geo["h1_pre"][FW.WEEDCROP]["pass"] is True,
          geo["h1_pre"][FW.WEEDCROP])
    gh = geo["matches"][FW.GREENHOUSE]["names"]
    check("H1-pre fails for an off-by-one class order, and the pairs name the swap",
          geo["h1_pre"][FW.GREENHOUSE]["pass"] is False and any(" -> " in k and k.split(" -> ")[0] != k.split(" -> ")[1]
                                                                  for k in gh["pairs"]), gh)
    cm = read_json(fd / "class_maps.json")
    mhp = {p["src_id"]: p for p in cm["proposals"] if p["source"] == FW.MH}
    check("class_maps.json: MH id 12 -> Sicklepod by card+geometry, proposed",
          mhp["12"]["map_to"] == "Sicklepod" and mhp["12"]["via"] == "card+geometry" and mhp["12"]["status"] == "proposed",
          mhp.get("12"))
    check("the card check found the upstream names equal to the card table's", mh["card_names"]["checked"]
          and not mh["card_names"]["disagree"], mh["card_names"])
    geo_bytes = (fd / "relation_geometry_v1.json").read_bytes()
    cm_bytes = (fd / "class_maps.json").read_bytes()
    time.sleep(1.1)                     # a rewrite would carry another built_utc (second resolution)
    R.run_geometry(fd / "prereg_v1.json", "weed", fd, A, testing=True)
    check("a rerun with the same inputs is a no-op: neither file is rewritten (class_maps.json's record of the "
          "geometry file stays current)", (fd / "relation_geometry_v1.json").read_bytes() == geo_bytes
          and (fd / "class_maps.json").read_bytes() == cm_bytes)
    cmi = read_json(fd / "class_maps.json")["inputs"]
    check("class_maps.json records the name status and taxonomy cache its taxonomy proposals came from",
          cmi["name_status_v2"]["sha256"] == C.sha256_file(fd / "name_status_v2.json") and "taxonomy_cache" in cmi,
          sorted(cmi))
    rec = cmi["relation_geometry"]
    check("... and class_maps.json's recorded geometry sha256 is the file's",
          rec["sha256"] == C.sha256_file(fd / "relation_geometry_v1.json"), rec)
    cm2 = dict(cm)
    cm2["proposals"] = cm["proposals"][1:]
    (fd / "class_maps.json").write_text(json.dumps(cm2))
    msg = raises(lambda: R.run_geometry(fd / "prereg_v1.json", "weed", fd, A, testing=True), RelationError)
    check("class_maps.json is immutable: other proposals refuse without --force", msg is not None and "immutable" in msg,
          msg)
    check("the refused run wrote nothing: relation_geometry_v1.json is unchanged",
          (fd / "relation_geometry_v1.json").read_bytes() == geo_bytes)
    nsp = fd / "name_status_v2.json"
    os.rename(nsp, fd / "ns.bak")
    msg = raises(lambda: R.run_geometry(fd / "prereg_v1.json", "weed", fd, A, force=True, testing=True),
                 RelationError)
    check("run_geometry refuses before the census (no name_status_v2.json): the taxonomy proposals would be "
          "missing from an immutable file", msg is not None and "census" in msg, msg)
    os.rename(fd / "ns.bak", nsp)
    R.run_geometry(fd / "prereg_v1.json", "weed", fd, A, force=True, testing=True)
    kt = A.known_truth("weed", fd)
    hid = "b:%s#%d" % (w.planted["hidden_target"]["key"], w.planted["hidden_target"]["box"])
    kt6 = {i["id"]: i for i in kt.get("KT6", [])}
    check("once H3a passes, KT6 holds the card-resolved id-12 boxes as Sicklepod", hid in kt6
          and kt6[hid]["truth"] == C.CLASS_NAMES.index("Sicklepod") and kt6[hid]["claimed"], kt6.get(hid))
    check("KT6 items are split into calibration and estimation halves by near-duplicate group",
          {i["role"] for i in kt6.values()} <= {"calibration", "estimation"} and kt6)
    gp = fd / "relation_geometry_v1.json"
    keep_geo = gp.read_bytes()
    g2 = json.loads(keep_geo)
    g2["matches"][FW.MH]["card_names"]["disagree"] = [{"up_id": 12, "upstream": "Harali", "card": "Sicklepod"}]
    gp.write_text(json.dumps(g2))
    A._KEYS_CACHE.clear()
    check("a card whose names disagree with the upstream's gives no KT6 truth, even with H3a's exact part passed",
          A.known_truth("weed", fd).get("KT6") == [])
    gp.write_bytes(keep_geo)
    os.unlink(fd / "cards" / "index.json")
    check("run_geometry refuses without the fetched cards (lever L11a)",
          raises(lambda: R.run_geometry(fd / "prereg_v1.json", "weed", fd, A), RelationError))


if __name__ == "__main__":
    try:
        test_upstream_and_alignment()
        test_relation_audit()
        test_card_proposals()
        test_world()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)
