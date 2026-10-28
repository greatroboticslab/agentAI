#!/usr/bin/env python3
"""Intake refuses copies of every exam (collect.intake, amendment 2026-10-28).

2SeasonWeedDet8 (zenodo_10762138) comes from the lab of CottonWeedDet12 and of
3SeasonWeedDet10: its 2021 part is derived from CottonWeedDet12 (whose dev and
test are never trained on), and its 2022 part holds every image of ood22, the
season hold-out exam of the registered V3F family, as byte copies (1,915 of
its 1,930 images), and 13 images of ood23 (docs/CONTINUOUS_LOOP.md, amendment
2026-10-28). GuardV2 holds
the v2 evaluation splits (dev, test, ImageWeeds); ood22/ood23 are not among
them (under splits v2 their images are base_v2's tsw22/tsw23 rows, so most
copies of them were refused as base_copy, but not the ood23 images base_v2
does not hold), and the funnel's H10d hold-out was not checked at intake.

Pinned, through collect.intake over the real inc2.guard.GuardV2 and a
synthetic LOCK v2 that names a LOCK v1 (derived_from), every image a
re-encoded or flipped copy:
  * a copy of a cwd12 test image is refused near_eval_* (GuardV2), of a dev
    image likewise;
  * a copy of an ood22 image and of an ood23 image that base_v2 does not hold
    are refused near_exam_season, named by exam and key;
  * a copy of an ood23 image that base_v2 holds stays base_copy and carries
    the exam it copies (match.exam);
  * a copy of an H10d domain-dev image (step1_r1's complete recovery record)
    is refused near_h10d;
  * one photograph in both season exams (as ood22__20221008_iPhoneSE_YL_2
    and ood23__20230810_HTRC_iPhone13_BD_29 are): a copy of it is named by
    the first and carries the other (match.also), and counts for both exams;
  * a clean picture is kept; guard.json and summary.json count the copies of
    each exam (exam_copies), and D28's base-copy share counts the new reasons;
  * an image whose EXIF orientation is not the identity is refused
    exif_orientation after every guard (its boxes would be drawn in a frame
    the trainer does not see), a copy with one is still refused as a copy,
    an orientation of 1 is kept; it counts among the images checked;
  * the legacy-label rule on 2SeasonWeedDet8's own 25 class names (9 of them
    the reference dataset's legacy labels) closes such a source at intake
    (copy_candidate), before any image is judged; the config's never-fetch
    entry for 2SeasonWeedDet8 (Zenodo 10762138, or its title under any
    record) closes it never_train before its files are read;
  * fail closed (GuardUnavailable): a production LOCK v2 naming no v1 LOCK; a
    v1 LOCK, v1 never-train index or exam manifest that does not hash as
    recorded; an index missing an exam image; a complete recovery record
    whose domain dev does not hash as recorded or whose image cannot be read.
    No recovery record (F9 not run) checks no H10d and says so.

Run:  python3 tests/test_collect_intake_exams.py
"""
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_intake as TI  # noqa: E402  (W.setup ran on import: TMP, INC_DIR)

W = TI.W
TMP = TI.TMP
check, raises = W.check, W.raises


def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def make_lock_v1(d, exams, extra=None):
    """splits/v1: the ood22/ood23 manifests, the v1 never-train index (every split's images, complete, 6 bits) and
    LOCK.json recording their sha256s."""
    from weed_optimizer_framework.tools.inc import common as C
    d.mkdir(parents=True, exist_ok=True)
    man, nt = {}, []
    for split, imgs in sorted(dict(exams, **(extra or {})).items()):
        rows = [{"key": "%s__%d" % (split, i), "image": str(p), "label": str(p) + ".txt", "sha256": C.sha256_file(p),
                 "label_sha256": "0" * 64, "source": split, "session": ""} for i, p in enumerate(imgs)]
        (d / ("%s.jsonl" % split)).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        man[split] = C.sha256_file(d / ("%s.jsonl" % split))
        nt += [[int(C.dhash(r["image"])), split, r["key"]] for r in rows]
    (d / "nevertrain_dhash.json").write_text(json.dumps({"entries": nt, "complete": True, "bits": 6,
                                                         "min_expected": len(nt)}))
    (d / "LOCK.json").write_text(json.dumps({"splits_version": "v1", "manifests": man,
                                             "nevertrain_sha256": C.sha256_file(d / "nevertrain_dhash.json")}))
    return d / "LOCK.json"


def link_v1(lock2, lock1, testing=True):
    from weed_optimizer_framework.tools.inc import common as C
    lk = json.loads(lock2.read_text())
    lk["derived_from"] = {"v1_lock_path": str(lock1), "v1_lock_sha256": C.sha256_file(lock1)}
    lk["testing"] = testing
    lock2.write_text(json.dumps(lk))


def make_h10d(inc, imgs):
    """step1_r1 as funnel.recover writes it: domain_dev.jsonl, domain_dev.json (its rows record) and a complete
    recovery.json whose domain_dev record names domain_dev.json."""
    from weed_optimizer_framework.tools.inc import common as C
    r1 = pathlib.Path(inc) / "step1_r1"
    r1.mkdir(parents=True, exist_ok=True)
    rows = [{"key": "dd_%d" % i, "image": str(p), "label": str(p) + ".txt", "sha256": C.sha256_file(p),
             "label_sha256": "0" * 64, "source": "rf_dd", "session": "g", "group": "g0"} for i, p in enumerate(imgs)]
    (r1 / "domain_dev.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    rsha = C.sha256_file(r1 / "domain_dev.jsonl")
    (r1 / "domain_dev.json").write_text(json.dumps({"rows": {"path": str(r1 / "domain_dev.jsonl"), "sha256": rsha,
                                                             "images": len(rows)}}))
    (r1 / "recovery.json").write_text(json.dumps({"format": "funnel-recovery/1", "status": "complete",
                                                  "domain_dev": {"path": str(r1 / "domain_dev.json"),
                                                                 "sha256": C.sha256_file(r1 / "domain_dev.json"),
                                                                 "rows_sha256": rsha}}))
    return r1


def test_exams(cfg):
    from PIL import Image
    from weed_optimizer_framework.tools.collect import GuardUnavailable
    from weed_optimizer_framework.tools.collect import intake as I
    print("intake refuses copies of the season exams, the H10d hold-out and the v2 evaluation splits")
    g = TMP / "ex"
    dev = W.grid_img(g / "dev.png", 920, fmt="PNG")
    test = W.grid_img(g / "test.png", 921, fmt="PNG")
    iw = W.grid_img(g / "iw.png", 922, fmt="PNG")
    o22 = W.grid_img(g / "o22.png", 923, fmt="PNG")
    o23a = W.grid_img(g / "o23a.png", 924, fmt="PNG")          # also a base_v2 row (tsw23)
    o23b = W.grid_img(g / "o23b.png", 925, fmt="PNG")          # an ood23 image base_v2 does not hold
    dd = W.grid_img(g / "dd.png", 926, fmt="PNG")
    sh = W.grid_img(g / "sh22.png", 928, fmt="PNG")            # one photograph in both season exams
    sh2 = g / "sh23.png"
    sh2.write_bytes(sh.read_bytes())
    lock2 = TI.make_lock_v2({"dev": [dev], "test": [test], "imageweeds": [iw]}, [o23a])
    lock1 = make_lock_v1(TMP / "inc" / "splits" / "v1", {"ood22": [o22, sh], "ood23": [o23a, o23b, sh2]},
                         extra={"dev": [dev], "test": [test], "imageweeds": [iw]})
    link_v1(lock2, lock1)
    import os
    inc = pathlib.Path(os.environ["INC_DIR"])
    _idx, rec = I.load_exam_guard(lock2, inc)
    check("no recovery record (F9 not run): the H10d hold-out is not checked, and says so; the season exams are",
          rec["season"]["checked"] and rec["season"]["images"] == {"ood22": 2, "ood23": 3}
          and rec["h10d"]["checked"] is False and "F9 has not written" in rec["h10d"]["why"], rec)
    make_h10d(inc, [dd])

    def copy_of(p, how):
        with Image.open(p) as im:
            if how == "flip":
                return png_bytes(im.transpose(Image.FLIP_LEFT_RIGHT))
            if how == "rot":
                return png_bytes(im.transpose(Image.ROTATE_90))
            return png_bytes(im.convert("RGB"))                  # re-encoded
    def jpeg(p, orientation):
        with Image.open(p) as im:
            ex = Image.Exif()
            ex[I.EXIF_ORIENTATION_TAG] = orientation
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=95, exif=ex.tobytes())
            return buf.getvalue()
    imgs = {"a_test.png": copy_of(test, "reenc"), "b_dev.png": copy_of(dev, "flip"),
            "c_ood22.png": copy_of(o22, "flip"), "d_ood23b.png": copy_of(o23b, "reenc"),
            "e_ood23a.png": copy_of(o23a, "rot"), "f_h10d.png": copy_of(dd, "flip"), "g_clean.png": W.img_bytes(927),
            "h_shared.png": copy_of(sh, "reenc"),
            "i_exif6.jpg": jpeg(W.grid_img(g / "x6.png", 929, fmt="PNG"), 6),
            "j_exif1.jpg": jpeg(W.grid_img(g / "x1.png", 930, fmt="PNG"), 1),
            "k_exif8_test.jpg": jpeg(test, 8)}
    cats = [{"id": 1, "name": "weed: amaranthus palmeri"}]
    anns = [{"id": i, "image_id": i, "category_id": 1, "bbox": [10, 20, 40, 40]} for i in range(len(imgs))]
    sid = TI.fetch_simple(cfg, "eeeeeeee-0000-0000-0000-000000000031", TI.coco_files(imgs, cats, anns))
    r = I.intake(cfg, sid, lock_path=lock2)
    bdir = pathlib.Path(r["dir"])
    dec = {d["rel"].split("/")[-1]: d for d in (json.loads(x) for x in (bdir / "decisions.jsonl").read_text()
                                                .splitlines()) if d.get("kind") == "image"}
    why = {k: v["reason"] for k, v in dec.items()}
    check("a copy of a cwd12 test image and of a dev image: refused by GuardV2 (near_eval_*)",
          why.get("a_test.png") in ("near_eval_v2", "near_eval_variant")
          and why.get("b_dev.png") in ("near_eval_v2", "near_eval_variant"), why)
    m22, m23 = (dec.get("c_ood22.png") or {}).get("match") or {}, (dec.get("d_ood23b.png") or {}).get("match") or {}
    check("a copy of an ood22 image and of an ood23 image base_v2 does not hold: near_exam_season, named",
          why.get("c_ood22.png") == "near_exam_season" and why.get("d_ood23b.png") == "near_exam_season"
          and m22.get("split") == "ood22" and m22.get("key") == "ood22__0" and m23.get("split") == "ood23"
          and m23.get("key") == "ood23__1", (why, m22, m23))
    ma = (dec.get("e_ood23a.png") or {}).get("match") or {}
    check("a copy of an ood23 image base_v2 holds: base_copy, carrying the exam it copies",
          why.get("e_ood23a.png") == "base_copy" and (ma.get("exam") or {}).get("split") == "ood23", (why, ma))
    md = (dec.get("f_h10d.png") or {}).get("match") or {}
    check("a copy of an H10d domain-dev image: near_h10d", why.get("f_h10d.png") == "near_h10d"
          and md.get("split") == "h10d" and md.get("key") == "dd_0", (why, md))
    mh = (dec.get("h_shared.png") or {}).get("match") or {}
    check("a copy of the photograph both season exams hold: named by ood22's image, carrying ood23's (also)",
          why.get("h_shared.png") == "near_exam_season" and mh.get("split") == "ood22" and mh.get("key") == "ood22__1"
          and mh.get("also") == [{"split": "ood23", "key": "ood23__2"}], (why.get("h_shared.png"), mh))
    mx = (dec.get("i_exif6.jpg") or {}).get("match") or {}
    check("a clean picture with EXIF orientation 6: refused exif_orientation (orientation recorded); one with 1 kept; "
          "a copy of a test image with 8 still refused as a copy",
          why.get("i_exif6.jpg") == "exif_orientation" and mx.get("orientation") == 6
          and why.get("j_exif1.jpg") == "kept"
          and why.get("k_exif8_test.jpg") in ("near_eval_v2", "near_eval_variant"), (why, mx))
    check("the clean pictures are kept; no copied or EXIF-rotated image is in the manifest",
          why.get("g_clean.png") == "kept" and r["rows"] == 2, (why, r["rows"]))
    gd = json.loads((bdir / "guard.json").read_text())
    sm = json.loads((bdir / "summary.json").read_text())
    check("guard.json and summary.json count the copies of each exam, whatever refused them (the shared "
          "photograph for both)",
          gd.get("exam_copies") == {"h10d": 1, "ood22": 2, "ood23": 3} and sm.get("exam_copies") == gd["exam_copies"]
          and (gd.get("exam_guard") or {}).get("season", {}).get("checked") is True
          and (gd.get("exam_guard") or {}).get("h10d", {}).get("checked") is True, (gd.get("exam_copies"),
                                                                                     gd.get("exam_guard")))
    g_ = sm.get("guard") or {}
    check("summary.json's guard counts the new reasons, and D28's base-copy share counts them with base_copy; the "
          "EXIF-refused image is among the images checked",
          g_.get("near_exam_season") == 3 and g_.get("near_h10d") == 1 and g_.get("base_copy") == 1
          and abs(sm["source_leak"]["base_share"] - round(5 / 11.0, 4)) < 1e-9 and sm.get("images") == 11
          and (sm.get("decisions") or {}).get("by_reason", {}).get("exif_orientation") == 1,
          (g_, sm.get("source_leak"), sm.get("images"), (sm.get("decisions") or {}).get("by_reason")))

    print("fail closed")
    from weed_optimizer_framework.tools.inc import common as C
    lk = json.loads(lock2.read_text())
    prod = lock2.parent / "LOCK_prod.json"
    prod.write_text(json.dumps(dict({k: v for k, v in lk.items() if k != "derived_from"}, testing=False)))
    e = raises(lambda: I.load_exam_guard(prod, inc), GuardUnavailable)
    check("a production LOCK v2 naming no v1 LOCK refuses", e is not None and "names no v1 LOCK" in str(e), e)
    for what, path, edit, msg in (
            ("LOCK v1", lock1, lambda b: b.replace(b'"v1"', b'"v1 "'), "LOCK v1"),
            ("the v1 never-train index", lock1.parent / "nevertrain_dhash.json", lambda b: b + b"\n", "v1 never-train"),
            ("the ood23 manifest", lock1.parent / "ood23.jsonl", lambda b: b + b"\n", "ood23 manifest")):
        keep = path.read_bytes()
        path.write_bytes(edit(keep))
        e = raises(lambda: I.load_exam_guard(lock2, inc), GuardUnavailable)
        path.write_bytes(keep)
        check("%s that no longer hashes as recorded refuses" % what, e is not None and msg in str(e), e)
    # an index that misses an exam image (recorded consistently: the LOCKs re-signed over it)
    ntp = lock1.parent / "nevertrain_dhash.json"
    keep_nt, keep_l1, keep_l2 = ntp.read_bytes(), lock1.read_bytes(), lock2.read_bytes()
    nt = json.loads(keep_nt)
    nt["entries"] = [e_ for e_ in nt["entries"] if e_[2] != "ood23__1"]
    ntp.write_text(json.dumps(nt))
    l1 = json.loads(keep_l1)
    l1["nevertrain_sha256"] = C.sha256_file(ntp)
    lock1.write_text(json.dumps(l1))
    link_v1(lock2, lock1)
    e = raises(lambda: I.load_exam_guard(lock2, inc), GuardUnavailable)
    check("a v1 index that misses an ood23 image refuses (its copies would be cleared)",
          e is not None and "does not hold exactly" in str(e) and "ood23__1" in str(e), e)
    ntp.write_bytes(keep_nt)
    lock1.write_bytes(keep_l1)
    lock2.write_bytes(keep_l2)
    ddj = inc / "step1_r1" / "domain_dev.jsonl"
    keep_dd = ddj.read_bytes()
    ddj.write_bytes(keep_dd + b"\n")
    e = raises(lambda: I.load_exam_guard(lock2, inc), GuardUnavailable)
    ddj.write_bytes(keep_dd)
    check("a complete recovery record whose domain dev does not hash as recorded refuses",
          e is not None and "H10d" in str(e), e)
    dd.rename(dd.with_suffix(".gone"))
    e = raises(lambda: I.load_exam_guard(lock2, inc), GuardUnavailable)
    dd.with_suffix(".gone").rename(dd)
    check("a domain-dev image that cannot be read refuses", e is not None and "cannot be hashed" in str(e), e)
    rj = inc / "step1_r1" / "recovery.json"
    keep_rj = rj.read_bytes()
    rj.write_text(json.dumps(dict(json.loads(keep_rj), status="refused")))
    _i, rec2 = I.load_exam_guard(lock2, inc)
    rj.write_bytes(keep_rj)
    check("a recovery record that is not complete holds no hold-out (recorded, not checked)",
          rec2["h10d"]["checked"] is False and "not complete" in rec2["h10d"]["why"], rec2["h10d"])
    e = raises(lambda: I.intake(cfg, "weedai_none", lock_path=prod), Exception)
    check("(the intake itself refuses under that production LOCK before writing anything)", e is not None, e)


def via_files_2swd8(names, seed0):
    """One image and one VIA file per class name, in 2SeasonWeedDet8's own layout (a per-image file whose regions
    are a columnar object, the class a Weed checkbox with one name checked)."""
    files = {}
    for i, n in enumerate(names):
        stem = "Year2022/img_%02d" % i
        data = W.img_bytes(seed0 + i)
        files[stem + ".png"] = data
        key = "img_%02d.png%d" % (i, len(data))
        files[stem + ".json"] = json.dumps({key: {
            "filename": "img_%02d.png" % i, "size": len(data), "file_attributes": {},
            "regions": {"shape_attributes": {"name": "rect", "x": 10, "y": 12, "width": 40, "height": 30},
                        "region_attributes": {"Weed": {n: True}}}}}).encode()
    return files


def test_same_lab_repackage(cfg):
    """2SeasonWeedDet8 at intake, as the re-intake of 2026-10-28 meets it."""
    from weed_optimizer_framework.tools.collect import Refusal, state as S
    from weed_optimizer_framework.tools.collect import intake as I
    from weed_optimizer_framework.tools.collect import batches_ledger
    print("2SeasonWeedDet8's class list and its never-fetch entry")
    doc = json.loads((pathlib.Path(__file__).resolve().parent / "fixtures" / "via_2swd8" / "FIXTURE.json")
                     .read_text())
    names = sorted(doc["survey_of_all_files"]["class_names"])
    legacy = sorted(set(names) & set(cfg.raw["prefilter"]["legacy_labels"]))
    check("fixture: the survey of all 6,664 files lists 25 class names, 9 of them legacy labels of the reference "
          "dataset", len(names) == 25 and len(legacy) == 9, (len(names), legacy))
    led = batches_ledger(None)
    before = led.read_text() if led.exists() else ""
    sid = TI.fetch_simple(cfg, "2a5e0000-0000-0000-0000-000000010762", via_files_2swd8(names, 940))
    e = raises(lambda: I.intake(cfg, sid, guard=W.FakeGuard()), Refusal)
    last = [r for r in S.read() if r["source"] == sid][-1]
    check("its own class names close it at intake (copy_candidate, action close, recorded), nothing committed",
          e is not None and e.code == "copy_candidate" and e.action == "close" and all(n in e.detail for n in legacy)
          and last["event"] == "closed" and last["reason"] == "copy_candidate"
          and (led.read_text() if led.exists() else "") == before, (e, last))
    for k, (what, kw, msg) in enumerate((("its Zenodo record", {"provider": "zenodo", "ref": "10762138"},
                           "zenodo:10762138 as never fetched"),
                          ("its title under another record", {"title": "2SeasonWeedDet8: a two-season, 8-class "
                                                                        "dataset for cross-season weed detection"},
                           "title matches a never-train dataset"),
                          ("a 'Two-Season' title", {"title": "Two-Season Weed Detection (cotton, Mississippi)"},
                           "title matches a never-train dataset"))):
        sid2 = TI.fetch_simple(cfg, "%08x-0000-0000-0000-000000010762" % (0x2b5e0000 + k),
                               TI.coco_files({"c0.png": W.img_bytes(970 + k)},
                                             [{"id": 1, "name": "weed: amaranthus palmeri"}],
                                             [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]}]))
        TI.edit_fetch(sid2, **kw)
        e = raises(lambda: I.intake(cfg, sid2, guard=W.FakeGuard()), Refusal)
        last = [r for r in S.read() if r["source"] == sid2][-1]
        check("the config's never-fetch entry of 2SeasonWeedDet8 closes it by %s (never_train), before any image "
              "is read" % what, e is not None and e.code == "never_train" and e.action == "close" and msg in e.detail
              and last["reason"] == "never_train", (e, last))
    sid3 = TI.fetch_simple(cfg, "2c5e9999-0000-0000-0000-000000010762",
                           TI.coco_files({"c1.png": W.img_bytes(990)}, [{"id": 1, "name": "weed: amaranthus palmeri"}],
                                         [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]}]))
    TI.edit_fetch(sid3, title="A 12-season weed detection set; 2 seasons of cotton")
    e = raises(lambda: I.intake(cfg, sid3, guard=W.FakeGuard()), Refusal)
    check("  (a title that only mentions seasons is not closed by it)", e is None or e.code != "never_train", e)


def main():
    try:
        cfg = W.config()
        W.build_cache()
        from weed_optimizer_framework.tools.collect import fetch as F
        F.disk_free = lambda p: 10 ** 13
        test_exams(cfg)
        test_same_lab_repackage(cfg)
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
