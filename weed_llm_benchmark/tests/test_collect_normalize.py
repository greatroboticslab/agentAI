#!/usr/bin/env python3
"""Annotation formats -> one form (docs/CONTINUOUS_LOOP.md §3.2 "Normalise";
group D acceptance "Normalisers: fixtures for COCO, VOC, VIA, WeedCOCO, MFWD
gt.csv and YOLO").

Each fixture is a small tree of generated pictures (144 x 128 px) and one
annotation file in the format, with the same two boxes on each picture:
  box A  x 10..50, y 20..60 px   -> (cx, cy, w, h) = (0.208333, 0.3125, 0.277778, 0.3125)
  box B  x 100..160, y 100..140  -> clipped to the frame: x 100..144, y 100..128
Pinned per format: the class list in source-id order, every image found with
its boxes normalised and clipped (the clip counted), a malformed box counted
as bad and left out, images without labels and labels without images listed;
a placed box that names no listed class (COCO, VOC, VIA) kept as NO_CLASS.
Also: WeedCOCO categories give their taxon as a hint; a box table (the
MFWD gt.csv layout: EPPO label ids, pixel corners, file names without their
extension, a tray column) finds its images by name and takes the tray as the
capture group, and rows naming images that were not fetched are counted; a
YOLO polygon line gives its bounding box and data.yaml gives the names; a
tree is listed once (every directory scanned once, no rglob anywhere in the
package's intake path); the dispatcher picks the format, or the one a known
item pins; the capture-group rule (column, regex, video, date, image).

Run:  python3 tests/test_collect_normalize.py
"""
import csv
import io
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_normalize_")
check, raises = W.check, W.raises
A = (0.208333, 0.3125, 0.277778, 0.3125)
B = ((100 + 144) / 2 / 144, (100 + 128) / 2 / 128, 44 / 144, 28 / 128)


def close(b, want, tol=1e-4):
    return all(abs(x - y) < tol for x, y in zip(b, want))


def pictures(root, n=3, sub="images", ext=".jpg"):
    out = []
    for i in range(n):
        p = pathlib.Path(root) / sub / ("pic_%d%s" % (i, ext))
        W.grid_img(p, 300 + i, fmt="PNG" if ext == ".png" else None)
        out.append(p)
    return out


def boxes_ok(res, n_items, cls_a, cls_b, name):
    items = sorted(res.items, key=lambda i: i["rel"])
    good = len(items) == n_items and all(len(i["boxes"]) == 2 for i in items)
    if good:
        for it in items:
            a, b = sorted(it["boxes"], key=lambda x: x[1])
            good = good and a[0] == cls_a and b[0] == cls_b and close(a[1:], A) and close(b[1:], B) and it["clipped"] == 1
    check("%s: %d images, boxes normalised and clipped (the clip counted)" % (name, n_items), good,
          [(i["rel"], i["boxes"], i["clipped"]) for i in items][:3])


def test_coco():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("COCO and WeedCOCO")
    root = TMP / "coco"
    pics = pictures(root, 3)
    imgs = [{"id": i, "file_name": p.name, "width": 144, "height": 128} for i, p in enumerate(pics)]
    imgs.append({"id": 9, "file_name": "missing.jpg", "width": 144, "height": 128})
    cats = [{"id": 7, "name": "b_second"}, {"id": 3, "name": "a_first"}]
    anns = []
    for i in range(3):
        anns.append({"id": 10 * i, "image_id": i, "category_id": 3, "bbox": [10, 20, 40, 40]})
        anns.append({"id": 10 * i + 1, "image_id": i, "category_id": 7, "bbox": [100, 100, 60, 40]})
    anns.append({"id": 99, "image_id": 0, "category_id": 3, "bbox": [10, 20, 0, 40]})
    (root / "annotations.json").write_text(json.dumps(W.coco_doc(imgs, cats, anns)))
    tree, res = N.read(root)
    check("coco: detected", res.format == "coco", res.format)
    check("coco: classes in source-id order", [c["name"] for c in res.classes] == ["a_first", "b_second"], res.classes)
    boxes_ok(res, 3, 0, 1, "coco")
    check("coco: a zero-width box is bad and left out", sum(i["bad"] for i in res.items) == 1)
    check("coco: an image the JSON lists but the tree lacks is listed", res.labels_without_images == ["missing.jpg"])
    root2 = TMP / "weedcoco"
    pics = pictures(root2, 2)
    imgs = [{"id": i, "file_name": "images/" + p.name, "width": 144, "height": 128} for i, p in enumerate(pics)]
    cats = [{"id": 0, "name": "weed: amaranthus palmeri (BBCH10-12)"}, {"id": 1, "name": "crop: gossypium", "role": "crop"}]
    anns = [{"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 20, 40, 40]},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [100, 100, 60, 40]},
            {"id": 3, "image_id": 1, "category_id": 0, "bbox": [10, 20, 40, 40]},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [100, 100, 60, 40],
             "segmentation": [[100, 100, 160, 100, 160, 140]]}]
    doc = dict(W.coco_doc(imgs, cats, anns), agcontexts=[{"id": 0}])
    (root2 / "weedcoco.json").write_text(json.dumps(doc))
    tree, res = N.read(root2)
    check("weedcoco: detected (agcontexts, role: taxon names)", res.format == "weedcoco", res.format)
    check("weedcoco: a category's taxon is its hint", res.classes[0]["hints"] == ["amaranthus palmeri"]
          and "gossypium" in res.classes[1]["hints"], res.classes)
    boxes_ok(res, 2, 0, 1, "weedcoco")
    root3 = TMP / "coco_seg"
    pics = pictures(root3, 1)
    (root3 / "a.json").write_text(json.dumps(W.coco_doc([{"id": 0, "file_name": pics[0].name, "width": 144,
                                                          "height": 128}], [{"id": 1, "name": "x"}],
                                                        [{"id": 1, "image_id": 0, "category_id": 1,
                                                          "segmentation": [[10, 20, 50, 20, 50, 60, 10, 60]]}])))
    _t, res = N.read(root3)
    check("coco: a segmentation-only annotation gives its polygon's box",
          len(res.items[0]["boxes"]) == 1 and close(res.items[0]["boxes"][0][1:], A), res.items[0]["boxes"])


def test_voc():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("Pascal VOC")
    root = TMP / "voc"
    pics = pictures(root, 3, sub="JPEGImages")
    for p in pics[:2]:
        xml = ("<annotation><filename>%s</filename><size><width>144</width><height>128</height></size>"
               "<object><name>zeta</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>"
               "<object><name>alpha</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>160</xmax><ymax>140</ymax></bndbox></object>"
               "<object><name>alpha</name><bndbox><xmin>x</xmin></bndbox></object>"
               "</annotation>") % p.name
        (root / "Annotations" / (p.stem + ".xml")).parent.mkdir(parents=True, exist_ok=True)
        (root / "Annotations" / (p.stem + ".xml")).write_text(xml)
    tree, res = N.read(root)
    check("voc: detected", res.format == "voc")
    check("voc: classes sorted by name", [c["name"] for c in res.classes] == ["alpha", "zeta"])
    boxes_ok(res, 2, 1, 0, "voc")
    check("voc: a malformed box is bad", sum(i["bad"] for i in res.items) == 2)
    check("voc: the picture without an XML is listed", res.images_without_labels == ["JPEGImages/pic_2.jpg"],
          res.images_without_labels)


def test_via():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("VIA")
    root = TMP / "via"
    pics = pictures(root, 2)
    meta = {}
    for p in pics:
        meta[p.name + "123"] = {"filename": p.name, "size": 123, "regions": [
            {"shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 40, "height": 40},
             "region_attributes": {"species": "one"}},
            {"shape_attributes": {"name": "polygon", "all_points_x": [100, 160, 130], "all_points_y": [100, 100, 140]},
             "region_attributes": {"species": "two"}}]}
    (root / "via_project.json").write_text(json.dumps({"_via_img_metadata": meta}))
    tree, res = N.read(root)
    check("via: detected", res.format == "via")
    check("via: classes from the region attribute", [c["name"] for c in res.classes] == ["one", "two"])
    boxes_ok(res, 2, 0, 1, "via")


def test_unlisted_class():
    """A box that can be placed but names no listed class keeps its place with
    normalize.NO_CLASS (intake makes it unmapped, so admission masks it); it
    is never counted bad and dropped, which would leave an unlabelled plant in
    a kept image (§3.2)."""
    from weed_optimizer_framework.tools.collect import normalize as N
    print("placed boxes of an unlisted class")
    root = TMP / "coco_unlisted"
    pics = pictures(root, 1)
    (root / "a.json").write_text(json.dumps(W.coco_doc(
        [{"id": 0, "file_name": pics[0].name, "width": 144, "height": 128}], [{"id": 1, "name": "x"}],
        [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]},
         {"id": 2, "image_id": 0, "category_id": 5, "bbox": [100, 100, 60, 40]},
         {"id": 3, "image_id": 0, "category_id": 5, "bbox": [10, 20, 0, 40]}])))
    _t, res = N.read(root)
    it = res.items[0]
    check("coco: a category id missing from the categories keeps its placed box as NO_CLASS; a degenerate one is bad",
          sorted(b[0] for b in it["boxes"]) == [N.NO_CLASS, 0] and it["bad"] == 1, (it["boxes"], it["bad"]))
    root = TMP / "voc_unlisted"
    pics = pictures(root, 1, sub="JPEGImages")
    (root / "Annotations").mkdir(parents=True, exist_ok=True)
    (root / "Annotations" / (pics[0].stem + ".xml")).write_text(
        "<annotation><filename>%s</filename><size><width>144</width><height>128</height></size>"
        "<object><name>zeta</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>"
        "<object><name></name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>160</xmax><ymax>140</ymax></bndbox></object>"
        "</annotation>" % pics[0].name)
    _t, res = N.read(root)
    it = res.items[0]
    check("voc: an object without a name keeps its placed box as NO_CLASS",
          sorted(b[0] for b in it["boxes"]) == [N.NO_CLASS, 0] and it["bad"] == 0, (it["boxes"], it["bad"]))
    root = TMP / "via_unlisted"
    pics = pictures(root, 1)
    meta = {pics[0].name: {"filename": pics[0].name, "size": 1, "regions": [
        {"shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 40, "height": 40},
         "region_attributes": {"species": "one"}},
        {"shape_attributes": {"name": "rect", "x": 100, "y": 100, "width": 60, "height": 40},
         "region_attributes": {}}]}}
    (root / "via_project.json").write_text(json.dumps({"_via_img_metadata": meta}))
    _t, res = N.read(root)
    it = res.items[0]
    check("via: a region without a class keeps its placed box as NO_CLASS",
          sorted(b[0] for b in it["boxes"]) == [N.NO_CLASS, 0] and it["bad"] == 0, (it["boxes"], it["bad"]))


def test_box_csv():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("box table (gt.csv layout)")
    root = TMP / "boxcsv"
    rows = []
    for tray in ("131801", "131802"):
        for k in range(2):
            stem = "POROL_%s_2021Y07M2%dD_00H49M09S_img" % (tray, k)
            p = root / "jpegs" / "POROL" / tray / (stem + ".jpg")
            W.grid_img(p, int(tray) + k)
            rows.append(["1", "POROL", "9", "10", "20", "50", "60", "POROL/%s/%s" % (tray, stem), tray])
            rows.append(["2", "SOLNI", "9", "100", "100", "160", "140", "POROL/%s/%s" % (tray, stem), tray])
    rows.append(["3", "ACHMI", "9", "1", "1", "5", "5", "ACHMI/133801/ACHMI_133801_img", "133801"])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["track_id", "label_id", "bbox_id", "xmin", "ymin", "xmax", "ymax", "filename", "tray_id"])
    w.writerows(rows)
    (root / "gt.csv").write_text(buf.getvalue())
    opts = {"format": "box_csv", "csv": "gt.csv", "label_column": "label_id", "file_column": "filename",
            "group_column": "tray_id"}
    tree, res = N.read(root, opts)
    check("box_csv: classes are the label ids, sorted", [c["name"] for c in res.classes] == ["POROL", "SOLNI"],
          res.classes)
    boxes_ok(res, 4, 0, 1, "box_csv")
    check("box_csv: the tray column is the capture group",
          sorted({i["group"] for i in res.items}) == ["131801", "131802"]
          and all(i["group_basis"] == "column" for i in res.items))
    check("box_csv: rows naming images that were not fetched are counted", res.rows_without_images == 1
          and res.labels_without_images == [], (res.rows_without_images, res.labels_without_images))
    _t, res2 = N.read(root)
    check("box_csv: detected without options too", res2.format == "box_csv")


def test_yolo():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("YOLO")
    root = TMP / "yolo"
    for split in ("train", "valid"):
        pics = pictures(root / split, 2)
        for p in pics:
            lab = root / split / "labels" / (p.stem + ".txt")
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("%d %.6f %.6f %.6f %.6f\n" % ((0,) + A) + "1 0.902778 0.937500 0.416667 0.312500\n"
                           + "bad line\n")
    pictures(root / "train", 1, sub="images_extra")
    orphan = root / "train" / "labels" / "orphan.txt"
    orphan.write_text("0 0.5 0.5 0.1 0.1\n")
    (root / "data.yaml").write_text("train: train/images\nval: valid/images\nnc: 2\nnames: ['first', 'second']\n")
    tree, res = N.read(root)
    check("yolo: detected; names from data.yaml", res.format == "yolo"
          and [c["name"] for c in res.classes] == ["first", "second"], (res.format, res.classes))
    boxes_ok(res, 4, 0, 1, "yolo")
    check("yolo: a malformed line is bad", all(i["bad"] == 1 for i in res.items))
    check("yolo: a label without an image is listed", "train/labels/orphan.txt" in res.labels_without_images,
          res.labels_without_images)
    root2 = TMP / "yolo_poly"
    pics = pictures(root2, 1)
    (root2 / "labels").mkdir(parents=True)
    (root2 / "labels" / "pic_0.txt").write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
    (root2 / "classes.txt").write_text("only\n")
    _t, res = N.read(root2)
    b = res.items[0]["boxes"][0]
    check("yolo: a polygon line gives its bounding box; classes.txt gives the names",
          res.classes[0]["name"] == "only" and close(b[1:], (0.2, 0.3, 0.2, 0.2)), (res.classes, b))
    root3 = TMP / "yolo_noname"
    pics = pictures(root3, 1)
    (root3 / "labels").mkdir(parents=True)
    (root3 / "labels" / "pic_0.txt").write_text("3 0.5 0.5 0.2 0.2\n")
    _t, res = N.read(root3)
    check("yolo: no class names file: numeric names, with the problem recorded",
          [c["name"] for c in res.classes] == ["0", "1", "2", "3"] and res.problems, (res.classes, res.problems))


def test_parquet():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("parquet shards")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        print("  skip hf_parquet: pyarrow is not installed here (%s)" % e)
        return
    root = TMP / "hf"
    root.mkdir()
    rows = []
    for i in range(2):
        rows.append({"image": {"bytes": W.img_bytes(400 + i), "path": "x%d.png" % i},
                     "objects": {"bbox": [[10.0, 20.0, 40.0, 40.0], [100.0, 100.0, 60.0, 40.0]], "categories": [0, 1]}})
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / "train-00000.parquet"
                   if (root / "data").mkdir() is None else None)
    tree, res = N.read(root, {"class_names": ["p", "q"]}, out_images=TMP / "hf_out")
    check("hf_parquet: detected, images written, class names from the card", res.format == "hf_parquet"
          and [c["name"] for c in res.classes] == ["p", "q"] and all(pathlib.Path(i["path"]).is_file() for i in res.items))
    boxes_ok(res, 2, 0, 1, "hf_parquet")


def test_listing_and_dispatch():
    from weed_optimizer_framework.tools.collect import normalize as N
    print("one listing; dispatch; capture groups")
    root = TMP / "coco"
    seen = []
    real = os.scandir

    def counting(path=None):
        seen.append(str(path))
        return real(path)
    os.scandir = counting
    try:
        N.read(root)
    finally:
        os.scandir = real
    dirs = [d for d in seen]
    check("every directory is listed once", len(dirs) == len(set(dirs)) and len(dirs) >= 2, dirs)
    col = W.ROOT / "weed_optimizer_framework" / "tools" / "collect"
    offenders = [p.name for p in sorted(col.rglob("*.py")) if ".rglob(" in p.read_text() or ".iterdir(" in p.read_text()]
    check("no rglob or iterdir in the collector's code", not offenders, offenders)
    e = raises(lambda: N.read(TMP / "coco", {"format": "nosuch"}), Exception)
    check("an unknown pinned format refuses", e is not None)
    empty = TMP / "empty"
    empty.mkdir()
    (empty / "readme.md").write_text("x")
    from weed_optimizer_framework.tools.collect import NormaliseError
    check("a tree with no known format refuses", raises(lambda: N.read(empty), NormaliseError) is not None)
    cg = N.capture_group
    check("capture group: the column wins", cg("a/b.jpg", {}, "t7") == ("t7", "column"))
    check("capture group: the options' regex", cg("site3/x_1.jpg", {"capture_group_regex": r"^(?P<group>site\d+)/"})
          == ("site3", "regex"))
    check("capture group: a video's frames", cg("v/clip.mp4_000123.png") == ("v/clip.mp4", "video"))
    check("capture group: a capture date", cg("d/IMG_20220129_1.jpg") == ("d/20220129", "date")
          and cg("POROL_1_2021Y07M28D_00H.jpg") == ("20210728", "date"))
    check("capture group: else the image alone", cg("a/IMG_1234.jpg") == ("a/IMG_1234.jpg", "image"))


def main():
    try:
        test_coco()
        test_voc()
        test_via()
        test_unlisted_class()
        test_box_csv()
        test_yolo()
        test_parquet()
        test_listing_and_dispatch()
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
