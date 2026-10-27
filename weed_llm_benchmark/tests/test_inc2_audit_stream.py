#!/usr/bin/env python3
"""The label audit with the stream's lookup (inc2/audit_stream.py; lever L4 since 2026-10-27).

What is pinned, and why:
  * inc/audit.py's audit is unchanged: on the audits of inc/audit.py's own test world (I1, Bswap, Breal, the
    defects, the overlap with the trusted set), whose images Step 1 holds or nothing holds, this module writes the
    same OUT.json (apart from the keys it adds), the same OUT_boxes.csv byte for byte and the same OUT.md up to its
    added section, with a stream present; the trusted boxes, probe and baseline are Step 1's only, also when the
    trusted manifest holds stream images;
  * a stream increment (whole and masked admissions over two committed batches), which inc/audit.py judges not at
    all (every image image_not_in_crops), is judged on every box the stream cropped: all but the box the intake
    left out as small, the crop whose embedding failed and the box whose label moved off its crop; a masked
    admission's label box j is judged on the crop of the kept box keep[j];
  * the statistics stay the audit's: a 40 % species swap of that increment is above its baseline with a noise
    estimate near 0.4, the increment itself is not;
  * every stream cause of a box without a crop maps to inc/audit.py's reasons (a row not in the queue or behind
    an unfinished last queue line, a batch not committed, a batch without crops.csv, no admission record, an
    admission neither whole nor masked, a label box past the kept ones, a box past the crops) and is counted; an
    image whose unmasked original is a trusted image is refused (in_trusted);
  * refusals, never guesses: a batch whose crops_info.json, emb.npz or verdicts.npz does not describe its
    crops.csv, whose files do not hash as batch.json recorded them, whose embedding misses a crop, or whose
    embedder is not Step 1's stops the audit; no output under the stream's directory;
  * a row of a batch without crops.csv is judged through its original: a rejoin row (masked or whole) on its
    original intake batch's crops under the original's key, a rejoin of a v1 row and a masked backfill row on
    Step 1's crops of the unmasked original, label box j on kept box keep[j]; their intake verdict and class are
    their own admission record's; an original that is not there, or an input that names none, leaves the row
    not judged (original_not_found, batch_without_crops);
  * an admission record that disagrees with its queue row (n_masked, the kept species counts) leaves that row
    not judged (admission_mismatch) and the audit runs on; a consistent one is judged;
  * the stream's directory is --stream-dir, else the step1_stream that the audited increments' stream.json
    names, else the default; increments of streams naming different directories are refused;
  * the intake verdict of every located box is reported beside the audit's; reruns are deterministic; the CLI
    takes --stream-dir.

The world: inc/audit.py's test world (tests/test_inc_audit.py build_world: Step 1 in verify's formats, a trusted P0)
plus a step1_stream directory in inc2.step1_stream's formats (queue/queue.jsonl, batches/bNNNN/{batch.json,
crops.csv, crops_skipped.csv, crops_info.json, emb.npz, verdicts.npz, admission.jsonl}). A stream box's feature is
its true class's direction (the mean of P0's Step 1 crops of that class) plus noise. No image is opened.

Run:  python3 tests/test_inc2_audit_stream.py
"""
import copy
import csv
import io
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc_audit as TA  # noqa: E402  (sets INC_DIR to its temporary directory before the package is imported)

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import audit as A  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import audit_stream as AS  # noqa: E402

TMP = TA.TMP
FAILURES = []
EMBEDDER = "fake"                      # test_inc_audit's Step 1 shards' embedder
VERDICT_CODES = ["verified", "conflict", "unknown", "other_ok", "small", "failed", "unmapped"]


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def refused(fn, needle):
    try:
        fn()
    except A.AuditError as e:
        return needle in str(e) or (print("    (refused with: %s)" % e) and False)
    return False


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def root():
    return AS.default_root()


# ------------------------------------------------------------- the stream
def class_dirs():
    """Per INC class, the unit mean of P0-session train_core crops' features of that (true) class."""
    crops = V.Crops()
    X, _ = V.load_embeddings(crops)
    dirs = {}
    for k in range(C.NC - 1):
        m = (crops.set == V.SETS.index("core")) & (crops.label == k)
        f = np.asarray(X[m], dtype=np.float32)
        f = f[np.isfinite(f).all(axis=1)]
        v = f.mean(axis=0)
        dirs[k] = v / np.linalg.norm(v)
    return dirs


def _geom(rng):
    return tuple(float("%.6f" % x) for x in (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
                                             rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3)))


class Batch(object):
    """A stream batch being written: its pool rows (key, image, sha, boxes), crops, skipped boxes, admission
    records, embeddings, intake verdicts."""

    def __init__(self, bid, kind="intake"):
        self.bid, self.kind = bid, kind
        self.crops, self.skipped, self.adm, self.X, self.verd = [], [], [], [], []

    def add(self, key, image, source, boxes, feats, admission="whole", keep=None, skip=(), nan=(), verd=None):
        for b, box in enumerate(boxes):
            if b in skip:
                self.skipped.append(["pool", key, b, skip[b] if isinstance(skip, dict) else "small"])
                continue
            self.crops.append(["pool", key, str(image), source, source, b, box[1:], box[0]])
            self.X.append(np.full(len(feats[b]), np.nan) if b in nan else feats[b])
            self.verd.append(VERDICT_CODES.index(verd[b] if verd else ("other_ok" if box[0] == C.OTHER_PLANT
                                                                       else "verified")))
        keep = list(range(len(boxes))) if keep is None else list(keep)
        self.adm.append({"admission": admission, "image_verdict": "admitted", "keep": keep, "key": key,
                         "labels": [int(b[0]) for b in boxes], "mask": [b for b in range(len(boxes)) if b not in keep],
                         "mask_record": None, "overlaps": [], "refusal": None, "source": source,
                         "verdicts": ["verified"] * len(boxes)})

    def write(self, committed=True, crops=True, embedder=EMBEDDER):
        d = root() / "batches" / self.bid
        d.mkdir(parents=True, exist_ok=True)
        (d / "admission.jsonl").write_text("".join(canon(r) + "\n" for r in self.adm))
        if crops:
            buf = io.StringIO()
            wr = csv.writer(buf, lineterminator="\r\n")
            wr.writerow(V.CROP_FIELDS)
            for cid, (st, key, image, source, group, b, g, lab) in enumerate(self.crops):
                wr.writerow([cid, st, key, image, source, group, b] + ["%.6f" % x for x in g]
                            + [640, 480, int(lab), ""])
            (d / "crops.csv").write_bytes(buf.getvalue().encode())
            buf = io.StringIO()
            wr = csv.writer(buf, lineterminator="\r\n")
            wr.writerow(V.SKIPPED_FIELDS)
            wr.writerows(self.skipped)
            (d / "crops_skipped.csv").write_bytes(buf.getvalue().encode())
            sha = C.sha256_file(d / "crops.csv")
            n = len(self.crops)
            V._write_json(d / "crops_info.json", {"crops": n, "crops_sha256": sha, "min_box_px": 16,
                                                  "skipped_sha256": C.sha256_file(d / "crops_skipped.csv")})
            X = np.asarray(self.X, dtype=np.float16).reshape(n, -1)
            V._save_npz(d / "emb.npz", {"crops_sha256": sha, "embedder": embedder, "dim": int(X.shape[1]),
                                        "crops": n, "stats": {}},
                        crop_ids=np.arange(n, dtype=np.int64), X=X)
            V._save_npz(d / "verdicts.npz", {"crops_sha256": sha, "verdict_codes": VERDICT_CODES, "verifier": "v1",
                                             "oof_judged": 0},
                        crop_id=np.arange(n, dtype=np.int64), verdict=np.asarray(self.verd, dtype=np.int8),
                        pred=np.zeros(n, dtype=np.int64), p=np.ones(n), cosine=np.ones(n))
        if committed:
            inputs = {}
            for name, f in (("crops", "crops.csv"), ("emb", "emb.npz"), ("verdicts", "verdicts.npz")):
                if (d / f).is_file():
                    inputs[name] = {"path": str(d / f), "sha256": C.sha256_file(d / f)}
            V._write_json(d / "batch.json", {"format": "inc2-step1-stream/1", "kind": self.kind,
                                             "input": "%s:%s" % (self.kind, self.bid), "inputs": inputs})
        return d


def queue_row(bid, key, image, label, source, admission="whole", unmasked=None, unmasked_sha=None, sha=None):
    sha = sha or C.sha256_text("img:" + key)
    return {"admission": admission, "batch": bid, "format": "inc2-stream-queue/1", "holds": [], "image": str(image),
            "key": key, "label": str(label), "label_sha256": C.sha256_file(label), "session": "", "sha256": sha,
            "source": source, "unmasked_image": str(unmasked or image), "unmasked_sha256": unmasked_sha or sha}


def manifest_row(q):
    return {k: q[k] for k in C.MANIFEST_KEYS}


def build_stream(p0_rows, paths):
    """The stream: b0001 (whole, masked, small, failed), b0002 (whole), b0000 (backfill, no crops.csv), b0003 (not
    committed), and the manifests S1 (the increment), S1swap (40 % of its species boxes swapped), Sdef (every
    cause of a box without a crop) and P0s (P0 plus two stream images, as a trusted set)."""
    shutil.rmtree(root(), ignore_errors=True)
    rng = np.random.default_rng(20261027)
    dirs = class_dirs()
    D = len(dirs[0])

    def feat(k):
        return dirs[k] + TA.NOISE * rng.normal(size=D) / np.sqrt(D)

    lab_dir = root() / "labels"
    queue, truth = [], {"masked": {}, "small": [], "failed": [], "moved": [], "s1_species": 0, "s1_boxes": 0}
    b1, b2, b0, b3 = Batch("b0001"), Batch("b0002"), Batch("b0000", "backfill"), Batch("b0003")

    def image_boxes(n, other=0.2):
        out = []
        for _ in range(n):
            k = C.OTHER_PLANT if rng.random() < other else int(rng.integers(0, 12))
            out.append((k,) + _geom(rng))
        return out

    s1 = []
    for i in range(40):
        key = "slugS__img_%03d" % i
        img = TMP / "repo" / "datasets" / "slugS" / "images" / ("img_%03d.jpg" % i)
        if 30 <= i < 38:                                         # masked: box 1 of 4 masked, keep [0, 2, 3]
            boxes = image_boxes(4)
            feats = [feat(b[0]) if b[0] < C.OTHER_PLANT else rng.normal(size=D) / np.sqrt(D) for b in boxes]
            b1.add(key, img, "slugS", boxes, feats, admission="masked", keep=[0, 2, 3])
            kept = [boxes[0], boxes[2], boxes[3]]
            lp = lab_dir / "slugS" / (key + ".txt")
            C.write_yolo(lp, kept)
            masked = root() / "images_masked" / "slugS" / (key + ".png")
            q = queue_row("b0001", key, masked, lp, "slugS", admission="masked", unmasked=img,
                          unmasked_sha=C.sha256_text("img:" + key), sha=C.sha256_text("masked:" + key))
            truth["masked"][key] = [0, 2, 3]
            label_boxes = kept
        else:
            boxes = image_boxes(3)
            feats = [feat(b[0]) if b[0] < C.OTHER_PLANT else rng.normal(size=D) / np.sqrt(D) for b in boxes]
            skip, nan = {}, ()
            if i == 38:
                skip = {1: "small"}
                truth["small"].append((key, 1))
            if i == 39:
                nan = (0,)
                truth["failed"].append((key, 0))
            b1.add(key, img, "slugS", boxes, feats, skip=skip, nan=nan)
            lp = lab_dir / "slugS" / (key + ".txt")
            C.write_yolo(lp, boxes)
            q = queue_row("b0001", key, img, lp, "slugS")
            label_boxes = boxes
        queue.append(q)
        s1.append(manifest_row(q))
        truth["s1_boxes"] += len(label_boxes)
        truth["s1_species"] += sum(1 for b in label_boxes if b[0] < C.OTHER_PLANT)
    # refused rows of the batch (not admitted, never queued) still have crops
    for i in range(3):
        key = "slugS__refused_%03d" % i
        boxes = image_boxes(2)
        b1.add(key, TMP / "repo" / "datasets" / "slugS" / "images" / ("ref_%03d.jpg" % i), "slugS", boxes,
               [feat(b[0]) if b[0] < C.OTHER_PLANT else rng.normal(size=D) / np.sqrt(D) for b in boxes],
               admission="not_admitted", keep=[])
    for i in range(10):
        key = "slugT__img_%03d" % i
        img = TMP / "repo" / "datasets" / "slugT" / "images" / ("img_%03d.jpg" % i)
        boxes = image_boxes(2)
        b2.add(key, img, "slugT", boxes,
               [feat(b[0]) if b[0] < C.OTHER_PLANT else rng.normal(size=D) / np.sqrt(D) for b in boxes],
               verd=["unknown", "conflict"])
        lp = lab_dir / "slugT" / (key + ".txt")
        C.write_yolo(lp, boxes)
        q = queue_row("b0002", key, img, lp, "slugT")
        queue.append(q)
        row = manifest_row(q)
        truth["s1_boxes"] += len(boxes)
        truth["s1_species"] += sum(1 for b in boxes if b[0] < C.OTHER_PLANT)
        if i == 0:                                           # its label moved box 0 off its crop (another label)
            moved = [(boxes[0][0], boxes[0][1] + 0.01) + boxes[0][2:], (boxes[1][0], boxes[1][1] + 0.00005)
                     + boxes[1][2:]]
            lp2 = TMP / "inc" / "relabel" / (key + ".txt")
            C.write_yolo(lp2, moved)
            row = dict(row, label=str(lp2), label_sha256=C.sha256_file(lp2))
            truth["moved"].append((key, 0))
        s1.append(row)

    # Sdef: every cause of a box without a crop
    sdef = []
    defects = {}

    def dq(bid, key, boxes, batch=None, admission="whole", keep=None, adm=True, add=True, unmasked=None,
           unmasked_sha=None, image=None, nboxes_crops=None):
        img = image or (TMP / "repo" / "datasets" / "slugD" / "images" / (key + ".jpg"))
        lp = lab_dir / "slugD" / (key + ".txt")
        C.write_yolo(lp, boxes)
        q = queue_row(bid, key, img, lp, "slugD", admission=admission, unmasked=unmasked, unmasked_sha=unmasked_sha)
        if batch is not None:
            cb = boxes[:nboxes_crops] if nboxes_crops is not None else boxes
            batch.add(key, unmasked or img, "slugD", cb, [feat(b[0]) for b in cb], admission=admission, keep=keep)
            if not adm:
                batch.adm.pop()
        if add:
            queue.append(q)
        sdef.append(manifest_row(q))
        return q

    sp = lambda n: [(int(rng.integers(0, 12)),) + _geom(rng) for _ in range(n)]  # noqa: E731
    dq("b0001", "slugD__noqueue", sp(2), add=False); defects["slugD__noqueue"] = AS.S_NOT_IN_QUEUE
    dq("b0003", "slugD__uncommitted", sp(2), batch=b3); defects["slugD__uncommitted"] = AS.S_NOT_COMMITTED
    pool_img = paths["pool_rows"][0]
    dq("b0000", "slugD__backfill", sp(2), batch=b0, admission="masked", keep=[0, 1],
       image=root() / "images_masked" / "slugD" / "bf.png", unmasked=pool_img["image"],
       unmasked_sha=pool_img["sha256"])
    defects["slugD__backfill"] = AS.S_NO_CROPS
    dq("b0002", "slugD__noadm", sp(2), batch=b2, adm=False); defects["slugD__noadm"] = AS.S_NO_ADMISSION
    q = dq("b0002", "slugD__notadm", sp(2), batch=b2)
    b2.adm[-1]["admission"] = "not_admitted"
    defects["slugD__notadm"] = AS.S_NOT_ADMITTED
    dq("b0002", "slugD__notkept", sp(3), batch=b2, admission="masked", keep=[0, 1], nboxes_crops=3,
       image=root() / "images_masked" / "slugD" / "nk.png", unmasked=TMP / "repo" / "slugD_nk.jpg")
    defects["slugD__notkept"] = AS.S_NOT_KEPT                       # box 2 of the label past keep [0, 1]
    dq("b0002", "slugD__extra", sp(3), batch=b2, nboxes_crops=2)
    defects["slugD__extra"] = AS.S_NO_CROP_ROW                      # box 2 of a whole admission past its crops
    t = p0_rows[7]
    dq("b0002", "slugD__trusted", sp(2), batch=b2, admission="masked", keep=[0, 1],
       image=root() / "images_masked" / "slugD" / "tr.png", unmasked=t["image"], unmasked_sha=t["sha256"])
    defects["slugD__trusted"] = "in_trusted"
    # the same key, image and bytes queued by b0001 and again by b0002 under another label (a rejoin): the row
    # with the manifest's label is b0001's
    dup = sp(2)
    q1 = dq("b0001", "slugD__dup", dup, batch=b1)
    lp2 = _wl(lab_dir / "slugD" / "dup_rejoined.txt", [((dup[0][0] + 1) % 12,) + dup[0][1:], dup[1]])
    b2.add("slugD__dup", q1["image"], "slugD", dup, [feat(b[0]) for b in dup])
    queue.append(dict(q1, batch="b0002", label=str(lp2), label_sha256=C.sha256_file(lp2)))
    defects["slugD__dup"] = "b0001"
    # a queue row of the key and image, but other bytes (the image changed after admission): not this row's
    q = dq("b0002", "slugD__othersha", sp(2), batch=b2)
    sdef[-1] = dict(sdef[-1], sha256="f" * 64)
    defects["slugD__othersha"] = AS.S_NOT_IN_QUEUE

    # Sorig: rows of batches without crops.csv, found through their original (a rejoin batch b0004; the backfill)
    b4 = Batch("b0004", "rejoin")
    sorig, orig = [], {"via": {}, "defects": {}, "mismatch": {}}

    def counts(labels, keep):
        sp_ = [0] * C.OTHER_PLANT
        for b_ in keep:
            if labels[b_] < C.OTHER_PLANT:
                sp_[labels[b_]] += 1
        return sp_, sum(1 for b_ in keep if labels[b_] == C.OTHER_PLANT)

    def adm_rec(batch, key, labels, keep, verdicts, admission, supersedes=None):
        rec = {"admission": admission, "image_verdict": "admitted", "keep": list(keep), "key": key,
               "labels": list(labels), "mask": [b_ for b_ in range(len(labels)) if b_ not in keep],
               "mask_record": None, "overlaps": [], "refusal": None, "source": "slugR", "verdicts": list(verdicts)}
        if supersedes:
            rec.update(supersedes=supersedes, changed_boxes=[0])
        batch.adm.append(rec)

    def orow(bid, key, boxes, keep, admission, inp, unmasked, unmasked_sha, supersedes=None, batch=None,
             verdicts=None, add_adm=True):
        """A queue row of a batch without crops.csv, its label the kept boxes, and its admission record."""
        labels = [int(b_[0]) for b_ in boxes]
        kept = [boxes[b_] for b_ in keep]
        lp = _wl(lab_dir / "slugR" / (key.replace("/", "_") + ".txt"), kept)
        if admission == "masked":
            image, sha = root() / "images_masked" / "slugR" / (key + ".png"), C.sha256_text("masked:" + key)
        else:
            image, sha = unmasked, unmasked_sha
        q = queue_row(bid, key, image, lp, "slugR", admission=admission, unmasked=unmasked,
                      unmasked_sha=unmasked_sha, sha=sha)
        sp_, ot_ = counts(labels, keep)
        q.update(input=inp, supersedes=supersedes, n_masked=len(boxes) - len(keep), species_boxes=sp_,
                 other_boxes=ot_)
        if add_adm:
            adm_rec(batch, key, labels, keep, verdicts or ["verified"] * len(boxes), admission, supersedes)
        queue.append(q)
        sorig.append(manifest_row(q))
        return q, labels

    def relabel(boxes, b_=0):
        out = [tuple(x) for x in boxes]
        out[b_] = ((out[b_][0] + 1) % C.OTHER_PLANT if out[b_][0] < C.OTHER_PLANT else 3,) + tuple(out[b_][1:])
        return out

    # (1) a masked rejoin of a whole b0001 row: its crops are b0001's, under the original's key
    k1 = "slugS__img_005"
    o1 = next(x for x in queue if x["key"] == k1)
    bx1 = relabel(C.read_yolo(o1["label"]), 1)
    _q, labs = orow("b0004", k1 + "__rj1", bx1, [0, 2], "masked", "rejoin_v1:b0001:intake:x", o1["image"],
                    o1["sha256"], supersedes=k1, batch=b4, verdicts=["verified", "conflict", "unknown"])
    orig["via"][k1 + "__rj1"] = ("b0001", k1, [0, 2], ["verified", "conflict", "unknown"], labs)
    # (2) a whole rejoin of a b0002 row: the same image and bytes as its original, another key
    k2 = "slugT__img_004"
    o2 = next(x for x in queue if x["key"] == k2)
    bx2 = relabel(C.read_yolo(o2["label"]), 0)
    _q, labs = orow("b0004", k2 + "__rj1", bx2, [0, 1], "whole", "rejoin_v1:b0002:intake:x", o2["image"],
                    o2["sha256"], supersedes=k2, batch=b4, verdicts=["unknown", "verified"])
    orig["via"][k2 + "__rj1"] = ("b0002", k2, [0, 1], ["unknown", "verified"], labs)
    # (3) a masked rejoin of a v1 row: its crops are Step 1's, of its unmasked original
    pr = paths["pool_rows"][4]
    bx3 = relabel(C.read_yolo(pr["label"]), 0)
    _q, labs = orow("b0004", pr["key"] + "__rj1", bx3, [0, 2, 3], "masked", "rejoin_v1:v1", pr["image"],
                    pr["sha256"], supersedes=pr["key"], batch=b4, verdicts=["verified", "conflict", "verified",
                                                                             "unknown"])
    orig["via"][pr["key"] + "__rj1"] = ("step1", pr["image"], [0, 2, 3], ["verified", "conflict", "verified",
                                                                           "unknown"], labs)
    # (4) a masked backfill row (b0000, input v1): Step 1 cut its unmasked original
    pb = paths["pool_rows"][7]
    bxb = [tuple(x) for x in C.read_yolo(pb["label"])]
    _q, labs = orow("b0000", pb["key"], bxb, [1, 2], "masked", "v1", pb["image"], pb["sha256"], batch=b0,
                    verdicts=["conflict", "verified", "verified", "conflict"])
    orig["via"][pb["key"]] = ("step1", pb["image"], [1, 2], ["conflict", "verified", "verified", "conflict"], labs)
    # defects: an original's batch that is not there; an input naming no original; a backfill row whose original
    # Step 1 does not hold
    orow("b0004", "slugR__lost__rj1", sp(2), [0, 1], "whole", "rejoin_v1:b0099:intake:x",
         TMP / "repo" / "slugR_lost.jpg", C.sha256_text("img:lost"), supersedes="slugR__lost", batch=b4)
    orig["defects"]["slugR__lost__rj1"] = AS.S_SOURCE_MISSING
    orow("b0004", "slugR__reg__rj1", sp(2), [0, 1], "whole", "rejoin_v1:registry_x", TMP / "repo" / "slugR_reg.jpg",
         C.sha256_text("img:reg"), supersedes="slugR__reg", batch=b4)
    orig["defects"]["slugR__reg__rj1"] = AS.S_NO_CROPS
    orow("b0000", "slugR__bfmiss", sp(2), [0], "masked", "v1", TMP / "repo" / "slugR_bfmiss.jpg",
         C.sha256_text("img:bfmiss"), batch=b0)
    orig["defects"]["slugR__bfmiss"] = AS.S_SOURCE_MISSING
    # admission records that do not describe their queue rows (b0002 has crops): n_masked, the species counts
    for key, field in (("slugR__mm_masked", "n_masked"), ("slugR__mm_species", "species_boxes")):
        img = TMP / "repo" / "datasets" / "slugR" / (key + ".jpg")
        bx = sp(2)
        b2.add(key, img, "slugR", bx, [feat(b_[0]) for b_ in bx])
        lp = _wl(lab_dir / "slugR" / (key + ".txt"), bx)
        q = queue_row("b0002", key, img, lp, "slugR")
        sp_, ot_ = counts([int(b_[0]) for b_ in bx], [0, 1])
        q.update(n_masked=0, species_boxes=sp_, other_boxes=ot_)
        if field == "n_masked":
            q["n_masked"] = 1
        else:
            q["species_boxes"] = [x + (1 if j_ == 0 else 0) for j_, x in enumerate(sp_)]
        queue.append(q)
        sorig.append(manifest_row(q))
        orig["mismatch"][key] = field
    # and one that does (the check passes a consistent row)
    key = "slugR__mm_ok"
    img = TMP / "repo" / "datasets" / "slugR" / (key + ".jpg")
    bx = sp(2)
    b2.add(key, img, "slugR", bx, [feat(b_[0]) for b_ in bx])
    q = queue_row("b0002", key, img, _wl(lab_dir / "slugR" / (key + ".txt"), bx), "slugR")
    sp_, ot_ = counts([int(b_[0]) for b_ in bx], [0, 1])
    q.update(n_masked=0, species_boxes=sp_, other_boxes=ot_)
    queue.append(q)
    sorig.append(manifest_row(q))
    orig["mismatch"][key] = None

    b1.write()
    b2.write()
    b0.write(crops=False)
    b3.write(committed=False)
    b4.write(crops=False)
    qp = root() / "queue" / "queue.jsonl"
    qp.parent.mkdir(parents=True, exist_ok=True)
    partial = queue_row("b0002", "slugD__partial", TMP / "repo" / "slugD_partial.jpg",
                        _wl(lab_dir / "slugD" / "partial.txt", sp(1)), "slugD")
    b2.add("slugD__partial", partial["image"], "slugD", [(0, 0.5, 0.5, 0.1, 0.1)], [feat(0)])
    b2.write()
    qp.write_text("".join(canon(r) + "\n" for r in queue) + canon(partial))   # an append still in progress
    sdef.append(manifest_row(partial))
    defects["slugD__partial"] = AS.S_NOT_IN_QUEUE

    mdir = TMP / "inc" / "stream_t" / "manifests"
    out = {"S1": mdir / "S1.jsonl", "Sdef": mdir / "Sdef.jsonl", "S1swap": mdir / "S1swap.jsonl",
           "P0s": mdir / "P0s.jsonl", "Sorig": mdir / "Sorig.jsonl"}
    C.write_manifest(out["S1"], s1)
    C.write_manifest(out["Sdef"], sdef)
    C.write_manifest(out["Sorig"], sorig)
    truth["orig"] = orig
    # S1swap: the same images, 40 % of the species boxes given one uniform wrong species
    srng = np.random.default_rng(7)
    swapped = []
    n_sp = n_sw = 0
    for r in s1:
        bx = C.read_yolo(r["label"])
        new = []
        for b in bx:
            if b[0] < C.OTHER_PLANT:
                n_sp += 1
                if srng.random() < 0.4:
                    n_sw += 1
                    b = ((b[0] + int(srng.integers(1, 12))) % 12,) + tuple(b[1:])
            new.append(b)
        lp = TMP / "inc" / "stream_t" / "labels" / "S1swap" / (r["key"] + ".txt")
        C.write_yolo(lp, new)
        swapped.append(dict(r, label=str(lp), label_sha256=C.sha256_file(lp)))
    C.write_manifest(out["S1swap"], swapped)
    truth["swap_share"] = n_sw / float(n_sp)
    C.write_manifest(out["P0s"], list(p0_rows) + [s1[0], s1[31]])
    return out, truth, defects


def _wl(path, boxes):
    C.write_yolo(path, boxes)
    return path


# ------------------------------------------------------------------ helpers
ADDED_TOP = ("lookup",)
ADDED_AUDIT = ("images_in_stream", "images_without_crops", "stream", "coverage")


def strip_new(res):
    r = copy.deepcopy(res)
    for k in ("built_utc", "seconds", "outputs") + ADDED_TOP:
        r.pop(k, None)
    r["trusted"].pop("lookup", None)
    r["inputs"].pop("step1_stream", None)
    r["notes"] = [n for n in r["notes"] if n != AS.STREAM_NOTE]
    for a in r["audits"].values():
        for k in ADDED_AUDIT:
            a.pop(k, None)
    return r


def boxes_rows(out):
    with open(A._paths(out)[2], newline="") as fh:
        return list(csv.DictReader(fh))


def audit_specs(paths, names):
    return [(n, paths[n]) for n in names]


# -------------------------------------------------------------------- tests
def t_equivalence(paths):
    print("inc/audit.py's audit unchanged on Step 1's boxes (a stream present)")
    names = ["I1", "Bswap", "Breal", "defects", "overlap"]
    o1, o2 = TMP / "eq" / "old" / "audit.json", TMP / "eq" / "new" / "audit.json"    # one name: OUT.md cites it
    old = A.run_audit(paths["P0"], audit_specs(paths, names), o1)
    new = AS.run_audit(paths["P0"], audit_specs(paths, names), o2)
    a, b = TA.strip(old), strip_new(new)
    check("OUT.json: the same document apart from the added keys (probe, baseline, every rate and test)",
          json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True),
          [k for k in a if a.get(k) != b.get(k)])
    check("OUT_boxes.csv byte for byte", A._paths(o1)[2].read_bytes() == A._paths(o2)[2].read_bytes())
    md1, md2 = A._paths(o1)[1].read_text(), A._paths(o2)[1].read_text()
    md2s = md2.replace("- %s\n" % AS.STREAM_NOTE, "", 1)
    check("OUT.md: inc/audit.py's (its notes plus the lookup's), then the stream section",
          md2s.startswith(md1) and "## Stream lookup" in md2s[len(md1):] and "- %s\n" % AS.STREAM_NOTE in md2)
    check("the added keys: lookup step1+step1_stream, the trusted lookup step1",
          new["lookup"] == "step1+step1_stream" and new["trusted"]["lookup"] == "step1")
    d = new["audits"]["defects"]
    check("the defects' absent image (not in Step 1, not in the queue): image_not_in_crops, stream not_in_queue",
          d["no_embedding_reasons"].get("image_not_in_crops") == old["audits"]["defects"]["no_embedding_reasons"].get(
              "image_not_in_crops") and d["stream"]["not_located"].get("not_in_queue", 0) > 0,
          (d["no_embedding_reasons"], d["stream"]))
    return new


def t_trusted_step1_only(paths, spaths, base):
    print("the trusted boxes are Step 1's only (a trusted set holding stream images)")
    o = TMP / "tr" / "a.json"
    res = AS.run_audit(spaths["P0s"], [("S1", spaths["S1"])], o)
    old = A.run_audit(spaths["P0s"], [("I1", paths["I1"])], TMP / "tr" / "old.json")
    check("the probe and the baseline equal inc/audit.py's for that trusted set (its stream rows unused)",
          res["baseline_cv"] == old["baseline_cv"] and res["trusted"]["species_boxes_used"]
          == old["trusted"]["species_boxes_used"] and res["probe"]["thresholds"] == old["probe"]["thresholds"])
    s1 = res["audits"]["S1"]
    check("the trusted set's two stream images are refused in S1 (in_trusted)", s1["images_refused_in_trusted"] == 2,
          s1["images_refused_in_trusted"])


def t_stream(spaths, truth):
    print("a stream increment: judged on every box the stream cropped")
    o_old, o_new = TMP / "st" / "old.json", TMP / "st" / "new.json"
    old = A.run_audit(TA_P0, [("S1", spaths["S1"])], o_old)
    new = AS.run_audit(TA_P0, [("S1", spaths["S1"]), ("S1swap", spaths["S1swap"])], o_new)
    a0, a = old["audits"]["S1"], new["audits"]["S1"]
    check("inc/audit.py: 0 boxes judged, every one image_not_in_crops",
          a0["species"]["boxes_judged"] == 0 and a0["no_embedding_reasons"] == {"image_not_in_crops": a0["boxes"]},
          a0["no_embedding_reasons"])
    st = a["boxes_by_status"]
    judged = sum(st[s] for s in A.JUDGED)
    want = truth["s1_boxes"] - 3
    check("with the stream: %d of %d boxes judged (all but the small, the failed and the moved box)"
          % (want, truth["s1_boxes"]), judged == want and a["no_embedding_reasons"]
          == {"small": 1, "failed": 1, "geometry_mismatch": 1}, (judged, a["no_embedding_reasons"]))
    cv = a["coverage"]
    check("coverage: species boxes judged / species-labelled boxes", cv["species_boxes"] == truth["s1_species"]
          and cv["species_judged"] == a["species"]["boxes_judged"] and cv["species_judged"] >= truth["s1_species"] - 3,
          cv)
    s = a["stream"]
    check("the stream block: located per batch, the two batches", sorted(s["per_batch"]) == ["b0001", "b0002"]
          and s["boxes_located"] == want and s["images"] == 50, s)
    rows = {(r["key"], int(r["box"])): r for r in boxes_rows(o_new) if r["manifest"] == "S1"}
    crops1 = V.Crops(root() / "batches" / "b0001" / "crops.csv")
    cid = {(crops1.key[i], int(crops1.box[i])): i for i in range(crops1.n)}
    ok = True
    for key, keep in truth["masked"].items():
        for j, k in enumerate(keep):
            r = rows[(key, j)]
            ok &= r["crop_set"] == "stream/b0001" and int(r["crop_id"]) == cid[(key, k)] and r["matched_by"] == "stream"
    check("a masked admission: label box j judged on the crop of kept box keep[j] (crop_set stream/b0001)", ok)
    k0, b0 = truth["small"][0]
    kf, bf = truth["failed"][0]
    km, bm = truth["moved"][0]
    check("the small box, the failed crop, the moved label box: not judged, each with its reason",
          rows[(k0, b0)]["reason"] == "small" and rows[(kf, bf)]["reason"] == "failed"
          and rows[(km, bm)]["reason"] == "geometry_mismatch" and rows[(km, 1)]["status"] in A.JUDGED,
          (rows[(k0, b0)]["reason"], rows[(kf, bf)]["reason"], rows[(km, bm)]["reason"]))
    check("the relabelled image matched its queue row by image and sha256 (another label)",
          new["inputs"]["step1_stream"]["queue"]["matched_with_another_label"] >= 1)
    check("the intake verdicts of the located boxes are reported (b0002's unknown / conflict among them)",
          s["intake_verdicts"].get("unknown", 0) >= 9 and sum(s["intake_verdicts"].values()) == want
          and sum(s["intake_vs_audit"].values()) == want, s["intake_verdicts"])
    sw = new["audits"]["S1swap"]["species"]
    check("the clean increment is not above its baseline", a["species"]["above_baseline"] is False,
          a["species"]["conflict_rate"])
    check("the 40 %% swap is above its baseline, noise estimate %.2f near the planted %.2f"
          % (sw["noise_estimate_corrected"] or -1, truth["swap_share"]),
          sw["above_baseline"] is True and abs((sw["noise_estimate_corrected"] or 0) - truth["swap_share"]) < 0.08, sw)
    check("the MD carries the stream section", "## Stream lookup" in A._paths(o_new)[1].read_text())
    return new


def t_defects(spaths, defects):
    print("every stream cause of a box without a crop")
    o = TMP / "df" / "a.json"
    res = AS.run_audit(TA_P0, [("Sdef", spaths["Sdef"])], o)
    rows = boxes_rows(o)
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    want_reason = {AS.S_NOT_IN_QUEUE: "image_not_in_crops", AS.S_NOT_COMMITTED: "image_not_in_crops",
                   AS.S_NO_CROPS: "image_not_in_crops", AS.S_NO_ADMISSION: "no_crop_row",
                   AS.S_NOT_ADMITTED: "no_crop_row"}
    for key, cause in sorted(defects.items()):
        rs = by[key]
        if cause in ("b0001",):
            check("%s: queued twice (b0001, b0002 rejoined): judged on the batch whose row has its label (%s)"
                  % (key, cause), all(r["status"] in A.JUDGED and r["crop_set"] == "stream/" + cause for r in rs),
                  [(r["status"], r["crop_set"]) for r in rs])
        elif cause == "in_trusted":
            check("%s: in_trusted (its unmasked original is a trusted image)" % key,
                  all(r["status"] == "in_trusted" for r in rs), [r["status"] for r in rs])
        elif cause in (AS.S_NOT_KEPT, AS.S_NO_CROP_ROW):
            check("%s: boxes 0, 1 judged; box 2 no_crop_row (%s)" % (key, cause),
                  rs[0]["status"] in A.JUDGED and rs[1]["status"] in A.JUDGED and rs[2]["reason"] == "no_crop_row",
                  [(r["status"], r["reason"]) for r in rs])
        else:
            check("%s: %s -> %s" % (key, cause, want_reason[cause]),
                  all(r["status"] == "no_embedding" and r["reason"] == want_reason[cause] for r in rs),
                  [(r["status"], r["reason"]) for r in rs])
    nl = res["audits"]["Sdef"]["stream"]["not_located"]
    check("the stream block counts each cause", nl.get(AS.S_NOT_IN_QUEUE) == 5 and nl.get(AS.S_NOT_COMMITTED) == 2
          and nl.get(AS.S_NO_CROPS) == 2 and nl.get(AS.S_NO_ADMISSION) == 2 and nl.get(AS.S_NOT_ADMITTED) == 2
          and nl.get(AS.S_NOT_KEPT) == 1 and nl.get(AS.S_NO_CROP_ROW) == 1 and nl.get("in_trusted") == 2, nl)
    check("the partial last queue line is not read (an append in progress)",
          res["inputs"]["step1_stream"]["queue"]["lines"] == len(
              (root() / "queue" / "queue.jsonl").read_bytes().split(b"\n")) - 1)


def t_originals(spaths, truth):
    print("rows of batches without crops.csv, found through their original (rejoin, backfill)")
    o = TMP / "og" / "a.json"
    res = AS.run_audit(TA_P0, [("Sorig", spaths["Sorig"])], o)
    a = res["audits"]["Sorig"]
    rows = {}
    for r in boxes_rows(o):
        rows.setdefault(r["key"], []).append(r)
    orig = truth["orig"]
    s1 = V.Crops()
    s1_id = {(s1.image[i], int(s1.box[i])): i for i in range(s1.n)}
    batch_id = {}
    for bid in ("b0001", "b0002"):
        cr = V.Crops(root() / "batches" / bid / "crops.csv")
        batch_id[bid] = {(cr.key[i], int(cr.box[i])): i for i in range(cr.n)}
    for key, (src, ref, keep, _verdicts, _labels) in sorted(orig["via"].items()):
        rs = rows[key]
        own = "b0000" if src == "step1" and "__rj" not in key else "b0004"
        want_ids = [s1_id[(ref, k)] if src == "step1" else batch_id[src][(ref, k)] for k in keep]
        check("%s: every label box judged on the crop of kept box keep[j] of its original (%s, crop_set "
              "stream/%s/via/%s)" % (key, src, own, src),
              len(rs) == len(keep) and all(r["status"] in A.JUDGED and r["matched_by"] == "stream"
                                           and r["crop_set"] == "stream/%s/via/%s" % (own, src)
                                           and int(r["crop_id"]) == w for r, w in zip(rs, want_ids)),
              [(r["status"], r["reason"], r["crop_set"], r["crop_id"]) for r in rs] + [want_ids])
    st = a["stream"]
    n_via = sum(len(v[2]) for v in orig["via"].values())
    check("the stream block counts the boxes found through their original (%d)" % n_via,
          sum(st["located_via_original"].values()) == n_via and st["located_via_original"].get(
              "b0004 via step1") == 3 and st["located_via_original"].get("b0000 via step1") == 2
          and st["located_via_original"].get("b0004 via b0001") == 2, st["located_via_original"])
    want_v = {}
    for key, (_src, _ref, keep, verdicts, labels) in orig["via"].items():
        for k in keep:
            want_v[verdicts[k]] = want_v.get(verdicts[k], 0) + 1
    got_v = dict(st["intake_verdicts"])
    for v_ in ("verified",):
        got_v[v_] = got_v.get(v_, 0) - 2            # slugR__mm_ok's two boxes (b0002's own crops, all verified)
    check("their intake verdicts are their own admission records' (the verdict under the label each admitted)",
          {k: v for k, v in got_v.items() if v} == want_v, (st["intake_verdicts"], want_v))
    check("their class at intake is the relabelled one (no label differs from intake)",
          st["label_differs_from_intake"] == 0, st["label_differs_from_intake"])
    check("the queue record counts the rows found through their original (step1 3, batch 3)",
          res["inputs"]["step1_stream"]["queue"]["rows_via_their_original"] == {"batch": 3, "step1": 3},
          res["inputs"]["step1_stream"]["queue"]["rows_via_their_original"])
    for key, cause in sorted(orig["defects"].items()):
        rs = rows[key]
        check("%s: %s -> %s" % (key, cause, AS.AUDIT_REASON[cause]),
              all(r["status"] == "no_embedding" and r["reason"] == AS.AUDIT_REASON[cause] for r in rs),
              [(r["status"], r["reason"]) for r in rs])
    for key, field in sorted(orig["mismatch"].items()):
        rs = rows[key]
        if field is None:
            check("%s: an admission record that describes its queue row: judged" % key,
                  all(r["status"] in A.JUDGED for r in rs), [(r["status"], r["reason"]) for r in rs])
        else:
            check("%s: an admission record whose %s disagrees with the queue row: not judged (no_crop_row), the "
                  "audit runs on" % (key, field), all(r["status"] == "no_embedding" and r["reason"] == "no_crop_row"
                                                        for r in rs), [(r["status"], r["reason"]) for r in rs])
    nl = st["not_located"]
    check("the stream block counts each cause (original_not_found 3, batch_without_crops 2, admission_mismatch 4)",
          nl.get(AS.S_SOURCE_MISSING) == 3 and nl.get(AS.S_NO_CROPS) == 2 and nl.get(AS.S_ADMISSION_MISMATCH) == 4,
          nl)
    check("coverage: why the species boxes were not judged",
          sum(a["coverage"]["species_not_judged"].values()) == a["coverage"]["species_boxes"]
          - a["coverage"]["species_judged"], a["coverage"])
    check("the MD notes the boxes found through their original", "Located through their original" in
          A._paths(o)[1].read_text())


def t_stream_root(spaths):
    print("the stream's Step 1 directory: --stream-dir, else the audited increments' stream.json, else the default")
    sdir = TMP / "inc" / "stream" / "s_t"
    (sdir / "increments").mkdir(parents=True, exist_ok=True)
    m = sdir / "increments" / "inc0001.jsonl"
    m.write_bytes(spaths["S1"].read_bytes())
    alt = TMP / "inc" / "elsewhere" / "step1_stream"
    shutil.copytree(str(root()), str(alt))
    V._write_json(sdir / "stream.json", {"sid": "s_t", "step1_stream": str(alt)})
    res = AS.run_audit(TA_P0, [("inc0001", m)], TMP / "sr" / "a.json")
    rec = res["inputs"]["step1_stream"]
    check("an increment of a stream naming its Step 1 directory: that directory, recorded with where it came from",
          rec["root"] == str(alt) and rec["root_from"] == "stream.json %s" % (sdir / "stream.json")
          and res["audits"]["inc0001"]["stream"]["boxes_located"] > 0, (rec["root"], rec["root_from"]))
    res = AS.run_audit(TA_P0, [("inc0001", m)], TMP / "sr" / "b.json", stream_dir=str(root()))
    check("--stream-dir wins over stream.json", res["inputs"]["step1_stream"]["root"] == str(root())
          and res["inputs"]["step1_stream"]["root_from"] == "--stream-dir")
    res = AS.run_audit(TA_P0, [("S1", spaths["S1"])], TMP / "sr" / "c.json")
    check("a manifest that is not a stream increment: the default directory",
          res["inputs"]["step1_stream"]["root"] == str(root())
          and res["inputs"]["step1_stream"]["root_from"] == "default")
    s2 = TMP / "inc" / "stream" / "s_u"
    (s2 / "increments").mkdir(parents=True, exist_ok=True)
    m2 = s2 / "increments" / "inc0002.jsonl"
    m2.write_bytes(spaths["S1"].read_bytes())
    V._write_json(s2 / "stream.json", {"sid": "s_u", "step1_stream": str(root())})
    check("increments of streams naming different directories: refused",
          refused(lambda: AS.run_audit(TA_P0, [("inc0001", m), ("inc0002", m2)], TMP / "sr" / "d.json"),
                  "name different Step 1 directories"))
    shutil.rmtree(str(alt), ignore_errors=True)


def t_refusals(spaths):
    print("refusals: a batch whose files do not describe its crops.csv")
    d = root() / "batches" / "b0001"
    keep = {f: (d / f).read_bytes() for f in ("crops_info.json", "emb.npz", "verdicts.npz", "batch.json",
                                              "crops_skipped.csv")}
    sha = C.sha256_file(d / "crops.csv")
    with np.load(d / "emb.npz") as z:
        emeta, ids, X = json.loads(str(z["meta"])), z["crop_ids"], z["X"]
    with np.load(d / "verdicts.npz") as z:
        vmeta, varr = json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}

    def recommit(name):
        """batch.json recording the file as it now is (a batch committed with it)."""
        bj = json.loads(keep["batch.json"])
        bj["inputs"][name]["sha256"] = C.sha256_file(d / {"emb": "emb.npz", "verdicts": "verdicts.npz"}[name])
        V._write_json(d / "batch.json", bj)

    def bj_crops():
        bj = json.loads(keep["batch.json"])
        bj["inputs"]["crops"]["sha256"] = "3" * 64
        V._write_json(d / "batch.json", bj)

    def emb(meta, ids_, X_):
        V._save_npz(d / "emb.npz", meta, crop_ids=ids_, X=X_)

    cases = [
        ("crops_info.json of another crops.csv",
         lambda: V._write_json(d / "crops_info.json", {"crops_sha256": "0" * 64}), "crops_info.json does not describe"),
        ("crops_skipped.csv changed after the crops were cut",
         lambda: (d / "crops_skipped.csv").write_bytes(keep["crops_skipped.csv"] + b"pool,x,0,small\r\n"),
         "crops_skipped.csv changed"),
        ("emb.npz of another crops.csv", lambda: (emb(dict(emeta, crops_sha256="1" * 64), ids, X), recommit("emb")),
         "emb.npz does not describe"),
        ("an embedding row missing", lambda: (emb(emeta, ids[1:], X[1:]), recommit("emb")),
         "emb.npz does not hold every crop"),
        ("another embedder", lambda: (emb(dict(emeta, embedder="another-model"), ids, X), recommit("emb")),
         "Step 1's by fake"),
        ("emb.npz changed after commit", lambda: emb(emeta, ids, X + np.float16(0.5)),
         "emb does not hash as batch.json recorded"),
        ("verdicts.npz of another crops.csv",
         lambda: (V._save_npz(d / "verdicts.npz", dict(vmeta, crops_sha256="2" * 64), **varr), recommit("verdicts")),
         "verdicts.npz does not describe"),
        ("verdicts.npz changed after commit",
         lambda: V._save_npz(d / "verdicts.npz", dict(vmeta, oof_judged=1), **varr),
         "verdicts does not hash as batch.json recorded"),
        ("crops.csv changed after commit", bj_crops, "crops does not hash as batch.json recorded"),
    ]
    out = TMP / "rf" / "a.json"
    for name, mutate, needle in cases:
        mutate()
        ok = refused(lambda: AS.run_audit(TA_P0, [("S1", spaths["S1"])], out), needle)
        for f, b in keep.items():
            (d / f).write_bytes(b)
        check("refused: %s" % name, ok)
    check("no output written by a refused run", not out.exists())
    check("the batch's files are back as written", C.sha256_file(d / "crops.csv") == sha
          and all((d / f).read_bytes() == b for f, b in keep.items()))
    check("--out under the stream's directory refused",
          refused(lambda: AS.run_audit(TA_P0, [("S1", spaths["S1"])], root() / "x" / "a.json"),
                  "stream's Step 1 directory") and not (root() / "x").exists())


def t_determinism_cli(spaths, first):
    print("determinism and the CLI")
    o = TMP / "st" / "new.json"
    again = AS.run_audit(TA_P0, [("S1", spaths["S1"]), ("S1swap", spaths["S1swap"])], o)
    a, b = dict(first), dict(again)
    for x in (a, b):
        x.pop("built_utc", None)
        x.pop("seconds", None)
    check("a rerun over its own outputs gives the same document", json.dumps(a, sort_keys=True)
          == json.dumps(b, sort_keys=True))
    out = TMP / "cli" / "a.json"
    rc = AS.main(["--trusted", str(TA_P0), "--audit", "S1=%s" % spaths["S1"], "--out", str(out),
                  "--stream-dir", str(root())])
    rc2 = AS.main(["--trusted", str(TA_P0), "--audit", "S1=%s" % spaths["S1"], "--out", str(TMP / "cli" / "b.json"),
                   "--stream-dir", str(TMP / "nowhere")])
    r2 = json.loads((TMP / "cli" / "b.json").read_text())
    check("main: --stream-dir; exit 0", rc == 0 and json.loads(out.read_text())["audits"]["S1"]["stream"]
          ["boxes_located"] > 0)
    check("main: a stream directory that is not there judges no stream box (not an error)",
          rc2 == 0 and r2["inputs"]["step1_stream"]["present"] is False
          and r2["audits"]["S1"]["species"]["boxes_judged"] == 0)
    check("main: a bad spec exits 2", AS.main(["--trusted", str(TA_P0), "--audit", "S1", "--out", str(out)]) == 2)
    sh = TA.ROOT / "run_inc_audit.sh"
    body = sh.read_text()
    mods = body.split("MODULES=(", 1)[1].split(")", 1)[0]
    check("run_inc_audit.sh runs this module with its arguments and hashes it with inc/audit.py",
          '-m weed_optimizer_framework.tools.inc2.audit_stream "$@"' in body
          and "tools/inc2/audit_stream.py" in mods and "tools/inc/audit.py" in mods and "tools/inc2/__init__.py" in mods)
    lines = body.splitlines()
    check("run_inc_audit.sh keeps the pilot command on lines 30-36 (the policy row cites them)",
          lines[29].strip().startswith("#   REPO=") and "--out" in lines[35])
    dep = (TA.ROOT / "deploy" / "deploy_funnel.sh").read_text()
    pkg = dep.split("PKG_PATHS=(", 1)[1].split("\n)\n", 1)[0]
    check("deploy_funnel.sh ships run_inc_audit.sh (and inc2, where this module lives)",
          "run_inc_audit.sh" in [x for ln in pkg.splitlines() for x in ln.split("#", 1)[0].split()]
          and "weed_optimizer_framework/tools/inc2" in pkg)


TA_P0 = None


def main():
    global TA_P0
    try:
        paths, truth0 = TA.build_world()
        TA_P0 = paths["P0"]
        p0_rows = C.read_manifest(paths["P0"])
        paths["pool_rows"] = C.read_manifest(V.POOL)
        spaths, truth, defects = build_stream(p0_rows, paths)
        t_equivalence(paths)
        t_trusted_step1_only(paths, spaths, None)
        first = t_stream(spaths, truth)
        t_defects(spaths, defects)
        t_originals(spaths, truth)
        t_stream_root(spaths)
        t_refusals(spaths)
        t_determinism_cli(spaths, first)
    except Exception as e:                       # noqa: BLE001
        import traceback
        check("the test ran", False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-3000:]))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
