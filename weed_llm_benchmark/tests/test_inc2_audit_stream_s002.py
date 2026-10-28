#!/usr/bin/env python3
"""The label audit of s002's inc0005 on the cluster's own records (tests/fixtures/audit_stream_s002/).

On 2026-10-27 the platform's L4 (inc/audit.py) judged 8 of inc0005's 1,784 species-labelled boxes (91 of its 1,867
boxes): its 7 images Step 1 holds; the other 1,357 images (1,776 boxes) were image_not_in_crops. This test rebuilds
what the stream lookup reads for inc0005 from the cluster's records and holds the new module to them:

  * the fixture is what it was extracted as: every file's sha256 (FIXTURE.json), every label file hashing to its
    manifest row's label_sha256, every manifest row matching one queue row (key, image, sha256, label_sha256), the
    copied crop rows of each batch from a crops.csv whose sha256 batch.json recorded at commit;
  * on it, inc/audit.py judges what the platform's record of 2026-10-27 says it judged (8 species boxes, 91 boxes,
    1,776 image_not_in_crops); inc2.audit_stream judges (near) all: every species box the stream cropped, the counts
    equal to the lab run on full copies of the cluster's files (lab_run in FIXTURE.json, when present);
  * D31 reads the old record as not judged (it predates the stream lookup) and the new one as judged, with no
    coverage shortfall.

What is real: the manifest, its label files, the queue rows, the batches' crop rows (geometry, class at intake,
original crop ids), the boxes the intake left out, the admission records, each crop's intake verdict and whether its
embedding is finite, the 7 Step 1 images' pool rows and crop rows, the old record. What is not: the embedding
vectors (2.7 MB; a stand-in per crop: its class's direction plus noise, NaN where the real one is not finite) and the
trusted set (a synthetic train_core of all 12 species; the real probe's 5,151 trusted boxes are not copied), so this
test pins coverage, not verdicts. The audit of the real vectors is the lab run (docs/CONTINUOUS_LOOP.md, Amendment
2026-10-27).

Run:  python3 tests/test_inc2_audit_stream_s002.py
"""
import csv
import gzip
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_audit_s002_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "audit_stream_s002"

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import audit as A  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import audit_stream as AS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402

assert C.INC_DIR == TMP / "inc", "fake INC_DIR not in effect"
FAILURES = []
D = 40
NOISE = 0.7
REC = json.loads((FIX / "FIXTURE.json").read_text())


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raw(name):
    return gzip.decompress((FIX / name).read_bytes())


def sha(b):
    return hashlib.sha256(b).hexdigest()


class FakeEv(object):
    def __init__(self, docs):
        self.docs = docs

    def json(self, name):
        return self.docs.get(name)

    def cite(self, name, ptr):
        return {"artifact": name, "pointer": ptr}


# ------------------------------------------------------------ the fixture
def t_fixture():
    print("the fixture is the cluster's records as extracted")
    bad = [f for f, s in REC["files"].items() if sha((FIX / f).read_bytes()) != s]
    bad += [f for f, x in (REC.get("lab_files") or {}).items() if sha((FIX / f).read_bytes()) != x["sha256"]]
    check("every fixture file hashes as FIXTURE.json recorded it (%d from the cluster, %d from the lab run)"
          % (len(REC["files"]), len(REC.get("lab_files") or {})), not bad, bad)
    man = raw("inc0005.jsonl.gz")
    check("the manifest is inc0005's bytes (sha256 %s...)" % REC["sources"]["inc0005"]["sha256"][:12],
          sha(man) == REC["sources"]["inc0005"]["sha256"])
    rows = [json.loads(x) for x in man.decode().splitlines() if x.strip()]
    labels = json.loads(raw("labels.json.gz"))
    check("every label file hashes to its row's label_sha256 (%d rows)" % len(rows),
          len(rows) == 1364 and all(sha(labels[r["label"]].encode()) == r["label_sha256"] for r in rows))
    q = [json.loads(x) for x in raw("queue_rows.jsonl.gz").decode().splitlines() if x.strip()]
    qk = {(x["key"], x["image"], x["sha256"], x["label_sha256"]) for x in q}
    check("every row has its queue row (key, image, sha256, label_sha256)",
          all((r["key"], r["image"], r["sha256"], r["label_sha256"]) in qk for r in rows))
    ok = all(b.get("crops.csv") is None or b["crops.csv"] == ((b.get("batch_json") or {}).get("inputs") or {}).get(
        "crops", {}).get("sha256") for b in REC["batches"].values())
    check("each batch's crops.csv is the one its batch.json recorded at commit", ok)
    return rows, labels, q


# --------------------------------------------------------------- the world
def class_dirs():
    rng = np.random.default_rng(20261027)
    basis = np.linalg.qr(rng.normal(size=(D, D)))[0]
    return basis[:C.NC], basis[C.NC:C.NC + 3]


def write_csv(path, header, rows):
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\r\n")
    wr.writerow(header)
    wr.writerows(rows)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue().encode())


def build(rows, labels):
    """Step 1 (a synthetic train_core of 12 sessions and the 7 real pool images inc0005 shares with it), the
    stream's batches as the fixture holds them, and inc0005 with its label files laid out under TMP."""
    rng = np.random.default_rng(5)
    cdir, odir = class_dirs()

    def feat(k):
        v = cdir[k] if k < C.OTHER_PLANT else odir[int(rng.integers(0, 3))]
        return v + NOISE * rng.normal(size=D) / np.sqrt(D)

    # label files, the manifest's paths rewritten under TMP (bytes and sha256 unchanged)
    man = []
    for r in rows:
        lp = TMP / "labels" / r["label"].lstrip("/")
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(labels[r["label"]])
        man.append(dict(r, label=str(lp)))
    mp = TMP / "inc" / "s002" / "manifests" / "inc0005.jsonl"
    C.write_manifest(mp, man)
    # Step 1
    core, crops, X = [], [], []
    for s in range(12):
        sess = "20210%02d01_S%02d" % (s % 9 + 1, s)
        for i in range(15):
            key = "core_s%02d_%03d" % (s, i)
            boxes = [(int((i * 4 + b + s) % 12),) + tuple(float("%.6f" % x) for x in
                                                           (rng.uniform(.2, .8), rng.uniform(.2, .8),
                                                            rng.uniform(.05, .3), rng.uniform(.05, .3)))
                     for b in range(4)]
            lp = TMP / "repo" / "cwd12" / "labels" / (key + ".txt")
            C.write_yolo(lp, boxes)
            img = TMP / "repo" / "cwd12" / "train" / "images" / (key + ".jpg")
            core.append({"image": str(img), "label": str(lp), "sha256": C.sha256_text("img:" + key),
                         "label_sha256": C.sha256_file(lp), "source": "cottonweeddet12/train", "session": sess,
                         "key": key})
            for b, box in enumerate(boxes):
                crops.append([len(crops), "core", key, str(img), "train_core", sess, b] + ["%.6f" % x for x in box[1:]]
                             + [640, 480, box[0], C.CLASS_NAMES[box[0]]])
                X.append(feat(box[0]))
    C.write_manifest(C.manifest_path("train_core"), core)
    V.STEP1.mkdir(parents=True, exist_ok=True)
    V.POOL.write_bytes(raw("step1/pool_rows.jsonl.gz"))
    V.POOL_META.write_bytes(b"")
    V.COPIES.write_bytes(b"")
    s1 = raw("step1/crops_rows.csv.gz").decode().splitlines()
    for t in csv.reader(s1[1:]):
        crops.append([len(crops)] + t[1:])
        X.append(feat(int(t[13])))
    write_csv(V.CROPS, V.CROP_FIELDS, crops)
    sk = raw("step1/skipped_rows.csv.gz").decode().splitlines()
    write_csv(V.CROPS_SKIPPED, V.SKIPPED_FIELDS, list(csv.reader(sk[1:])))
    V._write_json(V.CROPS_INFO, {"crops_sha256": C.sha256_file(V.CROPS), "inputs": V._input_hashes(),
                                 "skipped_sha256": C.sha256_file(V.CROPS_SKIPPED)})
    Xa = np.asarray(X, dtype=np.float16)
    V.EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(V.EMB_DIR / "emb_s000_of_001.npz",
             meta=np.array(json.dumps({"crops_sha256": C.sha256_file(V.CROPS), "embedder": "fake", "dim": D,
                                       "crops": len(crops), "stats": {}})),
             crop_ids=np.arange(len(crops)), X=Xa)
    p0 = TMP / "inc" / "s002" / "manifests" / "P0.jsonl"
    C.write_manifest(p0, core)
    # the stream
    root = AS.default_root()
    (root / "queue").mkdir(parents=True, exist_ok=True)
    (root / "queue" / "queue.jsonl").write_bytes(raw("queue_rows.jsonl.gz"))
    for bid, b in sorted(REC["batches"].items()):
        if not b.get("crops.csv"):
            continue
        d = root / "batches" / bid
        d.mkdir(parents=True, exist_ok=True)
        rows_ = list(csv.reader(raw("batches/%s/crops_rows.csv.gz" % bid).decode().splitlines()[1:]))
        intake = json.loads(raw("batches/%s/intake.json.gz" % bid))
        check("  %s: the copied crop rows are those the intake verdicts and finiteness describe (%d of %d)"
              % (bid, len(rows_), b["crops_rows"]), [int(t[0]) for t in rows_] == intake["crop_ids"])
        write_csv(d / "crops.csv", V.CROP_FIELDS, [[i] + t[1:] for i, t in enumerate(rows_)])
        skr = list(csv.reader(raw("batches/%s/skipped_rows.csv.gz" % bid).decode().splitlines()[1:]))
        write_csv(d / "crops_skipped.csv", V.SKIPPED_FIELDS, skr)
        csha = C.sha256_file(d / "crops.csv")
        V._write_json(d / "crops_info.json", {"crops": len(rows_), "crops_sha256": csha,
                                              "skipped_sha256": C.sha256_file(d / "crops_skipped.csv")})
        Xb = np.asarray([feat(int(t[13])) if fin else np.full(D, np.nan)
                         for t, fin in zip(rows_, intake["finite"])], dtype=np.float16).reshape(len(rows_), D)
        V._save_npz(d / "emb.npz", {"crops_sha256": csha, "embedder": "fake", "dim": D, "crops": len(rows_),
                                    "stats": {}}, crop_ids=np.arange(len(rows_)), X=Xb)
        vm = dict(b["verdicts_meta"], crops_sha256=csha)
        n = len(rows_)
        V._save_npz(d / "verdicts.npz", vm, crop_id=np.arange(n), verdict=np.asarray(intake["verdict"], np.int8),
                    pred=np.zeros(n, np.int64), p=np.ones(n), cosine=np.ones(n))
        (d / "admission.jsonl").write_bytes(raw("batches/%s/admission_rows.jsonl.gz" % bid))
        V._write_json(d / "batch.json", {"kind": b["batch_json"]["kind"], "input": b["batch_json"]["input"],
                                         "inputs": {k: {"sha256": C.sha256_file(d / f)} for k, f in
                                                    (("crops", "crops.csv"), ("emb", "emb.npz"),
                                                     ("verdicts", "verdicts.npz"))}})
    return p0, mp


# ------------------------------------------------------------------ tests
def t_audit(p0, mp):
    print("inc0005: inc/audit.py as the platform ran it, then with the stream's lookup")
    old_rec = json.loads(raw("label_audit_s002_2026-10-27.json.gz"))["audits"]["inc0005"]
    o1, o2 = TMP / "out" / "old" / "label_audit.json", TMP / "out" / "new" / "label_audit.json"
    old = A.run_audit(p0, [("inc0005", mp)], o1)["audits"]["inc0005"]
    new = AS.run_audit(p0, [("inc0005", mp)], o2)
    a = new["audits"]["inc0005"]

    def judged(blk):
        return sum(blk["boxes_by_status"][s] for s in A.JUDGED)
    check("the platform's record of 2026-10-27: 8 species boxes judged, 91 of 1,867 boxes, 1,776 image_not_in_crops",
          old_rec["species"]["boxes_judged"] == 8 and judged(old_rec) == 91 and old_rec["boxes"] == 1867
          and old_rec["no_embedding_reasons"] == {"image_not_in_crops": 1776})
    check("inc/audit.py on the fixture judges the same boxes (8 species, 91 in all, 1,776 image_not_in_crops)",
          old["species"]["boxes_judged"] == 8 and judged(old) == 91
          and old["no_embedding_reasons"] == {"image_not_in_crops": 1776},
          (old["species"]["boxes_judged"], judged(old), old["no_embedding_reasons"]))
    cv, st = a["coverage"], a["stream"]
    check("with the stream: %d of %d species boxes judged (%d of %d boxes); not judged %s"
          % (cv["species_judged"], cv["species_boxes"], judged(a), a["boxes"], a["no_embedding_reasons"] or "none"),
          cv["species_boxes"] == 1784 and cv["species_judged"] >= 0.9 * 1784 and judged(a) >= 0.9 * 1867, cv)
    check("the stream located the 1,357 images' boxes in their batches (%s)" % st["per_batch"],
          a["images_in_stream"] == 1357 and set(st["per_batch"]) <= set(REC["batches"]), st)
    lab = REC.get("lab_run")
    if lab:
        got = {"species_boxes": cv["species_boxes"], "species_judged": cv["species_judged"], "boxes_judged": judged(a),
               "no_embedding_reasons": a["no_embedding_reasons"], "per_batch": st["per_batch"],
               "not_located": st["not_located"]}
        want = {k: lab[k] for k in got}
        check("the same coverage as the lab run on full copies of the cluster's files (%s)" % lab.get("run"),
              got == want, (got, want))
    return json.loads(raw("label_audit_s002_2026-10-27.json.gz")), new


def t_d31(old, new):
    print("D31 on the two records")
    r_old = DS._audit_result(None, FakeEv({"x/audit/label_audit.json": old}), "x", ["inc0005"], 0.5, 0.9)[0]
    r_new = DS._audit_result(None, FakeEv({"x/audit/label_audit.json": new}), "x", ["inc0005"], 0.5, 0.9)[0]
    check("the record of 2026-10-27: not judged (8 of 1,784 species boxes), predates the stream lookup",
          r_old["verdict"] == "not_judged" and r_old["predates_stream"] and r_old["steps"]["inc0005"]["judged"] == 8
          and r_old["steps"]["inc0005"]["species_boxes"] == 1784, r_old)
    check("the stream lookup's record: judged (no longer 'not_judged'), no coverage shortfall",
          r_new["verdict"] in ("labels", "not_labels") and r_new["coverage_short"] == [] and r_new["stream_lookup"],
          (r_new["verdict"], r_new["coverage_short"], r_new["steps"]))


def main():
    try:
        rows, labels, _q = t_fixture()
        p0, mp = build(rows, labels)
        old, new = t_audit(p0, mp)
        t_d31(old, new)
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
