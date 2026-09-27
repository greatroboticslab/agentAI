#!/usr/bin/env python3
"""INC attribution item 4: the BioCLIP-2 label audit (inc/audit.py).

What is pinned, and why:
  * a copy of a clean manifest with 40 % of its boxes swapped to another
    species (pilot.make_bswap, the pilot's own corruption) gets a conflict
    rate close to 0.4 times the baseline's conflict recall, and a corrected
    noise estimate close to the planted share; its known-truth block counts
    the planted swaps exactly;
  * the clean manifest stays at the held-out CV baseline;
  * a harvested-style manifest with 25 % wrong species labels is estimated at
    about 0.25; its OtherPlant boxes of other plants are other_ok, and its
    OtherPlant boxes that hold a cwd12 species are conflicts;
  * in a noisier world (held-out false-conflict rate f about 0.13) the
    correction removes f: clean sets come out near 0 although their conflict
    rate is above 0.1, and the swap copy near 0.4 where c and c / r are not;
  * in a world where two confusable species are rare in P0 and make up most
    of the other sessions, a clean set of those sessions is clean against the
    baseline for its own species mix, where the baseline pooled over P0's mix
    would call it noisy; the swap copy of those sessions is still caught;
  * the probe is fitted on the trusted manifest's boxes only, with the labels
    of the trusted manifest's own label files: a trusted copy that relabels a
    species leaves the probe without it, and audited boxes labelled with it
    are label_unseen; the probe and the baseline do not depend on what is
    audited;
  * refusals, never guesses: a box whose label geometry differs from its crop
    row's by more than 1e-4 (and not one within it), a NaN feature row, a box
    left out as small, a box with no crop row, an image not among the Step 1
    images, an image whose bytes changed; the same bytes under another path
    are found by sha256;
  * an image of the trusted manifest is refused by each route on its own:
    the same path, the same sha256, a cwd12 copy of a trusted train_core
    image and the train_core twin of a trusted copy; a copy of an untrusted
    image is judged on its copy crop;
  * hard errors: a label file that changed after its manifest, crops.csv
    inputs changed since `verify crops`, a trusted set too small to
    cross-validate, bad --audit specs; an --out that does not end in .json,
    lies under the Step 1 directory, is an input, or would overwrite a file
    that is not an earlier audit output (the Step 1 files stay untouched);
  * the audit is deterministic and may rerun over its own outputs; verify
    prints no OtherPlant threshold warning for the probe without that class;
    the CLI and the sbatch script parse.

The world is synthetic, written in inc/verify.py's own formats (crops.csv,
crops_skipped.csv, crops_info.json, emb_sNNN_of_NNN.npz, pool.jsonl,
pool_meta.jsonl, cwd12_copies.jsonl, the train_core manifest) under a temp
INC_DIR, with two cwd12 copies (of a P0 and an I1 image). A box's feature is
its TRUE class's direction plus noise (other plants: directions of their
own), so it clusters by the truth whatever the label says. No image is
opened; no open_clip, no network.

Run:  python3 tests/test_inc_audit.py
"""
import contextlib
import csv
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_audit_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import audit as A  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import pilot as P  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402

assert C.INC_DIR == TMP / "inc" and str(V.STEP1).startswith(str(TMP)), "fake INC_DIR not in effect"

FAILURES = []


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
        return needle in str(e)
    return False


# ------------------------------------------------------------- the world
D = 40
NOISE = 0.7
N_SESS, IMGS, BOXES = 20, 20, 4
P0_SESS, I1_SESS, SW_SESS = range(0, 10), range(10, 14), range(14, 20)
N_POOL = 60
POOL_WRONG = 0.25
EXP = "pilot_t"


def _geom(rng):
    return tuple(float("%.6f" % x) for x in (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
                                             rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3)))


def _row(key, image, label, source, session, sha=None):
    return {"image": str(image), "label": str(label), "sha256": sha or C.sha256_text("img:" + key),
            "label_sha256": C.sha256_file(label), "source": source, "session": session, "key": key}


def build_world(noise=NOISE, hard=None):
    """Step 1 outputs in verify's formats plus the manifests to audit.
    Returns the manifest paths and the planted truth.

    noise: the feature noise (0.7: the probe almost never errs; 2.2: CV top-1
    about 0.85 and a held-out false-conflict rate above 0.1). hard = (a, b,
    pull, w_p0, w_rest): species a's direction is moved next to species b's
    (a confusable pair), and a box is w_p0 times as likely as another
    species' to be a or b in the P0 sessions, w_rest times in the others."""
    rng = np.random.default_rng(20260927)
    basis = np.linalg.qr(rng.normal(size=(D, D)))[0]
    class_dir, other_dir = basis[:C.NC].copy(), basis[C.NC:C.NC + 3]
    if hard:
        a, b = hard[0], hard[1]
        class_dir[a] = class_dir[b] + hard[2] * class_dir[a]
        class_dir[a] /= np.linalg.norm(class_dir[a])

    def feat(v):
        return v + noise * rng.normal(size=D) / np.sqrt(D)

    shutil.rmtree(TMP / "inc", ignore_errors=True)
    shutil.rmtree(TMP / "repo", ignore_errors=True)
    crops, emb, skipped = [], [], []       # crops: [set, key, image, source, group, box, geom, label]
    lab_core = TMP / "repo" / "cwd12" / "labels"
    img_core = TMP / "repo" / "cwd12" / "train" / "images"

    core, by_sess = [], {}
    for s in range(N_SESS):
        sess = "202108%02d_iPhoneSE_S%02d" % (s + 1, s)
        pw = None
        if hard:
            pw = np.ones(12)
            pw[[hard[0], hard[1]]] = hard[3] if s in P0_SESS else hard[4]
            pw /= pw.sum()
        for i in range(IMGS):
            key = "core_s%02d_%03d" % (s, i)
            boxes = [(int(rng.integers(0, 12)) if pw is None else int(rng.choice(12, p=pw)),)
                     + _geom(rng) for _ in range(BOXES)]
            lp = lab_core / (key + ".txt")
            C.write_yolo(lp, boxes)
            r = _row(key, img_core / (key + ".jpg"), lp, "cottonweeddet12/train", sess)
            core.append(r)
            by_sess.setdefault(s, []).append(r)
            for b, box in enumerate(boxes):
                if key == "core_s10_003" and b == 2:          # an I1 box too small to crop
                    skipped.append(["core", key, b, "small"])
                    continue
                crops.append(["core", key, r["image"], "train_core", sess, b, box[1:], box[0]])
                emb.append(np.full(D, np.nan) if (key == "core_s11_004" and b == 1)   # failed crop
                           else feat(class_dir[box[0]]))
    C.write_manifest(C.manifest_path("train_core"), core)

    # the pool: two harvested slugs; boxes of a species, of another plant, or
    # of a species under an OtherPlant label (a join that lost the species)
    pool, meta, pool_truth = [], [], {}
    for i in range(N_POOL):
        slug = "slugA" if i % 2 else "slugB"
        key = "%s__img_%03d" % (slug, i)
        truth, boxes = [], []
        for _ in range(BOXES):
            u = rng.random()
            g = _geom(rng)
            if u < 0.70:
                k = int(rng.integers(0, 12))
                truth.append(("sp", k))
                boxes.append((k,) + g)
            elif u < 0.90:
                truth.append(("other", int(rng.integers(0, 3))))
                boxes.append((C.OTHER_PLANT,) + g)
            else:
                k = int(rng.integers(0, 12))
                truth.append(("hidden", k))
                boxes.append((C.OTHER_PLANT,) + g)
        lp = V.LABELS_DIR / slug / (key + ".txt")
        C.write_yolo(lp, boxes)
        r = _row(key, TMP / "repo" / "datasets" / slug / "images" / ("img_%03d.jpg" % i), lp, slug, "")
        pool.append(r)
        meta.append({"key": key, "source": slug, "W": 640, "H": 480,
                     "dhash": int(rng.integers(0, 2 ** 62)),
                     "boxes": [list(b) for b in boxes], "src": [[b[0], "x"] for b in boxes]})
        pool_truth[key] = (truth, boxes)
        for b, (box, t) in enumerate(zip(boxes, truth)):
            crops.append(["pool", key, r["image"], slug, slug, b, box[1:], box[0]])
            emb.append(feat(other_dir[t[1]] if t[0] == "other" else class_dir[t[1]]))
    C.write_manifest(V.POOL, pool)
    V._write_jsonl(V.POOL_META, sorted(meta, key=lambda m: m["key"]))

    # cwd12 copies (verify pool: a harvested image within 6 dHash bits of a
    # train_core image; other path, other bytes): one of a P0 image, one of an
    # I1 image. Their features follow the train_core twin's true classes.
    crng = np.random.default_rng(7)
    copies = []
    for n, src in enumerate((by_sess[P0_SESS[0]][5], by_sess[I1_SESS[0]][5])):
        key = "slugC__cp_%03d" % n
        boxes = C.read_yolo(src["label"])
        lp = V.COPY_LABELS_DIR / "slugC" / (key + ".txt")
        C.write_yolo(lp, boxes)
        r = _row(key, TMP / "repo" / "datasets" / "slugC" / "images" / ("cp_%03d.jpg" % n), lp,
                 "slugC", "")
        r.update({"W": 640, "H": 480, "dhash": int(crng.integers(0, 2 ** 62)), "bits": 3,
                  "src": [[b[0], C.CLASS_NAMES[b[0]]] for b in boxes], "old_join": None,
                  "calibration_only": False, "train_core_key": src["key"],
                  "train_core_image": src["image"], "train_core_label": src["label"],
                  "train_core_session": src["session"]})
        copies.append(r)
        for b, box in enumerate(boxes):
            crops.append(["copy", key, r["image"], "slugC", "slugC", b, box[1:], box[0]])
            emb.append(class_dir[box[0]] + noise * crng.normal(size=D) / np.sqrt(D))
    V._write_jsonl(V.COPIES, copies)

    with open(V.CROPS, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        for cid, (st, key, image, source, group, b, g, lab) in enumerate(crops):
            wr.writerow([cid, st, key, image, source, group, b] + ["%.6f" % x for x in g]
                        + [640, 480, lab, C.CLASS_NAMES[lab]])
    with open(V.CROPS_SKIPPED, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.SKIPPED_FIELDS)
        wr.writerows(skipped)
    V._write_json(V.CROPS_INFO, {"crops_sha256": C.sha256_file(V.CROPS), "inputs": V._input_hashes(),
                                 "skipped_sha256": C.sha256_file(V.CROPS_SKIPPED)})
    X = np.asarray(emb, dtype=np.float16)
    ids = np.arange(len(crops))
    perm = rng.permutation(len(ids))                # shards hold crops in any order
    V.EMB_DIR.mkdir(parents=True, exist_ok=True)
    crops_sha = C.sha256_file(V.CROPS)
    for s, part in enumerate(np.array_split(perm, 2)):
        np.savez(V.EMB_DIR / ("emb_s%03d_of_002.npz" % s),
                 meta=np.array(json.dumps({"crops_sha256": crops_sha, "embedder": "fake", "dim": D,
                                           "crops": int(len(part)), "stats": {}})),
                 crop_ids=ids[part], X=X[part])

    # the manifests to audit
    mdir = TMP / "inc" / EXP / "manifests"
    rows = {name: [r for s in ss for r in by_sess[s]]
            for name, ss in (("P0", P0_SESS), ("I1", I1_SESS), ("swap_src", SW_SESS))}
    paths = {}
    for name in ("P0", "I1"):
        paths[name] = mdir / ("%s.jsonl" % name)
        C.write_manifest(paths[name], rows[name])
    bswap_rows, bswap = P.make_bswap(EXP, rows["swap_src"], TMP / "inc" / EXP / "labels" / "Bswap")
    paths["Bswap"] = mdir / "Bswap.jsonl"
    C.write_manifest(paths["Bswap"], bswap_rows)
    paths["Bswap_clean"] = mdir / "Bswap_clean.jsonl"      # the same images, their own labels
    C.write_manifest(paths["Bswap_clean"], rows["swap_src"])

    # Breal-style: the pool images with the pilot's join labels, 25 % of the
    # species boxes given another species
    brng = np.random.default_rng(99)
    sp_boxes = [(r["key"], b) for r in pool for b, t in enumerate(pool_truth[r["key"]][0])
                if t[0] == "sp"]
    n_sp, n_wrong = len(sp_boxes), int(round(POOL_WRONG * len(sp_boxes)))
    wrong = {sp_boxes[i] for i in brng.choice(n_sp, size=n_wrong, replace=False)}
    breal = []
    for r in pool:
        truth, boxes = pool_truth[r["key"]]
        new = [((k + int(brng.integers(1, 12))) % 12,) + box[1:] if (r["key"], b) in wrong else box
               for b, ((_kind, k), box) in enumerate(zip(truth, boxes))]
        lp = TMP / "inc" / EXP / "labels" / "Breal" / ("Breal__%s.txt" % r["key"])
        C.write_yolo(lp, new)
        breal.append(_row("Breal__" + r["key"], r["image"], lp, r["source"], "", sha=r["sha256"]))
    paths["Breal"] = mdir / "Breal.jsonl"
    C.write_manifest(paths["Breal"], breal)

    # defects: every way a box can lack an embedding to judge it on
    ddir = TMP / "inc" / EXP / "labels" / "defects"
    defects = []

    def add(key, image, boxes, sha):
        lp = ddir / (key + ".txt")
        C.write_yolo(lp, boxes)
        defects.append(_row(key, image, lp, "defects", "", sha=sha))

    p = {r["key"]: r for r in pool}
    b0 = pool_truth["slugB__img_000"][1]
    add("moved", "/elsewhere/img_000.jpg", b0, p["slugB__img_000"]["sha256"])       # same bytes
    add("changed", p["slugA__img_001"]["image"], pool_truth["slugA__img_001"][1], "f" * 64)
    add("absent", "/nowhere/x.jpg", b0, "e" * 64)
    g2 = [tuple(b) for b in pool_truth["slugB__img_002"][1]]
    g2[0] = (g2[0][0], g2[0][1] + 0.01) + g2[0][2:]                                 # moved box
    g2[1] = (g2[1][0], g2[1][1] + 0.00005) + g2[1][2:]                              # within tolerance
    add("geometry", p["slugB__img_002"]["image"], g2, p["slugB__img_002"]["sha256"])
    add("extra", p["slugA__img_003"]["image"],
        pool_truth["slugA__img_003"][1] + [(0, 0.5, 0.5, 0.1, 0.1)], p["slugA__img_003"]["sha256"])
    paths["defects"] = mdir / "defects.jsonl"
    C.write_manifest(paths["defects"], defects)

    # overlap: I1 plus three images of the trusted P0 (by path), a P0 image
    # under another path (by sha256), the cwd12 copy of a P0 image (the same
    # photograph: other path, other bytes) and the copy of an I1 image
    moved = dict(rows["P0"][3], image="/elsewhere/%s.jpg" % rows["P0"][3]["key"],
                 key="moved_" + rows["P0"][3]["key"])
    cp = {c["train_core_key"]: {k: c[k] for k in C.MANIFEST_KEYS} for c in copies}
    paths["overlap"] = mdir / "overlap.jsonl"
    C.write_manifest(paths["overlap"], rows["I1"] + rows["P0"][:3] + [moved]
                     + [cp[by_sess[P0_SESS[0]][5]["key"]], cp[by_sess[I1_SESS[0]][5]["key"]]])

    # a trusted copy whose labels call every CutleafGroundcherry box Goosegrass
    tdir = TMP / "inc" / EXP / "labels" / "P0_relabelled"
    relab = []
    for r in rows["P0"]:
        boxes = [(10 if b[0] == 11 else b[0],) + b[1:] for b in C.read_yolo(r["label"])]
        lp = tdir / (r["key"] + ".txt")
        C.write_yolo(lp, boxes)
        relab.append(dict(r, label=str(lp), label_sha256=C.sha256_file(lp)))
    paths["P0_relabelled"] = mdir / "P0_relabelled.jsonl"
    C.write_manifest(paths["P0_relabelled"], relab)

    # a trusted set of two sessions: too small to cross-validate the baseline
    paths["P0_two"] = mdir / "P0_two.jsonl"
    C.write_manifest(paths["P0_two"], by_sess[0] + by_sess[1])
    return paths, {"bswap": bswap, "breal_species": n_sp, "breal_wrong": n_wrong, "rows": rows,
                   "pool_truth": pool_truth, "copies": copies, "moved": moved}


def strip(res):
    return {k: v for k, v in res.items() if k not in ("built_utc", "seconds", "outputs")}


def boxes_csv(out):
    with open(A._paths(out)[2], newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ tests
def test_audit(paths, truth):
    print("audit on the synthetic world")
    audits = [(n, paths[n]) for n in ("I1", "Bswap", "Bswap_clean", "Breal", "defects", "overlap")]
    out = TMP / "out1" / "audit.json"
    res = A.run_audit(paths["P0"], audits, out)
    for p in A._paths(out):
        check("output %s written" % p.name, p.is_file())
    bl = res["baseline_cv"]
    f = bl["species"]["false_conflict_rate"]
    r = bl["swapped"]["conflict_recall_on_wrong"]
    check("baseline: 5 folds over the P0 sessions, every box judged", bl["folds"] == 5
          and bl["species"]["boxes_judged"] == res["trusted"]["species_boxes_used"], bl["species"])
    check("baseline: false conflicts are rare and a wrong species is caught", f <= 0.03 and r >= 0.85,
          (f, r))
    t = res["trusted"]
    check("trusted: every P0 box fits the probe, grouped by session",
          t["species_boxes_used"] == len(P0_SESS) * IMGS * BOXES and t["groups"] == len(P0_SESS)
          and t["grouped_by"] == {"session": t["species_boxes_used"], "source": 0}, t)

    sw = res["audits"]["Bswap"]
    share = truth["bswap"]["changed"] / float(truth["bswap"]["boxes"])
    c = sw["species"]["conflict_rate"]
    check("Bswap: 40 % of the boxes were swapped (pilot.make_bswap)", abs(share - 0.4) < 0.01, share)
    check("Bswap: conflict rate close to 0.4 x the conflict recall", abs(c - share * r) < 0.06
          and 0.3 < c < 0.46, (c, share, r))
    check("Bswap: corrected noise estimate close to the planted share",
          abs(sw["species"]["noise_estimate_corrected"] - share) < 0.06,
          (sw["species"]["noise_estimate_corrected"], share))
    check("Bswap: above the baseline", sw["species"]["above_baseline"])
    kt = sw["known_truth"]
    check("Bswap known truth: the planted swaps exactly, and the audit catches them",
          kt is not None and kt["wrong_judged"] == truth["bswap"]["changed"]
          and abs(kt["label_error_rate"] - share) < 1e-3 and kt["conflict_recall_on_wrong"] >= 0.85
          and kt["false_conflict_rate_on_correct"] <= 0.03, kt)
    rows = boxes_csv(out)
    changed = {(ch["key"], ch["line"]) for ch in truth["bswap"]["changes"]}
    sw_rows = [x for x in rows if x["manifest"] == "Bswap"]
    planted = [x for x in sw_rows if x["label"] != x["train_core_label"]]
    check("Bswap per-box rows: one per box, the wrong ones exactly the planted ones",
          len(sw_rows) == truth["bswap"]["boxes"] and len(planted) == len(changed)
          and {(x["key"], int(x["box"])) for x in planted} == changed, (len(planted), len(changed)))
    clean = res["audits"]["Bswap_clean"]["species"]
    check("the same images with their own labels stay at the baseline",
          clean["conflict_rate"] <= f + 0.03 and not clean["above_baseline"]
          and abs(clean["verified_rate"] - bl["species"]["verified_rate"]) < 0.08, clean)

    i1 = res["audits"]["I1"]
    check("I1 (clean): conflict rate near the baseline, not above it",
          i1["species"]["conflict_rate"] <= f + 0.03 and not i1["species"]["above_baseline"]
          and (i1["species"]["noise_estimate_corrected"] or 0.0) < 0.05, i1["species"])
    mx = i1["species"]["baseline_for_mix"]
    check("I1: its mix's baseline interval is one interval over its species, not a sum of per-species "
          "bounds (0.055 here)", mx["false_conflict_rate_ci95"][1] < 0.03
          and mx["boxes"] == i1["species"]["boxes_judged"] and len(mx["per_species"]) == 12, mx)
    check("I1: the small box and the failed crop are not judged, and say why",
          i1["no_embedding_reasons"] == {"small": 1, "failed": 1}
          and i1["boxes_by_status"]["no_embedding"] == 2, i1["no_embedding_reasons"])
    check("I1: no image in the trusted set, no shared session",
          i1["images_refused_in_trusted"] == 0 and i1["sessions_shared_with_trusted"] == [])

    br = res["audits"]["Breal"]
    est = br["species"]["noise_estimate_corrected"]
    want = truth["breal_wrong"] / float(truth["breal_species"])
    check("Breal-style: 25 % wrong species labels estimated at about that",
          abs(want - POOL_WRONG) < 0.01 and abs(est - want) < 0.08, (est, want, br["species"]))
    check("Breal-style: per source", set(br["per_source"]) == {"slugA", "slugB"}
          and sum(v["boxes"] for v in br["per_source"].values()) == br["boxes"])
    ob = [x for x in rows if x["manifest"] == "Breal" and x["label"] == str(C.OTHER_PLANT)]
    hidden = {("Breal__" + k, b) for k, (tr, _bx) in truth["pool_truth"].items()
              for b, t in enumerate(tr) if t[0] == "hidden"}
    hid = [x for x in ob if (x["key"], int(x["box"])) in hidden]
    oth = [x for x in ob if (x["key"], int(x["box"])) not in hidden]
    check("OtherPlant boxes of other plants are other_ok",
          oth and sum(x["status"] == "other_ok" for x in oth) >= 0.9 * len(oth),
          [x["status"] for x in oth][:10])
    check("OtherPlant boxes that hold a cwd12 species are conflicts",
          hid and sum(x["status"] == "conflict" for x in hid) >= 0.8 * len(hid),
          [x["status"] for x in hid])
    check("Breal-style: no known-truth block (no train_core crop)", br["known_truth"] is None)

    de = res["audits"]["defects"]
    by = {(x["key"], int(x["box"])): x for x in rows if x["manifest"] == "defects"}
    check("the same bytes under another path are found by sha256 and judged",
          de["images_matched_by_sha256"] == 1 and all(by[("moved", b)]["status"] in A.JUDGED
                                                       for b in range(BOXES)))
    check("an image whose bytes changed is refused",
          all(by[("changed", b)]["reason"] == "image_changed" for b in range(BOXES)))
    check("an image not among the Step 1 images is refused",
          all(by[("absent", b)]["reason"] == "image_not_in_crops" for b in range(BOXES)))
    check("a box moved by 0.01 is a geometry mismatch, not judged",
          by[("geometry", 0)]["status"] == "no_embedding"
          and by[("geometry", 0)]["reason"] == "geometry_mismatch"
          and abs(float(by[("geometry", 0)]["geom_delta"]) - 0.01) < 1e-5, by[("geometry", 0)])
    check("a box within the 1e-4 tolerance is judged",
          by[("geometry", 1)]["status"] in A.JUDGED
          and 0 < float(by[("geometry", 1)]["geom_delta"]) <= 1e-4, by[("geometry", 1)])
    check("a box beyond the crop rows has no crop row",
          by[("extra", BOXES)]["reason"] == "no_crop_row"
          and all(by[("extra", b)]["status"] in A.JUDGED for b in range(BOXES)))
    check("defects: the reasons are counted",
          de["no_embedding_reasons"] == {"image_not_in_crops": BOXES, "image_changed": BOXES,
                                         "no_crop_row": 1, "geometry_mismatch": 1}
          and de["geometry_mismatch_examples"][0]["key"] == "geometry", de["no_embedding_reasons"])

    ov = res["audits"]["overlap"]
    ov_rows = [x for x in rows if x["manifest"] == "overlap"]
    p0_keys = {r["key"] for r in truth["rows"]["P0"][:3]}
    cp_p0, cp_i1 = truth["copies"][0]["key"], truth["copies"][1]["key"]
    moved = truth["moved"]["key"]
    st = {}
    for x in ov_rows:
        st.setdefault(x["key"], set()).add(x["status"])
    check("an image of the trusted manifest is refused, never judged: by path",
          all(st[k] == {"in_trusted"} for k in p0_keys), [st[k] for k in p0_keys])
    check("... by sha256 (a trusted image under another path)",
          st[moved] == {"in_trusted"} and next(x for x in ov_rows if x["key"] == moved)["matched_by"]
          == "sha256", st[moved])
    check("... and a cwd12 copy of a trusted train_core image (the same photograph, other bytes)",
          st[cp_p0] == {"in_trusted"}, st[cp_p0])
    check("... counted", ov["images_refused_in_trusted"] == 5
          and ov["boxes_by_status"]["in_trusted"] == 5 * BOXES, ov["boxes_by_status"])
    cp_rows = [x for x in ov_rows if x["key"] == cp_i1]
    check("a copy of an untrusted image is judged on its copy crop",
          len(cp_rows) == BOXES and all(x["status"] in A.JUDGED and x["crop_set"] == "copy"
                                        and x["matched_by"] == "path" for x in cp_rows),
          [(x["status"], x["crop_set"]) for x in cp_rows])
    i1_rows = [(x["key"], x["box"], x["status"], x["pred"], x["p"])
               for x in rows if x["manifest"] == "I1"]
    ov_i1 = [(x["key"], x["box"], x["status"], x["pred"], x["p"])
             for x in ov_rows if x["key"] not in p0_keys | {cp_p0, cp_i1, moved}]
    check("the rest of the overlapping manifest is judged as on its own", i1_rows == ov_i1)
    check("OUT.md holds the table", "| Bswap |" in A._paths(out)[1].read_text()
          and "trusted, held-out CV (baseline)" in A._paths(out)[1].read_text())
    return res


def test_trusted_identity(paths, truth):
    print("in_trusted: each route on its own")
    crops = V.Crops()
    X = V.load_embeddings(crops)[0]
    i1 = {r["key"]: r for r in truth["rows"]["I1"]}
    cp = {k: v for k, v in truth["copies"][1].items() if k in C.MANIFEST_KEYS}
    twin = i1[truth["copies"][1]["train_core_key"]]
    other = next(r for k, r in sorted(i1.items()) if k != twin["key"])
    ghost_t = dict(other, image="/nowhere/t.jpg", sha256="d" * 64, key="ghost_t")
    ghost_a = dict(other, image="/other/t.jpg", sha256="d" * 64, key="ghost_a")
    same_path = dict(other, sha256="c" * 64, key="same_path")
    rows = [cp, twin, other, ghost_t, ghost_a, same_path]
    index = A.CropIndex(crops, X, [[(r, []) for r in rows]])

    def trusted(*rs):
        return A.trusted_ids(index, [(r, []) for r in rs])

    check("the same sha256 under other paths, neither a Step 1 image",
          index.image(ghost_t)[0] is None and index.image(ghost_a)[0] is None
          and A.in_trusted(index, ghost_a, trusted(ghost_t)))
    check("the same path under another sha256 (a changed image)",
          A.in_trusted(index, same_path, trusted(other)))
    check("the train_core twin of a trusted cwd12 copy", A.in_trusted(index, twin, trusted(cp))
          and not A.in_trusted(index, other, trusted(cp)))
    check("a cwd12 copy of a trusted train_core image", A.in_trusted(index, cp, trusted(twin))
          and not A.in_trusted(index, cp, trusted(other)))
    check("no trusted manifest: nothing refused", not A.in_trusted(index, cp, None))


def test_determinism_and_independence(paths, res1):
    print("determinism, independence of the probe from the audited sets")
    audits = [(n, paths[n]) for n in ("I1", "Bswap", "Bswap_clean", "Breal", "defects", "overlap")]
    out2 = TMP / "out2" / "audit.json"
    res2 = A.run_audit(paths["P0"], audits, out2)
    out1 = TMP / "out1" / "audit.json"
    check("a rerun gives the same result", strip(res1) == strip(res2))
    check("... the same per-box CSV and report",
          A._paths(out1)[2].read_bytes() == A._paths(out2)[2].read_bytes()
          and A._paths(out1)[1].read_bytes() == A._paths(out2)[1].read_bytes())
    res1b = A.run_audit(paths["P0"], audits, out1)
    check("a rerun over its own earlier outputs is allowed and gives the same", strip(res1b) == strip(res1)
          and A._paths(out1)[2].read_bytes() == A._paths(out2)[2].read_bytes())
    res3 = A.run_audit(paths["P0"], [("I1", paths["I1"])], TMP / "out3" / "audit.json")
    check("the probe and the baseline do not depend on what is audited",
          res3["probe"] == res1["probe"] and res3["baseline_cv"] == res1["baseline_cv"]
          and res3["trusted"] == res1["trusted"])
    check("... nor does I1's audit", res3["audits"]["I1"] == res1["audits"]["I1"])


def test_trusted_labels(paths):
    print("the probe reads the trusted manifest's own labels")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = A.run_audit(paths["P0_relabelled"], [("I1", paths["I1"])], TMP / "out4" / "audit.json")
    log = buf.getvalue()
    print(log, end="")
    check("no OtherPlant threshold warning from verify; the audit says once why there is no class",
          "OtherPlant: cv top-1" not in log and log.count("no OtherPlant class, by design") == 1)
    pr, t = res["probe"], res["trusted"]
    check("a species the trusted labels do not hold is not in the probe",
          pr["species_not_in_trusted"] == ["CutleafGroundcherry"]
          and t["used_per_species"]["CutleafGroundcherry"] == 0, pr["species_not_in_trusted"])
    i1 = res["audits"]["I1"]
    n11 = i1["boxes_per_class"]["CutleafGroundcherry"]
    unseen_ok = i1["per_species"]["CutleafGroundcherry"]["boxes_judged"] == 0
    check("audited boxes labelled with it are label_unseen, not conflicts",
          n11 > 0 and unseen_ok
          and i1["boxes_by_status"]["label_unseen"] == n11 - sum(
              1 for x in boxes_csv(TMP / "out4" / "audit.json")
              if x["label"] == "11" and x["status"] == "no_embedding"), i1["boxes_by_status"])


def test_refusals(paths):
    print("hard refusals")
    out = TMP / "out5" / "audit.json"
    check("too few trusted groups for the baseline CV is refused",
          refused(lambda: A.run_audit(paths["P0_two"], [("I1", paths["I1"])], out),
                  "fewer than 2 groups"))
    lp = pathlib.Path(C.read_manifest(paths["I1"])[5]["label"])
    text = lp.read_text()
    lp.write_text(text + "0 0.500000 0.500000 0.100000 0.100000\n")
    try:
        check("a label file changed after its manifest is refused",
              refused(lambda: A.run_audit(paths["P0"], [("I1", paths["I1"])], out), "label_sha256"))
    finally:
        lp.write_text(text)
    pool_text = V.POOL.read_text()
    V.POOL.write_text(pool_text + "\n")
    try:
        check("crops.csv made from other inputs (verify.check_fresh) is refused",
              refused(lambda: A.run_audit(paths["P0"], [("I1", paths["I1"])], out), "Step 1 inputs"))
    finally:
        V.POOL.write_text(pool_text)
    step1_before = {p: p.read_bytes() for p in (V.CROPS, V.CROPS_SKIPPED, V.CROPS_INFO, V.POOL, V.COPIES)}

    def run_out(o, audited=paths["I1"]):
        return lambda: A.run_audit(paths["P0"], [("I1", audited)], o)

    check("--out must end in .json: an input manifest is refused",
          refused(run_out(paths["I1"]), "must end in .json"))
    check("... and so is crops.csv", refused(run_out(V.CROPS), "must end in .json"))
    check("--out under the Step 1 directory is refused: crops_info.json",
          refused(run_out(V.CROPS_INFO), "Step 1 directory"))
    check("... and any new file there", refused(run_out(V.STEP1 / "audit" / "a.json"), "Step 1 directory")
          and not (V.STEP1 / "audit").exists())
    as_json = TMP / "inc" / EXP / "manifests" / "I1_as.json"
    shutil.copy(paths["I1"], as_json)
    check("--out that is an audited manifest is refused", refused(run_out(as_json, as_json), "is an input"))
    exp_json = TMP / "inc" / EXP / "exp.json"
    exp_json.write_text('{"exp": "pilot_t"}\n')
    check("an existing file that is not an earlier audit output is not overwritten",
          refused(run_out(exp_json), "not an earlier audit output")
          and exp_json.read_text() == '{"exp": "pilot_t"}\n')
    side = TMP / "out7" / "a.json"
    side.parent.mkdir(parents=True, exist_ok=True)
    (TMP / "out7" / "a.md").write_text("notes\n")
    check("... nor is an OUT.md beside it", refused(run_out(side), "not an earlier audit output")
          and not side.exists() and (TMP / "out7" / "a.md").read_text() == "notes\n")
    V.check_fresh(V.Crops())
    check("the Step 1 files are untouched by every refusal",
          all(p.read_bytes() == b for p, b in step1_before.items()))
    check("NAME=MANIFEST is required", refused(lambda: A.parse_audits(["I1"]), "NAME=MANIFEST"))
    check("a name given twice is refused",
          refused(lambda: A.parse_audits(["a=x.jsonl", "a=y.jsonl"]), "twice"))
    check("a name that is not path-safe is refused",
          refused(lambda: A.parse_audits(["a b=x.jsonl"]), "use letters"))
    check("a missing manifest is refused",
          refused(lambda: A.run_audit(paths["P0"], [("x", TMP / "nope.jsonl")], out), "not found"))


def test_noisy_world():
    print("a noisier world: held-out false conflicts are not rare, and the correction must remove them")
    paths, truth = build_world(noise=2.2)
    res = A.run_audit(paths["P0"], [(n, paths[n]) for n in ("I1", "Bswap", "Bswap_clean", "Breal")],
                      TMP / "noisy" / "audit.json")
    bl = res["baseline_cv"]
    f, r = bl["species"]["false_conflict_rate"], bl["swapped"]["conflict_recall_on_wrong"]
    check("baseline: false-conflict rate above 0.08, a wrong species still caught", f >= 0.08 and r >= 0.85,
          (f, r, res["probe"]["cv_top1_species"]))
    share = truth["bswap"]["changed"] / float(truth["bswap"]["boxes"])
    sw = res["audits"]["Bswap"]["species"]
    c = sw["conflict"] / float(sw["boxes_judged"])
    check("Bswap: the corrected estimate is the planted share, where neither c nor c / r is",
          abs(sw["noise_estimate_corrected"] - share) < 0.05 and abs(c - share) > 0.03
          and abs(c / r - share) > 0.05, (sw["noise_estimate_corrected"], c, c / r, share))
    for n in ("I1", "Bswap_clean"):
        sp = res["audits"][n]["species"]
        check("%s (clean): conflict rate above 0.08, corrected estimate near 0, not above the baseline" % n,
              sp["conflict_rate"] >= 0.08 and sp["noise_estimate_corrected"] <= 0.05
              and not sp["above_baseline"] and res["audits"][n]["known_truth"]["label_error_rate"] == 0.0,
              sp)
    br = res["audits"]["Breal"]["species"]
    want = truth["breal_wrong"] / float(truth["breal_species"])
    check("Breal-style: 25 % wrong species estimated at about that", abs(br["noise_estimate_corrected"] - want)
          < 0.06 and br["above_baseline"], (br["noise_estimate_corrected"], want))


HARD = (5, 11, 0.35, 0.15, 12.0)


def test_species_mix_world():
    print("a skewed species mix: the baseline must be the audited set's own mix")
    a, b = C.CLASS_NAMES[HARD[0]], C.CLASS_NAMES[HARD[1]]
    paths, truth = build_world(hard=HARD)
    res = A.run_audit(paths["P0"], [(n, paths[n]) for n in ("I1", "Bswap_clean", "Bswap")],
                      TMP / "mix" / "audit.json")
    bl = res["baseline_cv"]
    used = res["trusted"]["used_per_species"]
    pooled_f = bl["species"]["false_conflict_rate"]
    check("world: the pair is rare in P0 and hard (its held-out false-conflict rates are the highest)",
          (used[a] + used[b]) < 0.05 * res["trusted"]["species_boxes_used"]
          and min(bl["per_species"][a]["conflict_rate"], bl["per_species"][b]["conflict_rate"])
          > 5 * max(pooled_f, 0.005), (used[a], used[b], bl["per_species"][a], pooled_f))
    for n in ("I1", "Bswap_clean"):
        blk = res["audits"][n]
        sp, mx = blk["species"], blk["species"]["baseline_for_mix"]
        pair = blk["boxes_per_class"][a] + blk["boxes_per_class"][b]
        check("%s: every label right, most boxes of the hard pair" % n,
              blk["known_truth"]["label_error_rate"] == 0.0 and pair > 0.5 * blk["boxes"], pair)
        check("%s: the pooled baseline would call it noisy" % n,
              sp["above_pooled_baseline"] and sp["noise_estimate_pooled_baseline"] > 0.04, sp)
        check("%s: against its own mix it is clean: not above the baseline, corrected estimate near 0" % n,
              not sp["above_baseline"] and sp["noise_estimate_corrected"] < 0.02
              and mx["false_conflict_rate"] > 5 * pooled_f and mx["species_without_baseline"] == {}
              and mx["boxes"] == sp["boxes_judged"], (sp["noise_estimate_corrected"], mx))
    sw = res["audits"]["Bswap"]["species"]
    share = truth["bswap"]["changed"] / float(truth["bswap"]["boxes"])
    check("Bswap on the skewed sessions: still above the baseline, estimate within 0.07 of the share "
          "(the label mix only stands in for the wrong labels' true species)",
          sw["above_baseline"] and abs(sw["noise_estimate_corrected"] - share) < 0.07,
          (sw["noise_estimate_corrected"], share))
    one = A.mix_interval([(3, 40)], [1.0])
    check("mix_interval: one species is its Jeffreys interval; it holds its own point",
          one[0] < 3 / 40.0 < one[1] and A.mix_interval([(0, 50), (0, 50)], [0.5, 0.5])[0] == 0.0, one)


def test_cli(paths):
    print("CLI and sbatch script")
    out = TMP / "out6" / "audit.json"
    rc = A.main(["--trusted", str(paths["P0"]), "--audit", "I1=%s" % paths["I1"],
                 "Bswap=%s" % paths["Bswap"], "--audit", "Breal=%s" % paths["Breal"], "--out", str(out)])
    res = json.loads(out.read_text())
    check("main: NAME=MANIFEST lists, repeated --audit, exit 0",
          rc == 0 and res["audit_order"] == ["I1", "Bswap", "Breal"]
          and sorted(res["audits"]) == ["Breal", "Bswap", "I1"], (rc, res.get("audit_order")))
    rc = A.main(["--trusted", str(paths["P0"]), "--audit", "I1", "--out", str(out)])
    check("main: a bad spec exits 2", rc == 2)
    sh = ROOT / "run_inc_audit.sh"
    r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    body = sh.read_text()
    check("run_inc_audit.sh parses and passes its arguments to the module",
          r.returncode == 0 and '-m weed_optimizer_framework.tools.inc.audit "$@"' in body
          and "--gres=gpu:v100-32:1" in body and "--time=02:00:00" in body, r.stderr)


def main():
    try:
        paths, truth = build_world()
        res = test_audit(paths, truth)
        test_determinism_and_independence(paths, res)
        test_trusted_identity(paths, truth)
        test_trusted_labels(paths)
        test_refusals(paths)
        test_cli(paths)
        test_noisy_world()
        test_species_mix_world()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
