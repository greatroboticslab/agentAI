#!/usr/bin/env python3
"""DS1 Revision 1 and the fixes made with it before any DS1 data (docs/CONTINUOUS_LOOP.md, "Revision (2026-10-29):
DS1 Revision 1, before any DS1 data"; docs/prereg/DS1_rev1_2026-10-29.md).

- the cwd12 copy screen (DS.5.1 item 1(a), D28-v2): copy_hit (an embedding hit; a dHash hit at pair cosine >= the
  copy threshold, or without one; a split not kept counts; ImageWeeds and other refusals do not); the intake's own
  refusals of a source (decisions.jsonl, its eval_hits sidecar); Step 1's copy matches (ingest.jsonl,
  b0000's eval_hit_pairs); the screen: a source with an intake or a Step 1 copy refusal quarantined whole, a dHash
  hit the DINOv2 detector does not confirm only counted, one with no detector quarantining (fail closed);
- role C by capture names (item 1(b)): a source whose file names carry them is never ready, whatever its sheet;
- a screen revision written when a Step 1 batch came and nothing is new (the platform sees the screen done); none
  when nothing changed; phase 2 reads only the named sources;
- source classes (Revision item 2): the ambiguous INC classes, each box's traced source class, the untraced
  fallback, roles per source class, the names on the sheet; recover_box_src on the collector's own reader (a YOLO
  source staged as a fetch): traced, a box mismatch and a file the source no longer holds not traced, gone blobs;
- the draw's exclusions (Revision item 1): a group near a built manifest's row (dHash, DINOv2 only) drawn into
  neither exam; near base B or a stream pool, or holding a base B file, never into MT-W v2;
- the owner's seeded sample over several screens (item 3) and its use in the fold;
- the sheets (item 5): nothing pre-selected on an audit sheet; an audit recorded under the owner's id only; an LN
  image not marked checked and an MT-W image not answered refuse the file; an MT-W correction is the whole corrected
  label; the MT-W check's corrected labels; a render pulls the cluster's sheets state before listing its batch; an
  audit batch on the lab names its sources; LN and MT-W batches in the order their bundles were written;
- a source rejected (or role C, or quarantined) after its pool or its exams were written: out of every manifest
  (ready_pools), its MD-N and MT-W v2 never scored (md_exams, mtw_roots);
- MT-W v1.1 written when a test v1 source fails its check with no correction, without that source's rows;
- the reads: none before verdict v2, the challenger third; the substitution when S2x is not run (walltime); a
  verdict refused while an arm is missing; the arms' states from k_not_run.json and the long path's smokes; the
  0.90 interval with the capture-group SE (mtw_groups); a sealed read re-submitted after a partial failure
  completes;
- the long path keeps its train logs, removes its data and checkpoints, and moves an earlier attempt aside.

Run:  python3 tests/test_inc2_ds1_rev1.py
"""
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import traceback

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_rev1_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"
PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import base3 as B3  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_data as DD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_long as LG  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_sheets as SH  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402

FAILURES = []
DD.PROJECT_LOADER = False
SIU = DD.SIU


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


@contextlib.contextmanager
def inc(name):
    """A fresh INC_DIR for one test (DS1's modules read inc.common.INC_DIR when they run)."""
    prev = C.INC_DIR
    C.INC_DIR = TMP / name
    C.INC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield C.INC_DIR
    finally:
        C.INC_DIR = prev


def h64(seed):
    return int(np.random.default_rng(seed).integers(0, 2 ** 63)) | (1 << 63)


def flip(h, bits, start=0):
    for b in range(start, start + bits):
        h ^= 1 << b
    return h


def variants(h, seed):
    return [int(h)] + [h64(seed * 100 + i) for i in range(1, 8)]


def png(path, seed, size=(64, 48)):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return str(path)


def label(path, boxes):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in boxes))
    return str(path)


def jl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class StubScanner(object):
    """The DINOv2 detector: {image path: (split, key)} it calls copies."""

    def __init__(self, flagged=None):
        self.flagged = dict(flagged or {})

    def scan(self, rows, desc_path=None, procs=1):
        out = {}
        for r in rows:
            hit = self.flagged.get(str(r["image"]))
            if hit:
                out[r["key"]] = [{"eval_split": hit[0], "eval_key": hit[1], "cos": 0.97}]
        return out, set()


def agn_arrays(n_img, seed, quality, keys=None):
    rng = np.random.default_rng(seed)
    keys = keys or ["k%04d" % i for i in range(n_img)]
    tp, conf, img = [], [], []
    for i in range(n_img):
        for _j in range(4):
            good = rng.random() < quality
            k = int(rng.integers(5, 11)) if good else 0
            row = np.zeros(10, dtype=bool)
            row[:k] = True
            tp.append(row)
            conf.append(float(rng.random() * (0.5 + 0.5 * good)))
            img.append(i)
    return {"keys": np.asarray(keys), "tp": np.asarray(tp), "conf": np.asarray(conf), "pred_img": np.asarray(img),
            "n_gt": np.full(n_img, 3, dtype=np.int64)}


def fake_timestamp():
    """ds1/v1/timestamp.json as the delegate's record writes it (the jobs' gate), for the steps tested here."""
    D.write_json(D.timestamp_path(), {"format": D.FMT_TIMESTAMP, "sha256": D.PREREG_SHA256, "proofs": [
        {"file": "a.ots", "sha256": "a" * 64}, {"file": "b.ots", "sha256": "b" * 64}],
        "revisions": [{"file": r["file"], "sha256": r["sha256"], "proofs": [{"file": "c", "sha256": "c" * 64},
                                                                             {"file": "d", "sha256": "d" * 64}]}
                      for r in D.REVISIONS]})


# ------------------------------------------------------------------ the cwd12 copy screen
def t_copy_rules():
    print("the cwd12 copy screen's rule (DS.5.1 item 1(a), D28-v2)")
    cases = [
        ("an embedding hit on cwd12 test", ("near_eval_embed", {"eval_split": "test", "cos": 0.97}, None), True),
        ("a masked embedding hit on dev", ("masked_near_eval_embed", {"eval_split": "dev"}, None), True),
        ("an embedding hit on ImageWeeds (not cwd12)", ("near_eval_embed", {"eval_split": "imageweeds"}, None), False),
        ("a dHash hit on dev at pair cosine 0.50", ("near_eval_v2", {"split": "dev"}, 0.50), False),
        ("a dHash hit on dev at pair cosine 0.95", ("near_eval_variant", {"split": "dev"}, 0.95), True),
        ("a dHash hit on test with no pair cosine (fail closed)", ("near_eval_v2", {"split": "test"}, None), True),
        ("an embedding refusal whose match was not kept (fail closed)", ("near_eval_embed", None, None), True),
        ("a base copy", ("base_copy", {"split": "dev"}, None), False)]
    got = {n: bool(DD.copy_hit(*a)) for n, a, _w in cases}
    check("copy_hit: %s" % "; ".join("%s %s" % (n, "hit" if w else "no") for n, _a, w in cases),
          all(got[n] == w for n, _a, w in cases), got)
    with inc("inc_copy") as d:
        b1 = d / "intake" / "i0001_src_x"
        jl(b1 / "decisions.jsonl", [
            {"kind": "class", "source": "src_x", "src_id": 0, "name": "weed", "inc_id": 12},
            {"kind": "image", "source": "src_x", "key": "x1", "decision": "rejected", "reason": "near_eval_v2",
             "match": {"split": "test", "key": "t1"}, "pair_cos": 0.31},
            {"kind": "image", "source": "src_x", "key": "x2", "decision": "rejected", "reason": "near_eval_v2",
             "match": {"split": "dev", "key": "d1"}},
            {"kind": "image", "source": "src_x", "key": "x3", "decision": "rejected", "reason": "near_eval_variant",
             "match": {"split": "test", "key": "t2"}, "pair_cos": 0.97},
            {"kind": "image", "source": "src_x", "key": "x4", "decision": "rejected", "reason": "near_eval_embed",
             "match": {"eval_split": "imageweeds"}},
            {"kind": "image", "source": "src_x", "key": "x5", "decision": "kept", "reason": "kept"}])
        (b1 / "summary.json").write_text(json.dumps({"source": "src_x"}))
        (b1 / "eval_hits.json").write_text(json.dumps({"pairs": [{"key": "x2", "pair_cos": 0.99}]}))
        b2 = d / "intake" / "i0002_src_y"
        jl(b2 / "decisions.jsonl", [{"kind": "image", "source": "src_y", "key": "y1", "decision": "rejected",
                                     "reason": "near_eval_embed", "match": {"eval_split": "test"}}])
        (b2 / "summary.json").write_text(json.dumps({"source": "src_y"}))
        hits = DD.intake_cwd12_hits("src_x")
        check("the intake's refusals of src_x: the dHash hit at 0.99 by its sidecar and the one at 0.97, not the one "
              "at 0.31, the ImageWeeds hit or another source's", sorted(h["key"] for h in hits) == ["x2", "x3"], hits)
        sb = d / "step1_stream" / "batches" / "b0012"
        jl(sb / "ingest.jsonl", [{"key": "q1", "decision": "near_eval_embed", "guard_match": {
            "eval_split": "test", "eval_key": "t9", "cos": 0.96}}, {"key": "q2", "decision": "pool"}])
        im = DD.ingest_matches(sb, {"eval_hit_pairs": [{"key": "q3", "match": {"split": "dev"}, "pair_cos": 0.2}]})
        check("Step 1's copy matches: an ingest row's guard match, b0000's eval_hit_pairs", set(im) == {"q1", "q3"}
              and im["q1"]["guard_match"]["eval_split"] == "test" and im["q3"]["pair_cos"] == 0.2, im)
        row = DD.candidate_row("q1", {"refusal": "near_eval_embed", "source": "s"}, None,
                               {"label": label(d / "l.txt", [(12, .5, .5, .2, .2)]), "image": "/x.png"}, im["q1"])
        check("  a candidate refused at Step 1 carries its match (D28-v2's input)", row["guard_match"]["eval_split"] ==
              "test" and DD.copy_hit(row["refused"], row["guard_match"], row["pair_cos"]))


def screen_inputs(d, rows, scanners, intake_hits=None, record=None):
    return DD.Inputs(candidates=rows, scanners=scanners, intake_hits=intake_hits or {}, record=record or {},
                     exams={"dev": {"keys": ["dev__a"], "H": np.asarray([variants(h64(9002), 9002)],
                                                                        dtype=np.uint64), "ok": np.asarray([True])}},
                     hasher=lambda p: HASHES.get(str(p), (None, None)), production=False)


HASHES = {}


def cand(d, key, source, seed, h=None, rel=None, refused_=None, guard_match=None, boxes=None):
    img = png(d / "src" / source / ("%s.png" % key), seed, (400, 400))
    lab = label(d / "src" / source / ("%s.txt" % key), boxes or [(12, 0.5, 0.5, 0.2, 0.2)])
    hh = h if h is not None else h64(seed)
    HASHES[img] = (hh, {k: v for k, v in zip(B3.VARIANTS, variants(hh, seed))})
    return {"key": key, "source": source, "pool": "p3", "image": img, "sha256": C.sha256_file(img), "label": lab,
            "boxes": [tuple(b) for b in (boxes or [(12, 0.5, 0.5, 0.2, 0.2)])], "W": 400, "H": 400,
            "rel": rel or key, "series": None, "eppo": None, "capture": None, "stem": None, "session": "",
            "dhash": hh, "variants": None, "licence": "cc-by", "research_only": False, "holds": [],
            "refused": refused_, "admission": "whole" if not refused_ else "not_admitted", "test_v1": None,
            "batch": "b0012", "guard_match": guard_match, "pair_cos": None, "box_src": None}


def t_screen():
    print("the screen: copies quarantine a source whole, role C by capture names, revisions, phase 2's sources")
    with inc("inc_screen") as d:
        rows = [cand(d, "ok_%d" % i, "src_ok", 100 + i) for i in range(6)]
        rows += [cand(d, "ref_%d" % i, "src_ref", 200 + i) for i in range(5)]
        rows.append(cand(d, "ref_x", "src_ref", 210, refused_="near_eval_embed",
                         guard_match={"eval_split": "test", "cos": 0.96}))
        rows += [cand(d, "int_%d" % i, "src_int", 300 + i) for i in range(5)]
        rows += [cand(d, "dh_%d" % i, "src_dh", 400 + i, h=flip(h64(9002), 2) if i == 0 else None) for i in range(5)]
        rows += [cand(d, "IphoneSE_YL_Morningglory_%03d" % i, "src_cap", 500 + i,
                      rel="IphoneSE_YL_Morningglory_%03d_JPG.rf.ab.jpg" % i) for i in range(5)]
        inp = screen_inputs(d, rows, {"eval": StubScanner()}, intake_hits={"src_int": [{"key": "i9"}]},
                            record={"step1_batches": ["b0012"]})
        doc = DD.screen(1, inp)
        s = doc["sources"]
        check("Step 1's refusal of a cwd12 test copy quarantines its source whole; so does an intake refusal",
              s["src_ref"]["quarantine_whole"] and s["src_int"]["quarantine_whole"] and
              s["src_ref"]["cwd12_copies"]["step1_refusals"] == 1, {k: s[k].get("cwd12_copies") for k in s})
        check("  a dHash hit the DINOv2 detector does not confirm drops its image later but does not quarantine; a "
              "clean source is not quarantined", not s["src_dh"]["quarantine_whole"] and
              s["src_dh"]["cwd12_copies"]["dhash_unconfirmed"] == 1 and not s["src_ok"]["quarantine_whole"],
              s["src_dh"].get("cwd12_copies"))
        nos = DD.screen(1, screen_inputs(d, [cand(d, "dh2_0", "src_dh2", 600, h=flip(h64(9002), 3))], {}), write=False)
        check("  with no DINOv2 detector a dHash hit quarantines (fail closed)", nos["sources"]["src_dh2"][
            "quarantine_whole"])
        check("a source whose file names carry cwd12 capture names is role C: never ready, even accepted and audited",
              s["src_cap"]["role_c"] and DD.source_ready("src_cap", {"src_cap": {"decision": "accept"}},
                                                       s["src_cap"])[0] is False and
              "role C" in DD.source_ready("src_cap", {"src_cap": {"decision": "accept"}}, s["src_cap"])[1] and
              DD.settled_out("src_cap", {}, s["src_cap"]))
        check("  a quarantined source is settled out with no decision; a clean undecided one is not",
              DD.settled_out("src_ref", {}, s["src_ref"]) and not DD.settled_out("src_ok", {}, s["src_ok"]))
        n1 = len(DD.screen_revisions(1))
        again = DD.screen(1, inp)
        check("a second screen with nothing new and no new Step 1 batch writes no revision", len(
            DD.screen_revisions(1)) == n1 == 1 and "unchanged" in again.get("status", ""), again.get("status"))
        inp.record = {"step1_batches": ["b0012", "b0013"]}
        DD.screen(1, inp)
        last = DD._read_json(DD.screen_revisions(1)[-1])
        check("  a new Step 1 batch with no new row: a revision is written naming the batches it saw (the plan sees the "
              "screen done)", len(DD.screen_revisions(1)) == 2 and last["candidates"] == 0 and
              last["inputs"]["step1_batches"] == ["b0012", "b0013"])
        merged = DD.screen_records(1)
        check("  the sources' records merged over revisions keep the flags that settle them", merged["src_ref"][
            "quarantine_whole"] and merged["src_cap"]["role_c"])
        named = DD.phase2_ids()
        mf = next(iter(named))
        p2rows = [cand(d, "p2_ok_%d" % i, "src_late", 700 + i) for i in range(3)] + \
            [cand(d, "p2_mf_%d" % i, mf, 800 + i) for i in range(3)]
        doc2 = DD.screen(2, screen_inputs(d, p2rows, {"eval": StubScanner()}, record={"step1_batches": ["b0014"]}))
        check("phase 2 screens only the sources DS.5.2 names (PHASE2_SOURCES): %s's rows, never src_late's" % mf,
              set(doc2["sources"]) == {mf} and doc2["candidates"] == 3, sorted(doc2["sources"]))


# ------------------------------------------------------------------ source classes (Revision item 2)
def t_source_classes():
    print("each box's role on its source class (Revision item 2)")
    cmap = {"0": {"name": "amaranth", "inc_id": 3}, "1": {"name": "maize", "inc_id": 12},
            "2": {"name": "pigweed", "inc_id": 12}}
    check("the INC classes several source classes map to", DD.ambiguous_ids(cmap) == {12})
    r1 = {"key": "k1", "source": "s", "boxes": [(3, .5, .5, .2, .2), (12, .3, .3, .1, .1)]}
    r2 = {"key": "k2", "source": "s", "boxes": [(3, .5, .5, .2, .2), (12, .3, .3, .1, .1)]}
    DD.set_box_src([r1, r2], cmap, {"k1": ["0", "2"]})
    check("a traced row's boxes take their source classes; an untraced row's box of an ambiguous INC class keeps it",
          r1["box_src"] == ["0", "2"] and r2["box_src"] == ["0", None] and DD.unit_of(r1, 12, 1) == "src:2" and
          DD.unit_of(r2, 12, 1) == "inc:12" and DD.unit_of(r2, 3, 0) == "src:0")
    k, sp, dr, un = DD.weed_boxes(r1, {"src:0": "weed", "src:2": "weed"})
    k2, _s2, dr2, _u2 = DD.weed_boxes(dict(r1, box_src=["0", "1"]), {"src:0": "weed", "src:1": "crop"})
    check("roles per source class: pigweed kept as a weed, maize (the same INC class) dropped as a crop",
          len(k) == 2 and sp == ["src:0", "src:2"] and len(k2) == 1 and dr2 == {"crop": 1})
    _k3, _s3, _d3, un3 = DD.weed_boxes(r2, {"src:0": "weed", "inc:12": "unresolved"})
    check("  an untraced box of an INC class whose names differ in role is unresolved: the image is held",
          un3 == ["inc:12"])
    check("the sheet's suggestion for a source class is its INC class's; the names each unit stands for",
          DD.suggest_role("s", "src:0", {"0": 3}) == "weed" and DD.suggest_role("s", "src:2", {"2": 12}) ==
          "unresolved" and DD.unit_names("s", ["src:2", "inc:12"], cmap) == {"src:2": ["pigweed"],
                                                                             "inc:12": ["maize", "pigweed"]})
    with inc("inc_srcclass") as d:
        from weed_optimizer_framework.tools import collect as CL
        sdir = CL.staging_dir("src_cw", d)
        tree = TMP / "cw_tree"
        files = {}
        for i in range(3):
            files["images/a%d.jpg" % i] = pathlib.Path(png(tree / ("a%d.jpg" % i), 900 + i)).read_bytes()
            files["labels/a%d.txt" % i] = b"0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n"
        files["data.yaml"] = b"names: ['amaranth', 'maize', 'pigweed']\n"
        recs = []
        for name, data in sorted(files.items()):
            sha = hashlib.sha256(data).hexdigest()
            (sdir / "blobs").mkdir(parents=True, exist_ok=True)
            (sdir / "blobs" / sha).write_bytes(data)
            recs.append({"name": name, "sha256": sha, "bytes": len(data)})
        (sdir / "fetch.json").write_text(json.dumps({"format": CL.FORMATS["fetch"], "files": recs,
                                                     "provider": "test", "ref": "x"}))
        b = d / "intake" / "i0001_src_cw"
        jl(b / "decisions.jsonl", [{"kind": "class", "source": "src_cw", "src_id": i, "name": n, "inc_id": c}
                                   for i, (n, c) in enumerate((("amaranth", 3), ("maize", 12), ("pigweed", 12)))])
        (b / "summary.json").write_text(json.dumps({"source": "src_cw"}))
        rows = [{"key": "cw0", "rel": "images/a0.jpg", "boxes": [(3, .5, .5, .2, .2), (12, .3, .3, .1, .1)]},
                {"key": "cw1", "rel": "images/a1.jpg", "boxes": [(12, .5, .5, .2, .2), (12, .3, .3, .1, .1)]},
                {"key": "cw9", "rel": "images/zz.jpg", "boxes": [(3, .5, .5, .2, .2)]}]
        got, rec = DD.recover_box_src("src_cw", rows, d)
        check("recover_box_src reads the source again with the collector's own reader: a row whose boxes map to its "
              "intake label box for box is traced; a mismatching row and a file the source no longer holds are not",
              got == {"cw0": ["0", "2"]} and rec["traced"] == 1 and sum(rec["not_traced"].values()) == 2,
              (got, rec))
        (sdir / "blobs" / recs[0]["sha256"]).unlink()
        got2, rec2 = DD.recover_box_src("src_cw", rows, d)
        check("  a source whose staging blobs are gone keeps its INC units (recorded why)", got2 == {} and
              "gone" in rec2.get("why", ""), rec2)


# ------------------------------------------------------------------ the draw's exclusions (Revision item 1)
def t_draw_exclusions():
    print("the draw: groups near a built manifest's rows in neither exam; near base B never in MT-W v2")
    with inc("inc_draw") as d:
        base_b, s0 = {}, []
        for i in range(8):
            img = png(d / "b" / ("b%d.png" % i), 1000 + i)
            lab = label(d / "b" / ("b%d.txt" % i), [(12, .5, .5, .2, .2)])
            h = h64(1000 + i)
            row = {"key": "b%d" % i, "image": img, "label": lab, "sha256": C.sha256_file(img),
                   "label_sha256": C.sha256_file(lab), "source": "rf_b", "session": ""}
            base_b[row["key"]] = dict(row, original_sha256=row["sha256"], dhash=h, variants=variants(h, 1000 + i))
            if i < 4:
                s0.append(row)
        DD.write_arm("S0", s0)
        new = []
        for i in range(6):
            h = h64(2000 + i)
            if i == 0:
                h = flip(base_b["b0"]["dhash"], 5)            # 5 bits from an S0 row
            if i == 2:
                h = flip(base_b["b6"]["dhash"], 4)            # 4 bits from a base B row not in a manifest
            new.append({"key": "n%d" % i, "source": "src_n", "image": "/n/%d.png" % i, "dhash": h,
                        "variants": variants(h, 2000 + i), "sha256": base_b["b7"]["sha256"] if i == 3 else "s%d" % i})
        groups = {r["key"]: "G%d" % i for i, r in enumerate(new)}
        calls = []

        def ref_scan(rows, ref_rows, tag):
            calls.append((tag, sorted(r["key"] for r in ref_rows)))
            return {"n1": [{"exam": "ref", "exam_key": "b1", "how": "embed"}]}
        inp = DD.Inputs(base_b=base_b, ref_scan_fn=ref_scan, pool_ref_fn=lambda: [
            {"key": "pool|x|p1", "dhash": flip(h64(2005), 2), "variants": None}])
        ex, ex_mtw, rec = DD.draw_exclusions(new, groups, inp, "src_n")
        check("near an S0 row by dHash (n0) or by DINOv2 only (n1): in neither exam", ex == {"G0", "G1"}, (ex, rec))
        check("  near a base B row outside the manifests (n2), holding a base B file (n3), or near a stream pool row "
              "(n5): never in MT-W v2", ex_mtw == {"G2", "G3", "G5"}, ex_mtw)
        check("  the DINOv2 index is each built arm's own rows (tag by arm and manifest sha256)",
              calls and calls[0][0].startswith("arm_s0_") and calls[0][1] == ["b0", "b1", "b2", "b3"], calls)
        by = {"src_n": [{"key": "n%d" % i} for i in range(6)] + [{"key": "m%d" % i} for i in range(200)]}
        g2 = dict(groups, **{"m%d" % i: "M%d" % i for i in range(200)})
        dg = {g: hashlib.sha256(g.encode()).hexdigest() for g in set(g2.values())}
        dr = DD.draw_exams(by, g2, dg, {"src_n": 206}, new_sources={"src_n"}, excluded_groups=ex,
                           mtw_excluded_groups=ex_mtw)["src_n"]
        check("  the draw honours them: no excluded group in MD or MT-W v2, no MT-W-excluded group in MT-W v2",
              not (set(dr["md"] + dr["mtw"]) & ex) and not (set(dr["mtw"]) & ex_mtw), dr)


# ------------------------------------------------------------------ the owner's sample (Revision item 3)
def t_audit_sample():
    print("the owner's seeded sample over several screens (Revision item 3)")
    srcs = ["src_%02d" % i for i in range(20)]
    order = sorted(srcs, key=lambda s: (C.stable_int("inc2/ds1/audit/%s" % s), s))
    first = [s for s in srcs if s not in order[:3]][:8]          # revision 1 lacks the three lowest
    s1 = SH.sticky_audit_sample([first])
    s2 = SH.sticky_audit_sample([first, srcs])
    check("after revision 1: its 5 lowest; after revision 2 (the three lowest added) every source drawn before stays "
          "drawn, the 5 lowest of all join", len(s1) == 5 and s2 == s1 | set(order[:5]) and set(order[:3]) <= s2
          and len(s2) > 5, (sorted(s1), sorted(s2)))
    with inc("inc_audit") as d:
        sd = d / "splits" / "ds1" / "screen"
        sd.mkdir(parents=True)

        def rev(n, sources, extra=None):
            (sd / ("p1_r%03d.json" % n)).write_text(json.dumps({"sources": dict(
                {s: {"live": 3} for s in sources}, **(extra or {}))}))
        rev(1, first)
        jl(SH.decisions_path(d), [{"kind": "source", "source": s, "decision": "accept", "setting": "Field",
                                   "roles": {}, "decided_by": "delegate", "sheet_sha256": "x"}
                                  for s in srcs + ["src_capname"]])
        f1 = SH.fold(d)
        req1 = {s for s, v in f1.items() if v["audit_required"]}
        rev(2, srcs, {"src_capname": {"live": 3, "role_c": True}})
        f2 = SH.fold(d)
        req2 = {s for s, v in f2.items() if v["audit_required"]}
        check("the fold: a source drawn after revision 1 still needs his audit after revision 2; a role C source is "
              "never drawn", req1 == s1 and s1 <= req2 and "src_capname" in f2 and "src_capname" not in req2,
              (sorted(req1), sorted(req2)))


# ------------------------------------------------------------------ the sheets (Revision item 5)
def t_sheets():
    print("the sheets: every answer given, the owner's audit his, the whole corrected MT-W label, the lab's pull")
    with inc("inc_sheets") as d:
        it = [{"id": "src_a|admitted|k%d" % i, "group": "admitted", "source": "src_a", "key": "k%d" % i,
               "image": png(d / "i" / ("a%d.png" % i), 3000 + i), "boxes": [["inc:12", .5, .5, .2, .2]]}
              for i in range(3)]
        SH.write_bundle("source_p1_r001_src_a", it, {"kind": "source", "source": "src_a", "screen": {
            "units": {"inc:12": {"boxes": 3, "suggested": "unresolved"}}}}, d)
        jl(SH.decisions_path(d), [{"kind": "source", "source": "src_a", "decision": "accept", "setting": "Field",
                                   "roles": {"inc:12": "weed"}, "decided_by": "delegate", "sheet_sha256": "x"}])
        r = SH.render("audit", 1, inc=d, sources=["src_a"])
        html = pathlib.Path(r["html"]).read_text(encoding="utf-8")
        dec_radios = [x for x in html.split("<input") if 'name="dec|src_a"' in x]
        check("the owner's audit sheet pre-selects no decision, setting or role (the delegate's shown beside)",
              dec_radios and not any(" checked" in x.split(">")[0] for x in dec_radios) and
              html.count("— 请选择 choose —") >= 2 and "The delegate decided" in html and "delegate: weed" in html)
        doc = {"format": SH.FMT_DECISIONS, "sheet": {"kind": "audit", "batch": 1, "spec_sha256": r["spec_sha256"]},
               "decider": "Harry", "saved_utc": "2026-11-12T00:00:00Z",
               "sources": {"src_a": {"decision": "reject", "setting": "Field", "roles": {"inc:12": "crop"}}}}
        f = TMP / "audit.json"
        f.write_text(json.dumps(doc))
        check("an audit file recorded under another person's id is refused; under the owner's it is his",
              refused(SH.record, str(f), "human:someone@else.org", inc=d) is not None and
              SH.record(str(f), SH.OWNER_HUMAN, inc=d)[0]["decided_by"] == SH.OWNER_HUMAN)
        ln_items = [{"id": "ln|s%d" % i, "ln_group": "train_core", "key": "l%d" % i,
                     "image": png(d / "i" / ("l%d.png" % i), 3100 + i), "boxes": [["box", .5, .5, .2, .2]]}
                    for i in range(2)]
        SH.write_bundle("ln_001", ln_items, {"kind": "ln"}, d)
        rl = SH.render("ln", 1, inc=d)
        hl = pathlib.Path(rl["html"]).read_text(encoding="utf-8")
        ln = {"format": SH.FMT_DECISIONS, "sheet": {"kind": "ln", "batch": 1, "spec_sha256": rl["spec_sha256"]},
              "decider": "Harry", "saved_utc": "2026-11-12T00:00:00Z",
              "images": {"ln|s0": {"marks": ["correct"], "missing": 0, "checked": True},
                         "ln|s1": {"marks": ["correct"], "missing": 0, "checked": False}}}
        f2 = TMP / "ln.json"
        f2.write_text(json.dumps(ln))
        e = refused(SH.record, str(f2), SH.OWNER_HUMAN, inc=d)
        ln["images"]["ln|s1"]["checked"] = True
        f2.write_text(json.dumps(ln))
        check("an LN image not marked checked refuses the file (a sheet saved untouched is no measurement); each image "
              "has its checkbox", e is not None and 'name="seen|ln|s0"' in hl and
              len(SH.record(str(f2), SH.OWNER_HUMAN, inc=d)) == 2)
        mt_items = [{"id": "mtw|src_m|m%d" % i, "source": "src_m", "key": "m%d" % i, "total": 2, "flagged": False,
                     "image": png(d / "i" / ("m%d.png" % i), 3200 + i),
                     "boxes": [["weed", .5, .5, .2, .2], ["weed", .2, .2, .1, .1], ["weed", .8, .8, .1, .1]]}
                    for i in range(2)]
        SH.write_bundle("mtw_v2_src_m", mt_items, {"kind": "mtw", "source": "src_m", "total": 2}, d)
        rm = SH.render("mtw", 1, inc=d)
        hm = pathlib.Path(rm["html"]).read_text(encoding="utf-8")
        ok_radios = [x for x in hm.split("<input") if 'name="ok|mtw|src_m|m0"' in x]
        mt = {"format": SH.FMT_DECISIONS, "sheet": {"kind": "mtw", "batch": 1, "spec_sha256": rm["spec_sha256"]},
              "decider": "Harry", "saved_utc": "2026-11-12T00:00:00Z",
              "images": {"mtw|src_m|m0": {"verdict": "error", "error": True, "removed": [0],
                                          "added": [[0.4, 0.4, 0.1, 0.1]]},
                         "mtw|src_m|m1": {"removed": [], "added": []}}}
        f3 = TMP / "mt.json"
        f3.write_text(json.dumps(mt))
        e3 = refused(SH.record, str(f3), SH.OWNER_HUMAN, inc=d)
        mt["images"]["mtw|src_m|m1"].update(verdict="correct", error=False)
        f3.write_text(json.dumps(mt))
        rows = SH.record(str(f3), SH.OWNER_HUMAN, inc=d)
        corr = next(x for x in rows if x["item"].endswith("m0"))["corrected_boxes"]
        check("an MT-W image not answered refuses the file; each image answers correct or in error, nothing pre-set",
              e3 is not None and len(ok_radios) == 2 and not any(" checked" in x.split(">")[0] for x in ok_radios))
        check("  a correction is the image's whole corrected label: its boxes without the one removed, with the one "
              "drawn", corr == [[.2, .2, .1, .1], [.8, .8, .1, .1], [0.4, 0.4, 0.1, 0.1]], corr)
        chk = SH.mtw_checks(d)["src_m"]
        check("  the check passes (whole source, its error corrected) and its corrected label is the whole one",
              chk["verdict"] == "pass" and chk["corrected_labels"]["m0"] == corr, chk)
        jl(TMP / "inc_sheets_err" / "d.jsonl", [])
        bad = {"src_e": [{"kind": "mtw", "source": "src_e", "item": "mtw|src_e|e0", "error": True, "verdict": "error",
                          "removed": [], "added": [], "total": 1}]}
        dp = SH.decisions_path(TMP / "inc_sheets_err")
        jl(dp, bad["src_e"])
        check("  a whole source with an error left uncorrected fails", SH.mtw_checks(TMP / "inc_sheets_err")["src_e"][
            "verdict"] == "fail")
        # LN and MT-W batches in the order their bundles were written: a later bundle never renumbers a batch
        SH.write_bundle("mtw_v1_aaa", mt_items[:1], {"kind": "mtw", "source": "aaa", "total": 1}, d)
        check("MT-W batches in the order their bundles were written (a later mtw_v1_aaa after mtw_v2_src_m)",
              SH.bundles("mtw", d) == ["mtw_v2_src_m", "mtw_v1_aaa"], SH.bundles("mtw", d))
    # the lab: the cluster's sheets state pulled before the batch is listed
    lab = TMP / "lab_inc"
    cluster = TMP / "inc_sheets"
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        if "rsync" in argv and "--exclude" in argv:              # the pull (the push only records its call)
            dst = pathlib.Path(argv[-1])
            dst.mkdir(parents=True, exist_ok=True)
            src = SH.sheets_dir(cluster)
            for p in src.rglob("*"):
                rel = p.relative_to(src)
                if p.is_file() and rel.parts[0] not in ("rendered", "html"):
                    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(str(p), str(dst / rel))

        class R(object):
            returncode, stdout, stderr = 0, "", ""
        return R()
    r2 = SH.render("mtw", 2, inc=lab, pull=True, runner=runner)
    pulls = [a for a in calls if "rsync" in a and "--exclude" in a]
    check("a lab render pulls the cluster's sheets state first (bundles and decisions, not its renders or HTML), then "
          "lists its batch from it, renders and pushes its record", r2["bundles"][0]["name"] == "mtw_v1_aaa" and
          pulls and pulls[0][-2].endswith("/ds1/v1/sheets/") and "rendered/" in pulls[0] and
          any("rendered/" in str(a[-1]) for a in calls if a is not pulls[0]), calls[:2])
    check("  an audit batch rendered on the lab without its sources is refused", refused(
        SH.render, "audit", 2, inc=lab, pull=True, runner=runner) is not None)


# ------------------------------------------------------------------ sources out, ready pools, MT-W v1.1
def t_sources_out():
    print("a source rejected after its pool and exams: out of every manifest, MD and MT-W")
    with inc("inc_out") as d:
        sd = d / "splits" / "ds1" / "screen"
        sd.mkdir(parents=True)
        (sd / "p1_r001.json").write_text(json.dumps({"sources": {"src_r": {"live": 3}, "src_k": {"live": 3},
                                                                 "src_q": {"live": 2, "quarantine_whole": True}}}))
        for s in ("src_r", "src_k"):
            p = DD.out_dir() / "pools" / "p1" / ("%s.jsonl" % DD.slug(s))
            jl(p, [{"key": "%s_%d" % (s, i), "source": s, "image": "/x", "sha256": "s", "boxes": []} for i in range(3)])
            p.with_suffix(".json").write_text(json.dumps({"source": s}))
            for exam in (DD.mdn_exam(s), DD.mtw_exam(s)):
                fz = DD.exam_dir(exam) / "frozen.json"
                fz.parent.mkdir(parents=True, exist_ok=True)
                fz.write_text(json.dumps({"exam": exam, "frozen": True, "images": 3, "source": s,
                                          "root": str(d / ("root_%s" % exam))}))
        decs = {"src_r": {"decision": "reject"}, "src_k": {"decision": "accept_training_only"}}
        rp = DD.ready_pools(1, DD.Inputs(decisions=decs))
        check("ready_pools: the rejected source's pool never enters a manifest; the ready one does", set(rp) == {"src_k"},
              sorted(rp))
        jl(SH.decisions_path(d), [{"kind": "source", "source": "src_r", "decision": "accept", "setting": "Field",
                                   "roles": {}, "decided_by": "delegate", "sheet_sha256": "x"},
                                  {"kind": "audit", "source": "src_r", "decision": "reject", "setting": "Field",
                                   "roles": {}, "decided_by": SH.OWNER_HUMAN, "sheet_sha256": "y"}])
        out = DD.sources_out()
        check("sources_out: the owner's rejection after the delegate's accept, and a quarantined source",
              out == {"src_r", "src_q"}, out)
        check("  its MD-N is never scored (md_exams) and its MT-W v2 never in MT-W (mtw_roots)",
              D.md_exams() == [DD.mdn_exam("src_k")] and [x[2] for x in D.mtw_roots()][-1] == "mtw_v2:src_k" and
              not any("src_r" in x[2] for x in D.mtw_roots()), (D.md_exams(), [x[2] for x in D.mtw_roots()]))
        # MT-W v1.1: a failing test v1 source leaves the headline even when nothing was corrected
        hd = B3.out_dir() / B3.HOLDOUT_DIR
        for s, n in (("rf_unitec", 3), ("rf_ok", 3)):
            rows = []
            for i in range(n):
                img = png(d / "tv1" / ("%s_%d.png" % (s, i)), 4000 + i + (10 if s == "rf_ok" else 0))
                lab = label(d / "tv1" / ("%s_%d.txt" % (s, i)), [(12, .5, .5, .2, .2), (12, .2, .2, .1, .1)])
                rows.append({"key": "%s__%d" % (s, i), "source": s, "image": img, "label": lab,
                             "sha256": C.sha256_file(img), "label_sha256": C.sha256_file(lab), "session": "",
                             "group": "g%s%d" % (s, i)})
            jl(hd / ("%s.jsonl" % s), rows)
        st = DD.freeze_mtw_v1_1({"rf_unitec": {"verdict": "fail", "sha256": "a"},
                                 "rf_ok": {"verdict": "pass", "sha256": "b", "corrected_labels": {}}})
        man = C.read_manifest(pathlib.Path(st["root"]["root"]) / "splits" / "v1" / "mtw_v2.jsonl")
        check("MT-W v1.1 written when a source fails its check with no correction: the failing source's rows out",
              st["left_mtw"] == ["rf_unitec"] and st["corrections"] == 0 and {r["source"] for r in man} == {"rf_ok"},
              st)
        g = D.mtw_groups()
        check("mtw_groups: test v1's capture groups (the 0.90 interval's group SE)", g.get("rf_ok__0") == "v1:grf_ok0"
              and len(g) == 6, g)
        # DS1's own DINOv2 index reads the MD and MT-W rows' files from the never-train list
        DD.append_never_train([{"key": "nk1", "source": "src_k", "image": "/files/nk1.png", "sha256": "s1"}],
                              DD.mdn_exam("src_k"), {"nk1": "g1"})
        saved = DD.eval_group_rows
        DD.eval_group_rows = lambda: {}
        try:
            own = DD.own_exam_rows()
        finally:
            DD.eval_group_rows = saved
        check("DS1's own DINOv2 index describes every MD and MT-W row (the never-train list keeps its file)",
              {"key": "%s|nk1" % DD.mdn_exam("src_k"), "image": "/files/nk1.png", "sha256": "s1"} in
              own.get("ds1_exams", []), own.get("ds1_exams"))


# ------------------------------------------------------------------ the reads, the verdicts, the arms
def t_core():
    print("the reads, the substitution, the verdicts' arms, the 0.90 SE, a sealed read resumed")
    v1 = {"status": "decided", "primaries": {"P1": {"confirmed": True, "diff": 0.01}}}
    check("no sealed read before verdict v2, the challenger among them", D.read_order(v1, None) == [])
    check("  after v2: S0x, S2x, the challenger, the cwd12 test", [o["kind"] for o in D.read_order(
        v1, {"status": "decided"})] == ["reference", "candidate", "challenger", "cwd12"])
    check("the substitution when S2x's smoke recorded it not run (walltime), before the date",
          D.substitution_due("2026-11-20", True, s2x_walltime=True) and not D.substitution_due("2026-11-20", True))
    e = refused(D.verdict, 1, {a: "done" for a in ("K0", "KD", "KT", "KF", "S0", "S1CR", "S1C")}, write=False)
    e2 = refused(D.verdict, 2, {"S0X": "done", "S2": "done"}, write=False, on_date="2026-11-20")
    check("a verdict is refused while an arm of its phase has no score record and is not recorded failed or not run "
          "(S1U at v1, S2X at v2)", e is not None and "S1U" in str(e) and e2 is not None and "S2X" in str(e2), (e, e2))
    keys = ["md%03d" % i for i in range(30)]
    oo = ["ood23__2023080%d_iPhone12_WY_%d" % (i % 5, i) for i in range(30)]
    dev = ["dev%03d" % i for i in range(30)]
    gp = D.splits_dir() / "md_v1" / "groups.jsonl"
    jl(gp, [{"exam": "md_s0", "key": k, "group": "g%d" % (i // 2)} for i, k in enumerate(keys)])
    devp = C.manifest_path("dev")
    jl(devp, [{"image": "/x/%s.jpg" % k, "label": "/x/%s.txt" % k, "sha256": "0" * 64, "label_sha256": "0" * 64,
               "source": "cwd12", "session": "s%d" % (i % 4), "key": k} for i, k in enumerate(dev)])
    D.RESAMPLES = 30

    def loader(arm, exam, kind):
        out = []
        for s in range(3):
            if kind == "species":
                a = agn_arrays(30, 50 + s + len(arm), 0.6, dev)
                a = {"keys": a["keys"], "tp": a["tp"], "conf": a["conf"], "pred_img": a["pred_img"],
                     "pred_cls": np.zeros(len(a["conf"]), dtype=np.int64),
                     "target_img": np.repeat(np.arange(30), 3), "target_cls": np.zeros(90, dtype=np.int64)}
                from weed_optimizer_framework.tools.inc2 import v3fam as F
                out.append(({"species_map50_95": F.statistic_value(a, "species")}, a))
            else:
                out.append(({}, agn_arrays(30, 10 + s + len(arm), 0.6, keys if exam.startswith("md") else oo)))
        return out
    doc, _ext = D.verdict(2, {"S0X": "done", "S1UX": "done", "S1U": "done", "S0": "done", "S1C": "done"},
                          loader=loader, on_date="2026-11-20", s2_frozen=True, write=False, s2x_walltime=True)
    check("verdict v2 with S2x not run (walltime): P3 decided on S1ux over S0x, the substitution named",
          doc["primaries"]["P3"]["x"] == "S1UX" and doc["substitution"]["applied"] and
          doc["substitution"]["s2x_walltime"], doc.get("substitution"))
    with inc("inc_states") as d:
        jl(D.splits_dir() / "arms" / "k_not_run.json", [])
        (D.splits_dir() / "arms" / "k_not_run.json").write_text(json.dumps(
            {"arms": ["KD", "KT"], "status": "not run (the K source enters no DS1 manifest: rejected)"}))
        sp = LG.arm_dir("S2X") / "smoke.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"status": LG.NOT_RUN_WALLTIME}))
        st = D.arms_state_from_records()
        check("the arms' states from the records: KD and KT not run (k_not_run.json), S2X not run (walltime), the "
              "rest missing", st["KD"].startswith("not run") and st["KT"].startswith("not run") and
              st["S2X"] == LG.NOT_RUN_WALLTIME and st["S1U"] == "missing" and D.s2x_walltime_not_run(), st)
        # the 0.90 interval with the capture-group SE (test_report's own groups)
        root = d / "z1root"
        jl(root / "splits" / "v1" / "test_v1.jsonl", [{"key": "t%02d" % i, "source": "rf_x", "image": "/i",
                                                       "label": "/l", "sha256": "0", "label_sha256": "0",
                                                       "session": ""} for i in range(24)])
        jl(B3.out_dir() / B3.HOLDOUT_DIR / "rf_x.jsonl", [{"key": "t%02d" % i, "source": "rf_x", "group": "G%d" % (
            i // 6)} for i in range(24)])
        td = D.test_dir("S0X", "mtw")
        for s in range(3):
            sd = td / ("s%d" % s) / "test_v1"
            sd.mkdir(parents=True, exist_ok=True)
            a = agn_arrays(24, 70 + s, 0.7, ["t%02d" % i for i in range(24)])
            sha = SC.save_npz(sd / "test_v1.weed.npz", a)
            (sd / "test_v1.json").write_text(json.dumps({"format": D.FMT_SCORE, "stratum": "test_v1",
                                                         "root": str(root), "weed": {"name": "test_v1.weed.npz",
                                                                                     "sha256": sha}}))
        (td / "done.json").write_text(json.dumps({"read": "x"}))
        rep = D.test_report("S0X", "mtw", write=False)
        check("the MT-W report's 0.90 reading takes SE = max(image, capture-group SE) from MT-W's own groups",
              rep["groups"] == 4 and rep["se"] >= rep["se_image"] > 0 and rep["reading"]["se"] == rep["se"],
              {k: rep.get(k) for k in ("groups", "se", "se_image")})
    t_resume()


def t_resume():
    with inc("inc_resume") as d:
        fake_timestamp()
        rd = D.reads_dir() / "1_S0X_mtw"
        rd.mkdir(parents=True)
        (rd / "read.json").write_text(json.dumps({"format": D.FMT_READ, "arm": "S0X", "exam": "mtw"}))
        saved = {k: getattr(D, k) for k in ("build_source", "mtw_roots", "pass_in_root", "commit")}
        from weed_optimizer_framework.tools.inc2 import zoo as Z
        z_root = Z.root_dir
        passes = []
        fail = {"n": 1}

        def pass_in_root(w, ex, mode, arm, out, root_dir=None, read_json=None, **kw):
            if fail["n"] and len(passes) == 2:
                fail["n"] = 0
                raise RuntimeError("node failure")
            passes.append((w, ex))
            return {"result": {}, "weed": None}

        def commit(arm, s, ex, ps, cfg, source=None, extra=None, d=None, **kw):
            js = pathlib.Path(d) / D.file_names(ex)[0]
            js.parent.mkdir(parents=True, exist_ok=True)
            js.write_text(json.dumps({"format": D.FMT_SCORE, "arm": arm, "seed": s, "exam": ex}))
            return D.entry(js, "written")
        try:
            D.build_source = lambda arm, s: {"weights": "w%d" % s}
            D.mtw_roots = lambda: [("test_v1", d / "z", "test_v1")]
            Z.root_dir = lambda v: d / "z"
            D.pass_in_root, D.commit = pass_in_root, commit
            e = None
            try:
                D.score_test("S0X", "mtw")
            except RuntimeError as ex:
                e = ex
            first = list(passes)
            files = D.score_test("S0X", "mtw")
        finally:
            for k, v in saved.items():
                setattr(D, k, v)
            Z.root_dir = z_root
        check("a sealed read that failed partway (node failure after two parts) completes when re-submitted: the parts "
              "committed are kept, never scored again", e is not None and len(first) == 2 and len(passes) == 6 and
              len(files) == 6 and sum(1 for f in files if f["status"] == "kept") == 2 and
              (D.test_dir("S0X", "mtw") / "done.json").is_file(), (len(first), len(passes), [f["status"] for f in files]))


# ------------------------------------------------------------------ the long path's logs
def t_long():
    print("the long path keeps its train logs (data and checkpoints removed, recorded), an earlier attempt aside")
    with inc("inc_long"):
        fake_timestamp()
        d = TMP / "imgs_long"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(50):
            lab = d / ("l%d.txt" % i)
            lab.write_text("12 0.5 0.5 0.2 0.2\n")
            rows.append({"image": str(d / ("x%d.jpg" % i)), "label": str(lab), "sha256": "%064d" % i,
                         "label_sha256": C.sha256_file(lab), "source": "s", "session": "", "key": "S0_%05d" % i})
        DD.write_arm("S0", rows)

        def trainer(rows_, rcp, out_dir, rec, times):
            times.append((100.0, 101.0))
            out = pathlib.Path(out_dir)
            (out / "data" / "images").mkdir(parents=True, exist_ok=True)
            (out / "data" / "images" / "x.png").write_bytes(b"x")
            (out / "train" / "weights").mkdir(parents=True, exist_ok=True)
            (out / "train" / "results.csv").write_text("epoch,loss\n1,0.5\n")
            w = out / "train" / "weights" / "last.pt"
            w.write_bytes(b"weights")
            (out / "train" / "weights" / "best.pt").write_bytes(b"best")
            return w, out / "train"
        work = LG.arm_dir("S0X") / "smoke_work"
        (work / "train").mkdir(parents=True)
        (work / "train" / "results.csv").write_text("an earlier attempt\n")
        sm = LG.smoke("S0X", trainer=trainer)
        aside = pathlib.Path(sm["earlier_attempt"]["moved_to"])
        check("the smoke: an earlier attempt's work moved aside with its logs (never deleted); this run's logs kept, its "
              "data and checkpoints removed and recorded", (aside / "train" / "results.csv").read_text() ==
              "an earlier attempt\n" and (work / "train" / "results.csv").is_file() and not (work / "data").exists()
              and not (work / "train" / "weights" / "last.pt").exists() and
              any(x.endswith("last.pt") for x in sm["cleanup_removed"]), sm.get("cleanup_removed"))
        tr = LG.train("S0X", 0, trainer=trainer)
        sw = LG.arm_dir("S0X") / "s0" / "work"
        check("a seed: final_ema.pt exported, the train logs kept, data and checkpoints removed (train.json lists them)",
              (LG.arm_dir("S0X") / "s0" / "final_ema.pt").read_bytes() == b"weights" and
              (sw / "train" / "results.csv").is_file() and not (sw / "data").exists() and
              not (sw / "train" / "weights" / "best.pt").exists() and tr["cleanup_removed"], tr.get("cleanup_removed"))


class HueEmbedder(object):
    """A stand-in for the calibrated embedder: a hue histogram, L2-normalised (counts its calls)."""
    dim = 12

    def __init__(self, name="facebook/dinov2-base:cls"):
        self.name = name
        self.calls = 0

    def __call__(self, pils):
        self.calls += 1
        out = []
        for p in pils:
            hsv = np.asarray(p.convert("HSV"), dtype=np.int64)
            h = np.bincount((hsv[..., 0].reshape(-1) * self.dim) // 256, minlength=self.dim).astype(np.float32)
            out.append(h / (np.linalg.norm(h) or 1.0))
        return np.stack(out)


def t_memo():
    print("the job's descriptors described once across DS1's indexes (the same copies as EmbedScanner.scan)")
    from weed_optimizer_framework.tools.inc2 import guard as G
    d = TMP / "memo"
    def solid(path, hue):
        # one hue, random brightness: hue histograms apart, dHashes apart
        path.parent.mkdir(parents=True, exist_ok=True)
        v = np.random.default_rng(int(hue) + 7).integers(60, 255, (48, 64), dtype=np.uint8)
        hsv = np.stack([np.full((48, 64), int(hue), np.uint8), np.full((48, 64), 255, np.uint8), v], axis=-1)
        Image.fromarray(hsv, "HSV").convert("RGB").save(path)
        return str(path)
    ref = [{"key": "r%d" % i, "image": solid(d / ("r%d.png" % i), i * 21), "sha256": None} for i in range(6)]
    emb = HueEmbedder()
    idx = G.eval_index({"ref": ref}, emb, d / "ref_desc.npz")
    cal = {"embedder": emb.name, "cos_threshold": 0.999, "dhash_bits_max": 2, "file": "x", "format": "y",
           "calibration": {"ok": True, "cos_threshold": 0.999, "dhash_bits_max": 2}}
    sc = G.EmbedScanner(idx, cal)
    items = [{"key": "q%d" % i, "image": ref[i]["image"] if i < 2 else solid(d / ("q%d.png" % i), 140 + i * 21),
              "sha256": None} for i in range(5)]
    plain, _u = sc.scan(items)
    DD._DESC_MEMO.clear()
    c0 = emb.calls
    memo, _u2 = DD._memo_scan(items, sc)
    c1 = emb.calls
    memo2, _u3 = DD._memo_scan(items, sc)
    check("the same copies as the scanner's own scan (q0, q1 are r0, r1's files); a second index's scan of the same "
          "images describes nothing again", sorted(plain) == sorted(memo) == sorted(memo2) == ["q0", "q1"] and
          all(memo[k][0]["eval_key"] == plain[k][0]["eval_key"] for k in memo) and c1 > c0 and emb.calls == c1,
          (sorted(plain), sorted(memo), c0, c1, emb.calls))
    hits = DD.embed_hits([dict(it) for it in items], sc)
    check("  embed_hits takes that path for a calibrated scanner", sorted(hits) == ["q0", "q1"], sorted(hits))


def main():
    for fn in (t_copy_rules, t_screen, t_source_classes, t_draw_exclusions, t_audit_sample, t_sheets, t_sources_out,
               t_core, t_long, t_memo):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2500:]))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
