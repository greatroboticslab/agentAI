#!/usr/bin/env python3
"""V3F's system arms: E3RS and D8 (inc2/v3fam_systems.py), D3 (inc2/v3fam_wbf.py)
and D7's software check through Ultralytics' validator path
(tools/latency/two_stage_latency.py); docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-11): V3F, the v3 experiment family D0-D9", V3F.5 E3RS, D3, D7, D8.

Two groups. UNITS need no scoring pass (V3FAM_ONLY=units runs only them):
- the constants (the system arms, D8's builds and finals, D3's fixed WBF
  setting and the pinned ensemble_boxes file, the scaffold) and the six
  E3-frozen sha256s of V3F.1, asserted literally;
- the fusion against a hand computation on a 3-box case (score-weighted
  coordinates, the averaged score times min(3, n) / 3), the normalisation
  round trip, the 300 cap, a fourth seed refused, a box outside its image
  refused, another library file refused;
- D8's build arguments (LOCK v2's train_core by sha256, s640 / m640, seeds
  0-2, role baseline, finals dev and ImageWeeds, the cold table) and their
  refusals.
WORLD runs real Ultralytics CPU passes in the synthetic world of
tests/test_inc2_e3.py (E3's own score-arm writes E3's records first, as in
tests/test_inc2_v3fam.py) with a zoo root holding ooddev_v1:
- E3RS: the reproductions, E3-M, E3-A, E3-B through the family path equal to
  E3's recorded dev files exactly, the reference equal to its recorded dev
  score, O5's oracle reading on E3-B's own boxes, E3-B on ImageWeeds and
  ooddev_v1 for E1-A, E1-B and b_v2_m640 in the zoo's root (nothing written
  there), the dev-only platform record, a rerun keeping every file;
- D3: the fused boxes through inject_boxes (one system), the library and the
  inputs recorded, the P4 comparison computable on the arrays, E3-B's own
  stored stage-1 rows injected reproducing its family score, ImageWeeds and
  ooddev_v1 reported, a planted fourth seed refused, a rerun keeping it;
- D8: the builds' definitions, the scoring of six seeds through the identity
  pass (equal to the recorded protocol scores), the reported passes, D8-m's
  reference read's path (D8-s has none), a definition on another manifest
  refused;
- D7: the stage-1 boxes of Ultralytics' validator on a label-free view equal
  the family stage1 pass's (the software check's path), no label opened;
- the CLI.

Run:  python3 tests/test_inc2_v3fam_systems.py      (V3FAM_ONLY=units: the units only)
      ensemble_boxes 1.0.9 must import (V3FAM_ENSEMBLE_BOXES=<dir of the unpacked wheel> is put on sys.path;
      a numba the environment lacks is replaced by a pass-through jit, which the fusion never calls)
"""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)
import test_inc2_e3 as E3T  # noqa: E402
import test_inc2_native as N  # noqa: E402
import test_inc2_v3fam as FT  # noqa: E402

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import common as C2  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_wbf as WB  # noqa: E402

FAILURES = W.FAILURES
check = W.check
patched = E3T.patched
moved = E3T.moved
tamper = E3T.tamper
KW = FT.KW
EMB = E3T.EMB
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LATENCY = REPO_ROOT / "weed_llm_benchmark" / "weed_optimizer_framework" / "tools" / "latency" / "two_stage_latency.py"
WORLD = {}
FROZEN = {"twostage": "8fe2edc0e0d2a00d07182c3746407281a112c9e50c76662bc9f48a558b328ddc",
          "scorer": "18c008371cd2c021104d034d06b837564016f4509e8a969c2a3f850b09dd2de4",
          "scorer_sidecar": "de7aacda903780b32944866be0b4daa5c6b1f76e2c86965f7a99d85f443d5e52",
          "scorer_native": "ba9eeb928a2b5a0756ae8ac0144b54df2fdfb3570fd011f66b4a8022039d6b7d",
          "verify": "602eba3c935f375179d6f922dce28bbf89006054e5bcec5894f0c9cc7b87aa10",
          "semisup_labeler": "03b28c50699b30a32720e383ff2b713a2638926650007acb92a367d1f450458c"}


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (F.V3famRefused, TS.E3Refused, S.ScorerRefused) as e:
        return e
    return None


def ensemble_boxes_ready():
    """ensemble_boxes importable (V3F's pinned 1.0.9): V3FAM_ENSEMBLE_BOXES is put on sys.path, and a missing
    numba (which only the package's other fusions use) is replaced by a pass-through jit."""
    d = os.environ.get("V3FAM_ENSEMBLE_BOXES")
    if d and d not in sys.path:
        sys.path.insert(0, d)
    try:
        import numba  # noqa: F401
    except Exception:                        # noqa: BLE001 - absent or incompatible with this numpy
        stub = types.ModuleType("numba")

        def jit(*a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return lambda f: f
        stub.jit = jit
        sys.modules["numba"] = stub
    try:
        import ensemble_boxes  # noqa: F401
        return True
    except ImportError:
        return False


def latency_module():
    spec = importlib.util.spec_from_file_location("v3fam_two_stage_latency", str(LATENCY))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_ooddev_root():
    """Z1's root for the tests: ooddev_v1 (weed boxes only, class 12) with its manifest, LOCK and exam."""
    root = F.root_of("ooddev_v1")
    rows = []
    for i in range(4):
        img = W.TMP / "src" / "ooddev" / ("od_%03d.jpg" % i)
        W.make_image(img, 950 + i)
        rows.append(W.row_for("od_%03d" % i, img, [(12, 0.4, 0.4, 0.3, 0.3), (12, 0.72, 0.7, 0.2, 0.25)],
                              "rf_main-otq0a__weed-in-paddy-field-wmjlr"))
    (root / "splits" / "v1").mkdir(parents=True, exist_ok=True)
    sha = C.write_manifest(root / "splits" / "v1" / "ooddev_v1.jsonl", rows)
    (root / "splits" / "v1" / "LOCK.json").write_text(json.dumps({"manifests": {"ooddev_v1": sha}, "scorer_sha256":
                                                                  W.sha(pathlib.Path(S.__file__).resolve())}))
    C.materialise(rows, root / "exams" / "v1" / "ooddev_v1")
    return rows


def root_tree():
    r = F.root_of("ooddev_v1")
    return sorted(str(p.relative_to(r)) for p in r.rglob("*"))


# ================================================================== UNITS
def u_constants():
    print("the system arms' constants (pre-registered)")
    check("the system arms E3RS, D3, D6, D8 are family score arms; E3's arms through the family path in E3M, E3A, "
          "E3B, the reference in REF, E1's one-class boxes in E1A, E1B",
          SYS.SYSTEM_ARMS == ("E3RS", "D3", "D6", "D8") and set(SYS.SYSTEM_ARMS) <= set(F.SCORE_ARMS)
          and SYS.E3_DIRS == {"M": "E3M", "A": "E3A", "B": "E3B"} and SYS.REF_DIR == "REF"
          and SYS.E1_DIRS == {"A": "E1A", "B": "E1B"} and SYS.ORACLE_VARIANT == "oracle")
    check("D8: v3_d8_s640_tc (s640) and v3_d8_m640_tc (m640) on LOCK v2's train_core, finals dev and ImageWeeds, "
          "role baseline; D8-m the reference read",
          SYS.D8_ARMS == {"D8S": {"exp": "v3_d8_s640_tc", "arm": "s640"}, "D8M": {"exp": "v3_d8_m640_tc",
                                                                                "arm": "m640"}}
          and SYS.D8_MANIFEST == "train_core" and SYS.D8_FINALS == ("dev", "imageweeds") and SYS.D8_ROLE == "baseline"
          and SYS.D8_READ == "D8M" and SYS.D8_READ in F.REFERENCE_READS and "D8S" not in F.REFERENCE_READS
          and set(SYS.D8_ARMS) <= set(F.BUILD_ARMS))
    check("D3: iou_thr 0.55, skip_box_thr 0.001, conf_type avg, weights [1, 1, 1], one label, 300 per image; "
          "ensemble_boxes 1.0.9 by its file's sha256; E1-B seed 0 the scaffold; inputs E3RS's E3-B records "
          "(ooddev_v1: E1-B's)", WB.WBF_IOU_THR == 0.55 and WB.WBF_SKIP_BOX_THR == 0.001
          and WB.WBF_CONF_TYPE == "avg" and WB.WBF_WEIGHTS == (1, 1, 1) and WB.MAX_FUSED == 300
          and WB.ENSEMBLE_BOXES_VERSION == "1.0.9"
          and WB.ENSEMBLE_BOXES_WBF_SHA256 == "6189c54cf9582d1f7afadcb1ceb751c39e86e8e2ae7a35b7f0f44db1c2fe4efb"
          and WB.SCAFFOLD == ("B", 0) and WB.INPUT_DIRS == {"dev": "E3B", "imageweeds": "E3B", "ooddev_v1": "E1B"}
          and WB.ONE_CLASS_ID == C.OTHER_PLANT)


def u_frozen():
    print("the six E3-frozen files (V3F.1), asserted literally")
    now = {k: C.sha256_file(pathlib.Path(v).resolve()) for k, v in TS.CODE_FILES.items()}
    check("FROZEN_SHA256 is V3F.1's six sha256s, and the files on disk hash so", F.FROZEN_SHA256 == FROZEN
          and now == FROZEN, {k: now.get(k, "")[:12] for k in FROZEN})


def u_wbf():
    print("D3's fusion (ensemble_boxes 1.0.9), by hand")
    if not ensemble_boxes_ready():
        check("ensemble_boxes is importable (set V3FAM_ENSEMBLE_BOXES to the unpacked 1.0.9 wheel)", False)
        return
    fn, impl = WB.wbf_impl()
    check("the library: ensemble_boxes 1.0.9, ensemble_boxes_wbf.py by its pinned sha256", impl["version"] == "1.0.9"
          and impl["sha256"] == WB.ENSEMBLE_BOXES_WBF_SHA256, impl)
    with patched(WB, "ENSEMBLE_BOXES_WBF_SHA256", "0" * 64):
        e = refused(WB.wbf_impl)
    check("  another file is refused", e is not None, e)
    s0 = {"xyxy": [[10, 10, 50, 50]], "q": [0.9]}
    s1 = {"xyxy": [[12, 10, 52, 50], [70, 70, 90, 90]], "q": [0.6, 0.5]}
    s2 = {"xyxy": [[10, 12, 50, 52]], "q": [0.3]}
    xy, sc, st = WB.fuse_image([s0, s1, s2], 100, 100, fn)
    want_box = [(0.9 * 10 + 0.6 * 12 + 0.3 * 10) / 1.8, (0.9 * 10 + 0.6 * 10 + 0.3 * 12) / 1.8,
                (0.9 * 50 + 0.6 * 52 + 0.3 * 50) / 1.8, (0.9 * 50 + 0.6 * 50 + 0.3 * 52) / 1.8]
    check("the 3-box case by hand: one cluster at the score-weighted coordinates (%s) with score mean(0.9, 0.6, 0.3) x "
          "3/3 = 0.6; the lone box at 0.5 x 1/3; score order" % np.round(want_box, 4),
          len(sc) == 2 and np.allclose(xy[0], want_box, atol=1e-3) and abs(sc[0] - 0.6) < 1e-6
          and np.allclose(xy[1], [70, 70, 90, 90], atol=1e-3) and abs(sc[1] - 0.5 / 3) < 1e-6
          and st["rows_in"] == 4 and st["kept"] == 2, (xy, sc, st))
    a = np.array([[0, 0, 640, 480], [1.5, 2.25, 300.75, 479.0]])
    check("normalisation round trip (by the image's W, H) exact to 1e-9; the normalised boxes in [0, 1]",
          np.allclose(WB.denormalise(WB.normalise(a, 640, 480), 640, 480), a, atol=1e-9, rtol=0)
          and WB.normalise(a, 640, 480).max() <= 1.0)
    many = {"xyxy": [[i * 2.0, 0, i * 2.0 + 1.0, 1.0] for i in range(400)], "q": list(np.linspace(0.9, 0.01, 400))}
    empty = {"xyxy": np.zeros((0, 4)), "q": np.zeros(0)}
    xy, sc, st = WB.fuse_image([many, empty, empty], 1000, 10, fn)
    check("the 300 cap: 400 separate boxes fuse to 400, the 300 highest kept in score order (100 capped)",
          len(sc) == 300 and st["fused"] == 400 and st["capped"] == 100 and np.all(np.diff(sc) <= 0)
          and abs(sc[0] - 0.9 / 3) < 1e-6, st)
    out = {"a fourth seed": refused(WB.fuse_image, [s0, s1, s2, s0], 100, 100, fn),
           "two seeds": refused(WB.fuse_image, [s0, s1], 100, 100, fn),
           "a box outside its image": refused(WB.fuse_image, [{"xyxy": [[10, 10, 120, 50]], "q": [0.5]}, s1, s2],
                                              100, 100, fn),
           "boxes without scores": refused(WB.fuse_image, [{"xyxy": [[1, 1, 2, 2]], "q": []}, s1, s2], 100, 100, fn)}
    rec = {"keys": np.array(["a", "b"]), "img": np.array([0, 1]), "xyxy": np.array([[1, 1, 5, 5], [2, 2, 6, 6.0]]),
           "q": np.array([0.5, 0.4])}
    other = dict(rec, keys=np.array(["a", "c"]))
    out["seeds in another key order"] = refused(WB.fuse_exam, [rec, rec, other], {"a": (10, 10), "b": (10, 10)}, fn)
    out["four seeds of an exam"] = refused(WB.fuse_exam, [rec] * 4, {"a": (10, 10), "b": (10, 10)}, fn)
    out["sizes of other images"] = refused(WB.fuse_exam, [rec] * 3, {"a": (10, 10)}, fn)
    check("refused: %s" % ", ".join(out), all(e is not None for e in out.values()), {k: str(v)[:80]
                                                                                       for k, v in out.items()})
    fused, cnt = WB.fuse_exam([rec] * 3, {"a": (10, 10), "b": (10, 10)}, fn)
    check("fuse_exam: three identical seeds fuse to the same boxes at the same scores, every key present",
          sorted(fused) == ["a", "b"] and np.allclose(fused["a"]["xyxy"], [[1, 1, 5, 5]], atol=1e-5)
          and abs(float(fused["b"]["conf"][0]) - 0.4) < 1e-6 and cnt["kept"] == 2, (fused, cnt))


def u_d8_kwargs():
    print("D8's build arguments (L23I)")
    kw = SYS.d8_build_kwargs("D8M", exp="v3_d8_m640_tc", seeds="0,1,2", role="baseline")
    sha = C2.verify_manifest_against_lock_v2("train_core")
    check("D8M: LOCK v2's train_core (by sha256), arm m640, seeds 0-2, role baseline, finals dev and ImageWeeds, a "
          "v3fam record naming the arm and the cold table", kw["manifest"] == str(C2.v2_manifest_path("train_core"))
          and kw["arm"] == "m640" and kw["seeds"] == [0, 1, 2] and kw["role"] == "baseline"
          and kw["final_exams"] == ["dev", "imageweeds"] and kw["extra"]["v3fam"]["arm"] == "D8M"
          and kw["extra"]["v3fam"]["manifest_sha256"] == sha and kw["extra"]["v3fam"]["recipe"] == "cold", kw)
    out = {"another experiment": refused(SYS.d8_build_kwargs, "D8M", exp="v3_d8_s640_tc"),
           "seeds 0,1": refused(SYS.d8_build_kwargs, "D8S", seeds="0,1"),
           "role capacity": refused(SYS.d8_build_kwargs, "D8S", role="capacity"),
           "a test final": refused(SYS.d8_build_kwargs, "D8S", final_exams=("dev", "imageweeds", "test")),
           "an unknown arm": refused(SYS.d8_build_kwargs, "D8L")}
    check("refused: %s" % ", ".join(out), all(e is not None for e in out.values()), {k: str(v)[:80]
                                                                                       for k, v in out.items()})


def u_vs_recorded():
    print("a family identity score against a recorded one (D8, the reference)")
    rec = {"exam": "dev", "manifest_sha256": "m" * 64, "key_order_sha256": "k" * 64, "n_images": 8,
           "weights_sha256": "w" * 64, "n_gt": 10, "scorer_sha256": S.TEST_PREFIX + "a" * 64, "map50_95": 0.5,
           "species_map50_95": 0.4, "agnostic_map50_95": 0.6, "per_class": {"Waterhemp": 0.4, "OtherPlant": 0.1}}

    def off(dx, **over):
        d = json.loads(json.dumps(rec))
        d["per_class"]["Waterhemp"] += dx
        d.update(over)
        return d
    same = SYS.vs_recorded(off(0.0), rec, "x")
    dev = SYS.vs_recorded(off(0.001), rec, "x")
    out = {"0.01 off": refused(SYS.vs_recorded, off(0.01), rec, "x"),
           "other weights": refused(SYS.vs_recorded, off(0.0, weights_sha256="v" * 64), rec, "x"),
           "another exam's key order": refused(SYS.vs_recorded, off(0.0, key_order_sha256="j" * 64), rec, "x")}
    check("equal: passed; 0.001 off: a deviation; refused: %s" % ", ".join(out), same["status"] == "passed"
          and dev["status"] == "deviation" and all(e is not None for e in out.values()), (same, dev, out))


UNITS = ("u_constants", "u_frozen", "u_wbf", "u_d8_kwargs", "u_vs_recorded")


# ================================================================== WORLD
def w_setup():
    print("the world: E3's records (tests/test_inc2_v3fam.py's setup) and a zoo root holding ooddev_v1")
    FT.WORLD["rows"] = WORLD["rows"]
    FT.w_setup()
    WORLD["ooddev"] = build_ooddev_root()
    check("the zoo root holds ooddev_v1's manifest, LOCK and exam", (F.root_of("ooddev_v1") / "exams" / "v1" /
                                                                     "ooddev_v1" / "data.yaml").is_file())


def w_e3rs():
    print("E3RS: the comparators through the family path")
    t0 = time.time()
    tree0 = root_tree()
    rec = SYS.score_e3rs(embedder=EMB, **KW)
    WORLD["e3rs"] = rec
    got = {(f["dir"], f["seed"], f["variant"]) for f in rec["dev"]}
    want = {(d, s, None) for d in ("E3M", "E3A", "E3B", "REF") for s in F.SEEDS} | {("E3B", s, "oracle")
                                                                                     for s in F.SEEDS}
    check("E3RS writes 15 dev files (E3-M, E3-A, E3-B two-stage, the reference identity, O5 on E3-B) and the "
          "reproductions first (%.0fs)" % (time.time() - t0), got == want and all(
              (F.equiv_dir() / n).exists() for n in ("dev_identity.json", "injection.json", "two_stage_M.json",
                                                    "two_stage_A.json", "two_stage_B.json",
                                                    "identity_imageweeds.json", "identity_ooddev_v1.json")),
          sorted(got ^ want))
    same = []
    for letter, d in SYS.E3_DIRS.items():
        for s in F.SEEDS:
            fam = json.loads((F.arm_dir(d, s) / "dev.json").read_text())
            e3 = json.loads((TS.arm_dir(letter, s) / "dev.json").read_text())
            same.append(all(fam[k] == e3[k] for k in ("map50_95", "per_class", "species_map50_95",
                                                      "agnostic_map50_95"))
                        and fam["e3"]["status"] == "passed" and fam["settings"]["nms"] == TS.NMS[letter]
                        and fam["stage2"]["classifier"]["npz_sha256"] == TS.load_classifier()[5]["npz_sha256"])
    check("E3-M, E3-A, E3-B through the family path equal E3's recorded dev files exactly (each seed), E3's NMS "
          "per arm, E3's pinned classifier", all(same) and len(same) == 9, same)
    refs = []
    for s in F.SEEDS:
        fam = json.loads((F.arm_dir("REF", s) / "dev.json").read_text())
        rd = json.loads((C.INC_DIR / TS.REFERENCE / "runs" / ("final__base__s%d" % s) / "scores" / "dev.json")
                        .read_text())
        refs.append(fam["species_map50_95"] == rd["species_map50_95"] and fam["per_class"] == rd["per_class"]
                    and fam["vs_recorded"]["status"] == "passed" and fam["settings"]["mode"] == "identity")
    check("the reference's family identity scores equal its recorded protocol dev scores exactly", all(refs), refs)
    orc = []
    for s in F.SEEDS:
        o = json.loads((F.arm_dir("E3B", s) / "dev.oracle.json").read_text())
        b = json.loads((F.arm_dir("E3B", s) / "dev.json").read_text())
        orc.append((o["settings"]["mode"] == "oracle" and o["stage1_vs_e3"]["rows"] == b["stage1"]["n_boxes"]
                    and o["species_map50_95"] >= b["species_map50_95"] - 1e-12, o["species_map50_95"],
                    b["species_map50_95"]))
    check("O5: the oracle reading names E3-B's own boxes (its stage-1 rows E3's) and scores at least E3-B's two-stage "
          "species mean (%s)" % [(round(x[1], 4), round(x[2], 4)) for x in orc], all(x[0] for x in orc), orc)
    cap = json.loads(SYS.score_record_path("E3RS").read_text())
    names = [f["file"] for f in cap["dev"]]
    check("capacity/v3fam_score_e3rs.json: dev files only (names relative to v3fam/v1, their sha256s), the dev "
          "reproductions, the reported passes counted only (%s)" % cap["reported"],
          all(f["exam"] == "dev" and W.sha(F.root() / f["file"]) == f["sha256"] for f in cap["dev"])
          and not any("imageweeds" in n or "ooddev" in n for n in names)
          and not any("imageweeds" in k or "ooddev" in k for k in cap["equivalence"])
          and cap["reported"] == {"status": "complete", "passes": 12, "failed": 0} and cap["e_trees"]["files"] > 10,
          cap["reported"])
    rep = json.loads(SYS.reported_path("E3RS").read_text())
    written = {(p["dir"], p["exam"]) for p in rep["passes"] if p["status"] == "written"}
    check("reported: E3-B two-stage on ImageWeeds (3) and ooddev_v1 for E1-A, E1-B (stage1 mode) and b_v2_m640 "
          "(identity), 9 passes in the zoo's root; nothing written in the root",
          written == {("E3B", "imageweeds"), ("E1A", "ooddev_v1"), ("E1B", "ooddev_v1"), ("REF", "ooddev_v1")}
          and len(rep["passes"]) == 12 and root_tree() == tree0, rep["passes"])
    od = json.loads((F.arm_dir("E1B", 0) / "ooddev_v1.json").read_text())
    check("  E1-B's ooddev_v1 record keeps its box set (stage1 npz) and its agnostic AP; the E3-B ImageWeeds record its "
          "stage-1 rows without features", od["settings"]["mode"] == "stage1" and od["stage1_images"]["sha256"]
          and "feat" not in SC.load_npz(F.arm_dir("E3B", 0) / "imageweeds.stage1.npz"))
    files0 = sorted(str(p) for p in F.root().rglob("*.json"))
    rec2 = SYS.score_e3rs(embedder=EMB, **KW)
    check("a rerun keeps every file (status kept) and writes no new score", all(
        f["status"] == "kept" for f in rec2["dev"]) and sorted(str(p) for p in F.root().rglob("*.json")) == files0
          and rec2["reported"]["failed"] == 0)
    shutil.rmtree(str(F.arm_dir("E3A", 0)))
    e3f = TS.arm_dir("A", 0) / "dev.json"
    first = sorted(json.loads(e3f.read_text())["per_class"])[0]
    with tamper(e3f, lambda d: d["per_class"].update({first: d["per_class"][first] + 0.01})):
        e = refused(SYS.score_e3rs, embedder=EMB, **KW)
    check("a family two-stage pass that does not reproduce E3's file (0.01 off) refuses, writing no score",
          e is not None and "E3-A" in str(e) and not (F.arm_dir("E3A", 0) / "dev.json").exists(), e)
    SYS.score_e3rs(embedder=EMB, **KW)
    check("  with E3's file restored the missing score is taken again", (F.arm_dir("E3A", 0) / "dev.json").exists())


def w_d3():
    print("D3: WBF of E3-B's three stage-1 seeds through inject_boxes")
    if not ensemble_boxes_ready():
        check("ensemble_boxes is importable", False)
        return
    t0 = time.time()
    rec = WB.score_d3(embedder=EMB, **KW)
    d = F.root() / "arms" / "D3" / "ens"
    doc = json.loads((d / "dev.json").read_text())
    ins = doc["fusion"]["inputs"]
    check("D3's dev score: one system (no seed), mode inject_boxes on E1-B s0's scaffold, E3's classifier, the "
          "library and the three E3-B records it fused by sha256 (%.0fs)" % (time.time() - t0),
          doc["seed"] is None and doc["one_system"] is True and doc["settings"]["mode"] == "inject_boxes"
          and doc["scaffold"]["exp"] == "e1_b_m640" and doc["scaffold"]["seed"] == 0
          and doc["fusion"]["library"]["sha256"] == WB.ENSEMBLE_BOXES_WBF_SHA256
          and [i["seed"] for i in ins] == [0, 1, 2]
          and all(i["sha256"] == W.sha(F.arm_dir("E3B", i["seed"]) / "dev.json") for i in ins)
          and doc["stage2"]["classifier"]["npz_sha256"] == TS.load_classifier()[5]["npz_sha256"]
          and rec["dev"][0]["file"] == "arms/D3/ens/dev.json", doc.get("fusion"))
    s1 = SC.load_npz(d / doc["stage1_images"]["name"])
    stage1s = [SYS.family_side("E3B", s, "dev", "stage1_images")[1] for s in F.SEEDS]
    fn, _impl = WB.wbf_impl()
    fused, _cnt = WB.fuse_exam(stage1s, SYS.image_sizes(SYS.exam_rows("dev")), fn)
    q_all = np.concatenate([fused[str(k)]["conf"] for k in s1["keys"]]).astype(np.float64)
    check("  the boxes it named are the fused boxes (scores equal, %d rows)" % len(s1["q"]),
          len(s1["q"]) == len(q_all) and np.allclose(np.sort(s1["q"]), np.sort(q_all), atol=1e-6))
    x = SYS.family_runs("D3", "dev", ens="ens")
    y = SYS.family_runs("E3B", "dev")
    cmp_ = F.comparison("P4", [x[0][0]["species_map50_95"]], [v[0]["species_map50_95"] for v in y],
                        [x[0][1]], [v[1] for v in y], one_system=True, resamples=50)
    check("P4 is computable on these arrays (one system: D %.4f, its 2-sd bar 2 x sd(E3-B))"
          % cmp_["decision"]["diff"], cmp_["decision"]["one_system"] and cmp_["decision"]["pooled_sd"] is None)
    doc0, st0 = SYS.family_side("E3B", 0, "dev", "stage1_images")
    cuts = np.searchsorted(st0["img"], np.arange(len(st0["keys"]) + 1))
    inj = {str(k): {"xyxy": st0["xyxy"][cuts[i]:cuts[i + 1]], "conf": st0["q"][cuts[i]:cuts[i + 1]]}
           for i, k in enumerate(st0["keys"])}
    ccfg, _r, _s = TS._clf_cfg()
    with TS._workers(None) as wk:
        ps = F._family_pass(E3T.src_weights("B", 0), "dev", dict(ccfg, mode="inject_boxes", nms="locked",
                                                                 embedder=EMB, workers=wk, inject=inj), "D3", **KW)
    diff = abs(ps["result"]["species_map50_95"] - doc0["species_map50_95"])
    check("the injection path: E3-B s0's own stored stage-1 rows injected reproduce its family dev score (diff %.2g; "
          "boxes clipped to the image)" % diff, diff <= F.DEVIATION_TOL, diff)
    rep = json.loads(SYS.reported_path("D3").read_text())
    od = json.loads((d / "ooddev_v1.json").read_text())
    check("reported: ImageWeeds (inject_boxes) and ooddev_v1 (inject_rows of the one-class detectors' class, in the "
          "zoo's root) written; dev only in the platform's record",
          {p["exam"]: p["status"] for p in rep["passes"]} == {"imageweeds": "written", "ooddev_v1": "written"}
          and od["settings"]["mode"] == "inject_rows" and od["fusion"]["input_dir"] == "E1B"
          and json.loads(SYS.score_record_path("D3").read_text())["reported"]["failed"] == 0, rep)
    shutil.copytree(str(F.arm_dir("E3B", 2)), str(F.arm_dir("E3B", 3)))
    try:
        e = refused(WB.load_inputs, "dev")
    finally:
        shutil.rmtree(str(F.arm_dir("E3B", 3)))
    check("a planted fourth seed (arms/E3B/s3) is refused", e is not None and "s3" in str(e), e)
    rec2 = WB.score_d3(embedder=EMB, **KW)
    check("a rerun keeps D3's files", rec2["dev"][0]["status"] == "kept")


def make_d8_runs(arm, wseed0):
    """The runs a D8 build's jobs would leave: three done base runs (real checkpoints) and final runs scored on dev
    and ImageWeeds by the locked scorer (CPU, test mode)."""
    exp = SYS.D8_ARMS[arm]["exp"]
    for s in F.SEEDS:
        rd = N.make_final(exp, s, wseed0 + s, protocol=False)
        w = C.INC_DIR / exp / "runs" / ("base__s%d" % s) / "weights" / "final.pt"
        (C.INC_DIR / exp / "runs" / ("base__s%d" % s) / "run.json").write_text(json.dumps(
            {"status": "done", "testing": True, "weights_sha256": W.sha(w)}))
        for exam in SYS.D8_FINALS:
            S.score(rd / "weights" / "final.pt", exam, rd / "scores" / ("%s.json" % exam), lock_check=True,
                    imgsz=640, batch=32, device="cpu")


def w_d8():
    print("D8: YOLO11s and YOLO11m 12-class on train_core")
    for model in ("yolo11s.pt", "yolo11m.pt"):
        (C.REPO / model).write_bytes(os.urandom(2048))
    sha = C2.verify_manifest_against_lock_v2("train_core")
    defs = {}
    for arm in SYS.D8_ORDER:
        _summ, defn, res = SYS.build_d8(arm, testing=True, init=False)
        defs[arm] = defn
        (C.INC_DIR / SYS.D8_ARMS[arm]["exp"] / "exp.json").write_text(json.dumps(defn))     # what driver init pins
        check("%s: exp.json on LOCK v2's train_core (by sha256), arm %s, role baseline, finals dev and ImageWeeds, the "
              "cold recipe, seeds 0-2, the v3fam record; no driver init in the test" % (arm, SYS.D8_ARMS[arm]["arm"]),
              res is None and defn["base"]["manifest_sha256"] == sha and defn["arm"]["id"] == SYS.D8_ARMS[arm]["arm"]
              and defn["role"] == "baseline" and defn["final_exams"] == ["dev", "imageweeds"]
              and defn["base"]["recipe"] == RC.cold(SYS.D8_ARMS[arm]["arm"]) and defn["seeds"] == [0, 1, 2]
              and defn["v3fam"]["arm"] == arm and "test" not in defn["final_exams"], defn.get("v3fam"))
    e = refused(SYS.build_d8, "D8M", testing=True, init=False)
    check("  a second build of the same experiment (exp.json pinned by the driver's init) is refused", e is not None, e)
    make_d8_runs("D8S", 300)
    make_d8_runs("D8M", 310)
    t0 = time.time()
    rec = SYS.score_d8(**KW)
    files = {(f["dir"], f["seed"]) for f in rec["dev"]}
    vs = [json.loads((F.arm_dir(a, s) / "dev.json").read_text())["vs_recorded"] for a in SYS.D8_ORDER
          for s in F.SEEDS]
    check("score_d8: six dev identity scores, each equal to its run's recorded protocol dev score (%.0fs)"
          % (time.time() - t0), files == {(a, s) for a in SYS.D8_ORDER for s in F.SEEDS}
          and all(v["status"] == "passed" and v["diff"]["max_abs_diff"] == 0 for v in vs), vs[:1])
    rep = json.loads(SYS.reported_path("D8").read_text())
    check("reported: ImageWeeds (6) and ooddev_v1 in the zoo's root (6), written",
          sorted((p["dir"], p["seed"], p["exam"]) for p in rep["passes"] if p["status"] == "written")
          == sorted((a, s, e) for a in SYS.D8_ORDER for s in F.SEEDS for e in ("imageweeds", "ooddev_v1")),
          [p for p in rep["passes"] if p["status"] != "written"])
    src = SYS.reference_sources("D8M")
    check("D8-m's reference read scores v3_d8_m640_tc's three base runs (by their weights' sha256); D8-s, D3 and "
          "E3RS have none", src["exp"] == "v3_d8_m640_tc" and [x["seed"] for x in src["seeds"]] == [0, 1, 2]
          and all(x["weights_sha256"] == W.sha(C.INC_DIR / "v3_d8_m640_tc" / "runs" / ("base__s%d" % x["seed"])
                                                / "weights" / "final.pt") for x in src["seeds"])
          and all(refused(SYS.reference_sources, a) is not None for a in ("D8S", "D3", "E3RS")), src)
    ej = C.INC_DIR / "v3_d8_s640_tc" / "exp.json"
    with tamper(ej, lambda d: d["base"].update(manifest_sha256="0" * 64)):
        e1 = refused(SYS.check_d8_definition, "D8S", testing_ok=True)
    with tamper(ej, lambda d: d["base"]["recipe"].update(epochs=99)):
        e2 = refused(SYS.check_d8_definition, "D8S", testing_ok=True)
    e3 = refused(SYS.check_d8_definition, "D8S")
    check("refused: a definition on another manifest, another recipe, a testing build outside the tests",
          e1 is not None and e2 is not None and e3 is not None, (e1, e2, e3))


def w_latency_check():
    print("D7: the software check's path (Ultralytics' validator on a label-free view) against the family stage1 pass")
    TL = latency_module()
    w = E3T.src_weights("B", 0)
    rows = [{"key": r["key"], "image": r["image"], "sha256": r["sha256"]} for r in SYS.exam_rows("dev")]
    labels = {os.path.realpath(r["label"]) for r in SYS.exam_rows("dev")}
    keys = [r["key"] for r in rows][:5]
    opened = []
    WORLD["audit_on"] = True

    def hook(event, args):
        if WORLD.get("audit_on") and event == "open" and args and isinstance(args[0], str):
            opened.append(os.path.realpath(args[0]))
    if not WORLD.get("hooked"):
        sys.addaudithook(hook)
        WORLD["hooked"] = True
    with E3T.no_oracle():
        fam = F._family_pass(w, "dev", {"mode": "stage1", "nms": "locked"}, "D4F", **KW)
        WORLD["audit_on"] = True
        got = TL.stage1_validator_boxes(w, rows, "cpu", set(keys))
        WORLD["audit_on"] = False
    cmp_ = TL.compare_stage1(got, fam["stage1"], keys)
    check("the stage-1 box sets of Ultralytics' validator on a label-free view (batch 32, rect, conf 0.001, one box per "
          "location, the inverse letterbox) equal the family stage1 pass's on %d dev images (max %.2g px)"
          % (len(keys), cmp_["max_diff_px"]), cmp_["matched"] and cmp_["max_diff_px"] <= 1e-3, cmp_)
    check("  no dev label was opened", not (set(opened) & labels), sorted(set(opened) & labels)[:3])
    shifted = {k: (v[0] + 2.0, v[1]) for k, v in got.items()}
    check("  boxes 2 px off are not matched (the run would be labelled 'different software')",
          not TL.compare_stage1(shifted, fam["stage1"], keys)["matched"])


def w_cli():
    print("the CLI")
    env = dict(os.environ)
    pkg = str(pathlib.Path(F.__file__).resolve().parents[3])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def cli(*a):
        p = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc2.v3fam_systems"] + list(a),
                           env=env, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    rc1, _o, e1 = cli("score", "--arm", "D9")
    rc2, _o, e2 = cli("sources", "--arm", "D8S")
    rc3, out3, _e = cli("sources", "--arm", "D8M")
    rc4, _o, _e4 = cli("nonsense")
    check("usage errors and refusals exit 2 with [inc2.v3fam_systems] ERROR; sources --arm D8M prints the read's "
          "inputs", rc1 == 2 and "ERROR" in e1 and rc2 == 2 and "no reference read" in e2 and rc3 == 0
          and json.loads(out3)["exp"] == "v3_d8_m640_tc" and rc4 == 2, (rc1, e1[-200:], rc2, e2[-200:], rc3))


WORLD_TESTS = ("w_setup", "w_e3rs", "w_d3", "w_d8", "w_latency_check", "w_cli")


def main():
    t0 = time.time()
    only = os.environ.get("V3FAM_ONLY")
    upto = os.environ.get("V3FAM_UPTO")
    try:
        WORLD["rows"] = W.build_world()
        seq = UNITS + (() if only == "units" else WORLD_TESTS)
        for name in seq:
            globals()[name]()
            if name == upto:
                break
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
