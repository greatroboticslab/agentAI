#!/usr/bin/env python3
"""inc2/common.py: where every v2 name resolves (docs/CONTINUOUS_LOOP.md §4.3).

Pinned:
  * the v2 constants: SPLITS_VERSION v2, SPLITS_DIR INC_DIR/splits/v2, the v2
    LOCK, never-train and base-copy index paths, EVAL_SPLITS (dev, test,
    imageweeds), TRAIN_SPLITS (train_core, tsw22, tsw23), FINAL_EXAMS (dev,
    imageweeds, test); EXAMS_DIR stays the v1 exam directory;
  * the re-export: every public name of inc.common is in inc2.common, the
    same object unless it is one of the documented overrides; importing
    inc2.common leaves inc.common's own globals at v1;
  * resolution: train_manifest_path -> v2 files, eval_manifest_path -> the v1
    files (the v1 scorer's), manifest_path dispatches and refuses ood22/ood23;
    the argument-less v1 functions (read_lock, verify_manifest_against_lock)
    refuse as ambiguous; read_lock_v2 refuses a missing or v1 LOCK;
    verify_manifest_against_lock_v2 catches a changed manifest;
  * NeverTrainGuard.load() without a path refuses in inc2; with a path it
    loads, and nevertrain_v2 refuses an index lock has not marked complete;
    its check() covers the 8 flips and rotations (§8): a flipped copy of an
    indexed image, which the v1 check (stored dHash only) passes, is a hit;
    missing variants, or variants of other pixels, are unhashable;
  * no module of the inc2 package calls NeverTrainGuard.load() without a path
    (a grep over every inc2/*.py, other groups' modules included);
  * inc2/common.py imports only the standard library and inc.common at module
    level (AST);
  * the atomic writers: the file appears whole or not at all; a failed write
    leaves the old file and no temporary file.

Run:  python3 tests/test_inc2_common.py
"""
import ast
import json
import os
import pathlib
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc2_common_"))
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_DIR"] = str(TMP / "inc")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weed_optimizer_framework.tools.inc import common as C1  # noqa: E402
from weed_optimizer_framework.tools.inc2 import common as C2  # noqa: E402

INC2 = ROOT / "weed_optimizer_framework" / "tools" / "inc2"
FAILURES = []

# The names inc2.common deliberately redefines (§4.3); every other public name
# of inc.common must be the very same object.
OVERRIDES = {"SPLITS_VERSION", "SPLITS_DIR", "LOCK_PATH", "NEVER_TRAIN_INDEX", "EVAL_SPLITS", "TRAIN_SPLITS",
             "manifest_path", "read_lock", "verify_manifest_against_lock", "NeverTrainGuard"}


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        return text in str(e)
    return False


def test_constants():
    print("constants")
    check("C1 and C2 read the same INC_DIR (the fake one)", C1.INC_DIR == C2.INC_DIR == TMP / "inc")
    check("SPLITS_VERSION v2, SPLITS_DIR INC_DIR/splits/v2",
          C2.SPLITS_VERSION == "v2" and C2.SPLITS_DIR == TMP / "inc" / "splits" / "v2")
    check("LOCK, never-train and base-copy indexes live in splits/v2",
          C2.LOCK_PATH == C2.SPLITS_DIR / "LOCK.json"
          and C2.NEVER_TRAIN_INDEX == C2.SPLITS_DIR / "nevertrain_dhash.json"
          and C2.BASE_COPIES_INDEX == C2.SPLITS_DIR / "base_copies_dhash.json")
    check("EVAL_SPLITS (dev, test, imageweeds), TRAIN_SPLITS (train_core, tsw22, tsw23)",
          C2.EVAL_SPLITS == ("dev", "test", "imageweeds") and C2.TRAIN_SPLITS == ("train_core", "tsw22", "tsw23"))
    check("FINAL_EXAMS (dev, imageweeds, test), a subset of the v1 evaluation splits (the pinned driver's)",
          C2.FINAL_EXAMS == ("dev", "imageweeds", "test") and set(C2.FINAL_EXAMS) <= set(C1.EVAL_SPLITS))
    check("EXAMS_DIR is the v1 exam directory (the v1 scorer reads it)", C2.EXAMS_DIR == C1.EXAMS_DIR
          and C2.EXAMS_DIR == TMP / "inc" / "exams" / "v1")
    check("inc.common is untouched: still v1 after importing inc2.common",
          C1.SPLITS_VERSION == "v1" and C1.SPLITS_DIR == TMP / "inc" / "splits" / "v1"
          and C1.EVAL_SPLITS == ("dev", "test", "ood22", "ood23", "imageweeds") and C1.TRAIN_SPLITS == ("train_core",))
    check("the byte copies are dev, test, imageweeds and train_core",
          set(C2.BYTE_COPIES) == {"dev", "test", "imageweeds", "train_core"})


def test_reexport():
    print("re-export")
    public = [n for n in dir(C1) if not n.startswith("_")]
    missing = [n for n in public if not hasattr(C2, n)]
    check("every public name of inc.common is in inc2.common", not missing, missing)
    same = [n for n in public if n not in OVERRIDES and getattr(C2, n) is not getattr(C1, n)]
    check("every name but the documented overrides is the same object", not same, same)
    differ = [n for n in OVERRIDES if getattr(C2, n) is getattr(C1, n) or getattr(C2, n) == getattr(C1, n)]
    check("every documented override differs from v1", not differ, differ)
    check("the class space, manifest keys, stable_int and dhash are v1's",
          C2.CLASS_NAMES == C1.CLASS_NAMES and C2.NC == 13 and C2.MANIFEST_KEYS == C1.MANIFEST_KEYS
          and C2.stable_int is C1.stable_int and C2.dhash is C1.dhash and C2.write_manifest is C1.write_manifest)


def test_resolution():
    print("resolution")
    v2 = TMP / "inc" / "splits" / "v2"
    v1 = TMP / "inc" / "splits" / "v1"
    check("train_manifest_path: tsw22, tsw23, train_core, base_v2 -> splits/v2",
          all(C2.train_manifest_path(s) == v2 / ("%s.jsonl" % s)
              for s in ("tsw22", "tsw23", "train_core", "base_v2")))
    check("train_manifest_path refuses an evaluation split and ood22",
          raises(lambda: C2.train_manifest_path("dev"), C2.Inc2Error, "not a v2 training manifest")
          and raises(lambda: C2.train_manifest_path("ood22"), C2.Inc2Error))
    check("eval_manifest_path: dev, test, imageweeds -> the v1 files",
          all(C2.eval_manifest_path(s) == v1 / ("%s.jsonl" % s) == C1.manifest_path(s)
              for s in ("dev", "test", "imageweeds")))
    check("eval_manifest_path refuses ood22, ood23 and training splits",
          all(raises(lambda s=s: C2.eval_manifest_path(s), C2.Inc2Error)
              for s in ("ood22", "ood23", "tsw22", "train_core")))
    check("manifest_path dispatches: dev -> v1, tsw22 / train_core / base_v2 -> v2",
          C2.manifest_path("dev") == v1 / "dev.jsonl" and C2.manifest_path("tsw22") == v2 / "tsw22.jsonl"
          and C2.manifest_path("train_core") == v2 / "train_core.jsonl"
          and C2.manifest_path("base_v2") == v2 / "base_v2.jsonl")
    check("manifest_path refuses ood22 and ood23 (not v2 splits)",
          raises(lambda: C2.manifest_path("ood22"), C2.Inc2Error, "not a v2 split")
          and raises(lambda: C2.manifest_path("ood23"), C2.Inc2Error))
    check("v2_manifest_path covers the byte copies too", C2.v2_manifest_path("dev") == v2 / "dev.jsonl"
          and raises(lambda: C2.v2_manifest_path("ood22"), C2.Inc2Error))
    check("read_lock and verify_manifest_against_lock refuse as ambiguous",
          raises(C2.read_lock, C2.AmbiguousV1Call, "read_lock_v2")
          and raises(lambda: C2.verify_manifest_against_lock("dev"), C2.AmbiguousV1Call))
    check("read_lock_v2 refuses a missing LOCK", raises(C2.read_lock_v2, C2.Inc2Error, "not locked"))
    v2.mkdir(parents=True, exist_ok=True)
    C2.write_json_atomic(C2.LOCK_PATH, {"splits_version": "v1", "manifests": {}})
    check("read_lock_v2 refuses a LOCK that is not v2", raises(C2.read_lock_v2, C2.Inc2Error, "not a v2 LOCK"))
    rows = [{"image": "/x.png", "label": "/x.txt", "sha256": "a" * 64, "label_sha256": "b" * 64, "source": "s",
             "session": "", "key": "tsw22__x"}]
    sha = C1.write_manifest(C2.v2_manifest_path("tsw22"), rows)
    C2.write_json_atomic(C2.LOCK_PATH, {"splits_version": "v2", "manifests": {"tsw22": sha}})
    check("verify_manifest_against_lock_v2 passes an unchanged manifest",
          C2.verify_manifest_against_lock_v2("tsw22") == sha)
    rows[0]["session"] = "changed"
    C1.write_manifest(C2.v2_manifest_path("tsw22"), rows)
    check("... and refuses a changed one", raises(lambda: C2.verify_manifest_against_lock_v2("tsw22"),
                                                  C2.Inc2Error, "changed since it was locked"))
    check("... and a split the LOCK does not hold", raises(lambda: C2.verify_manifest_against_lock_v2("tsw23"),
                                                           C2.Inc2Error, "not in LOCK v2"))
    os.unlink(C2.LOCK_PATH)


def test_guard():
    print("never-train guard")
    check("NeverTrainGuard.load() without a path refuses in inc2",
          raises(lambda: C2.NeverTrainGuard.load(), C2.Inc2Error, "explicit index path"))
    idx = TMP / "idx.json"
    C2.write_json_atomic(idx, {"entries": [[12345, "dev", "dev__a"]], "bits": 6, "complete": True,
                               "min_expected": 1})
    g = C2.NeverTrainGuard.load(idx)
    check("with a path it loads (a subclass of the v1 guard)", g.n == 1 and isinstance(g, C1.NeverTrainGuard))
    check("nevertrain_v2 loads the v2 index path",
          raises(C2.nevertrain_v2, FileNotFoundError) or raises(C2.nevertrain_v2, OSError))
    C2.write_json_atomic(C2.NEVER_TRAIN_INDEX, {"entries": [[1, "dev", "dev__a"]], "bits": 6, "complete": False,
                                                "min_expected": 2})
    check("nevertrain_v2 refuses an index lock has not marked complete",
          raises(C2.nevertrain_v2, RuntimeError, "expected >= 2"))
    C2.write_json_atomic(C2.NEVER_TRAIN_INDEX, {"entries": [[1, "dev", "dev__a"]], "bits": 6, "complete": True,
                                                "min_expected": 1})
    check("... and loads a complete one", C2.nevertrain_v2().n == 1)
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print("SKIP: the v2 NeverTrainGuard variant checks (numpy / PIL not importable: %s)" % e)
        return
    rng = np.random.default_rng(7)
    a = (rng.integers(0, 4, size=(8, 9)) * 60 + 20).astype(np.uint8)
    img = Image.fromarray(np.repeat(np.repeat(a, 12, axis=0), 12, axis=1)).convert("RGB")
    orig, flip, other = TMP / "orig.png", TMP / "flip.png", TMP / "other.png"
    img.save(orig)
    img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(flip)
    Image.fromarray(np.repeat(np.repeat((rng.integers(0, 4, size=(8, 9)) * 60 + 20).astype(np.uint8), 12, axis=0),
                              12, axis=1)).convert("RGB").save(other)
    far = bin(C1.dhash(orig) ^ C1.dhash(flip)).count("1")
    g = C2.NeverTrainGuard([[C1.dhash(orig), "test", "test__orig"]])
    v1_hits, _u = C1.NeverTrainGuard.check(g, [flip])
    hits, un = g.check([flip, other])
    check("v2 check: a horizontal flip (%d bits from the original as stored) is a hit; v1's check passed it" % far,
          far > 6 and v1_hits == [] and [(h[0], h[1], h[2]) for h in hits] == [(str(flip), "test", "test__orig")]
          and not un, (v1_hits, hits, un))
    check("... an unrelated image passes, assert_trainable raises on the flip",
          g.check([other]) == ([], []) and raises(lambda: g.assert_trainable([flip]), RuntimeError, "flips"))
    check("... no variants, or variants of other pixels, are unhashable (fail closed)",
          g.check([other], variants_fn=lambda p: None) == ([], [str(other)])
          and g.check([other], variants_fn=lambda p: {k: 0 for k in ("id", "hflip", "vflip", "rot90", "rot180",
                                                                      "rot270", "transpose", "transverse")})
          == ([], [str(other)]))


def pathless_loads(source):
    """Lines of calls NeverTrainGuard.load() / X.NeverTrainGuard.load() with no
    argument (code only: strings and comments do not count)."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"):
            continue
        owner = node.func.value
        name = owner.id if isinstance(owner, ast.Name) else owner.attr if isinstance(owner, ast.Attribute) else None
        if name == "NeverTrainGuard" and not node.args and not node.keywords:
            out.append(node.lineno)
    return out


def test_no_pathless_load():
    print("no argument-less NeverTrainGuard.load() in the inc2 package")
    files = sorted(p for p in INC2.iterdir() if p.suffix == ".py")
    hits, unparsed = {}, []
    for p in files:
        try:
            lines = pathless_loads(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            unparsed.append(p.name)
            print("SKIP: %s does not parse (%s); not checked" % (p.name, e))
            continue
        if lines:
            hits[p.name] = lines
    check("no module of %d calls NeverTrainGuard.load() without a path (AST)" % (len(files) - len(unparsed)),
          not hits, hits)
    check("the check catches C.NeverTrainGuard.load() and NeverTrainGuard.load( ), not a docstring or a path",
          pathless_loads("g = C.NeverTrainGuard.load( )\nh = NeverTrainGuard.load()\n") == [1, 2]
          and pathless_loads('"""NeverTrainGuard.load()"""\n# NeverTrainGuard.load()\n'
                             'g = C.NeverTrainGuard.load(p)\nk = C.NeverTrainGuard.load(path=p)\n') == [])


def test_imports():
    print("module-level imports of inc2/common.py")
    tree = ast.parse((INC2 / "common.py").read_text())
    bad = []
    stdlib = {"__future__", "datetime", "json", "os", "pathlib"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] not in stdlib]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").split(".")[0] not in stdlib:
                bad.append(node.module)
            if node.level > 0 and node.module not in ("inc", "inc.common"):
                bad.append("." * node.level + str(node.module))
    check("only the standard library and inc.common", not bad, bad)


def test_atomic():
    print("atomic writers")
    p = TMP / "a" / "x.json"
    sha = C2.write_json_atomic(p, {"b": 1, "a": [1, 2]})
    check("write_json_atomic returns the file's sha256 and writes sorted JSON",
          sha == C1.sha256_file(p) and json.loads(p.read_text()) == {"a": [1, 2], "b": 1}
          and p.read_text().index('"a"') < p.read_text().index('"b"'))
    check("a failed write (NaN) leaves the old file and no temporary file",
          raises(lambda: C2.write_json_atomic(p, {"x": float("nan")}), ValueError)
          and json.loads(p.read_text()) == {"a": [1, 2], "b": 1} and not list(p.parent.glob("*.tmp")))
    q = TMP / "a" / "y.jsonl"
    C2.write_jsonl_atomic(q, [{"k": 2}, {"k": 1}])
    check("write_jsonl_atomic keeps the given order", [json.loads(ln)["k"] for ln in q.read_text().splitlines()] == [2, 1])
    c = TMP / "a" / "copy.bin"
    src = TMP / "a" / "src.bin"
    src.write_bytes(b"\x00\x01abc")
    check("atomic_copy is a byte copy", C2.atomic_copy(src, c) == C1.sha256_file(src) and c.read_bytes() == b"\x00\x01abc")
    r = C2.file_record(c)
    check("file_record: path, sha256, bytes", r["bytes"] == 5 and r["sha256"] == C1.sha256_file(c))
    check("file_record refuses a missing file", raises(lambda: C2.file_record(TMP / "nope"), C2.Inc2Error))
    s = C2.write_csv_atomic(TMP / "a" / "t.csv", ("a", "b"), [[1, None], ["x,y", 2]])
    check("write_csv_atomic quotes and writes None as empty",
          (TMP / "a" / "t.csv").read_text() == 'a,b\n1,\n"x,y",2\n' and s == C1.sha256_file(TMP / "a" / "t.csv"))


def main():
    test_constants()
    test_reexport()
    test_resolution()
    test_guard()
    test_no_pathless_load()
    test_imports()
    test_atomic()
    print()
    if FAILURES:
        print("%d FAILED: %s" % (len(FAILURES), FAILURES))
        return 1
    print("all inc2.common checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
