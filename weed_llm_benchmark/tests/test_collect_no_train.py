#!/usr/bin/env python3
"""The collector never trains and never imports a trainer (docs/CONTINUOUS_LOOP.md
§8 "No mega_trainer training"; group D acceptance "Imports").

The rule, as §8 [review] restates it: collect/ may reach mega_trainer only
through inc.common.dhash (the project's one dHash, which imports
mega_trainer._dhash when it is called). So:
  * static: no module of collect/ imports mega_trainer, ultralytics, torch,
    inc.train, inc2.train, inc.lora, lora_yolo, yolo_trainer,
    hot_reload_trainer, train_yolo_on_verified or roboflow_sync (every
    import statement, relative ones resolved); no code names mega_trainer as
    a Python name; the only string that names it is the file the never-train
    slugs are parsed from (prefilter.TRAINER_SOURCE, parsed with ast, never
    imported);
  * runtime (a fresh interpreter): after importing every collect module,
    loading the config and running a guard check on an image (the dHash and
    its eight variants, the real inc2.guard when it is importable), neither
    ultralytics nor torch is loaded, and no trainer module beyond those the
    package root weed_optimizer_framework/__init__.py loads by itself (its
    orchestrator imports yolo_trainer, which loads no Ultralytics at import);
  * the registry annotation intake_v1 is not one the pinned Step 1 or the
    old merge reads (verify.VALID_ANNOTATIONS).

Run:  python3 tests/test_collect_no_train.py
"""
import ast
import io
import json
import pathlib
import subprocess
import sys
import tokenize

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
COL = ROOT / "weed_optimizer_framework" / "tools" / "collect"
PKG = "weed_optimizer_framework.tools.collect"
FORBIDDEN = ("mega_trainer", "ultralytics", "torch", "weed_optimizer_framework.tools.inc.train",
             "weed_optimizer_framework.tools.inc2.train", "weed_optimizer_framework.tools.inc.lora", "lora_yolo",
             "yolo_trainer", "hot_reload_trainer", "train_yolo_on_verified", "roboflow_sync", "train_rfdetr")
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def module_name(p):
    rel = p.relative_to(COL).with_suffix("")
    parts = [x for x in rel.parts if x != "__init__"]
    return ".".join([PKG] + parts)


def imports_of(p):
    mod = module_name(p)
    is_pkg = p.name == "__init__.py"
    out = []
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = mod.split(".")
                base = base if is_pkg else base[:-1]
                base = base[:len(base) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            out.append(target)
            out.extend("%s.%s" % (target, a.name) for a in node.names)
    return out


def test_static():
    print("static: the import graph and the names")
    files = sorted(p for p in COL.rglob("*.py") if "__pycache__" not in p.parts)
    bad = []
    for p in files:
        for imp in imports_of(p):
            for f in FORBIDDEN:
                if imp == f or imp.startswith(f + ".") or imp.endswith("." + f) or ("." + f + ".") in imp \
                        or imp.split(".")[-1] == f.split(".")[-1] and f.split(".")[-1] in ("mega_trainer", "roboflow_sync"):
                    bad.append((p.name, imp))
    check("no collect module imports a trainer, ultralytics, torch or a labelling sync (%d files)" % len(files),
          not bad and len(files) >= 15, bad)
    names, strings = [], []
    for p in files:
        toks = list(tokenize.generate_tokens(io.StringIO(p.read_text()).readline))
        for i, t in enumerate(toks):
            if t.type == tokenize.NAME and t.string == "mega_trainer":
                names.append((p.name, t.start[0]))
            if t.type == tokenize.STRING and "mega_trainer" in t.string and not t.string.startswith(('"""', "'''")):
                strings.append((p.name, t.start[0], t.string))
    check("no code names mega_trainer as a Python name", not names, names)
    check("the only string naming it is the parsed never-train source file",
          strings == [("prefilter.py", strings[0][1], '"mega_trainer.py"')] if strings else False, strings)
    src = (COL / "prefilter.py").read_text()
    check("... which is parsed with ast.literal_eval, never imported", "ast.parse" in src and "literal_eval" in src
          and "import_module" not in src)


RUNTIME = r'''
import json, os, sys, tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp(prefix="collect_notrain_"))
os.environ["INC_DIR"] = str(tmp / "inc"); os.environ["REPO"] = str(tmp / "repo")
sys.path.insert(0, sys.argv[1])
import importlib, pkgutil
TRAINERS = (".inc.train", ".inc2.train", ".inc.lora", "lora_yolo", "yolo_trainer", "hot_reload_trainer",
            "train_yolo_on_verified", "roboflow_sync", "train_rfdetr")
import weed_optimizer_framework
baseline = sorted(k for k in sys.modules if k.endswith(TRAINERS))
import weed_optimizer_framework.tools.collect as COLP
mods = [COLP.__name__]
for m in pkgutil.walk_packages(COLP.__path__, COLP.__name__ + "."):
    importlib.import_module(m.name); mods.append(m.name)
after_import = sorted(k for k in sys.modules if k.startswith(("ultralytics", "torch")))
from weed_optimizer_framework.tools.collect import config as CF, prefilter as PF
cfg = CF.load("weed")
PF.never_train_slugs()
from PIL import Image
p = tmp / "a.png"
Image.new("RGB", (40, 30), (10, 200, 30)).save(p)
from weed_optimizer_framework.tools.inc import common as C
h = C.dhash(p)
from weed_optimizer_framework.tools.funnel import leak as L
with Image.open(p) as im:
    v = L.dhash_variants(im)
real_guard = False
try:
    from weed_optimizer_framework.tools.inc2 import guard as G2
    G2.image_hashes(p)
    real_guard = True
except ImportError:
    pass
loaded = sorted(k for k in sys.modules if k.startswith(("ultralytics", "torch"))
                or (k.endswith(TRAINERS) and k not in baseline))
print(json.dumps({"modules": mods, "after_import": after_import, "loaded": loaded, "baseline": baseline,
                  "dhash": h is not None,
                  "variants": len(v), "real_guard": real_guard,
                  "mega_trainer_loaded": "weed_optimizer_framework.tools.mega_trainer" in sys.modules}))
'''


def test_runtime():
    print("runtime: a fresh interpreter")
    out = subprocess.run([sys.executable, "-c", RUNTIME, str(ROOT)], capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        check("the runtime probe ran", False, out.stderr[-2000:])
        return
    res = json.loads(out.stdout.strip().splitlines()[-1])
    check("every collect module imports (%d)" % len(res["modules"]), len(res["modules"]) >= 15, res["modules"])
    check("importing the collector loads neither ultralytics nor torch", res["after_import"] == [], res["after_import"])
    check("after a config load and a guard check, neither ultralytics nor torch is loaded, and the collector adds "
          "no trainer module to those the package root itself loads (%s)" % ", ".join(res["baseline"]) or "none",
          res["loaded"] == [] and res["dhash"] and res["variants"] == 8, res)
    check("mega_trainer is reached only through inc.common.dhash (loaded by the dHash call)",
          res["mega_trainer_loaded"], res)
    if not res["real_guard"]:
        print("  note: inc2.guard was not importable here; the guard check ran on inc.common.dhash and funnel.leak")


def test_annotation():
    print("the registry annotation")
    sys.path.insert(0, str(ROOT))
    from weed_optimizer_framework.tools.collect import prefilter as PF
    from weed_optimizer_framework.tools.inc import verify as V
    check("intake_v1 is not an annotation the pinned Step 1 or the old merge reads",
          PF.INTAKE_ANNOTATION == "intake_v1" and PF.INTAKE_ANNOTATION not in V.VALID_ANNOTATIONS)


def main():
    test_static()
    test_runtime()
    test_annotation()
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
