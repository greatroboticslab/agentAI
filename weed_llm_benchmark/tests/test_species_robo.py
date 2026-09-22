#!/usr/bin/env python3
"""Roboflow upload, download and train paths name cwd12 boxes by species.

Uploads: a cwd12-id label file is uploaded with {id: species} written in the
target project's vocabulary, so the three projects created from our legacy
uploads keep exactly the class names they have and every other project gets
the species. Boxes on photographs that are not cwd12 photographs are written
as names download-merge reads back as the same species (round trip). Registry slugs resolve each class through class_species (cwd12
copies by id, others by real name). OWL proposal files (one species, id 0)
upload under that species, not under cwd12 id 0's name.

Downloads: images that are cwd12 photographs (holdout or train, within
NEAR_DUP_BITS by dHash) are dropped whole; the remaining boxes are read as
real names, and names that are not cwd12 species are dropped with a count.
train_from_roboflow maps species to trainer slots and the holdout GT by the
fixed id permutation.

Run:  python3 tests/test_species_robo.py
"""
import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402
from weed_optimizer_framework.tools import roboflow_sync as RS  # noqa: E402
from weed_optimizer_framework.tools import merge_roboflow_projects as MR  # noqa: E402
from weed_optimizer_framework.tools import train_from_roboflow as TR  # noqa: E402
from weed_optimizer_framework.tools import owl_precision as OP  # noqa: E402
from weed_optimizer_framework.tools import owl_preannotate as PA  # noqa: E402
from weed_optimizer_framework.tools import owl_upload_proposals as OU  # noqa: E402
from weed_optimizer_framework.tools import export_owl_exemplars as EX  # noqa: E402
from weed_optimizer_framework.tools import mega_trainer as M  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def _img(path, seed, size=(96, 72)):
    from PIL import Image
    import random
    rnd = random.Random(seed)
    im = Image.new("RGB", size)
    px = im.load()
    for x in range(size[0]):
        for y in range(size[1]):
            v = (x * (seed % 7 + 1) + y * (seed % 5 + 2) + rnd.randint(0, 40)) % 256
            px[x, y] = (v, (v * 3) % 256, (255 - v))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95)
    return path


class _FakeProject:
    def __init__(self):
        self.calls = []

    def single_upload(self, **kw):
        self.calls.append(kw)


class _FakeWS:
    def __init__(self):
        self.projects = {}

    def project(self, name):
        return self.projects.setdefault(name, _FakeProject())


def test_upload_labelmaps():
    print("upload labelmaps")
    for proj in S.LEGACY_ROBOFLOW_PROJECTS:
        check("legacy project %s keeps its class names" % proj,
              RS._cwd12_labelmap(proj) == dict(enumerate(S.CWD12_LEGACY_LABELS)))
    check("other project gets species",
          RS._cwd12_labelmap("weed-crop-agent-clean") == dict(enumerate(S.CWD12_SPECIES)))

    hold = {"class_names": ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]}
    lm = RS._registry_labelmap("cottonweed_holdout", hold, "weed-crop-agent-v4")
    check("holdout: 12 ids by id space, stored 4 names ignored",
          lm == dict(enumerate(S.CWD12_SPECIES)), lm)
    lm = RS._registry_labelmap("cottonweed_holdout", hold, "weed-crop-agent-dataset")
    check("holdout into legacy project: legacy label of each id",
          lm == dict(enumerate(S.CWD12_LEGACY_LABELS)), lm)
    sp8 = {"class_names": S.TRAINER_SLOT_LEGACY[:8]}
    lm = RS._registry_labelmap("cottonweed_sp8", sp8, "weed-crop-agent-clean")
    check("sp8: slot species", lm == dict(enumerate(S.TRAINER_SLOT_SPECIES[:8])), lm)

    zig = {"class_names": ["Carpet weed", "Crabgrass", "Eclipta", "Goosegrass",
                           "Morning glory", "Nutsedge", "crab_grass ", "0", ""]}
    lm = RS._registry_labelmap("rf_zig-zag-lnodr__weed-detection-vanpe", zig,
                               "weed-crop-agent-clean")
    check("real names -> species / cleaned real name",
          lm == {0: "Carpetweed", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
                 4: "MorningGlory", 5: "Nutsedge", 6: "crab grass"}, lm)
    lm = RS._registry_labelmap("rf_zig-zag-lnodr__weed-detection-vanpe", zig,
                               "weed-crop-agent-dataset")
    check("real names into legacy project: the class that reads back as the species",
          lm[0] == "Carpetweeds" and lm[4] == "Morningglory" and lm[3] == "Goosegrass"
          and lm[2] == "Eclipta", lm)
    check("real crabgrass/nutsedge do not join the legacy classes",
          lm[1] == "Crabgrass (non-cwd12)" and lm[5] == "Nutsedge (non-cwd12)", lm)
    # a harvested (non-cwd12) slug pushed into each legacy project comes back
    # from download-merge as the same species
    harvest = {"class_names": ["Waterhemp", "Carpet weed", "Common ragweed", "Crabgrass",
                               "Goosegrass", "Cutleaf groundcherry", "Morning glory",
                               "Palmer amaranth", "Sicklepod", "Spotted spurge",
                               "Purslane", "Eclipta", "Prickly sida", "Nutsedge"]}
    for proj in sorted(S.LEGACY_ROBOFLOW_PROJECTS) + ["weed-crop-agent-clean"]:
        lm = RS._registry_labelmap("some_harvest_slug", harvest, proj)
        names = [lm[i] for i in sorted(lm)]
        remap, _unk = MR._multiclass_remap(names)
        back = {i: S.CWD12_SPECIES[remap[j]] for j, i in enumerate(sorted(lm)) if j in remap}
        want = {i: S.species_of(n) for i, n in enumerate(harvest["class_names"])
                if S.species_of(n)}
        check("round trip %s: harvest species come back unchanged" % proj,
              back == want, (lm, back))
        check("round trip %s: one class per species" % proj,
              len({lm[i] for i in want}) == len(want), lm)
    check("push-slug labelmap is the same resolver",
          RS._labelmap_for(zig, "rf_x", "weed-crop-agent-clean")
          == RS._registry_labelmap("rf_x", zig, "weed-crop-agent-clean"))

    check("require-cwd12: crabgrass + nutsedge only is not cwd12",
          not RS._slug_has_cwd12(["Crab Grass", "Nutsedge", "weed"], "rf_a"))
    check("require-cwd12: waterhemp is cwd12", RS._slug_has_cwd12(["Waterhemp"], "rf_b"))
    check("require-cwd12: cwd12 copy by id space",
          RS._slug_has_cwd12(["whatever"], "cottonweed_holdout"))
    check("require-cwd12: cyperus is not cwd12", not RS._slug_has_cwd12(["Cyperus"], "rf_c"))


def test_bulk_upload():
    print("bulk-upload")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "img").mkdir()
        (td / "lbl").mkdir()
        for stem, lines in (("a", "3 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n0 0.1 0.1 0.1 0.1"),
                            ("b", "9 0.5 0.5 0.1 0.1"), ("c", None)):
            (td / "img" / (stem + ".jpg")).write_bytes(b"x")
            if lines is not None:
                (td / "lbl" / (stem + ".txt")).write_text(lines)
        check("primary species by cwd12 id",
              RS._primary_species(td / "lbl" / "a.txt") == "SpottedSpurge")
        orig = RS._workspace
        try:
            for proj, single, want0, tag_a in (
                    ("weed-crop-agent-dataset", None, "Carpetweeds", "Goosegrass"),
                    ("weed-crop-agent-clean", None, "Waterhemp", "SpottedSpurge"),
                    ("weed-crop-agent-dataset", "SpottedSpurge", "Goosegrass", "Goosegrass"),
                    ("weed-crop-agent-clean", "SpottedSpurge", "SpottedSpurge", "SpottedSpurge")):
                ws = _FakeWS()
                RS._workspace = lambda ws=ws: ws
                RS.cmd_bulk_upload(argparse.Namespace(
                    images=str(td / "img"), labels=str(td / "lbl"), split="train",
                    batch="red", workers=1, per_species=0, project=proj,
                    single_species=single))
                calls = {pathlib.Path(c["image_path"]).stem: c for c in ws.projects[proj].calls}
                lm = calls["a"].get("annotation_labelmap", {})
                check("%s single=%s: id 0 -> %s" % (proj, single, want0),
                      lm.get(0) == want0, lm)
                if single:
                    check("%s single: labelmap holds only id 0" % proj, list(lm) == [0], lm)
                check("%s single=%s: batch/tag %s" % (proj, single, tag_a),
                      calls["a"]["batch_name"] == "red-" + tag_a
                      and tag_a in calls["a"]["tag_names"], calls["a"])
                check("%s: unlabeled image has no labelmap" % proj,
                      "annotation_labelmap" not in calls["c"])
            # --photos other: boxes read back as real names by download-merge
            for single, want0, tag_a in ((None, "Waterhemp", "SpottedSpurge"),
                                         ("SpottedSpurge", "SpottedSpurge", "SpottedSpurge"),
                                         ("Carpetweed", "Carpetweeds", "Carpetweeds")):
                ws = _FakeWS()
                RS._workspace = lambda ws=ws: ws
                RS.cmd_bulk_upload(argparse.Namespace(
                    images=str(td / "img"), labels=str(td / "lbl"), split="train",
                    batch="red", workers=1, per_species=0,
                    project="weed-crop-agent-dataset", single_species=single,
                    photos="other"))
                calls = {pathlib.Path(c["image_path"]).stem: c
                         for c in ws.projects["weed-crop-agent-dataset"].calls}
                lm = calls["a"].get("annotation_labelmap", {})
                check("legacy project, other photos, single=%s: id 0 -> %s" % (single, want0),
                      lm.get(0) == want0, lm)
                check("legacy project, other photos, single=%s: batch red-%s" % (single, tag_a),
                      calls["a"]["batch_name"] == "red-" + tag_a, calls["a"])
                sp0 = single or S.CWD12_SPECIES[0]
                remap, _u = MR._multiclass_remap([lm[0]])
                check("legacy project, other photos, single=%s: reads back as %s" % (single, sp0),
                      remap.get(0) == S.CWD12_SPECIES.index(sp0), remap)
        finally:
            RS._workspace = orig


def test_merge_remap():
    print("download-merge remap")
    names = ["Ragweed", "Crabgrass", "Carpet weed", "Waterhemp", "spurge",
             "Morningglory", "Nutsedge"]
    remap, unknown = MR._multiclass_remap(names)
    check("real ragweed -> cwd12 id 5", remap.get(0) == 5, remap)
    check("carpet weed -> cwd12 id 4", remap.get(2) == 4, remap)
    check("waterhemp -> cwd12 id 0 (was dropped)", remap.get(3) == 0, remap)
    check("morningglory -> cwd12 id 1", remap.get(5) == 1, remap)
    check("crabgrass / nutsedge / bare 'spurge' not joined",
          set(unknown) == {"Crabgrass", "spurge", "Nutsedge"}, unknown)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        loc = td / "dl"
        for stem, lines in (("hold", "0 .5 .5 .1 .1"), ("train", "0 .5 .5 .1 .1"),
                            ("ok", "0 .5 .5 .1 .1\n1 .2 .2 .1 .1\n3 .3 .3 .1 .1"),
                            ("oov", "1 .5 .5 .1 .1")):
            (loc / "train" / "images").mkdir(parents=True, exist_ok=True)
            (loc / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (loc / "train" / "images" / (stem + ".jpg")).write_bytes(b"x")
            (loc / "train" / "labels" / (stem + ".txt")).write_text(lines)
        kinds = {"hold": "holdout", "train": "cwd12_train"}
        out_i, out_l = td / "o" / "images", td / "o" / "labels"
        out_i.mkdir(parents=True)
        out_l.mkdir(parents=True)
        st = MR.merge_project_dir(loc, "p", remap, names,
                                  lambda p: kinds.get(p.stem), out_i, out_l)
        check("cwd12 photographs dropped whole, by kind",
              st["dropped_cwd12_photos"] == 2
              and st["dropped_cwd12_by_kind"] == {"holdout": 1, "cwd12_train": 1}, st)
        check("kept image, boxes in cwd12 ids",
              (out_l / "p_ok.txt").read_text().split("\n")[:2] == ["5 .5 .5 .1 .1", "0 .3 .3 .1 .1"],
              (out_l / "p_ok.txt").read_text())
        check("non-cwd12 boxes counted by name",
              st["dropped_oov_by_name"] == {"Crabgrass": 2}, st)
        check("image with no cwd12 box dropped and counted",
              st["dropped_no_cwd12_box"] == 1 and st["images"] == 1, st)


def test_species_project():
    print("--species project slug")
    check("PricklySida -> cwd12-pricklysida",
          MR._species_project("PricklySida") == "cwd12-pricklysida")
    check("Waterhemp -> cwd12-waterhemp", MR._species_project("Waterhemp") == "cwd12-waterhemp")
    try:
        MR._species_project("Goosegrass")
        check("Goosegrass refused (cwd12-goosegrass held SpottedSpurge)", False)
    except SystemExit as e:
        check("Goosegrass refused (cwd12-goosegrass held SpottedSpurge)", e.code == 2)
    ns = argparse.Namespace(species="Sicklepod", legacy_per_species=False, project="")
    try:
        MR._resolve_dl_targets(ns)
        check("download-merge --species Sicklepod refused", False)
    except SystemExit as e:
        check("download-merge --species Sicklepod refused", e.code == 2)
    ns = argparse.Namespace(species="", legacy_per_species=True, project="")
    t = dict(MR._resolve_dl_targets(ns))
    check("--legacy-per-species: cwd12-goosegrass is SpottedSpurge",
          t["cwd12-goosegrass"] == ("species", "SpottedSpurge"), t)


def test_photo_index():
    print("cwd12 photograph index")
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        hold = _img(td / "ref" / "h.jpg", 11, (320, 240))
        train = _img(td / "ref" / "t.jpg", 23, (320, 240))
        other = _img(td / "x" / "o.jpg", 37, (320, 240))
        # a Roboflow-style re-export: stretched to a square and re-encoded
        re = td / "x" / "h_rf.jpg"
        Image.open(hold).resize((256, 256)).save(re, quality=70)
        orig = MR._cwd12_ref_images
        try:
            MR._cwd12_ref_images = lambda: iter([("holdout", hold), ("cwd12_train", train)])
            idx = MR.cwd12_photo_index(td / "cache.json")
            check("re-exported holdout copy is a holdout photograph",
                  MR.cwd12_photo_kind(idx, re) == "holdout")
            check("train photograph found", MR.cwd12_photo_kind(idx, train) == "cwd12_train")
            check("other photograph passes", MR.cwd12_photo_kind(idx, other) is None)
            check("hash cache written", (td / "cache.json").is_file())
            idx2 = MR.cwd12_photo_index(td / "cache.json")
            check("cached index equal", len(idx2) == len(idx))
            MR._cwd12_ref_images = lambda: iter([("holdout", hold)])
            try:
                MR.cwd12_photo_index(td / "cache2.json")
                check("no train photographs -> refuse", False)
            except RuntimeError:
                check("no train photographs -> refuse", True)
        finally:
            MR._cwd12_ref_images = orig


def test_train_from_roboflow():
    print("train_from_roboflow")
    check("slot names are the trainer slot species",
          TR.V3_NAMES == S.TRAINER_SLOT_SPECIES == M.CANONICAL_12_SPECIES)
    mapping, unknown = TR._slot_mapping(S.CWD12_SPECIES)
    check("merge output (species, cwd12 ids) -> CWD12_ID_TO_SLOT",
          mapping == S.CWD12_ID_TO_SLOT and not unknown, mapping)
    mapping, unknown = TR._slot_mapping(["Ragweed", "Crabgrass", "Purslane", "Waterhemp"])
    check("real ragweed -> the slot holding Ragweed",
          TR.V3_NAMES[mapping[0]] == "Ragweed" and TR.V3_NAMES[mapping[2]] == "Purslane"
          and TR.V3_NAMES[mapping[3]] == "Waterhemp", mapping)
    check("crabgrass not aliased", unknown == {"Crabgrass": 1}, unknown)

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        si, sl = td / "src" / "images", td / "src" / "labels"
        si.mkdir(parents=True)
        sl.mkdir(parents=True)
        for stem in ("stemleak", "hold", "tr", "ok"):
            (si / (stem + ".jpg")).write_bytes(b"x")
            (sl / (stem + ".txt")).write_text("0 .5 .5 .1 .1\n1 .2 .2 .1 .1")
        kinds = {"hold": "holdout", "tr": "cwd12_train"}
        stats = {}
        ni, nl, leak = TR._remap_split(si, sl, ["Ragweed", "Nutsedge"], td / "out",
                                       {"stemleak"}, lambda p: kinds.get(p.stem), stats)
        check("stem + hash holdout copies counted as leak", leak == 2, leak)
        check("cwd12 train photograph skipped", stats["skipped_cwd12_train_photo"] == 1, stats)
        check("one image kept", ni == 1 and nl == 1, (ni, nl))
        slot = TR.V3_NAMES.index("Ragweed")
        check("ragweed box written to its slot",
              (td / "out" / "labels" / "ok.txt").read_text() == "%d .5 .5 .1 .1" % slot)
        check("nutsedge box dropped and counted",
              stats["dropped_oov_by_name"] == {"Nutsedge": 1}, stats)

        # holdout GT by id permutation, whatever data.yaml says
        base = td / "cwd12"
        (base / "valid" / "images").mkdir(parents=True)
        (base / "valid" / "labels").mkdir(parents=True)
        (base / "data.yaml").write_text("names: ['x']\n")
        _img(base / "valid" / "images" / "v.jpg", 3)
        (base / "valid" / "labels" / "v.txt").write_text(
            "\n".join("%d .5 .5 .1 .1" % i for i in range(12)))
        orig = TR.CWD12_YAML
        try:
            TR.CWD12_YAML = base / "data.yaml"
            TR.build_holdout_yaml("valid", td / "stage")
        finally:
            TR.CWD12_YAML = orig
        got = [int(line.split()[0]) for line in
               (td / "stage" / "holdout_valid" / "labels" / "v.txt").read_text().splitlines()]
        check("holdout GT id -> CWD12_ID_TO_SLOT",
              got == [S.CWD12_ID_TO_SLOT[i] for i in range(12)], got)
        import yaml
        y = yaml.safe_load(open(td / "stage" / "holdout_valid" / "data.yaml"))
        check("holdout data.yaml names are slot species", y["names"] == S.TRAINER_SLOT_SPECIES)


def test_owl():
    print("OWL chain")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        prop, gt = td / "prop", td / "gt"
        prop.mkdir()
        gt.mkdir()
        (prop / "a.txt").write_text("0 0.5 0.5 0.2 0.2  # red conf=0.9 src=owlv2")
        (gt / "a.txt").write_text("3 0.5 0.5 0.2 0.2\n10 0.1 0.1 0.1 0.1")
        r = OP.evaluate("SpottedSpurge", prop, gt)
        check("SpottedSpurge scored against cwd12 id 3",
              r["gt_cid"] == 3 and r["precision"] == 1.0, r)
        r = OP.evaluate("Goosegrass", prop, gt)
        check("Goosegrass scored against cwd12 id 10", r["gt_cid"] == 10 and r["precision"] == 0.0, r)
        try:
            OP.evaluate("Crabgrass", prop, gt)
            check("non-cwd12 species refused", False)
        except ValueError:
            check("non-cwd12 species refused", True)

        check("legacy exemplar config refused",
              PA.config_species({"species": "Goosegrass", "exemplars": []}) is None)
        check("species exemplar config accepted",
              PA.config_species({"species": "SpottedSpurge", "vocabulary": "species"})
              == "SpottedSpurge")
        try:
            PA.species_arg("Nutsedge")
            check("--species Nutsedge refused", False)
        except argparse.ArgumentTypeError:
            check("--species Nutsedge refused", True)
        check("--species 'Spotted spurge' -> SpottedSpurge",
              PA.species_arg("Spotted spurge") == "SpottedSpurge")

        # owl_upload_proposals: manifest required, id 0 uploaded as the species
        calls = []
        orig_call, orig_argv, orig_repo = OU.subprocess.call, sys.argv, OU.REPO
        cwd12_valid = td / "downloads" / "cottonweeddet12" / "valid" / "images"
        _img(cwd12_valid / "a.jpg", 1)
        _img(td / "field" / "a.jpg", 2)
        _img(td / "holdout_test" / "zz.jpg", 2)
        try:
            OU.REPO = td
            OU.subprocess.call = lambda argv, cwd=None: calls.append(argv) or 0
            sys.argv = ["x", "--species", "SpottedSpurge", "--prop-dir", str(prop),
                        "--gt-dir", str(gt), "--min-precision", "0.5"]
            check("proposal dir without manifest refused", OU.main() == 2 and not calls)
            (prop / PA.PROPOSALS_MANIFEST).write_text(json.dumps(
                {"species": "Goosegrass", "vocabulary": "species"}))
            check("manifest of another species refused", OU.main() == 2 and not calls)
            (prop / PA.PROPOSALS_MANIFEST).write_text(json.dumps(
                {"species": "SpottedSpurge", "vocabulary": "species"}))
            check("manifest without target_dir and no --images refused",
                  OU.main() == 2 and not calls)
            (prop / PA.PROPOSALS_MANIFEST).write_text(json.dumps(
                {"species": "SpottedSpurge", "vocabulary": "species",
                 "target_dir": str(cwd12_valid)}))
            rc = OU.main()
            check("gate passes and uploads", rc == 0 and len(calls) == 1, (rc, calls))
            if calls:
                a = calls[0]
                check("bulk-upload gets --single-species SpottedSpurge",
                      a[a.index("--single-species") + 1] == "SpottedSpurge", a)
                check("default holdout images: --photos cwd12",
                      a[a.index("--photos") + 1] == "cwd12", a)
                check("default --images is the manifest target_dir",
                      a[a.index("--images") + 1] == str(cwd12_valid), a)
            calls.clear()
            sys.argv = ["x", "--species", "SpottedSpurge", "--prop-dir", str(prop),
                        "--gt-dir", str(gt), "--min-precision", "0.5",
                        "--images", str(td / "holdout_test"), "--photos", "cwd12"]
            check("--images sharing no stem with the proposals refused",
                  OU.main() == 2 and not calls)
            sys.argv = ["x", "--species", "SpottedSpurge", "--prop-dir", str(prop),
                        "--gt-dir", str(gt), "--min-precision", "0.5",
                        "--images", str(td / "field")]
            check("other --images without --photos refused", OU.main() == 2 and not calls)
            sys.argv += ["--photos", "other"]
            rc = OU.main()
            check("other --images with --photos other uploads",
                  rc == 0 and calls and calls[0][calls[0].index("--photos") + 1] == "other",
                  (rc, calls))
        finally:
            OU.subprocess.call, sys.argv, OU.REPO = orig_call, orig_argv, orig_repo

        # owl_preannotate: species-era root, and no reuse of another run's dir
        check("default proposals dir is under owl_red_proposals_species",
              PA.default_proposals_dir("Goosegrass").parent.name
              == "owl_red_proposals_species")
        legacy = td / "legacy_goose"
        legacy.mkdir()
        (legacy / "old_a.txt").write_text("0 .5 .5 .1 .1")
        check("unstamped dir with proposals refused",
              PA.out_dir_conflict(legacy, "Goosegrass", cwd12_valid) is not None)
        check("same species + target reuses its dir",
              PA.out_dir_conflict(prop, "SpottedSpurge", cwd12_valid) is None)
        check("same species, other target refused",
              PA.out_dir_conflict(prop, "SpottedSpurge", td / "field") is not None)
        check("empty dir accepted",
              PA.out_dir_conflict(td / "nonexistent", "Goosegrass", cwd12_valid) is None)

        # export_owl_exemplars: species-keyed output, legacy-keyed bank dir.
        # The holdout guard fails closed without cwd12 test/valid images, so
        # this fake repository seeds it with an empty stem set.
        from weed_optimizer_framework.tools import synth_cutpaste as SC
        try:
            SC._holdout_guard()
            check("holdout guard fails closed without holdout images", False)
        except RuntimeError:
            check("holdout guard fails closed without holdout images", True)
        SC._HOLDOUT_GUARD.update(never=frozenset(M.NEVER_TRAIN_SLUGS),
                                 stems=frozenset(), _imgs=[])
        bank = td / "bank"
        _img(bank / "Goosegrass" / "c1.jpg", 5)
        _img(bank / "SpottedSpurge" / "c1.jpg", 6)
        orig_bank = EX.BANK_DIR
        try:
            EX.BANK_DIR = bank
            r = EX.export_one("SpottedSpurge", "bank", 5, td / "out")
        finally:
            EX.BANK_DIR = orig_bank
        cfg = json.load(open(td / "out" / "SpottedSpurge.json"))
        check("SpottedSpurge exemplars come from the legacy 'Goosegrass' bank dir",
              r["n"] == 1 and "/Goosegrass/" in cfg["exemplars"][0]["image"], cfg)
        check("exported config is accepted by owl_preannotate",
              PA.config_species(cfg) == "SpottedSpurge", cfg)

        # a bank marked "species" is keyed by species; legacy holdout crops skipped
        sbank = td / "sbank"
        _img(sbank / "SpottedSpurge" / "rf_x_1.jpg", 7)
        _img(sbank / "Goosegrass" / "rf_x_2.jpg", 8)
        (sbank / ".vocabulary").write_text("species\n")
        _img(bank / "Goosegrass" / "cottonweed_holdout_9.jpg", 9)
        try:
            EX.BANK_DIR = sbank
            d, _u = EX._bank_dir_for("SpottedSpurge")
            check("species bank: SpottedSpurge folder", d.name == "SpottedSpurge", d)
            EX.BANK_DIR = bank
            ex = EX._bank_exemplars("SpottedSpurge", 10)
            check("legacy bank: holdout crops not used as exemplars",
                  len(ex) == 1 and "cottonweed_holdout_" not in ex[0]["image"], ex)
        finally:
            EX.BANK_DIR = orig_bank


def test_policy_template():
    print("policy template")
    p = pathlib.Path(__file__).resolve().parents[1] / "weed_optimizer_framework" / \
        "tools" / "brain" / "policy_actions.json"
    d = json.load(open(p))
    tpl = json.dumps(d)
    check("owl_upload_proposals template names SpottedSpurge",
          "owl_upload_proposals --species SpottedSpurge" in tpl
          and "--species Goosegrass" not in tpl)


if __name__ == "__main__":
    test_upload_labelmaps()
    test_bulk_upload()
    test_merge_remap()
    test_species_project()
    test_photo_index()
    test_train_from_roboflow()
    test_owl()
    test_policy_template()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        sys.exit(1)
    print("all passed")
