#!/usr/bin/env python3
"""INC Steps 0.3 and 0.4: the platform fixes the incremental protocol stands on.

docs/INCREMENTAL_PROTOCOL.md lists defects that made earlier runs unmeasurable;
these tests pin the fixes in the three live modules that had them.

mega_trainer
  * The optimizer is always passed to model.train. Ultralytics' default
    "auto" discards lr0 (MuSGD at 0.01 against a configured 0.001,
    REGRESSION_DIAGNOSIS.md §5a); the installed Ultralytics is checked too,
    so a version that changes this fails here, not silently in a campaign.
  * A merge clears what the previous merge into the same dir left behind
    (the stale oversample_* links of §5i), leaves anything else in the dir,
    and refuses a dir outside results/framework.
  * The merge refuses unless the whole sealed holdout is found: exactly
    HOLDOUT_IMAGES (1,977) distinct photographs, every one hashable. A copy
    missing one photograph is refused (under the old 1,900 floor a copy
    missing 77 was accepted and their renamed re-uploads trained); so is an
    unexpected extra one. An image that cannot be hashed is skipped and
    counted instead of merged unchecked.
  * With an INC dev manifest, val is dev (labels in trainer slots) and dev
    images are kept out of train, by the never-train index when it exists
    and by the dev manifest's own hashes otherwise. Without it, val stays
    the sealed holdout and the log says so loudly. A dev manifest that
    cannot be staged exactly (empty, a missing or changed file, a class with
    no trainer slot, a manifest that differs from LOCK.json) is refused
    before the previous merge is cleared.
  * train_yolo_mega's summary (what run_m1_merged_seeds.sh keeps as the job
    artifact) says which split results.csv's mAP was computed on.
  * The min_dino_score gate refuses slug scores that do not record a complete
    reference-pool guard, before anything is cleared.
dataset_discovery
  * The Roboflow phase is asked for the remaining quota, not max(quota, 8);
    harvest_new_datasets still returns at most max_new datasets when a source
    returns more than asked, and what it cut is neither registered (so never
    marked used_for_training) nor left on disk.
dinov2_curator
  * The reference pool leaves out holdout stems and every image within
    HOLDOUT_NEAR_DUP_BITS of a holdout / dev / exam image, and the weed pool
    is not built unless the whole holdout is found.
  * score-all refuses a weed pool whose meta does not show a complete guard
    (every pool built before the guard), and stamps the guard on each score.
  * A slug's candidate sample is seeded by the slug, not its registry index,
    and drawn from the sorted listing of the whole slug, so it does not
    depend on the filesystem's enumeration order or the slug's location.

No GPU, no network: synthetic images in a temp dir, Ultralytics from
yolo11n.yaml on the CPU.

Run:  python3 tests/test_inc_step0_fixes.py
"""
import copy
import json
import logging
import os
import pathlib
import random
import shutil
import sys
import tempfile
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_step0_"))
# Before any import: common.py resolves INC_DIR, and dinov2_curator creates its
# output dir under REPO_ROOT, at import time. Nothing may touch the real ones.
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO_ROOT"] = str(TMP / "repo")
os.environ["YOLO_OFFLINE"] = "true"
os.environ.pop("DINO_DOMAIN", None)
os.environ.pop("BRAIN_MAX_NEW", None)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.config import Config  # noqa: E402
from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402
from weed_optimizer_framework.tools import mega_trainer as M  # noqa: E402
from weed_optimizer_framework.tools import dataset_discovery as DD  # noqa: E402
from weed_optimizer_framework.tools import dinov2_curator as CUR  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402

assert str(C.INC_DIR) == str(TMP / "inc"), C.INC_DIR
for _h in logging.getLogger().handlers:      # dinov2_curator's basicConfig
    _h.setLevel(logging.WARNING)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


class LogCapture(logging.Handler):
    def __init__(self, *names):
        super().__init__(logging.INFO)
        self.records = []
        self.loggers = [logging.getLogger(n) for n in names]

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))

    def __enter__(self):
        for lg in self.loggers:
            lg.addHandler(self)
            lg.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc):
        for lg in self.loggers:
            lg.removeHandler(self)

    def warnings(self):
        return [m for lvl, m in self.records if lvl >= logging.WARNING]


class Patch:
    """Set attributes for the duration of a with-block, then restore them."""

    def __init__(self, *triples):
        self.triples = triples
        self.saved = []

    def __enter__(self):
        for obj, name, value in self.triples:
            self.saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


# ------------------------------------------------------------------ images
def grid_img(path, seed, paint=()):
    """A 9x8 grid of random grey levels blown up to 144x128. Its dHash is the
    grid's own left-right comparisons, so copies are 0 bits apart, different
    seeds ~32 bits apart, and `paint` rows (last column pushed past its left
    neighbour) move the hash by a bit or two: a re-encoded near copy."""
    from PIL import Image
    rng = random.Random(seed)
    grid = [[rng.randrange(256) for _ in range(9)] for _ in range(8)]
    for r in paint:
        grid[r][8] = 255 if grid[r][8] <= grid[r][7] else 0
    im = Image.new("L", (9, 8))
    im.putdata([v for row in grid for v in row])
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.resize((144, 128), Image.NEAREST).convert("RGB").save(path, quality=95)
    return path


def label(path, text="0 0.5 0.5 0.2 0.2\n"):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def bits(a, b):
    return bin(M._dhash(a) ^ M._dhash(b)).count("1")


# ----------------------------------------------------------------- fixtures
FW = TMP / "fw" / "results" / "framework"
HOLDOUT = TMP / "cwd12"                     # a stand-in for downloads/cottonweeddet12
HOLDOUT_IMGS = [grid_img(HOLDOUT / "test" / "images" / ("2021_hold_%d.jpg" % i), 100 + i)
                for i in range(3)]
for _p in HOLDOUT_IMGS:
    label(HOLDOUT / "test" / "labels" / (_p.stem + ".txt"))
DEV_IMGS = [grid_img(HOLDOUT / "train" / "images" / ("2021_dev_%d.jpg" % i), 200 + i)
            for i in range(2)]
# INC ids: dev_0 holds Ragweed (5), dev_1 Waterhemp (0) and Goosegrass (10).
DEV_LABELS = [label(HOLDOUT / "train" / "labels" / "2021_dev_0.txt", "5 0.5 0.5 0.2 0.2\n"),
              label(HOLDOUT / "train" / "labels" / "2021_dev_1.txt",
                    "0 0.3 0.3 0.1 0.1\n10 0.7 0.7 0.1 0.1\n")]
EXAM_IMG = grid_img(TMP / "exam" / "ood22_a.jpg", 300)

def holdout_patch(dirs=None, expected=3):
    """The fixture's three holdout images are the whole sealed holdout."""
    dirs = [HOLDOUT / "test" / "images"] if dirs is None else dirs
    return Patch((M, "_holdout_image_dirs", lambda: list(dirs)),
                 (M, "HOLDOUT_IMAGES", expected))


def holdout_variant(name, keep=(0, 1, 2), extra=()):
    """A holdout image dir holding copies of HOLDOUT_IMGS[keep] plus `extra`
    [(file name, seed or bytes)]; a partial, padded or damaged copy."""
    d = TMP / "holdout_variants" / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i in keep:
        shutil.copyfile(HOLDOUT_IMGS[i], d / HOLDOUT_IMGS[i].name)
    for fname, src in extra:
        if isinstance(src, bytes):
            (d / fname).write_bytes(src)
        else:
            grid_img(d / fname, src)
    return d


def write_dev_manifest():
    rows = [{"image": str(p), "label": str(lb), "sha256": C.sha256_file(p),
             "label_sha256": C.sha256_file(lb), "source": "cwd12_train",
             "session": "2021_dev", "key": "cwd12_train__" + p.stem}
            for p, lb in zip(DEV_IMGS, DEV_LABELS)]
    return C.write_manifest(C.manifest_path("dev"), rows)


def write_nevertrain_index():
    entries = ([[M._dhash(p), "test", "cwd12_test__" + p.stem] for p in HOLDOUT_IMGS]
               + [[M._dhash(p), "dev", "cwd12_train__" + p.stem] for p in DEV_IMGS]
               + [[M._dhash(EXAM_IMG), "ood22", "ood22__a"]])
    C.NEVER_TRAIN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    C.NEVER_TRAIN_INDEX.write_text(json.dumps({"entries": entries,
                                               "min_expected": len(entries)}))


def clear_inc():
    shutil.rmtree(C.INC_DIR, ignore_errors=True)


class FakeDisc:
    """DatasetDiscovery with an in-memory registry: the real one reads and
    writes results/framework/dataset_registry.json."""
    registry = None

    def __init__(self):
        pass

    def mark_as_used(self, *a, **k):
        raise AssertionError("mark_as_used must not run in these tests")


def build_source(name, files):
    """A registry dataset <TMP>/data/<name>/{images,labels}; files is
    [(image_name, src_path_or_seed_or_bytes)]."""
    root = TMP / "data" / name
    for img_name, src in files:
        dst = root / "images" / img_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, bytes):
            dst.write_bytes(src)
        elif isinstance(src, int):
            grid_img(dst, src)
        else:
            shutil.copyfile(src, dst)
        label(root / "labels" / (pathlib.Path(img_name).stem + ".txt"))
    return {"local_path": str(root), "annotation": "yolo",
            "class_names": ["Waterhemp", "Ragweed"], "status": "downloaded"}


def merge_env(registry):
    FakeDisc.registry = {"datasets": registry}
    return Patch((M, "DatasetDiscovery", FakeDisc),
                 (M, "update_registry", lambda path, fn: fn({"datasets": {}})),
                 (Config, "FRAMEWORK_DIR", str(FW)))


# ------------------------------------------------------------------- tests
def test_fixture_hashes():
    print("fixture sanity (the dHash distances the tests rely on)")
    near = grid_img(TMP / "probe" / "near.jpg", 100, paint=(0, 1))
    b = bits(near, HOLDOUT_IMGS[0])
    check("a painted copy is a near copy (1..%d bits)" % M.HOLDOUT_NEAR_DUP_BITS,
          1 <= b <= M.HOLDOUT_NEAR_DUP_BITS, b)
    far = min(bits(a, c) for a in HOLDOUT_IMGS + DEV_IMGS + [EXAM_IMG]
              for c in HOLDOUT_IMGS + DEV_IMGS + [EXAM_IMG] if a != c)
    check("distinct fixture images are far apart (> %d bits)" % M.HOLDOUT_NEAR_DUP_BITS,
          far > M.HOLDOUT_NEAR_DUP_BITS, far)


def test_merge_guards():
    print("mega_trainer._merge_datasets: holdout guard, clearing, unhashable")
    clear_inc()
    reg = {"t_src": build_source("t_src", [
        ("a_first.jpg", 1), ("b_keep.jpg", 2), ("c_keep.jpg", 3),
        ("renamed_holdout.jpg", HOLDOUT_IMGS[0]),
        ("broken.jpg", b"not an image at all"),
    ])}
    out = FW / "merged_iter_t1"
    stale_img = out / "train" / "images" / "oversample_3_9_stale_from_last_round.jpg"
    stale_lbl = out / "train" / "labels" / "oversample_3_9_stale_from_last_round.txt"
    run_ckpt = out / "train2" / "weights" / "best.pt"

    def plant():
        stale_img.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(stale_img):
            os.symlink(str(HOLDOUT_IMGS[1]), stale_img)
        label(stale_lbl)
        run_ckpt.parent.mkdir(parents=True, exist_ok=True)
        run_ckpt.write_bytes(b"ckpt")

    # 1. no holdout reachable -> refuse, and touch nothing
    plant()
    with merge_env(reg), Patch((M, "_holdout_image_dirs", lambda: [])):
        try:
            M._merge_datasets(str(out))
            check("a merge with no holdout images is refused", False)
        except RuntimeError as e:
            check("a merge with no holdout images is refused",
                  "holdout guard incomplete" in str(e), e)
    check("the refused merge left the previous merge dir untouched",
          os.path.lexists(stale_img) and not (out / "data.yaml").exists())

    # 2. the whole holdout, not a floor under it
    check("the sealed holdout is pinned at exactly 1,977 photographs",
          M.HOLDOUT_IMAGES == 1977, M.HOLDOUT_IMAGES)
    with merge_env(reg), holdout_patch(expected=4):
        try:
            M._merge_datasets(str(out))
            check("3 holdout images where 4 are expected is refused", False)
        except RuntimeError as e:
            check("3 holdout images where 4 are expected is refused",
                  "3 distinct holdout image(s)" in str(e), e)

    # 2a. one photograph missing from the holdout copy, and a renamed re-upload of
    # exactly that photograph in the registry. Neither the stem filter nor the
    # dHash guard knows it, so a guard that accepts a partial copy trains on it.
    partial = holdout_variant("partial", keep=(0, 1))
    leak = dict(reg, t_leak=build_source("t_leak", [
        (HOLDOUT_IMGS[2].stem + "_jpg.rf.abc.jpg", HOLDOUT_IMGS[2])]))
    with merge_env(leak), holdout_patch(dirs=[partial]):
        try:
            M._merge_datasets(str(out))
            check("a holdout missing one photograph is refused", False)
        except RuntimeError as e:
            check("a holdout missing one photograph is refused",
                  "2 distinct holdout image(s)" in str(e)
                  and "expected exactly 3" in str(e), e)
    check("... before the previous merge dir is touched",
          os.path.lexists(stale_img) and not (out / "data.yaml").exists())

    # 2b. an extra photograph: not the holdout the count was taken from
    padded = holdout_variant("padded", extra=[("2021_extra.jpg", 150)])
    with merge_env(reg), holdout_patch(dirs=[padded]):
        try:
            M._merge_datasets(str(out))
            check("a holdout with an unexpected extra image is refused", False)
        except RuntimeError as e:
            check("a holdout with an unexpected extra image is refused",
                  "4 distinct holdout image(s)" in str(e), e)

    # 2c. every stem present, one file damaged: it cannot be hash-guarded
    damaged = holdout_variant("damaged", keep=(0, 1),
                              extra=[(HOLDOUT_IMGS[2].name, b"truncated")])
    with merge_env(reg), holdout_patch(dirs=[damaged]):
        try:
            M._merge_datasets(str(out))
            check("a holdout image that cannot be hashed is refused", False)
        except RuntimeError as e:
            check("a holdout image that cannot be hashed is refused",
                  "1 cannot be hashed" in str(e), e)

    # 2d. two roots holding the same holdout (the cwd-relative and the absolute
    # candidate on the cluster) are one complete holdout, not twice as many
    twin = holdout_variant("twin")
    with holdout_patch(dirs=[HOLDOUT / "test" / "images", twin]):
        stems, hashes = M._load_holdout_guard()
    check("the same holdout under two roots is complete",
          len(stems) == 3 and len(hashes) == 3, (len(stems), len(hashes)))

    # 3. an empty holdout dir -> refuse (dirs exist, images do not)
    empty = TMP / "empty_holdout"
    empty.mkdir(exist_ok=True)
    with merge_env(reg), Patch((M, "_holdout_image_dirs", lambda: [empty])):
        try:
            M._merge_datasets(str(out))
            check("an empty holdout dir is refused", False)
        except RuntimeError:
            check("an empty holdout dir is refused", True)

    # 4. a merge dir outside results/framework is refused before anything is cleared
    outside = TMP / "not_framework" / "merged"
    label(outside / "train" / "labels" / "keep.txt")
    with merge_env(reg), holdout_patch():
        try:
            M._merge_datasets(str(outside))
            check("a merge dir outside results/framework is refused", False)
        except RuntimeError as e:
            check("a merge dir outside results/framework is refused",
                  "refusing to clear" in str(e), e)
        try:
            M._clear_merge_output(str(FW))
            check("results/framework itself is refused", False)
        except RuntimeError:
            check("results/framework itself is refused", True)
    check("the refused outside dir still holds its files",
          (outside / "train" / "labels" / "keep.txt").exists())

    # 5. a guarded merge: stale outputs cleared, run dirs kept, unhashable skipped
    with merge_env(reg), holdout_patch(), LogCapture(M.__name__) as cap:
        _, data_yaml, stats, used, names = M._merge_datasets(str(out))
    train_imgs = sorted(p.name for p in (out / "train" / "images").iterdir())
    check("the stale oversample link of the previous merge is gone",
          not os.path.lexists(stale_img) and not stale_lbl.exists(), train_imgs)
    check("the stale link's target was not deleted", HOLDOUT_IMGS[1].exists())
    check("a run dir beside the merge outputs survives (hot_reload trains there)",
          run_ckpt.exists())
    check("the unhashable image is skipped and counted",
          stats["skipped_unhashable"] == 1
          and not any("broken" in n for n in train_imgs), stats)
    check("the renamed holdout copy is blocked by its hash",
          stats["skipped_holdout_hash"] == 1
          and not any("renamed_holdout" in n for n in train_imgs), stats)
    check("the three clean images are merged",
          stats["images"] == 3 and used == ["t_src"], stats)
    check("without INC, val stays the merged split", stats["val_source"] == "merged_split")
    check("the counts are in the merge summary log",
          any("Unhashable skipped: 1" in m for _, m in cap.records))

    # 6. a second merge into the same dir produces the same train set
    first = train_imgs
    with merge_env(reg), holdout_patch():
        _, _, stats2, _, _ = M._merge_datasets(str(out))
    second = sorted(p.name for p in (out / "train" / "images").iterdir())
    check("re-merging into the same dir does not accumulate oversample copies",
          first == second, (len(first), len(second)))


def test_merge_inc_dev():
    print("mega_trainer._merge_datasets: INC dev as val, dev and exams out of train")
    clear_inc()
    reg = {"t_inc": build_source("t_inc", [
        ("a_first.jpg", 11), ("b_keep.jpg", 12),
        ("dev_copy_0.jpg", DEV_IMGS[0]), ("dev_copy_1.jpg", DEV_IMGS[1]),
        ("exam_copy.jpg", EXAM_IMG),
    ])}
    out = FW / "merged_iter_t2"
    val_root = str(HOLDOUT)

    # a. no INC files: today's behaviour, val = sealed holdout, loud warning
    with merge_env(reg), holdout_patch(), LogCapture(M.__name__) as cap:
        _, data_yaml, stats, _, _ = M._merge_datasets(str(out), val_dataset_root=val_root)
    yaml_text = pathlib.Path(data_yaml).read_text()
    check("without a dev manifest val is the staged sealed holdout",
          stats["val_source"] == "cwd12_holdout"
          and "val: %s" % (out / "cwd12_holdout" / "images") in yaml_text, yaml_text)
    check("... and a warning says best.pt is selected on the sealed holdout",
          any("SEALED HOLDOUT" in m for m in cap.warnings()), cap.warnings())
    check("... and the dev/exam copies are not excluded (no INC files yet)",
          stats["skipped_inc_nevertrain"] == 0 and stats["images"] == 5, stats)

    # b. dev manifest, no never-train index: exclusion by the dev images' own hashes
    write_dev_manifest()
    with merge_env(reg), holdout_patch(), LogCapture(M.__name__) as cap:
        _, data_yaml, stats, _, _ = M._merge_datasets(str(out), val_dataset_root=val_root)
    yaml_text = pathlib.Path(data_yaml).read_text()
    val_dir = out / M.INC_DEV_VAL_DIR
    check("with a dev manifest val is the materialised dev split",
          stats["val_source"] == "inc_dev"
          and "val: %s" % (val_dir / "images") in yaml_text, yaml_text)
    check("... and the stats name the val path", stats.get("val_path") == str(val_dir / "images"),
          stats.get("val_path"))
    check("... and the sealed holdout is not staged",
          not (out / "cwd12_holdout").exists())
    check("... and the stale holdout val of the previous merge is gone",
          "cwd12_holdout" not in yaml_text)
    check("both dev copies are kept out of train (dev manifest hashes)",
          stats["skipped_inc_nevertrain"] == 2
          and stats.get("inc_nevertrain_by_split") == {"dev": 2}, stats)
    check("the exam copy still trains when only the dev manifest exists",
          stats["images"] == 3, stats)
    got = {p.name: p.read_text().split() for p in (val_dir / "labels").glob("*.txt")}
    slot = S.CWD12_ID_TO_SLOT
    check("dev labels are mapped from INC ids to trainer slots",
          got.get("cwd12_train__2021_dev_0.txt", [None])[0] == str(slot[5])
          and [got.get("cwd12_train__2021_dev_1.txt", [])[i] for i in (0, 5)]
          == [str(slot[0]), str(slot[10])], got)
    check("the slot really holds that species (Ragweed -> slot %d)" % slot[5],
          M.CANONICAL_12_SPECIES[slot[5]] == "Ragweed"
          and M.CANONICAL_12_SPECIES[slot[10]] == "Goosegrass")
    check("coordinates are carried over verbatim",
          got.get("cwd12_train__2021_dev_1.txt", [])[1:5] == ["0.3", "0.3", "0.1", "0.1"], got)
    check("the manifest's own label files are not edited",
          DEV_LABELS[0].read_text() == "5 0.5 0.5 0.2 0.2\n")
    imgs = sorted(p.name for p in (val_dir / "images").iterdir())
    check("the dev val holds exactly the dev images, as symlinks",
          imgs == ["cwd12_train__2021_dev_0.jpg", "cwd12_train__2021_dev_1.jpg"]
          and all(os.path.islink(val_dir / "images" / n) for n in imgs), imgs)
    check("no INC data.yaml (nc 13) is left inside the slot-space val dir",
          not (val_dir / "data.yaml").exists())
    check("no 'sealed holdout' warning once dev is the val set",
          not any("SEALED HOLDOUT" in m for m in cap.warnings()), cap.warnings())

    # c. never-train index present: dev and exam copies both excluded
    write_nevertrain_index()
    with merge_env(reg), holdout_patch():
        _, _, stats, _, _ = M._merge_datasets(str(out), val_dataset_root=val_root)
    check("with the never-train index the exam copy is excluded too",
          stats["skipped_inc_nevertrain"] == 3
          and stats.get("inc_nevertrain_by_split") == {"dev": 2, "ood22": 1}
          and stats["images"] == 2, stats)

    # d-g. a dev manifest that cannot be staged exactly is refused before the
    # previous merge (the one from c) is cleared: before the fix an empty
    # manifest or an OtherPlant box raised only after the train set was
    # rewritten, and a missing image became a dangling symlink that Ultralytics
    # drops as corrupt while the run still reports val_source=inc_dev.
    good = C.read_manifest(C.manifest_path("dev"))
    yaml_before = (out / "data.yaml").read_text()

    def refused(name, rows, needle):
        C.write_manifest(C.manifest_path("dev"), rows)
        with merge_env(reg), holdout_patch():
            try:
                M._merge_datasets(str(out))
                check(name, False)
            except RuntimeError as e:
                check(name, needle in str(e), e)
        check("... and the previous merge is left as it was",
              (out / "data.yaml").is_file()
              and (out / "data.yaml").read_text() == yaml_before
              and len(list((val_dir / "images").iterdir())) == 2)

    bad = label(TMP / "bad_dev" / "x.txt", "12 0.5 0.5 0.2 0.2\n")
    refused("an OtherPlant box in dev is refused",
            [dict(good[0], label=str(bad), label_sha256=C.sha256_file(bad))] + good[1:],
            "not a cwd12 species")
    refused("an empty dev manifest is refused", [], "is empty")
    refused("a dev image that does not exist is refused (index present, so no dHash pass)",
            [dict(good[0], image=str(TMP / "gone" / "missing.jpg"))] + good[1:],
            "image missing")
    changed = grid_img(TMP / "changed_dev" / "2021_dev_0.jpg", 777)
    refused("a dev image that differs from its manifest sha256 is refused",
            [dict(good[0], image=str(changed))] + good[1:],
            "differs from its manifest sha256")
    changed_lbl = label(TMP / "changed_dev" / "2021_dev_0.txt", "5 0.4 0.4 0.2 0.2\n")
    refused("a dev label that differs from its manifest sha256 is refused",
            [dict(good[0], label=str(changed_lbl))] + good[1:],
            "differs from its manifest sha256")

    # h. LOCK.json: a dev manifest that matches its lock is staged; one that
    # does not is refused
    dev_sha = C.write_manifest(C.manifest_path("dev"), good)
    C.LOCK_PATH.write_text(json.dumps({"manifests": {"dev": dev_sha}}))
    with merge_env(reg), holdout_patch():
        _, _, stats, _, _ = M._merge_datasets(str(out))
    check("a dev manifest that matches LOCK.json is staged",
          stats["val_source"] == "inc_dev", stats["val_source"])
    yaml_before = (out / "data.yaml").read_text()
    C.LOCK_PATH.write_text(json.dumps({"manifests": {"dev": "0" * 64}}))
    refused("a dev manifest that differs from LOCK.json is refused", good,
            "changed since it was locked")
    clear_inc()


def test_dino_gate():
    print("mega_trainer._merge_datasets: min_dino_score needs guarded slug scores")
    clear_inc()
    reg = {"t_hi": build_source("t_hi", [("hi.jpg", 31)]),
           "t_lo": build_source("t_lo", [("lo.jpg", 32)])}
    out = FW / "merged_iter_t3"
    scores = TMP / "dino" / "slug_scores.json"
    scores.parent.mkdir(parents=True, exist_ok=True)
    complete = {"source": "mega_trainer_holdout", "n_hashes": 1977,
                "n_holdout_stems": 1977, "complete": True}

    def write_scores(guard):
        recs = {"t_hi": {"slug": "t_hi", "status": "ok", "score": 0.8},
                "t_lo": {"slug": "t_lo", "status": "ok", "score": 0.1},
                "t_err": {"slug": "t_err", "status": "error", "score": None}}
        if guard is not None:
            for r in recs.values():
                r["ref_guard"] = guard
        scores.write_text(json.dumps(recs))

    with merge_env(reg), holdout_patch():
        M._merge_datasets(str(out))
    yaml_before = (out / "data.yaml").read_text()

    for name, guard in (("scores with no guard record (every pool before INC 0.3)", None),
                        ("scores from a pool whose guard was incomplete",
                         dict(complete, complete=False))):
        write_scores(guard)
        with merge_env(reg), holdout_patch(), Patch((M, "DINO_SCORES_PATH", str(scores))):
            try:
                M._merge_datasets(str(out), min_dino_score=0.5)
                check("%s are refused" % name, False)
            except RuntimeError as e:
                check("%s are refused" % name, "complete holdout guard" in str(e), e)
        check("... before the previous merge is cleared",
              (out / "data.yaml").is_file() and (out / "data.yaml").read_text() == yaml_before)

    write_scores(complete)
    with merge_env(reg), holdout_patch(), Patch((M, "DINO_SCORES_PATH", str(scores))):
        _, _, stats, used, _ = M._merge_datasets(str(out), min_dino_score=0.5)
    check("guarded scores drive the gate", used == ["t_hi"] and stats["skipped_low_dino"] == 1,
          (used, stats["skipped_low_dino"]))

    with merge_env(reg), holdout_patch(), \
            Patch((M, "DINO_SCORES_PATH", str(TMP / "dino" / "absent.json"))), \
            LogCapture(M.__name__) as cap:
        _, _, stats, used, _ = M._merge_datasets(str(out), min_dino_score=0.5)
    check("a missing score file still disables the gate with a warning (run_m1 "
          "refuses that case itself)",
          sorted(used) == ["t_hi", "t_lo"] and any("DISABLED" in m for m in cap.warnings()),
          (used, cap.warnings()))


class FakeYOLO:
    calls = []

    def __init__(self, weights):
        self.weights = weights

    def add_callback(self, event, fn):
        pass

    def train(self, **kw):
        FakeYOLO.calls.append(kw)
        save = pathlib.Path(kw["project"]) / kw["name"]
        (save / "weights").mkdir(parents=True, exist_ok=True)
        (save / "weights" / "best.pt").write_bytes(b"x")
        self.trainer = types.SimpleNamespace(save_dir=save, epoch=0)


def test_optimizer_kwargs():
    print("mega_trainer.train_yolo_mega: the optimizer reaches model.train")
    import ultralytics
    dev_val = str(TMP / "m" / M.INC_DEV_VAL_DIR / "images")
    fake_merge = lambda out_dir, **kw: (out_dir, str(TMP / "d.yaml"),
                                        {"images": 150, "val_source": "inc_dev",
                                         "val_path": dev_val}, [],
                                        list(M.CANONICAL_12_SPECIES))
    FakeDisc.registry = {"datasets": {}, "mega_round_count": 0}
    trace = TMP / "trace.jsonl"
    os.environ["WEED_ALLOW_CPU_TRAIN"] = "1"
    try:
        with Patch((ultralytics, "YOLO", FakeYOLO), (M, "_merge_datasets", fake_merge),
                   (M, "DatasetDiscovery", FakeDisc),
                   (M, "update_registry", lambda path, fn: {"mega_round_count": 1}),
                   (Config, "FRAMEWORK_DIR", str(FW))):
            _, summary = M.train_yolo_mega({"base_model": "fake.pt", "lr": 0.0123,
                                            "epochs": 1, "trace_path": str(trace)},
                                           "t_opt", run_tag="r1")
            M.train_yolo_mega({"base_model": "fake.pt", "lr": 0.0123, "epochs": 1,
                               "optimizer": "AdamW"}, "t_opt", run_tag="r2")
    finally:
        os.environ.pop("WEED_ALLOW_CPU_TRAIN", None)
    kw1, kw2 = FakeYOLO.calls[0], FakeYOLO.calls[1]
    check("the default optimizer is passed explicitly as SGD",
          kw1.get("optimizer") == "SGD", kw1.get("optimizer"))
    check("lr0 is passed alongside it", kw1.get("lr0") == 0.0123, kw1.get("lr0"))
    check("a strategy's optimizer is passed through", kw2.get("optimizer") == "AdamW")
    recs = [json.loads(ln) for ln in trace.read_text().splitlines()]
    start = [r for r in recs if r.get("kind") == "start"]
    end = [r for r in recs if r.get("kind") == "end"]
    check("the run trace records the optimizer and the val source",
          start and start[0]["strategy"].get("optimizer") == "SGD"
          and start[0].get("val_source") == "inc_dev"
          and end and end[0].get("val_source") == "inc_dev", recs)
    # run_m1_merged_seeds.sh keeps this summary as the job artifact, and the
    # round scheduler reads that run's results.csv as a metric: once dev is the
    # val set, that mAP is a dev number and the artifact has to say so.
    check("the returned summary says which split results.csv's mAP is on",
          summary.get("val_source") == "inc_dev" and summary.get("val_path") == dev_val,
          {k: summary.get(k) for k in ("val_source", "val_path")})


def test_optimizer_ultralytics():
    """The premise, on the installed Ultralytics: "auto" discards lr0, a named
    optimizer uses it. Read from trainer.optimizer at on_train_start, after
    the scheduler has stamped each group's initial_lr."""
    print("Ultralytics: optimizer=SGD honours lr0, auto does not")
    from PIL import Image
    from ultralytics import YOLO
    import ultralytics
    ds = TMP / "yolo_tiny"
    for i in range(4):
        (ds / "images").mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (96, 96), (30 * i, 90, 160))
        im.paste((250, 40, 40), (20, 20, 60, 60))
        im.save(ds / "images" / ("%d.jpg" % i))
        label(ds / "labels" / ("%d.txt" % i), "%d 0.416 0.416 0.416 0.416\n" % (i % 2))
    yaml = ds / "data.yaml"
    yaml.write_text("path: %s\ntrain: images\nval: images\nnc: 2\nnames: [a, b]\n" % ds)

    def initial_lrs(optimizer):
        seen = {}

        def grab(trainer):
            opt = getattr(trainer, "optimizer", None)
            assert opt is not None and opt.param_groups, (
                "Ultralytics %s: trainer.optimizer missing at on_train_start; "
                "this check reads it" % ultralytics.__version__)
            seen["lr"] = sorted({round(g.get("initial_lr", g["lr"]), 8)
                                 for g in opt.param_groups})
            seen["kind"] = type(opt).__name__
        model = YOLO("yolo11n.yaml")
        model.add_callback("on_train_start", grab)
        model.train(data=str(yaml), epochs=1, imgsz=64, batch=2, device="cpu",
                    workers=0, plots=False, val=False, optimizer=optimizer,
                    lr0=0.0123, project=str(TMP / "yolo_runs"), name=optimizer,
                    exist_ok=True, verbose=False, amp=False)
        return seen

    sgd = initial_lrs("SGD")
    auto = initial_lrs("auto")
    check("Ultralytics %s: optimizer=SGD trains at lr0" % ultralytics.__version__,
          sgd.get("kind") == "SGD" and sgd.get("lr") == [0.0123], sgd)
    check("Ultralytics %s: optimizer=auto ignores lr0 (why it is never left implicit)"
          % ultralytics.__version__, 0.0123 not in auto.get("lr", [0.0123]), auto)


def test_harvest_cap():
    print("dataset_discovery.harvest_new_datasets: Roboflow asked for the quota, "
          "the result capped at max_new")
    import huggingface_hub
    from weed_optimizer_framework.tools import extra_sources, roboflow_source

    class FakeApi:
        def list_datasets(self, **kw):
            return []

    data_dir = TMP / "harvest" / "datasets"
    outside_dir = TMP / "harvest" / "elsewhere" / "rf_outside"
    rf_asked = []

    def entry(slug, local_path, src):
        pathlib.Path(local_path).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(local_path) / "img.jpg").write_bytes(b"x")
        return {"slug": slug, "hf_id": src + "/" + slug, "reason": src + ":weed",
                "stats": {"status": "downloaded", "images": 60, "labeled": 60},
                "info": {"source": src, "status": "downloaded", "local_path": str(local_path),
                         "annotation": "yolo", "description": "weed detection " + slug,
                         "used_for_training": False, "training_runs": []}}

    def fake_gh(data_dir, queries, already_known_cb, max_new=3):
        return [entry("gh_a", pathlib.Path(data_dir) / "gh_a", "github")]

    def fake_rf(data_dir, queries, already_known_cb, max_new=10):
        # A source that returns more than it was asked for: what the final cut
        # is still there for.
        rf_asked.append(max_new)
        out = [entry("rf_%d" % i, pathlib.Path(data_dir) / ("rf_%d" % i), "roboflow")
               for i in range(7)]
        out.append(entry("rf_outside", outside_dir, "roboflow"))
        return out

    disc = object.__new__(DD.DatasetDiscovery)   # no registry file, no data dir scan
    disc.data_dir = str(data_dir)
    disc.registry = {"datasets": {"pre_existing": {"status": "downloaded",
                                                   "used_for_training": True}},
                     "current_round": 4}
    saved = []
    disc._save_registry = lambda registry=None: saved.append(copy.deepcopy(disc.registry))
    with Patch((huggingface_hub, "HfApi", FakeApi),
               (extra_sources, "harvest_github_datasets", fake_gh),
               (extra_sources, "harvest_kaggle_datasets", lambda **kw: []),
               (roboflow_source, "harvest_roboflow_datasets", fake_rf)), \
            LogCapture(DD.__name__) as cap:
        ret = disc.harvest_new_datasets(max_new=3, queries=["weed detection"],
                                        strict_topic=False, domain="weed")
    kept = [r["slug"] for r in ret["results"]]
    cut = ["rf_%d" % i for i in range(2, 7)] + ["rf_outside"]
    check("the Roboflow phase is asked for the remaining quota (3 - 1 GitHub = 2), "
          "not max(quota, 8)", rf_asked == [2], rf_asked)
    check("the result holds max_new datasets, the earliest found",
          kept == ["gh_a", "rf_0", "rf_1"], kept)
    check("what was cut is reported", ret.get("cut_over_max_new") == cut,
          ret.get("cut_over_max_new"))
    reg = saved[-1]["datasets"] if saved else {}
    check("no cut dataset is registered (so none can be marked used_for_training)",
          saved and not any(s in reg for s in cut), sorted(reg))
    check("the kept datasets and the earlier registry survive",
          all(s in reg for s in kept + ["pre_existing"]), sorted(reg))
    check("kept datasets are stamped with the current round",
          all(reg.get(s, {}).get("harvest_round") == 4 for s in kept))
    check("the cut downloads are removed from disk",
          not any((data_dir / s).exists() for s in cut if s != "rf_outside"))
    check("a cut download outside the datasets dir is left alone",
          outside_dir.exists())
    check("kept downloads stay on disk", all((data_dir / s).exists() for s in kept))
    check("the cut is logged", any("cut" in m and "rf_6" in m for m in cap.warnings()),
          cap.warnings())
    check("the registry is saved once, after the cut", len(saved) == 1, len(saved))


def test_curator():
    print("dinov2_curator: reference pool guard and slug-stable seeds")
    import numpy as np
    clear_inc()
    cw = TMP / "trusted" / "cottonweeddet12"
    clean = [grid_img(cw / "train" / "images" / ("clean_%d.jpg" % i), 400 + i)
             for i in range(4)]
    (cw / "test" / "images").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HOLDOUT_IMGS[0], cw / "test" / "images" / HOLDOUT_IMGS[0].name)
    grid_img(cw / "valid" / "images" / HOLDOUT_IMGS[2].name, 999)   # holdout stem, other pixels
    shutil.copyfile(HOLDOUT_IMGS[1], cw / "train" / "images" / "renamed_copy.jpg")
    grid_img(cw / "train" / "images" / "near_copy.jpg", 100, paint=(0, 1))
    (cw / "train" / "images" / "broken.jpg").write_bytes(b"garbage")
    shutil.copyfile(DEV_IMGS[0], cw / "train" / "images" / "dev_copy.jpg")
    shutil.copyfile(EXAM_IMG, cw / "train" / "images" / "exam_copy.jpg")
    registry = {"datasets": {"cottonweeddet12": {"local_path": str(cw)}}}
    embedded = []

    def fake_embed(model, proc, paths, batch_size=16):
        embedded.append([str(p) for p in paths])
        return np.random.RandomState(0).rand(len(paths), 8).astype(np.float32)

    def build():
        embedded.clear()
        with holdout_patch(), Patch((CUR, "_load_registry", lambda: registry),
                                    (CUR, "_load_dinov2", lambda: (None, None)),
                                    (CUR, "_embed_images", fake_embed),
                                    (CUR, "SAMPLES_PER_TRUSTED_SLUG", 1000)):
            CUR.build_reference_pool()
        return (sorted(pathlib.Path(p).name for p in embedded[0]),
                json.loads(CUR.REF_META.read_text()))

    names, meta = build()
    want = sorted(p.name for p in clean) + ["dev_copy.jpg", "exam_copy.jpg"]
    check("without the INC index: holdout stems, holdout copies (exact and near) "
          "and unhashable files are left out of the pool", names == want, names)
    ex = meta["slugs"]["cottonweeddet12"].get("excluded", {})
    check("... and the exclusions are recorded per slug",
          ex == {"stem": 2, "near": 2, "unhashable": 1}, ex)
    check("... with the guard's source, marked complete",
          meta.get("guard", {}).get("source") == "mega_trainer_holdout"
          and meta["guard"].get("complete") is True, meta.get("guard"))

    write_nevertrain_index()
    names, meta = build()
    check("with the INC index the dev and exam copies are left out too",
          names == sorted(p.name for p in clean), names)
    check("... and the source is the never-train index",
          meta.get("guard", {}).get("source") == "inc_nevertrain_index")
    clear_inc()

    with Patch((M, "_holdout_image_dirs", lambda: []),
               (CUR, "_load_registry", lambda: registry),
               (CUR, "_load_dinov2", lambda: (None, None))):
        try:
            CUR.build_reference_pool()
            check("the weed pool is not built without a holdout guard", False)
        except RuntimeError as e:
            check("the weed pool is not built without a holdout guard",
                  "reference-pool guard" in str(e), e)
    partial = holdout_variant("curator_partial", keep=(0, 1))
    with holdout_patch(dirs=[partial]), Patch((CUR, "_load_registry", lambda: registry),
                                              (CUR, "_load_dinov2", lambda: (None, None))):
        try:
            CUR.build_reference_pool()
            check("the weed pool is not built from a holdout missing one photograph", False)
        except RuntimeError as e:
            check("the weed pool is not built from a holdout missing one photograph",
                  "expected exactly 3" in str(e), e)

    # another domain's pool does not need the cwd12 holdout; it is built and
    # its guard recorded as incomplete
    fruit = TMP / "trusted" / "fruit_a"
    for i in range(3):
        grid_img(fruit / ("f_%d.jpg" % i), 500 + i)
    fruit_reg = {"datasets": {"fruit_a": {"local_path": str(fruit), "domain": "fruit",
                                          "source": "manual_upload"}}}
    embedded.clear()
    with Patch((M, "_holdout_image_dirs", lambda: []), (CUR, "DINO_DOMAIN", "fruit"),
               (CUR, "_load_registry", lambda: fruit_reg),
               (CUR, "_load_dinov2", lambda: (None, None)),
               (CUR, "_embed_images", fake_embed)):
        CUR.build_reference_pool()
    fmeta = json.loads(CUR.REF_META.read_text())
    check("another domain's pool is built without the holdout, guard marked incomplete",
          len(embedded[0]) == 3 and fmeta["guard"].get("complete") is False, fmeta.get("guard"))

    # score-all only against a pool its meta shows was built under a complete guard
    slugs = ["rf_alpha", "kg_beta", "gh_gamma", "cottonweed_sp8"]
    seeds = []

    def fake_score(slug, info, model, proc, ref, seed=0):
        seeds[-1][slug] = seed
        return {"slug": slug, "status": "ok", "score": 0.5}

    def score_all(order):
        seeds.append({})
        reg = {"datasets": {s: {} for s in order}}
        with Patch((CUR, "_load_registry", lambda: reg),
                   (CUR, "_load_dinov2", lambda: (None, None)),
                   (CUR, "score_one_slug", fake_score)):
            CUR.score_all_slugs()

    good_guard = {"source": "mega_trainer_holdout", "n_hashes": 3,
                  "n_holdout_stems": 3, "complete": True}
    np.save(CUR.REF_POOL, np.ones((1, 8), dtype=np.float32))
    for name, meta in (
            ("a pool whose meta has no guard (every pool before INC 0.3)",
             {"pool_shape": [1, 8], "slugs": {}}),
            ("a pool whose guard was incomplete",
             {"pool_shape": [1, 8], "guard": dict(good_guard, complete=False)}),
            ("a pool the meta does not describe (another shape)",
             {"pool_shape": [4, 8], "guard": good_guard})):
        CUR.REF_META.write_text(json.dumps(meta))
        try:
            score_all(slugs)
            check("score-all refuses %s" % name, False)
        except RuntimeError as e:
            check("score-all refuses %s" % name, "build-reference" in str(e), e)
    CUR.REF_META.unlink()
    try:
        score_all(slugs)
        check("score-all refuses a pool with no meta at all", False)
    except RuntimeError as e:
        check("score-all refuses a pool with no meta at all", "unreadable" in str(e), e)

    CUR.REF_META.write_text(json.dumps({"pool_shape": [1, 8], "guard": good_guard}))
    seeds.clear()
    for order in (slugs, list(reversed(slugs)), slugs[1:] + slugs[:1]):
        score_all(order)
    written = json.loads(CUR.SCORES_PATH.read_text())
    check("every score carries the guard of the pool it was scored against",
          sorted(written) == sorted(slugs)
          and all(r.get("ref_guard") == good_guard for r in written.values()), written)
    check("a slug's sample seed does not depend on registry order",
          seeds[0] == seeds[1] == seeds[2], seeds)
    check("the seed is common.stable_int(slug)",
          all(seeds[0][s] == C.stable_int(s) for s in slugs), seeds[0])
    check("different slugs get different seeds", len(set(seeds[0].values())) == len(slugs))

    # candidate sampling is a function of the slug's contents alone
    d = cw / "train" / "images"
    a = CUR._sample_images(d, 3, seed=C.stable_int("x"))
    b = CUR._sample_images(d, 3, seed=C.stable_int("x"))
    check("a seed gives the same sample twice", a == b and len(a) == 3)

    many = TMP / "sample_src" / "slug"
    for i in range(300):
        f = many / ("sub%d" % (i % 3)) / ("img_%03d.jpg" % i)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    (many / "sub0" / "notes.txt").write_text("not an image")
    seed = C.stable_int("rf_alpha")

    def rel(paths, root):
        return [pathlib.Path(p).relative_to(root).as_posix() for p in paths]

    first = rel(CUR._sample_images(many, 10, seed=seed), many)
    real_walk = os.walk

    def reversed_walk(top, *args, **kw):
        for root, dirs, files in real_walk(top, *args, **kw):
            dirs.reverse()           # in place: the walk descends in reverse too
            yield root, dirs, list(reversed(files))

    with Patch((os, "walk", reversed_walk)):
        backwards = rel(CUR._sample_images(many, 10, seed=seed), many)
    check("same files, same seed, reversed enumeration order -> the same sample",
          first == backwards, (first, backwards))
    moved = TMP / "sample_elsewhere" / "deeper" / "slug"
    shutil.copytree(many, moved)
    check("the same slug under another root (lab server vs cluster) -> the same sample",
          rel(CUR._sample_images(moved, 10, seed=seed), moved) == first)
    listing = sorted(p.relative_to(many).as_posix() for p in many.rglob("*.jpg"))
    want = random.Random(seed).sample(listing, 10)
    check("the sample is drawn from the whole sorted listing", first == want, (first, want))
    check("... including files past the old first-n*20 cut",
          max(listing.index(x) for x in first) >= 200, [listing.index(x) for x in first])


def main():
    try:
        test_fixture_hashes()
        test_merge_guards()
        test_merge_inc_dev()
        test_dino_gate()
        test_optimizer_kwargs()
        test_optimizer_ultralytics()
        test_harvest_cap()
        test_curator()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print()
    print("%d failure(s)" % len(FAILURES))
    if FAILURES:
        print("FAILED: %s" % FAILURES)
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
