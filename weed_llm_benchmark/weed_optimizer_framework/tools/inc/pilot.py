"""INC Step 4, the pilot on known answers, Step 0.6, B0, and the baseline of any
training manifest (Step 1.6, base B): the experiment builders
(docs/INCREMENTAL_PROTOCOL_RUNNER.md, "Pilot build" and "Baseline build").

    python -m weed_optimizer_framework.tools.inc.pilot build    --exp pilot_v1 [--replay-mode {sample,full}]
                                                                [--gate-flips-mode {negative,net}]
                                                                [--testing | --testing-settings JSON]
    python -m weed_optimizer_framework.tools.inc.pilot build-b0 --exp b0_v1   [--testing | --testing-settings JSON]
    python -m weed_optimizer_framework.tools.inc.pilot build-baseline --exp base_b_v1 --manifest PATH
                                                                [--seeds 0,1,2]
                                                                [--testing | --testing-settings JSON]

--testing writes "testing": true into exp.json, --testing-settings the
executor's test-mode object (keys imgsz, batch, device, lock_check; inc/train.py);
either needs INC_SCORER_TESTING=1, and every output of the experiment then says
TESTING. Production builds pass neither.

Every builder writes its manifests under INC_DIR/<exp>/manifests/, a build summary
(INC_DIR/<exp>/build_summary.json) and the experiment definition, and then calls
driver init, which writes exp.json and state.json and submits the first runs.
An experiment is built once: a second build under the same name refuses.

build (a 'chain' experiment):
  * P0: train_core's sessions sorted by size (descending, then name), taken
    until P0 holds >= 50% of train_core's images.
  * The other sessions are packed into 6 bins, largest session first into the
    currently smallest bin (ties: lowest index); the bins are then sorted by
    their lexicographically smallest session. Bin 3 is the Bswap source, the
    others in order are I1..I5. A bin outside +-35% of the mean is flagged.
    Sessions are never split.
  * Bswap: the source bin with 40% of its boxes (round half up) given another
    species, drawn uniformly from the other 11, with
    numpy.random.default_rng(stable_int(exp + '/Bswap')): first the boxes (a
    sample without replacement over the bin's boxes in key, line order), then
    one draw per chosen box in that order. The relabelled copies are written
    to INC_DIR/<exp>/labels/Bswap/ (untouched lines byte for byte, a changed
    line with only its class token replaced); every change is listed in
    labels/Bswap_changes.json. Keys are 'Bswap__<train_core key>'.
  * Breal: every labelled image of REPO/datasets/<slug>/{images,labels} for
    the two BREAL_SLUGS (<slug>/<split>/{images,labels} when there is no
    top-level images/), read with the registry's class_names order
    (results/framework/dataset_registry.json): a name species_of() knows
    becomes its cwd12 id, every other name OtherPlant (12). An image within 6
    dHash bits of any evaluation image (NeverTrainGuard.check), or that cannot
    be hashed, is dropped and counted; so is an image within 6 bits of any
    train_core image (the protocol: "near-copies of any split image are
    removed"; the runner doc names only evaluation images), since such a copy
    would sit in cand and union sets next to its original in P0 or an I_k.
    N images are then sampled with default_rng(stable_int(exp + '/Breal')),
    N = the median size of I1..I5. That the label ids index class_names is an
    assumption, recorded as unverified.
  * Sequence [I1, I2, Bswap, I3, Breal, I4, I5]; clean = I1..I5. Recipes full,
    freeze (backbone layers 0-10) and lora (r 16, alpha 32); base and truth
    runs use the protocol's cold recipe.
  * --replay-mode (default sample) goes into exp.json and the build summary
    as "replay_mode": 'sample' is pilot_v1's replay (cand on D_k + R1, null
    on R1 + R2, |R1| = |R2| = |D_k|); 'full' is full rehearsal (cand on the
    chain's whole accepted pool + D_k, null on the pool). Everything else in
    the definition is the same for both.
  * --gate-flips-mode (default negative) goes into exp.json and the build
    summary as the gate block, "gate": {"flips_mode": ...}: 'negative' is
    protocol v1's flips guard (the driver decides exactly as without the
    block), 'net' protocol v2's (negative - positive flips;
    docs/INCREMENTAL_PROTOCOL.md, Gate, "Protocol v2 (net flips)").
    build-b0 and build-baseline make no gate decision and refuse the flag.
  * Attribution: the driver runs protocol attribution steps 1-3 only; step 4
    (BioCLIP-2 label audit) and step 5 (leave-one-source-out, which Breal's
    two sources would call for at a REJECT) are out of scope for the pilot,
    recorded as such in exp.json ("attribution_scope") and in every gate entry.

A production build (neither --testing flag) needs the splits locked
(LOCK.json) and train_core verifying against it; a testing build only warns.

build-b0 (a 'baseline' experiment): train_core (copied into the experiment,
same bytes) cold, 3 seeds.

build-baseline (a 'baseline' experiment, exactly as build-b0 but on any
training manifest; base B is INC_DIR/step1/base_B.jsonl -> base_b_v1): the
manifest, copied into the experiment as manifests/<its sanitised stem>.jsonl
(same bytes), cold with the protocol's recipe, one base run per --seeds seed
(default 0,1,2), then final runs on every exam, test included, after the base
runs. Before anything is written the manifest must pass
check_training_manifest, which fails closed:
  * inc/train.py's own manifest check, the one every training run of it
    repeats: every row has the manifest keys, keys are unique and path-safe,
    every label line is a box of the INC class space (ids 0..12, coordinates
    in [0, 1]), label and image bytes hash as the manifest says, and it is
    not an evaluation split's manifest;
  * the never-train guard over the dHashes that check computed: an image
    within 6 bits of a dev / test / exam image, or one that cannot be hashed,
    refuses the build.
A production build also needs LOCK.json and the never-train index LOCK.json
recorded (never_train_status); a testing build only warns. When the
manifest is the base_B.jsonl an inc.select build wrote (select_summary.json
next to it names its sha256), that build must have read the train_core
LOCK.json records and the never-train index in place now
(select_provenance): a base selected under an earlier lock is refused in
production, warned about in testing. The summary records the manifest's
sha256, images, sessions, boxes per class, images and boxes per source, the
duplicate counts, the guard and that select record (select_build, null for
any other manifest).

base_b_v1 and the real loop (inc/realloop.py) train the same base: the
loop's base arm is the same three cold runs (same manifest bytes, recipe,
seeds and init), and its final runs read base B's test again. The driver
cannot take the base weights from another experiment, so running both costs
three more cold runs and a second read of B's test; docs/INCREMENTAL_PROTOCOL.md
(Step 1.6) says when base_b_v1 is built.

Recipes. The protocol fixes: cold = 100 epochs, SGD, lr0 0.01, cosine, lrf
0.01, warmup 3 epochs; incremental = 30 epochs, SGD, lr0 0.002 (LoRA 0.01),
warmup 1 epoch, cosine to lrf 0.01; imgsz 640, batch 32, deterministic. Keys it
does not fix keep Ultralytics' defaults (momentum 0.937, weight_decay 0.0005,
close_mosaic 10, and warmup_bias_lr 0.1 for the cold recipe); the incremental
warmup_bias_lr equals lr0, as the runner doc says.

Warmup as run. Ultralytics (BaseTrainer._do_train, 8.4.x) warms up for
nw = max(round(warmup_epochs * nb), 100) iterations, nb = ceil(images /
batch): never fewer than 100 iterations. In replay mode 'sample' the cand and
null sets hold 2|D_k| images, a few hundred on the real train_core (476-548
in pilot_v1, 15-18 iterations per epoch at batch 32), so "warmup 1 epoch" runs
as 5.6-6.7 of the 30 epochs, the same in both arms of a step; a cold run on P0
or T_k keeps its 3 epochs. In replay mode 'full' cand holds the accepted pool
+ D_k and null the pool (P0 alone held 1,540 images in pilot_v1, 49
iterations per epoch), so the floor stretches the warmup to about 1-2 epochs;
the sizes depend on which earlier increments the chain accepted, so each is
recorded at both ends (pool_min = P0, pool_max = P0 + every earlier
increment), and the driver records the actual sizes per step. The build
records the effective warmup of every run type (build_summary.json "warmup",
exp.json "effective_warmup"); the recipe itself is unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

from . import common as C
from . import driver as D
from ..cwd12_species import species_of
from ..near_dup import HOLDOUT_NEAR_DUP_BITS, NearHashIndex
from .scorer import TEST_ENV
from .splits import sanitise

P0_SHARE = 0.5
N_BINS = 6
BSWAP_BIN = 3
BIN_TOLERANCE = 0.35
BSWAP_SHARE = 0.4
N_SPECIES = C.OTHER_PLANT            # 12: the ids a Bswap box may be reassigned among
SEQUENCE = ("I1", "I2", "Bswap", "I3", "Breal", "I4", "I5")
CLEAN = ("I1", "I2", "I3", "I4", "I5")
BREAL_SLUGS = ("project_agml__weed_crop_detection",
               "project_agml__imageweeds_aerial_weed_detection")
BUILD_SUMMARY = "build_summary.json"
CLASS_ORDER_NOTE = ("label id i is read as the registry's class_names[i]; the order is not "
                    "checked against the source's own metadata (unverified)")
WARMUP_FLOOR = 100                   # Ultralytics' minimum warmup iterations
ATTRIBUTION_SCOPE = {
    "run": ["1 recipe vs data", "2 classification vs localisation", "3 per-species deltas"],
    "not_run": {"4": "label audit (BioCLIP-2 re-reading D_k's boxes): out of scope for the pilot",
                "5": "leave-one-source-out cand runs for a multi-source increment (Breal mixes two "
                     "sources): out of scope for the pilot"},
}


class PilotError(RuntimeError):
    """A condition under which the experiment must not be built."""


def log(msg):
    print("[inc.pilot] %s" % msg, flush=True)


# --------------------------------------------------------------- recipes
def cold_recipe():
    """The protocol's cold ('union') run; seed is set per run by the driver."""
    return {"trainer": "full", "epochs": 100, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
            "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3, "warmup_bias_lr": 0.1,
            "cos_lr": True, "freeze": None, "lora": None, "imgsz": 640, "batch": 32,
            "cache": "ram", "workers": 5, "close_mosaic": 10, "deterministic": True}


def inc_recipes():
    """The pilot's three incremental recipes (runner doc, "Recipes")."""
    full = dict(cold_recipe(), epochs=30, lr0=0.002, warmup_epochs=1, warmup_bias_lr=0.002)
    freeze = dict(full, trainer="freeze", freeze=11)
    lora = dict(full, trainer="lora", lr0=0.01, warmup_bias_lr=0.01,
                lora={"rank": 16, "alpha": 32})
    return {"full": full, "freeze": freeze, "lora": lora}


def effective_warmup(recipe, n_images):
    """The warmup Ultralytics actually runs for a recipe on n_images:
    nb = ceil(n / batch) iterations per epoch, nw = max(round(warmup_epochs *
    nb), 100) warmup iterations (none when warmup_epochs is 0)."""
    nb = max(1, int(math.ceil(n_images / float(recipe["batch"]))))
    total = nb * int(recipe["epochs"])
    we = recipe["warmup_epochs"]
    nw = max(int(round(we * nb)), WARMUP_FLOOR) if we > 0 else 0
    return {"images": int(n_images), "iterations_per_epoch": nb, "warmup_iterations": nw,
            "warmup_epochs_nominal": we, "warmup_epochs_effective": round(min(nw, total) / float(nb), 2),
            "epochs": int(recipe["epochs"]), "covers_whole_run": nw >= total}


def warmup_table(recipes, cold, p0_n, steps, replay_mode=D.DEFAULT_REPLAY_MODE):
    """Effective warmup per run type: steps = [(name, |D_k|, clean)] in order.
    Replay mode 'sample': cand and null hold 2|D_k| images (the table pilot_v1
    recorded). 'full': cand holds the accepted pool + D_k, null the pool; each
    step records both at pool_min (P0: nothing accepted before it) and pool_max
    (P0 + every earlier increment), for incremental[recipe][step][cand|null]."""
    out = {"note": "Ultralytics warms up for max(round(warmup_epochs * iterations per epoch), %d) "
                   "iterations; cand and null sets hold 2|D_k| images" % WARMUP_FLOOR,
           "cold": {"base": effective_warmup(cold, p0_n), "truth": {}}, "incremental": {}}
    t = p0_n
    for i, (name, n, clean) in enumerate(steps, 1):
        out["cold"]["truth"]["s%02d_%s" % (i, name)] = effective_warmup(cold, t + n)
        if clean:
            t += n
    if replay_mode == "full":
        out["note"] = ("Ultralytics warms up for max(round(warmup_epochs * iterations per epoch), %d) "
                       "iterations; replay mode full: cand holds the accepted pool + D_k and null the pool, "
                       "recorded at pool_min (P0) and pool_max (P0 + every earlier increment); the driver "
                       "records each step's actual sizes" % WARMUP_FLOOR)
        out["replay_mode"] = "full"
        for r, rec in recipes.items():
            per, before = {}, 0
            for i, (name, n, _) in enumerate(steps, 1):
                per["s%02d_%s" % (i, name)] = {
                    "cand": {"pool_min": effective_warmup(rec, p0_n + n),
                             "pool_max": effective_warmup(rec, p0_n + before + n)},
                    "null": {"pool_min": effective_warmup(rec, p0_n),
                             "pool_max": effective_warmup(rec, p0_n + before)}}
                before += n
            out["incremental"][r] = per
        eff = [v["warmup_epochs_effective"] for rv in out["incremental"].values() for sv in rv.values()
               for arm in sv.values() for v in arm.values()]
    else:
        for r, rec in recipes.items():
            out["incremental"][r] = {"s%02d_%s" % (i, name): effective_warmup(rec, 2 * n)
                                     for i, (name, n, _) in enumerate(steps, 1)}
        eff = [v["warmup_epochs_effective"] for rv in out["incremental"].values() for v in rv.values()]
    out["incremental_effective_epochs"] = {"min": min(eff), "max": max(eff)} if eff else None
    return out


# ---------------------------------------------------------------- inputs
def _check_testing(testing):
    """exp.json's 'testing' value: False, True, or the executor's settings object."""
    if isinstance(testing, dict):
        unknown = sorted(set(testing) - set(D.TESTING_KEYS))
        if unknown:
            raise PilotError("testing settings %s are not among %s" % (unknown, list(D.TESTING_KEYS)))
    elif not isinstance(testing, bool):
        raise PilotError("testing must be true / false or a settings object")
    if testing and os.environ.get(TEST_ENV) != "1":
        raise PilotError("--testing defines an experiment gated on test-mode scores; it needs "
                         "%s=1" % TEST_ENV)
    return testing if testing else False


def _check_new(paths):
    """Refuse once the experiment is defined (exp.json or state.json exist). The
    outputs of a build that stopped before driver init (manifests/, labels/,
    the summary) are not used by any run yet; they are removed and rebuilt."""
    if paths.exp_json.exists() or paths.state.exists():
        raise PilotError("experiment %s is already built at %s; an experiment is built once "
                         "(use a new --exp name)" % (paths.exp, paths.root))
    stale = [p for p in (paths.manifests, paths.root / "labels") if p.exists()]
    if stale or (paths.root / BUILD_SUMMARY).exists():
        log("WARNING: %s holds the outputs of a build that never reached driver init; "
            "rebuilding them" % paths.root)
        for p in stale:
            shutil.rmtree(p)
        if (paths.root / BUILD_SUMMARY).exists():
            (paths.root / BUILD_SUMMARY).unlink()


def load_train_core(testing=False):
    """(rows, sha256, locked, path) of the train_core manifest, checked against
    LOCK.json. A production build refuses unlocked splits; a testing build
    only warns."""
    path = C.manifest_path("train_core")
    if not path.is_file():
        raise PilotError("no train_core manifest at %s (run inc.splits build)" % path)
    locked = C.LOCK_PATH.exists()
    if locked:
        try:
            C.verify_manifest_against_lock("train_core")
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            raise PilotError("train_core does not verify against LOCK.json: %s" % e)
    elif not testing:
        raise PilotError("%s does not exist: a production experiment is built only on locked splits "
                         "(run inc.splits lock first)" % C.LOCK_PATH)
    else:
        log("WARNING: %s does not exist; train_core is used unlocked (testing build)" % C.LOCK_PATH)
    rows = sorted(C.read_manifest(path), key=lambda r: r["key"])
    if not rows:
        raise PilotError("train_core is empty")
    return rows, C.sha256_file(path), locked, path


def sessions_of(rows):
    by = collections.defaultdict(list)
    for r in rows:
        s = r.get("session")
        if not s:
            raise PilotError("train_core row %s has no session; the pilot splits by whole "
                             "sessions" % r["key"])
        by[s].append(r)
    return by


# -------------------------------------------------------------- P0, bins
def choose_p0(sizes, share=P0_SHARE):
    """Sessions by size (descending, then name), taken until P0 holds >= share
    of all images."""
    total = sum(sizes.values())
    p0, n = [], 0
    for s in sorted(sizes, key=lambda s: (-sizes[s], s)):
        if n >= share * total:
            break
        p0.append(s)
        n += sizes[s]
    return p0


def pack_bins(sizes, sessions, n_bins=N_BINS):
    """Greedy packing: largest session first (ties by name) into the currently
    smallest bin (ties: lowest index); then bins sorted by their smallest
    session name. Each bin's sessions are returned sorted."""
    order = sorted(sessions, key=lambda s: (-sizes[s], s))
    if len(order) < n_bins:
        raise PilotError("%d sessions left after P0 cannot fill %d bins" % (len(order), n_bins))
    bins, tot = [[] for _ in range(n_bins)], [0] * n_bins
    for s in order:
        i = min(range(n_bins), key=lambda j: (tot[j], j))
        bins[i].append(s)
        tot[i] += sizes[s]
    return sorted((sorted(b) for b in bins), key=lambda b: b[0])


def bin_report(bins, sizes, roles):
    counts = [sum(sizes[s] for s in b) for b in bins]
    mean = sum(counts) / float(len(counts))
    out = []
    for i, (b, n) in enumerate(zip(bins, counts)):
        dev = (n - mean) / mean if mean else 0.0
        out.append({"index": i, "role": roles[i], "sessions": b, "images": n,
                    "deviation": round(dev, 4), "flagged": abs(dev) > BIN_TOLERANCE + 1e-12})
    return out, mean


# ----------------------------------------------------------------- Bswap
def make_bswap(exp, rows, label_dir):
    """(Bswap rows, changes): see the module docstring."""
    import numpy as np
    rows = sorted(rows, key=lambda r: r["key"])
    texts, boxes = [], []
    for i, r in enumerate(rows):
        with open(r["label"]) as fh:
            lines = fh.read().splitlines()
        texts.append(lines)
        for j, ln in enumerate(lines):
            t = ln.split()
            if not t:
                continue
            if len(t) != 5:
                raise PilotError("%s:%d: %d columns, expected 5" % (r["label"], j + 1, len(t)))
            c = float(t[0])
            if c != int(c) or not 0 <= int(c) < N_SPECIES:
                raise PilotError("%s:%d: class %s; a Bswap box must be one of the %d species"
                                 % (r["label"], j + 1, t[0], N_SPECIES))
            boxes.append((i, j, int(c)))
    if not boxes:
        raise PilotError("the Bswap source bin holds no boxes")
    n_change = int(math.floor(BSWAP_SHARE * len(boxes) + 0.5))
    rng = np.random.default_rng(C.stable_int(exp + "/Bswap"))
    chosen = sorted(int(x) for x in rng.choice(len(boxes), size=n_change, replace=False))
    changes = []
    for b in chosen:
        i, j, c = boxes[b]
        others = [x for x in range(N_SPECIES) if x != c]
        new = others[int(rng.integers(0, len(others)))]
        t = texts[i][j].split()
        t[0] = str(new)
        texts[i][j] = " ".join(t)
        changes.append({"key": "Bswap__" + rows[i]["key"], "source_key": rows[i]["key"],
                        "line": j, "box": b, "from": c, "to": new,
                        "from_name": C.CLASS_NAMES[c], "to_name": C.CLASS_NAMES[new]})
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for r, lines in zip(rows, texts):
        key = "Bswap__" + r["key"]
        path = label_dir / (key + ".txt")
        with open(path, "w") as fh:
            fh.write("".join(ln + "\n" for ln in lines))
        out.append({"image": r["image"], "label": str(path), "sha256": r["sha256"],
                    "label_sha256": C.sha256_file(path), "source": "Bswap",
                    "session": r["session"], "key": key})
    return out, {"boxes": len(boxes), "changed": n_change, "share": n_change / float(len(boxes)),
                 "seed_text": exp + "/Bswap", "changes": changes}


# ----------------------------------------------------------------- Breal
def registry_path():
    return C.REPO / "results" / "framework" / "dataset_registry.json"


def names_list(names):
    """A registry class_names value (list, or {id: name}) as a list by id."""
    if not names:
        return []
    if isinstance(names, dict):
        try:
            ids = {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            raise PilotError("class_names dict with non-integer keys: %s" % list(names)[:5])
        out = [""] * (max(ids) + 1)
        for i, v in ids.items():
            if i < 0:
                raise PilotError("negative class id %d in class_names" % i)
            out[i] = v
        return out
    return [str(n) for n in names]


def class_join(names):
    """{source id: INC id}: species_of(name) -> its cwd12 id, else OtherPlant."""
    out = {}
    for i, n in enumerate(names):
        sp = species_of(n) if n else None
        out[i] = C.CLASS_NAMES.index(sp) if sp else C.OTHER_PLANT
    return out


def _image_dirs(root):
    """[(split, images dir, labels dir)]: root/images, else root/<split>/images."""
    if (root / "images").is_dir():
        return [("", root / "images", root / "labels")]
    out = []
    for name in sorted(os.listdir(root)):
        if (root / name / "images").is_dir():
            out.append((name, root / name / "images", root / name / "labels"))
    return out


def convert_label(path, join):
    """(INC boxes, None, ids outside class_names) or (None, reason, 0)."""
    boxes, outside = [], 0
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if not t:
                continue
            if len(t) != 5:
                return None, "columns", 0
            try:
                c = float(t[0])
                v = [float(x) for x in t[1:]]
            except ValueError:
                return None, "not_numeric", 0
            if c != int(c) or c < 0:
                return None, "bad_class", 0
            if min(v) < 0 or max(v) > 1:
                return None, "out_of_range", 0
            c = int(c)
            if c not in join:
                outside += 1
            boxes.append((join.get(c, C.OTHER_PLANT),) + tuple(v))
    return boxes, None, outside


def breal_candidates(slugs=BREAL_SLUGS, registry=None):
    """Every usable labelled image of the Breal sources, and per-slug counts."""
    from .. import mega_trainer as MT
    if registry is None:
        with open(registry_path()) as fh:
            registry = json.load(fh)
    entries = registry.get("datasets", registry)
    cands, info = [], {}
    for slug in slugs:
        if slug in MT.NEVER_TRAIN_SLUGS:
            raise PilotError("%s is a never-train slug" % slug)
        entry = entries.get(slug)
        if not isinstance(entry, dict):
            raise PilotError("%s is not in the registry %s" % (slug, registry_path()))
        names = names_list(entry.get("class_names"))
        if not names:
            raise PilotError("the registry names no classes for %s; its boxes cannot be joined" % slug)
        join = class_join(names)
        root = C.REPO / "datasets" / slug
        if not root.is_dir() and entry.get("local_path"):
            lp = Path(entry["local_path"])
            root = lp if lp.is_absolute() else C.REPO / lp
        if not root.is_dir():
            raise PilotError("%s: no dataset dir at %s" % (slug, root))
        dirs = _image_dirs(root)
        if not dirs:
            raise PilotError("%s: no images/ dir under %s" % (slug, root))
        drops = collections.Counter()
        found, outside, n = 0, 0, 0
        short = sanitise(slug[len("project_agml__"):] if slug.startswith("project_agml__") else slug)
        for split, img_dir, lbl_dir in dirs:
            for name in sorted(os.listdir(img_dir)):
                if name.startswith(".") or os.path.splitext(name)[1].lower() not in C.IMG_EXTS:
                    continue
                found += 1
                stem = os.path.splitext(name)[0]
                lbl = lbl_dir / (stem + ".txt")
                if not lbl.is_file():
                    drops["no_label"] += 1
                    continue
                boxes, why, out = convert_label(lbl, join)
                if boxes is None:
                    drops["bad_label_" + why] += 1
                    continue
                outside += out
                n += 1
                rel = "%s/%s" % (split, name) if split else name
                cands.append({"slug": slug, "image": str(img_dir / name), "boxes": boxes,
                              "rel": rel, "base_key": "Breal__%s__%s" % (short, sanitise(stem))})
        info[slug] = {"root": str(root), "layout": [s or "." for s, _, _ in dirs],
                      "class_names": names,
                      "join": {names[i]: C.CLASS_NAMES[j] for i, j in join.items()},
                      "images_found": found, "usable": n, "dropped": dict(drops),
                      "boxes_with_ids_outside_class_names": outside}
    counts = collections.Counter(c["base_key"] for c in cands)
    for c in cands:
        c["key"] = (c["base_key"] if counts[c["base_key"]] == 1
                    else "%s__%s" % (c["base_key"], C.sha256_text(c["slug"] + "/" + c["rel"])[:8]))
    if len({c["key"] for c in cands}) != len(cands):
        raise PilotError("Breal keys collide after disambiguation")
    return cands, info


def train_core_index(rows, hash_fn=None):
    """(NearHashIndex of train_core's dHashes at HOLDOUT_NEAR_DUP_BITS, [unhashable images])."""
    hash_fn = hash_fn or C.dhash
    index, unhashable = NearHashIndex(), []
    for r in rows:
        h = hash_fn(r["image"])
        if h is None:
            unhashable.append(r["image"])
            continue
        index.add(int(h), r["key"], max_bits=HOLDOUT_NEAR_DUP_BITS)
    return index, unhashable


def make_breal(exp, n, label_dir, train_core_rows, guard=None, hash_fn=None, registry=None,
               slugs=BREAL_SLUGS):
    """(Breal rows, summary): the guard drops near-copies of evaluation images
    and unhashable images, then near-copies of train_core images are dropped,
    then n images are sampled."""
    import numpy as np
    hash_fn = hash_fn or C.dhash
    cands, info = breal_candidates(slugs, registry)
    hashes = {c["image"]: hash_fn(c["image"]) for c in cands}
    guard = guard if guard is not None else C.NeverTrainGuard.load()
    hits, unhashable = guard.check([c["image"] for c in cands], hash_fn=hashes.get)
    hit_images = {h[0] for h in hits}
    bad = hit_images | set(unhashable)
    core_index, core_unhashable = train_core_index(train_core_rows, hash_fn)
    if core_unhashable:
        log("WARNING: %d train_core image(s) cannot be hashed; Breal is not checked against them: %s"
            % (len(core_unhashable), core_unhashable[:3]))
    near_core = []
    for c in cands:
        if c["image"] in bad:
            continue
        m = core_index.find(int(hashes[c["image"]]))
        if m is not None:
            near_core.append((c["image"], m[0], m[1]))
    near_core_images = {x[0] for x in near_core}
    per_slug = collections.Counter()
    for c in cands:
        if c["image"] in hit_images:
            per_slug[(c["slug"], "near_eval")] += 1
        elif c["image"] in bad:
            per_slug[(c["slug"], "unhashable")] += 1
        elif c["image"] in near_core_images:
            per_slug[(c["slug"], "near_train_core")] += 1
    for (slug, why), k in per_slug.items():
        info[slug]["dropped"][why] = k
    bad |= near_core_images
    pool = sorted((c for c in cands if c["image"] not in bad), key=lambda c: c["key"])
    if len(pool) < n:
        raise PilotError("Breal has %d usable images after the guard, fewer than the %d a pilot "
                         "increment holds" % (len(pool), n))
    rng = np.random.default_rng(C.stable_int(exp + "/Breal"))
    picked = [pool[i] for i in sorted(int(x) for x in rng.choice(len(pool), size=n, replace=False))]
    label_dir = Path(label_dir)
    rows = []
    for c in picked:
        path = label_dir / (c["key"] + ".txt")
        C.write_yolo(path, c["boxes"])
        rows.append({"image": c["image"], "label": str(path), "sha256": C.sha256_file(c["image"]),
                     "label_sha256": C.sha256_file(path), "source": c["slug"], "session": "",
                     "key": c["key"]})
    summary = {"slugs": info, "n": n, "seed_text": exp + "/Breal", "candidates": len(cands),
               "after_guard": len(pool),
               "dropped_near_eval": len(hit_images), "dropped_unhashable": len(set(unhashable)),
               "dropped_near_train_core": len(near_core_images),
               "train_core_unhashable": len(core_unhashable),
               "near_eval_examples": [{"image": h[0], "split": h[1], "eval_image": h[2], "bits": h[3]}
                                      for h in hits[:20]],
               "near_train_core_examples": [{"image": x[0], "train_core_key": x[1], "bits": x[2]}
                                            for x in near_core[:20]],
               "sampled_per_slug": dict(collections.Counter(c["slug"] for c in picked)),
               "class_order_verified": False, "class_order_assumption": CLASS_ORDER_NOTE}
    return rows, summary


# ----------------------------------------------------------------- build
def _entry(name, path, rows, sha, clean=None, **extra):
    e = {"name": name, "manifest": str(path), "manifest_sha256": sha, "n_images": len(rows)}
    if clean is not None:
        e["clean"] = clean
    e.update(extra)
    return e


def _describe(rows):
    counts = C.class_counts(rows)
    return {"images": len(rows), "boxes_total": sum(counts.values()), "boxes": counts,
            "sources": dict(collections.Counter(r["source"] for r in rows))}


def gate_block(flips_mode):
    """exp.json's gate block for a builder's --gate-flips-mode."""
    if flips_mode not in D.FLIPS_MODES:
        raise PilotError("gate flips mode %r not in %s" % (flips_mode, D.FLIPS_MODES))
    return {"flips_mode": flips_mode}


def build_pilot(exp="pilot_v1", testing=False, backend=None, guard=None, hash_fn=None,
                registry=None, init=True, quiet=False, replay_mode=D.DEFAULT_REPLAY_MODE,
                gate_flips_mode=D.DEFAULT_FLIPS_MODE):
    """Build the pilot's manifests and definition, then driver init. Returns
    (summary, definition, init result or None)."""
    if replay_mode not in D.REPLAY_MODES:
        raise PilotError("replay mode %r not in %s" % (replay_mode, D.REPLAY_MODES))
    gate = gate_block(gate_flips_mode)
    testing = _check_testing(testing)
    paths = D.Paths(exp)
    _check_new(paths)
    rows, core_sha, locked, core_path = load_train_core(testing=bool(testing))
    by = sessions_of(rows)
    sizes = {s: len(v) for s, v in by.items()}
    p0 = choose_p0(sizes)
    bins = pack_bins(sizes, [s for s in sizes if s not in set(p0)])
    clean_names = iter(CLEAN)
    roles = ["Bswap_source" if i == BSWAP_BIN else next(clean_names) for i in range(N_BINS)]
    bins_info, mean = bin_report(bins, sizes, roles)
    for b in bins_info:
        if b["flagged"]:
            log("WARNING: bin %d (%s) holds %d images, %+.0f%% from the mean %.1f (tolerance "
                "+-%.0f%%)" % (b["index"], b["role"], b["images"], 100 * b["deviation"], mean,
                               100 * BIN_TOLERANCE))

    part = {"P0": [r for s in p0 for r in by[s]]}
    for b, role in zip(bins, roles):
        part[role] = [r for s in b for r in by[s]]
    mdir = paths.manifests
    manifests = {}
    for name in ("P0",) + CLEAN:
        path = mdir / ("%s.jsonl" % name)
        manifests[name] = (path, C.write_manifest(path, part[name]), part[name])

    bswap_rows, bswap = make_bswap(exp, part["Bswap_source"], paths.root / "labels" / "Bswap")
    changes_path = paths.root / "labels" / "Bswap_changes.json"
    D._write_json(changes_path, {"exp": exp, "testing": bool(testing), **bswap})
    path = mdir / "Bswap.jsonl"
    manifests["Bswap"] = (path, C.write_manifest(path, bswap_rows), bswap_rows)

    n_breal = int(statistics.median(len(part[n]) for n in CLEAN))
    breal_rows, breal = make_breal(exp, n_breal, paths.root / "labels" / "Breal", rows, guard=guard,
                                   hash_fn=hash_fn, registry=registry)
    path = mdir / "Breal.jsonl"
    manifests["Breal"] = (path, C.write_manifest(path, breal_rows), breal_rows)

    extra = {"Bswap": {"planted": "40% of boxes relabelled to another species",
                       "source_sessions": bins[BSWAP_BIN]},
             "Breal": {"planted": "harvested sources with measured label precision < 0.90",
                       "class_order_verified": False}}
    p_path, p_sha, p_rows = manifests["P0"]
    warmup = warmup_table(inc_recipes(), cold_recipe(), len(p_rows),
                          [(n, len(manifests[n][2]), n in CLEAN) for n in SEQUENCE], replay_mode)
    defn = {
        "exp": exp, "type": "chain", "builder": "inc.pilot build", "testing": testing,
        "replay_mode": replay_mode, "gate": gate,
        "seeds": list(D.SEEDS), "init_weights": D.COLD_INIT, "decision_exam": D.DECISION_EXAM,
        "final_exams": list(D.FINAL_EXAMS),
        "base": _entry("P0", p_path, p_rows, p_sha, recipe=cold_recipe(), sessions=p0),
        "steps": [_entry(n, manifests[n][0], manifests[n][2], manifests[n][1], clean=n in CLEAN,
                         **(extra[n] if n in extra else {"sessions": bins[roles.index(n)]}))
                  for n in SEQUENCE],
        "recipes": inc_recipes(),
        "truth": True, "truth_recipe": cold_recipe(),
        "train_core": {"manifest": str(core_path), "sha256": core_sha, "locked": locked},
        "effective_warmup": warmup, "attribution_scope": ATTRIBUTION_SCOPE,
    }
    summary = {
        "exp": exp, "testing": bool(testing), "built_utc": D._utc(), "replay_mode": replay_mode,
        "gate": dict(gate),
        "train_core": {"manifest": str(core_path), "sha256": core_sha, "locked": locked,
                       "images": len(rows), "sessions": len(sizes)},
        "p0": dict(_describe(part["P0"]), sessions=p0, share=len(part["P0"]) / float(len(rows))),
        "bins": bins_info, "bin_mean": mean, "bin_tolerance": BIN_TOLERANCE,
        "bins_flagged": [b["index"] for b in bins_info if b["flagged"]],
        "increments": {n: dict(_describe(manifests[n][2]), manifest=str(manifests[n][0]),
                               manifest_sha256=manifests[n][1], clean=n in CLEAN)
                       for n in SEQUENCE},
        "bswap": {"source_bin": BSWAP_BIN, "source_sessions": bins[BSWAP_BIN],
                  "boxes": bswap["boxes"], "changed": bswap["changed"], "share": bswap["share"],
                  "seed_text": bswap["seed_text"], "changes_file": str(changes_path)},
        "breal": breal,
        "sequence": list(SEQUENCE), "clean": list(CLEAN),
        "recipes": inc_recipes(), "cold_recipe": cold_recipe(),
        "warmup": warmup, "attribution_scope": ATTRIBUTION_SCOPE,
    }
    D._write_json(paths.root / BUILD_SUMMARY, summary)
    log("%s: P0 %d images (%.1f%%, %d sessions); increments %s; Bswap %d/%d boxes changed; "
        "Breal %d sampled of %d after the guards (%d near-eval, %d near-train_core dropped); "
        "replay mode %s; gate flips %s; incremental warmup runs %s-%s epochs"
        % (exp, len(part["P0"]), 100 * summary["p0"]["share"], len(p0),
           {n: len(manifests[n][2]) for n in SEQUENCE}, bswap["changed"], bswap["boxes"],
           n_breal, breal["after_guard"], breal["dropped_near_eval"], breal["dropped_near_train_core"],
           replay_mode, gate["flips_mode"], warmup["incremental_effective_epochs"]["min"],
           warmup["incremental_effective_epochs"]["max"]))
    result = D.Driver(exp, backend=backend, quiet=quiet).init(defn) if init else None
    return summary, defn, result


def build_b0(exp="b0_v1", testing=False, backend=None, init=True, quiet=False):
    """B0 (Step 0.6): a 'baseline' experiment, train_core cold, 3 seeds."""
    testing = _check_testing(testing)
    paths = D.Paths(exp)
    _check_new(paths)
    rows, sha, locked, core_path = load_train_core(testing=bool(testing))
    dst = paths.manifests / "train_core.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(core_path, dst)
    if C.sha256_file(dst) != sha:
        raise PilotError("copy of %s does not hash like the original" % core_path)
    defn = {"exp": exp, "type": "baseline", "builder": "inc.pilot build-b0", "testing": testing,
            "seeds": list(D.SEEDS), "init_weights": D.COLD_INIT, "decision_exam": D.DECISION_EXAM,
            "final_exams": list(D.FINAL_EXAMS),
            "base": _entry("train_core", dst, rows, sha, recipe=cold_recipe(),
                           source_manifest=str(core_path), source_locked=locked)}
    summary = {"exp": exp, "testing": bool(testing), "built_utc": D._utc(),
               "train_core": dict(_describe(rows), manifest=str(core_path), sha256=sha, locked=locked,
                                  sessions=len({r.get("session") for r in rows})),
               "cold_recipe": cold_recipe(), "warmup": {"base": effective_warmup(cold_recipe(), len(rows))}}
    D._write_json(paths.root / BUILD_SUMMARY, summary)
    log("%s: B0 on train_core, %d images, seeds %s" % (exp, len(rows), list(D.SEEDS)))
    result = D.Driver(exp, backend=backend, quiet=quiet).init(defn) if init else None
    return summary, defn, result


# ------------------------------------------------- any training manifest
def check_seeds(seeds):
    """Seeds as a list of distinct non-negative ints, from a list or from text
    such as '0,1,2'."""
    if isinstance(seeds, str):
        try:
            seeds = [int(s) for s in seeds.split(",") if s.strip()]
        except ValueError:
            raise PilotError("--seeds must be comma-separated integers, got %r" % seeds)
    seeds = list(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)):
        raise PilotError("seeds must be distinct non-negative integers, got %r" % (seeds,))
    return seeds


def never_train_status(testing=False):
    """The never-train index every training set is checked against, and
    whether LOCK.json vouches for it. A production build refuses a missing
    LOCK.json, one that records no never-train sha256, or an index that is
    not the one it recorded; a testing build only warns. A missing index
    refuses either build."""
    idx = Path(C.NEVER_TRAIN_INDEX)
    if not idx.is_file():
        raise PilotError("no never-train index at %s (run inc.splits build)" % idx)
    sha = C.sha256_file(idx)
    out = {"index": str(idx), "sha256": sha, "locked": C.LOCK_PATH.exists(), "matches_lock": None}
    problem = None
    if out["locked"]:
        try:
            want = C.read_lock().get("nevertrain_sha256")
        except (OSError, ValueError) as e:
            raise PilotError("cannot read %s: %s" % (C.LOCK_PATH, e))
        if want is None:
            problem = "%s records no nevertrain_sha256" % C.LOCK_PATH
        else:
            out["matches_lock"] = want == sha
            if want != sha:
                problem = ("the never-train index %s (%s) is not the one LOCK.json recorded (%s)"
                           % (idx, sha[:12], want[:12]))
    else:
        problem = ("%s does not exist: a production experiment is built only on locked splits "
                   "(run inc.splits lock first)" % C.LOCK_PATH)
    if problem:
        if not testing:
            raise PilotError(problem)
        log("WARNING: %s (testing build)" % problem)
    return out


def select_base_summary(path, sha):
    """The inc.select summary (select_summary.json next to path) when path is
    the base_B.jsonl that summary names by sha256, else None."""
    from . import select as S
    sel = S._read_json(Path(path).parent / S.SUMMARY)
    if not isinstance(sel, dict) or ((sel.get("outputs") or {}).get(S.BASE_B) or {}).get("sha256") != sha:
        return None
    return sel


def select_provenance(sel, testing=False, what="base B"):
    """What an inc.select build read, against what is current: the train_core
    sha256 LOCK.json records and the never-train index in place now. A base
    selected under another lock or index is a stale base (its train_core rows
    or its guard are not the current ones): a production build refuses it, a
    testing build only warns. Returns the record."""
    inputs = (sel or {}).get("inputs") or {}
    core = inputs.get("train_core") or {}
    nt = inputs.get("never_train") or {}
    idx = Path(C.NEVER_TRAIN_INDEX)
    nt_now = C.sha256_file(idx) if idx.is_file() else None
    lock_core = None
    if C.LOCK_PATH.exists():
        try:
            lock_core = (C.read_lock().get("manifests") or {}).get("train_core")
        except (OSError, ValueError) as e:
            raise PilotError("cannot read %s: %s" % (C.LOCK_PATH, e))
    out = {"train_core": {"path": core.get("path"), "sha256": core.get("sha256"),
                          "lock_sha256": lock_core,
                          "matches_lock": core.get("sha256") is not None and core.get("sha256") == lock_core},
           "never_train": {"path": nt.get("path"), "sha256": nt.get("sha256"), "current_sha256": nt_now,
                           "matches_current": nt.get("sha256") is not None and nt.get("sha256") == nt_now}}
    short = lambda h: (h or "none")[:12]  # noqa: E731
    problems = []
    if not out["train_core"]["matches_lock"]:
        problems.append("it read train_core %s, LOCK.json records %s"
                        % (short(core.get("sha256")), short(lock_core)))
    if not out["never_train"]["matches_current"]:
        problems.append("it read the never-train index %s, the index now is %s"
                        % (short(nt.get("sha256")), short(nt_now)))
    if problems:
        msg = ("%s was selected by an inc.select build under other splits: %s (rerun inc.verify and "
               "inc.select build on the current lock and index)" % (what, "; ".join(problems)))
        if not testing:
            raise PilotError(msg)
        log("WARNING: %s (testing build)" % msg)
    return out


def check_training_manifest(path, guard=None, what="training manifest"):
    """(rows, {image: dHash}, info) of a manifest that passes the checks every
    training run of it repeats, before anything is built on it. Fails closed:
      * inc/train.py check_manifest: every row has the manifest keys; keys
        are unique and path-safe; every label line is a box of the INC class
        space (ids 0..12, coordinates in [0, 1]); label and image bytes hash
        as the manifest says; it is not an evaluation split's manifest;
      * the never-train guard (NeverTrainGuard.load() unless one is given)
        over the dHashes that check computed: any image within
        HOLDOUT_NEAR_DUP_BITS of an evaluation image, or that cannot be
        hashed, refuses.
    info is check_manifest's (sizes, train_class_counts, train_sources,
    duplicate counts) plus the guard record."""
    from . import train as T
    try:
        rows, dhashes, info = T.check_manifest(path)
    except T.RunError as e:
        raise PilotError("%s %s refused: %s" % (what, path, e))
    if guard is None:
        try:
            guard = C.NeverTrainGuard.load()
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            raise PilotError("cannot load the never-train index %s: %s" % (C.NEVER_TRAIN_INDEX, e))
    paths = [r["image"] for r in rows]
    hits, unhashable = guard.check(paths, hash_fn=dhashes.get)
    if hits or unhashable:
        raise PilotError("%s %s: %d image(s) within %d dHash bits of an evaluation image and %d that "
                         "cannot be hashed (never-train guard, fail closed); first: %s %s"
                         % (what, path, len(hits), HOLDOUT_NEAR_DUP_BITS, len(unhashable),
                            hits[:3], unhashable[:3]))
    info["guard"] = {"index": str(C.NEVER_TRAIN_INDEX), "index_images": guard.n, "checked": len(paths),
                     "bits": HOLDOUT_NEAR_DUP_BITS, "hits": 0, "unhashable": 0}
    return rows, dhashes, info


def by_source(rows):
    """{source: {"images", "boxes": boxes per INC class}}, sources sorted."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
    return {s: {"images": len(v), "boxes": C.class_counts(v)} for s, v in sorted(groups.items())}


def manifest_summary(rows, sha, info):
    """What a build records of a checked training manifest."""
    return {"manifest_sha256": sha, "images": len(rows),
            "sessions": len({r.get("session") for r in rows if r.get("session")}),
            "boxes_total": info["n_train_boxes"], "boxes": info["train_class_counts"],
            "sources": info["train_sources"], "by_source": by_source(rows),
            "duplicate_images": info["duplicate_images"], "dhash0_collisions": info["dhash0_collisions"],
            "duplicate_label_rows": info["duplicate_label_rows"], "guard": info["guard"]}


def build_baseline(exp, manifest, seeds=D.SEEDS, testing=False, backend=None, guard=None, init=True,
                   quiet=False):
    """A 'baseline' experiment on any training manifest (base B, Step 1.6): the
    manifest copied into the experiment (same bytes), cold with the protocol's
    recipe, one base run per seed; after the base runs, final runs on every
    exam, test included. Refuses, before anything is written, a manifest that
    fails check_training_manifest."""
    testing = _check_testing(testing)
    seeds = check_seeds(seeds)
    paths = D.Paths(exp)
    _check_new(paths)
    src = Path(os.path.abspath(str(manifest)))
    nt = never_train_status(testing=bool(testing))
    rows, _dhashes, info = check_training_manifest(src, guard=guard)
    sha = info["train_manifest_sha256"]
    sel = select_base_summary(src, sha)
    select_build = None
    if sel is not None:
        from . import select as S
        select_build = dict(select_provenance(sel, testing=bool(testing), what=str(src)),
                            summary={"path": str(src.parent / S.SUMMARY),
                                     "sha256": C.sha256_file(src.parent / S.SUMMARY)})
    name = sanitise(src.stem)
    dst = paths.manifests / ("%s.jsonl" % name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if C.sha256_file(dst) != sha:
        raise PilotError("copy of %s does not hash like the manifest that was checked" % src)
    cold = cold_recipe()
    defn = {"exp": exp, "type": "baseline", "builder": "inc.pilot build-baseline", "testing": testing,
            "seeds": list(seeds), "init_weights": D.COLD_INIT, "decision_exam": D.DECISION_EXAM,
            "final_exams": list(D.FINAL_EXAMS),
            "base": _entry(name, dst, rows, sha, recipe=cold, source_manifest=str(src))}
    summary = {"exp": exp, "testing": bool(testing), "built_utc": D._utc(),
               "builder": "inc.pilot build-baseline", "seeds": list(seeds),
               "manifest": dict(manifest_summary(rows, sha, info), name=name, source=str(src), copy=str(dst)),
               "never_train": nt, "select_build": select_build,
               "cold_recipe": cold, "warmup": {"base": effective_warmup(cold, len(rows))}}
    D._write_json(paths.root / BUILD_SUMMARY, summary)
    log("%s: baseline on %s (%s), %d images, %d boxes, %d source(s), seeds %s"
        % (exp, src, sha[:12], len(rows), info["n_train_boxes"], len(info["train_sources"]), list(seeds)))
    result = D.Driver(exp, backend=backend, quiet=quiet).init(defn) if init else None
    return summary, defn, result


def testing_arg(flag, settings_text):
    """The CLI's --testing / --testing-settings as the builders' testing value."""
    if settings_text is None:
        return flag
    try:
        testing = json.loads(settings_text)
    except ValueError as e:
        raise PilotError("--testing-settings is not JSON: %s" % e)
    if not isinstance(testing, dict) or not testing:
        raise PilotError("--testing-settings must be a non-empty JSON object")
    return testing


def main(argv=None):
    ap = argparse.ArgumentParser(description="INC experiment builders: the Step 4 pilot, B0 and the "
                                             "baseline of any training manifest.")
    ap.add_argument("command", choices=("build", "build-b0", "build-baseline"))
    ap.add_argument("--exp", default=None, help="default pilot_v1 (build) / b0_v1 (build-b0); "
                                                "required for build-baseline")
    ap.add_argument("--manifest", default=None,
                    help="build-baseline: the training manifest (base B: INC_DIR/step1/base_B.jsonl)")
    ap.add_argument("--seeds", default=None, help="build-baseline: comma-separated seeds (default 0,1,2)")
    ap.add_argument("--testing", action="store_true",
                    help="gate on test-mode scores (needs %s=1); every output says TESTING" % TEST_ENV)
    ap.add_argument("--testing-settings", default=None,
                    help="as --testing, with inc/train.py's test-mode settings as a JSON object, "
                         "e.g. '{\"imgsz\": 64, \"device\": \"cpu\"}'")
    ap.add_argument("--replay-mode", choices=D.REPLAY_MODES, default=D.DEFAULT_REPLAY_MODE,
                    help="build: 'sample' (default; pilot_v1: cand on D_k + R1, null on R1 + R2) or 'full' "
                         "(full rehearsal: cand on the accepted pool + D_k, null on the pool)")
    ap.add_argument("--gate-flips-mode", choices=D.FLIPS_MODES, default=D.DEFAULT_FLIPS_MODE,
                    help="build: what the gate's flips guard counts: 'negative' (default; protocol v1) or "
                         "'net' (protocol v2: negative - positive flips)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        testing = testing_arg(a.testing, a.testing_settings)
        if a.command != "build-baseline" and (a.manifest is not None or a.seeds is not None):
            raise PilotError("--manifest and --seeds apply to build-baseline")
        if a.command == "build":
            build_pilot(a.exp or "pilot_v1", testing=testing, quiet=a.quiet, replay_mode=a.replay_mode,
                        gate_flips_mode=a.gate_flips_mode)
        elif a.command == "build-b0":
            if a.replay_mode != D.DEFAULT_REPLAY_MODE:
                raise PilotError("--replay-mode applies to build (a chain); build-b0 has no replay")
            if a.gate_flips_mode != D.DEFAULT_FLIPS_MODE:
                raise PilotError("--gate-flips-mode applies to build (a chain); build-b0 makes no gate decision")
            build_b0(a.exp or "b0_v1", testing=testing, quiet=a.quiet)
        else:
            if a.replay_mode != D.DEFAULT_REPLAY_MODE:
                raise PilotError("--replay-mode applies to build (a chain); build-baseline has no replay")
            if a.gate_flips_mode != D.DEFAULT_FLIPS_MODE:
                raise PilotError("--gate-flips-mode applies to build (a chain); build-baseline makes no gate "
                                 "decision")
            if not a.exp or not a.manifest:
                raise PilotError("build-baseline needs --exp and --manifest")
            build_baseline(a.exp, a.manifest, seeds=D.SEEDS if a.seeds is None else a.seeds,
                           testing=testing, quiet=a.quiet)
    except (PilotError, D.DriverError) as e:
        print("[inc.pilot] ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
