"""INC Step 1.4-1.5: the harvested part of base B, and the increments drawn later.

    python -m weed_optimizer_framework.tools.inc.select build --base-frac 0.6 --seed 0
    python -m weed_optimizer_framework.tools.inc.select increments --exp real_v1 --n 6 --other-heavy
        [--sources relevance] [--relevance INC_DIR/step1/relevance.json]
    python -m weed_optimizer_framework.tools.inc.select increments --exp real_v1 --n 6 --other-heavy
        --sources evidence [--min-evidence 1] [--admit-summary INC_DIR/step1/admit_summary.json]
    python -m weed_optimizer_framework.tools.inc.select summary

docs/INCREMENTAL_PROTOCOL.md, Step 1.4: selection mixes three criteria instead of
applying one filter (the DINOv3 data recipe).

Inputs are inc/verify.py's Step 1 outputs, read through its own path constants
and readers (verify.Crops, verify.load_embeddings), so the file names and what
counts as a complete, current set of embedding shards are defined once:
    step1/verified.jsonl      admitted images (a manifest, INC-id labels)
    step1/pool_meta.jsonl     per pool image: source, size, dHash, boxes
    step1/crops.csv           one row per box crop of train_core ("core"), the
                              cwd12 copies ("copy") and the pool ("pool")
    step1/emb/emb_sNNN_of_NNN.npz   BioCLIP-2 features by crop_id (NaN = failed)
    step1/pool_verdicts.npz   box verdicts of every pool crop (other_ok, ...)
    step1/admit_summary.json  must name this verified.jsonl and crops.csv, and
                              the shard set ("embeddings") admit judged with
plus the locked train_core manifest (splits/v1/train_core.jsonl) and the
never-train index (splits/v1/nevertrain_dhash.json).

Box embeddings are L2-normalised. An image's feature is the L2-normalised mean
of its box embeddings. verify admits an image only if it has a box, and every
species box of an admitted image was embedded and verified. An admitted image
whose boxes are all OtherPlant boxes too small to crop has no feature, so it is
never selected and goes to the increment pool.

0. Near-duplicate groups. verify drops only exact dHash duplicates inside the
   pool, but a re-exported fork of a photograph under another slug (Roboflow
   resize, JPEG re-encode) moves a few bits of the hash (near_dup.py). Images
   within NEAR_DUP_BITS of each other, transitively, form one group (dHash from
   pool_meta.jsonl; no image is opened). The group is the unit of everything
   below: it goes to base B or to the increment pool whole, and an increment
   draws it whole, so no increment replays a photograph that is already in the
   base or in another increment. A false match (letterboxed exports) only keeps
   unrelated images on the same side. The summary reports the group sizes.
1. Retrieval (on-domain), on one scale for every image: a percentile of
   train_core's own crops.
   Species k's prototype is the normalised mean of train_core's crops of k.
   k's reference is the cosine of each train_core crop of k to the mean of the
   others (leave-one-out). A box of species k scores the share of that
   reference at or below its own cosine (0..1; train_core's own crops are
   uniform on it). An image with species boxes (score_kind cwd12) scores the
   mean over them.
   OtherPlant has no train_core crops, and an OtherPlant crop's closeness to
   other OtherPlant crops says nothing about train_core's domain. An image
   whose boxes are all OtherPlant (score_kind other) takes its source's score:
   the median percentile of every embedded species box of that source that
   verify did not call a conflict, admitted image or not (the source's domain,
   read from its species boxes without the admission filter). A source with no
   such box gives no evidence; its OtherPlant-only images are not selected.
   Among one source's OtherPlant-only images, typicality breaks the tie: the
   mean over the image's boxes of the highest cosine to OtherPlant prototypes
   (k-means centres of other_ok pool crops). Crops are split into two folds by
   a hash of their image key, and an image is scored against the prototypes of
   the other fold, never against centres made from its own crops.
   Gate: an image scoring below --gate (GATE = 0.10: its boxes sit, on
   average, in train_core's bottom tenth) is off-domain and is not selected.
   The threshold is moderate, not strict (OWL-ST); the summary records the
   share of train_core's own images the same rule passes, and the gated
   images per source.
   A group scores as its lowest-scoring member (a group passes the gate only
   if every member with a feature does).
2. Coverage. Two-level k-means over the image features. Level 1 has
   k1 = round(sqrt(N / 10)) clusters, clamped to [8, 256]. Each level-1 cluster
   is split into up to 8 sub-clusters (one per 10 images). A group sits in the
   clusters of its first member with a feature.
   The base quota Q = round(F * |verified|) images is shared equally across
   level-1 clusters, then across each cluster's sub-clusters; a small
   cluster's unused share is redistributed (water-filling). Inside a
   sub-cluster, groups are taken by descending retrieval score.
   This is one round-robin over the groups that pass the gate: each level-1
   cluster in turn gives its next group, and inside a cluster each
   sub-cluster in turn does. Groups are taken in that pick order until Q
   images are taken (a group that would overshoot Q is passed over). The
   order in which clusters are visited is a seeded permutation.
3. OtherPlant-only budget. Base B is a species base; the protocol tests an
   OtherPlant-heavy increment separately (Steps 2-3). OtherPlant-only images
   may be at most --other-only-frac (OTHER_ONLY_FRAC = 0.10) of the selected
   images. The pick passes over OtherPlant-only groups beyond floor(frac * Q);
   after the cap drops (4), the last-picked OtherPlant-only groups are dropped
   until they are within floor(frac * selected) again.
4. Per-species caps (per-class pruning, Sorscher et al. 2022). The selected
   boxes of species k may not exceed cap_mult x train_core's boxes of k.
   While any species is over its cap, the species furthest over it (largest
   selected / cap ratio) loses its last-picked selected carrier. Drops are
   last-in-first-out over the pick order, per species. Inside a sub-cluster
   the pick order is descending retrieval, so every cluster loses its
   lowest-retrieval carriers first, and the drop keeps the cluster balance.
   Taking the worst species first matters when many species are over at
   once: the images it loses also carry the others, which often brings them
   under their caps with no drop of their own. Going through the species in
   id order, or dropping every carrier of any over-cap species in one sweep,
   pruned the others far below their caps in a 100k-image trial.
   Dropped images go to the increment pool. With --refill, the pick order is
   walked once more, and groups that keep every species within its cap and
   OtherPlant-only images within their budget are added until Q is reached.

Never-train: before anything is written, every verified image (selected and
increment pool alike) is checked against the never-train index at
HOLDOUT_NEAR_DUP_BITS, with its dHash from pool_meta.jsonl; a hit or an image
without a hash is a hard error. `increments` checks its draw the same way.

Outputs, in step1/ (every file is written atomically, the summary last):
    base_selected.jsonl     the selected harvested images
    base_B.jsonl            train_core rows + base_selected rows (base B's manifest)
    increment_pool.jsonl    every other verified image
    select_clusters.csv     per verified image: near-dup group, l1, l2, scores,
                            boxes, pick rank (of its group), status
    select_summary.json     sizes, boxes per species, sources, clusters,
                            retrieval, OtherPlant, near-dup and never-train
                            sections, parameters, input and output sha256s

`increments` draws N disjoint increments of M images each from
increment_pool.jsonl; M defaults to INC_FRAC (10 %) of base B's images, as the
protocol fixes it. It uses the build's own clusters (images without a feature
form one more cluster) and near-dup groups, in a seeded order:
    stable_int(exp) permutes the clusters and shuffles each sub-cluster;
    groups are drawn in the same two-level round-robin until N*M images;
    the drawn groups are dealt to the increments cluster by cluster, each to
    the emptiest increment it fits in, so every cluster's share splits across
    the increments to within one group and a group is never split. A group
    that fits in no increment is left in the pool and replaced by the next.
Each increment is therefore itself cluster-balanced. With --other-heavy it
first draws one more increment of the same size, the same way, from the
OtherPlant-heavy groups (at least OTHER_HEAVY_MIN of their boxes OtherPlant),
seeded by stable_int(exp + "/otherplant"), and the N regular increments come
from what is left. It writes INC_DIR/<exp>/manifests/inc_01.jsonl ..
inc_NN.jsonl, inc_otherplant.jsonl and increments_summary.json.
The protocol's unverified-source increment is not drawn here: its images are
the ones verify did not admit, which select never loads (no verified labels,
no cluster). The Steps 2-3 builder draws it, with the never-train guard and
with near-copies of base B kept out.

--relevance PATH (inc/relevance.py's relevance.json for this build): the
verifier certifies species labels, not relevance, so an OtherPlant box on a
non-plant image is admitted. With the file, every increment-pool image of a
source whose relevance status is not 'pass' leaves the pool before the draw
(both the regular and the OtherPlant-heavy increments). The file must be the
one made for this build's increment_pool.jsonl and base_selected.jsonl, hold
every pool source with the pool's image counts, and have each status follow
from its recorded numbers (relevance.load); otherwise the draw refuses. The
file's sha256 joins the draw's parameters, and the summary's "relevance"
section records it, the excluded sources with their images, and the near-dup
groups the exclusion split (their members of passing sources stay drawable).
Without --relevance the draw and its parameters are exactly as before.

--sources picks the relevance criterion of the draw. 'relevance' (the
default) is the above: --relevance if given, else no filter, byte for byte
as before. 'evidence' is source-level species evidence, the second criterion
(docs/INCREMENTAL_PROTOCOL.md, Steps 2-3; adopted at an R4 review after the
zero-shot relevance build failed its own calibration check): a pool source is
eligible for the regular and OtherPlant-heavy draws only if verify admit
judged at least --min-evidence (MIN_EVIDENCE = 1) of its cwd12-species boxes
'verified' (admit_summary.json per_slug boxes; --admit-summary, default
verify's own file). Every increment-pool image of any other source leaves the
pool before the draw. It is conservative: a genuine weed dataset of species
outside cwd12 holds no verified box and is excluded too. load_evidence
refuses a --min-evidence below 1; an admit summary without per-source
verdict counts; one that does not name the verified.jsonl and crops.csv this
build read; and, where select_summary.json records retrieval.source_evidence,
a disagreement between the two files: a verified box is a species crop with
features that verify did not call a conflict, which is what that record
counts per source, so every source needs species_crops >= its verified boxes,
and every source it counts needs a per_slug entry. apply_evidence refuses a
pool source without a per_slug entry. 'evidence' takes no --relevance.
The mode, --min-evidence and the admit summary's sha256 join the draw's
parameters; the summary's "evidence" section records the evidenced sources
with their verified boxes and pool images, the excluded sources with their
images and verified boxes, the cross-check and the near-dup groups the
exclusion split. pool_capacity says what a filtered pool can supply, so
realloop can refuse before it writes anything.

Both commands are deterministic. k-means runs single-threaded: sklearn's Lloyd
step adds the per-thread partial centres in whatever order the threads
finish, so on many threads the clusters depend on the core count. The thread
count and the sklearn and numpy versions are parameters, so a rebuild under
other versions refuses without --force. A rerun with the same inputs and
parameters is a no-op. A rerun that would overwrite outputs made from
different inputs or parameters refuses unless --force is given, because an
experiment may already train on them.

Nothing here opens an image.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import common as C

N_SPECIES = C.OTHER_PLANT            # ids 0..11 are the cwd12 species

# Inputs: every Step 1 file name comes from inc/verify.py (see _verify()); this
# pattern (verify._shard_name, verify.load_embeddings) only fingerprints them.
EMB_SHARD_RE = r"emb_s\d{3}_of_%03d\.npz"

# --- outputs ------------------------------------------------------------------
BASE_SELECTED = "base_selected.jsonl"
BASE_B = "base_B.jsonl"
POOL = "increment_pool.jsonl"
CLUSTERS = "select_clusters.csv"
SUMMARY = "select_summary.json"
INC_SUMMARY = "increments_summary.json"
INC_NAME = "inc_%02d.jsonl"
INC_OTHER = "inc_otherplant.jsonl"
CLUSTER_COLS = ("key", "dup_group", "l1", "l2", "score_kind", "cosine", "score", "typicality",
                "species_boxes", "other_boxes", "rank", "status")

# --- parameters ---------------------------------------------------------------
BASE_FRAC = 0.6
CAP_MULT = 3.0
GATE = 0.10                             # retrieval gate, on the train_core percentile scale
OTHER_ONLY_FRAC = 0.10                  # OtherPlant-only images: at most this share of base
INC_FRAC = 0.10                         # increment size: this share of base B's images
OTHER_HEAVY_MIN = 0.5                   # OtherPlant-heavy group: this share of its boxes
SOURCES_RELEVANCE, SOURCES_EVIDENCE = "relevance", "evidence"     # increments --sources
SOURCE_MODES = (SOURCES_RELEVANCE, SOURCES_EVIDENCE)
MIN_EVIDENCE = 1                        # --sources evidence: verified cwd12-species boxes a source needs
EVIDENCE_RULE = ("source-level species evidence: an increment-pool source is eligible for the regular and "
                 "OtherPlant-heavy draws only if verify admit judged at least min_evidence of its cwd12-species "
                 "boxes 'verified' (admit_summary.json per_slug boxes); every image of any other source leaves "
                 "the draw")
K1_PER, K1_MIN, K1_MAX = 10, 8, 256     # k1 = round(sqrt(N / K1_PER)) in [K1_MIN, K1_MAX]
K2_PER, K2_MAX = 10, 8                  # k2 = min(K2_MAX, n_cluster // K2_PER), >= 1
N_INIT, MAX_ITER = 3, 100               # sklearn KMeans, fixed so runs repeat exactly
KMEANS_THREADS = 1                      # see the module docstring (determinism)
OTHER_SAMPLE, OTHER_K_MAX = 50_000, 64  # OtherPlant prototypes (both folds together)
PROTO_SEED = 0                          # prototypes do not depend on --seed
LARGE_GROUP = 50                        # near-dup groups above this are reported
CHUNK = 65_536                          # crops per float32 chunk
SCORE_CHUNK = 16_384                    # crops per float64 chunk
WORKERS = 5


class SelectError(RuntimeError):
    pass


def log(msg):
    print("[inc.select %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def step1_dir():
    return C.INC_DIR / "step1"


def _verify():
    """inc/verify.py, imported when needed: its STEP1 paths, crops.csv reader
    (Crops) and shard loader (load_embeddings) define Step 1's inputs."""
    from . import verify as V
    return V


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _file_info(path, rows=None):
    info = {"path": str(path), "sha256": C.sha256_file(path)}
    if rows is not None:
        info["rows"] = rows
    return info


def _versions():
    import numpy as np
    import sklearn
    return {"numpy": np.__version__, "sklearn": sklearn.__version__}


# ------------------------------------------------------------------ inputs
def _count_label(path):
    """Boxes per INC class in one label file. Unlike common.read_yolo, a missing
    file raises: a verified row without its label is an error, not a background."""
    counts = [0] * C.NC
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if len(t) < 5:
                continue
            c = int(float(t[0]))
            if not 0 <= c < C.NC:
                raise SelectError("%s: class id %d outside the INC class space" % (path, c))
            counts[c] += 1
    return counts


def box_counts(rows, workers=WORKERS, what="labels"):
    """(len(rows), NC) int array of boxes per class. Label files are read in
    threads (Lustre latency, not CPU, is the cost), in row order."""
    import numpy as np
    out = np.zeros((len(rows), C.NC), dtype=np.int64)
    paths = [r["label"] for r in rows]
    step = max(1, len(paths) // 10)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, c in enumerate(ex.map(_count_label, paths)):
            out[i] = c
            if len(paths) >= 20_000 and (i + 1) % step == 0:
                log("%s: %d/%d label files read" % (what, i + 1, len(paths)))
    return out


def _species_dict(vec):
    return {C.CLASS_NAMES[i]: int(vec[i]) for i in range(len(vec))}


def _check_rows(rows, what):
    for r in rows:
        missing = [k for k in C.MANIFEST_KEYS if k not in r]
        if missing:
            raise SelectError("%s row missing %s: %r" % (what, missing, r))
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        dup = [k for k, n in collections.Counter(keys).items() if n > 1]
        raise SelectError("%s has duplicate keys, e.g. %s" % (what, dup[:5]))


def _never_train_slugs():
    """NEVER_TRAIN_SLUGS plus the slugs that hold cwd12 valid/test copies
    (mega_trainer.CWD12_COPY_ID_MAPS; verify._skip_reason skips both)."""
    from ..mega_trainer import CWD12_COPY_ID_MAPS, NEVER_TRAIN_SLUGS
    return set(NEVER_TRAIN_SLUGS) | set(CWD12_COPY_ID_MAPS)


def check_pool_against_core(pool, core, never_train=None):
    """The pool is harvested data only: no key, image path or image bytes
    shared with train_core, and nothing from a never-train dataset (cwd12
    itself, the exam sources, the slugs holding cwd12 copies). Raises on any."""
    core_keys = {r["key"] for r in core}
    clash = [r["key"] for r in pool if r["key"] in core_keys]
    if clash:
        raise SelectError("%d verified key(s) are also train_core keys (base_B needs "
                          "unique keys), e.g. %s" % (len(clash), clash[:5]))
    core_imgs = {r["image"] for r in core}
    same = [r["image"] for r in pool if r["image"] in core_imgs]
    if same:
        raise SelectError("%d verified image(s) are train_core images, e.g. %s"
                          % (len(same), same[:3]))
    slugs = _never_train_slugs() if never_train is None else set(never_train)
    bad = [r for r in pool
           if r["source"].split("/")[0] in slugs or set(Path(r["image"]).parts) & slugs]
    if bad:
        raise SelectError("%d verified image(s) come from a never-train dataset (%s), e.g. %s"
                          % (len(bad), sorted(slugs), [(r["source"], r["image"]) for r in bad[:3]]))
    core_sha = {r["sha256"]: r["key"] for r in core}
    copies = [(r["key"], core_sha[r["sha256"]]) for r in pool if r["sha256"] in core_sha]
    if copies:
        raise SelectError("%d verified image(s) are byte-identical to a train_core image "
                          "(verify could not hash that train_core image?), e.g. %s"
                          % (len(copies), copies[:3]))


def read_pool_dhash(path, keys):
    """{key: dHash} from verify's pool_meta.jsonl for the given keys; raises if
    one is missing (an image we cannot compare is an image we cannot place)."""
    want = set(keys)
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m["key"] in want:
                out[m["key"]] = m.get("dhash")
    missing = [k for k in keys if out.get(k) is None]
    if missing:
        raise SelectError("%d image(s) have no dHash in %s, e.g. %s: it was made by another "
                          "`verify pool`" % (len(missing), path, missing[:3]))
    return out


def _load_guard(path=None):
    path = Path(path or C.NEVER_TRAIN_INDEX)
    try:
        return C.NeverTrainGuard.load(path)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        raise SelectError("never-train index %s: %s (run inc/splits.py build)" % (path, e))


def guard_rows(guard, rows, dh, what):
    """NeverTrainGuard.assert_trainable over rows, hashing through dh (key ->
    dHash). Returns the number of images checked."""
    by_image = {r["image"]: dh.get(r["key"]) for r in rows}
    try:
        guard.assert_trainable([r["image"] for r in rows], hash_fn=by_image.get)
    except RuntimeError as e:
        raise SelectError("%s: %s; rerun `verify pool` with the current never-train index"
                          % (what, e))
    return len(rows)


def _finite_rows(X, idx):
    """Mask over idx: rows of X that are finite (verify writes NaN rows for crops
    it could not cut or embed). Chunked, so no full-size temporary."""
    import numpy as np
    out = np.zeros(len(idx), dtype=bool)
    for s in range(0, len(idx), CHUNK):
        out[s:s + CHUNK] = np.isfinite(X[idx[s:s + CHUNK]]).all(axis=1)
    return out


def read_pool_verdicts(path, crops):
    """Box verdict of every pool crop (verify admit) as an int8 array over all
    crop ids (-1: not a pool crop), and the verdict names."""
    import numpy as np
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        cid, verdict = d["crop_id"], d["verdict"]
    if meta.get("crops_sha256") != crops.sha:
        raise SelectError("%s was made from another crops.csv; rerun verify admit" % path)
    out = np.full(crops.n, -1, dtype=np.int8)
    out[cid.astype(np.int64)] = verdict
    return out, list(meta["verdict_codes"])


def fold_of(key):
    """Typicality fold (0 or 1) of an image, by its key."""
    return C.stable_int("inc.select/fold/" + key) % 2


def crop_index(crops, pool_index, core_keys, verdict, codes, E, V):
    """The crops select uses, as crop ids into verify's X:
        core, core_img     train_core crops with features, and their image
        pool, pool_img     crops of admitted images, and their image (every
                           one must have features)
        src, src_id        species-labelled pool crops with features of any
                           image, that verify did not call a conflict (their
                           source's domain evidence), and their source's index
                           into `sources`
        other, other_fold  pool crops verify judged other_ok with features,
                           and their fold (fold_of their image key)
    Returns (J, info)."""
    import numpy as np
    core_all = crops.where("core")
    stale = [crops.key[i] for i in core_all if crops.key[i] not in core_keys]
    if stale:
        raise SelectError("crops.csv has %d train_core crop(s) of images that are not in the "
                          "train_core manifest, e.g. %s: it was made from another train_core"
                          % (len(stale), stale[:3]))
    core_ix = {k: i for i, k in enumerate(sorted(core_keys))}
    pool_all = crops.where("pool")
    img = np.fromiter((pool_index.get(crops.key[i], -1) for i in pool_all),
                      dtype=np.int64, count=len(pool_all))
    keep = img >= 0
    pool_j, pool_img = pool_all[keep].astype(np.int64), img[keep]
    fin = _finite_rows(E, pool_j)
    if not fin.all():
        raise SelectError("%d crop(s) of admitted images have no features (NaN), e.g. crop ids "
                          "%s: the embeddings changed after verify admit"
                          % (int((~fin).sum()), pool_j[~fin][:5].tolist()))
    fin = _finite_rows(E, core_all)
    core_j = core_all[fin].astype(np.int64)
    core_img = np.array([core_ix[crops.key[i]] for i in core_j], dtype=np.int64)
    v = verdict[pool_all]
    if (v < 0).any():
        raise SelectError("%d pool crop(s) have no verdict in pool_verdicts.npz"
                          % int((v < 0).sum()))
    lab = crops.label[pool_all]
    no_evidence = np.isin(v, [codes.index(V.CONFLICT), codes.index(V.FAILED)])
    src_j = pool_all[(lab < N_SPECIES) & ~no_evidence].astype(np.int64)
    src_j = src_j[_finite_rows(E, src_j)]
    names = [crops.source[i] for i in src_j]
    sources, src_id = (np.unique(names, return_inverse=True) if names
                       else (np.array([], dtype=str), np.zeros(0, dtype=np.int64)))
    other_all = pool_all[v == codes.index(V.OTHER_OK)].astype(np.int64)
    other_j = other_all[_finite_rows(E, other_all)]
    fold_cache = {}
    other_fold = np.array([fold_cache.setdefault(crops.key[i], fold_of(crops.key[i]))
                           for i in other_j], dtype=np.int64)
    J = {"core": core_j, "core_img": core_img, "pool": pool_j, "pool_img": pool_img,
         "src": src_j, "src_id": np.asarray(src_id, dtype=np.int64),
         "sources": [str(s) for s in sources], "other": other_j, "other_fold": other_fold}
    info = {"crops_csv_rows": int(crops.n), "train_core_crops": int(len(core_all)),
            "train_core_crops_failed": int(len(core_all) - len(core_j)),
            "pool_crops": int(len(pool_all)), "admitted_image_crops": int(keep.sum()),
            "source_evidence_crops": int(len(src_j)),
            "other_ok_crops": int((v == codes.index(V.OTHER_OK)).sum()),
            "other_ok_crops_with_features": int(len(other_j))}
    return J, info


def _unit(X):
    import numpy as np
    X = X.astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _unit64(X):
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------ near-dup groups
_POP8 = None


def _popcount64(x):
    """Set bits of every uint64 in x (same shape)."""
    import numpy as np
    global _POP8
    x = np.ascontiguousarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    if _POP8 is None:
        _POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
    return _POP8[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def dup_groups(hashes, bits=None):
    """Near-duplicate group of every image: images whose dHashes are within
    `bits` (near_dup.NEAR_DUP_BITS) of each other, transitively, share a group.
    Groups are numbered 0.. by their first image.

    Pigeonhole, as near_dup.NearHashIndex: two hashes within `bits` agree
    exactly on at least one of bits + 1 disjoint blocks, so only hashes that
    share a block value are compared, vectorised per bucket (letterboxed
    exports put many hashes in one bucket)."""
    import numpy as np
    from ..near_dup import NEAR_DUP_BITS
    bits = NEAR_DUP_BITS if bits is None else int(bits)
    H = np.array([int(h) for h in hashes], dtype=np.uint64)
    n = len(H)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    parent = list(range(n))

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nb = bits + 1
    shift = 0
    for i in range(nb):
        w = 64 // nb + (1 if i < 64 % nb else 0)
        key = (H >> np.uint64(shift)) & np.uint64((1 << w) - 1)
        shift += w
        o = np.argsort(key, kind="stable")
        ks = key[o]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], n]
        for a, b in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
            run = o[a:b]
            hr = H[run]
            if len(run) <= 256:
                d = _popcount64(hr[:, None] ^ hr[None, :])
                x, y = np.nonzero(np.triu(d <= bits, 1))
                for p, q in zip(x, y):
                    union(run[p], run[q])
            else:
                for t in range(len(run) - 1):
                    near = np.flatnonzero(_popcount64(hr[t + 1:] ^ hr[t]) <= bits)
                    for q in near:
                        union(run[t], run[t + 1 + q])
    return canonical_labels(np.array([find(i) for i in range(n)], dtype=np.int64))


# ------------------------------------------------------------- clustering
def _fit_kmeans(X, k, seed):
    import warnings
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    with warnings.catch_warnings(), threadpool_limits(limits=KMEANS_THREADS):
        warnings.simplefilter("ignore")        # "fewer distinct points than clusters"
        return KMeans(n_clusters=k, n_init=N_INIT, max_iter=MAX_ITER,
                      random_state=int(seed)).fit(X)


def _kmeans(X, k, seed):
    import numpy as np
    k = min(int(k), len(X))
    if k <= 1:
        return np.zeros(len(X), dtype=np.int64)
    return canonical_labels(_fit_kmeans(X, k, seed).labels_)


def _kmeans_centres(X, k, seed):
    import numpy as np
    k = min(int(k), len(X))
    if k <= 1:
        return X.mean(0, keepdims=True)
    return np.asarray(_fit_kmeans(X, k, seed).cluster_centers_, dtype=np.float32)


def canonical_labels(lab):
    """Renumber cluster labels 0.. by first appearance, so ids do not depend on
    k-means' internal numbering and empty clusters leave no gap."""
    import numpy as np
    lab = np.asarray(lab)
    if lab.size == 0:
        return lab.astype(np.int64)
    uniq, first = np.unique(lab, return_index=True)
    remap = np.empty(int(uniq.max()) + 1, dtype=np.int64)
    remap[uniq[np.argsort(first)]] = np.arange(len(uniq))
    return remap[lab]


def k1_for(n):
    return max(1, min(n, int(min(K1_MAX, max(K1_MIN, round(math.sqrt(n / K1_PER)))))))


def hierarchical_kmeans(F, seed):
    """(l1, l2): level-1 cluster and sub-cluster of every row of F."""
    import numpy as np
    n = len(F)
    l1 = _kmeans(F, k1_for(n), seed)
    l2 = np.zeros(n, dtype=np.int64)
    for c in range(int(l1.max()) + 1 if n else 0):
        idx = np.flatnonzero(l1 == c)
        k2 = min(K2_MAX, max(1, len(idx) // K2_PER))
        if k2 > 1:
            l2[idx] = _kmeans(F[idx], k2, C.stable_int("%d/l2/%d" % (seed, c)))
    return l1, l2


def round_robin(groups):
    """Interleave index arrays: the first item of every group, then the second,
    and so on, skipping exhausted groups. Any prefix of length q gives each group
    min(len, t) or min(len, t + 1) items, the extra ones going to the earliest
    groups: an equal split with the unused share of small groups redistributed."""
    import numpy as np
    groups = [np.asarray(g, dtype=np.int64) for g in groups if len(g)]
    if not groups:
        return np.empty(0, dtype=np.int64)
    items = np.concatenate(groups)
    pos = np.concatenate([np.arange(len(g)) for g in groups])
    grp = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    return items[np.lexsort((grp, pos))]


def cluster_lists(idx, l1, l2, within, rng):
    """One pick list per level-1 cluster, clusters in a seeded order: a round-robin
    over the cluster's sub-clusters (seeded order), each sub-cluster ordered by
    within(indices)."""
    import numpy as np
    idx = np.asarray(idx, dtype=np.int64)
    if not len(idx):
        return []
    a, b = l1[idx], l2[idx]
    per_cluster = []
    for c in rng.permutation(np.unique(a)):
        in_c = idx[a == c]
        bc = b[a == c]
        per_cluster.append(round_robin([within(in_c[bc == s]) for s in rng.permutation(np.unique(bc))]))
    return per_cluster


def balanced_order(idx, l1, l2, within, rng):
    """Pick order over idx: round-robin over cluster_lists."""
    return round_robin(cluster_lists(idx, l1, l2, within, rng))


# --------------------------------------------------------------- features
def image_features(n_img, E, pool_j, pool_img):
    """(F, has): (n_img, D) float32 unit image features (the normalised mean of
    the image's unit box embeddings; zero rows where has is False)."""
    import numpy as np
    D = E.shape[1]
    sums = np.zeros((n_img, D), dtype=np.float32)
    nbox = np.zeros(n_img, dtype=np.int64)
    o = np.argsort(pool_img, kind="stable")
    pj, pimg = pool_j[o], pool_img[o]
    for s in range(0, len(pj), CHUNK):
        X = _unit(E[pj[s:s + CHUNK]])
        ii = pimg[s:s + CHUNK]
        starts = np.flatnonzero(np.r_[True, ii[1:] != ii[:-1]])
        sums[ii[starts]] += np.add.reduceat(X, starts, axis=0)
        nbox += np.bincount(ii, minlength=n_img)
    has = nbox > 0
    F = np.zeros_like(sums)
    F[has] = _unit(sums[has])
    return F, has


def species_reference(E, core_j, lab):
    """(P, ref, loo): unit species prototypes [12, D] (float64); per species,
    the sorted leave-one-out cosines of train_core's crops (the percentile
    scale); and every core_j crop's own leave-one-out cosine."""
    import numpy as np
    D = E.shape[1]
    S = np.zeros((N_SPECIES, D), dtype=np.float64)
    n = np.zeros(N_SPECIES, dtype=np.int64)
    y_all = lab[core_j]
    for s in range(0, len(core_j), SCORE_CHUNK):
        X = _unit64(E[core_j[s:s + SCORE_CHUNK]])
        y = y_all[s:s + SCORE_CHUNK]
        for k in range(N_SPECIES):
            m = y == k
            if m.any():
                S[k] += X[m].sum(0)
                n[k] += int(m.sum())
    if (n < 2).any():
        raise SelectError("fewer than 2 embedded train_core crops for %s: the retrieval scale "
                          "needs train_core crops in crops.csv and the shards"
                          % [C.CLASS_NAMES[k] for k in np.flatnonzero(n < 2)])
    ss = (S * S).sum(1)
    loo = np.empty(len(core_j), dtype=np.float64)
    for s in range(0, len(core_j), SCORE_CHUNK):
        X = _unit64(E[core_j[s:s + SCORE_CHUNK]])
        y = y_all[s:s + SCORE_CHUNK]
        xs = (X * S[y]).sum(1)
        xx = (X * X).sum(1)
        loo[s:s + len(y)] = (xs - xx) / np.sqrt(np.maximum(ss[y] - 2.0 * xs + xx, 1e-24))
    ref = [np.sort(loo[y_all == k]) for k in range(N_SPECIES)]
    return _unit64(S), ref, loo


def percentile(ref, cos, y):
    """Share of species y's reference at or below cos, per box."""
    import numpy as np
    cos = np.asarray(cos, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    u = np.empty(len(cos), dtype=np.float64)
    for k in np.unique(y):
        m = y == k
        u[m] = np.searchsorted(ref[k], cos[m], side="right") / len(ref[k])
    return u


def species_box_scores(E, j, lab, P, ref):
    """(cos, u) of species-labelled crops j: cosine to their species'
    prototype and its percentile."""
    import numpy as np
    cos = np.empty(len(j), dtype=np.float64)
    for s in range(0, len(j), SCORE_CHUNK):
        jj = j[s:s + SCORE_CHUNK]
        cos[s:s + len(jj)] = (_unit64(E[jj]) * P[lab[jj]]).sum(1)
    return cos, percentile(ref, cos, lab[j])


def typicality(E, other_j, other_fold, box_j, box_img, img_fold, n_img):
    """Per image: mean over its boxes box_j (box_img = their image) of the
    highest cosine to the OtherPlant prototypes of the OTHER fold (NaN for an
    image without such a box, or whose other fold has no crops). Prototypes of
    fold f: k-means centres of up to OTHER_SAMPLE / 2 of fold f's other_ok
    crops. Returns (typ, info)."""
    import numpy as np
    rng = np.random.default_rng(PROTO_SEED)
    protos, info = [None, None], {"sample": [0, 0], "prototypes": [0, 0]}
    for f in (0, 1):
        oj = other_j[other_fold == f]
        if len(oj) > OTHER_SAMPLE // 2:
            oj = np.sort(rng.choice(oj, OTHER_SAMPLE // 2, replace=False))
        if len(oj):
            k = int(min(OTHER_K_MAX, max(1, round(math.sqrt(len(oj) / K1_PER)))))
            protos[f] = _unit(_kmeans_centres(_unit(E[oj]), k, PROTO_SEED))
            info["sample"][f], info["prototypes"][f] = int(len(oj)), int(len(protos[f]))
    s_t = np.zeros(n_img, dtype=np.float64)
    n_t = np.zeros(n_img, dtype=np.int64)
    for s in range(0, len(box_j), CHUNK):
        X = _unit(E[box_j[s:s + CHUNK]])
        ii = box_img[s:s + CHUNK]
        f_img = img_fold[ii]
        for f in (0, 1):
            O = protos[1 - f]
            m = f_img == f
            if O is None or not m.any():
                continue
            s_t += np.bincount(ii[m], weights=(X[m] @ O.T).max(1), minlength=n_img)
            n_t += np.bincount(ii[m], minlength=n_img)
    typ = np.full(n_img, np.nan)
    typ[n_t > 0] = s_t[n_t > 0] / n_t[n_t > 0]
    return typ, info


def retrieval(n_img, E, lab, J, img_src, img_fold, gate=GATE):
    """Per image: (kind, cosine, score, typ, info). kind 1 = scored on its
    species boxes, 2 = OtherPlant-only (its source's score), 0 = none.
    cosine: kind 1's mean raw cosine. score: the percentile score (NaN if
    none). typ: kind 2's typicality. See the module docstring, 1."""
    import numpy as np
    P, ref, loo = species_reference(E, J["core"], lab)
    # train_core's own images on the same scale: how many would pass the gate
    u_core = percentile(ref, loo, lab[J["core"]])
    n_ci = int(J["core_img"].max()) + 1 if len(J["core_img"]) else 0
    cnt = np.bincount(J["core_img"], minlength=n_ci)
    core_score = (np.bincount(J["core_img"], weights=u_core, minlength=n_ci)[cnt > 0]
                  / cnt[cnt > 0])
    # images with species boxes
    pj, pimg = J["pool"], J["pool_img"]
    sp = lab[pj] < N_SPECIES
    cos, u = species_box_scores(E, pj[sp], lab, P, ref)
    n_sp = np.bincount(pimg[sp], minlength=n_img)
    s_cos = np.bincount(pimg[sp], weights=cos, minlength=n_img)
    s_u = np.bincount(pimg[sp], weights=u, minlength=n_img)
    # the sources' evidence
    _c, u_src = species_box_scores(E, J["src"], lab, P, ref)
    n_src = len(J["sources"])
    src_med = np.full(n_src, np.nan)
    src_n = np.bincount(J["src_id"], minlength=n_src)
    o = np.argsort(J["src_id"], kind="stable")
    for s, part in zip(range(n_src), np.split(u_src[o], np.cumsum(src_n)[:-1])):
        if len(part):
            src_med[s] = float(np.median(part))
    # OtherPlant-only images
    has_box = np.bincount(pimg, minlength=n_img) > 0
    kind = np.zeros(n_img, dtype=np.int64)
    cosine = np.full(n_img, np.nan)
    score = np.full(n_img, np.nan)
    m1 = n_sp > 0
    kind[m1] = 1
    cosine[m1] = s_cos[m1] / n_sp[m1]
    score[m1] = s_u[m1] / n_sp[m1]
    m2 = has_box & ~m1
    kind[m2] = 2
    ok = m2 & (img_src >= 0)
    score[ok] = src_med[img_src[ok]]
    ot = lab[pj] == C.OTHER_PLANT
    in2 = m2[pimg]
    typ, tinfo = typicality(E, J["other"], J["other_fold"], pj[ot & in2], pimg[ot & in2],
                            img_fold, n_img)
    info = {"scale": "percentile of train_core's leave-one-out crop-to-prototype cosine",
            "gate": float(gate),
            "train_core_images": int(len(core_score)),
            "train_core_images_passing_gate": (round(float((core_score >= gate).mean()), 6)
                                               if len(core_score) else None),
            "train_core_image_score_q05_q50": ([round(float(x), 6) for x in
                                                np.quantile(core_score, [0.05, 0.5])]
                                               if len(core_score) else None),
            "train_core_crops_per_species": _species_dict(
                np.array([len(r) for r in ref])),
            "source_evidence": {J["sources"][s]: {"species_crops": int(src_n[s]),
                                                  "median": round(float(src_med[s]), 6)}
                                for s in range(n_src) if src_n[s]},
            "other_typicality": {"other_ok_crops": int(len(J["other"])), **tinfo},
            "dim": int(E.shape[1])}
    return kind, cosine, score, typ, info


# ------------------------------------------------------------------ build
def choose(order, q, counts, caps, sizes=None, other_only=None, other_frac=None, refill=False):
    """The selection over units (near-dup groups) in pick order; see the
    module docstring, 2-4.

    order: unit ids in pick order. q: the quota in images. counts: (U, NC)
    boxes per unit. caps: species caps (ids 0..len(caps)-1). sizes: images
    per unit (default 1). other_only: units without a species box.
    other_frac: their budget as a share of the selected images (None or >= 1:
    no budget). Returns a dict: sel (bool mask over units), status ({unit:
    'other_budget' | 'dropped_cap' | 'dropped_other_budget' | 'refilled'}),
    cap_drops (units, in drop order) and picked_boxes (boxes of the first pick,
    before any drop)."""
    import numpy as np
    order = np.asarray(order, dtype=np.int64)
    n = counts.shape[0]
    caps = np.asarray(caps, dtype=np.int64)
    ns = len(caps)
    size = np.ones(n, dtype=np.int64) if sizes is None else np.asarray(sizes, dtype=np.int64)
    oth = np.zeros(n, dtype=bool) if other_only is None else np.asarray(other_only, dtype=bool)
    frac = None if other_frac is None or other_frac >= 1 else float(other_frac)

    def budget(total):
        return math.inf if frac is None else math.floor(frac * total + 1e-9)

    sel = np.zeros(n, dtype=bool)
    status = {}
    picked = []
    have = have_o = 0
    for u in order:
        if have >= q:
            break
        u, s = int(u), int(size[u])
        if have + s > q:
            continue
        if oth[u] and have_o + s > budget(q):
            status[u] = "other_budget"
            continue
        sel[u] = True
        picked.append(u)
        have += s
        have_o += s * int(oth[u])
    tot = counts[picked].sum(0) if picked else np.zeros(counts.shape[1], dtype=np.int64)
    picked_boxes = tot.copy()
    rev = np.array(picked[::-1], dtype=np.int64)
    # LIFO carrier lists; only species over cap now can ever be over (drops only lower counts)
    carriers = {k: rev[counts[rev, k] > 0] for k in np.flatnonzero(tot[:ns] > caps)}
    ptr = dict.fromkeys(carriers, 0)
    cap_drops = []
    while True:
        over = tot[:ns] > caps
        if not over.any():
            break
        ratio = np.where(over, tot[:ns] / np.maximum(caps, 0.5), -1.0)
        k = int(np.argmax(ratio))                      # ties: the lowest id
        lst, p = carriers[k], ptr[k]
        while not sel[lst[p]]:                         # already dropped for another species
            p += 1
        ptr[k] = p + 1
        u = int(lst[p])
        sel[u] = False
        tot = tot - counts[u]
        have -= int(size[u])
        have_o -= int(size[u]) * int(oth[u])
        status[u] = "dropped_cap"
        cap_drops.append(u)
    for u in rev:                                      # the budget, on what is left
        if have_o <= budget(have):
            break
        u = int(u)
        if sel[u] and oth[u]:
            sel[u] = False
            have -= int(size[u])
            have_o -= int(size[u])
            status[u] = "dropped_other_budget"
    if refill:
        for u in order:
            if have >= q:
                break
            u = int(u)
            if sel[u] or status.get(u) in ("dropped_cap", "dropped_other_budget"):
                continue
            s, c = int(size[u]), counts[u]
            if have + s > q or not np.all(tot[:ns] + c[:ns] <= caps):
                continue
            if oth[u] and have_o + s > budget(have + s):
                continue
            sel[u] = True
            tot = tot + c
            have += s
            have_o += s * int(oth[u])
            status[u] = "refilled"
    return {"sel": sel, "status": status, "cap_drops": cap_drops, "picked_boxes": picked_boxes}


def _write_clusters(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(CLUSTER_COLS)
        wr.writerows(rows)
    os.replace(tmp, path)
    return C.sha256_file(path)


def read_clusters(path):
    with open(path, newline="") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def _sources(rows):
    return dict(sorted(collections.Counter(r["source"] for r in rows).items()))


def _outputs_match(summary, out_dir, names):
    outs = (summary or {}).get("outputs", {})
    for n in names:
        p = Path(out_dir) / n
        if n not in outs or not p.exists() or C.sha256_file(p) != outs[n]["sha256"]:
            return False
    return True


def _fmt(x):
    import numpy as np
    return "" if x is None or not np.isfinite(x) else "%.6f" % x


def _group_info(group):
    import numpy as np
    size = np.bincount(group) if len(group) else np.zeros(0, dtype=np.int64)
    multi = size[size > 1]
    return {"bits": _near_dup_bits(), "groups": int(len(size)),
            "multi_image_groups": int(len(multi)), "images_in_multi_groups": int(multi.sum()),
            "largest_group": int(size.max()) if len(size) else 0,
            "groups_over_%d" % LARGE_GROUP: int((size > LARGE_GROUP).sum())}


def _near_dup_bits():
    from ..near_dup import NEAR_DUP_BITS
    return NEAR_DUP_BITS


def build(base_frac=BASE_FRAC, seed=0, cap_mult=CAP_MULT, refill=False, out_dir=None,
          core_manifest=None, lock_check=True, workers=WORKERS, force=False, never_train=None,
          gate=GATE, other_only_frac=OTHER_ONLY_FRAC):
    import numpy as np
    V = _verify()
    out_dir = Path(out_dir or V.STEP1)
    core_manifest = Path(core_manifest or C.manifest_path("train_core"))
    if not 0.0 <= base_frac <= 1.0:
        raise SelectError("--base-frac must be in [0, 1], got %r" % base_frac)
    if cap_mult < 0:
        raise SelectError("--cap-mult must be >= 0, got %r" % cap_mult)
    if not 0.0 <= gate <= 1.0:
        raise SelectError("--gate must be in [0, 1], got %r" % gate)
    if not 0.0 <= other_only_frac <= 1.0:
        raise SelectError("--other-only-frac must be in [0, 1], got %r" % other_only_frac)
    t0 = time.time()
    verified_path, crops_path = Path(V.VERIFIED_MANIFEST), Path(V.CROPS)
    for p in (verified_path, crops_path, Path(V.POOL_VERDICTS), Path(V.ADMIT_SUMMARY),
              Path(V.POOL_META), core_manifest, C.NEVER_TRAIN_INDEX):
        if not p.exists():
            raise SelectError("missing input %s (run inc/splits.py and inc/verify.py first)" % p)
    if lock_check:
        if not C.LOCK_PATH.exists():
            raise SelectError("no %s: lock the splits (inc/splits.py lock) before selecting"
                              % C.LOCK_PATH)
        want = C.read_lock()["manifests"].get("train_core")
        got = C.sha256_file(core_manifest)
        if got != want:
            raise SelectError("train_core manifest %s does not match LOCK.json (%s != %s)"
                              % (core_manifest, got[:12], str(want)[:12]))
    admit = _read_json(V.ADMIT_SUMMARY) or {}
    admit_emb = admit.get("embeddings") or {}
    if not admit_emb.get("nshards"):
        raise SelectError("%s does not record the embedding shards admit judged with; rerun "
                          "verify admit" % V.ADMIT_SUMMARY)
    nshards = int(admit_emb["nshards"])
    # the shard set admit used (verify's own name pattern), not chunks of a running embed
    shards = sorted(n for n in os.listdir(V.EMB_DIR) if re.fullmatch(EMB_SHARD_RE % nshards, n)) \
        if Path(V.EMB_DIR).is_dir() else []
    inputs = {"verified": _file_info(verified_path), "train_core": _file_info(core_manifest),
              "crops": _file_info(crops_path), "pool_verdicts": _file_info(V.POOL_VERDICTS),
              "pool_meta": _file_info(V.POOL_META), "never_train": _file_info(C.NEVER_TRAIN_INDEX),
              "emb_files": [_file_info(Path(V.EMB_DIR) / n) for n in shards]}
    if (admit.get("verified_sha256") != inputs["verified"]["sha256"]
            or admit.get("crops_sha256") != inputs["crops"]["sha256"]):
        raise SelectError("%s does not name this verified.jsonl and crops.csv: they changed "
                          "after verify admit; rerun verify admit" % V.ADMIT_SUMMARY)
    params = {"base_frac": float(base_frac), "seed": int(seed), "cap_mult": float(cap_mult),
              "refill": bool(refill), "gate": float(gate), "other_only_frac": float(other_only_frac),
              "k1": [K1_PER, K1_MIN, K1_MAX], "k2": [K2_PER, K2_MAX],
              "n_init": N_INIT, "max_iter": MAX_ITER, "kmeans_threads": KMEANS_THREADS,
              "other_sample": OTHER_SAMPLE, "other_k_max": OTHER_K_MAX, "proto_seed": PROTO_SEED,
              "near_dup_bits": _near_dup_bits(), "versions": _versions()}
    outputs = (BASE_SELECTED, BASE_B, POOL, CLUSTERS)
    old = _read_json(out_dir / SUMMARY)
    if old is not None:
        same = old.get("params") == params and old.get("inputs") == inputs
        if same and not force and _outputs_match(old, out_dir, outputs):
            log("up to date: %s (same inputs and parameters)" % (out_dir / SUMMARY))
            return old
        if not force:
            raise SelectError("%s exists from a different build (inputs, parameters or outputs "
                              "differ); an experiment may train on it. Pass --force to "
                              "overwrite." % (out_dir / SUMMARY))
    elif not force and any((out_dir / n).exists() for n in outputs):
        raise SelectError("outputs exist in %s without a summary; pass --force" % out_dir)

    pool = sorted(C.read_manifest(verified_path), key=lambda r: r["key"])
    core = sorted(C.read_manifest(core_manifest), key=lambda r: r["key"])
    _check_rows(pool, "verified")
    _check_rows(core, "train_core")
    check_pool_against_core(pool, core, never_train)
    n = len(pool)
    log("verified pool: %d images; train_core: %d images" % (n, len(core)))
    pool_index = {r["key"]: i for i, r in enumerate(pool)}
    core_keys = {r["key"] for r in core}

    # never-train guard and near-dup groups, from verify's dHashes
    dh = read_pool_dhash(V.POOL_META, [r["key"] for r in pool])
    guard = _load_guard()
    n_checked = guard_rows(guard, pool, dh, "verified pool")
    group = dup_groups([dh[r["key"]] for r in pool])
    ginfo = _group_info(group)
    log("never-train guard: %d images clear of %d evaluation images; near-dup groups: %s"
        % (n_checked, guard.n, ginfo))
    if ginfo["groups_over_%d" % LARGE_GROUP]:
        log("WARNING: %d near-dup group(s) hold more than %d images (largest %d): likely chains "
            "of false matches (letterboxed exports); each stays on one side"
            % (ginfo["groups_over_%d" % LARGE_GROUP], LARGE_GROUP, ginfo["largest_group"]))

    try:
        crops = V.Crops(crops_path)
        E, emb_info = V.load_embeddings(crops, nshards)
    except V.VerifyError as e:
        raise SelectError("verify inputs: %s" % e)
    if emb_info != admit_emb:
        raise SelectError("the embedding shards are not the ones verify admit judged with "
                          "(admit %s, now %s); rerun verify admit" % (admit_emb, emb_info))
    verdict, codes = read_pool_verdicts(V.POOL_VERDICTS, crops)
    J, crop_info = crop_index(crops, pool_index, core_keys, verdict, codes, E, V)
    lab = crops.label
    src_index = {s: i for i, s in enumerate(J["sources"])}
    del crops, verdict
    crop_info["embeddings"] = emb_info
    log("crops: %d train_core, %d of admitted images, %d source evidence, %d other_ok "
        "(embeddings: %s)" % (len(J["core"]), len(J["pool"]), len(J["src"]), len(J["other"]),
                              emb_info))
    F, has = image_features(n, E, J["pool"], J["pool_img"])
    img_src = np.array([src_index.get(r["source"], -1) for r in pool], dtype=np.int64)
    img_fold = np.array([fold_of(r["key"]) for r in pool], dtype=np.int64)
    kind, cosine, score, typ, ret_info = retrieval(n, E, lab, J, img_src, img_fold, gate)
    del E
    log("features: %d/%d images have one (dim %d); %d scored on their species boxes, %d "
        "OtherPlant-only on their source; train_core images passing the gate %.3f"
        % (int(has.sum()), n, ret_info["dim"], int((kind == 1).sum()), int((kind == 2).sum()),
           ret_info["train_core_images_passing_gate"] or 0.0))

    counts = box_counts(pool, workers, "verified")
    core_counts = box_counts(core, workers, "train_core").sum(0)
    caps = [int(math.floor(cap_mult * core_counts[k] + 1e-9)) for k in range(N_SPECIES)]
    other_only = counts[:, :N_SPECIES].sum(1) == 0

    fidx = np.flatnonzero(has)
    l1 = np.full(n, -1, dtype=np.int64)
    l2 = np.full(n, -1, dtype=np.int64)
    if len(fidx):
        a, b = hierarchical_kmeans(F[fidx], seed)
        l1[fidx], l2[fidx] = a, b
    del F
    log("clusters: %d level-1, %d sub-clusters" % (
        len(np.unique(l1[fidx])), len({(int(x), int(y)) for x, y in zip(l1[fidx], l2[fidx])})))

    # units: near-dup groups
    U = int(group.max()) + 1 if n else 0
    usize = np.bincount(group, minlength=U)
    ucounts = np.zeros((U, C.NC), dtype=np.int64)
    np.add.at(ucounts, group, counts)
    uhas = np.bincount(group, weights=has.astype(np.float64), minlength=U) > 0
    uscore = np.full(U, np.inf)
    np.minimum.at(uscore, group[has], score[has])          # NaN (no evidence) propagates
    uscore[~uhas] = np.nan
    utyp = np.full(U, np.inf)
    np.minimum.at(utyp, group[has], np.where(np.isnan(typ), -2.0, typ)[has])
    utyp[~uhas] = -2.0
    rep = np.full(U, n, dtype=np.int64)
    np.minimum.at(rep, group[has], fidx)
    ul1 = np.where(uhas, l1[np.minimum(rep, max(n - 1, 0))], -1)
    ul2 = np.where(uhas, l2[np.minimum(rep, max(n - 1, 0))], -1)
    uother = ucounts[:, :N_SPECIES].sum(1) == 0
    with np.errstate(invalid="ignore"):
        eligible = uhas & np.isfinite(uscore) & (uscore >= gate)
    sort_typ = utyp

    def by_score(ix):
        return ix[np.lexsort((ix, -sort_typ[ix], -uscore[ix]))]

    rng = np.random.default_rng(int(seed))
    order = balanced_order(np.flatnonzero(eligible), ul1, ul2, by_score, rng)
    q = int(round(base_frac * n))
    res = choose(order, q, ucounts, caps, usize, uother, other_only_frac, refill)
    usel, ustatus = res["sel"], res["status"]
    urank = np.full(U, -1, dtype=np.int64)
    urank[order] = np.arange(len(order))
    sel = usel[group]
    rank = urank[group]

    status = []
    for i in range(n):
        g = int(group[i])
        if not uhas[g]:
            status.append("no_feature")
        elif not np.isfinite(uscore[g]):
            status.append("no_evidence")
        elif not eligible[g]:
            status.append("below_gate")
        elif usel[g]:
            status.append("refilled" if ustatus.get(g) == "refilled" else "selected")
        else:
            status.append(ustatus.get(g, "pool"))
    selected = [pool[i] for i in range(n) if sel[i]]
    rest = [pool[i] for i in range(n) if not sel[i]]
    if len(selected) + len(rest) != n or {r["key"] for r in selected} & {r["key"] for r in rest}:
        raise SelectError("internal: selected and pool do not partition the verified set")
    split = np.bincount(group, weights=sel.astype(np.float64), minlength=U)
    if ((split > 0) & (split < usize)).any():
        raise SelectError("internal: a near-dup group is split between base and pool")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    out[BASE_SELECTED] = {"sha256": C.write_manifest(out_dir / BASE_SELECTED, selected),
                          "rows": len(selected)}
    out[BASE_B] = {"sha256": C.write_manifest(out_dir / BASE_B, core + selected),
                   "rows": len(core) + len(selected)}
    out[POOL] = {"sha256": C.write_manifest(out_dir / POOL, rest), "rows": len(rest)}
    sp_boxes = counts[:, :N_SPECIES].sum(1)
    crow = [(pool[i]["key"], int(group[i]), int(l1[i]), int(l2[i]),
             ("", "cwd12", "other")[int(kind[i])], _fmt(cosine[i]), _fmt(score[i]), _fmt(typ[i]),
             int(sp_boxes[i]), int(counts[i, C.OTHER_PLANT]), int(rank[i]), status[i])
            for i in range(n)]
    out[CLUSTERS] = {"sha256": _write_clusters(out_dir / CLUSTERS, crow), "rows": n}
    for name in out:
        out[name]["path"] = str(out_dir / name)

    st = np.array(status)
    picked_img = np.isin(st, ("selected", "dropped_cap", "dropped_other_budget"))
    level1 = []
    for c in (np.unique(l1[fidx]) if len(fidx) else []):
        m = l1 == c
        level1.append({"id": int(c), "images": int(m.sum()),
                       "sub_clusters": int(len(np.unique(l2[m]))),
                       "below_gate": int((m & (st == "below_gate")).sum()),
                       "picked": int((m & picked_img).sum()),
                       "selected": int((m & sel).sum()),
                       "mean_score": _mean(score[m & np.isfinite(score)])})
    sel_counts = counts[sel].sum(0)
    n_oo = int((other_only & sel).sum())
    gated_by_source = collections.Counter(pool[i]["source"] for i in np.flatnonzero(
        st == "below_gate"))
    summary = {
        "step": "INC Step 1.4-1.5 base selection (inc/select.py)",
        "created": _now(),
        "params": params,
        "inputs": inputs,
        "outputs": out,
        "sizes": {"verified": n, "with_feature": int(has.sum()),
                  "no_feature": int((st == "no_feature").sum()),
                  "no_evidence": int((st == "no_evidence").sum()),
                  "below_gate": int((st == "below_gate").sum()),
                  "eligible": int(eligible[group].sum()), "quota": q,
                  "dropped_by_cap": int((st == "dropped_cap").sum()),
                  "dropped_other_budget": int((st == "dropped_other_budget").sum()),
                  "passed_over_other_budget": int((st == "other_budget").sum()),
                  "refilled": int((st == "refilled").sum()),
                  "selected": len(selected), "increment_pool": len(rest),
                  "train_core": len(core), "base_B": len(core) + len(selected),
                  "selected_frac_of_verified": round(len(selected) / n, 6) if n else 0.0},
        "boxes": {"train_core": _species_dict(core_counts),
                  "selected": _species_dict(sel_counts),
                  "increment_pool": _species_dict(counts[~sel].sum(0)),
                  "base_B": _species_dict(core_counts + sel_counts),
                  "caps": {C.CLASS_NAMES[k]: caps[k] for k in range(N_SPECIES)},
                  "capped_species": [C.CLASS_NAMES[k] for k in range(N_SPECIES)
                                     if res["picked_boxes"][k] > caps[k]]},
        "otherplant": {"only_images_budget_frac": float(other_only_frac),
                       "only_images_verified": int(other_only.sum()),
                       "only_images_eligible": int((other_only & eligible[group]).sum()),
                       "only_images_selected": n_oo,
                       "only_images_share_of_selected": (round(n_oo / len(selected), 6)
                                                         if selected else 0.0),
                       "boxes_share_of_selected": (round(float(sel_counts[C.OTHER_PLANT])
                                                         / max(1, int(sel_counts.sum())), 6))},
        "near_dup": ginfo,
        "never_train": {"index": str(C.NEVER_TRAIN_INDEX),
                        "sha256": inputs["never_train"]["sha256"],
                        "evaluation_images": guard.n, "checked": n_checked, "hits": 0},
        "sources": {"train_core": _sources(core), "selected": _sources(selected),
                    "increment_pool": _sources(rest),
                    "below_gate": dict(sorted(gated_by_source.items()))},
        "clusters": {"k1_target": k1_for(len(fidx)) if len(fidx) else 0,
                     "level1": len(level1),
                     "sub_clusters": int(sum(c["sub_clusters"] for c in level1)),
                     "per_level1": level1},
        "retrieval": {"scored_on_species": int((kind == 1).sum()),
                      "scored_on_source": int(((kind == 2) & np.isfinite(score)).sum()),
                      "mean_selected": _mean(score[sel & np.isfinite(score)]),
                      "mean_increment_pool": _mean(score[~sel & np.isfinite(score)]),
                      **ret_info},
        "crops": crop_info,
        "seconds": round(time.time() - t0, 1),
    }
    _atomic_json(out_dir / SUMMARY, summary)
    log("base B: %d train_core + %d selected (quota %d; %d below the gate, %d dropped by caps, "
        "%d by the OtherPlant-only budget, %d refilled; OtherPlant-only %.3f of selected); "
        "increment pool %d; %s"
        % (len(core), len(selected), q, summary["sizes"]["below_gate"],
           summary["sizes"]["dropped_by_cap"], summary["sizes"]["dropped_other_budget"],
           summary["sizes"]["refilled"], summary["otherplant"]["only_images_share_of_selected"],
           len(rest), out_dir / SUMMARY))
    return summary


def _mean(a):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    return round(float(a.mean()), 6) if a.size else None


# ------------------------------------------------------------- increments
def draw_parts(lists, sizes, n, m):
    """n parts of exactly m images from per-cluster unit lists (cluster_lists):
    units are drawn in round-robin order until n * m images (a unit that would
    overshoot is passed over), then dealt cluster by cluster, each unit to the
    emptiest part it fits in (ties: the lowest part). A unit that fits in no
    part is left undrawn and replaced by the next ones. Returns (parts,
    left_out) as lists of unit ids."""
    import numpy as np
    order = round_robin(lists)
    need = n * m
    drawn, total, pos, left_out = set(), 0, 0, []
    while True:
        while total < need and pos < len(order):
            u = int(order[pos])
            pos += 1
            if total + int(sizes[u]) <= need:
                drawn.add(u)
                total += int(sizes[u])
        if total < need:
            raise SelectError("%d increments of %d images need %d, the draw found %d"
                              % (n, m, need, total))
        fill = np.zeros(n, dtype=np.int64)
        parts, failed = [[] for _ in range(n)], []
        for L in lists:
            for u in L:
                u = int(u)
                if u not in drawn:
                    continue
                room = np.flatnonzero(fill + int(sizes[u]) <= m)
                if not len(room):
                    failed.append(u)
                    continue
                j = int(room[np.argmin(fill[room])])
                parts[j].append(u)
                fill[j] += int(sizes[u])
        if not failed:
            return parts, left_out
        for u in failed:
            drawn.discard(u)
            total -= int(sizes[u])
        left_out.extend(failed)


def load_relevance(path, build_summary):
    """inc/relevance.py's file, checked against this select build (relevance.load)."""
    from . import relevance as REL
    try:
        return REL.load(path, build_summary)
    except REL.RelevanceError as e:
        raise SelectError("relevance: %s" % e)


def apply_relevance(pool, rel, cl):
    """(the pool rows of passing sources, the record): every increment-pool
    image of a source whose relevance status is not 'pass' leaves the draw.
    The file must hold every pool source, with the pool's image counts."""
    have = collections.Counter(r["source"] for r in pool)
    unknown = sorted(set(have) - set(rel["status"]))
    if unknown:
        raise SelectError("relevance: %d increment-pool source(s) have no entry in %s, e.g. %s"
                          % (len(unknown), rel["path"], unknown[:3]))
    if dict(have) != rel["images"]:
        diff = sorted(s for s in set(have) | set(rel["images"]) if have.get(s) != rel["images"].get(s))
        raise SelectError("relevance: %s counts other increment-pool images per source than %s holds, "
                          "e.g. %s" % (rel["path"], POOL, diff[:3]))
    keep = [r for r in pool if rel["status"][r["source"]] == "pass"]
    drop = [r for r in pool if rel["status"][r["source"]] != "pass"]
    g_drop = {cl[r["key"]]["dup_group"] for r in drop}
    split = len({cl[r["key"]]["dup_group"] for r in keep} & g_drop)
    excluded = collections.Counter(r["source"] for r in drop)
    info = {"path": rel["path"], "sha256": rel["sha256"], "tau": rel["tau"], "rule": rel["rule"],
            "pool_images_before": len(pool), "excluded_images": len(drop),
            "excluded_sources": {s: {"images": int(n), "status": rel["status"][s]}
                                 for s, n in sorted(excluded.items())},
            "passing_sources": sum(1 for s in have if rel["status"][s] == "pass"),
            "near_dup_groups_split": split,
            "source_status": dict(sorted(rel["status"].items()))}
    return keep, info


def _count(x, what):
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise SelectError("evidence: %s is not a box count: %r" % (what, x))
    return x


def load_evidence(build_summary, admit_path=None, min_evidence=MIN_EVIDENCE):
    """Source-level species evidence for this select build (module
    docstring, --sources evidence): per source, the cwd12-species boxes verify
    admit judged 'verified', from admit_summary.json's per_slug verdict counts.
    Returns {path, sha256, min_evidence, rule, verified_boxes (every per_slug
    source), evidenced (those with >= min_evidence), cross_check}."""
    V = _verify()
    if isinstance(min_evidence, bool) or not isinstance(min_evidence, int) or min_evidence < 1:
        raise SelectError("--min-evidence must be an integer >= 1, got %r" % (min_evidence,))
    path = Path(os.path.abspath(str(admit_path or V.ADMIT_SUMMARY)))
    admit = _read_json(path) if path.is_file() else None
    if not isinstance(admit, dict):
        raise SelectError("evidence: cannot read %s (run inc.verify admit)" % path)
    per_slug = admit.get("per_slug")
    if not isinstance(per_slug, dict) or not per_slug:
        raise SelectError("evidence: %s holds no per_slug verdict counts; rerun inc.verify admit" % path)
    inputs = (build_summary or {}).get("inputs") or {}
    for name, field in (("verified", "verified_sha256"), ("crops", "crops_sha256")):
        want = (inputs.get(name) or {}).get("sha256")
        if not want or admit.get(field) != want:
            raise SelectError("evidence: %s and select_summary.json disagree: the admit summary names another %s "
                              "than the one this select build read (%s != %s); rerun select build after verify admit"
                              % (path, "verified.jsonl" if name == "verified" else "crops.csv",
                                 str(admit.get(field))[:12], str(want)[:12]))
    verified = {}
    for s, d in sorted(per_slug.items()):
        boxes = (d or {}).get("boxes") if isinstance(d, dict) else None
        if not isinstance(boxes, dict):
            raise SelectError("evidence: %s per_slug[%r] has no box verdict counts" % (path, s))
        verified[s] = _count(boxes.get(V.VERIFIED, 0), "per_slug[%r] boxes.%s" % (s, V.VERIFIED))
    se = ((build_summary or {}).get("retrieval") or {}).get("source_evidence")
    if se is None:
        cross = {"checked": False,
                 "reason": "select_summary.json records no retrieval.source_evidence to compare with"}
    else:
        if not isinstance(se, dict):
            raise SelectError("evidence: select_summary.json retrieval.source_evidence is not a table")
        bad = []
        for s in sorted(set(verified) | set(se)):
            n_sp = _count((se.get(s) or {}).get("species_crops", 0), "source_evidence[%r].species_crops" % s)
            if s not in verified:
                bad.append("%s: %d species crop(s) in select_summary.json, no per_slug entry in %s"
                           % (s, n_sp, path.name))
            elif verified[s] > n_sp:
                bad.append("%s: %d verified box(es) in %s, %d species crop(s) in select_summary.json"
                           % (s, verified[s], path.name, n_sp))
        if bad:
            raise SelectError("evidence: %s and select_summary.json disagree for %d source(s) (a verified box is a "
                              "species crop that select counts as source evidence, so every source needs "
                              "species_crops >= verified boxes): %s" % (path, len(bad), "; ".join(bad[:5])))
        cross = {"checked": True, "sources_compared": len(set(verified) | set(se)),
                 "rule": "select_summary.json retrieval.source_evidence species_crops >= admit_summary.json "
                         "per_slug verified boxes for every source, and every source it counts has a per_slug "
                         "entry"}
    return {"path": str(path), "sha256": C.sha256_file(path), "min_evidence": int(min_evidence),
            "rule": EVIDENCE_RULE, "verified_boxes": verified,
            "evidenced": {s: n for s, n in verified.items() if n >= min_evidence}, "cross_check": cross}


def apply_evidence(pool, ev, cl):
    """(the pool rows of evidenced sources, the record): every increment-pool
    image of a source with fewer than min_evidence verified cwd12-species
    boxes leaves the draw. Every pool source must have a per_slug entry."""
    have = collections.Counter(r["source"] for r in pool)
    unknown = sorted(set(have) - set(ev["verified_boxes"]))
    if unknown:
        raise SelectError("evidence: %d increment-pool source(s) have no per_slug entry in %s, e.g. %s"
                          % (len(unknown), ev["path"], unknown[:3]))
    keep = [r for r in pool if r["source"] in ev["evidenced"]]
    drop = [r for r in pool if r["source"] not in ev["evidenced"]]
    g_drop = {cl[r["key"]]["dup_group"] for r in drop}
    split = len({cl[r["key"]]["dup_group"] for r in keep} & g_drop)
    excluded = collections.Counter(r["source"] for r in drop)
    info = {"criterion": SOURCES_EVIDENCE, "rule": ev["rule"], "min_evidence": ev["min_evidence"],
            "admit_summary": {"path": ev["path"], "sha256": ev["sha256"]},
            "pool_images_before": len(pool), "excluded_images": len(drop), "kept_images": len(keep),
            "evidenced_sources": {s: {"verified_boxes": n, "pool_images": int(have.get(s, 0))}
                                  for s, n in sorted(ev["evidenced"].items())},
            "excluded_sources": {s: {"images": int(k), "verified_boxes": ev["verified_boxes"][s]}
                                 for s, k in sorted(excluded.items())},
            "near_dup_groups_split": split,
            "cross_check": ev["cross_check"]}
    return keep, info


def other_heavy_units(g_img, n_sp, n_ot, n_units):
    """Mask over near-dup groups: OtherPlant-heavy (at least OTHER_HEAVY_MIN of
    the group's boxes OtherPlant). g_img: each image's group; n_sp / n_ot: its
    species / OtherPlant boxes."""
    import numpy as np
    usp = np.bincount(g_img, weights=n_sp, minlength=n_units)
    uot = np.bincount(g_img, weights=n_ot, minlength=n_units)
    return (uot > 0) & (uot >= OTHER_HEAVY_MIN * (usp + uot))


def pool_capacity(pool, cl):
    """What the draw can take from these increment-pool rows: every image for
    the regular increments ("images"), and the images of OtherPlant-heavy
    near-dup groups, by the draw's own rule, for the OtherPlant-heavy one
    ("other_heavy_images"). Necessary, not sufficient: the draw keeps groups
    whole, so it can still fall short by less than a group."""
    import numpy as np
    if not pool:
        return {"images": 0, "other_heavy_images": 0}
    g_img = canonical_labels(np.array([int(cl[r["key"]]["dup_group"]) for r in pool], dtype=np.int64))
    n_sp = np.array([int(cl[r["key"]]["species_boxes"]) for r in pool], dtype=np.int64)
    n_ot = np.array([int(cl[r["key"]]["other_boxes"]) for r in pool], dtype=np.int64)
    n_units = int(g_img.max()) + 1
    heavy = other_heavy_units(g_img, n_sp, n_ot, n_units)
    return {"images": len(pool),
            "other_heavy_images": int(np.bincount(g_img, minlength=n_units)[heavy].sum())}


def increments(exp, n_inc, size=None, other_heavy=False, base_dir=None, exp_dir=None,
               workers=WORKERS, force=False, relevance=None, sources=SOURCES_RELEVANCE, min_evidence=None,
               admit_summary=None):
    import numpy as np
    if not exp or "/" in exp or exp.startswith("."):
        raise SelectError("bad experiment name %r" % exp)
    if n_inc < 1 or (size is not None and size < 1):
        raise SelectError("--n and --size must be >= 1")
    if sources not in SOURCE_MODES:
        raise SelectError("--sources must be one of %s, got %r" % (SOURCE_MODES, sources))
    if sources == SOURCES_EVIDENCE and relevance is not None:
        raise SelectError("--sources evidence does not take --relevance: the source-evidence criterion replaces "
                          "the relevance filter")
    if sources == SOURCES_RELEVANCE and (min_evidence is not None or admit_summary is not None):
        raise SelectError("--min-evidence and --admit-summary apply to --sources evidence only")
    base_dir = Path(base_dir or step1_dir())
    exp_dir = Path(exp_dir or (C.INC_DIR / exp))
    man_dir = exp_dir / "manifests"
    build_summary = _read_json(base_dir / SUMMARY)
    if build_summary is None:
        raise SelectError("no %s in %s: run `select build` first" % (SUMMARY, base_dir))
    if not _outputs_match(build_summary, base_dir, (POOL, CLUSTERS)):
        raise SelectError("%s or %s changed since the build that wrote %s"
                          % (POOL, CLUSTERS, base_dir / SUMMARY))
    meta_info = build_summary["inputs"]["pool_meta"]
    if not Path(meta_info["path"]).exists() or C.sha256_file(meta_info["path"]) != meta_info["sha256"]:
        raise SelectError("%s changed since the build; rerun `select build`" % meta_info["path"])
    if not C.NEVER_TRAIN_INDEX.exists():
        raise SelectError("missing never-train index %s" % C.NEVER_TRAIN_INDEX)
    rel = load_relevance(relevance, build_summary) if relevance is not None else None
    ev = (load_evidence(build_summary, admit_summary, MIN_EVIDENCE if min_evidence is None else min_evidence)
          if sources == SOURCES_EVIDENCE else None)
    if size is None:
        size = max(1, int(round(INC_FRAC * build_summary["sizes"]["base_B"])))
    seed = C.stable_int(exp)
    params = {"exp": exp, "n": int(n_inc), "size": int(size), "seed": seed,
              "other_heavy": bool(other_heavy), "other_heavy_min": OTHER_HEAVY_MIN,
              "pool_sha256": build_summary["outputs"][POOL]["sha256"],
              "clusters_sha256": build_summary["outputs"][CLUSTERS]["sha256"],
              "never_train_sha256": C.sha256_file(C.NEVER_TRAIN_INDEX)}
    if rel is not None:                  # absent without a filter: an unfiltered draw's params as before
        params["relevance_sha256"] = rel["sha256"]
    if ev is not None:                   # absent in the default mode: its params as before
        params.update(sources=SOURCES_EVIDENCE, min_evidence=ev["min_evidence"], admit_summary_sha256=ev["sha256"])
    names = [INC_NAME % (j + 1) for j in range(n_inc)] + ([INC_OTHER] if other_heavy else [])
    old = _read_json(man_dir / INC_SUMMARY)
    if old is not None:
        if old.get("params") == params and not force and _outputs_match(old, man_dir, names):
            log("up to date: %s" % (man_dir / INC_SUMMARY))
            return old
        if not force:
            raise SelectError("%s exists from a different draw; an experiment may train on "
                              "it. Pass --force to overwrite." % (man_dir / INC_SUMMARY))
    elif not force and any((man_dir / nm).exists() for nm in names):
        raise SelectError("increment manifests exist in %s without a summary; pass --force"
                          % man_dir)

    pool = sorted(C.read_manifest(base_dir / POOL), key=lambda r: r["key"])
    cl = read_clusters(base_dir / CLUSTERS)
    missing = [r["key"] for r in pool if r["key"] not in cl]
    if missing:
        raise SelectError("%d pool image(s) have no cluster row, e.g. %s" % (len(missing), missing[:3]))
    rel_info = None
    if rel is not None:
        pool, rel_info = apply_relevance(pool, rel, cl)
        log("relevance filter %s: %d of %d increment-pool images leave the draw (%d source(s) not passing), "
            "%d stay" % (rel_info["path"], rel_info["excluded_images"], rel_info["pool_images_before"],
                         len(rel_info["excluded_sources"]), len(pool)))
    ev_info = None
    if ev is not None:
        pool, ev_info = apply_evidence(pool, ev, cl)
        log("source evidence (>= %d verified cwd12-species box(es), %s): %d of %d increment-pool images leave the "
            "draw (%d source(s) without it), %d stay from %d evidenced source(s)"
            % (ev_info["min_evidence"], ev_info["admit_summary"]["path"], ev_info["excluded_images"],
               ev_info["pool_images_before"], len(ev_info["excluded_sources"]), len(pool),
               sum(1 for e in ev_info["evidenced_sources"].values() if e["pool_images"])))
    total = (n_inc + int(other_heavy)) * size
    if total > len(pool):
        raise SelectError("%d increments of %d images need %d, the increment pool holds %d%s"
                          % (n_inc + int(other_heavy), size, total, len(pool),
                             " after the relevance filter" if rel is not None else
                             " after the source-evidence filter" if ev is not None else ""))
    g_img = canonical_labels(np.array([int(cl[r["key"]]["dup_group"]) for r in pool],
                                      dtype=np.int64))
    l1 = np.array([int(cl[r["key"]]["l1"]) for r in pool], dtype=np.int64)
    l2 = np.array([max(0, int(cl[r["key"]]["l2"])) for r in pool], dtype=np.int64)
    n_sp = np.array([int(cl[r["key"]]["species_boxes"]) for r in pool], dtype=np.int64)
    n_ot = np.array([int(cl[r["key"]]["other_boxes"]) for r in pool], dtype=np.int64)
    U = int(g_img.max()) + 1 if len(pool) else 0
    members = [[] for _ in range(U)]
    for i, g in enumerate(g_img):
        members[int(g)].append(i)
    usize = np.array([len(x) for x in members], dtype=np.int64)
    # a group sits in the clusters of its first member with a feature; images
    # without a feature (l1 = -1) form one more level-1 cluster
    ul1 = np.full(U, -1, dtype=np.int64)
    ul2 = np.zeros(U, dtype=np.int64)
    for u, mem in enumerate(members):
        f = [i for i in mem if l1[i] >= 0]
        if f:
            ul1[u], ul2[u] = l1[f[0]], l2[f[0]]
    heavy = other_heavy_units(g_img, n_sp, n_ot, U)
    avail = np.ones(U, dtype=bool)
    plan = []                                        # (name, kind, unit ids)
    left_out = []
    if other_heavy:
        rng_o = np.random.default_rng(C.stable_int(exp + "/otherplant"))
        idx = np.flatnonzero(heavy)
        if usize[idx].sum() < size:
            raise SelectError("an OtherPlant-heavy increment of %d images needs %d, the pool "
                              "holds %d" % (size, size, int(usize[idx].sum())))
        (part,), lo = draw_parts(cluster_lists(idx, ul1, ul2, rng_o.permutation, rng_o),
                                 usize, 1, size)
        avail[part] = False
        left_out += lo
        plan.append((INC_OTHER, "otherplant_heavy", part))
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(avail)
    if usize[idx].sum() < n_inc * size:
        raise SelectError("%d increments of %d images need %d, the pool holds %d after the "
                          "OtherPlant-heavy increment" % (n_inc, size, n_inc * size,
                                                          int(usize[idx].sum())))
    parts, lo = draw_parts(cluster_lists(idx, ul1, ul2, rng.permutation, rng), usize, n_inc, size)
    left_out += lo
    plan = [(INC_NAME % (j + 1), "verified", p) for j, p in enumerate(parts)] + plan
    rows_of = {nm: [pool[i] for u in part for i in members[u]] for nm, _k, part in plan}
    if any(len(r) != size for r in rows_of.values()):
        raise SelectError("internal: increments of sizes %s, not %d"
                          % ([len(r) for r in rows_of.values()], size))
    drawn_rows = [r for rows in rows_of.values() for r in rows]
    if len({r["key"] for r in drawn_rows}) != len(drawn_rows):
        raise SelectError("internal: increments overlap")
    dh = read_pool_dhash(meta_info["path"], [r["key"] for r in drawn_rows])
    guard = _load_guard()
    guard_rows(guard, drawn_rows, dh, "increments")
    counts = box_counts(drawn_rows, workers, "increments")
    cnt = {r["key"]: counts[t] for t, r in enumerate(drawn_rows)}
    img_of = {r["key"]: i for i, r in enumerate(pool)}

    man_dir.mkdir(parents=True, exist_ok=True)
    if old is not None:
        for stale in old.get("outputs", {}):
            if stale not in names and (man_dir / stale).exists():
                os.remove(man_dir / stale)
    outs, per = {}, []
    for nm, kind, part in plan:
        rows = rows_of[nm]
        outs[nm] = {"sha256": C.write_manifest(man_dir / nm, rows), "rows": len(rows),
                    "path": str(man_dir / nm)}
        box = sum((cnt[r["key"]] for r in rows), np.zeros(C.NC, dtype=np.int64))
        ii = np.array([img_of[r["key"]] for r in rows], dtype=np.int64)
        per.append({"manifest": nm, "kind": kind, "images": len(rows),
                    "boxes": _species_dict(box),
                    "otherplant_box_share": round(float(box[C.OTHER_PLANT]) / max(1, int(box.sum())), 6),
                    "sources": _sources(rows),
                    "level1_clusters": int(len(np.unique(l1[ii]))),
                    "no_feature_images": int((l1[ii] < 0).sum()),
                    "near_dup_groups": len(part),
                    "multi_image_groups": int((usize[np.array(part, dtype=np.int64)] > 1).sum())})
    summary = {"step": "INC increments (inc/select.py increments)", "created": _now(),
               "params": params, "build_summary": str(base_dir / SUMMARY),
               "pool_images": len(pool), "drawn": len(drawn_rows),
               "left_in_pool": len(pool) - len(drawn_rows),
               "pool_level1_clusters": int(len(np.unique(l1))),
               "groups_left_out_for_size": len(left_out),
               "never_train": {"index": str(C.NEVER_TRAIN_INDEX), "checked": len(drawn_rows),
                               "hits": 0},
               "unverified_source_increment": "not drawn here: its images are the ones verify "
                                              "did not admit (see the module docstring)",
               "relevance": rel_info,
               "outputs": outs, "increments": per}
    if ev_info is not None:              # absent in the default mode: its summary as before
        summary.update(increment_sources=SOURCES_EVIDENCE, evidence=ev_info)
    _atomic_json(man_dir / INC_SUMMARY, summary)
    log("%d increments of %d images%s from a pool of %d -> %s"
        % (n_inc, size, " + 1 OtherPlant-heavy" if other_heavy else "", len(pool), man_dir))
    return summary


# -------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc.select",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="choose base B's harvested images; write the increment pool")
    b.add_argument("--base-frac", type=float, default=BASE_FRAC,
                   help="fraction of the verified pool that goes to base B (default %(default)s)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--cap-mult", type=float, default=CAP_MULT,
                   help="per-species cap as a multiple of train_core's boxes (default %(default)s)")
    b.add_argument("--gate", type=float, default=GATE,
                   help="retrieval gate on the train_core percentile scale (default %(default)s)")
    b.add_argument("--other-only-frac", type=float, default=OTHER_ONLY_FRAC,
                   help="most OtherPlant-only images, as a share of the selected images "
                        "(default %(default)s)")
    b.add_argument("--refill", action="store_true",
                   help="after the cap drops, refill toward the quota within the caps")
    b.add_argument("--out-dir", default=None, help="default: INC_DIR/step1, next to the inputs")
    b.add_argument("--workers", type=int, default=WORKERS)
    b.add_argument("--force", action="store_true", help="overwrite a different earlier build")
    i = sub.add_parser("increments", help="draw N fixed-size increments from the increment pool")
    i.add_argument("--exp", required=True)
    i.add_argument("--n", type=int, required=True)
    i.add_argument("--size", type=int, default=None,
                   help="images per increment (default: %d%% of base B's images)"
                        % round(100 * INC_FRAC))
    i.add_argument("--other-heavy", action="store_true",
                   help="also draw one OtherPlant-heavy increment of the same size")
    i.add_argument("--base-dir", default=None, help="where build wrote (default INC_DIR/step1)")
    i.add_argument("--sources", choices=SOURCE_MODES, default=SOURCES_RELEVANCE,
                   help="the relevance criterion: 'relevance' (default: --relevance if given, else no filter, "
                        "as before) or 'evidence' (only sources with --min-evidence verified cwd12-species boxes)")
    i.add_argument("--relevance", default=None,
                   help="inc.relevance's relevance.json for this build: draw only from sources that "
                        "pass it (default: no filter)")
    i.add_argument("--min-evidence", type=int, default=None,
                   help="--sources evidence: verified cwd12-species boxes a source needs (default %d)"
                        % MIN_EVIDENCE)
    i.add_argument("--admit-summary", default=None,
                   help="--sources evidence: verify admit's admit_summary.json (default: verify's own path)")
    i.add_argument("--workers", type=int, default=WORKERS)
    i.add_argument("--force", action="store_true")
    s = sub.add_parser("summary", help="print the build summary")
    s.add_argument("--base-dir", default=None)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            build(args.base_frac, args.seed, args.cap_mult, args.refill, args.out_dir,
                  workers=args.workers, force=args.force, gate=args.gate,
                  other_only_frac=args.other_only_frac)
        elif args.cmd == "increments":
            increments(args.exp, args.n, args.size, args.other_heavy, args.base_dir,
                       workers=args.workers, force=args.force, relevance=args.relevance, sources=args.sources,
                       min_evidence=args.min_evidence, admit_summary=args.admit_summary)
        else:
            sm = _read_json(Path(args.base_dir or step1_dir()) / SUMMARY)
            if sm is None:
                raise SelectError("no build summary")
            sm = dict(sm)
            sm["clusters"] = {k: v for k, v in sm["clusters"].items() if k != "per_level1"}
            print(json.dumps({k: sm[k] for k in ("sizes", "boxes", "otherplant", "near_dup",
                                                 "never_train", "sources", "clusters",
                                                 "retrieval", "params")}, indent=1))
    except SelectError as e:
        print("inc.select: error: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
