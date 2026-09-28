"""DINOv2 features for the funnel audit (contract docs/FUNNEL_AUDIT.md §4.3,
§6 H6, DEC-9; runner docs/FUNNEL_AUDIT_RUNNER.md §4.12, §5.3.1).

Three kinds of feature file, all written atomically:

  crop shards   FUNNEL_DIR/emb_dinov2/emb_sXXX_of_NNN.npz, in the Step 1
                verifier's shard format (arrays crop_ids int64 and X float16,
                meta a JSON string with crops_sha256, embedder, dim, stats,
                shard, nshards, crops, seconds, built_utc). They cover every
                row of the Step 1 crop table. Crops are cut by the verifier's
                own worker (inc/verify._cut_task: its square crop, grey
                padding, 224 px), so a DINOv2 row and a Step 1 row describe the
                same pixels. Shards resume per chunk of images, and a crop that
                cannot be cut or embedded is a NaN row that shifts no
                neighbour. In an array job, task i builds shard i.
  crop tables   one npz per table in the verifier's CROP_FIELDS format that
                is not the Step 1 crop table (the independent-truth photos,
                refetched images): the same arrays and meta, one file.
  image files   whole-image descriptors for the copy detector (leak.py):
                arrays keys, X, and optionally H (per-image dHash values) and
                extra (JSON per row). The view is the image with its EXIF
                orientation applied, resized so its shorter edge is VIEW_EDGE
                px (bicubic), then the model's own processor (resize and
                centre crop). The resize runs in the worker processes, so a
                32-megapixel field photograph never crosses a process boundary
                at full size.

The model comes from the domain config (judges.features.model and pooling);
the loader and batch embedder are semisup_labeler's (_load_backbone,
_embed_batch), the ones the Step 1 verifier and the existing curator tools
use, so there is one definition of how an image becomes a DINOv2 feature.

Nothing here names a domain. torch, transformers and PIL are imported inside
the functions that need them.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from . import EmbedError, canonical_json, sha256_bytes, utc
from ..inc import common as C
from ..inc import verify as V

SHARD_RE = re.compile(r"emb_s(\d{3})_of_(\d{3})\.npz")
POOLINGS = ("cls", "mean")
VIEW_EDGE = 256                   # shorter edge of the whole-image view, before the processor
VIEW = "exif_transpose; shorter edge %d px (bicubic); model processor (resize, centre crop)" % VIEW_EDGE
TABLE_FORMAT = "funnel-crop-table-features/1"
IMAGE_FORMAT = "funnel-image-descriptors/1"
PROGRESS_EVERY = 2000


def log(msg):
    print("[funnel.embed] %s" % msg, flush=True)


def _raw(domain):
    raw = getattr(domain, "raw", domain)
    if not isinstance(raw, dict):
        raise EmbedError("a domain config (or its raw dict) is required, got %r" % (domain,))
    return raw


def features_config(domain):
    """(model, pooling) of the domain's feature extractor (judges.features)."""
    feats = (_raw(domain).get("judges") or {}).get("features")
    if not isinstance(feats, dict) or not feats.get("model"):
        raise EmbedError("the domain config has no judges.features.model")
    pooling = feats.get("pooling", "cls")
    if pooling not in POOLINGS:
        raise EmbedError("judges.features.pooling %r is not one of %s" % (pooling, POOLINGS))
    if str(feats["model"]).startswith("hf-hub:"):
        raise EmbedError("judges.features.model %r is an open_clip hub model; the funnel features are a "
                         "transformers backbone with %s pooling" % (feats["model"], POOLINGS))
    return str(feats["model"]), pooling


def embedder_name(model, pooling):
    return "%s:%s" % (model, pooling)


class Dinov2Embedder:
    """A transformers self-supervised backbone (default facebook/dinov2-base,
    CLS token) through semisup_labeler's loader and batch embedder. Called
    with a list of PIL images, returns float32 [n, dim]."""

    def __init__(self, model="facebook/dinov2-base", pooling="cls", device=None):
        from PIL import Image
        from .. import semisup_labeler as SL
        if pooling not in POOLINGS:
            raise EmbedError("pooling %r is not one of %s" % (pooling, POOLINGS))
        if str(model).startswith("hf-hub:"):
            raise EmbedError("%s is an open_clip hub model, not a transformers backbone" % model)
        self._SL = SL
        self.model_name = str(model)
        self.pooling = pooling
        self.device = device or SL._device()
        self.name = embedder_name(self.model_name, pooling)
        log("loading %s on %s" % (self.name, self.device))
        self.model, self.proc = SL._load_backbone(self.model_name, self.device)
        grey = Image.new("RGB", (SL.CROP_PX, SL.CROP_PX), (124, 124, 124))
        self.dim = int(self([grey]).shape[1])

    def __call__(self, pils):
        rgb = [p if p.mode == "RGB" else p.convert("RGB") for p in pils]
        out = self._SL._embed_batch(self.model, self.proc, rgb, (self.pooling,), self.device)
        return np.asarray(out[self.pooling], dtype=np.float32)


class LazyEmbedder:
    """A Dinov2Embedder that loads its model on first use. Its name is known
    before loading, so a no-op rerun never loads the model, and a caller that
    forks worker processes first (as the verifier does, before the model is on
    the GPU) gets the model loaded after the fork."""

    def __init__(self, model="facebook/dinov2-base", pooling="cls"):
        self.model_name, self.pooling = str(model), pooling
        self.name = embedder_name(self.model_name, pooling)
        self._emb = None

    def _get(self):
        if self._emb is None:
            self._emb = Dinov2Embedder(self.model_name, self.pooling)
        return self._emb

    @property
    def dim(self):
        return self._get().dim

    def __call__(self, pils):
        return self._get()(pils)


def default_embedder(domain):
    """The domain's feature embedder, loaded on first use."""
    model, pooling = features_config(domain)
    return LazyEmbedder(model, pooling)


# ------------------------------------------------------------ crop tables
class CropTable:
    """A table in the verifier's CROP_FIELDS format with any `set` values (the
    Step 1 crop table has its own reader, verify.Crops; this one reads the
    other tables). crop_id must be 0..n-1 in order. Offers what the
    verifier's embedding helpers use: n, sha, row(i), images()."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise EmbedError("crop table %s not found" % self.path)
        self.sha = C.sha256_file(self.path)
        cols = {f: [] for f in V.CROP_FIELDS}
        with open(self.path, newline="") as fh:
            rd = csv.reader(fh)
            try:
                header = next(rd)
            except StopIteration:
                raise EmbedError("%s is empty" % self.path)
            if tuple(header) != V.CROP_FIELDS:
                raise EmbedError("%s: columns %s, expected %s" % (self.path, header, list(V.CROP_FIELDS)))
            for row in rd:
                if len(row) != len(V.CROP_FIELDS):
                    raise EmbedError("%s: a row with %d fields" % (self.path, len(row)))
                for f, v in zip(V.CROP_FIELDS, row):
                    cols[f].append(v)
        self.n = len(cols["crop_id"])
        try:
            ids = [int(x) for x in cols["crop_id"]]
        except ValueError:
            raise EmbedError("%s: a crop_id is not an integer" % self.path)
        if ids != list(range(self.n)):
            raise EmbedError("%s: crop_id is not 0..n-1 in order" % self.path)
        self.set, self.key, self.image = cols["set"], cols["key"], cols["image"]
        self.source, self.group, self.src_name = cols["source"], cols["group"], cols["src_name"]
        try:
            self.box = np.array(cols["box"], dtype=np.int64)
            for f in ("cx", "cy", "w", "h"):
                setattr(self, f, np.array(cols[f], dtype=np.float64))
            self.W = np.array(cols["W"], dtype=np.int64)
            self.H = np.array(cols["H"], dtype=np.int64)
            self.label = np.array(cols["label"], dtype=np.int64)
        except ValueError as e:
            raise EmbedError("%s: a numeric column does not parse (%s)" % (self.path, e))

    def row(self, i):
        return {"crop_id": int(i), "cx": float(self.cx[i]), "cy": float(self.cy[i]),
                "w": float(self.w[i]), "h": float(self.h[i]), "W": int(self.W[i]), "H": int(self.H[i])}

    def images(self):
        """[(image, [crop ids])] in crop order, as verify.Crops.images."""
        out, idx = [], {}
        for i, img in enumerate(self.image):
            j = idx.get(img)
            if j is None:
                idx[img] = len(out)
                out.append((img, [i]))
            else:
                out[j][1].append(i)
        return out


def _npz_meta(path):
    return V._npz_meta(path)


def _save_npz(path, meta, **arrays):
    V._save_npz(path, meta, **arrays)


def _check_shard_args(shard, nshards):
    try:
        s, n = int(shard), int(nshards)
    except (TypeError, ValueError):
        raise EmbedError("shard and nshards must be integers (got %r, %r)" % (shard, nshards))
    if n < 1 or n > 999 or not 0 <= s < n:
        raise EmbedError("shard %d of %d is out of range" % (s, n))
    return s, n


def embed_crops(crop_table, out_dir, shard, nshards, embedder=None, procs=5, batch=64, chunk_images=2000,
                force=False, model=None, pooling="cls"):
    """Build shard `shard` of `nshards` of the crop table's features in out_dir.

    Images are split into shards exactly as the verifier splits them (the
    crops of one image stay in one shard). A finished shard made from this
    crop table by the same embedder is a no-op; with force it is rebuilt.
    Chunks of chunk_images images are written as part files and skipped on a
    resumed run. Returns the shard's meta."""
    s, n = _check_shard_args(shard, nshards)
    out_dir = Path(out_dir)
    if embedder is not None:
        want_name = embedder.name
    elif model:
        want_name = embedder_name(model, pooling)
    else:
        raise EmbedError("embed_crops needs an embedder or a model name")
    images = crop_table.images()
    final = out_dir / V._shard_name(s, n)
    meta = _npz_meta(final) if final.exists() else None
    if meta and not force and meta.get("crops_sha256") == crop_table.sha and meta.get("embedder") == want_name:
        log("shard %d/%d: done already (%s)" % (s, n, final.name))
        return meta
    if meta:
        log("shard %d/%d: %s was made from another crop table or embedder; rebuilding" % (s, n, final.name))
    lo, hi = s * len(images) // n, (s + 1) * len(images) // n
    mine = images[lo:hi]
    want = np.array([c for _img, cs in mine for c in cs], dtype=np.int64)
    chunk = max(1, int(chunk_images))
    n_parts = (len(mine) + chunk - 1) // chunk
    log("shard %d/%d: %d images, %d crops, %d chunk(s) of %d images" % (s, n, len(mine), len(want), n_parts, chunk))
    holder = {"e": embedder}

    def get_embedder():
        if holder["e"] is None:
            holder["e"] = Dinov2Embedder(model, pooling)
        if holder["e"].name != want_name:
            raise EmbedError("the embedder is %s, the shard wants %s" % (holder["e"].name, want_name))
        return holder["e"]

    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    t0 = time.time()
    done = 0
    # workers first: fork before the model is on the GPU (as the verifier does)
    with V._Workers(procs) as workers:
        for j in range(n_parts):
            part = out_dir / V._shard_name(s, n, j)
            parts.append(part)
            pm = _npz_meta(part) if part.exists() else None
            if (pm and not force and pm.get("crops_sha256") == crop_table.sha
                    and pm.get("chunk_images") == chunk and pm.get("embedder") == want_name):
                continue
            emb = get_embedder()
            t1 = time.time()
            ids, X, st = V._embed_images(mine[j * chunk:(j + 1) * chunk], crop_table, emb, workers, int(batch))
            _save_npz(part, {"crops_sha256": crop_table.sha, "chunk_images": chunk, "embedder": emb.name,
                             "dim": int(emb.dim), "stats": dict(st), "seconds": round(time.time() - t1, 1)},
                      crop_ids=ids, X=X.astype(np.float16))
            done += len(ids)
            log("  shard %d/%d chunk %d/%d: %d crops (%d failed), %.1f crops/s"
                % (s, n, j + 1, n_parts, len(ids), st.get("failed_crops", 0), done / max(time.time() - t0, 1e-6)))
    ids, Xs, names, dims, stats, secs = [], [], set(), set(), collections.Counter(), 0.0
    for part in parts:
        with np.load(part) as d:
            pm = json.loads(str(d["meta"]))
            ids.append(d["crop_ids"])
            Xs.append(d["X"])
        names.add(pm["embedder"])
        dims.add(pm["dim"])
        stats.update(pm.get("stats", {}))
        secs += pm.get("seconds", 0.0)
    if len(names) > 1 or len(dims) > 1:
        raise EmbedError("shard %d/%d: chunks from different embedders %s; rerun with force" % (s, n, sorted(names)))
    ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
    if not np.array_equal(np.sort(ids), np.sort(want)) or len(set(ids.tolist())) != len(ids):
        raise EmbedError("shard %d/%d: the chunks do not cover the shard's crops exactly once" % (s, n))
    if names:
        name, dim = names.pop(), int(dims.pop())
    else:                                      # an empty shard: nothing was embedded
        name, dim = want_name, int(getattr(embedder, "dim", 0) or 0)
    X = np.concatenate(Xs) if Xs else np.zeros((0, dim), dtype=np.float16)
    meta = {"format": "funnel-crop-features/1", "crops_sha256": crop_table.sha, "shard": s, "nshards": n,
            "images": len(mine), "crops": int(len(ids)), "embedder": name, "dim": dim,
            "stats": dict(stats), "seconds": round(secs, 1), "built_utc": utc()}
    _save_npz(final, meta, crop_ids=ids, X=X)
    for part in parts:
        part.unlink()
    log("shard %d/%d: %d crops, %d failed -> %s" % (s, n, len(ids), stats.get("failed_crops", 0), final))
    return meta


def _files_digest(files):
    """sha256 of the canonical JSON {file name: sha256}: one hash for a shard set."""
    return sha256_bytes(canonical_json({Path(p).name: C.sha256_file(p) for p in files}).encode("utf-8"))


def load(crop_table, emb_dir, nshards=None, embedder=None):
    """(X float16 [n, dim], info) from a complete, current set of shards in
    emb_dir: the same checks as the verifier's load_embeddings (every crop id
    exactly once, every shard made from this crop table, one embedder).

    crop_table is a crop table (verify.Crops or CropTable) or its sha256;
    with a bare sha256 the table size is taken from the shards, which must
    then cover 0..n-1 exactly. info records every shard file's sha256 and
    their digest ("sha256"), so a consumer can record the features it read."""
    emb_dir = Path(emb_dir)
    sha = crop_table if isinstance(crop_table, str) else crop_table.sha
    n_rows = None if isinstance(crop_table, str) else int(crop_table.n)
    files = collections.defaultdict(dict)
    if emb_dir.is_dir():
        for name in os.listdir(emb_dir):
            m = SHARD_RE.fullmatch(name)
            if m:
                files[int(m.group(2))][int(m.group(1))] = emb_dir / name
    problems = []
    for n in ([int(nshards)] if nshards else sorted(files)):
        have = files.get(n, {})
        missing = [s for s in range(n) if s not in have]
        metas = {s: _npz_meta(p) for s, p in have.items()}
        stale = [s for s, m in metas.items() if not m or m.get("crops_sha256") != sha]
        if missing or stale:
            problems.append("nshards=%d: missing shards %s, stale shards %s" % (n, missing, stale))
            continue
        filled = [m for m in metas.values() if m.get("crops", 0) > 0]
        names = {m["embedder"] for m in filled}
        dims = {m["dim"] for m in filled}
        if len(names) != 1 or len(dims) != 1:
            problems.append("nshards=%d: shards from different embedders %s" % (n, sorted(names)))
            continue
        name = names.pop()
        if embedder is not None and name != embedder:
            problems.append("nshards=%d: shards by %s, wanted %s" % (n, name, embedder))
            continue
        dim = dims.pop()
        total = sum(int(m.get("crops", 0)) for m in metas.values())
        rows = n_rows if n_rows is not None else total
        X = np.full((rows, dim), np.nan, dtype=np.float16)
        seen = np.zeros(rows, dtype=np.int64)
        failed = 0
        bad_ids = False
        for s in range(n):
            with np.load(have[s]) as d:
                ids = d["crop_ids"]
                if len(ids):
                    if ids.min() < 0 or ids.max() >= rows:
                        bad_ids = True
                        break
                    X[ids] = d["X"]
                    np.add.at(seen, ids, 1)
            failed += int(metas[s].get("stats", {}).get("failed_crops", 0))
        if bad_ids or not (seen == 1).all():
            problems.append("nshards=%d: crop ids not covering 0..%d exactly once" % (n, rows - 1))
            continue
        shard_files = [have[s] for s in range(n)]
        info = {"nshards": n, "embedder": name, "dim": int(dim), "failed_crops": failed,
                "nan_rows": int((~np.isfinite(X).all(axis=1)).sum()), "crops_sha256": sha,
                "path": str(emb_dir), "files": {p.name: C.sha256_file(p) for p in shard_files}}
        info["sha256"] = sha256_bytes(canonical_json(info["files"]).encode("utf-8"))
        return X, info
    raise EmbedError("no complete, current set of feature shards in %s (%s); run embed-judges --stage embed"
                     % (emb_dir, "; ".join(problems) or "none found"))


def embed_table(csv_path, out_path, embedder, procs=1, batch=64, force=False):
    """Features of every row of a CROP_FIELDS table (not the Step 1 crop table)
    in one npz: crop_ids, X float16 (NaN for a crop that failed), meta with
    the table's sha256 and the embedder. A current file is a no-op."""
    table = CropTable(csv_path)
    out_path = Path(out_path)
    meta = _npz_meta(out_path) if out_path.exists() else None
    if (meta and not force and meta.get("crops_sha256") == table.sha
            and meta.get("embedder") == embedder.name):
        log("%s: done already" % out_path.name)
        return meta
    t0 = time.time()
    with V._Workers(procs) as workers:
        ids, X, st = V._embed_images(table.images(), table, embedder, workers, int(batch))
    if len(ids) != table.n or sorted(ids.tolist()) != list(range(table.n)):
        raise EmbedError("%s: the embedded rows do not cover the table exactly once" % table.path)
    order = np.argsort(ids, kind="stable")
    meta = {"format": TABLE_FORMAT, "crops_sha256": table.sha, "table": str(table.path), "crops": int(table.n),
            "images": len(table.images()), "embedder": embedder.name, "dim": int(embedder.dim),
            "stats": dict(st), "seconds": round(time.time() - t0, 1), "built_utc": utc()}
    _save_npz(out_path, meta, crop_ids=ids[order], X=X[order].astype(np.float16))
    log("%s: %d crops, %d failed" % (out_path.name, table.n, st.get("failed_crops", 0)))
    return meta


def load_table(table, path, embedder=None):
    """(X float16 [n, dim], meta) of an embed_table file, checked against the
    table (a CropTable, or its csv path, or its sha256 with n from the file)."""
    path = Path(path)
    if not path.is_file():
        raise EmbedError("%s not found; run embed-judges" % path)
    if isinstance(table, (str, Path)) and Path(str(table)).is_file():
        table = CropTable(table)
    sha = table if isinstance(table, str) else table.sha
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        ids, X = d["crop_ids"], d["X"]
    if meta.get("crops_sha256") != sha:
        raise EmbedError("%s was made from another table (%s, not %s)"
                         % (path.name, str(meta.get("crops_sha256"))[:12], sha[:12]))
    if embedder is not None and meta.get("embedder") != embedder:
        raise EmbedError("%s was made by %s, not %s" % (path.name, meta.get("embedder"), embedder))
    n = int(meta.get("crops", len(ids)))
    if not isinstance(table, str) and n != table.n:
        raise EmbedError("%s holds %d crops, the table %d" % (path.name, n, table.n))
    if sorted(ids.tolist()) != list(range(n)):
        raise EmbedError("%s: crop ids do not cover 0..%d exactly once" % (path.name, n - 1))
    out = np.full((n, X.shape[1]), np.nan, dtype=np.float16)
    out[ids] = X
    meta = dict(meta, sha256=C.sha256_file(path), path=str(path))
    return out, meta


# ------------------------------------------------------------ whole images
def view_small(image):
    """The whole-image view before the processor: RGB, shorter edge VIEW_EDGE
    px (bicubic; long edge rounded)."""
    from PIL import Image
    im = image if image.mode == "RGB" else image.convert("RGB")
    W, H = im.size
    if W < 1 or H < 1:
        raise EmbedError("an image of size %dx%d" % (W, H))
    if W <= H:
        size = (VIEW_EDGE, max(1, int(round(H * VIEW_EDGE / float(W)))))
    else:
        size = (max(1, int(round(W * VIEW_EDGE / float(H)))), VIEW_EDGE)
    if im.size == size:
        return im.copy()
    return im.resize(size, Image.BICUBIC)


def prepare_view(row):
    """The default image preparation: (the EXIF-transposed RGB image, no hashes,
    no extra). A module-level function, so worker processes can run it."""
    from PIL import Image, ImageOps
    with Image.open(row["image"]) as im0:
        im = ImageOps.exif_transpose(im0).convert("RGB")
    return im, None, None


def _image_task(task):
    """Worker: (index, small uint8 array | None, hashes | None, extra | None, error)."""
    idx, prepare, row = task
    try:
        img, hashes, extra = prepare(row)
        small = np.asarray(view_small(img), dtype=np.uint8)
        return idx, small, (None if hashes is None else [int(h) for h in hashes]), extra, None
    except Exception as e:  # noqa: BLE001 - a failed image is a NaN row, never a crash
        return idx, None, None, None, "%s: %s" % (type(e).__name__, e)


def rows_digest(rows):
    """sha256 of the rows an image file was made from (key, image, sha256 and
    any augmentation spec, in order)."""
    body = [[r["key"], str(r["image"]), r.get("sha256") or "", r.get("spec")] for r in rows]
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def _prepare_name(prepare):
    return "%s.%s" % (getattr(prepare, "__module__", "?").rsplit(".", 1)[-1], getattr(prepare, "__qualname__", "?"))


def _embed_rows(rows, embedder, prepare, n_hashes, workers, batch):
    """(X float32 [n, dim], H uint64 [n, n_hashes], hash_ok bool [n], extra list, stats)."""
    from PIL import Image
    n = len(rows)
    dim = int(embedder.dim)
    X = np.full((n, dim), np.nan, dtype=np.float32)
    H = np.zeros((n, max(0, int(n_hashes))), dtype=np.uint64)
    hash_ok = np.zeros(n, dtype=bool)
    extra = [None] * n
    stats = collections.Counter()
    buf, buf_idx = [], []

    def flush():
        F = V._embed_safely(embedder, buf, stats)
        for i, v in zip(buf_idx, F):
            X[i] = v

    tasks = [(i, prepare, rows[i]) for i in range(n)]
    for i, small, hashes, ext, err in workers.imap(_image_task, tasks, chunksize=4):
        if err is not None:
            stats["failed_images"] += 1
            if stats["failed_images"] <= 5:
                log("  WARNING: cannot read %s: %s" % (rows[i].get("image"), err))
            continue
        extra[i] = ext
        if n_hashes:
            if hashes is not None and len(hashes) == n_hashes:
                H[i] = np.array(hashes, dtype=np.uint64)
                hash_ok[i] = True
            else:
                stats["failed_hashes"] += 1
        buf.append(Image.fromarray(small))
        buf_idx.append(i)
        if len(buf) >= batch:
            flush()
            buf, buf_idx = [], []
    if buf:
        flush()
    bad = ~np.isfinite(X).all(axis=1)
    stats["failed_descriptors"] = int(bad.sum())
    return X, H, hash_ok, extra, stats


def _part_path(out_path, j):
    return out_path.with_name("%s.part%05d.npz" % (out_path.name[:-len(".npz")], j))


def _pack(keys, X, H, hash_ok, extra):
    arrays = {"keys": np.array(keys, dtype=str), "X": X.astype(np.float16),
              "extra": np.array([json.dumps(e, sort_keys=True) for e in extra], dtype=str)}
    if H.shape[1]:
        arrays["H"] = H
        arrays["hash_ok"] = hash_ok
    return arrays


def image_ident(rows, embedder, prepare=None, n_hashes=0):
    """The identity of an embed_images file made from `rows`: its rows digest,
    the embedder's name, the preparation function, the hash count and the
    view. A file whose meta holds all of these is current."""
    return {"rows_sha256": rows_digest(rows), "embedder": embedder.name,
            "prepare": _prepare_name(prepare or prepare_view), "n_hashes": int(n_hashes), "view": VIEW}


def image_file_current(path, ident):
    """True when the embed_images file at path exists and carries ident."""
    path = Path(path)
    meta = _npz_meta(path) if path.is_file() else None
    return bool(meta) and all(meta.get(k) == v for k, v in ident.items())


def load_images(path, rows=None):
    """The content of an embed_images file: {"keys", "X" (float16), "H",
    "hash_ok", "extra" (list), "meta", "path", "sha256"}; with rows, the file
    must have been made from exactly those rows."""
    path = Path(path)
    if not path.is_file():
        raise EmbedError("%s not found" % path)
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        out = {"keys": [str(k) for k in d["keys"]], "X": d["X"],
               "extra": [json.loads(str(e)) for e in d["extra"]], "meta": meta,
               "H": d["H"] if "H" in d else np.zeros((len(d["keys"]), 0), dtype=np.uint64),
               "hash_ok": d["hash_ok"] if "hash_ok" in d else np.zeros(len(d["keys"]), dtype=bool)}
    if rows is not None and meta.get("rows_sha256") != rows_digest(rows):
        raise EmbedError("%s was made from other rows" % path.name)
    out["path"] = str(path)
    out["sha256"] = C.sha256_file(path)
    return out


def embed_images(rows, out_path, embedder, view="processor", prepare=None, n_hashes=0, procs=1, batch=32,
                 chunk=2000, force=False):
    """Whole-image descriptors of `rows` ([{"key", "image", ...}], keys unique).

    prepare(row) -> (PIL image, hashes | None, extra | None) is a module-level
    function (worker processes pickle it by name); the default applies the
    EXIF orientation and nothing else. The image is reduced to the view
    (view_small) in the worker and embedded in this process. A row whose image
    cannot be read or embedded is a NaN row; a row whose hashes fail has
    hash_ok False.

    With out_path, the file is written atomically, resumes per chunk of rows,
    and a current file (same rows, embedder, preparation and hash count) is
    returned as is. Without out_path nothing is written. Returns load_images'
    dict."""
    if view != "processor":
        raise EmbedError("view %r: only the processor view is defined (%s)" % (view, VIEW))
    prepare = prepare or prepare_view
    keys = [str(r["key"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise EmbedError("embed_images: duplicate keys")
    ident = image_ident(rows, embedder, prepare, n_hashes)
    if out_path is not None:
        out_path = Path(out_path)
        if not out_path.name.endswith(".npz"):
            raise EmbedError("%s: an image descriptor file ends in .npz" % out_path)
        if not force and image_file_current(out_path, ident):
            return load_images(out_path)
    t0 = time.time()
    step = max(1, int(chunk))
    n_parts = (len(rows) + step - 1) // step
    Xs, Hs, oks, extras, stats = [], [], [], [], collections.Counter()
    with V._Workers(procs) as workers:
        for j in range(n_parts):
            part_rows = rows[j * step:(j + 1) * step]
            part = _part_path(out_path, j) if out_path is not None else None
            pm = _npz_meta(part) if (part is not None and part.exists()) else None
            want_part = dict(ident, rows_sha256=rows_digest(part_rows), chunk=step)
            if pm and not force and all(pm.get(k) == v for k, v in want_part.items()):
                with np.load(part) as d:
                    Xs.append(d["X"].astype(np.float32))
                    Hs.append(d["H"] if "H" in d else np.zeros((len(part_rows), 0), dtype=np.uint64))
                    oks.append(d["hash_ok"] if "hash_ok" in d else np.zeros(len(part_rows), dtype=bool))
                    extras.extend(json.loads(str(e)) for e in d["extra"])
                stats.update(pm.get("stats", {}))
                continue
            t1 = time.time()
            X, H, ok, ext, st = _embed_rows(part_rows, embedder, prepare, int(n_hashes), workers, int(batch))
            if part is not None:
                _save_npz(part, dict(want_part, stats=dict(st), seconds=round(time.time() - t1, 1)),
                          **_pack(keys[j * step:(j + 1) * step], X, H, ok, ext))
            Xs.append(X)
            Hs.append(H)
            oks.append(ok)
            extras.extend(ext)
            stats.update(st)
            if n_parts > 1:
                log("  %s chunk %d/%d: %d images" % (out_path.name if out_path else "images", j + 1, n_parts,
                                                      len(part_rows)))
    dim = int(Xs[0].shape[1]) if Xs else int(embedder.dim)     # a fully resumed run loads no model
    X = np.concatenate(Xs) if Xs else np.zeros((0, dim), dtype=np.float32)
    H = np.concatenate(Hs) if Hs else np.zeros((0, max(0, int(n_hashes))), dtype=np.uint64)
    ok = np.concatenate(oks) if oks else np.zeros(0, dtype=bool)
    meta = dict(ident, format=IMAGE_FORMAT, n=len(rows), dim=dim, stats=dict(stats),
                seconds=round(time.time() - t0, 1), built_utc=utc())
    arrays = _pack(keys, X, H, ok, extras)
    if out_path is None:
        return {"keys": keys, "X": arrays["X"], "H": H, "hash_ok": ok, "extra": extras, "meta": meta,
                "path": None, "sha256": None}
    _save_npz(out_path, meta, **arrays)
    for j in range(n_parts):
        p = _part_path(out_path, j)
        if p.exists():
            p.unlink()
    return load_images(out_path)
