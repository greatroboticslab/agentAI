#!/usr/bin/env python3
"""funnel/embed.py: DINOv2 crop shards, crop-table files and whole-image
descriptors (runner docs/FUNNEL_AUDIT_RUNNER.md §4.12, §5.3.1).

The world is synthetic: tiny PNG images and a crops.csv in the Step 1
verifier's CROP_FIELDS format (sets core, copy and pool), plus one row whose
image does not exist. The embedder is a fake (a deterministic function of
the pixels it is given, which also records every crop it sees), so no model
is loaded.

Pinned:
  * over 3 shards every crop id is covered exactly once, and load() returns
    one row per crop;
  * a crop whose image cannot be read is a NaN row, and its neighbours keep
    their own features (each row equals the fake embedder applied to that
    crop as verify._cut_task cuts it);
  * the crop fed to the embedder is byte-identical to verify._cut_task's;
  * a finished shard is a no-op; a killed run resumes and embeds only the
    chunks that were not finished;
  * load() refuses shards made from another crops.csv, or by another
    embedder, or an incomplete set, or a set covering a crop twice;
  * a finished shard, or chunks, made by another embedder are rebuilt, not
    reused; load_table refuses a same-size table with other content;
    embed_images re-embeds a file made from other rows or by another
    embedder;
  * embed_table / load_table on a table with a set outside the verifier's;
  * embed_images: the view (shorter edge 256), NaN rows for unreadable
    images, hashes through a preparation function, per-chunk resume, a
    current file returned as is, and load_images refusing other rows;
  * the feature model comes from the domain config (weed.json: DINOv2-base,
    CLS); Dinov2Embedder refuses an open_clip hub model and an unknown pooling
    before loading anything.

Run:  python3 tests/test_funnel_embed.py
"""
import csv
import os
import pathlib
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_embed_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
HERE = pathlib.Path(__file__).resolve()
try:                                  # PYTHONPATH may name a package copy (the mutation harness)
    import weed_optimizer_framework  # noqa: F401
except ImportError:
    sys.path.insert(0, str(HERE.parents[1]))
REAL_REPO = HERE.parents[2]
for src, dst in ((REAL_REPO / "docs" / "FUNNEL_AUDIT.md", TMP / "repo" / "docs" / "FUNNEL_AUDIT.md"),
                 (HERE.parents[1] / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json",
                  TMP / "inc" / "funnel" / "prereg_v1.json")):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def skip(name, reason):
    print("SKIP: %s (%s)" % (name, reason))
    SKIPS.append(name)


try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None


def raises(fn, err, contains=None):
    try:
        fn()
    except err as e:
        if contains and contains not in str(e):
            print("       raised without %r: %s" % (contains, e))
            return None
        return str(e) or "raised"
    return None


class Stop(BaseException):
    """Kills a run mid-way (not an Exception, so the verifier's per-crop retry
    does not swallow it)."""


class FakeEmbedder:
    name = "fake/crop:cls"
    dim = 6

    def __init__(self, stop_after=None):
        self.seen = []
        self.calls = 0
        self.stop_after = stop_after

    @staticmethod
    def feature(a):
        a = np.asarray(a, dtype=np.float64) / 255.0
        h, w = a.shape[:2]
        left, right = a[:, : w // 2].mean(), a[:, w // 2:].mean()
        top, bot = a[: h // 2].mean(), a[h // 2:].mean()
        return np.array([a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean(), left - right, top - bot,
                         a.std()], dtype=np.float32)

    def __call__(self, pils):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            raise Stop()
        out = []
        for p in pils:
            a = np.asarray(p)
            self.seen.append(a.copy())
            out.append(self.feature(a))
        return np.stack(out)


class OtherEmbedder(FakeEmbedder):
    """The same features under another model name."""
    name = "other/crop:cls"


def write_image(path, seed, size=(48, 40)):
    rng = np.random.default_rng(seed)
    w, h = size
    base = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    im = Image.fromarray(base).resize((w, h), Image.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def build_crops(root):
    """crops.csv with 7 images (one missing on disk) and 14 crops, sets core,
    copy and pool; returns (path, rows)."""
    from weed_optimizer_framework.tools.inc import verify as V
    imgs = []
    for i in range(7):
        p = root / "img" / ("im%d.png" % i)
        if i != 3:
            write_image(p, i, size=(40 + 4 * i, 36 + 2 * i))
        imgs.append(p)
    sets = ["core", "core", "copy", "pool", "pool", "pool", "pool"]
    rows = []
    for i, p in enumerate(imgs):
        W, H = 40 + 4 * i, 36 + 2 * i
        for b in range(2):
            cx, cy = 0.3 + 0.4 * b, 0.5
            rows.append([len(rows), sets[i], "k%d" % i, str(p), "src%d" % (i % 3), "g%d" % (i % 2), b,
                         "%.6f" % cx, "%.6f" % cy, "%.6f" % 0.3, "%.6f" % 0.5, W, H, b, "name%d" % b])
    path = root / "crops.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        wr.writerows(rows)
    return path, rows


def main():
    if np is None:
        skip("all", "numpy is not installed")
        return
    if Image is None:
        skip("all", "PIL is not installed")
        return
    from weed_optimizer_framework.tools.funnel import EmbedError
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import embed as E
    from weed_optimizer_framework.tools.inc import verify as V

    root = TMP / "world"
    crops_path, rows = build_crops(root)
    crops = V.Crops(crops_path)
    n = crops.n
    missing_img = str(root / "img" / "im3.png")
    bad_ids = [r[0] for r in rows if r[3] == missing_img]

    print("shards")
    emb = FakeEmbedder()
    out = TMP / "funnel" / "emb_dinov2"
    metas = [E.embed_crops(crops, out, s, 3, embedder=emb, procs=1, batch=3, chunk_images=1) for s in range(3)]
    ids = []
    for s in range(3):
        with np.load(out / V._shard_name(s, 3)) as d:
            ids.extend(d["crop_ids"].tolist())
    check("3 shards cover every crop id exactly once", sorted(ids) == list(range(n)) and len(ids) == n, ids)
    check("shard meta: crops_sha256, embedder, dim, shard, nshards",
          all(m["crops_sha256"] == crops.sha and m["embedder"] == emb.name and m["dim"] == 6
              and m["nshards"] == 3 for m in metas) and [m["shard"] for m in metas] == [0, 1, 2])
    check("no part files left", not [p for p in os.listdir(out) if ".part" in p], os.listdir(out))
    X, info = E.load(crops, out)
    check("load: one row per crop, float16", X.shape == (n, 6) and X.dtype == np.float16, X.shape)
    check("load info: shard files and their digest", info["nshards"] == 3 and len(info["files"]) == 3
          and len(info["sha256"]) == 64 and info["embedder"] == emb.name)
    nan_rows = [i for i in range(n) if not np.isfinite(X[i].astype(np.float32)).all()]
    check("the unreadable image's crops are NaN rows, and only they", nan_rows == bad_ids, (nan_rows, bad_ids))
    # every other row is the fake feature of verify's own cut of that crop
    good = True
    for img, cids in crops.images():
        if img == missing_img:
            continue
        _i, got_ids, arrays, err = V._cut_task((img, [crops.row(c) for c in cids]))
        for c, a in zip(got_ids, arrays):
            want = FakeEmbedder.feature(a).astype(np.float16)
            if not np.array_equal(X[c], want):
                good = False
    check("each row is its own crop's feature (a NaN row shifts no neighbour)", good)
    # byte identity of the crop fed to the embedder
    probe = FakeEmbedder()
    img = str(root / "img" / "im5.png")
    cids = [r[0] for r in rows if r[3] == img]
    E.embed_crops(crops, TMP / "probe", 0, 1, embedder=probe, procs=1, batch=64, chunk_images=100)
    _i, _ids, arrays, _err = V._cut_task((img, [crops.row(c) for c in cids]))
    fed = [a for a in probe.seen if any(np.array_equal(a, b) for b in arrays)]
    check("the crop fed to the embedder is byte-identical to verify._cut_task's",
          len(fed) == len(arrays) and all(a.dtype == np.uint8 and a.shape == (224, 224, 3) for a in fed))

    print("no-op and resume")
    again = FakeEmbedder()
    m0 = E.embed_crops(crops, out, 0, 3, embedder=again, procs=1, batch=3, chunk_images=1)
    check("a finished shard is a no-op", again.calls == 0 and m0["crops_sha256"] == crops.sha)
    res_dir = TMP / "resume"
    killer = FakeEmbedder(stop_after=2)
    try:
        E.embed_crops(crops, res_dir, 0, 1, embedder=killer, procs=1, batch=64, chunk_images=1)
        killed = False
    except Stop:
        killed = True
    parts = sorted(p for p in os.listdir(res_dir) if ".part" in p)
    check("a killed run leaves its finished chunks", killed and len(parts) == 2, parts)
    resumed = FakeEmbedder()
    E.embed_crops(crops, res_dir, 0, 1, embedder=resumed, procs=1, batch=64, chunk_images=1)
    n_images_with_crops = len([1 for img, _c in crops.images() if img != missing_img])
    check("the resumed run embeds only the unfinished chunks",
          resumed.calls == n_images_with_crops - 2, (resumed.calls, n_images_with_crops))
    Xr, _ = E.load(crops, res_dir)
    check("the resumed shard equals the uninterrupted one",
          np.array_equal(np.nan_to_num(Xr.astype(np.float32), nan=-9), np.nan_to_num(X.astype(np.float32), nan=-9)))

    print("load refusals")
    rows2 = [list(r) for r in rows]
    rows2[0][7] = "0.310000"
    other = root / "crops_other.csv"
    with open(other, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        wr.writerows(rows2)
    check("load refuses shards made from another crops.csv",
          raises(lambda: E.load(V.Crops(other), out), EmbedError, "stale") is not None)
    check("load refuses shards of another embedder",
          raises(lambda: E.load(crops, out, embedder="another:cls"), EmbedError) is not None)
    (out / V._shard_name(1, 3)).unlink()
    check("load refuses an incomplete shard set",
          raises(lambda: E.load(crops, out), EmbedError, "missing shards [1]") is not None)
    # shard 1 replaced by a copy of shard 0: every shard is present and current, but
    # shard 0's crops are covered twice and shard 1's not at all
    with np.load(out / V._shard_name(0, 3)) as d:
        m0_meta = __import__("json").loads(str(d["meta"]))
        dup_ids, dup_X = d["crop_ids"].copy(), d["X"].copy()
    V._save_npz(out / V._shard_name(1, 3), dict(m0_meta, shard=1), crop_ids=dup_ids, X=dup_X)
    check("load refuses a shard set that covers a crop twice",
          raises(lambda: E.load(crops, out), EmbedError, "exactly once") is not None)
    (out / V._shard_name(1, 3)).unlink()
    X2, info2 = E.load(crops.sha, res_dir)
    check("load by sha256 alone infers the size", X2.shape == (n, 6))
    check("embed_crops refuses a shard out of range",
          raises(lambda: E.embed_crops(crops, out, 3, 3, embedder=emb), EmbedError) is not None)

    print("another embedder")
    other = OtherEmbedder()
    m_other = E.embed_crops(crops, res_dir, 0, 1, embedder=other, procs=1, batch=64, chunk_images=1)
    check("a finished shard made by another embedder is rebuilt, not reused",
          other.calls > 0 and m_other["embedder"] == OtherEmbedder.name, (other.calls, m_other.get("embedder")))
    mixed = TMP / "mixed"
    try:
        E.embed_crops(crops, mixed, 0, 1, embedder=FakeEmbedder(stop_after=2), procs=1, batch=64, chunk_images=1)
    except Stop:
        pass
    other2 = OtherEmbedder()
    try:
        m_mixed = E.embed_crops(crops, mixed, 0, 1, embedder=other2, procs=1, batch=64, chunk_images=1)
        err = None
    except EmbedError as e:
        m_mixed, err = {}, str(e)
    check("chunks another embedder left behind are re-embedded, not resumed",
          err is None and other2.calls == n_images_with_crops and m_mixed.get("embedder") == OtherEmbedder.name,
          (err, other2.calls, n_images_with_crops))

    print("crop tables")
    tab = root / "kt7.csv"
    trows = []
    for i in range(4):
        p = write_image(root / "kt7" / ("p%d.png" % i), 100 + i, size=(30 + i, 50))
        trows.append([i, "kt7", "t7:o%d/p%d" % (i, i), str(p), "kt7", "o%d" % i, 0, "0.5", "0.5", "1", "1",
                      30 + i, 50, 0, "Taxon a"])
    with open(tab, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        wr.writerows(trows)
    check("verify.Crops cannot read a table outside its sets (why CropTable exists)",
          raises(lambda: V.Crops(tab), (ValueError, V.VerifyError)) is not None)
    te = FakeEmbedder()
    meta = E.embed_table(tab, TMP / "funnel" / "emb_x_kt7.npz", te)
    Xt, mt = E.load_table(E.CropTable(tab), TMP / "funnel" / "emb_x_kt7.npz", embedder=te.name)
    check("embed_table covers the table; load_table returns one row per crop",
          Xt.shape == (4, 6) and np.isfinite(Xt.astype(np.float32)).all() and meta["crops"] == 4
          and mt["sha256"] == __import__("hashlib").sha256(open(TMP / "funnel" / "emb_x_kt7.npz", "rb").read()).hexdigest())
    te2 = FakeEmbedder()
    E.embed_table(tab, TMP / "funnel" / "emb_x_kt7.npz", te2)
    check("a current table file is a no-op", te2.calls == 0)
    check("load_table refuses another table",
          raises(lambda: E.load_table(E.CropTable(crops_path), TMP / "funnel" / "emb_x_kt7.npz"), EmbedError)
          is not None)
    tab2 = root / "kt7_edited.csv"
    trows2 = [list(r) for r in trows]
    trows2[1][7] = "0.4"                              # same size, one box moved
    with open(tab2, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        wr.writerows(trows2)
    check("load_table refuses a table of the same size with other content",
          raises(lambda: E.load_table(E.CropTable(tab2), TMP / "funnel" / "emb_x_kt7.npz"), EmbedError,
                 "another table") is not None)

    print("whole images")
    irows = [{"key": "a", "image": str(root / "img" / "im0.png")}, {"key": "b", "image": missing_img},
             {"key": "c", "image": str(root / "img" / "im6.png")}]
    ie = FakeEmbedder()
    res = E.embed_images(irows, None, ie, n_hashes=0)
    check("embed_images: NaN only for the unreadable image",
          [bool(np.isfinite(res["X"][i].astype(np.float32)).all()) for i in range(3)] == [True, False, True])
    shapes = sorted({a.shape[:2] for a in ie.seen})
    check("the view has a shorter edge of %d px" % E.VIEW_EDGE, all(min(s) == E.VIEW_EDGE for s in shapes), shapes)
    big = Image.new("RGB", (1000, 600), (10, 20, 30))
    check("view_small keeps the aspect ratio", E.view_small(big).size == (427, 256), E.view_small(big).size)
    hp = TMP / "funnel" / "emb_dinov2_images_test.npz"
    he = FakeEmbedder()
    r1 = E.embed_images(irows, hp, he, prepare=_prepare_with_hash, n_hashes=2, chunk=1)
    check("hashes through the preparation function; a failed image has hash_ok False",
          list(r1["hash_ok"]) == [True, False, True] and int(r1["H"][0][1]) == 7
          and int(r1["H"][0][0]) == len("a"), r1["H"].tolist())
    check("extra per row is kept", r1["extra"][0] == {"k": "a"} and r1["extra"][1] is None)
    he2 = FakeEmbedder()
    r2 = E.embed_images(irows, hp, he2, prepare=_prepare_with_hash, n_hashes=2, chunk=1)
    check("a current image file is returned as is", he2.calls == 0 and r2["sha256"] == r1["sha256"])
    check("load_images refuses other rows",
          raises(lambda: E.load_images(hp, rows=irows[:2]), EmbedError, "other rows") is not None)
    rows_new = [irows[0], irows[2], {"key": "d", "image": str(root / "img" / "im4.png")}]
    he3 = FakeEmbedder()
    r4 = E.embed_images(rows_new, hp, he3, prepare=_prepare_with_hash, n_hashes=2, chunk=1)
    check("a file made from other rows is re-embedded, not returned",
          he3.calls > 0 and r4["keys"] == ["a", "c", "d"], (he3.calls, r4["keys"]))
    he4 = OtherEmbedder()
    r5 = E.embed_images(rows_new, hp, he4, prepare=_prepare_with_hash, n_hashes=2, chunk=1)
    check("a file made by another embedder is re-embedded, not returned",
          he4.calls > 0 and r5["meta"]["embedder"] == OtherEmbedder.name, (he4.calls, r5["meta"].get("embedder")))
    check("image_file_current: the identity of the file just written, and no other",
          E.image_file_current(hp, E.image_ident(rows_new, he4, _prepare_with_hash, 2))
          and not E.image_file_current(hp, E.image_ident(rows_new, FakeEmbedder(), _prepare_with_hash, 2))
          and not E.image_file_current(hp, E.image_ident(irows, he4, _prepare_with_hash, 2)))
    kill = FakeEmbedder(stop_after=1)
    hp2 = TMP / "funnel" / "emb_dinov2_images_resume.npz"
    try:
        E.embed_images(irows, hp2, kill, n_hashes=0, chunk=1)
    except Stop:
        pass
    left = sorted(p for p in os.listdir(hp2.parent) if p.startswith(hp2.name[:-4] + ".part"))
    again = FakeEmbedder()
    r3 = E.embed_images(irows, hp2, again, n_hashes=0, chunk=1)
    check("embed_images resumes per chunk", len(left) >= 1 and again.calls == 1 and len(r3["keys"]) == 3,
          (left, again.calls))
    check("embed_images refuses duplicate keys",
          raises(lambda: E.embed_images(irows + irows[:1], None, FakeEmbedder()), EmbedError) is not None)

    print("model from the config")
    weed = D.load("weed")
    check("weed.json names DINOv2-base with CLS pooling",
          E.features_config(weed) == ("facebook/dinov2-base", "cls"))
    check("Dinov2Embedder refuses an open_clip hub model before loading",
          raises(lambda: E.Dinov2Embedder("hf-hub:some/model"), EmbedError) is not None)
    check("Dinov2Embedder refuses an unknown pooling before loading",
          raises(lambda: E.Dinov2Embedder("facebook/dinov2-base", pooling="max"), EmbedError) is not None)
    bad = dict(weed.raw, judges=dict(weed.raw["judges"], features={"model": "x/y", "pooling": "max"}))
    check("features_config refuses an unknown pooling",
          raises(lambda: E.features_config(bad), EmbedError) is not None)
    # the loader wiring, with semisup_labeler's loader and batch embedder stubbed
    from weed_optimizer_framework.tools import semisup_labeler as SL
    calls = []
    saved = (SL._load_backbone, SL._embed_batch, SL._device)
    try:
        SL._device = lambda: "cpu"
        SL._load_backbone = lambda name, device: (calls.append(("load", name, device)) or "model", "proc")
        SL._embed_batch = lambda model, proc, pils, poolings, device: (
            calls.append(("batch", model, proc, len(pils), poolings, tuple(p.mode for p in pils)))
            or {poolings[0]: np.ones((len(pils), 5), dtype=np.float32)})
        de = E.Dinov2Embedder("facebook/dinov2-base", "cls")
        F = de([Image.new("L", (30, 20)), Image.new("RGB", (10, 10))])
        first = list(calls)
        lazy = E.default_embedder(weed)
        n_before = len(calls)
        lazy_name_free = lazy.name == "facebook/dinov2-base:cls" and len(calls) == n_before
        lazy([Image.new("RGB", (8, 8))])
        lazy([Image.new("RGB", (8, 8))])
        lazy_loads_once = [c[0] for c in calls[n_before:]].count("load") == 1 and lazy.dim == 5
    finally:
        SL._load_backbone, SL._embed_batch, SL._device = saved
    check("Dinov2Embedder: semisup_labeler's loader, CLS pooling, RGB input, name model:pooling, probed dim",
          first[0] == ("load", "facebook/dinov2-base", "cpu") and de.name == "facebook/dinov2-base:cls"
          and de.dim == 5 and F.shape == (2, 5) and first[-1][4] == ("cls",) and first[-1][5] == ("RGB", "RGB"),
          first)
    check("the domain's default embedder is lazy: its name costs no load, and it loads once",
          lazy_name_free and lazy_loads_once)


def _prepare_with_hash(row):
    from PIL import Image as _I, ImageOps
    with _I.open(row["image"]) as im0:
        im = ImageOps.exif_transpose(im0).convert("RGB")
    return im, [len(row["key"]), 7], {"k": row["key"]}


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), SKIPS))
    sys.exit(1 if FAILURES else 0)
