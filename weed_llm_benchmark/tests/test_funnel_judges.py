#!/usr/bin/env python3
"""funnel/judges.py: J-zs, J-knn1, J-knn2 and J1 on the independent-truth
photos (contract docs/FUNNEL_AUDIT.md §4.1, §4.3, DEC-9; runner
docs/FUNNEL_AUDIT_RUNNER.md §4.12, §5.3.2, §7.2, §7.6 M7).

Direct checks of the two scoring rules:
  * zero_shot: rows sum to 1, equal a softmax computed here, a non-finite
    row stays NaN, a label without a prompt and a bad scale refuse;
  * knn: a bank entry that shares any of the four keys with the query (lab,
    source, near_dup3, provenance) and sits at cosine 1.0 does not change P
    (the check mutation point M7 switches off); a missing key value never
    shares; on random keys the vectorised exclusion equals a kNN over the
    bank filtered by qualify.shares (the one definition of "shares");
    exclusion by session drops the query's own session and its provenance
    twin; a query with no eligible neighbour is a NaN row; bad banks refuse.

The per-cell cap: a synthetic config with "kt5_per_cell_max": 2; the kept
items per (source, source name) cell are the ones the seed text
funnel/v1/knn2/kt5/<cell> draws (recomputed here), and a cap on a set that is
not in the bank refuses.

score_all on the real weed.json with a synthetic world (crops.csv, DINOv2
shards written here, a KT7 table of tiny PNG photos, fake embedders, a fake
text tower with logit scale 50, a fake Step 1 probe, a fake resolver):
  * every score file of the panel is written with the header, its labels,
    and (kNN) the bank's sha256, which equals the index's and banks()'s;
  * P rows sum to 1; a crop without a DINOv2 feature is a NaN row, top -1;
  * the J-knn1 and J-knn2 rows equal kNNs recomputed here over the bank
    filtered by session / by qualify.shares (with qualify.item_keys: KT7
    photos are compared per observation, so a KT7 query can use another
    observation's exemplar);
  * J1 on KT7 comes from the adapter's probe, with its cosines;
  * the prompts on weed.json: one per target (template, lineage, common
    name), per attractor and per remaining named non-target, and the
    non-object prompts, all distinct;
  * a rerun is a no-op (same files); with force the arrays are identical
    (deterministic, fake text tower included); a changed parameter refuses
    without force; a missing KT7 table refuses naming the fetch;
  * the kNN vote equals exp(cos / temperature) over the top k by hand;
  * check_records passes on every score file's and the index's inputs (no
    directory or pseudo-path as a path); the Step 1 features are recorded by
    a digest of their values, so changed values refuse without force;
  * a sample-lock amendment leaves the score files current (prereg core);
  * a changed query key (one unit's lab) refuses the kNN files without
    force (query_keys_sha256);
  * a KT7 item without a crop row refuses with JudgeError;
  * judges.py names no model class (inspect.getsource).

Run:  python3 tests/test_funnel_judges.py
"""
import csv
import inspect
import os
import pathlib
import shutil
import sys
import tempfile
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_judges_"))
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


def raises(fn, err, contains=None):
    try:
        fn()
    except err as e:
        if contains and contains not in str(e):
            print("       raised without %r: %s" % (contains, e))
            return None
        return str(e) or "raised"
    return None


try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None


# ------------------------------------------------------------ direct rules
def test_zero_shot(J):
    print("zero_shot")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5))
    X[3] = np.nan
    T = rng.normal(size=(6, 5))
    groups = [0, 0, 1, 2, 2, 2]
    P = J.zero_shot(X, T, 20.0, groups, 3)
    fin = [i for i in range(7) if i != 3]
    check("rows sum to 1", np.allclose(P[fin].sum(axis=1), 1.0))
    check("a non-finite row stays NaN", np.isnan(P[3]).all())
    Xn = X[0] / np.linalg.norm(X[0])
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    L = 20.0 * Tn @ Xn
    e = np.exp(L - L.max())
    e /= e.sum()
    want = np.array([e[0] + e[1], e[2], e[3] + e[4] + e[5]])
    check("P is the softmax over prompts summed per label", np.allclose(P[0], want, atol=1e-12))
    check("a label without a prompt refuses",
          raises(lambda: J.zero_shot(X, T, 20.0, [0, 0, 0, 2, 2, 2], 3), J.JudgeError, "no prompt") is not None)
    check("a non-positive scale refuses", raises(lambda: J.zero_shot(X, T, 0.0, groups, 3), J.JudgeError) is not None)


def _k(**kw):
    base = {"source": None, "near_dup3": None, "provenance": None, "lab": None, "session": None}
    base.update(kw)
    return base


def test_knn(J, QF):
    print("knn")
    q = np.array([[1.0, 0.0, 0.0]])
    B = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 1.0, 0.0]])
    labels = [0, 1, 2, 2]
    bkeys = [_k(source="s0", near_dup3="n0", provenance="p0", lab="L1"),
             _k(source="s1", near_dup3="n1", provenance="p1", lab="L2"),
             _k(source="s2", near_dup3="n2", provenance="p2", lab="L3"),
             _k(source="s3", near_dup3="n3", provenance="p3", lab="L4")]
    qk = [_k(source="sq", near_dup3="nq", provenance="pq", lab="L1")]
    P = J.knn(q, qk, B, labels, bkeys, 3, k=2, temperature=0.1, exclude="disjoint")
    Pw = J.knn(q, qk, B[1:], labels[1:], bkeys[1:], 3, k=2, temperature=0.1, exclude="none")
    Pn = J.knn(q, qk, B, labels, bkeys, 3, k=2, temperature=0.1, exclude="none")
    check("M7: a same-lab bank entry at cosine 1.0 does not change P", np.allclose(P, Pw) and P[0, 0] == 0.0,
          (P, Pw))
    check("... and would have, had it been seen (the check matters)", Pn[0, 0] > 0.5, Pn)
    # the rule by hand: the top k = 3 by cosine (1.0, 0.8, 0.6), weights exp(cos / tau), normalised
    Ph = J.knn(q, qk, B, labels, bkeys, 3, k=3, temperature=0.1, exclude="none")
    w = np.exp(np.array([1.0, 0.8, 0.6]) / 0.1)
    check("P is the exp(cos / temperature) vote of the top k, normalised (by hand)",
          np.allclose(Ph[0], w / w.sum(), atol=1e-6), (Ph, w / w.sum()))
    for kind in ("source", "near_dup3", "provenance", "lab"):
        kq = [_k(**{"source": "x", "near_dup3": "x", "provenance": "x", "lab": "x", kind: bkeys[0][kind]})]
        Pk = J.knn(q, kq, B, labels, bkeys, 3, k=2, temperature=0.1, exclude="disjoint")
        check("sharing only %s excludes the entry" % kind, np.allclose(Pk, Pw), Pk)
    Pm = J.knn(q, [_k()], B, labels, [_k() for _ in range(4)], 3, k=2, temperature=0.1, exclude="disjoint")
    check("missing key values never share", np.allclose(Pm, Pn), Pm)
    # equivalence with qualify.shares on random keys
    rng = np.random.default_rng(1)
    nq, nb, d = 30, 40, 6
    Qx, Bx = rng.normal(size=(nq, d)), rng.normal(size=(nb, d))
    lab_b = rng.integers(0, 4, size=nb)
    vocab = [None, "a", "b", "c", "d"]

    def rk():
        return _k(**{kind: vocab[int(rng.integers(len(vocab)))] for kind in ("source", "near_dup3", "provenance", "lab")})
    qks, bks = [rk() for _ in range(nq)], [rk() for _ in range(nb)]
    Pv = J.knn(Qx, qks, Bx, lab_b, bks, 4, k=5, temperature=0.07, exclude="disjoint")
    same = True
    for i in range(nq):
        keep = [j for j in range(nb) if not any(QF.shares(qks[i], bks[j]).values())]
        if not keep:
            same &= bool(np.isnan(Pv[i]).all())
            continue
        Pi = J.knn(Qx[i:i + 1], [qks[i]], Bx[keep], lab_b[keep], [bks[j] for j in keep], 4, k=5,
                   temperature=0.07, exclude="none")
        same &= bool(np.allclose(Pv[i], Pi[0], atol=1e-6))
    check("the vectorised exclusion equals a kNN over the bank filtered by qualify.shares", same)
    # session exclusion
    qs = [_k(session="S1", provenance="p9")]
    Bs = np.array([[1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.5, 0.86, 0.0]])
    bs = [_k(session="S1", provenance="a"), _k(session="S2", provenance="p9"), _k(session="S2", provenance="b")]
    Ps = J.knn(q, qs, Bs, [0, 1, 2], bs, 3, k=3, temperature=0.07, exclude="session")
    check("session exclusion: own session and provenance twin unseen", np.allclose(Ps, [[0, 0, 1]]), Ps)
    Ps2 = J.knn(q, [_k(provenance="p9")], Bs, [0, 1, 2], bs, 3, k=3, temperature=0.07, exclude="session")
    check("a query without a session loses only its twin", Ps2[0, 1] == 0 and Ps2[0, 0] > 0, Ps2)
    Pnan = J.knn(np.array([[np.nan, 0, 0], [1.0, 0, 0]]), [_k(), _k(lab="L")], B[:1], [0], [_k(lab="L")], 3,
                 k=2, exclude="disjoint")
    check("a NaN query and a query with no eligible neighbour are NaN rows", np.isnan(Pnan).all(), Pnan)
    check("a NaN bank row refuses",
          raises(lambda: J.knn(q, qk, np.array([[np.nan, 0, 0]]), [0], [_k()], 3), J.JudgeError) is not None)
    check("a bank label outside the space refuses",
          raises(lambda: J.knn(q, qk, B, [0, 1, 2, 3], bkeys, 3), J.JudgeError) is not None)
    check("an unknown exclusion refuses",
          raises(lambda: J.knn(q, qk, B, labels, bkeys, 3, exclude="some"), J.JudgeError) is not None)


def test_cap(J, D):
    print("per-cell cap")
    from weed_optimizer_framework.tools.inc import common as C
    raw = {"domain": "toy", "adapter": "none",
           "classes": {"targets": [{"id": 0, "name": "Alpha", "common": "alpha", "taxon": "Genusa alpha",
                                    "rank": "species", "not": []},
                                   {"id": 1, "name": "Gamma", "common": "gamma", "taxon": "Genusg gamma",
                                    "rank": "species", "not": []}],
                       "other": {"id": 2, "name": "Rest"}},
           "judges": {"features": {"model": "m/x", "pooling": "cls"},
                      "panel": [{"id": "J-knn2", "kind": "knn", "bank": ["KT1", "KT5"], "holdout": "disjoint",
                                 "k": 3, "temperature": 0.1, "kt5_per_cell_max": 2}]},
           "known_truth": {"KT1": {"independent": False}, "KT5": {"independent": False}},
           "sources": {"reference": "ref", "lab_groups": {}}}
    dom = D.Domain(raw, TMP / "toy.json", "0" * 64)
    names = []
    items = {"KT1": [], "KT5": []}
    X = []

    def add(kt, source, name, truth, kind):
        cid = len(names)
        names.append(name)
        X.append(np.eye(4)[cid % 4] + 0.01 * cid)
        items[kt].append({"id": "b:%s#%d" % (source, cid), "kt": kt, "crop_id": cid, "crop_set": "pool",
                          "truth": truth, "truth_kind": kind, "source": source, "lab": "L" + source,
                          "near_dup3": "n%d" % cid, "provenance": "p%d" % cid, "session": None, "role": None})
    for i in range(3):
        add("KT1", "ref", "Alpha", i % 2, "target")
    for i in range(5):
        add("KT5", "s1", "nameA", 2, "attractor")
    add("KT5", "s1", "nameB", 2, "attractor")
    for i in range(3):
        add("KT5", "s2", "nameA", 2, "attractor")
    table = types.SimpleNamespace(src_name=names)
    b = J.banks(dom, None, {"crops": np.array(X)}, known_truth=items, crop_table=table)["J-knn2"]

    def expect(cell, ids):
        ids = sorted(ids)
        if len(ids) <= 2:
            return set(ids)
        perm = np.random.default_rng(C.stable_int("funnel/v1/knn2/kt5/%s" % cell)).permutation(len(ids))
        return {ids[i] for i in perm[:2]}
    want = set(it["id"] for it in items["KT1"])
    for cell, src, nm in (("s1|nameA", "s1", "nameA"), ("s1|nameB", "s1", "nameB"), ("s2|nameA", "s2", "nameA")):
        want |= expect(cell, [it["id"] for it in items["KT5"]
                              if it["source"] == src and names[it["crop_id"]] == nm])
    check("KT5 is capped at 2 per (source, name) cell, by the pinned seed text",
          set(b["ids"]) == want and len(b["ids"]) == 3 + 2 + 1 + 2, (sorted(b["ids"]), sorted(want)))
    check("the cap's seed texts are recorded",
          b["seeds"] == {"KT5/s1|nameA": "funnel/v1/knn2/kt5/s1|nameA", "KT5/s2|nameA": "funnel/v1/knn2/kt5/s2|nameA"},
          b["seeds"])
    check("labels: targets plus other (KT5 attractors are 'other')",
          b["label_names"] == ["Alpha", "Gamma", "other"] and sorted(set(b["labels"].tolist())) == [0, 1, 2])
    raw3 = dict(raw, judges=dict(raw["judges"], panel=[dict(raw["judges"]["panel"][0],
                                                          bank=["KT1", "KT5", "KT6:calibration"])]),
                known_truth=dict(raw["known_truth"], KT6={"independent": False}))
    items3 = dict(items, KT6=[dict(items["KT1"][0], id="b:mh#99", kt="KT6", role="estimation")])
    b3 = J.banks(D.Domain(raw3, TMP / "toy.json", "0" * 64), None, {"crops": np.array(X)}, known_truth=items3,
                 crop_table=table)["J-knn2"]
    check("a bank set with no member of the role (KT6 before H3a) is recorded as empty, not refused",
          b3["empty"] == ["KT6:calibration"] and b3["ids"] == b["ids"] and b3["kt"] == ["KT1", "KT5"],
          (b3["empty"], b3["kt"]))
    raw2 = dict(raw, judges=dict(raw["judges"], panel=[dict(raw["judges"]["panel"][0], kt6_per_cell_max=2)]))
    check("a cap on a set outside the bank refuses",
          raises(lambda: J.banks(D.Domain(raw2, TMP / "toy.json", "0" * 64), None, {"crops": np.array(X)},
                                 known_truth=items, crop_table=table), J.JudgeError, "not in its bank")
          is not None)


# ------------------------------------------------------------ score_all world
class GridEmbedder:
    """Features = grey means over a grid of the crop (dim = rows * cols)."""

    def __init__(self, name, rows, cols):
        self.name, self.rows, self.cols = name, rows, cols
        self.dim = rows * cols
        self.calls = 0

    def __call__(self, pils):
        self.calls += 1
        out = []
        for p in pils:
            a = np.asarray(p.convert("L"), dtype=np.float64) / 255.0
            h, w = a.shape
            f = [a[r * h // self.rows:(r + 1) * h // self.rows, c * w // self.cols:(c + 1) * w // self.cols].mean()
                 for r in range(self.rows) for c in range(self.cols)]
            out.append(np.array(f, dtype=np.float32) + 0.01)
        return np.stack(out)


class FakeText:
    name = "fake text tower"
    logit_scale = 50.0

    def __init__(self, dim):
        self.dim = dim
        self.calls = 0

    def __call__(self, texts):
        from weed_optimizer_framework.tools.inc import common as C
        self.calls += 1
        return np.stack([np.random.default_rng(C.stable_int(t)).normal(size=self.dim) for t in texts])


class FakeResolver:
    def lineage_string(self, taxon):
        return "Plantae Tracheophyta %s" % taxon


def build_world(root, dom):
    """Crop table, DINOv2 shards, KT7 table and photos, known truth, unit
    keys; returns the fake adapter module and the pieces the checks need."""
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.inc import verify as V
    n_t = len(dom.targets)
    other = dom.other["id"]
    rng = np.random.default_rng(7)
    rows, items, keys, dino = [], {k: [] for k in ("KT1", "KT2", "KT4", "KT5", "KT6", "KT7")}, {}, []

    def dirn(c):
        v = np.zeros(16)
        v[c % 16] = 1.0
        return v

    def crop(set_, key, source, box, label, name, kt=None, truth=None, kind=None, lab="Lx", prov=None,
             session=None, role=None, feat=None):
        cid = len(rows)
        rows.append([cid, set_, key, "/nowhere/%s.png" % key, source, session or source, box, "0.5", "0.5",
                     "0.2", "0.2", 100, 100, label, name])
        pre = {"core": "t1", "copy": "t2", "pool": "b"}[set_]
        uid = "%s:%s#%d" % (pre, key, box)
        k = {"source": source, "lab": lab, "near_dup3": "n:%s" % key, "provenance": prov or "prov:%s" % key}
        keys[uid] = k
        dino.append(feat if feat is not None else dirn(truth if truth is not None else label) + 0.2 * rng.normal(size=16))
        if kt:
            items[kt].append(dict(k, id=uid, kt=kt, crop_id=cid, crop_set=set_, truth=truth, truth_taxon=None,
                                  truth_kind=kind, claimed=kt in ("KT4", "KT5", "KT6"), session=session, role=role))
        return cid, uid
    for i in range(3 * n_t):
        c = i % n_t
        crop("core", "c%02d" % i, "train_core", 0, c, dom.targets[c]["name"], kt="KT1", truth=c, kind="target",
             lab="LuLab", session="S%d" % (i % 3))
    for i in range(4):
        crop("copy", "cp%d" % i, "cottonweed_holdout", 0, i % n_t, "x", kt="KT2", truth=i % n_t, kind="target",
             lab="src:cottonweed_holdout", prov="prov:c%02d" % i)
    for i in range(6):
        crop("pool", "wc%d" % i, "project_agml__weed_crop_detection", 0, 5, "Ragweed", kt="KT4", truth=5,
             kind="target", lab="NDSU")
    for i in range(6):
        crop("pool", "gh%d" % i, "project_agml__greenhouse_crop_weed_detection", 0, other, "redroot pigweed",
             kt="KT5", truth=other, kind="attractor", lab="NDSU")
    for i in range(4):
        crop("pool", "mh%d" % i, "project_agml__mh_weed16_weed_detection", 0, other, "12", kt="KT6", truth=9,
             kind="target", lab="src:project_agml__mh_weed16_weed_detection",
             role="calibration" if i < 2 else "estimation")
    nan_cid, nan_uid = crop("pool", "rfx0", "rf_x", 0, other, "", lab="src:rf_x", feat=np.full(16, np.nan))
    for i in range(1, 5):
        crop("pool", "rfx%d" % i, "rf_x", 0, other, "", lab="src:rf_x")
    path = root / "step1" / "crops.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        wr.writerows(rows)
    crops = V.Crops(path)
    fdir = root / "funnel"
    X = np.array(dino, dtype=np.float32)
    V._save_npz(fdir / "emb_dinov2" / V._shard_name(0, 1),
                {"crops_sha256": crops.sha, "embedder": "facebook/dinov2-base:cls", "dim": 16, "crops": len(rows),
                 "shard": 0, "nshards": 1, "images": len(rows), "stats": {"failed_crops": 1}},
                crop_ids=np.arange(len(rows), dtype=np.int64), X=X.astype(np.float16))
    # KT7 photos: observation o<i>, one photo each; roles by position
    kt7_rows = []
    for i in range(12):
        target = i < 9
        truth = (i % n_t) if target else other
        shade = int(30 + 20 * (i % 9))
        img = Image.new("RGB", (40, 30), (shade, 255 - shade, (7 * i) % 255))
        p = fdir / "kt7" / "photos" / ("o%d_p%d.png" % (i, i))
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        uid = "t7:o%d/p%d" % (i, i)
        role = "exemplar" if i % 3 == 0 else ("g0" if i % 3 == 1 else "sentinel")
        kt7_rows.append([i, "kt7", uid, str(p), "kt7", "o%d" % i, 0, "0.5", "0.5", "1", "1", 40, 30, truth,
                         "Taxon %d" % i])
        items["KT7"].append({"id": uid, "kt": "KT7", "crop_id": i, "crop_set": "kt7", "truth": truth,
                             "truth_taxon": "Taxon %d" % i, "truth_kind": "target" if target else "attractor",
                             "claimed": False, "source": "kt7", "lab": "src:kt7", "near_dup3": "n:%s" % uid,
                             "provenance": "prov:%s" % uid, "session": None, "role": role})
    with open(fdir / "kt7" / "crops_kt7.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        wr.writerows(kt7_rows)
    X1 = rng.normal(size=(len(rows), 12)).astype(np.float32)
    W = np.random.default_rng(3).normal(size=(12, n_t + 1))
    Pro = np.random.default_rng(4).normal(size=(n_t + 1, 12))

    def j1_scores(Xq):
        Xq = np.asarray(Xq, dtype=np.float64)
        L = Xq @ W
        e = np.exp(L - L.max(axis=1, keepdims=True))
        Xn = Xq / np.linalg.norm(Xq, axis=1, keepdims=True)
        Pn = Pro / np.linalg.norm(Pro, axis=1, keepdims=True)
        return e / e.sum(axis=1, keepdims=True), Xn @ Pn.T

    ad = types.ModuleType("fake_step1_adapter")
    ad.__file__ = str(HERE)
    ad.crop_table = lambda: V.Crops(path)
    ad.known_truth = lambda domain, funnel_dir: items
    ad.unit_keys = lambda uids: {u: dict(keys[u]) for u in uids if u in keys}
    ad.step1_features = lambda: (X1.astype(np.float16), {"embedder": "fake-step1", "nshards": 1, "dim": 12})
    ad.text_encoder = lambda: FakeText(12)
    ad.j1_scores = j1_scores
    ad.bioclip_embedder = lambda: GridEmbedder("fake-step1", 3, 4)
    return ad, {"crops": crops, "X": X, "items": items, "keys": keys, "nan_cid": nan_cid, "X1": X1,
                "j1": j1_scores, "fdir": fdir}


def test_score_all(J, D, QF):
    print("score_all on weed.json")
    from weed_optimizer_framework.tools.inc import common as C
    dom = D.load("weed")
    pre = D.load_prereg(TMP / "inc" / "funnel" / "prereg_v1.json")
    root = TMP / "world"
    ad, w = build_world(root, dom)
    fdir = w["fdir"]
    dino = GridEmbedder("facebook/dinov2-base:cls", 4, 4)
    text = FakeText(12)
    index = J.score_all(pre, dom, fdir, ad, text_encoder=text, resolver=FakeResolver(), dinov2_embedder=dino,
                        testing=True)
    want = {"J1__kt7", "J-zs__crops", "J-zs__kt7", "J-knn1__crops", "J-knn1__kt7", "J-knn2__crops", "J-knn2__kt7"}
    check("every score file of the panel (J-vlm is scored from the labeller's answers)",
          set(index["files"]) == want, sorted(index["files"]))
    check("the index carries the header", index["format"] == "funnel-judge-index/1"
          and index["prereg"]["core_sha256"] == pre.core_sha256 and index["domain"] == "weed")

    def load(name):
        with np.load(fdir / "judges" / ("%s.npz" % name)) as d:
            import json
            return {k: d[k] for k in d.files if k != "meta"}, json.loads(str(d["meta"]))
    arr, meta = load("J-knn2__crops")
    n_t = len(dom.targets)
    check("J-knn2 meta: labels (targets + other), the bank's sha256 and the header",
          meta["labels"] == dom.target_names + ["other"] and meta["bank"]["sha256"] == index["banks"]["J-knn2"]["sha256"]
          and meta["format"] == "funnel-judge-scores/1" and meta["exclusion"] == "disjoint"
          and meta["crops_sha256"] == w["crops"].sha and meta["testing"] is True)
    X7, _m7 = __import__("weed_optimizer_framework.tools.funnel.embed", fromlist=["x"]).load_table(
        fdir / "kt7" / "crops_kt7.csv", fdir / "emb_dinov2_kt7.npz")
    b = J.banks(dom, ad, {"crops": w["X"], "kt7": X7}, known_truth=w["items"], crop_table=w["crops"])
    check("the bank sha256 in the file equals banks() recomputed", meta["bank"]["sha256"] == b["J-knn2"]["sha256"])
    P = arr["P"].astype(np.float32)
    fin = np.isfinite(P).all(axis=1)
    check("finite rows sum to 1", np.allclose(P[fin].sum(axis=1), 1.0, atol=5e-3))
    check("a crop without a DINOv2 feature is a NaN row with top -1",
          not fin[w["nan_cid"]] and int(arr["top"][w["nan_cid"]]) == -1 and fin.sum() == len(P) - 1)
    # J-knn2 rows equal a kNN over the bank filtered by qualify.shares
    crops = w["crops"]
    uids = J.crop_unit_ids(crops)
    bb = b["J-knn2"]
    bank_keys = [{k: bb["keys"][k][i] for k in bb["keys"]} for i in range(len(bb["ids"]))]
    ok = True
    for cid in range(crops.n):
        if cid == w["nan_cid"]:
            continue
        qk = dict(w["keys"][uids[cid]])
        keep = [j for j in range(len(bank_keys)) if not any(QF.shares(qk, bank_keys[j]).values())]
        Pi = J.knn(w["X"][cid:cid + 1], [qk], bb["X"][keep], bb["labels"][keep], [bank_keys[j] for j in keep],
                   bb["n_labels"], k=10, temperature=0.07, exclude="none")
        ok &= bool(np.allclose(P[cid], Pi[0], atol=2e-3))
    check("every J-knn2 crop row equals a kNN over the bank filtered by qualify.shares", ok)
    arr7, meta7 = load("J-knn2__kt7")
    items7 = sorted(w["items"]["KT7"], key=lambda it: it["id"])
    ok7, used_other_obs = True, False
    for r, it in enumerate(items7):
        qk = QF.item_keys(it, dom)
        keep = [j for j in range(len(bank_keys)) if not any(QF.shares(qk, bank_keys[j]).values())]
        used_other_obs |= any(bb["kt"] and bb["ids"][j].startswith("t7:") for j in keep)
        Pi = J.knn(X7[int(it["crop_id"]):int(it["crop_id"]) + 1], [qk], bb["X"][keep], bb["labels"][keep],
                   [bank_keys[j] for j in keep], bb["n_labels"], k=10, temperature=0.07, exclude="none")
        ok7 &= bool(np.allclose(arr7["P"][r].astype(np.float32), Pi[0], atol=2e-3))
    check("J-knn2 KT7 rows: per-observation keys (other observations' exemplars stay usable)",
          ok7 and used_other_obs and list(arr7["unit_index"]) == [int(it["crop_id"]) for it in items7])
    # J-knn1: bank KT1, own session and provenance twin excluded
    arr1, meta1 = load("J-knn1__crops")
    b1 = b["J-knn1"]
    check("J-knn1: targets only, holdout by session", meta1["labels"] == dom.target_names
          and meta1["exclusion"] == "session" and b1["n"] == 3 * n_t)
    sess = {it["id"]: it["session"] for kt in w["items"].values() for it in kt if it.get("session")}
    ok1 = True
    for cid in range(crops.n):
        if cid == w["nan_cid"]:
            continue
        u = uids[cid]
        keep = [j for j in range(b1["n"]) if not ((sess.get(u) and b1["keys"]["session"][j] == sess.get(u))
                                                 or b1["keys"]["provenance"][j] == w["keys"][u]["provenance"])]
        Pi = J.knn(w["X"][cid:cid + 1], [{}], b1["X"][keep], b1["labels"][keep], [{} for _ in keep], n_t, k=10,
                   temperature=0.07, exclude="none")
        ok1 &= bool(np.allclose(arr1["P"][cid].astype(np.float32), Pi[0], atol=2e-3))
    check("every J-knn1 row equals a kNN without the query's session and provenance twin", ok1)
    # J1 on KT7
    arrj, metaj = load("J1__kt7")
    Pj, cj = w["j1"](X7_s1(fdir)[[int(it["crop_id"]) for it in items7]].astype(np.float32))
    check("J1 on KT7 is the adapter's probe on the Step 1 features of the photos, with its cosines",
          np.allclose(arrj["P"].astype(np.float32), Pj, atol=2e-3) and np.allclose(arrj["cos"].astype(np.float32), cj,
                                                                                 atol=2e-3)
          and metaj["labels"] == dom.target_names + ["other"] and (fdir / "emb_bioclip_kt7.npz").exists())
    # J-zs
    arrz, metaz = load("J-zs__crops")
    texts, groups, labels = J.prompts(dom, FakeResolver())
    check("J-zs: labels (targets, other, non_object), prompts recorded, the tower's scale",
          metaz["labels"] == dom.target_names + ["other", "non_object"] and len(metaz["prompts"]) == len(texts)
          and metaz["text_encoder"]["scale"] == 50.0)
    Tz = text(texts)
    Pz = J.zero_shot(w["X1"].astype(np.float16).astype(np.float32), Tz, 50.0, groups, len(labels))
    check("J-zs rows are zero_shot over the Step 1 crop features", np.allclose(arrz["P"].astype(np.float32), Pz,
                                                                              atol=2e-3, equal_nan=True))
    # prompts on weed.json
    tt = [t["taxon"] for t in dom.targets]
    att = []
    for a in dom.attractors:
        if a["taxon"] not in tt and a["taxon"] not in att:
            att.append(a["taxon"])
    nots = []
    for t in dom.targets:
        for x in t.get("not", []):
            if x not in tt and x not in att and x not in nots:
                nots.append(x)
    zs = [e for e in dom.raw["judges"]["panel"] if e["kind"] == "zero_shot"][0]
    t0 = dom.targets[0]
    check("prompts: one per target, per attractor, per remaining named non-target, and the non-object ones",
          len(texts) == len(tt) + len(att) + len(nots) + len(zs["non_object_prompts"])
          and groups[:n_t] == list(range(n_t)) and set(groups[n_t:len(texts) - len(zs["non_object_prompts"])]) == {n_t}
          and groups[-1] == n_t + 1 and len(set(texts)) == len(texts)
          and texts[0] == zs["prompt_template"].format(lineage="Plantae Tracheophyta %s" % t0["taxon"],
                                                      common=t0["common"]), (len(texts), texts[:2]))
    # every recorded input re-hashes (runner §1.2 freshness): no pseudo-path a consumer cannot open
    from weed_optimizer_framework.tools.funnel import FunnelError, check_records
    bad = []
    for k in sorted(want):
        try:
            check_records(load(k)[1].get("inputs"))
        except FunnelError as e:
            bad.append("%s: %s" % (k, e))
    try:
        check_records(index.get("inputs"))
    except FunnelError as e:
        bad.append("index: %s" % e)
    check("check_records passes on every score file's and the index's inputs", not bad, bad)
    s1_rec = metaz["inputs"]["features"]
    check("the Step 1 features are recorded without a path, by a digest of their values",
          s1_rec.get("path") is None and s1_rec.get("what") == "adapter.step1_features" and len(s1_rec["sha256"]) == 64,
          s1_rec)
    # reruns
    before = {k: C.sha256_file(v["path"]) for k, v in index["files"].items()}
    text2 = FakeText(12)
    J.score_all(pre, dom, fdir, ad, text_encoder=text2, resolver=FakeResolver(), dinov2_embedder=dino, testing=True)
    after = {k: C.sha256_file(v["path"]) for k, v in index["files"].items()}
    check("a rerun on the same inputs is a no-op", before == after, (before, after))
    # the sample lock (an amendment) leaves the score files current (runner §3.3)
    pre_path = TMP / "inc" / "funnel" / "prereg_v1.json"
    D.append_amendment(pre_path, {"id": "T-lock", "kind": "sample_lock", "date": "2026-09-28"})
    pre_locked = D.load_prereg(pre_path)
    try:
        J.score_all(pre_locked, dom, fdir, ad, text_encoder=FakeText(12), resolver=FakeResolver(),
                    dinov2_embedder=dino, testing=True)
        locked_err = None
    except J.JudgeError as e:
        locked_err = str(e)
    after_lock = {k: C.sha256_file(v["path"]) for k, v in index["files"].items()}
    check("after a sample-lock amendment a rerun is still a no-op (the prereg core is unchanged)",
          locked_err is None and after_lock == before and pre_locked.core_sha256 == pre.core_sha256
          and pre_locked.sha256 != pre.sha256, locked_err)
    # other Step 1 feature values under the same embedder name are another input
    s1_saved = ad.step1_features
    ad.step1_features = lambda: ((w["X1"] + 0.5).astype(np.float16), {"embedder": "fake-step1", "nshards": 1,
                                                                     "dim": 12})
    check("changed Step 1 feature values (same embedder, same shape) refuse without force",
          raises(lambda: J.score_all(pre, dom, fdir, ad, text_encoder=FakeText(12), resolver=FakeResolver(),
                                     dinov2_embedder=dino, testing=True), J.JudgeError, "--force") is not None)
    ad.step1_features = s1_saved
    keys_saved = ad.unit_keys
    moved = sorted(w["keys"])[0]

    def unit_keys_moved(uids):
        out = keys_saved(uids)
        if moved in out:
            out[moved] = dict(out[moved], lab="another-lab")
        return out
    ad.unit_keys = unit_keys_moved
    check("a changed query key (one unit's lab) refuses the kNN files without force",
          raises(lambda: J.score_all(pre, dom, fdir, ad, text_encoder=FakeText(12), resolver=FakeResolver(),
                                     dinov2_embedder=dino, testing=True), J.JudgeError, "--force") is not None
          and len(load("J-knn2__crops")[1].get("query_keys_sha256", "")) == 64)
    ad.unit_keys = keys_saved
    known_saved = ad.known_truth
    broken = {k: [dict(it) for it in v] for k, v in w["items"].items()}
    broken["KT7"][0]["crop_id"] = None
    ad.known_truth = lambda domain, funnel_dir: broken
    check("a KT7 item without a crop row refuses with the judges' own error, naming the fetch",
          raises(lambda: J.score_all(pre, dom, fdir, ad, text_encoder=FakeText(12), resolver=FakeResolver(),
                                     dinov2_embedder=dino, testing=True), J.JudgeError, "fetch --what kt7")
          is not None)
    ad.known_truth = known_saved
    snap = {k: load(k)[0] for k in want}
    J.score_all(pre, dom, fdir, ad, text_encoder=FakeText(12), resolver=FakeResolver(), dinov2_embedder=dino,
                testing=True, force=True)
    again = {k: load(k)[0] for k in want}
    check("with force every array is identical (deterministic, fake text tower included)",
          all(all(np.array_equal(snap[k][a], again[k][a], equal_nan=True) for a in snap[k]) for k in want))
    raw2 = __import__("copy").deepcopy(dom.raw)
    for e in raw2["judges"]["panel"]:
        if e["id"] == "J-knn2":
            e["k"] = 5
    dom2 = D.Domain(raw2, dom.path, dom.sha256)
    check("a changed parameter refuses without force",
          raises(lambda: J.score_all(pre, dom2, fdir, ad, text_encoder=FakeText(12), resolver=FakeResolver(),
                                     dinov2_embedder=dino, testing=True), J.JudgeError, "--force") is not None)
    (fdir / "kt7" / "crops_kt7.csv").rename(fdir / "kt7" / "moved.csv")
    check("a missing KT7 table refuses, naming the fetch",
          raises(lambda: J.score_all(pre, dom, fdir, ad, text_encoder=FakeText(12), resolver=FakeResolver(),
                                     dinov2_embedder=dino, testing=True), J.JudgeError, "fetch --what kt7")
          is not None)


def X7_s1(fdir):
    from weed_optimizer_framework.tools.funnel import embed as E
    X, _m = E.load_table(fdir / "kt7" / "crops_kt7.csv", fdir / "emb_bioclip_kt7.npz")
    return X


def test_source(J):
    print("engine source")
    src = inspect.getsource(J)
    low = src.lower()
    check("judges.py names no model class", "bioclip" not in low and "textencoder" not in low
          and "open_clip" not in low)


def main():
    if np is None:
        skip("all", "numpy is not installed")
        return
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import judges as J
    from weed_optimizer_framework.tools.funnel import qualify as QF
    test_zero_shot(J)
    test_knn(J, QF)
    test_cap(J, D)
    test_source(J)
    if Image is None:
        skip("score_all", "PIL is not installed")
        return
    test_score_all(J, D, QF)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), SKIPS))
    sys.exit(1 if FAILURES else 0)
