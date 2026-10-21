#!/usr/bin/env python3
"""V3F deviation 4 (2026-10-20; docs/prereg/V3F_dev4_2026-10-20.md): the data build (inc2.v3fam_data, L23H) on base
rows as they are on the cluster, and over a failed build's partial outputs.

On 2026-10-12 both data builds failed with funnel.embed's "embed_images: duplicate keys": base B holds base v3's
re-written PNG of a base_v2 key and base_v2 the original file under the same key (6,805 of the 6,811 keys), and the
build scanned once per sha256 but keyed the scan by manifest key. tests/test_inc2_v3fam_data.py's fixture gives base
B and base_v2 the same file for each key, so it never met the case. This world does what the cluster holds: base B's
and base A's rows of every base_v2 key point at a PNG re-write (other bytes, other sha256), base_v2's at the original
JPG.

- the build completes (before the fix: EmbedError "duplicate keys" from the first scan; the stand-in scanners sit
  behind funnel.embed.embed_images' own key check, as inc2.guard.EmbedScanner does);
- each scan is given every distinct file once, keyed by its sha256 (the manifest key beside it), the season scan
  both files of an A' key;
- a hit counts for its own file and its key: a DINOv2 eval hit on an NDSU row's original JPG is recorded on base_v2's
  file only (base B's PNG has none), kept (a base_v2 row); a season hit on a train_core row's PNG drops the key from
  A'1, A'12 and B' alike, recorded with the file it is of; every hit names its file (file_sha256);
- the arms equal V3F.4/V3F.5's rules on this world (the registered lists before the checks; D5B doses' rows);
- a failed build's partial outputs: the probe's copies already under splits/v3fam/d5b (as job 49061040 left 476 of
  them), an unrelated file there and v3fam/v1/leak/ood_desc.npz are kept byte for byte; the copies made again are
  the same files; summary.json lists them (preexisting: files, copies made again, the ones not used); a file
  present at the start that changed refuses before the summary;
- a build cut off inside a DINOv2 scan (node lost, scancel, a requeue, ENOSPC) leaves funnel.embed's resume parts
  (<name>.partNNNNN.npz) and temp files in v3fam/v1/leak/: the next attempt's scan (the real funnel.embed.
  embed_images) resumes from the parts and removes them, and the build does not refuse over them (they are listed
  apart as transient, with whether they remain); an output beside them still may not change.

Run:  python3 tests/test_inc2_v3fam_dev4_data.py
"""
import hashlib
import json
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_v3fam_data as T  # noqa: E402  (sets INC_DIR / REPO first, through test_inc2_train)
import test_inc2_v3fam_sweep as SWT  # noqa: E402

from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.funnel import embed as E  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_data as D  # noqa: E402

W = T.W
FAILURES = T.FAILURES
check = T.check


def rewritten_fixture():
    """test_inc2_v3fam_data's fixture with base B's and base A's rows of every base_v2 key on a PNG re-write of the
    original (base v3's materialisation: other bytes, so another sha256), base_v2's and train_core's on the JPG."""
    rows, v2keys, ext, paths = T.build_fixture()
    md = T.FIX / "manifests_rw"
    for k in v2keys:
        src = pathlib.Path(rows[k]["r1"]["image"])
        dst = T.FIX / "rewritten" / ("%s.png" % k)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").save(dst, format="PNG")
        rows[k]["r1"] = dict(rows[k]["r1"], image=str(dst), sha256=W.sha(dst))
        rows[k]["png"] = str(dst)
    paths = {"base_b": md / "base_v3_weed.jsonl", "base_a1": md / "base_v2_weed.jsonl",
             "base_v2": md / "base_v2.jsonl", "train_core": md / "train_core.jsonl"}
    C.write_manifest(paths["base_b"], [rows[k]["r1"] for k in rows])
    C.write_manifest(paths["base_a1"], [rows[k]["r1"] for k in v2keys])
    C.write_manifest(paths["base_v2"], [rows[k]["r12"] for k in v2keys])
    C.write_manifest(paths["train_core"], [rows[k]["r12"] for k in v2keys if k.startswith("train_core__")])
    return rows, v2keys, ext, paths


class _KeysAccepted(Exception):
    pass


class _Emb(object):
    name = "stand-in:cls"
    dim = 4


def embed_key_check(rows):
    """funnel.embed.embed_images' own check of a scan's rows (inc2.guard.EmbedScanner.scan hands them to it): run
    up to the point where it would describe the images, so a repeated key raises its EmbedError ('embed_images:
    duplicate keys', the error of 2026-10-12)."""
    real = E.image_ident

    def stop(*_a, **_k):
        raise _KeysAccepted()
    E.image_ident = stop
    try:
        E.embed_images([{"key": str(r["key"]), "image": str(r["image"]), "sha256": r.get("sha256")} for r in rows],
                       None, _Emb())
    except _KeysAccepted:
        pass
    finally:
        E.image_ident = real


class SeenScanner(T.FakeScanner):
    """FakeScanner, recording each row it is given, behind funnel.embed's key check as the real scanner is."""

    def __init__(self, rule):
        T.FakeScanner.__init__(self, rule)
        self.rows = []

    def scan(self, rows, desc_path=None, procs=1):
        embed_key_check(rows)
        self.rows.extend(dict(r) for r in rows)
        return T.FakeScanner.scan(self, rows, desc_path, procs)


def w_rewritten():
    print("the data build where base B's file of a base_v2 key is a re-written PNG (as on the cluster)")
    rows, v2keys, ext, paths = T.WORLD["fixture"]
    inputs, P = T.make_inputs(plants=False)
    n1, t1 = P["N1"], P["T3"]
    n1_jpg = rows[n1]["r12"]["image"]
    t1_png = rows[t1]["png"]
    check("the world: every base_v2 key's base B file is another file than its base_v2 file (as 6,805 of 6,811 keys "
          "on the cluster)", all(inputs["base_b"]["rows"][k]["sha256"] != inputs["base_v2"]["rows"][k]["sha256"]
                                 for k in v2keys) and len(v2keys) > 20, len(v2keys))
    ev = SeenScanner(lambda r: {"eval_split": "dev", "eval_key": "dev_x", "cos": 0.97, "bits": 30}
                     if r["image"] == n1_jpg else None)
    ood = SeenScanner(lambda r: {"eval_split": "ood23", "eval_key": "ood_y", "cos": 0.96, "bits": 25}
                      if r["image"] == t1_png else None)
    inputs["scanners"] = {"eval": ev, "ood": ood, "catch_factory": inputs["scanners"]["catch_factory"]}
    ts = F.root() / "timestamp.json"
    ts.parent.mkdir(parents=True, exist_ok=True)
    ts.write_text(json.dumps({"ots": "fixture", "revisions": [{"sha256": F.REVISION_SHA256}]}))
    # a failed build's partial outputs (job 49061040): the probe's copies, written before its scan refused, an
    # unrelated file beside them, and the season index's cache
    with T.small_probe(P["kfm"]):
        for k, _i, fam in D.copy_plan(P["probe"]):
            D.make_copy(inputs["base_v2"]["rows"][k], fam, D.out_dir() / "d5b")
    orphan = D.out_dir() / "d5b" / "images" / "ff" / ("f" * 64 + ".png")
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"a copy another code version wrote")
    cache = F.root() / "leak" / "ood_desc.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"the season index's descriptors")
    before = T.tree(D.out_dir(), F.root() / "leak")
    mt = {p: pathlib.Path(p).stat().st_mtime_ns for p in before}
    time.sleep(0.01)
    err = None
    try:
        with T.small_probe(P["kfm"]):
            s = D.build(inputs=inputs, testing=True, procs=1, embedder=T.EMB)
    except Exception as e:                       # noqa: BLE001  (the defect of 2026-10-12 raises EmbedError)
        err, s = e, None
    check("the build completes (on 2026-10-12: funnel.embed's 'embed_images: duplicate keys' from the first scan)",
          err is None and s is not None and s["status"] == "complete", "%s: %s" % (type(err).__name__, err))
    if s is None:
        return
    lists = P["lists"]
    shas = {k: (inputs["base_b"]["rows"][k]["sha256"], inputs["base_v2"]["rows"][k]["sha256"]) for k in v2keys}
    ev_keys = [r["key"] for r in ev.rows]
    ev_files = {(r["sha256"], r["key"], r.get("row_key")) for r in ev.rows}
    want_ev = sum(2 for k in v2keys if inputs["base_v2"]["rows"][k]["source"] not in D.TSW_SOURCES) + len(P["probe"])
    check("the eval scan: every distinct file once (both files of each train_core and NDSU key, and the copies), "
          "keyed by its sha256, the manifest key beside it", len(ev_keys) == len(set(ev_keys)) == want_ev
          and all(sha == key for sha, key, _rk in ev_files) and all(rk for _s, _k, rk in ev_files)
          and {rk for _s, _k, rk in ev_files} >= {n1, t1}, (len(ev_keys), want_ev))
    s_keys = [r["key"] for r in ood.rows]
    want_s = len(set(lists["BPRIME"])) + len(set(lists["APRIME"]))
    check("the season scan: both files of every A' key and every B' file, once each (%d)" % want_s,
          len(s_keys) == len(set(s_keys)) == want_s == s["nevertrain"]["embed_scanned"]["season"]
          and s["nevertrain"]["embed_scanned"]["eval"] == want_ev, (len(s_keys), want_s))
    nt = s["nevertrain"]
    n1_v2 = (nt["base_v2_kept"].get(n1) or {}).get("hits") or []
    n1_b = (nt["base_b_kept"].get(n1) or {}).get("hits") or []
    check("a DINOv2 hit on an NDSU row's original JPG is its file's: recorded on base_v2's file (kept, a base_v2 row), "
          "none on base B's PNG", [h["file_sha256"] for h in n1_v2 if h["how"] == "embed"] == [shas[n1][1]]
          and not [h for h in n1_b if h["how"] == "embed"] and n1 in s["recorded_only"]["D6_base_v2"]["hits"],
          (n1_v2, n1_b))
    A = {a: set(T.keys_of(v["manifest"])) for a, v in s["arms"].items()}
    t1_hits = nt["season_dropped"].get(t1) or []
    check("a season hit on a train_core row's PNG drops the key from A'1, A'12 and B' alike (its base_v2 JPG had none); "
          "the hit names the PNG", all(t1 not in A[a] for a in ("D1A1", "D1A12", "D1B1"))
          and t1 in A["D4F"] and t1 in A["D5B0"] and [h["file_sha256"] for h in t1_hits] == [shas[t1][0]],
          t1_hits)
    hits = [h for v in (nt["base_b_kept"], nt["base_v2_kept"]) for x in v.values() for h in x["hits"]] + \
        [h for v in (nt["base_b_dropped"], nt["season_dropped"], nt["copies_dropped"]) for x in v.values() for h in x]
    check("every recorded hit names the file it is of", hits and all(h.get("file_sha256") for h in hits), len(hits))
    kb = s["key_lists_before_checks"]
    check("the key lists before the checks are V3F.4's on this world; D4's arms keep every row but the season's",
          all(kb[n]["sha256"] == T.lsha(lists[n]) for n in ("F", "N", "R15", "BPRIME", "APRIME"))
          and A["D4F"] == set(lists["F"]) - {P["TURF"]} and A["D1A1"] == set(lists["APRIME"]) - {t1},
          {n: kb[n]["rows"] for n in kb})
    pre = s["preexisting"]
    n_copy_files = 2 * len(P["probe"])
    after = T.tree(D.out_dir(), F.root() / "leak")
    kept = all(after.get(p) == h and pathlib.Path(p).stat().st_mtime_ns == mt[p] for p, h in before.items())
    check("a failed build's partial outputs coexist: every file present at the start (%d: the probe's %d copy files, "
          "an unrelated file, ood_desc.npz) kept byte for byte and untouched; the copies made again are the same "
          "files; the summary lists them" % (len(before), n_copy_files),
          kept and pre["files"] == len(before) == n_copy_files + 2 and pre["unchanged"] is True
          and pre["copy_files_made_again"] == n_copy_files
          and sorted(pre["not_used"]) == sorted([str(orphan), str(cache)])
          and all(r["sha256"] in {pathlib.Path(p).stem for p in before} for r in s["copies"]["records"]),
          {k: pre.get(k) for k in ("files", "copy_files_made_again", "not_used")})
    listing = [[p, pathlib.Path(p).stat().st_size, before[p]] for p in sorted(before)]
    check("  the summary's listing_sha256 is the sha256 of [path, bytes, sha256] of those files",
          pre["listing_sha256"] == hashlib.sha256(json.dumps(listing).encode("utf-8")).hexdigest())


def w_changed():
    print("a file present at the start that changes during the build refuses before the summary")
    d = T.TMP / "pre_changed"
    p = d / "a.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    real = D.out_dir
    D.out_dir = lambda: d
    try:
        pre = D._preexisting()
        rec_ok = D._preexisting_record(pre, {})
        p.write_bytes(b"y")
        e = None
        try:
            D._preexisting_record(pre, {})
        except D.DataRefused as ex:
            e = ex
        p.unlink()
        e2 = None
        try:
            D._preexisting_record(pre, {})
        except D.DataRefused as ex:
            e2 = ex
    finally:
        D.out_dir = real
    check("a changed file and a removed file each refuse ('never over them'); an unchanged one is listed unused",
          str(p) in rec_ok["not_used"] and e is not None and "never over them" in str(e) and e2 is not None,
          (rec_ok, e, e2))


class _Killed(BaseException):
    """A job killed in the middle of a scan (not an Exception: funnel's batch retry does not catch it)."""


class _KilledAfter(SWT.StubEmbedder):
    """The stand-in embedder, killed at its n-th call (one call per chunk of one row here)."""

    def __init__(self, n):
        self.left = n

    def __call__(self, pils):
        self.left -= 1
        if self.left < 0:
            raise _Killed()
        return SWT.StubEmbedder.__call__(self, pils)


def w_resume_parts():
    print("a build cut off inside a DINOv2 scan: the next attempt resumes from funnel.embed's parts and does not "
          "refuse over them")
    d = T.TMP / "pre_parts"
    imgs = d / "imgs"
    imgs.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(4):
        p = imgs / ("im%d.png" % i)
        Image.new("RGB", (32, 24), (40 * i, 90, 200 - 30 * i)).save(p)
        sha = W.sha(p)
        rows.append({"key": sha, "image": str(p), "sha256": sha})
    leak = d / "v3fam_root" / "leak"
    out = leak / "eval_scan_desc.npz"
    cache = leak / "ood_desc.npz"
    leak.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"the season index's descriptors")
    real_out, real_root = D.out_dir, F.root
    D.out_dir = lambda: d / "splits_v3fam"
    F.root = lambda: d / "v3fam_root"
    try:
        killed = False
        try:                                      # attempt 1: killed after two of the scan's four chunks
            E.embed_images(rows, out, _KilledAfter(2), chunk=1, procs=1)
        except _Killed:
            killed = True
        parts = sorted(p.name for p in leak.glob("eval_scan_desc.part*.npz"))
        stale = leak / "eval_scan_desc.part00002.npz.tmp"    # a part write cut off by the kill
        stale.write_bytes(b"half a part")
        pre = D._preexisting()
        got = E.embed_images(rows, out, SWT.EMB, chunk=1, procs=1)      # attempt 2: resumes, then removes the parts
        left = sorted(p.name for p in leak.iterdir())
        err, rec = None, None
        try:
            rec = D._preexisting_record(pre, {})
        except D.DataRefused as e:
            err = e
        cache.write_bytes(b"other descriptors")
        err2 = None
        try:
            D._preexisting_record(pre, {})
        except D.DataRefused as e:
            err2 = e
    finally:
        D.out_dir, F.root = real_out, real_root
    check("attempt 1 is killed inside the scan and leaves two resume parts; attempt 2 completes from them and "
          "removes them (and replaces the cut-off temp file)", killed and parts == ["eval_scan_desc.part00000.npz",
                                                                                  "eval_scan_desc.part00001.npz"]
          and len(got["keys"]) == 4 and left == ["eval_scan_desc.npz", "ood_desc.npz"], (killed, parts, left))
    tr = {pathlib.Path(x[0]).name: x[3] for x in (rec or {}).get("transient") or []}
    check("the build does not refuse over them: the outputs present at the start (ood_desc.npz) unchanged, the parts "
          "and the temp file listed apart as transient and gone",
          err is None and rec["files"] == 1 and rec["not_used"] == [str(cache)]
          and tr == {"eval_scan_desc.part00000.npz": False, "eval_scan_desc.part00001.npz": False,
                     "eval_scan_desc.part00002.npz.tmp": False}, (str(err), rec))
    check("  an output beside them that changed still refuses", err2 is not None and "never over them" in str(err2),
          err2)


def main():
    t0 = time.time()
    try:
        T.WORLD["world"] = W.build_world()
        T.WORLD["fixture"] = rewritten_fixture()
        for fn in (w_rewritten, w_changed, w_resume_parts):
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                import traceback
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-1500:]))
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
