"""Intake: staging -> one intake batch (docs/CONTINUOUS_LOOP.md §3.2, §7.4, §8).

    collect intake --source ID

Under the intake lock, fail closed at every step, nothing written before the
guard loads:
  1. arrival check: fetch.json and every blob it lists must hash as recorded
     (a lab fetch synced to the cluster is verified here, by sha256);
     an intake of the same fetch record already committed is a no-op, unless
     its committed batches left images deferred (1b);
  1b. continuation shards (amendment 2026-10-03): when the committed batches
     of this fetch record (same fetch sha256) recorded images 'deferred' that
     none of them decided since, the intake takes the next shard instead of
     the no-op: the items no earlier batch of the fetch decided, drawn by
     cap_items with the same seed (so the shards partition the source and a
     rerun takes the same shard), at most the cap; a deferred image the
     source as read now lacks is rejected as not_in_source, so the shards
     always end. Inside the fetch the shards are judged as one batch: the
     earlier shards' images seed the exact-duplicate check
     (exact_dup_intake) and stay out of the guard's intake index
     (near_dup_intake), which keeps every other batch; every other check of
     the guard runs as for any batch. A shard whose fetch's extracted tree
     is complete, and records the fetch sha256 it was extracted from after
     its blobs hashed as recorded, reuses the tree without hashing the blobs
     again (no blob is read then; fetch.json is still hashed). The tree is
     kept while images remain deferred and removed by the batch that leaves
     none. summary.json's `shard` records the shard's number, the earlier
     batches and the images still deferred;
  2. the gates that do not need the images: the licence (P6: unresolved ->
     held unless a person's licence override names the source, refused ->
     closed whatever an override says), the registry (quarantined -> closed),
     the never-train slugs;
  3. the copy guard: inc2.guard.GuardV2.load(LOCK v2) (group A); a guard that
     cannot be loaded refuses the intake. The images of earlier intake
     batches are added as its intake index (near_dup_intake, 3 bits). Then
     the images decision L-5 dropped from base v2 outright (splits v2's
     l5_excluded.jsonl, checked against LOCK v2): an image whose bytes, or
     any of whose eight variants within 6 bits, copy one is refused as
     l5_copy (a re-upload of those sources would otherwise bring them back).
     Then the embedding calibration that will judge the batch's rows held
     h6_scan (decision L-9(c)): splits v2's embed_calibration_v2.json, the
     one LOCK v2 records (per-image false positives on hard same-domain
     negatives, never below the funnel's threshold), checked against its
     sha256 and its own gates. A production LOCK that records none, or a
     file that does not hash or load, refuses the intake (fail closed). Its
     threshold is bound to every held row (copy_scan_calibration) and
     recorded in guard.json and summary.json; inc2.step1_stream's copy
     scan judges the rows by it;
  4. archives extracted (zip, tar; no path escapes the work directory) into
     intake/work/<source>/<fetch sha12>/x/, other files linked in;
  5. normalise (normalize.read, the known item's format options) and the
     class map (classmap.build); pending names -> intake/work/<source>/
     pending_names.json, a held event, and a NamesPending refusal (lever L26);
  6. per image, in path order: its boxes in the intake class space; no box ->
     rejected; the image copied (hard link when possible) to
     intake/<batch>/images/<key><ext>; an exact byte duplicate of an image of
     this batch -> rejected; the guard's first refusing check -> rejected
     (the copy removed); else its label written to labels/<key>.txt and a
     manifest row;
  6b. D28-v2 (amendment 2026-10-03): each image refused as a dHash copy of an
     evaluation image (near_eval_v2, near_eval_variant) is weighed before its
     copy is removed: its pair cosine with the evaluation image the guard
     matched, both described by the v2 calibration's embedder
     (score_eval_hits, inc2.eval_hits). It goes into its decision
     (pair_cos) and summary.json (eval_hits), which the autopilot's D28
     reads. Each hit is weighed against the guard's match and every other
     evaluation image within the never-train radius, and keeps the highest.
     Only a batch with such a hit loads the embedder; a hit that cannot be
     weighed is recorded so, and D28 reads it as a leak (fail closed). The
     copies are removed in a finally around the whole per-image loop, so a
     failure anywhere in it never leaves one in the batch directory. A batch
     committed before the amendment is weighed again into a sidecar,
     intake/<batch>/eval_hits.json (rescore_eval_hits, which step1_stream
     eval-hits runs), from its decisions and its staging blobs;
  7. the batch files: decisions.jsonl (every source, class, file and image
     decision, kept or rejected, with its reason), manifest.jsonl,
     guard.json (counts per reason, the indexes' shas), sources.json and,
     last, summary.json (the commit marker); batches.jsonl and sources.jsonl
     appended; the registry entry written with annotation intake_v1 and
     status intake (neither pinned Step 1 nor the old merge reads it).

Manifest row (§3.2): key, image, sha256, label, label_sha256, source,
licence, research_only, lab_group, capture_group, dhash, plus session (the
capture group, so inc.common.MANIFEST_KEYS readers work), batch, hold_until
(h6_scan unless the source is provenance-cleared; licence when a person let an
unresolved licence through), provenance_cleared, exhaustive_labels,
licence_class, licence_override (the person's decision, or null), boxes,
target_boxes, unmapped_boxes, width, height, rel and intake_utc. Under an
override licence and licence_class stay as fetched ("unresolved") and
research_only is the licence's OR the override's; summary.json and the
registry entry's provenance record both. Labels hold class ids 0..n-1
(targets), the reject class and the unmapped id, which exists only in intake
labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

from . import (FORMATS, CollectError, GuardUnavailable, NamesPending, NormaliseError, Refusal, StaleInput,
               append_chained, batches_ledger, header, intake_dir, intake_lock, read_json,
               read_jsonl, safe_name, sha256_file, sha256_text, staging_dir, utc, write_json_atomic,
               write_jsonl_atomic)
from . import classmap as CM
from . import normalize as NZ
from .config import licence_override
from . import prefilter as PF
from . import state as S

ANNOTATION = PF.INTAKE_ANNOTATION
KEY_MAX = 180
ARCHIVE_EXTS = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.xz", ".tar.bz2")
SOURCE_LEAK_EVAL_SHARE = 0.05       # D28: >= 5 % refused by the never-train guard
SOURCE_LEAK_BASE_SHARE = 0.20       # D28: >= 20 % base copies
EVAL_REASONS = ("near_eval_v2", "near_eval_variant", "near_eval_embed")
DHASH_EVAL_REASONS = ("near_eval_v2", "near_eval_variant")    # inc2.eval_hits.DHASH_HIT_REASONS (a test checks)


# ------------------------------------------------------------------ guard
def default_guard(lock_path=None):
    """inc2.guard.GuardV2.load(lock) (group A); GuardUnavailable otherwise."""
    try:
        from ..inc2 import guard as G2
    except ImportError as e:
        raise GuardUnavailable("inc2.guard is not installed (%s): intake refuses (fail closed)" % e)
    try:
        return G2.GuardV2.load(lock_path)
    except Exception as e:  # noqa: BLE001 - any failure to build the guard refuses
        raise GuardUnavailable("GuardV2 does not load from %s: %s" % (lock_path or "LOCK v2", e))


L5_NAME = "l5_excluded.jsonl"
L5_REASON = "l5_copy"
L5_BITS = 6


def load_l5(lock_path=None):
    """The images decision L-5 (§2.6) drops from base v2 outright (splits
    v2's l5_excluded.jsonl beside the LOCK, which records its sha256): (their
    sha256 set, a 6-bit dHash index, record). GuardV2 does not index them and
    inc2.train drops them by exact sha256 only, so a re-upload of those
    sources (a fork, re-encoded) would otherwise come back through intake as
    new data. Fails closed (GuardUnavailable): a LOCK that records the list
    whose file is missing or changed, or a production LOCK that records none.
    A testing LOCK without it gives an empty set, recorded."""
    from ..near_dup import NearHashIndex
    try:
        from ..inc2 import common as C2
    except ImportError as e:
        raise GuardUnavailable("inc2.common is not installed (%s): intake refuses (fail closed)" % e)
    lock_path = Path(lock_path or C2.LOCK_PATH)
    try:
        lock = read_json(lock_path, "LOCK v2")
    except CollectError as e:
        raise GuardUnavailable("the L-5 exclusion list cannot be checked: %s" % e)
    path = lock_path.parent / L5_NAME
    want = lock.get("l5_excluded_sha256")
    idx, shas = NearHashIndex(), set()
    if want is None:
        if not lock.get("testing"):
            raise GuardUnavailable("LOCK v2 %s records no l5_excluded_sha256: the L-5 exclusions cannot be checked "
                                   "(fail closed)" % lock_path)
        return shas, idx, {"path": str(path), "sha256": None, "images": 0, "note": "a testing LOCK records none"}
    if not path.is_file() or sha256_file(path) != want:
        raise GuardUnavailable("%s is missing or does not hash to the %s LOCK v2 records" % (path, str(want)[:12]))
    n = 0
    for r in read_jsonl(path):
        shas.add(str(r.get("sha256")))
        if r.get("dhash") is not None:
            idx.add(int(r["dhash"]), r.get("key"), max_bits=L5_BITS)
        n += 1
    return shas, idx, {"path": str(path), "sha256": want, "images": n, "bits": L5_BITS}


def load_copy_scan(lock_path=None):
    """(record or None, what guard.json records): the v2 embedding
    calibration (decision L-9(c)) that judges the rows an intake holds for
    the copy scan, beside LOCK v2, which records its sha256
    (inc2.embed_calibration.locked). Fails closed (GuardUnavailable): a
    production LOCK that records none, a file that is missing, changed or
    does not show a usable calibration. A testing LOCK without one gives
    None, recorded."""
    try:
        from ..inc2 import common as C2
        from ..inc2 import embed_calibration as EC
    except ImportError as e:
        raise GuardUnavailable("inc2.embed_calibration is not installed (%s): intake refuses (fail closed)" % e)
    lock_path = Path(lock_path or C2.LOCK_PATH)
    try:
        rec, why = EC.locked(lock_path, production=True)
    except Exception as e:  # noqa: BLE001 - any failure to bind the calibration refuses
        raise GuardUnavailable("the v2 embedding calibration LOCK v2 records cannot be used: %s (fail closed)" % e)
    if rec is None:
        return None, {"checked": False, "why": why, "lock": str(lock_path)}
    return rec, {"checked": True, "decided_by": EC.DECISION, "protocol": rec["protocol"], "file": rec["file"],
                 "cos_threshold": rec["cos_threshold"], "strict_threshold": rec["strict_threshold"],
                 "embedder": rec["embedder"], "p_false": rec["p_false"], "role": rec["role"],
                 "known_limits": [x.get("family") for x in rec["known_limits"]],
                 "judged_by": "inc2.step1_stream's copy scan (it releases or refuses the h6_scan hold)"}


def l5_hit(l5, sha, variants):
    """(key, bits) of the L-5 image the intake image copies (its bytes, or any
    of its eight variants within 6 bits of an L-5 image's dHash, as GuardV2's
    base_copy reads the base), else None."""
    shas, idx, _rec = l5
    if sha in shas:
        return ("sha256", 0)
    best = None
    for v in (variants or {}).values():
        m = idx.find(int(v))
        if m is not None and (best is None or m[1] < best[1]):
            best = m
    return best


def _eval_matches(g, variants):
    """inc2.eval_hits.eval_matches(g, variants): every evaluation image within
    the never-train radius of a refused image, or None (a stand-in guard)."""
    try:
        from ..inc2 import eval_hits as EH
        return EH.eval_matches(g, variants)
    except Exception:  # noqa: BLE001 - without the list the guard's own match is weighed alone
        return None


def score_eval_hits(hits, copy_scan, lock_path=None, embedder=None, eval_desc=None):
    """D28-v2's evidence for this batch (inc2.eval_hits; docs/CONTINUOUS_LOOP.md,
    amendment 2026-10-03): the pair cosine of every image the guard refused as
    a dHash copy of an evaluation image (near_eval_v2, near_eval_variant), with
    the evaluation image its match names. It is computed here because the
    guard drops those images and nothing after intake can describe them. Both
    images are described as the copy scan describes them, by the embedder the
    v2 calibration bound to this intake names (copy_scan, from LOCK v2); the
    evaluation image comes from the evaluation manifests LOCK v2 records. Only
    the refused hits are described, so a batch without one loads no model.

    Each hit is weighed against the guard's match and every other evaluation
    image within the never-train radius ("also", _eval_matches); its pair
    cosine is the highest. eval_desc(split, key): the copy scan's own
    evaluation descriptors, when the caller holds them (rescore_eval_hits run
    by step1_stream eval-hits); the others are described from the manifests.

    Returns (record, {hit key: result}); the record goes into summary.json as
    "eval_hits" (inc2.eval_hits.record). Without a bound calibration (a testing
    LOCK, a stand-in guard), or when anything fails, the hits are recorded
    unscored with the reason, and D28 reads them by its one-hit rule (fail
    closed). Never raises: the guard has already refused the images, this only
    weighs them."""
    try:
        from ..inc2 import eval_hits as EH
    except ImportError as e:
        return {"hits": len(hits), "scored": 0, "why": "inc2.eval_hits is not installed (%s)" % e}, {}
    if not hits:
        return EH.record([], {}), {}
    if copy_scan is None:
        return EH.record(hits, {}, why="no v2 embedding calibration is bound to this intake: the hits are not "
                                       "weighed"), {}
    name = copy_scan.get("embedder")
    kw = {"embedder_name": name, "copy_threshold": copy_scan.get("cos_threshold"),
          "calibration": {"path": (copy_scan.get("file") or {}).get("path"),
                          "sha256": (copy_scan.get("file") or {}).get("sha256"), "format": copy_scan.get("format")}}
    try:
        from ..inc2 import guard as G2
        if embedder is None:
            from ..funnel import embed as FE
            model, _, pooling = str(name).rpartition(":")
            embedder = FE.LazyEmbedder(model, pooling or "cls")
        if embedder.name != name:
            return EH.record(hits, {}, why="the embedder %s is not the calibration's %s" % (embedder.name, name),
                             **kw), {}
        paths = {(str(s), str(r["key"])): r["image"] for s, rows in G2.v2_eval_rows(lock_path).items() for r in rows}
    except Exception as e:  # noqa: BLE001 - no embedder or no evaluation rows: unscored, read fail closed
        return EH.record(hits, {}, why="the hits cannot be weighed (%s: %s)" % (type(e).__name__, str(e)[:200]),
                         **kw), {}
    items = [dict(h, eval_image=paths.get((str(h.get("split")), str(h.get("eval_key")))),
                  also=[dict(a, eval_image=paths.get((str(a.get("split")), str(a.get("eval_key")))))
                        for a in h.get("also") or () if isinstance(a, dict)]) for h in hits]
    scored = EH.pair_cosines(items, embedder, eval_desc=eval_desc)
    return EH.record(hits, scored, **kw), scored


# ------------------------------------------------------------------ keys
class Keys(object):
    """Unique, path-safe, length-capped keys, the same on every run. `used`
    holds the keys of earlier intake batches, so a key is unique across every
    batch (a later shard or a re-fetched version of a source never reuses an
    earlier row's key); `salt` (the fetch record's sha256) tells such a row
    apart."""

    def __init__(self, used=(), salt=None):
        self.used = set(used)
        self.salt = salt

    def make(self, source, rel):
        stem = os.path.splitext(rel)[0]
        key = "%s__%s" % (safe_name(source), safe_name(stem.replace("/", "__"), keep="._-"))
        tag = sha256_text("%s/%s" % (source, rel))[:8]
        if len(key.encode("utf-8")) > KEY_MAX:
            key = key.encode("utf-8")[:KEY_MAX - 10].decode("utf-8", "ignore") + "__" + tag
        if key in self.used:
            key = "%s__%s" % (key, tag)
        if key in self.used and self.salt:
            key = "%s__%s" % (key, str(self.salt)[:8])
        if key in self.used:
            raise CollectError("key collision for %s/%s" % (source, rel))
        self.used.add(key)
        return key


# ------------------------------------------------------------------ extraction
def _is_archive(name):
    n = name.lower()
    return any(n.endswith(e) for e in ARCHIVE_EXTS)


def _strip_archive_ext(name):
    n = name
    for e in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if n.lower().endswith(e):
            return n[:-len(e)]
    return n


def _safe_member(root, name):
    target = (root / name).resolve()
    if not str(target).startswith(str(root.resolve()) + os.sep) and target != root.resolve():
        raise CollectError("archive member %r escapes the work directory" % name)
    return target


def extract(blob, name, dest):
    """Extract one archive into dest (members may not escape it); returns the
    number of files written."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(blob) as zf:
            for m in zf.infolist():
                if m.is_dir():
                    continue
                t = _safe_member(dest, m.filename)
                t.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(t, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                n += 1
        return n
    with tarfile.open(blob) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            t = _safe_member(dest, m.name)
            t.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(t, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            n += 1
    return n


def materialise(sdir, fetch_doc, root, fetch_sha=None):
    """The source tree: archives extracted under root/<archive name without
    its extension>/ (nested archives once more), other files linked at their
    names. A tree marked complete is reused. The marker records the fetch
    record's sha256 when the caller gives it (intake, after the arrival check
    hashed every blob): tree_fetch_sha reads it."""
    root = Path(root)
    done = root / ".complete"
    if done.is_file():
        return {"reused": True}
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    files = 0
    for f in fetch_doc["files"]:
        blob = sdir / "blobs" / f["sha256"]
        rel = f["name"].lstrip("/")
        if _is_archive(rel):
            sub = root / _strip_archive_ext(rel)
            files += extract(blob, rel, sub)
            inner = []
            for dirpath, _dirs, fnames in os.walk(sub):      # one listing of the fresh tree
                inner.extend(Path(dirpath) / fn for fn in fnames if _is_archive(fn))
            for p in sorted(inner):
                files += extract(p, p.name, p.parent / _strip_archive_ext(p.name))
                os.unlink(p)
        else:
            t = _safe_member(root, rel)
            t.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(blob, t)
            except OSError:
                shutil.copyfile(blob, t)
            files += 1
    done.write_text(json.dumps({"utc": utc(), "fetch_sha256": fetch_sha}) + "\n")
    return {"reused": False, "files": files}


def tree_fetch_sha(root):
    """The fetch sha256 a complete tree's marker records (materialise), else
    None: no tree, an incomplete one, or a marker written without it (before
    2026-10-03, or by rescore_eval_hits)."""
    try:
        doc = json.loads((Path(root) / ".complete").read_text())
    except (OSError, ValueError):
        return None
    return str(doc["fetch_sha256"]) if isinstance(doc, dict) and doc.get("fetch_sha256") else None


# ------------------------------------------------------------------ helpers
def _read_options(cfg, fetch_doc):
    """normalize.read's options for a fetch: its known item's format and
    format options, else the class names the fetch declares."""
    ki = cfg.known_item(fetch_doc["known_item"]) if fetch_doc.get("known_item") else None
    opts = dict((ki or {}).get("format_options") or {})
    if (ki or {}).get("format"):
        opts["format"] = ki["format"]
    if not opts.get("class_names") and fetch_doc.get("classes"):
        opts["class_names"] = [c.get("name") for c in fetch_doc["classes"]]
    return opts


def read_fetch(sdir):
    """(fetch.json, its sha256), its format checked; the blobs unread."""
    p = sdir / "fetch.json"
    if not p.is_file():
        raise Refusal("not_fetched", "no fetch record at %s: fetch the source first (lever L16)" % p, action="refuse")
    doc = read_json(p, "fetch record")
    if doc.get("format") != FORMATS["fetch"]:
        raise StaleInput("%s is not a %s" % (p, FORMATS["fetch"]))
    return doc, sha256_file(p)


def check_blobs(sdir, doc, hashed=True):
    """Every blob the fetch record lists: present, and hashing as recorded
    (StaleInput on the first that is missing or changed). hashed=False checks
    presence and recorded size only, for a continuation shard that reads no
    blob (intake, step 1b)."""
    for f in doc.get("files") or []:
        b = sdir / "blobs" / f["sha256"]
        if not b.is_file():
            raise StaleInput("staging blob of %s (%s) is missing" % (f["name"], f["sha256"][:12]))
        if not hashed:
            if f.get("bytes") is not None and b.stat().st_size != int(f["bytes"]):
                raise StaleInput("staging blob of %s (%s) has %d bytes, recorded %s"
                                 % (f["name"], f["sha256"][:12], b.stat().st_size, f["bytes"]))
            continue
        got = sha256_file(b)
        if got != f["sha256"]:
            raise StaleInput("staging blob of %s changed in transit (%s, recorded %s)"
                             % (f["name"], got[:12], f["sha256"][:12]))


def verify_staging(sdir):
    """fetch.json and every blob it lists, checked (StaleInput on the first
    that is missing or changed)."""
    doc, sha = read_fetch(sdir)
    check_blobs(sdir, doc)
    return doc, sha


def cap_items(items, cap, seed):
    """(rels taken, record) when more than `cap` items carry a box (config
    budgets.intake_max_images), else (None, None). The taken items are drawn
    round-robin over class sets (the image's source class ids), inside one
    class set round-robin over capture groups, inside a group in an order
    seeded by `seed` (the fetch record's sha256): every class set, then every
    group (a video's frames), is reached before any gets a second image, and
    the same fetch takes the same images on every run. The rest are deferred,
    not judged, and recorded as such (decision 'deferred'): the next intake of
    the fetch takes its next shard from them (shard_state), drawn here again
    over the items still undecided, with the same seed."""
    boxed = [it for it in items if it["boxes"]]
    if not cap or len(boxed) <= int(cap):
        return None, None
    cap = int(cap)

    def h(*xs):
        return hashlib.sha256("\0".join([str(seed)] + [str(x) for x in xs]).encode()).hexdigest()

    by_cls = {}
    for it in boxed:
        ck = ",".join(sorted({str(b[0]) for b in it["boxes"]}))
        by_cls.setdefault(ck, {}).setdefault(str(it.get("group") or it["rel"]), []).append(it)
    queues = {}
    for ck, groups in by_cls.items():
        lists = [sorted(v, key=lambda x: h("item", x["rel"]))
                 for _g, v in sorted(groups.items(), key=lambda kv: h("group", ck, kv[0]))]
        q = []
        for j in range(max(len(l) for l in lists)):
            q.extend(l[j] for l in lists if j < len(l))
        queues[ck] = q
    order = sorted(queues, key=lambda ck: h("class", ck))
    taken, j = [], 0
    while len(taken) < cap:
        for ck in order:
            if j < len(queues[ck]):
                taken.append(queues[ck][j]["rel"])
                if len(taken) == cap:
                    break
        j += 1
    return set(taken), {"cap": cap, "eligible": len(boxed), "taken": len(taken), "deferred": len(boxed) - len(taken),
                        "class_sets": len(by_cls), "groups": sum(len(g) for g in by_cls.values()),
                        "seed": str(seed)[:12], "basis": "round-robin: class sets, capture groups, seeded order"}


def _earlier_manifests(inc):
    out = []
    for b in read_jsonl(batches_ledger(inc), missing_ok=True):
        p = intake_dir(inc) / b["batch"] / "manifest.jsonl"
        if p.is_file():
            out.extend(read_jsonl(p))
    return out


def _deferred_after(summary):
    """The images a committed batch left deferred: its shard record's, else
    (a batch committed before shards existed, always the first of its fetch)
    its yield's images_deferred."""
    sh = summary.get("shard") if isinstance(summary.get("shard"), dict) else {}
    if sh.get("deferred_remaining") is not None:
        return int(sh["deferred_remaining"])
    return int(((summary.get("yield") or {}).get("images_deferred")) or 0)


def shard_state(inc, source_id, fetch_sha):
    """The committed batches of this fetch record (step 1b), or None when
    there is none: {"batches": [names, in commit order], "n": the next
    shard's number, "decided": the rels they decided (any decision but
    'deferred'), "remaining": the rels deferred and not decided since}. Each
    batch's decisions.jsonl must hash to the sha256 its summary.json records
    (CollectError otherwise: the partition cannot be trusted). When the last
    batch records no image deferred, its decisions are not read: the fetch is
    complete and the intake is the no-op."""
    rows = [b for b in read_jsonl(batches_ledger(inc), missing_ok=True)
            if b.get("source") == source_id and b.get("fetch_sha256") == fetch_sha]
    if not rows:
        return None
    names = [str(b["batch"]) for b in rows]
    summ = {}
    for nm in names:
        p = intake_dir(inc) / nm / "summary.json"
        if not p.is_file():
            raise CollectError("intake batch %s is in batches.jsonl without its %s" % (nm, p))
        summ[nm] = read_json(p, "intake summary")
    out = {"batches": names, "n": len(names) + 1, "decided": set(), "remaining": set(), "last": rows[-1]}
    if _deferred_after(summ[names[-1]]) <= 0:
        return out
    deferred = set()
    for nm in names:
        dp = intake_dir(inc) / nm / "decisions.jsonl"
        want = ((summ[nm].get("decisions") or {}) if isinstance(summ[nm].get("decisions"), dict) else {}).get("sha256")
        if not dp.is_file() or not want or sha256_file(dp) != want:
            raise CollectError("%s does not hash to the sha256 its summary.json records: the next shard of %s is "
                               "not drawn" % (dp, source_id))
        for d in read_jsonl(dp):
            if d.get("kind") not in ("image", "label") or not d.get("rel"):
                continue
            if d.get("decision") == "deferred":
                deferred.add(str(d["rel"]))
            else:
                out["decided"].add(str(d["rel"]))
    out["remaining"] = deferred - out["decided"]
    return out


def _link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _label_text(boxes):
    return "".join("%d %.6f %.6f %.6f %.6f\n" % (int(b[0]), b[1], b[2], b[3], b[4]) for b in boxes)


def _registry_count(path):
    """The number of datasets of an existing registry, read strictly: None
    when the file is absent; a CollectError when it exists but is not a
    registry (unparseable, or no datasets map). registry_lock.update_registry
    reads an unparseable file as an empty registry and would write that back,
    erasing every other source: the collector never lets it."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise CollectError("the dataset registry %s exists but does not read as JSON (%s): not registered; "
                           "restore it before the next intake" % (p, e))
    if not isinstance(doc, dict) or not isinstance(doc.get("datasets"), dict):
        raise CollectError("the dataset registry %s has no datasets map: not registered" % p)
    return len(doc["datasets"])


def _register(cfg, source_id, fetch_doc, bdir, batch, n_rows, class_names, lic, registry_path=None,
              research_only=None, override=None):
    from ..registry_lock import update_registry
    path = registry_path or PF.registry_path()
    before = _registry_count(path)
    ro = bool(lic.get("research_only")) if research_only is None else bool(research_only)

    def mutate(reg):
        if before is not None and (not isinstance(reg.get("datasets"), dict) or len(reg["datasets"]) < before):
            raise CollectError("the dataset registry %s read under its lock holds %s datasets, %d before: it did "
                               "not read whole, so nothing is written" % (path, len(reg.get("datasets") or {}),
                                                                           before))
        ds = reg.setdefault("datasets", {})
        cur = ds.get(source_id)
        if isinstance(cur, dict) and cur.get("annotation") != ANNOTATION:
            raise CollectError("%s is already a registry source with annotation %r: never registered twice"
                               % (source_id, cur.get("annotation")))
        e = dict(cur or {})
        batches = list(e.get("intake_batches") or [])
        if batch not in batches:
            batches.append(batch)
        e.update({"source": fetch_doc["provider"], "annotation": ANNOTATION, "status": "intake",
                  "local_path": str(bdir), "intake_batches": batches, "class_names": class_names,
                  "source_ref": fetch_doc["ref"], "source_version": fetch_doc.get("version"),
                  "license": lic["id"], "provenance": {"license": lic["id"], "license_class": lic["class"],
                                                       "license_evidence": lic.get("evidence"), "research_only": ro,
                                                       "licence_override": override},
                  "images": int(e.get("images") or 0) + int(n_rows), "registered_by": "collect intake",
                  "used_for_training": False, "updated_utc": utc()})
        ds[source_id] = e
    update_registry(str(path), mutate)
    return str(path)


# ------------------------------------------------------------------ intake
def intake(cfg, source_id, inc=None, guard=None, names=None, targets=None, testing=False, registry_path=None,
           lock_path=None, now=None, keep_work=False, embedder=None):
    """Intake one fetched source (module docstring). Returns the result
    record; a refusal raises (Refusal, NamesPending, GuardUnavailable,
    StaleInput) after recording a held or closed event where the state
    changes. embedder: what describes the dHash hits for D28-v2
    (score_eval_hits); None makes the v2 calibration's own embedder when a
    batch has a hit."""
    from . import names as NM
    from .targets import Targets
    t0 = time.time()
    with intake_lock(inc, what="collect intake %s" % source_id):
        sdir = staging_dir(source_id, inc)
        fetch_doc, fetch_sha = read_fetch(sdir)
        # 1b. the committed batches of this fetch record, and what they left deferred
        shard = shard_state(inc, source_id, fetch_sha)
        work = intake_dir(inc) / "work" / safe_name(source_id)
        root = work / fetch_sha[:12] / "x"
        # a continuation shard over a complete tree extracted from this very fetch record, after its blobs hashed
        # as recorded (only intake's materialise writes the fetch sha256 into a tree's marker, and only after
        # check_blobs hashed them), reads no blob: materialise reuses the tree. So the blobs are not read again
        # (tens of GB for the largest sources); the batch's every image comes from that tree. Every other intake
        # hashes them, as before
        reuse = bool(shard and shard["remaining"]) and tree_fetch_sha(root) == fetch_sha
        check_blobs(sdir, fetch_doc, hashed=not reuse)
        arrival = {"fetch_sha256": fetch_sha, "blobs": "presence and size (the tree extracted from them after they "
                                                       "hashed as recorded is reused; no blob is read)"
                   if reuse else "sha256"}
        if shard is not None and not shard["remaining"]:
            b = shard["last"]
            return {"status": "already_intaken", "source": source_id, "batch": b["batch"], "rows": b.get("rows"),
                    "shards": len(shard["batches"]), "deferred_remaining": 0}
        # 2. gates that need no image
        lic = fetch_doc.get("licence") or {}
        if lic.get("class") == "refused":
            S.append(inc, source_id, "closed", reason="licence_refused", codes=["licence_refused"], stage="intake")
            raise Refusal("licence_refused", "licence %r is not research-usable" % lic.get("id"), action="close")
        person_licence = licence_override(cfg, source_id)
        if lic.get("class") == "unresolved" and not person_licence:
            S.append(inc, source_id, "held", reason="licence_unresolved", codes=["licence_unresolved"],
                     risk="R3", stage="intake")
            raise Refusal("licence_unresolved", "licence %r is unresolved (P6)" % lic.get("id"), action="hold",
                          risk="R3")
        reg = PF.load_registry(registry_path).get(source_id)
        if isinstance(reg, dict) and str(reg.get("status")) == "quarantined":
            S.append(inc, source_id, "closed", reason="quarantined", codes=["quarantined"], stage="intake")
            raise Refusal("quarantined", "the registry quarantines %s" % source_id, action="close")
        if isinstance(reg, dict) and reg.get("annotation") != ANNOTATION:
            raise Refusal("registered_outside_intake", "%s is a registry source with annotation %r"
                          % (source_id, reg.get("annotation")), action="close")
        nf = PF._never_fetch(dict(fetch_doc, source_id=source_id), cfg, PF.never_train_slugs())
        if nf:
            S.append(inc, source_id, "closed", reason="never_train", codes=["never_train"], stage="intake",
                     detail=nf[:500])
            raise Refusal("never_train", "%s: %s" % (source_id, nf), action="close")
        # 3. the guard, before anything is written
        g = guard if guard is not None else default_guard(lock_path)
        # the L-5 exclusions (a stand-in guard given without a LOCK, in tests, checks none: recorded)
        l5 = load_l5(lock_path) if (guard is None or lock_path is not None) else None
        # the embedding calibration that judges the rows held for the copy scan (L-9(c))
        if guard is None or lock_path is not None:
            copy_scan, copy_scan_rec = load_copy_scan(lock_path)
        else:
            copy_scan, copy_scan_rec = None, {"checked": False, "why": "a stand-in guard was given without a LOCK"}
        earlier = _earlier_manifests(inc)
        # the earlier shards of this fetch are judged with this one as one batch (step 1b): their images seed the
        # exact-duplicate check and stay out of the intake index, as a batch's own images do. In a capped batch of
        # video frames, 46 % of the rows were measured within 3 bits of another row of the batch
        # (docs/CONTINUOUS_LOOP.md, amendment 2026-10-03), so near_dup_intake against the earlier shards would
        # refuse about half of every later shard
        mine = set(shard["batches"]) if shard else set()
        same_fetch = [r for r in earlier if r.get("batch") in mine]
        for r in earlier:
            if r.get("dhash") is not None and r.get("batch") not in mine:
                g.add_intake(int(r["dhash"]), (r.get("batch"), r.get("key")))
        # 4-5. tree, normalise, class map
        try:
            tree_rec = materialise(sdir, fetch_doc, root, fetch_sha=fetch_sha)
        except (zipfile.BadZipFile, tarfile.TarError, EOFError) as e:
            S.append(inc, source_id, "held", reason="archive_unreadable", codes=["archive_unreadable"], risk="R3",
                     stage="intake", detail=str(e)[:500])
            raise Refusal("archive_unreadable", "an archive of %s does not read (%s)" % (source_id, e), action="hold",
                          risk="R3")
        except CollectError as e:                   # a member that would leave the work directory: hostile
            S.append(inc, source_id, "closed", reason="archive_escape", codes=["archive_escape"], stage="intake",
                     detail=str(e)[:500])
            raise Refusal("archive_escape", str(e), action="close")
        try:
            tree, res = NZ.read(root, _read_options(cfg, fetch_doc),
                                out_images=work / fetch_sha[:12] / "parquet_images")
        except NormaliseError as e:
            S.append(inc, source_id, "held", reason="normalise_failed", codes=["normalise_failed"], risk="R3",
                     stage="intake", detail=str(e)[:500])
            raise Refusal("normalise_failed", str(e), action="hold", risk="R3")
        items = sorted(res.items, key=lambda x: x["rel"])
        orphans = []
        if shard is not None:
            # a continuation shard: the items no earlier batch of this fetch decided (its candidates); a deferred
            # image the source as read now no longer holds is decided here (not_in_source), so the shards end
            items = [it for it in items if it["rel"] not in shard["decided"]]
            orphans = sorted(shard["remaining"] - {it["rel"] for it in items})
        # the legacy-label copy rule (§7.2), again on the class list the files declare: a provider that
        # declares none before download (Kaggle, a repository archive) is judged here, before anything is written
        pf = cfg.raw["prefilter"]
        legacy = set(pf["legacy_labels"])
        legacy_equal = sorted({str(c.get("name")) for c in res.classes if str(c.get("name")) in legacy})
        if len(legacy_equal) >= int(pf["copy_candidate_min"]):
            S.append(inc, source_id, "closed", reason="copy_candidate", codes=["copy_candidate"], stage="intake",
                     detail="%d class names equal the reference dataset's legacy labels: %s"
                            % (len(legacy_equal), legacy_equal))
            raise Refusal("copy_candidate", "%s declares %d of the reference dataset's legacy labels (%s): a probable "
                          "re-export, closed" % (source_id, len(legacy_equal), ", ".join(legacy_equal)),
                          action="close")
        names = names if names is not None else NM.load(cfg, inc)
        targets = targets or Targets(cfg, names)
        cmap = CM.build(source_id, res.classes, cfg, names, targets)
        if cmap["pending"]:
            write_json_atomic(work / "pending_names.json",
                              dict(header("pending_names", cfg, testing=testing), source=source_id,
                                   names=cmap["pending"], classes=[{"id": c["id"], "name": c["name"],
                                                                    "hints": c.get("hints")} for c in res.classes]))
            S.append(inc, source_id, "held", reason="names_pending", codes=["names_pending"], stage="intake",
                     names=cmap["pending"][:200])
            raise NamesPending(source_id, cmap["pending"])
        # 6. the batch
        batches = read_jsonl(batches_ledger(inc), missing_ok=True)
        batch = "i%04d_%s" % (len(batches) + 1, safe_name(source_id)[:60])
        bdir = intake_dir(inc) / batch
        if bdir.exists():
            if (bdir / "summary.json").exists():
                raise CollectError("%s exists and is committed but not in batches.jsonl" % bdir)
            shutil.rmtree(bdir)                            # a killed intake: redo it whole
        (bdir / "images").mkdir(parents=True)
        (bdir / "labels").mkdir(parents=True)
        lab_group = fetch_doc.get("lab_group")
        # cleared only when the fetch said so AND the config still clears this very record (the primary
        # record of a cleared known item, outside the evaluation labs): a fetch record is never trusted alone
        cleared = bool(fetch_doc.get("provenance_cleared")) and not fetch_doc.get("evaluation_lab") \
            and lab_group not in cfg.evaluation_labs() \
            and PF.cleared_by_config(cfg, source_id, fetch_doc.get("provider"), fetch_doc.get("ref"))
        holds = ([] if cleared else ["h6_scan"]) + (["licence"] if lic.get("class") == "unresolved" else [])
        hold = holds[0] if holds else None
        exhaustive = fetch_doc.get("exhaustive_labels")
        lic_rec = dict(lic) if not person_licence else dict(lic, override=person_licence)
        # P6, §8: research-only when the licence is, or when the person's override says so; the licence and its
        # class stay as the fetch recorded them (still "unresolved" under an override), so provenance is honest
        research_only = bool(lic.get("research_only")) or bool(isinstance(person_licence, dict)
                                                               and person_licence.get("research_only") is True)
        n_targets = len(cfg.targets)
        keys = Keys(used=(r.get("key") for r in earlier if r.get("key")), salt=fetch_sha)
        decisions, manifest = [], []
        eval_copies = []                   # dHash copies of evaluation images, kept until they are weighed (D28-v2)
        # an exact byte copy of an earlier shard's row is a duplicate within the fetch's one batch (step 1b)
        seen_sha = {r["sha256"]: r.get("key") for r in same_fetch if r.get("sha256")}
        reasons = {}
        per_id = {}
        per_src = {}
        cmap_names = {c["src_id"]: c["name"] for c in cmap["classes"]}
        now_s = utc()
        decisions.append({"kind": "source", "source": source_id, "decision": "kept", "reason": "fetched",
                          "provider": fetch_doc["provider"], "ref": fetch_doc["ref"],
                          "prefilter": fetch_doc.get("decision")})
        for c in cmap["classes"]:
            decisions.append({"kind": "class", "source": source_id, "src_id": c["src_id"], "name": c["name"],
                              "decision": "mapped", "inc_id": c["inc_id"], "cls": c["cls"], "status": c["status"],
                              "via": c["via"], "basis": c["basis"], "reason": c["status"] or "pending"})
        for f in fetch_doc["files"]:
            decisions.append({"kind": "file", "source": source_id, "name": f["name"], "sha256": f["sha256"],
                              "decision": "archive" if _is_archive(f["name"]) else "file", "reason": "fetched"})
        # the files of the source that are not items were decided by its first batch: a later shard decides only
        # its own images
        done = shard["decided"] if shard is not None else set()
        no_label = [rel for rel in res.images_without_labels if rel not in done]
        for rel in no_label:
            decisions.append({"kind": "image", "source": source_id, "rel": rel, "decision": "rejected",
                              "reason": "no_label"})
            reasons["no_label"] = reasons.get("no_label", 0) + 1
        for rel in res.labels_without_images:
            if rel in done:
                continue
            decisions.append({"kind": "label", "source": source_id, "rel": rel, "decision": "rejected",
                              "reason": "no_image"})
            reasons["no_image"] = reasons.get("no_image", 0) + 1
        if res.rows_without_images and shard is None:
            decisions.append({"kind": "table_rows", "source": source_id, "decision": "rejected",
                              "reason": "image_not_fetched", "count": res.rows_without_images})
        for rel in orphans:
            decisions.append({"kind": "image", "source": source_id, "rel": rel, "decision": "rejected",
                              "reason": "not_in_source"})
            reasons["not_in_source"] = reasons.get("not_in_source", 0) + 1
        candidates = sum(1 for it in items if it["boxes"])
        taken, cap_rec = cap_items(items, cfg.intake_max_images(), fetch_sha)
        t_budget = cfg.intake_max_seconds()
        time_deferred = 0
        if taken is not None:
            for it in items:
                if it["boxes"] and it["rel"] not in taken:
                    decisions.append({"kind": "image", "source": source_id, "rel": it["rel"], "decision": "deferred",
                                      "reason": "over_intake_cap"})
            items = [it for it in items if not it["boxes"] or it["rel"] in taken]
        try:
            for idx, it in enumerate(items):
                rel = it["rel"]
                if t_budget and it["boxes"] and time.time() - t0 > t_budget:
                    # the job's wall clock (budgets.intake_max_seconds): the rest is deferred, not judged, so
                    # the batch commits inside the job's limit instead of timing out with nothing
                    late = [x for x in items[idx:] if x["boxes"]]
                    for x in late:
                        decisions.append({"kind": "image", "source": source_id, "rel": x["rel"],
                                          "decision": "deferred", "reason": "over_intake_time"})
                    time_deferred = len(late)
                    break
                # a box whose source class is not in the class list (normalize.NO_CLASS, or an id the list lacks)
                # is a plant of unknown class: it keeps its place with the unmapped id, so per-box admission masks
                # it, never dropped (a dropped box leaves an unlabelled plant in a kept image, §3.2)
                boxes = [(cmap["by_src"].get(b[0], cfg.unmapped_id),) + tuple(b[1:]) for b in it["boxes"]]
                unlisted = sum(1 for b in it["boxes"] if b[0] not in cmap["by_src"])
                if not boxes:
                    why = "no_box"
                    decisions.append({"kind": "image", "source": source_id, "rel": rel, "decision": "rejected",
                                      "reason": why, "bad_boxes": it["bad"]})
                    reasons[why] = reasons.get(why, 0) + 1
                    continue
                key = keys.make(source_id, rel)
                ext = os.path.splitext(rel)[1].lower() or ".jpg"
                img = bdir / "images" / (key + ext)
                _link_or_copy(Path(it["path"]), img)
                sha = sha256_file(img)
                if sha in seen_sha:
                    os.unlink(img)
                    decisions.append({"kind": "image", "source": source_id, "rel": rel, "key": key,
                                      "decision": "rejected", "reason": "exact_dup_intake", "twin": seen_sha[sha]})
                    reasons["exact_dup_intake"] = reasons.get("exact_dup_intake", 0) + 1
                    continue
                reason, match, (dh, _var) = g.check_path(img)
                if not reason and l5 is not None:
                    hit = l5_hit(l5, sha, _var)
                    if hit is not None:
                        reason, match = L5_REASON, {"key": hit[0], "bits": hit[1], "why": "decision L-5 drops these "
                                                    "images from training outright"}
                if reason:
                    dec = {"kind": "image", "source": source_id, "rel": rel, "key": key, "decision": "rejected",
                           "reason": reason, "match": match, "sha256": sha}
                    decisions.append(dec)
                    reasons[reason] = reasons.get(reason, 0) + 1
                    if reason in DHASH_EVAL_REASONS:
                        # refused like any other; the file is removed once its pair cosine is taken (below)
                        m = match if isinstance(match, dict) else {}
                        eval_copies.append({"key": key, "source": source_id, "image": str(img), "split": m.get("split"),
                                            "eval_key": m.get("key"), "also": _eval_matches(g, _var), "decision": dec})
                    else:
                        os.unlink(img)
                    continue
                seen_sha[sha] = key
                text = _label_text(boxes)
                lab = bdir / "labels" / (key + ".txt")
                lab.write_text(text)
                for b in it["boxes"]:
                    nm = cmap_names.get(b[0], "(unlisted class %s)" % b[0])
                    per_src[nm] = per_src.get(nm, 0) + 1
                counts = {}
                for b in boxes:
                    counts[b[0]] = counts.get(b[0], 0) + 1
                    per_id[b[0]] = per_id.get(b[0], 0) + 1
                tb = sum(v for k, v in counts.items() if 0 <= k < n_targets)
                wh = it.get("wh") or (None, None)
                row = {"key": key, "image": str(img), "sha256": sha, "label": str(lab),
                       "label_sha256": sha256_text(text), "source": source_id, "session": it["group"],
                       "capture_group": it["group"], "capture_group_basis": it["group_basis"], "licence": lic.get("id"),
                       "licence_class": lic.get("class"), "research_only": research_only,
                       "licence_override": person_licence, "lab_group": lab_group, "dhash": int(dh),
                       "batch": batch, "hold_until": hold, "holds": holds, "provenance_cleared": cleared,
                       "exhaustive_labels": exhaustive,
                       "boxes": len(boxes), "target_boxes": tb, "unmapped_boxes": counts.get(cfg.unmapped_id, 0),
                       "class_ids": sorted(counts), "width": wh[0], "height": wh[1], "rel": rel, "intake_utc": now_s,
                       "bad_boxes": it["bad"], "unlisted_class_boxes": unlisted, "clipped_boxes": it["clipped"]}
                if "h6_scan" in holds and copy_scan is not None:
                    row["copy_scan_calibration"] = {"file": copy_scan["file"]["path"],
                                                    "sha256": copy_scan["file"]["sha256"],
                                                    "cos_threshold": copy_scan["cos_threshold"]}
                manifest.append(row)
                decisions.append({"kind": "image", "source": source_id, "rel": rel, "key": key, "decision": "kept",
                                  "reason": "kept", "target_boxes": tb, "boxes": len(boxes)})
            # D28-v2: weigh each dHash copy of an evaluation image by its pair cosine
            eval_rec, scored = score_eval_hits([{k: v for k, v in h.items() if k != "decision"}
                                                for h in eval_copies], copy_scan, lock_path, embedder)
        finally:
            # the copies of evaluation images never outlive this block, whatever fails in it (an unreadable
            # image, a full disk, the embedder): they are removed here, before anything else is written
            for h in eval_copies:
                if os.path.exists(h["image"]):
                    os.unlink(h["image"])
        for h in eval_copies:
            got = scored.get(h["key"]) or {}
            h["decision"]["pair_cos"] = got.get("pair_cos")
            if got.get("pair_cos") is None:
                h["decision"]["pair_cos_why"] = got.get("why") or eval_rec.get("why")
            elif got.get("best") and got.get("weighed", 0) > 1:
                h["decision"]["pair_cos_best"] = got["best"]
        manifest.sort(key=lambda r: r["key"])
        # 7. the batch files, summary last
        m_sha = write_jsonl_atomic(bdir / "manifest.jsonl", manifest)
        d_sha = write_jsonl_atomic(bdir / "decisions.jsonl", decisions)
        gcounts = {k: int(v) for k, v in dict(getattr(g, "counts", {}) or {}).items()}
        gdoc = header("guard", cfg, testing=testing)
        gdoc.update({"source": source_id, "batch": batch, "counts": {k: reasons.get(k, 0) for k in sorted(reasons)},
                     "guard_counts": gcounts, "index": g.index_record() if hasattr(g, "index_record") else None,
                     "l5_excluded": l5[2] if l5 is not None else {"checked": False, "why": "a stand-in guard was "
                                                                  "given without a LOCK"},
                     "earlier_intake_images": len(earlier) - len(same_fetch),
                     "same_fetch_shard_images": {"images": len(same_fetch), "batches": sorted(mine),
                                                 "why": "the earlier shards of this fetch: exact duplicates only, as "
                                                        "within one batch (not in the near_dup_intake index)"},
                     "copy_scan": copy_scan_rec,
                     "seen_index": "not read here: inc2.step1_stream applies exact_dup with its own seen index"})
        write_json_atomic(bdir / "guard.json", gdoc)
        src = header("sources", cfg, inputs={"fetch": {"path": str(sdir / "fetch.json"), "sha256": fetch_sha}},
                     testing=testing)
        src.update({"source_id": source_id, "provider": fetch_doc["provider"], "ref": fetch_doc["ref"],
                    "version": fetch_doc.get("version"), "title": fetch_doc.get("title"), "url": fetch_doc.get("url"),
                    "known_item": fetch_doc.get("known_item"), "licence": lic_rec, "lab_group": lab_group,
                    "lab_group_basis": fetch_doc.get("lab_group_basis"), "provenance_cleared": cleared,
                    "exhaustive_labels": exhaustive, "format": res.format, "normalise": res.summary(),
                    "classes": res.classes, "class_map": cmap, "fetch": {"sha256": fetch_sha, "bytes":
                                                                        fetch_doc.get("bytes"),
                                                                        "files": fetch_doc.get("files")}})
        write_json_atomic(bdir / "sources.json", src)
        _register(cfg, source_id, fetch_doc, bdir, batch, len(manifest), [c["name"] for c in res.classes], lic,
                  registry_path=registry_path, research_only=research_only, override=person_licence)
        kept_tb = sum(r["target_boxes"] for r in manifest)
        left = (cap_rec or {}).get("deferred", 0) + time_deferred      # the fetch's boxed images no batch decided
        seen_imgs = len(items) - time_deferred + len(no_label)
        eval_hits = sum(reasons.get(k, 0) for k in EVAL_REASONS)
        base_hits = reasons.get("base_copy", 0) + reasons.get(L5_REASON, 0)
        checked = len(manifest) + sum(reasons.get(k, 0) for k in ("unhashable",) + EVAL_REASONS + (
            "base_copy", L5_REASON, "exact_dup", "near_dup_intake"))
        leak = {"eval_share": round(eval_hits / checked, 4) if checked else 0.0,
                "base_share": round(base_hits / checked, 4) if checked else 0.0}
        leak["fires"] = leak["eval_share"] >= SOURCE_LEAK_EVAL_SHARE or leak["base_share"] >= SOURCE_LEAK_BASE_SHARE
        gb = float(fetch_doc.get("bytes") or 0) / 1e9
        yld = {"images_seen": seen_imgs, "images_kept": len(manifest), "target_images": sum(
            1 for r in manifest if r["target_boxes"] > 0), "target_boxes": kept_tb,
            "boxes_by_id": {str(k): v for k, v in sorted(per_id.items())},
            "boxes_by_class": {_class_name(cfg, k): v for k, v in sorted(per_id.items())},
            "boxes_by_source_class": dict(sorted(per_src.items())),
            "rejected": dict(sorted(reasons.items())), "bytes": fetch_doc.get("bytes"),
            "images_deferred": left,
            "target_boxes_per_gb": round(kept_tb / gb, 3) if gb > 0 else None}
        summ = header("summary", cfg, inputs={"manifest": {"path": str(bdir / "manifest.jsonl"), "sha256": m_sha},
                                              "decisions": {"path": str(bdir / "decisions.jsonl"), "sha256": d_sha},
                                              "fetch": {"path": str(sdir / "fetch.json"), "sha256": fetch_sha}},
                      testing=testing)
        guard_reasons = ("unhashable",) + EVAL_REASONS + ("base_copy", L5_REASON, "exact_dup", "near_dup_intake")
        by_reason = dict(sorted(reasons.items()))
        by_reason["kept"] = len(manifest)
        summ.update({"source": source_id, "batch": batch, "rows": len(manifest),
                     # the flat fields the autopilot reads (D28: guard refusals over the images the guard
                     # checked; the zero-yield card: every decision reason)
                     "images": checked, "guard": {k: reasons.get(k, 0) for k in guard_reasons},
                     "decisions": {"by_reason": by_reason, "file": "decisions.jsonl", "sha256": d_sha},
                     "yield": yld, "source_leak": leak,
                     # D28-v2: the pair cosines of the dHash copies of evaluation images (inc2.eval_hits.record)
                     "eval_hits": eval_rec,
                     "zero_yield": kept_tb == 0, "zero_yield_reasons": dict(sorted(reasons.items())) if kept_tb == 0
                     else None, "hold_until": hold, "lab_group": lab_group, "research_only": research_only,
                     "licence_override": person_licence,
                     "format": res.format, "copy_scan": copy_scan_rec, "intake_cap": cap_rec,
                     "intake_time": {"budget_s": t_budget, "deferred": time_deferred},
                     # step 1b: this batch's shard of its fetch record and what it leaves deferred (the stream
                     # proposes the next shard while deferred_remaining > 0)
                     "shard": {"n": shard["n"] if shard else 1, "fetch_sha256": fetch_sha,
                               "earlier_batches": list(shard["batches"]) if shard else [], "candidates": candidates,
                               "deferred_remaining": left, "tree": "reused" if tree_rec.get("reused") else "extracted",
                               "arrival": arrival},
                     "seconds": round(time.time() - t0, 3)})
        write_json_atomic(bdir / "summary.json", summ)
        append_chained(batches_ledger(inc), {"format": FORMATS["batch"], "ts": utc(), "batch": batch,
                                             "source": source_id, "fetch_sha256": fetch_sha, "manifest_sha256": m_sha,
                                             "rows": len(manifest), "target_boxes": kept_tb,
                                             "shard": shard["n"] if shard else 1, "deferred_remaining": left})
        S.append(inc, source_id, "intaken", provider=fetch_doc["provider"], ref=fetch_doc["ref"], batch=batch,
                 seconds=round(time.time() - t0, 3), shard=shard["n"] if shard else 1, deferred_remaining=left,
                 **{"yield": yld})
        if not keep_work and not left:
            # the extracted tree is kept while images remain deferred, so the next shard does not extract it again;
            # the batch that leaves none removes it
            shutil.rmtree(work / fetch_sha[:12], ignore_errors=True)
    return {"status": "intaken", "source": source_id, "batch": batch, "rows": len(manifest),
            "target_boxes": kept_tb, "rejected": dict(sorted(reasons.items())), "source_leak": leak,
            "zero_yield": kept_tb == 0, "dir": str(bdir), "shard": shard["n"] if shard else 1,
            "deferred_remaining": left}


# ------------------------------------------------------------------ D28-v2 sidecar
EVAL_HITS_NAME = "eval_hits.json"


def _dhash_hits(summary):
    """The dHash copies of evaluation images an intake summary counts (its
    guard's near_eval_v2 + near_eval_variant)."""
    g = summary.get("guard") if isinstance(summary.get("guard"), dict) else {}
    return sum(int(g.get(k) or 0) for k in DHASH_EVAL_REASONS)


def eval_hits_needed(summary):
    """True when an intake summary counts dHash copies of evaluation images
    that its own record (eval_hits) does not weigh in full: a batch committed
    before D28-v2, or one whose hits could not be weighed."""
    n = _dhash_hits(summary)
    if n <= 0:
        return False
    eh = summary.get("eval_hits") if isinstance(summary.get("eval_hits"), dict) else {}
    row = ((eh.get("per_source") or {}) if isinstance(eh.get("per_source"), dict) else {}).get(
        str(summary.get("source")))
    cos = (row or {}).get("pair_cos") if isinstance(row, dict) else None
    return not (isinstance(cos, list) and len([c for c in cos if c is not None]) >= n)


def rescore_eval_hits(cfg, batch, inc=None, lock_path=None, embedder=None, eval_desc=None, guard=None,
                      testing=False, force=False):
    """D28-v2's sidecar of an intake batch (docs/CONTINUOUS_LOOP.md, amendment
    2026-10-03): intake/<batch>/eval_hits.json, the pair cosines of the images
    its guard refused as dHash copies of evaluation images, for a batch whose
    summary.json does not weigh them (committed before the amendment, or a
    weighing that failed). The batch's own files are never rewritten.

    Deterministic, from the files the batch was judged on: decisions.jsonl
    (checked against the sha256 summary.json records) names each hit's rel,
    key, sha256 and the guard's match; the fetch it was intaken from
    (summary.json's fetch sha256; the staging record and blobs checked by
    verify_staging) is materialised and normalised again in a work directory
    of its own, removed at the end whatever happens; each hit image is the
    file at its rel, which must hash to the decision's sha256 and be refused
    again by GuardV2 (LOCK v2) with the same reason and match. It is then
    weighed as intake weighs a hit (score_eval_hits: the v2 calibration's
    embedder, the guard's match and every evaluation image within the
    radius; eval_desc, the copy scan's evaluation descriptors, when the
    caller holds them). A hit that cannot be re-derived or weighed is
    recorded unweighed with the reason, and D28 reads it by the one-hit rule
    (fail closed). The sidecar, which the snapshot ships, holds the record
    (inc2.eval_hits.record), the hit keys, their reasons and pair cosines:
    no pixels, no evaluation paths and no evaluation keys (decisions.jsonl
    names each hit's match).

    Returns {"status": "written" | "not_needed" | "exists", "batch", "path",
    "hits", "scored"}. Raises CollectError when the batch is not committed or
    its decisions do not hash as its summary records, GuardUnavailable when
    GuardV2 or the v2 calibration cannot be loaded, CollectError when hits
    were re-derived but none could be weighed (the embedder, not the batch),
    and LockHeld while another intake writes: no sidecar then, so the batch
    is tried again."""
    from ..inc2 import eval_hits as EH
    t0 = time.time()
    bdir = intake_dir(inc) / safe_name(batch)
    if safe_name(batch) != str(batch) or not (bdir / "summary.json").is_file():
        raise CollectError("intake batch %r is not committed (no %s)" % (batch, bdir / "summary.json"))
    summ = read_json(bdir / "summary.json", "intake summary")
    side = bdir / EVAL_HITS_NAME
    n = _dhash_hits(summ)
    out = {"batch": batch, "path": str(side), "hits": n, "scored": None}
    if n <= 0 or (not eval_hits_needed(summ) and not force):
        return dict(out, status="not_needed")
    if side.is_file() and not force:
        return dict(out, status="exists")
    dpath = bdir / "decisions.jsonl"
    want_d = ((summ.get("decisions") or {}) if isinstance(summ.get("decisions"), dict) else {}).get("sha256")
    if not dpath.is_file() or not want_d or sha256_file(dpath) != want_d:
        raise CollectError("%s does not hash to the sha256 %s records" % (dpath, bdir / "summary.json"))
    source = str(summ.get("source"))
    hits = [d for d in read_jsonl(dpath) if d.get("kind") == "image" and d.get("decision") == "rejected"
            and d.get("reason") in DHASH_EVAL_REASONS]
    if len(hits) != n:
        raise CollectError("%s lists %d dHash copies of evaluation images, its summary counts %d"
                           % (dpath, len(hits), n))
    want_f = (((summ.get("inputs") or {}).get("fetch") or {}) if isinstance(summ.get("inputs"), dict)
              else {}).get("sha256")
    failed, items = {}, []
    with intake_lock(inc, what="collect eval-hits %s" % batch):
        sdir = staging_dir(source, inc)
        work = intake_dir(inc) / "work" / safe_name(source) / ("eval_hits_%s" % str(want_f or "none")[:12])
        try:
            why = None
            try:
                fetch_doc, fetch_sha = verify_staging(sdir)
                if fetch_sha != want_f:
                    why = "the staging of %s now holds another fetch (%s, the batch was intaken from %s)" % (
                        source, fetch_sha[:12], str(want_f)[:12])
            except CollectError as e:
                fetch_doc, why = None, "the staging of %s cannot be read again (%s)" % (source, e)
            by_rel, g, copy_scan = {}, None, None
            if why is None:
                try:
                    materialise(sdir, fetch_doc, work / "x")
                    _tree, res = NZ.read(work / "x", _read_options(cfg, fetch_doc), out_images=work / "parquet_images")
                    by_rel = {it["rel"]: it["path"] for it in res.items}
                except Exception as e:  # noqa: BLE001 - nothing re-derived: every hit stays unweighed (fail closed)
                    why = "the batch's images cannot be re-derived (%s: %s)" % (type(e).__name__, str(e)[:200])
            if why is None:
                # the guard and the calibration are the environment, not the batch: without them this raises
                # (GuardUnavailable), and no sidecar uses up the batch's attempt
                g = guard if guard is not None else default_guard(lock_path)
                copy_scan, _rec = load_copy_scan(lock_path)
            for d in hits:
                key, m = str(d.get("key")), d.get("match") if isinstance(d.get("match"), dict) else {}
                if why is not None:
                    failed[key] = why
                    continue
                path = by_rel.get(d.get("rel"))
                if path is None:
                    failed[key] = "its rel %r is not in the source read again" % d.get("rel")
                    continue
                try:
                    img = work / "hits" / (key + (os.path.splitext(str(d.get("rel")))[1].lower() or ".jpg"))
                    _link_or_copy(Path(path), img)
                    if sha256_file(img) != d.get("sha256"):
                        failed[key] = "the file at its rel no longer hashes to the decision's sha256"
                        continue
                    reason, match, (_dh, var) = g.check_path(img)
                except Exception as e:  # noqa: BLE001 - this hit stays unweighed (fail closed)
                    failed[key] = "the re-derived image cannot be checked (%s: %s)" % (type(e).__name__, str(e)[:200])
                    continue
                mm = match if isinstance(match, dict) else {}
                if reason != d.get("reason") or (mm.get("split"), mm.get("key")) != (m.get("split"), m.get("key")):
                    # reasons only: this goes into the sidecar the snapshot ships, which names no evaluation image
                    # (decisions.jsonl keeps the recorded match)
                    failed[key] = "GuardV2 judges the re-derived image otherwise (%s, recorded %s%s)" % (
                        reason, d.get("reason"), "" if reason != d.get("reason")
                        else "; another evaluation image than the recorded match")
                    continue
                items.append({"key": key, "source": source, "image": str(img), "split": m.get("split"),
                              "eval_key": m.get("key"), "also": _eval_matches(g, var)})
            rec0, scored = score_eval_hits(items, copy_scan, lock_path, embedder, eval_desc) if items else ({}, {})
            if items and not any((scored.get(it["key"]) or {}).get("pair_cos") is not None for it in items):
                # re-derived but not one weighed: the calibration, the embedder or the evaluation images failed,
                # not the batch; a sidecar now would use up its attempt, so none is written and the job fails
                raise CollectError("the re-derived hits of %s could not be weighed (%s): no sidecar is written"
                                   % (batch, rec0.get("why") or sorted({(scored.get(it["key"]) or {}).get("why")
                                                                        for it in items} - {None})[:2]))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    merged = {}
    for d in hits:
        key = str(d.get("key"))
        got = scored.get(key) or {}
        if key in failed:
            merged[key] = {"pair_cos": None, "why": failed[key]}
        elif got.get("pair_cos") is not None:
            merged[key] = got
        else:
            merged[key] = {"pair_cos": None, "why": got.get("why") or rec0.get("why") or "not weighed"}
    rec = EH.record([{"key": str(d.get("key")), "source": source} for d in hits], merged,
                    embedder_name=rec0.get("embedder") or (copy_scan or {}).get("embedder"),
                    copy_threshold=rec0.get("copy_threshold") if rec0 else (copy_scan or {}).get("cos_threshold"),
                    calibration=rec0.get("calibration"))
    pairs = EH.sidecar_pairs([{"key": str(d.get("key")), "source": source, "reason": d.get("reason"),
                               "match": d.get("match")} for d in hits], merged, eval_keys=False)
    doc = header("eval_hits", cfg, inputs={"summary": {"path": str(bdir / "summary.json"),
                                                       "sha256": sha256_file(bdir / "summary.json")},
                                           "decisions": {"path": str(dpath), "sha256": want_d},
                                           "fetch": {"path": str(staging_dir(source, inc) / "fetch.json"),
                                                     "sha256": want_f}}, testing=testing)
    doc.update(EH.sidecar(batch, rec, pairs, "intake", source=source, seconds=round(time.time() - t0, 3)))
    write_json_atomic(side, doc)
    return dict(out, status="written", scored=rec["scored"],
                max_pair_cos=max((c for v in rec["per_source"].values() for c in v["pair_cos"]), default=None),
                why=sorted(set(failed.values()))[:5] or None)


def _class_name(cfg, cid):
    if cid == cfg.unmapped_id:
        return "unmapped"
    try:
        return cfg.funnel.class_name(cid)
    except Exception:  # noqa: BLE001
        return str(cid)
