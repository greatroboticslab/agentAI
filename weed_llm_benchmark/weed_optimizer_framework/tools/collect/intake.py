"""Intake: staging -> one intake batch (docs/CONTINUOUS_LOOP.md §3.2, §7.4, §8).

    collect intake --source ID

Under the intake lock, fail closed at every step, nothing written before the
guard loads:
  1. arrival check: fetch.json and every blob it lists must hash as recorded
     (a lab fetch synced to the cluster is verified here, by sha256);
     an intake of the same fetch record already committed is a no-op;
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


def materialise(sdir, fetch_doc, root):
    """The source tree: archives extracted under root/<archive name without
    its extension>/ (nested archives once more), other files linked at their
    names. A tree marked complete is reused."""
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
    done.write_text(utc() + "\n")
    return {"reused": False, "files": files}


# ------------------------------------------------------------------ helpers
def verify_staging(sdir):
    """fetch.json and every blob it lists, checked (StaleInput on the first
    that is missing or changed)."""
    p = sdir / "fetch.json"
    if not p.is_file():
        raise Refusal("not_fetched", "no fetch record at %s: fetch the source first (lever L16)" % p, action="refuse")
    doc = read_json(p, "fetch record")
    if doc.get("format") != FORMATS["fetch"]:
        raise StaleInput("%s is not a %s" % (p, FORMATS["fetch"]))
    for f in doc.get("files") or []:
        b = sdir / "blobs" / f["sha256"]
        if not b.is_file():
            raise StaleInput("staging blob of %s (%s) is missing" % (f["name"], f["sha256"][:12]))
        got = sha256_file(b)
        if got != f["sha256"]:
            raise StaleInput("staging blob of %s changed in transit (%s, recorded %s)"
                             % (f["name"], got[:12], f["sha256"][:12]))
    return doc, sha256_file(p)


def _earlier_manifests(inc):
    out = []
    for b in read_jsonl(batches_ledger(inc), missing_ok=True):
        p = intake_dir(inc) / b["batch"] / "manifest.jsonl"
        if p.is_file():
            out.extend(read_jsonl(p))
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
           lock_path=None, now=None, keep_work=False):
    """Intake one fetched source (module docstring). Returns the result
    record; a refusal raises (Refusal, NamesPending, GuardUnavailable,
    StaleInput) after recording a held or closed event where the state
    changes."""
    from . import names as NM
    from .targets import Targets
    t0 = time.time()
    with intake_lock(inc, what="collect intake %s" % source_id):
        sdir = staging_dir(source_id, inc)
        fetch_doc, fetch_sha = verify_staging(sdir)
        for b in read_jsonl(batches_ledger(inc), missing_ok=True):
            if b.get("source") == source_id and b.get("fetch_sha256") == fetch_sha:
                return {"status": "already_intaken", "source": source_id, "batch": b["batch"],
                        "rows": b.get("rows")}
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
        for r in earlier:
            if r.get("dhash") is not None:
                g.add_intake(int(r["dhash"]), (r.get("batch"), r.get("key")))
        # 4-5. tree, normalise, class map
        work = intake_dir(inc) / "work" / safe_name(source_id)
        root = work / fetch_sha[:12] / "x"
        try:
            materialise(sdir, fetch_doc, root)
        except (zipfile.BadZipFile, tarfile.TarError, EOFError) as e:
            S.append(inc, source_id, "held", reason="archive_unreadable", codes=["archive_unreadable"], risk="R3",
                     stage="intake", detail=str(e)[:500])
            raise Refusal("archive_unreadable", "an archive of %s does not read (%s)" % (source_id, e), action="hold",
                          risk="R3")
        except CollectError as e:                   # a member that would leave the work directory: hostile
            S.append(inc, source_id, "closed", reason="archive_escape", codes=["archive_escape"], stage="intake",
                     detail=str(e)[:500])
            raise Refusal("archive_escape", str(e), action="close")
        ki = cfg.known_item(fetch_doc["known_item"]) if fetch_doc.get("known_item") else None
        opts = dict((ki or {}).get("format_options") or {})
        if (ki or {}).get("format"):
            opts["format"] = ki["format"]
        if not opts.get("class_names") and fetch_doc.get("classes"):
            opts["class_names"] = [c.get("name") for c in fetch_doc["classes"]]
        try:
            tree, res = NZ.read(root, opts, out_images=work / fetch_sha[:12] / "parquet_images")
        except NormaliseError as e:
            S.append(inc, source_id, "held", reason="normalise_failed", codes=["normalise_failed"], risk="R3",
                     stage="intake", detail=str(e)[:500])
            raise Refusal("normalise_failed", str(e), action="hold", risk="R3")
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
                     names=cmap["pending"][:50])
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
        seen_sha = {}
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
        for rel in res.images_without_labels:
            decisions.append({"kind": "image", "source": source_id, "rel": rel, "decision": "rejected",
                              "reason": "no_label"})
            reasons["no_label"] = reasons.get("no_label", 0) + 1
        for rel in res.labels_without_images:
            decisions.append({"kind": "label", "source": source_id, "rel": rel, "decision": "rejected",
                              "reason": "no_image"})
            reasons["no_image"] = reasons.get("no_image", 0) + 1
        if res.rows_without_images:
            decisions.append({"kind": "table_rows", "source": source_id, "decision": "rejected",
                              "reason": "image_not_fetched", "count": res.rows_without_images})
        for it in sorted(res.items, key=lambda x: x["rel"]):
            rel = it["rel"]
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
                decisions.append({"kind": "image", "source": source_id, "rel": rel, "key": key, "decision": "rejected",
                                  "reason": "exact_dup_intake", "twin": seen_sha[sha]})
                reasons["exact_dup_intake"] = reasons.get("exact_dup_intake", 0) + 1
                continue
            reason, match, (dh, _var) = g.check_path(img)
            if not reason and l5 is not None:
                hit = l5_hit(l5, sha, _var)
                if hit is not None:
                    reason, match = L5_REASON, {"key": hit[0], "bits": hit[1], "why": "decision L-5 drops these "
                                                "images from training outright"}
            if reason:
                os.unlink(img)
                decisions.append({"kind": "image", "source": source_id, "rel": rel, "key": key, "decision": "rejected",
                                  "reason": reason, "match": match, "sha256": sha})
                reasons[reason] = reasons.get(reason, 0) + 1
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
            row = {"key": key, "image": str(img), "sha256": sha, "label": str(lab), "label_sha256": sha256_text(text),
                   "source": source_id, "session": it["group"], "capture_group": it["group"],
                   "capture_group_basis": it["group_basis"], "licence": lic.get("id"), "licence_class": lic.get("class"),
                   "research_only": research_only, "licence_override": person_licence, "lab_group": lab_group,
                   "dhash": int(dh),
                   "batch": batch, "hold_until": hold, "holds": holds, "provenance_cleared": cleared, "exhaustive_labels": exhaustive,
                   "boxes": len(boxes), "target_boxes": tb, "unmapped_boxes": counts.get(cfg.unmapped_id, 0),
                   "class_ids": sorted(counts), "width": wh[0], "height": wh[1], "rel": rel, "intake_utc": now_s,
                   "bad_boxes": it["bad"], "unlisted_class_boxes": unlisted, "clipped_boxes": it["clipped"]}
            if "h6_scan" in holds and copy_scan is not None:
                row["copy_scan_calibration"] = {"file": copy_scan["file"]["path"], "sha256": copy_scan["file"]["sha256"],
                                                "cos_threshold": copy_scan["cos_threshold"]}
            manifest.append(row)
            decisions.append({"kind": "image", "source": source_id, "rel": rel, "key": key, "decision": "kept",
                              "reason": "kept", "target_boxes": tb, "boxes": len(boxes)})
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
                     "earlier_intake_images": len(earlier), "copy_scan": copy_scan_rec,
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
        seen_imgs = len(res.items) + len(res.images_without_labels)
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
                     "zero_yield": kept_tb == 0, "zero_yield_reasons": dict(sorted(reasons.items())) if kept_tb == 0
                     else None, "hold_until": hold, "lab_group": lab_group, "research_only": research_only,
                     "licence_override": person_licence,
                     "format": res.format, "copy_scan": copy_scan_rec,
                     "seconds": round(time.time() - t0, 3)})
        write_json_atomic(bdir / "summary.json", summ)
        append_chained(batches_ledger(inc), {"format": FORMATS["batch"], "ts": utc(), "batch": batch,
                                             "source": source_id, "fetch_sha256": fetch_sha, "manifest_sha256": m_sha,
                                             "rows": len(manifest), "target_boxes": kept_tb})
        S.append(inc, source_id, "intaken", provider=fetch_doc["provider"], ref=fetch_doc["ref"], batch=batch,
                 seconds=round(time.time() - t0, 3), **{"yield": yld})
        if not keep_work:
            shutil.rmtree(work / fetch_sha[:12], ignore_errors=True)
    return {"status": "intaken", "source": source_id, "batch": batch, "rows": len(manifest),
            "target_boxes": kept_tb, "rejected": dict(sorted(reasons.items())), "source_leak": leak,
            "zero_yield": kept_tb == 0, "dir": str(bdir)}


def _class_name(cfg, cid):
    if cid == cfg.unmapped_id:
        return "unmapped"
    try:
        return cfg.funnel.class_name(cid)
    except Exception:  # noqa: BLE001
        return str(cid)
