"""The label audit (lever L4) over Step 1's crops and the stream's own: inc/audit.py's audit, unchanged, with each
audited box's crop and BioCLIP-2 embedding also looked up in the stream's committed Step 1 batches
(docs/CONTINUOUS_LOOP.md, Amendment 2026-10-27).

    python -m weed_optimizer_framework.tools.inc2.audit_stream --trusted TRUSTED.jsonl \\
        --audit NAME=MANIFEST.jsonl [NAME=MANIFEST.jsonl ...] --out OUT.json [--nshards N] [--stream-dir DIR]

run_inc_audit.sh, the job script of lever L4, runs this module. Its arguments are inc/audit.py's, plus
--stream-dir. Without it the directory is the step1_stream that the audited increments' stream names
(<stream>/stream.json, read for each audited manifest at <stream>/increments/<name>.jsonl), else INC_DIR/step1_stream
(inc2.stream's default, which every stream so far uses).

Why. inc/audit.py (pinned) finds a box's crop only among the images Step 1's crops.csv was cut from. The stream's
increments are cut from step1_stream queue rows, whose boxes were cropped, embedded and judged at their own intake
batch (inc2.step1_stream: batches/bNNNN/crops.csv, emb.npz, verdicts.npz). So the audit judged 0 of 4,092 boxes of
s001's increments and 8 of 1,867 species boxes of s002's inc0005 (2026-10-05, 2026-10-27): every other image was
image_not_in_crops.

What is inc/audit.py's, unchanged: reading and checking the manifests and their label files (read_rows); the Step 1
inputs and their freshness (verify.Crops, check_fresh, load_embeddings); the Step 1 lookup of every box (CropIndex,
resolve, in_trusted); the probe, fitted on the trusted manifest's Step 1 crops only (fit_trusted), and the held-out
baseline (cv_baseline); the verdicts (judge); every rate, interval, per-mix baseline, noise estimate and
above_baseline test (summarise); the three output files (_write_boxes, _write_md, verify._write_json) and the --out
rule (check_out). A box Step 1 holds is found and judged exactly as inc/audit.py finds and judges it, and the trusted
boxes, the probe and the baseline are the ones it fits (the trusted lookup is Step 1 only, so a record of this module
and one of inc/audit.py share them byte for byte).

What is added: the lookup of an audited image Step 1 does not hold (inc/audit.py's image_not_in_crops) in the stream.
  * The queue row. The stream's queue (queue/queue.jsonl, append-only) holds one row per admitted image; an
    increment's manifest row is its queue row's inc.common.MANIFEST_KEYS (inc2.stream.manifest_fields). A row is
    the queue row of the same key, image path and image sha256, of a committed batch (batches/<batch>/batch.json):
    the one with the same label_sha256, else (the label changed after admission) the latest; among equals the
    latest batch, then the last line. A row with none stays image_not_in_crops.
  * The batch's files, each refused unless it describes the batch's crops.csv (verify.Crops): crops_info.json
    (crops_sha256, and skipped_sha256 of crops_skipped.csv), emb.npz and verdicts.npz (their meta crops_sha256),
    and batch.json's own record of crops.csv, emb.npz and verdicts.npz (inputs.<file>.sha256). A batch that fails
    one is an AuditError: no output, as inc/audit.py refuses stale Step 1 inputs.
  * The admission record. batch.json records no hash of admission.jsonl, so each record is checked against the
    queue row it admitted instead: the same admission (whole / masked), as many masked boxes (n_masked), and per
    class as many kept boxes as the row counts (species_boxes, other_boxes). A record that disagrees leaves the
    row's boxes not judged (admission_mismatch, counted as no_crop_row); the rest of the audit runs.
  * A batch without crops.csv. Its rows were cut where their unmasked original was: a rejoin row
    (inc2.step1_stream.rejoin: key <original>__rjN, supersedes <original>, input rejoin_vN:<the original's input>)
    in the original's intake batch (input bNNNN:...), under the original's key, or in Step 1 for a v1 row (input
    v1); a backfill row (b0000, input v1, whole or masked) in Step 1. Step 1's crops of the unmasked original are
    found as inc/audit.py finds an image (CropIndex: by path, else sha256). The keep list is the row's own batch's
    admission record (a rejoin relabels in place: its boxes are the original's, in the original's order); the
    intake verdict and class reported beside the audit's are that record's (the verdict under the label it
    admitted). A row of a batch without crops.csv and without such an original stays image_not_in_crops.
  * The box. Label box j of a whole admission is the image's box j; of a masked admission it is box keep[j] of the
    batch's admission record (admission.jsonl; inc2.mask.decide keeps the label's boxes in that order). Its crop row
    (key, box) is used only when its (cx, cy, w, h) lie within inc/audit.py's GEOM_TOL of the label's
    (geometry_mismatch otherwise) and its embedding row is finite (failed otherwise); a box the batch left out is
    small / no_size as its crops_skipped.csv says (unmapped: no_crop_row). Every box the stream does not give a crop
    is not judged, with inc/audit.py's own reasons (the stream's own cause is counted beside it).
  * The trusted set. An audited image is refused (in_trusted) by inc/audit.py's routes, and also when its queue
    row's unmasked image (path or sha256) is one of the trusted manifest's images or photographs.

A stream crop is a BioCLIP-2 feature of the same embedder as Step 1's (inc2.step1_stream pins it and checks it on
its canary against v1's features), so the probe judges it as it judges a Step 1 crop. Its rows in OUT_boxes.csv
carry crop_set stream/<batch>, matched_by stream and the crop id local to that batch.

Added to OUT.json, beside inc/audit.py's keys (none of which changes): "lookup" ("step1+step1_stream"; the trusted
block's "lookup" is "step1"); inputs.step1_stream (the root and where it came from, the queue's sha256 over the
lines read, each batch's files and their sha256); per audited manifest images_in_stream, images_without_crops and
a "stream" block (boxes located per batch, the stream's causes for the rest, the intake verdict of each located
box against the audit's, labels that differ from the class at intake, rows found through their original) and
"coverage" (species-labelled boxes, those judged, and why the others were not). OUT.md gets one more section. A
box found through its original has crop_set stream/<its batch>/via/<the original's batch or step1>.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..inc import audit as AU
from ..inc import common as C
from ..inc import verify as V

LOOKUP = "step1+step1_stream"
TRUSTED_LOOKUP = "step1"
STREAM_SET = "stream"
MATCHED_BY_STREAM = "stream"
WHOLE, MASKED = "whole", "masked"          # inc2.mask.WHOLE, MASKED
QUEUE = Path("queue") / "queue.jsonl"      # inc2.step1_stream.Layout.queue, relative to the root
BATCHES = "batches"                        # inc2.step1_stream.Layout.batches
BATCH_FILES = ("crops", "emb", "verdicts")  # batch.json inputs.<name>.sha256, for crops.csv, emb.npz, verdicts.npz
KEY_RE = re.compile(rb'"key":"((?:[^"\\]|\\.)*)"')
STREAM_NOTE = ("lookup step1+step1_stream: an audited box whose image Step 1 does not hold is looked up in the "
               "stream's committed batches (its queue row by key, image and sha256; the batch's crops.csv, "
               "emb.npz and verdicts.npz, checked against its crops.csv); the trusted boxes, the probe and the "
               "baseline are Step 1's only, as inc/audit.py fits them.")

# The stream's own cause of a box without a crop to judge, and the reason inc/audit.py's summaries count it under.
S_NOT_IN_QUEUE = "not_in_queue"                  # no queue row of the key with this image and sha256
S_NOT_COMMITTED = "batch_not_committed"          # its batch has no batch.json
S_NO_CROPS = "batch_without_crops"               # its batch holds no crops.csv (the backfill b0000)
S_NO_ADMISSION = "no_admission_record"           # its batch's admission.jsonl holds no record of the key
S_NOT_ADMITTED = "not_whole_or_masked"           # the record admits the image neither whole nor masked
S_NOT_KEPT = "label_box_not_kept"                # the label holds more boxes than the admission kept
S_NO_CROP_ROW = "no_crop_row"
S_UNMAPPED = "unmapped"                          # crops_skipped.csv: an unmapped class id (never embedded)
S_ADMISSION_MISMATCH = "admission_mismatch"      # the admission record does not describe the queue row
S_SOURCE_MISSING = "original_not_found"          # a row of a batch without crops.csv: its original's crops not found
AUDIT_REASON = {S_NOT_IN_QUEUE: AU.IMAGE_NOT_IN_CROPS, S_NOT_COMMITTED: AU.IMAGE_NOT_IN_CROPS,
                S_NO_CROPS: AU.IMAGE_NOT_IN_CROPS, S_NO_ADMISSION: AU.NO_CROP_ROW, S_NOT_ADMITTED: AU.NO_CROP_ROW,
                S_NOT_KEPT: AU.NO_CROP_ROW, S_NO_CROP_ROW: AU.NO_CROP_ROW, S_UNMAPPED: AU.NO_CROP_ROW,
                S_ADMISSION_MISMATCH: AU.NO_CROP_ROW, S_SOURCE_MISSING: AU.IMAGE_NOT_IN_CROPS,
                AU.IMAGE_CHANGED: AU.IMAGE_CHANGED,
                AU.GEOMETRY_MISMATCH: AU.GEOMETRY_MISMATCH, V.FAILED: V.FAILED, "small": "small",
                "no_size": "no_size"}
# Where a row of a batch without crops.csv was cut (inc2.step1_stream: rejoin and backfill)
VIA_STEP1 = "step1"
KIND_BACKFILL = "backfill"                       # inc2.step1_stream.KIND_BACKFILL
REJOIN_INPUT = re.compile(r"^rejoin_v\d+:(.*)$")  # a rejoin row's input: rejoin_v<N>:<the original's input>
BATCH_INPUT = re.compile(r"^(b\d{4,}):")          # a stream row's input as a rejoin records it: <batch>:<input>


def log(msg):
    print("[inc2.audit_stream] %s" % msg, flush=True)


def default_root():
    """INC_DIR/step1_stream (inc2.step1_stream.Layout().root)."""
    return C.INC_DIR / "step1_stream"


def stream_root(audits):
    """(root, where it came from) of the stream's Step 1 directory for the audited manifests [(name, path)]: the
    step1_stream that stream.json names for every audited manifest that is a stream increment
    (<stream>/increments/<name>.jsonl beside <stream>/stream.json; inc2.stream writes the directory there, and a
    fork inherits it), else default_root(). Increments of streams naming different directories are refused."""
    named = {}
    for _n, p in audits:
        p = Path(p)
        sj = p.parent.parent / "stream.json"
        if p.parent.name != "increments" or not sj.is_file():
            continue
        try:
            d = json.loads(sj.read_text())
        except (OSError, ValueError) as e:
            raise AU.AuditError("%s: unreadable (%s)" % (sj, e))
        s1 = d.get("step1_stream") if isinstance(d, dict) else None
        if s1:
            named.setdefault(str(s1), str(sj))
    if len(named) > 1:
        raise AU.AuditError("the audited increments' streams name different Step 1 directories (%s): pass "
                            "--stream-dir" % ", ".join("%s in %s" % kv for kv in sorted(named.items())))
    if named:
        s1, sj = next(iter(named.items()))
        return Path(s1), "stream.json %s" % sj
    return default_root(), "default"


def _sha_file(path):
    return C.sha256_file(path)


def _npz(path):
    """(meta, {array name: array}) of an npz written by verify._save_npz."""
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        return meta, {k: d[k] for k in d.files if k != "meta"}


def _jsonl_by_key(path, keys, parse_all=False):
    """{key: [(line number, row)]} of the JSON lines of path whose top-level "key" is in keys; a last line without
    its newline (an append in progress) is not read. Returns (found, sha256 of the lines read, lines read, bytes).
    Without parse_all a line is parsed only when the first "key" it holds is one of keys: right for the queue,
    whose rows hold no object with a "key" before their own (canonical JSON, keys sorted); an admission record's
    guard_match can, so admission.jsonl is read with parse_all."""
    found = collections.defaultdict(list)
    h = hashlib.sha256()
    n = size = 0
    with open(path, "rb") as fh:
        for line in fh:
            if not line.endswith(b"\n"):
                break
            h.update(line)
            n += 1
            size += len(line)
            if not parse_all:
                m = KEY_RE.search(line)
                if m is None:
                    continue
                raw = m.group(1)
                key = raw.decode("utf-8") if b"\\" not in raw else json.loads(b'"' + raw + b'"')
                if key not in keys:
                    continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("key") in keys:
                found[row["key"]].append((n, row))
    return found, h.hexdigest(), n, size


# ------------------------------------------------------------- the batches
class StreamBatch:
    """One committed batch's crops of the keys asked for: crops.csv rows, the boxes left out, the admission records,
    the embedding rows and the intake verdicts, after checking that each file describes the batch's crops.csv."""

    def __init__(self, root, bid, keys):
        self.bid = bid
        self.dir = Path(root) / BATCHES / bid
        self.committed = (self.dir / "batch.json").is_file()
        self.has_crops = (self.dir / "crops.csv").is_file()
        self.record = {"batch": bid, "committed": self.committed, "has_crops": self.has_crops}
        self.row_of, self.skipped, self.admission = {}, {}, {}
        self.kind = None
        if not self.committed:
            return
        d = self.dir
        doc = V._read_json(d / "batch.json") or {}
        self.kind = doc.get("kind")
        if (d / "admission.jsonl").is_file():
            got, asha, an, _ab = _jsonl_by_key(d / "admission.jsonl", keys, parse_all=True)
            self.admission = {k: v[-1][1] for k, v in got.items()}
            self.record["admission"] = {"sha256_of_lines_read": asha, "lines": an, "records_of_keys": len(got)}
        if not self.has_crops:
            self.record.update({"kind": self.kind, "input": doc.get("input")})
            return
        try:
            crops = V.Crops(d / "crops.csv")
        except V.VerifyError as e:
            raise AU.AuditError("stream batch %s: %s" % (bid, e))
        self.crops = crops
        info = V._read_json(d / "crops_info.json") if (d / "crops_info.json").is_file() else None
        if not info or info.get("crops_sha256") != crops.sha:
            raise AU.AuditError("stream batch %s: crops_info.json does not describe its crops.csv" % bid)
        if info.get("skipped_sha256") and _sha_file(d / "crops_skipped.csv") != info["skipped_sha256"]:
            raise AU.AuditError("stream batch %s: crops_skipped.csv changed after its crops.csv was cut" % bid)
        shas = {"crops": crops.sha, "emb": _sha_file(d / "emb.npz"), "verdicts": _sha_file(d / "verdicts.npz")}
        inputs = doc.get("inputs") or {}
        for name in BATCH_FILES:
            want = (inputs.get(name) or {}).get("sha256")
            if want and want != shas[name]:
                raise AU.AuditError("stream batch %s: %s does not hash as batch.json recorded it at commit "
                                    "(%s != %s)" % (bid, name, shas[name][:12], str(want)[:12]))
        emeta, earr = _npz(d / "emb.npz")
        vmeta, varr = _npz(d / "verdicts.npz")
        for name, meta in (("emb.npz", emeta), ("verdicts.npz", vmeta)):
            if meta.get("crops_sha256") != crops.sha:
                raise AU.AuditError("stream batch %s: %s does not describe its crops.csv" % (bid, name))
        ids = np.asarray(earr["crop_ids"], dtype=np.int64)
        X = earr["X"]
        pos = np.full(crops.n, -1, dtype=np.int64)
        pos[ids] = np.arange(len(ids))
        if len(ids) != crops.n or (pos < 0).any():
            raise AU.AuditError("stream batch %s: emb.npz does not hold every crop of its crops.csv once" % bid)
        self.X, self.pos = X, pos
        vids = np.asarray(varr["crop_id"], dtype=np.int64)
        if len(vids) != crops.n or sorted(vids.tolist()) != list(range(crops.n)):
            raise AU.AuditError("stream batch %s: verdicts.npz does not hold every crop of its crops.csv once" % bid)
        codes = list(vmeta.get("verdict_codes") or [])
        vorder = np.argsort(vids)
        self.intake = [codes[int(c)] if 0 <= int(c) < len(codes) else "?" for c in np.asarray(varr["verdict"])[vorder]]
        for i, key in enumerate(crops.key):
            if key in keys:
                self.row_of.setdefault((key, int(crops.box[i])), []).append(i)
        with open(d / "crops_skipped.csv", newline="") as fh:
            rd = csv.reader(fh)
            if tuple(next(rd)) != V.SKIPPED_FIELDS:
                raise AU.AuditError("stream batch %s: crops_skipped.csv has unexpected columns" % bid)
            for _st, key, box, reason in rd:
                if key in keys:
                    self.skipped[(key, int(box))] = reason
        self.record.update({"kind": doc.get("kind"), "input": doc.get("input"),
                            "crops": {"sha256": crops.sha, "rows": int(crops.n)},
                            "emb": {"sha256": shas["emb"], "embedder": emeta.get("embedder"), "dim": emeta.get("dim")},
                            "verdicts": {"sha256": shas["verdicts"], "verifier": vmeta.get("verifier")},
                            "batch_json_sha256": _sha_file(d / "batch.json")})

    def finite(self, i):
        return bool(np.isfinite(np.asarray(self.X[self.pos[i]], dtype=np.float32)).all())

    def vector(self, i):
        return self.X[self.pos[i]]


def admission_matches(adm, q):
    """True when a batch's admission record of a key describes the queue row it admitted (inc2.step1_stream writes
    both from one decision): the same admission, as many masked boxes as the row's n_masked, and per class as many
    kept boxes (labels[keep]) as the row's species_boxes and other_boxes, each where the row records it.
    admission.jsonl is not among the files batch.json hashes, so this is how a stale or foreign record is caught."""
    if q.get("admission") is not None and q.get("admission") != adm.get("admission"):
        return False
    if q.get("n_masked") is not None and int(q["n_masked"]) != len(adm.get("mask") or []):
        return False
    sp, ob = q.get("species_boxes"), q.get("other_boxes")
    if sp is None and ob is None:
        return True
    labels, keep = adm.get("labels"), adm.get("keep")
    if not isinstance(labels, list) or not isinstance(keep, list):
        return False
    try:
        kept = [int(labels[int(b)]) for b in keep]
    except (IndexError, TypeError, ValueError):
        return False
    if sp is not None:
        got = [0] * len(sp)
        for lab in kept:
            if 0 <= lab < len(sp):
                got[lab] += 1
        if got != [int(x) for x in sp]:
            return False
    return ob is None or sum(1 for lab in kept if lab == AU.OTHER) == int(ob)


def crop_source(q, batch):
    """Where the crops of a queue row of a batch without crops.csv were cut: VIA_STEP1, a batch id, or None. A
    rejoin row (input rejoin_vN:<the original's input>, supersedes <the original's key>) was cut where its original
    was: Step 1 for a v1 row (input v1), its intake batch for a stream row (input bNNNN:...). A backfill row (b0000,
    input v1) is a v1 pool row, its unmasked original cut by Step 1."""
    inp = str(q.get("input") or "")
    m = REJOIN_INPUT.match(inp)
    if m and q.get("supersedes"):
        rest = m.group(1)
        if rest == "v1":
            return VIA_STEP1
        mb = BATCH_INPUT.match(rest)
        return mb.group(1) if mb else None
    if batch.kind == KIND_BACKFILL and inp == "v1":
        return VIA_STEP1
    return None


def _unmasked(q):
    """The pseudo manifest row of a queue row's unmasked original (CropIndex looks up image, then sha256)."""
    return {"image": q.get("unmasked_image") or q.get("image"), "sha256": q.get("unmasked_sha256") or q.get("sha256")}


class StreamIndex:
    """Where, among the stream's committed batches, each audited row's queue row and crops are. An entry is
    (queue row, its batch, the crops' source, the source's key): the source is the row's own batch, the batch of
    its original (rejoin), or VIA_STEP1 with the Step 1 entry of its unmasked original (rejoin of a v1 row,
    backfill)."""

    def __init__(self, root, rows, root_from="--stream-dir"):
        self.root = Path(root)
        self.queue_path = self.root / QUEUE
        self.present = self.queue_path.is_file() and (self.root / BATCHES).is_dir()
        self.batches, self.match, self.via = {}, {}, {}
        self.s1 = None
        self.queue = {"path": str(self.queue_path), "present": self.present}
        self.root_from = root_from
        keys = {r["key"] for r in rows}
        if not self.present or not keys:
            return
        found, sha, n, size = _jsonl_by_key(self.queue_path, keys)
        self.queue.update({"sha256_of_lines_read": sha, "lines": n, "bytes": size,
                           "rows_of_audited_keys": sum(len(v) for v in found.values())})
        ambiguous = label_changed = 0
        for r in rows:
            cands = [(n, q) for n, q in found.get(r["key"], ())
                     if q.get("image") == r["image"] and q.get("sha256") == r["sha256"]]
            if not cands:
                continue
            same = [(n, q) for n, q in cands if q.get("label_sha256") == r["label_sha256"]]
            pick = same or cands
            label_changed += 0 if same else 1
            pick.sort(key=lambda nq: (str(nq[1].get("batch")), nq[0]))
            ambiguous += 1 if len(pick) > 1 else 0
            self.match[(r["key"], r["image"], r["sha256"])] = pick[-1][1]
        # a rejoin row's crops are its original's, under the original's key
        bkeys = keys | {str(q["supersedes"]) for q in self.match.values() if q.get("supersedes")}
        bids = sorted({str(q.get("batch")) for v in found.values() for _n, q in v if q.get("batch")})
        for bid in bids:
            self.batches[bid] = StreamBatch(self.root, bid, bkeys)
        for mk, q in self.match.items():
            b = self.batches.get(str(q.get("batch")))
            if b is None or not b.committed or b.has_crops:
                continue
            src = crop_source(q, b)
            if src is None:
                continue
            self.via[mk] = src
            if src != VIA_STEP1 and src not in self.batches:
                self.batches[src] = StreamBatch(self.root, src, bkeys)
        self.queue.update({"rows_matched": len(self.match), "matched_among_several": ambiguous,
                           "matched_with_another_label": label_changed,
                           "rows_via_their_original": dict(sorted(collections.Counter(
                               "step1" if v == VIA_STEP1 else "batch" for v in self.via.values()).items()))})

    def attach_step1(self, crops, X):
        """Step 1's crops of the unmasked originals of the rows cut there (inc/audit.py's CropIndex, unchanged)."""
        need = [(_unmasked(self.match[mk]), ()) for mk, v in sorted(self.via.items()) if v == VIA_STEP1]
        if need:
            self.s1 = AU.CropIndex(crops, X, [need])

    def queue_row(self, row):
        return self.match.get((row["key"], row["image"], row["sha256"]))

    def entry(self, row):
        """(queue row, batch, source, source key) of row, or (None, the stream's cause)."""
        q = self.queue_row(row)
        if q is None:
            return None, S_NOT_IN_QUEUE
        b = self.batches.get(str(q.get("batch")))
        if b is None or not b.committed:
            return None, S_NOT_COMMITTED
        if b.has_crops:
            return (q, b, b, q["key"]), None
        src = self.via.get((row["key"], row["image"], row["sha256"]))
        if src is None:
            return None, S_NO_CROPS
        if src == VIA_STEP1:
            if self.s1 is None:
                return None, S_SOURCE_MISSING
            e1, how = self.s1.image(_unmasked(q))
            if e1 is None:
                return None, AU.IMAGE_CHANGED if how == AU.IMAGE_CHANGED else S_SOURCE_MISSING
            return (q, b, VIA_STEP1, e1), None
        sb = self.batches.get(src)
        if sb is None or not sb.committed or not sb.has_crops:
            return None, S_SOURCE_MISSING
        return (q, b, sb, str(q["supersedes"])), None

    def box(self, entry, j, box):
        """(crop id in the source, the stream's cause or None, geometry delta, the original's box index) of label
        box j of the entry's image."""
        q, b, src, ref = entry
        adm = b.admission.get(q["key"])
        if adm is None:
            return -1, S_NO_ADMISSION, None, None
        if adm.get("admission") not in (WHOLE, MASKED):
            return -1, S_NOT_ADMITTED, None, None
        if not admission_matches(adm, q):
            return -1, S_ADMISSION_MISMATCH, None, None
        if adm.get("admission") == MASKED:
            keep = list(adm.get("keep") or [])
            if j >= len(keep):
                return -1, S_NOT_KEPT, None, None
            k = int(keep[j])
        else:
            k = j
        if src == VIA_STEP1:
            i, why, dl = self.s1.box(ref, k, box)       # inc/audit.py's lookup: geometry, finiteness, skipped
            return i, why, dl, k
        ids = src.row_of.get((ref, k), [])
        if ids:
            cr = src.crops
            deltas = [max(abs(cr.cx[i] - box[1]), abs(cr.cy[i] - box[2]), abs(cr.w[i] - box[3]),
                          abs(cr.h[i] - box[4])) for i in ids]
            ok = [i for i, dl in zip(ids, deltas) if dl <= AU.GEOM_TOL]
            if not ok:
                jj = int(np.argmin(deltas))
                return ids[jj], AU.GEOMETRY_MISMATCH, float(deltas[jj]), k
            i = ok[0]
            dl = float(deltas[ids.index(i)])
            return (i, None, dl, k) if src.finite(i) else (i, V.FAILED, dl, k)
        why = src.skipped.get((ref, k))
        if why is not None:
            return -1, (why if why in ("small", "no_size") else S_UNMAPPED if why == S_UNMAPPED else S_NO_CROP_ROW), None, k
        return -1, S_NO_CROP_ROW, None, k

    @staticmethod
    def crop_set(entry):
        """crop_set of a located box: stream/<batch>, or stream/<batch>/via/<the original's batch or step1>."""
        q, b, src, _ref = entry
        if src is b:
            return "%s/%s" % (STREAM_SET, b.bid)
        return "%s/%s/via/%s" % (STREAM_SET, b.bid, VIA_STEP1 if src == VIA_STEP1 else src.bid)

    def intake(self, entry, k, i):
        """(intake verdict, class at intake) of the crop: the batch's verdicts.npz for its own crops; for a row cut
        through its original, its own admission record's (the verdict under the label it admitted)."""
        q, b, src, _ref = entry
        if src is b:
            return b.intake[i], int(b.crops.label[i])
        adm = b.admission.get(q["key"]) or {}
        vs, labs = adm.get("verdicts") or [], adm.get("labels") or []
        return (str(vs[k]) if k < len(vs) else "?"), (int(labs[k]) if k < len(labs) else -1)

    def vector(self, entry, i):
        _q, _b, src, _ref = entry
        return self.s1.X[i] if src == VIA_STEP1 else src.vector(i)

    def record(self):
        return {"root": str(self.root), "root_from": self.root_from, "present": self.present, "queue": self.queue,
                "batches": {bid: b.record for bid, b in sorted(self.batches.items())}}


# ------------------------------------------------------------ the lookup
class JoinedX:
    """Step 1's embedding rows [0, n1) followed by the located stream rows: what inc/audit.py's judge indexes."""

    def __init__(self, X1, rows):
        self.X1, self.n1 = X1, len(X1)
        dim = X1.shape[1] if len(X1.shape) == 2 else (len(rows[0]) if rows else 0)
        self.X2 = np.asarray(rows, dtype=np.float16).reshape(len(rows), dim) if rows else np.zeros((0, dim), np.float16)

    def __len__(self):
        return self.n1 + len(self.X2)

    def __getitem__(self, ids):
        if isinstance(ids, (int, np.integer)):
            i = int(ids)
            return self.X1[i] if i < self.n1 else self.X2[i - self.n1]
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), self.X1.shape[1]), dtype=np.float16)
        m = ids < self.n1
        if m.any():
            out[m] = self.X1[ids[m]]
        if (~m).any():
            out[~m] = self.X2[ids[~m] - self.n1]
        return out


def in_trusted_stream(q, trusted):
    """True when the queue row's unmasked image is one of the trusted manifest's images (path or sha256) or
    photographs (trusted = inc/audit.py's trusted_ids)."""
    if trusted is None or q is None:
        return False
    img, sha = q.get("unmasked_image") or q.get("image"), q.get("unmasked_sha256") or q.get("sha256")
    return img in trusted[0] or img in trusted[2] or sha in trusted[1]


def resolve(index, sindex, rows, trusted, vectors):
    """inc/audit.py's resolve over Step 1 (unchanged), then the stream lookup for every box whose image Step 1
    does not hold. Stream records get crop_id = the row of `vectors` (appended: the JoinedX row is n1 + that),
    crop_set stream/<batch>, matched_by stream, and the extra keys _stream (the stream's cause or 'located'),
    _batch, _local, _intake and _intake_label."""
    recs = AU.resolve(index, rows, trusted)
    it = iter(recs)
    for r, boxes in rows:
        mine = [next(it) for _b in boxes]
        e1, how1 = index.image(r)
        if e1 is not None or how1 != AU.IMAGE_NOT_IN_CROPS:
            continue
        entry, cause = sindex.entry(r)
        q = sindex.queue_row(r)
        refuse = mine and mine[0]["status"] == AU.IN_TRUSTED or in_trusted_stream(q, trusted)
        for k, (rec, b) in enumerate(zip(mine, boxes)):
            if refuse:
                rec.update(status=AU.IN_TRUSTED, reason=None)
                rec["matched_by"] = MATCHED_BY_STREAM if entry is not None else rec["matched_by"]
                rec["_stream"] = "in_trusted"
                continue
            if entry is None:
                rec.update(status=AU.NO_EMB, reason=AUDIT_REASON[cause])
                rec["_stream"] = cause
                continue
            batch = entry[1]
            rec["matched_by"] = MATCHED_BY_STREAM
            i, why, d, ko = sindex.box(entry, k, b)
            rec["geom_delta"] = d
            rec["_batch"] = batch.bid
            if entry[2] is not batch:
                rec["_via"] = VIA_STEP1 if entry[2] == VIA_STEP1 else entry[2].bid
            if i >= 0:
                rec["crop_set"] = sindex.crop_set(entry)
                rec["_local"] = int(i)
                rec["_intake"], rec["_intake_label"] = sindex.intake(entry, ko, i)
            if why is not None:
                rec.update(status=AU.NO_EMB, reason=AUDIT_REASON.get(why, why))
                rec["_stream"] = why
                if i >= 0:
                    rec["crop_id"] = int(i)
                continue
            rec.update(status=None, reason=None)
            rec["_stream"] = "located"
            rec["crop_id"] = len(vectors)
            vectors.append(sindex.vector(entry, i))
    return recs


def stream_block(recs):
    """The per-manifest stream lookup and coverage summary (beside inc/audit.py's summarise)."""
    st = [r for r in recs if "_stream" in r]
    loc = [r for r in st if r["_stream"] == "located"]
    per_batch = collections.Counter(r["_batch"] for r in loc)
    causes = collections.Counter(r["_stream"] for r in st if r["_stream"] != "located")
    intake = collections.Counter(r["_intake"] for r in loc)
    pairs = collections.Counter("%s -> %s" % (r["_intake"], r["status"]) for r in loc)
    differs = sum(1 for r in loc if r["_intake_label"] != r["label"])
    via = collections.Counter("%s via %s" % (r["_batch"], r["_via"]) for r in loc if r.get("_via"))
    return {"images": len({r["key"] for r in st}), "boxes_looked_up": len(st), "boxes_located": len(loc),
            "per_batch": dict(sorted(per_batch.items())), "not_located": dict(sorted(causes.items())),
            "intake_verdicts": dict(sorted(intake.items())),
            "intake_vs_audit": dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))),
            "label_differs_from_intake": differs, "located_via_original": dict(sorted(via.items()))}


def coverage(recs):
    """Species-labelled boxes, those the audit gave a species verdict (the share D31 reads), and why the others
    were not (the no_embedding reason, else the status: in_trusted, label_unseen)."""
    sp = [r for r in recs if r["label"] < AU.OTHER]
    judged = sum(1 for r in sp if r["status"] in AU.SPECIES_VERDICTS)
    ot = [r for r in recs if r["label"] == AU.OTHER]
    why = collections.Counter((r["reason"] if r["status"] == AU.NO_EMB else r["status"])
                              for r in sp if r["status"] not in AU.SPECIES_VERDICTS)
    return {"boxes": len(recs), "species_boxes": len(sp), "species_judged": judged,
            "species_judged_share": AU._rate(judged, len(sp)), "otherplant_boxes": len(ot),
            "otherplant_judged": sum(1 for r in ot if r["status"] in AU.JUDGED),
            "species_not_judged": dict(sorted(why.items()))}


# ------------------------------------------------------------------ the run
def check_out(out, inputs, root):
    """inc/audit.py's --out rule, and no output under the stream's directory either."""
    for p in AU._paths(out):
        rp, rr = Path(p).resolve(), Path(root).resolve()
        if rp == rr or rr in rp.parents:
            raise AU.AuditError("--out: %s lies under the stream's Step 1 directory %s; the audit never writes "
                                "there" % (p, rr))
    AU.check_out(out, inputs)


def run_audit(trusted_path, audits, out, nshards=None, stream_dir=None):
    """inc/audit.py's run_audit with the stream lookup for audited boxes; writes OUT.json, OUT.md and
    OUT_boxes.csv. Returns the result."""
    t0 = time.time()
    root, root_from = (Path(stream_dir), "--stream-dir") if stream_dir else stream_root(audits)
    trusted_path = Path(trusted_path)
    out_json, out_md, out_csv = AU._paths(out)
    manifests = [trusted_path] + [Path(p) for _n, p in audits]
    check_out(out, manifests, root)
    t_rows = AU.read_rows(trusted_path, "trusted")
    a_rows = [(name, Path(p), AU.read_rows(p, name)) for name, p in audits]
    inputs = (manifests + [r["label"] for rows in [t_rows] + [x for _n, _p, x in a_rows] for r, _b in rows]
              + [C.manifest_path("train_core"), V.COPIES, V.POOL, V.POOL_META, root / QUEUE])
    check_out(out, inputs, root)
    log("trusted %s: %d images; audited %s; stream %s"
        % (trusted_path, len(t_rows), {n: len(r) for n, _p, r in a_rows}, root))
    try:
        crops = V.Crops()
        fresh = V.check_fresh(crops)
        X, emb = V.load_embeddings(crops, nshards)
    except V.VerifyError as e:
        raise AU.AuditError("Step 1 inputs: %s" % e)
    log("crops.csv: %d crops; embeddings %s" % (crops.n, emb))
    index = AU.CropIndex(crops, X, [t_rows] + [r for _n, _p, r in a_rows])

    # the probe and its baseline: inc/audit.py's, on the trusted manifest's Step 1 crops only
    t_recs = AU.resolve(index, t_rows)
    use = [r for r in t_recs if r["status"] is None and r["label"] < AU.OTHER]
    if not use:
        raise AU.AuditError("no trusted species box has a Step 1 embedding")
    Xc = V._norm(X[np.array([r["crop_id"] for r in use], dtype=np.int64)])
    yc = np.array([r["label"] for r in use], dtype=np.int64)
    gc = AU._groups(use)
    log("probe: %d trusted species boxes from %d groups; no OtherPlant class, by design (verify's "
        "OtherPlant sample could hold the audited images)" % (len(use), len(np.unique(gc))))
    ver = AU.fit_trusted(Xc, yc, gc)
    baseline = AU.cv_baseline(Xc, yc, gc)
    classes = set(int(c) for c in ver.probe.classes_)
    t_reasons = collections.Counter(r["reason"] for r in t_recs if r["status"] == AU.NO_EMB)
    t_sessions = {r.get("session") for r, _b in t_rows if r.get("session")}
    group_kind = collections.Counter(g.split(":", 1)[0] for g in gc)
    trusted = {
        "manifest": str(trusted_path), "manifest_sha256": C.sha256_file(trusted_path),
        "images": len(t_rows), "boxes": len(t_recs),
        "species_boxes_used": len(use),
        "used_per_species": {C.CLASS_NAMES[k]: int((yc == k).sum()) for k in range(AU.N_SPECIES)},
        "otherplant_boxes_left_out": sum(1 for r in t_recs if r["label"] == AU.OTHER),
        "no_embedding": sum(t_reasons.values()),
        "no_embedding_reasons": {k: int(t_reasons.get(k, 0)) for k in AU.REASONS if t_reasons.get(k)},
        "images_matched_by_sha256": len({r["key"] for r in t_recs if r["matched_by"] == "sha256"}),
        "groups": int(len(np.unique(gc))),
        "grouped_by": {"session": int(group_kind.get("session", 0)),
                       "source": int(group_kind.get("source", 0))},
        "lookup": TRUSTED_LOOKUP,
    }
    pc = ver.info.get("per_class") or {}
    probe = {k: ver.info.get(k) for k in ("folds", "recall_target", "max_cond_pass", "threshold_rule",
                                          "n_species_crops", "cv_top1_species",
                                          "cv_recall_species_min", "cv_recall_species_mean")}
    probe.update({
        "classes": [C.CLASS_NAMES[k] for k in sorted(classes)],
        "species_not_in_trusted": [C.CLASS_NAMES[k] for k in range(AU.N_SPECIES) if k not in classes],
        "species_never_verified": [C.CLASS_NAMES[k] for k in range(AU.N_SPECIES)
                                   if k in classes and not np.isfinite(ver.tau_p[k])],
        "thresholds": {C.CLASS_NAMES[k]: {"tau_p": V._f(ver.tau_p[k]), "sigma": V._f(ver.sigma[k])}
                       for k in range(C.NC)},
        "per_class": {n: pc[n] for n in C.CLASS_NAMES if n in pc},
        "otherplant_class": False,
        "note": "verify.fit_verifier on the trusted species boxes only (no OtherPlant sample): the "
                "probe never predicts OtherPlant"})
    if probe["species_not_in_trusted"]:
        log("WARNING: the trusted boxes hold no %s: a box labelled so is label_unseen"
            % probe["species_not_in_trusted"])

    # the stream: the audited rows whose image Step 1 does not hold
    need = [r for _n, _p, rows in a_rows for r, _b in rows
            if index.image(r) == (None, AU.IMAGE_NOT_IN_CROPS)]
    sindex = StreamIndex(root, need, root_from)
    sindex.attach_step1(crops, X)
    log("stream %s (%s): %d audited image(s) not in Step 1; queue %s; batches %s"
        % (root, root_from, len({r["key"] for r in need}), {k: v for k, v in sindex.queue.items() if k != "path"},
           sorted(sindex.batches)))
    check_out(out, inputs + [b.dir / f for b in sindex.batches.values()
                             for f in ("crops.csv", "emb.npz", "verdicts.npz", "admission.jsonl")], root)
    for b in sindex.batches.values():
        e = b.record.get("emb")
        if e and (e.get("embedder") != emb.get("embedder") or int(e.get("dim") or -1) != int(emb.get("dim") or -2)):
            raise AU.AuditError("stream batch %s: its crops were embedded by %s (dim %s), Step 1's by %s (dim %s); "
                                "the probe judges Step 1's features only"
                                % (b.bid, e.get("embedder"), e.get("dim"), emb.get("embedder"), emb.get("dim")))

    results, box_rows, vectors, per = {}, [], [], []
    t_ids = AU.trusted_ids(index, t_rows)
    for name, path, rows in a_rows:
        recs = resolve(index, sindex, rows, t_ids, vectors)
        per.append((name, path, rows, recs))
    jx = JoinedX(X, vectors)
    for name, path, rows, recs in per:
        for r in recs:
            if r.get("_stream") == "located":
                r["crop_id"] = jx.n1 + r["crop_id"]
        AU.judge(ver, SimpleNamespace(X=jx), recs)
        for r in recs:
            if r.get("_stream") == "located":
                r["crop_id"] = r["_local"]
        blk = AU.summarise(recs, baseline)
        refused = {r["key"] for r in recs if r["status"] == AU.IN_TRUSTED}
        shared = sorted({r.get("session") for r, _b in rows if r.get("session") in t_sessions})
        how = collections.Counter()
        for r, _b in rows:
            e, h = index.image(r)
            how[h if e is not None else "none"] += 1
        sblk = stream_block(recs)
        blk.update({
            "manifest": str(path), "manifest_sha256": C.sha256_file(path), "images": len(rows),
            "sources": dict(sorted(collections.Counter(r["source"] for r, _b in rows).items())),
            "boxes_per_class": {C.CLASS_NAMES[k]: sum(1 for r in recs if r["label"] == k)
                                for k in range(C.NC)},
            "images_refused_in_trusted": len(refused),
            "images_matched_by_sha256": int(how.get("sha256", 0)),
            "images_without_step1_image": int(how.get("none", 0)),
            "sessions_shared_with_trusted": shared,
            "rows_in_shared_sessions": sum(1 for r, _b in rows if r.get("session") in t_sessions),
            "images_in_stream": len({r["key"] for r in recs if r.get("_batch")}),
            "images_without_crops": len({r["key"] for r in recs if r.get("_stream") in
                                         (S_NOT_IN_QUEUE, S_NOT_COMMITTED, S_NO_CROPS, S_SOURCE_MISSING,
                                          AU.IMAGE_CHANGED)}),
            "stream": sblk, "coverage": coverage(recs)})
        results[name] = blk
        sp, cv = blk["species"], blk["coverage"]
        mx = sp["baseline_for_mix"]
        log("%s: %d images, %d boxes; species boxes judged %d of %d, conflict rate %s %s, verified %s; "
            "not judged %s" % (name, len(rows), len(recs), sp["boxes_judged"], cv["species_boxes"],
                               sp["conflict_rate"], sp["conflict_rate_ci95"], sp["verified_rate"],
                               {s: n for s, n in blk["boxes_by_status"].items() if s not in AU.JUDGED and n}
                               or "none"))
        log("%s: stream: %d box(es) located in %s, not located %s"
            % (name, sblk["boxes_located"], sblk["per_batch"] or "-", sblk["not_located"] or "-"))
        log("%s: baseline for its species mix f %s %s, r %s; noise estimate %s, above baseline %s "
            "(pooled baseline: %s, %s)" % (name, mx["false_conflict_rate"], mx["false_conflict_rate_ci95"],
                                           mx["conflict_recall_on_wrong"], sp["noise_estimate_corrected"],
                                           sp["above_baseline"], sp["noise_estimate_pooled_baseline"],
                                           sp["above_pooled_baseline"]))
        if mx["species_without_baseline"]:
            log("WARNING: %s: species without a baseline entry, left out of the noise estimate: %s"
                % (name, mx["species_without_baseline"]))
        if refused:
            log("WARNING: %s: %d image(s) are in the trusted manifest; refused" % (name, len(refused)))
        if shared:
            log("WARNING: %s: %d row(s) share a capture session with the trusted rows"
                % (name, blk["rows_in_shared_sessions"]))
        if blk["no_embedding_reasons"].get(AU.GEOMETRY_MISMATCH):
            log("WARNING: %s: %d box(es) do not match their crop row's geometry; not judged"
                % (name, blk["no_embedding_reasons"][AU.GEOMETRY_MISMATCH]))
        for r in recs:
            box_rows.append((name, r))

    import sklearn
    result = AU._clean({
        "built_utc": AU._utc(), "seconds": round(time.time() - t0, 1),
        "what": AU.WHAT,
        "lookup": LOOKUP,
        "trusted": trusted, "probe": probe, "baseline_cv": baseline, "audits": results,
        "audit_order": list(results),
        "inputs": {"crops": {"path": str(crops.path), "sha256": crops.sha, "rows": crops.n},
                   "crops_made_from": fresh.get("inputs"), "embeddings": emb, "geom_tol": AU.GEOM_TOL,
                   "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__},
                   "step1_stream": sindex.record()},
        "outputs": {"json": str(out_json), "md": str(out_md), "boxes_csv": str(out_csv)},
        "notes": [
            "conflict_rate: conflicts / species-labelled boxes with a verdict (verified, conflict, "
            "unknown); the label-noise estimate. Boxes not judged (no_embedding, in_trusted, "
            "label_unseen) are counted, never guessed.",
            "baseline_cv: held-out trusted boxes judged by a verifier fitted, thresholds included, on "
            "the other folds (grouped by capture session, else source): the false-conflict rate f on "
            "true labels, and the conflict recall r under one uniform wrong species per box.",
            "baseline_for_mix: f and r of each species (baseline_cv.per_species) weighted by the "
            "audited set's judged species-labelled boxes per label; f's 95 % interval from seeded "
            "draws of each species' f from its Jeffreys posterior. Species without a baseline entry "
            "are listed and left out, with their boxes.",
            "noise_estimate_corrected = (c - f) / (r - f) with f and r of baseline_for_mix, clipped "
            "to [0, 1]; it assumes in-domain labels and uniform swaps, so it is indicative only for "
            "another domain.",
            "above_baseline: the conflict rate's Wilson 95 % interval lies wholly above "
            "baseline_for_mix's interval of f.",
            "noise_estimate_pooled_baseline, above_pooled_baseline: the same against the baseline "
            "pooled over the trusted set's own species mix; context only, since per-species f and r "
            "differ and an audited set's mix need not be the trusted set's.",
            "known_truth: boxes judged on a train_core crop, whose true label is the crop's "
            "train_core label.",
            STREAM_NOTE]})
    AU._write_boxes(out_csv, box_rows)
    AU._write_md(out_md, result)
    _append_md(out_md, result)
    V._write_json(out_json, result)
    log("wrote %s, %s, %s (%.0fs)" % (out_json, out_md, out_csv, time.time() - t0))
    return result


def _append_md(path, res):
    """OUT.md's stream section, after inc/audit.py's tables."""
    L = ["", "## Stream lookup (lookup %s)" % res["lookup"], "",
         "Boxes whose image Step 1 does not hold, looked up in the stream's committed batches (`%s`). The trusted "
         "boxes, the probe and the baseline are Step 1's only." % res["inputs"]["step1_stream"]["root"], "",
         "| Set | Species boxes judged / labelled | Looked up in the stream | Located (per batch) | Not located | "
         "Intake verdict -> audit verdict | Label differs from intake |", "|---|---|---|---|---|---|---|"]
    for name in res["audit_order"]:
        a = res["audits"][name]
        s, cv = a["stream"], a["coverage"]
        L.append("| %s | %d / %d | %d | %d (%s) | %s | %s | %d |" % (
            name, cv["species_judged"], cv["species_boxes"], s["boxes_looked_up"], s["boxes_located"],
            ", ".join("%s %d" % kv for kv in s["per_batch"].items()) or "-",
            ", ".join("%s %d" % kv for kv in s["not_located"].items()) or "-",
            ", ".join("%s %d" % kv for kv in list(s["intake_vs_audit"].items())[:6]) or "-",
            s["label_differs_from_intake"]))
    via = {name: res["audits"][name]["stream"].get("located_via_original") for name in res["audit_order"]}
    if any(via.values()):
        L += ["", "Located through their original (a rejoin row in its original's batch, a backfill or v1 rejoin row "
              "in Step 1): %s." % "; ".join("%s: %s" % (n, ", ".join("%s %d" % kv for kv in v.items()))
                                           for n, v in via.items() if v)]
    L.append("")
    path = Path(path)
    text = path.read_text() + "\n".join(L)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --------------------------------------------------------------------- CLI
def build_parser():
    ap = AU.build_parser()
    ap.prog = "python -m weed_optimizer_framework.tools.inc2.audit_stream"
    ap.description = ("INC attribution item 4 (lever L4): the BioCLIP-2 label audit of manifests against a probe "
                      "fitted on a trusted one, each box looked up in Step 1's crops and the stream's")
    ap.add_argument("--stream-dir", default=None,
                    help="the stream's Step 1 directory (default: INC_DIR/step1_stream)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        run_audit(a.trusted, AU.parse_audits(a.audit), a.out, nshards=a.nshards, stream_dir=a.stream_dir)
    except AU.AuditError as e:
        log("FAILED: %s" % e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
