#!/usr/bin/env python3
"""Intake (docs/CONTINUOUS_LOOP.md §3.2, §7.4, §8; group D acceptance "Intake"),
end to end from a fetched source to a committed batch, with a copy guard of
inc2.guard.GuardV2's interface (tests/test_collect_world.FakeGuard).

The source: an annotation-index archive (WeedCOCO) of eight pictures:
  p0..p3   a pinned growth-stage category (the collector's card -> the
           target), a companion species (the reject class) and a numeric
           category (unmapped, 13) in p0
  p4       a re-encoded near copy of an evaluation image -> near_eval_v2
  p5       a copy of a base image -> base_copy
  p6       the same bytes as p1 -> exact_dup_intake
  p7       listed with no annotation -> no_box
plus a picture on disk the JSON never lists -> no_label.

Pinned:
  * fail closed: no guard (no inc2 package or no LOCK v2) refuses before
    anything is written; a staging blob changed in transit refuses (StaleInput);
  * the batch: manifest rows with every field of §3.2 (key, image, sha256,
    label, label_sha256, source, licence, research_only, lab_group,
    capture_group, dhash) plus session, batch, hold_until, holds and the
    counts; labels in the intake class space (the target id, 12 and 13);
    refused images removed from the batch; decisions.jsonl holds the source,
    every class, every fetched file and every image, kept or rejected, with
    its reason; guard.json counts per reason; sources.json the class map and
    its provenance; summary.json (last) the yield, the zero-yield flag, D28's
    source-leak shares; batches.jsonl and sources.jsonl appended with intact
    hash chains; a rerun of the same fetch is a no-op;
  * the registry entry has annotation intake_v1 and status intake, and the
    real verify._skip_reason and the real mega_trainer merge skip it (a
    control source beside it is merged);
  * names the caches lack: pending_names.json, a held event and a
    NamesPending refusal (lever L26 first);
  * the licence is gated again at intake (an unresolved licence holds, R3);
    a non-commercial source's rows are research_only;
  * a record-server source over FTP (the box-table layout: EPPO label ids, a
    tray column) intakes with the tray as the capture group and the target
    code's boxes as the target;
  * a placed box whose class the source does not list keeps its place as
    unmapped (13); a fetch record's provenance clearance is re-derived from
    the config (not trusted); a never-fetched title refuses at intake too;
    summary.json carries images, guard counts and decisions by reason (the
    fields the autopilot's D28 and zero-yield card read);
  * the legacy-label copy rule runs again on the class list the files
    declare (a provider that declares none before download); a key an
    earlier batch used is never reused;
  * an archive member may not leave the work directory ("../" or an absolute
    path refuses); a tar's links are never written;
  * a registry that exists but does not read is never written over (the
    intake refuses and commits nothing);
  * against the real inc2.guard.GuardV2 over a synthetic LOCK v2: a flipped
    dev copy, a rotated base image and a near copy of an earlier batch are
    refused, a clean image kept;
  * decision L-9(c): the rows held for the copy scan are bound to the v2
    embedding calibration LOCK v2 records (manifest, guard.json,
    summary.json); a file that does not hash as recorded, or a production
    LOCK that records none, refuses the intake.

Run:  python3 tests/test_collect_intake.py
"""
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_intake_")
check, raises = W.check, W.raises
PAGS = "5c78d067-8750-4803-9cbe-57df8fae55e4"
SID = "weedai_" + PAGS
STAGE = "weed: amaranthus palmeri (BBCH10-12)"
MANIFEST_FIELDS = ("key", "image", "sha256", "label", "label_sha256", "source", "licence", "research_only",
                   "lab_group", "capture_group", "dhash")


def source_zip():
    eval_img = W.grid_img(TMP / "eval" / "dev_a.jpg", 900)
    base_img = W.grid_img(TMP / "base" / "b.jpg", 901)
    files = {}
    for i in range(4):
        files["images/p%d.jpg" % i] = W.img_bytes(500 + i)
    W.grid_img(TMP / "near.png", 900, paint=(2,), fmt="PNG")
    files["images/p4.png"] = (TMP / "near.png").read_bytes()
    files["images/p5.jpg"] = base_img.read_bytes()
    files["images/p6.jpg"] = files["images/p1.jpg"]
    files["images/p7.jpg"] = W.img_bytes(507)
    files["images/unlisted.jpg"] = W.img_bytes(508)
    cats = [{"id": 1, "name": STAGE}, {"id": 2, "name": "weed: chenopodium album"}, {"id": 3, "name": "weed: 12"}]
    imgs = [{"id": i, "file_name": "p%d.%s" % (i, "png" if i == 4 else "jpg"), "width": 144, "height": 128}
            for i in range(8)]
    anns = []
    for i in range(7):
        anns.append({"id": 10 * i, "image_id": i, "category_id": 1, "bbox": [10, 20, 40, 40]})
        anns.append({"id": 10 * i + 1, "image_id": i, "category_id": 2, "bbox": [80, 60, 30, 30]})
    anns.append({"id": 999, "image_id": 0, "category_id": 3, "bbox": [100, 90, 20, 20]})
    files["weedcoco.json"] = json.dumps(dict(W.coco_doc(imgs, cats, anns), agcontexts=[{"id": 0}])).encode()
    return W.zip_bytes(files), eval_img, base_img


def weedai_net(cfg, ref, z, licence="https://creativecommons.org/licenses/by/4.0/", cats=(STAGE,)):
    base = cfg.provider("weedai")["base_url"]
    info = {"metadata": {"name": "Plants %s" % ref[:4], "license": licence, "description": "boxes"},
            "agcontexts": [{"n_images": 8, "category_statistics": {c: {"image_count": 2, "bounding_box_count": 2}
                                                                   for c in cats}}], "head_version": 1}
    return W.make_net({("GET", base + "/api/upload_info/" + ref): (200, info),
                       ("GET", base + "/code/download/%s.zip" % ref): (200, z, {"Content-Length": str(len(z))})})


def test_fail_closed(cfg):
    from weed_optimizer_framework.tools.collect import GuardUnavailable, StaleInput, intake_dir, staging_dir
    from weed_optimizer_framework.tools.collect import intake as I
    print("fail closed")
    e = raises(lambda: I.intake(cfg, SID), GuardUnavailable)
    check("without LOCK v2 the guard does not load and intake refuses", e is not None, e)
    check("... before anything is written", not (intake_dir() / "batches.jsonl").exists()
          and not list((intake_dir()).glob("i0*")))
    blob = next((staging_dir(SID) / "blobs").glob("*"))
    keep = blob.read_bytes()
    blob.write_bytes(keep + b"x")
    e = raises(lambda: I.intake(cfg, SID, guard=W.FakeGuard()), StaleInput)
    check("a staging blob changed in transit refuses (the arrival check)", e is not None and "changed" in str(e), e)
    blob.write_bytes(keep)


def test_batch(cfg, eval_img, base_img):
    from weed_optimizer_framework.tools.collect import (batches_ledger, intake_dir, sources_ledger, state as S,
                                                        verify_chain)
    from weed_optimizer_framework.tools.collect import intake as I
    print("the batch")
    g = W.FakeGuard(eval_paths=[eval_img], base_paths=[base_img])
    r = I.intake(cfg, SID, guard=g, testing=True)
    b = pathlib.Path(r["dir"])
    rows = [json.loads(x) for x in (b / "manifest.jsonl").read_text().splitlines()]
    dec = [json.loads(x) for x in (b / "decisions.jsonl").read_text().splitlines()]
    check("committed: batch i0001 with 4 rows", r["status"] == "intaken" and r["batch"].startswith("i0001_")
          and r["rows"] == 4, r)
    check("every manifest row has the contract's fields", all(all(k in row for k in MANIFEST_FIELDS) for row in rows),
          [k for k in MANIFEST_FIELDS if k not in rows[0]])
    r0 = [x for x in rows if x["rel"].endswith("p0.jpg")][0]
    lab = (pathlib.Path(r0["label"]).read_text()).split("\n")
    ids = sorted(int(x.split()[0]) for x in lab if x.strip())
    check("labels are in the intake class space: the target (8), the reject class (12), unmapped (13)",
          ids == [8, 12, 13] and r0["target_boxes"] == 1 and r0["unmapped_boxes"] == 1, ids)
    check("hashes recorded: image and label sha256, the dHash", r0["sha256"] == hashlib.sha256(
        pathlib.Path(r0["image"]).read_bytes()).hexdigest() and r0["label_sha256"] == hashlib.sha256(
        pathlib.Path(r0["label"]).read_bytes()).hexdigest() and isinstance(r0["dhash"], int))
    check("provenance-cleared source: no hold; licence, lab group, session = capture group",
          r0["hold_until"] is None and r0["holds"] == [] and r0["licence"] == "cc-by-4.0" and r0["lab_group"] == "TAMU"
          and r0["session"] == r0["capture_group"] and r0["research_only"] is False, r0)
    reasons = {d.get("rel", "").split("/")[-1]: d["reason"] for d in dec if d["kind"] == "image"}
    check("the guard's refusals are recorded per image", reasons.get("p4.png") == "near_eval_v2"
          and reasons.get("p5.jpg") == "base_copy", reasons)
    check("an exact byte duplicate within the batch is rejected", reasons.get("p6.jpg") == "exact_dup_intake")
    check("a listed image with no annotation, and an unlisted one, are rejected with their reasons",
          reasons.get("p7.jpg") == "no_box" and reasons.get("unlisted.jpg") == "no_label", reasons)
    kinds = {d["kind"] for d in dec}
    check("decisions hold the source, every class, every file and every image",
          {"source", "class", "file", "image"} <= kinds and len([d for d in dec if d["kind"] == "class"]) == 3)
    imgs = sorted(p.name for p in (b / "images").iterdir())
    check("refused images are not in the batch", len(imgs) == 4 and not any("p4" in n or "p5" in n or "p6" in n
                                                                            for n in imgs), imgs)
    gd = json.loads((b / "guard.json").read_text())
    check("guard.json counts per reason", gd["counts"].get("near_eval_v2") == 1 and gd["counts"].get("base_copy") == 1
          and gd["format"] == "collect-intake-guard/1", gd["counts"])
    sd = json.loads((b / "sources.json").read_text())
    cm = sd["class_map"]
    check("sources.json holds the class map with provenance and the fetch checksum",
          cm["by_src"]["0"] == 8 if isinstance(next(iter(cm["by_src"])), str) else cm["by_src"][0] == 8,
          cm["by_src"])
    check("... the card, taxonomy cache and EPPO table are recorded", cm["provenance"]["card"]["origin"] ==
          "collect_config" and cm["provenance"]["taxonomy_cache"] and sd["fetch"]["sha256"])
    sm = json.loads((b / "summary.json").read_text())
    check("summary.json: the yield by class, D28's source-leak shares", sm["yield"]["target_boxes"] == 4
          and sm["yield"]["boxes_by_class"].get("PalmerAmaranth") == 4 and sm["source_leak"]["eval_share"] > 0.05
          and sm["source_leak"]["fires"] and sm["zero_yield"] is False, sm["yield"])
    check("the ledgers' hash chains verify", verify_chain(batches_ledger()) == []
          and verify_chain(sources_ledger()) == [])
    d28_reads(sm, r["batch"])
    ev = [e for e in S.read() if e["source"] == SID]
    check("sources.jsonl: intaken, with the batch and the yield", ev[-1]["event"] == "intaken"
          and ev[-1]["batch"] == r["batch"] and ev[-1]["yield"]["target_boxes"] == 4)
    r2 = I.intake(cfg, SID, guard=W.FakeGuard(eval_paths=[eval_img], base_paths=[base_img]))
    check("a rerun of the same fetch is a no-op", r2["status"] == "already_intaken" and r2["batch"] == r["batch"]
          and len((batches_ledger()).read_text().splitlines()) == 1, r2)
    check("the work tree is removed after the commit", not list((intake_dir() / "work" / SID).glob("*/x")))
    return b


def d28_reads(sm, batch):
    """The autopilot's own D28 (inc_autopilot.diagnose_stream.d28, read-only)
    over this summary.json: the leak the collector measured (1 near-eval copy
    of 6 images the guard checked) must fire there too, and a summary without
    the leak fields would stay silent (the check can fail). summary.json
    carries both forms D28 may read: source_leak {eval_share, base_share} and
    the flat images and guard counts."""
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS
        from weed_optimizer_framework.tools.inc_autopilot import evidence as E
        from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS
        th = LS.load_thresholds()
    except Exception as e:  # noqa: BLE001 - group F's package is optional here
        print("  skip D28 interop: the autopilot's stream diagnoses do not load here (%s)" % e)
        return

    class Ev(object):
        def __init__(self, arts):
            self.arts = dict(arts, **{E.CONTEXT: {"sid": "s"}})
            self.artifacts = sorted(k for k in arts)

        def json(self, name):
            return self.arts.get(name)

        def cite(self, name, ptr):
            return {"name": name, "ptr": ptr}

        def get(self, name, ptr, default=None):
            return default

    name = "intake/%s/summary.json" % batch
    d = DS.d28(DS.View(Ev({name: json.loads(json.dumps(sm))}), {"sid": "s"}, th))
    check("the autopilot's D28 fires on this summary.json (the fields it reads are there)",
          d.get("fired") and SID in json.dumps(d.get("detail") or {}), d)
    bare = {k: v for k, v in sm.items() if k not in ("images", "guard", "source_leak", "decisions")}
    d0 = DS.d28(DS.View(Ev({name: bare}), {"sid": "s"}, th))
    check("... and it would stay silent on a summary without the leak fields (so the check above can fail)",
          not d0.get("fired"), d0)


def test_registry(cfg, bdir):
    from weed_optimizer_framework.config import Config
    from weed_optimizer_framework.tools import mega_trainer as M
    from weed_optimizer_framework.tools.inc import verify as V
    print("the registry entry and the old trainers")
    reg = json.loads(pathlib.Path(os.environ["COLLECT_REGISTRY"]).read_text())["datasets"]
    e = reg[SID]
    check("registered with annotation intake_v1 and status intake", e["annotation"] == "intake_v1"
          and e["status"] == "intake" and e["local_path"] == str(bdir) and e["license"] == "cc-by-4.0", e)
    check("the real verify._skip_reason skips it", V._skip_reason(SID, e, {}, M) == "annotation_not_bbox")
    ctrl = TMP / "ctrl"
    for i in range(2):
        W.grid_img(ctrl / "images" / ("c%d.jpg" % i), 700 + i)
        (ctrl / "labels").mkdir(parents=True, exist_ok=True)
        (ctrl / "labels" / ("c%d.txt" % i)).write_text("0 0.5 0.5 0.2 0.2\n")
    registry = {SID: e, "control_src": {"local_path": str(ctrl), "annotation": "yolo", "status": "downloaded",
                                         "class_names": ["Waterhemp"]}}
    hold = TMP / "holdout"
    for i in range(3):
        W.grid_img(hold / ("h%d.jpg" % i), 800 + i)

    class FakeDisc(object):
        def __init__(self):
            self.registry = {"datasets": registry}

    fw = TMP / "fw" / "results" / "framework"
    fw.mkdir(parents=True)
    saved = [(M, "DatasetDiscovery", M.DatasetDiscovery), (M, "update_registry", M.update_registry),
             (M, "_holdout_image_dirs", M._holdout_image_dirs), (M, "HOLDOUT_IMAGES", M.HOLDOUT_IMAGES),
             (Config, "FRAMEWORK_DIR", Config.FRAMEWORK_DIR)]
    M.DatasetDiscovery = FakeDisc
    M.update_registry = lambda path, fn: fn({"datasets": {}})
    M._holdout_image_dirs = lambda: [hold]
    M.HOLDOUT_IMAGES = 3
    Config.FRAMEWORK_DIR = str(fw)
    try:
        _d, _y, stats, used, _n = M._merge_datasets(str(fw / "merged_t"))
    finally:
        for obj, name, val in saved:
            setattr(obj, name, val)
    check("the real mega_trainer merge skips the intake_v1 source and merges the control beside it",
          SID not in used and "control_src" in used and stats["images"] == 2, (used, stats.get("images")))


def test_pending_and_licence(cfg):
    from weed_optimizer_framework.tools.collect import NamesPending, Refusal, intake_dir, staging_dir, state as S
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import intake as I
    print("pending names; the licence at intake")
    ref = "aaaaaaaa-0000-0000-0000-000000000001"
    sid = "weedai_" + ref
    files = {"images/q0.jpg": W.img_bytes(600),
             "weedcoco.json": json.dumps(dict(W.coco_doc(
                 [{"id": 0, "file_name": "q0.jpg", "width": 144, "height": 128}],
                 [{"id": 1, "name": "weed: amaranthus palmeri"}, {"id": 2, "name": "weed: plantago major"}],
                 [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]},
                  {"id": 2, "image_id": 0, "category_id": 2, "bbox": [60, 60, 20, 20]}]), agcontexts=[])).encode()}
    z = W.zip_bytes(files)
    cands = TMP / "lab" / "cands.json"
    cands.parent.mkdir(parents=True, exist_ok=True)
    cands.write_text(json.dumps({"format": "collect-candidates/1", "candidates": [
        {"source_id": sid, "provider": "weedai", "ref": ref, "title": "Plants"}]}))
    F.fetch(cfg, sid, candidates_path=cands, net=weedai_net(cfg, ref, z, cats=("weed: amaranthus palmeri",)))
    e = raises(lambda: I.intake(cfg, sid, guard=W.FakeGuard()), NamesPending)
    pn = intake_dir() / "work" / sid / "pending_names.json"
    check("a name the caches lack refuses the intake (lever L26 first)", e is not None and e.code == "names_pending"
          and "plantago major" in e.names, e)
    check("... pending_names.json lists it and a held event records it", pn.is_file() and "plantago major" in
          json.loads(pn.read_text())["names"] and S.read()[-1]["reason"] == "names_pending")
    check("... and no batch was written", not [p for p in intake_dir().glob("i0*") if sid in p.name])
    fj = staging_dir(SID).parent / sid / "fetch.json"
    doc = json.loads(fj.read_text())
    doc["licence"] = dict(doc["licence"], id="unresolved", **{"class": "unresolved"})
    fj.write_text(json.dumps(doc))
    e = raises(lambda: I.intake(cfg, sid, guard=W.FakeGuard()), Refusal)
    check("the licence is gated again at intake: unresolved holds, R3", e is not None
          and e.code == "licence_unresolved" and e.risk == "R3", e)


def test_research_only(cfg):
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import intake as I
    print("research-only rows")
    ref = "bbbbbbbb-0000-0000-0000-000000000002"
    sid = "weedai_" + ref
    files = {"images/r0.jpg": W.img_bytes(610),
             "weedcoco.json": json.dumps(dict(W.coco_doc(
                 [{"id": 0, "file_name": "r0.jpg", "width": 144, "height": 128}],
                 [{"id": 1, "name": "weed: amaranthus palmeri"}],
                 [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]}]), agcontexts=[])).encode()}
    z = W.zip_bytes(files)
    cands = TMP / "lab" / "cands_nc.json"
    cands.write_text(json.dumps({"format": "collect-candidates/1", "candidates": [
        {"source_id": sid, "provider": "weedai", "ref": ref, "title": "Plants"}]}))
    F.fetch(cfg, sid, candidates_path=cands, net=weedai_net(
        cfg, ref, z, licence="https://creativecommons.org/licenses/by-nc/4.0/", cats=("weed: amaranthus palmeri",)))
    r = I.intake(cfg, sid, guard=W.FakeGuard())
    row = json.loads((pathlib.Path(r["dir"]) / "manifest.jsonl").read_text().splitlines()[0])
    sm = json.loads((pathlib.Path(r["dir"]) / "summary.json").read_text())
    check("a non-commercial source's rows are research_only, and held for the scan (not provenance-cleared)",
          row["research_only"] is True and row["licence_class"] == "research_only" and sm["research_only"] is True
          and row["hold_until"] == "h6_scan", row)


def fetch_simple(cfg, ref, files, cats=("weed: amaranthus palmeri",), licence="https://creativecommons.org/licenses/by/4.0/"):
    """Fetch a one-archive annotation-index source of the given files; returns its source id."""
    from weed_optimizer_framework.tools.collect import fetch as F
    sid = "weedai_" + ref
    cands = TMP / "lab" / ("cands_%s.json" % ref[:8])
    cands.parent.mkdir(parents=True, exist_ok=True)
    cands.write_text(json.dumps({"format": "collect-candidates/1", "candidates": [
        {"source_id": sid, "provider": "weedai", "ref": ref, "title": "Plants"}]}))
    F.fetch(cfg, sid, candidates_path=cands, net=weedai_net(cfg, ref, W.zip_bytes(files), licence=licence, cats=cats))
    return sid


def coco_files(imgs, cats, anns):
    files = {"images/%s" % n: data for n, data in imgs.items()}
    doc = W.coco_doc([{"id": i, "file_name": n, "width": 144, "height": 128} for i, n in enumerate(sorted(imgs))],
                     cats, anns)
    files["weedcoco.json"] = json.dumps(dict(doc, agcontexts=[])).encode()
    return files


def edit_fetch(sid, **kw):
    from weed_optimizer_framework.tools.collect import staging_dir
    fj = staging_dir(sid) / "fetch.json"
    doc = json.loads(fj.read_text())
    doc.update(kw)
    fj.write_text(json.dumps(doc))


def test_unlisted_and_clearance(cfg):
    from weed_optimizer_framework.tools.collect import Refusal, state as S
    from weed_optimizer_framework.tools.collect import intake as I
    print("a box of an unlisted class; clearance and never-fetch at intake")
    cats = [{"id": 1, "name": "weed: amaranthus palmeri"}]
    anns = [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]},
            {"id": 2, "image_id": 0, "category_id": 9, "bbox": [80, 60, 30, 30]}]
    sid = fetch_simple(cfg, "cccccccc-0000-0000-0000-000000000003", coco_files({"u0.jpg": W.img_bytes(630)}, cats,
                                                                              anns))
    # the fetch record claims clearance for a source the config does not clear (a stale or edited record)
    edit_fetch(sid, provenance_cleared=True, hold_until=None, lab_group="TAMU")
    r = I.intake(cfg, sid, guard=W.FakeGuard())
    row = json.loads((pathlib.Path(r["dir"]) / "manifest.jsonl").read_text().splitlines()[0])
    ids = sorted(int(x.split()[0]) for x in pathlib.Path(row["label"]).read_text().splitlines())
    check("a placed box whose category the source does not list keeps its place as unmapped (13), never dropped",
          ids == [8, 13] and row["unlisted_class_boxes"] == 1 and row["unmapped_boxes"] == 1, (ids, row))
    check("a fetch record's clearance is not trusted: the config does not clear this source, rows are held h6_scan",
          row["hold_until"] == "h6_scan" and "h6_scan" in row["holds"] and row["provenance_cleared"] is False, row)
    sm = json.loads((pathlib.Path(r["dir"]) / "summary.json").read_text())
    check("summary.json carries the fields the autopilot reads: images, guard counts, decisions by reason",
          sm["images"] == 1 and set(sm["guard"]) >= {"near_eval_v2", "near_eval_variant", "base_copy"}
          and sm["decisions"]["by_reason"].get("kept") == 1, (sm.get("images"), sm.get("guard"), sm.get("decisions")))
    sid2 = fetch_simple(cfg, "dddddddd-0000-0000-0000-000000000004", coco_files({"t0.jpg": W.img_bytes(640)}, cats,
                                                                               anns[:1]))
    edit_fetch(sid2, title="3SeasonWeedDet10 (all seasons)")
    e = raises(lambda: I.intake(cfg, sid2, guard=W.FakeGuard()), Refusal)
    check("a never-fetched dataset's title refuses at intake too (closed), whatever the fetch did",
          e is not None and e.code == "never_train" and e.action == "close"
          and [r for r in S.read() if r["source"] == sid2][-1]["event"] == "closed", e)


def test_copy_rule_and_keys(cfg):
    from weed_optimizer_framework.tools.collect import Refusal, state as S
    from weed_optimizer_framework.tools.collect import intake as I
    print("the legacy-label copy rule on the files' class list; keys unique across batches")
    legacy = cfg.raw["prefilter"]["legacy_labels"][:4]
    files = {"images/k0.jpg": W.img_bytes(660), "labels/k0.txt": b"0 0.5 0.5 0.2 0.2\n",
             "data.yaml": ("names: [%s]\n" % ", ".join(legacy)).encode()}
    sid = fetch_simple(cfg, "12121212-0000-0000-0000-000000000008", files)
    e = raises(lambda: I.intake(cfg, sid, guard=W.FakeGuard()), Refusal)
    check("a source whose files declare four legacy labels of the reference dataset is closed at intake "
          "(a provider that declares no class list before download is judged here)",
          e is not None and e.code == "copy_candidate" and e.action == "close"
          and [r for r in S.read() if r["source"] == sid][-1]["event"] == "closed"
          and [r for r in S.read() if r["source"] == sid][-1]["reason"] == "copy_candidate", e)
    sid_x = fetch_simple(cfg, "34343434-0000-0000-0000-000000000010", {"images/x0.jpg": W.img_bytes(661),
                                                                       "../../escape.jpg": W.img_bytes(662)})
    e = raises(lambda: I.intake(cfg, sid_x, guard=W.FakeGuard()), Refusal)
    check("a fetched archive whose member would leave the work directory closes the source, recorded",
          e is not None and e.code == "archive_escape" and e.action == "close"
          and [r for r in S.read() if r["source"] == sid_x][-1]["reason"] == "archive_escape", e)
    k = I.Keys()
    first = k.make("src", "a/b.jpg")
    tag = first and I.Keys().make("src", "a/b.jpg")
    k2 = I.Keys(used=[first, first + "__" + __import__("hashlib").sha256(b"src/a/b.jpg").hexdigest()[:8]],
                salt="abcdef0123456789")
    again = k2.make("src", "a/b.jpg")
    check("a key an earlier batch used is never reused (the fetch record's sha tells them apart)",
          again not in (first, tag) and again.endswith("__abcdef01") and not I.Keys(used=[first]).make(
              "src", "a/b.jpg") == first, (first, again))


def test_archive_escape():
    """An archive member may not leave the work directory (a zip or tar
    "../" path or an absolute one refuses the extraction); a tar's links are
    never followed or written."""
    import io
    import tarfile
    from weed_optimizer_framework.tools.collect import CollectError
    from weed_optimizer_framework.tools.collect import intake as I
    print("archive members stay inside the work directory")
    work = TMP / "esc"
    z = work / "a.zip"
    z.parent.mkdir(parents=True, exist_ok=True)
    z.write_bytes(W.zip_bytes({"ok.txt": b"1", "../escaped.txt": b"2"}))
    e = raises(lambda: I.extract(z, "a.zip", work / "x"), CollectError)
    check("a zip member '../escaped.txt' refuses the extraction", e is not None and not (work / "escaped.txt").exists(),
          e)
    t = work / "b.tar"
    with tarfile.open(t, "w") as tf:
        for name, data in (("fine.txt", b"1"), ("/tmp/abs_escape_%s.txt" % TMP.name, b"2")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        ln = tarfile.TarInfo("link_out")
        ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(ln)
    e = raises(lambda: I.extract(t, "b.tar", work / "y"), CollectError)
    check("a tar member with an absolute path refuses the extraction",
          e is not None and not pathlib.Path("/tmp/abs_escape_%s.txt" % TMP.name).exists(), e)
    t2 = work / "c.tar"
    with tarfile.open(t2, "w") as tf:
        ti = tarfile.TarInfo("fine.txt")
        ti.size = 1
        tf.addfile(ti, io.BytesIO(b"1"))
        ln = tarfile.TarInfo("link_out")
        ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(ln)
    n = I.extract(t2, "c.tar", work / "z")
    check("a tar's symbolic link is never written (only regular files)", n == 1
          and not (work / "z" / "link_out").exists(), n)


def test_registry_guard(cfg):
    from weed_optimizer_framework.tools.collect import CollectError
    from weed_optimizer_framework.tools.collect import intake as I
    print("the registry is never written over when it does not read")
    cats = [{"id": 1, "name": "weed: amaranthus palmeri"}]
    anns = [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]}]
    sid = fetch_simple(cfg, "eeeeeeee-0000-0000-0000-000000000005", coco_files({"g0.jpg": W.img_bytes(650)}, cats,
                                                                              anns))
    regp = pathlib.Path(os.environ["COLLECT_REGISTRY"])
    good = regp.read_bytes()
    n_before = len(json.loads(good)["datasets"])
    broken = good[: len(good) // 2]
    regp.write_bytes(broken)
    import weed_optimizer_framework.tools.registry_lock as RL
    sleep = RL.time.sleep
    RL.time.sleep = lambda s: None
    try:
        e = raises(lambda: I.intake(cfg, sid, guard=W.FakeGuard()), CollectError)
    finally:
        RL.time.sleep = sleep
    check("an unreadable registry refuses the intake", e is not None and "registry" in str(e), e)
    check("... and is left as it was (registry_lock would have written an empty one back)",
          regp.read_bytes() == broken)
    from weed_optimizer_framework.tools.collect import batches_ledger
    committed = [json.loads(x)["source"] for x in batches_ledger().read_text().splitlines()]
    check("... and the batch is not committed", sid not in committed, committed)
    # the registry writer's own guards, called directly (intake's earlier strict read refuses first, above)
    fd = {"provider": "weedai", "ref": "x"}
    lic = {"id": "cc-by-4.0", "class": "permissive"}
    RL.time.sleep = lambda s: None
    try:
        e = raises(lambda: I._register(cfg, "src_new", fd, TMP / "b", "i9999_src_new", 1, ["a"], lic,
                                       registry_path=str(regp)), CollectError)
        check("_register refuses a registry that does not read, and leaves it as it was",
              e is not None and regp.read_bytes() == broken, e)
        regp.write_bytes(good)
        real_read = RL.safe_read_json
        RL.safe_read_json = lambda path, **k: None           # the read under the lock comes back empty
        try:
            e = raises(lambda: I._register(cfg, "src_new", fd, TMP / "b", "i9999_src_new", 1, ["a"], lic,
                                           registry_path=str(regp)), CollectError)
        finally:
            RL.safe_read_json = real_read
        check("... and one whose read under the lock comes back with fewer sources is never written back",
              e is not None and "did not read whole" in str(e) and regp.read_bytes() == good, e)
        other = next(k for k, v in json.loads(good)["datasets"].items() if v.get("annotation") != "intake_v1") \
            if any(v.get("annotation") != "intake_v1" for v in json.loads(good)["datasets"].values()) else None
        if other is None:
            d = json.loads(good)
            d["datasets"]["legacy_src"] = {"annotation": "yolo", "status": "downloaded"}
            regp.write_text(json.dumps(d))
            other = "legacy_src"
        before = regp.read_bytes()
        e = raises(lambda: I._register(cfg, other, fd, TMP / "b", "i9999_x", 1, ["a"], lic,
                                       registry_path=str(regp)), CollectError)
        check("... and a source registered outside intake is never registered again",
              e is not None and "never registered twice" in str(e) and regp.read_bytes() == before, e)
    finally:
        RL.time.sleep = sleep
    regp.write_bytes(good)
    r = I.intake(cfg, sid, guard=W.FakeGuard())
    check("once the registry reads again the intake commits and adds its entry",
          r["status"] == "intaken" and len(json.loads(regp.read_text())["datasets"]) == n_before + 1, r)


def make_lock_v2(eval_imgs, base_imgs, l5_imgs=(), testing=True):
    """splits/v2 in inc2.guard's format: dev, test and imageweeds manifests, the
    never-train and base-copy indexes (complete, 6 bits), l5_excluded.jsonl
    (the images decision L-5 drops, with dHash and variants) and LOCK.json."""
    from weed_optimizer_framework.tools.inc import common as C
    d = TMP / "inc" / "splits" / "v2"
    d.mkdir(parents=True, exist_ok=True)
    man, nt = {}, []
    for split in ("dev", "test", "imageweeds"):
        rows = [{"key": "%s_%d" % (split, i), "image": str(p), "label": str(p) + ".txt", "sha256": C.sha256_file(p),
                 "label_sha256": "0" * 64, "source": split, "session": split}
                for i, p in enumerate(eval_imgs.get(split) or [])]
        (d / ("%s.jsonl" % split)).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        man[split] = C.sha256_file(d / ("%s.jsonl" % split))
        nt += [[int(C.dhash(r["image"])), split, r["key"]] for r in rows]
    bc = [[int(C.dhash(p)), "train_core", "b%d" % i] for i, p in enumerate(base_imgs)]
    (d / "nevertrain_dhash.json").write_text(json.dumps({"entries": nt, "complete": True, "bits": 6,
                                                         "min_expected": len(nt)}))
    (d / "base_copies_dhash.json").write_text(json.dumps({"entries": bc, "complete": True, "bits": 6,
                                                          "min_expected": len(bc)}))
    from weed_optimizer_framework.tools.inc2 import guard as G2
    l5 = []
    for i, p in enumerate(l5_imgs):
        h, v = G2.image_hashes(p)
        l5.append({"key": "l5_%d" % i, "source": "rf_x__y", "image": str(p), "sha256": C.sha256_file(p), "dhash": h,
                   "variants": [v[n] for n in G2.VARIANTS], "decided_by": "L-5"})
    (d / "l5_excluded.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in l5))
    lock = {"splits_version": "v2", "eval_splits": ["dev", "test", "imageweeds"], "testing": testing, "manifests": man,
            "nevertrain_sha256": C.sha256_file(d / "nevertrain_dhash.json"),
            "base_copies_sha256": C.sha256_file(d / "base_copies_dhash.json"),
            "l5_excluded_sha256": C.sha256_file(d / "l5_excluded.jsonl"),
            "nevertrain_entries": len(nt), "base_copies_entries": len(bc)}
    (d / "LOCK.json").write_text(json.dumps(lock))
    return d / "LOCK.json"


def test_real_guard(cfg):
    """The collector against group A's real inc2.guard.GuardV2 (not the test
    double): a flipped copy of a dev image, a rotated copy of a base image, a
    near copy of an image of an earlier batch; a clean image is kept."""
    from PIL import Image
    from weed_optimizer_framework.tools.collect import intake as I
    print("the real GuardV2 over a LOCK v2")
    try:
        from weed_optimizer_framework.tools.inc2 import guard as G2
    except ImportError as e:
        print("  skip the real guard: inc2.guard is not importable (%s)" % e)
        return
    dev = W.grid_img(TMP / "rg" / "dev.png", 910, fmt="PNG")
    base = W.grid_img(TMP / "rg" / "base.png", 911, fmt="PNG")
    l5img = W.grid_img(TMP / "rg" / "l5.png", 915, fmt="PNG")
    evals = {"dev": [dev], "test": [W.grid_img(TMP / "rg" / "test.png", 912, fmt="PNG")],
             "imageweeds": [W.grid_img(TMP / "rg" / "iw.png", 913, fmt="PNG")]}
    lock = make_lock_v2(evals, [base], [l5img])
    G2.GuardV2.load(lock)                         # the LOCK is one the real loader accepts

    def png(im):
        import io
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    with Image.open(dev) as im:
        flipped = png(im.transpose(Image.FLIP_LEFT_RIGHT))
    with Image.open(base) as im:
        rotated = png(im.transpose(Image.ROTATE_90))
    W.grid_img(TMP / "rg" / "l5_reenc.jpg", 915, paint=(1,))          # a re-encoded, near copy of the L-5 image
    cats = [{"id": 1, "name": "weed: amaranthus palmeri"}]
    imgs = {"a_flip.png": flipped, "b_rot.png": rotated, "c_clean.png": W.img_bytes(914),
            "d_l5.jpg": (TMP / "rg" / "l5_reenc.jpg").read_bytes()}
    anns = [{"id": i, "image_id": i, "category_id": 1, "bbox": [10, 20, 40, 40]} for i in range(4)]
    sid = fetch_simple(cfg, "ffffffff-0000-0000-0000-000000000006", coco_files(imgs, cats, anns))
    from weed_optimizer_framework.tools.collect import GuardUnavailable
    l5f = lock.parent / "l5_excluded.jsonl"
    keep_l5 = l5f.read_bytes()
    l5f.write_bytes(keep_l5 + b"\n")
    e = raises(lambda: I.intake(cfg, sid, lock_path=lock), GuardUnavailable)
    check("an L-5 list that no longer hashes as LOCK v2 records refuses the intake (fail closed)",
          e is not None and "l5_excluded" in str(e), e)
    l5f.write_bytes(keep_l5)
    ntf = lock.parent / "nevertrain_dhash.json"
    keep_nt = ntf.read_bytes()
    ntf.write_bytes(keep_nt.replace(b'"complete": true', b'"complete": false'))
    e = raises(lambda: I.intake(cfg, sid, lock_path=lock), GuardUnavailable)
    check("a never-train index that no longer hashes as LOCK v2 records: GuardV2 does not load, intake refuses",
          e is not None and "GuardV2" in str(e), e)
    ntf.write_bytes(keep_nt)
    r = I.intake(cfg, sid, lock_path=lock)
    dec = [json.loads(x) for x in (pathlib.Path(r["dir"]) / "decisions.jsonl").read_text().splitlines()]
    why = {d["rel"].split("/")[-1]: d["reason"] for d in dec if d["kind"] == "image"}
    check("real GuardV2: a flipped dev copy is near_eval_variant, a rotated base image base_copy, a clean one kept",
          why.get("a_flip.png") == "near_eval_variant" and why.get("b_rot.png") == "base_copy"
          and why.get("c_clean.png") == "kept", why)
    check("a re-encoded copy of an image decision L-5 dropped is refused (l5_copy), counted as a base copy for D28",
          why.get("d_l5.jpg") == "l5_copy" and json.loads((pathlib.Path(r["dir"]) / "summary.json").read_text())
          ["guard"].get("l5_copy") == 1, why)
    gd = json.loads((pathlib.Path(r["dir"]) / "guard.json").read_text())
    check("... guard.json records the real guard's LOCK and index shas", (gd.get("index") or {}).get(
        "nevertrain_sha256") and (gd.get("index") or {}).get("eval_entries") == 3, gd.get("index"))
    W.grid_img(TMP / "rg" / "near.png", 914, paint=(3,), fmt="PNG")
    sid2 = fetch_simple(cfg, "99999999-0000-0000-0000-000000000007",
                        coco_files({"n0.png": (TMP / "rg" / "near.png").read_bytes()}, cats, anns[:1]))
    r2 = I.intake(cfg, sid2, lock_path=lock)
    dec2 = [json.loads(x) for x in (pathlib.Path(r2["dir"]) / "decisions.jsonl").read_text().splitlines()]
    check("... and a near copy of an earlier batch's image is near_dup_intake (the intake index)",
          [d["reason"] for d in dec2 if d["kind"] == "image"] == ["near_dup_intake"], dec2[-1:])
    prod = make_lock_v2(evals, [base], [], testing=False)
    lk = json.loads(prod.read_text())
    lk.pop("l5_excluded_sha256")
    prod.write_text(json.dumps(lk))
    sid3 = fetch_simple(cfg, "88888888-0000-0000-0000-000000000009",
                        coco_files({"z0.png": W.img_bytes(916)}, cats, anns[:1]))
    e = raises(lambda: I.intake(cfg, sid3, lock_path=prod), GuardUnavailable)
    check("a production LOCK v2 that records no L-5 list refuses the intake (the exclusions cannot be checked)",
          e is not None and "l5_excluded_sha256" in str(e), e)
    test_copy_scan_binding(cfg, evals, base, cats, anns)


def write_v2_calibration(d, threshold=0.91, testing=True):
    """splits v2's embed_calibration_v2.json (group A's format, decision
    L-9(c)): a record that shows it passed (every family at recall 1, the hard
    tier within the gate), on a base the funnel's threshold 0.8256."""
    from weed_optimizer_framework.tools.funnel import leak as L
    from weed_optimizer_framework.tools.inc2 import embed_calibration as EC
    tier = {"n": 12, "false_hits": 0, "fpr": 0.0, "ub": 0.26, "constraining": True}
    cal = {"ok": True, "why": [], "protocol": EC.PROTOCOL, "testing": testing, "cos_threshold": threshold,
           "strict_threshold": 0.97, "dhash_bits_max": 6, "recall_min": L.RECALL_MIN, "fpr_max": L.FPR_MAX,
           "min_negatives": 5 if testing else EC.MIN_NEGATIVES, "floor": 0.8256, "constraining": ["hard"],
           "base": {"source": "funnel_leak_v1", "role": "funnel", "cos_threshold": 0.8256,
                    "file": {"path": "/cluster/inc/funnel/leak_v1.json", "sha256": "b" * 64}},
           "positives": {f: {"n": 10, "hits": 10, "recall": 1.0} for f in L.FAMILIES}, "known_limits": [],
           "negatives": {"hard": tier}}
    doc = {"format": EC.FORMAT, "calibration": cal, "detector": {"descriptor": {"embedder": "facebook/dinov2-base:cls"}}}
    p = pathlib.Path(d) / EC.NAME
    p.write_text(json.dumps(doc, sort_keys=True))
    from weed_optimizer_framework.tools.inc import common as C
    return C.sha256_file(p)


def test_copy_scan_binding(cfg, evals, base, cats, anns):
    """Decision L-9(c): the rows an intake holds for the copy scan are bound to
    the v2 embedding calibration LOCK v2 records (the threshold step1_stream's
    scan judges them by); a production LOCK without one, or a file that does
    not hash as recorded, refuses the intake."""
    from weed_optimizer_framework.tools.collect import GuardUnavailable
    from weed_optimizer_framework.tools.collect import intake as I
    print("the v2 embedding calibration LOCK v2 records (L-9(c))")
    lock = make_lock_v2(evals, [base], [])
    sha = write_v2_calibration(lock.parent)
    lk = json.loads(lock.read_text())
    lk["embed_calibration_v2_sha256"] = sha
    lock.write_text(json.dumps(lk))
    sid = fetch_simple(cfg, "77777777-0000-0000-0000-00000000000a", coco_files({"q0.png": W.img_bytes(917)}, cats,
                                                                                anns[:1]))
    r = I.intake(cfg, sid, lock_path=lock)
    rows = [json.loads(x) for x in (pathlib.Path(r["dir"]) / "manifest.jsonl").read_text().splitlines()]
    gd = json.loads((pathlib.Path(r["dir"]) / "guard.json").read_text())
    sm = json.loads((pathlib.Path(r["dir"]) / "summary.json").read_text())
    check("intake binds every row it holds for the copy scan to the v2 calibration (threshold 0.91, by sha256)",
          rows and all("h6_scan" in x["holds"] and (x.get("copy_scan_calibration") or {}).get("sha256") == sha
                       and x["copy_scan_calibration"]["cos_threshold"] == 0.91 for x in rows), rows[:1])
    check("... and guard.json and summary.json record it (its threshold, protocol and who judges the rows)",
          gd["copy_scan"]["checked"] is True and gd["copy_scan"]["cos_threshold"] == 0.91
          and gd["copy_scan"]["file"]["sha256"] == sha and "step1_stream" in gd["copy_scan"]["judged_by"]
          and sm["copy_scan"]["cos_threshold"] == 0.91, gd.get("copy_scan"))
    p = lock.parent / "embed_calibration_v2.json"
    keep = p.read_bytes()
    p.write_bytes(keep + b" ")
    sid2 = fetch_simple(cfg, "66666666-0000-0000-0000-00000000000b", coco_files({"q1.png": W.img_bytes(918)}, cats,
                                                                                 anns[:1]))
    e = raises(lambda: I.intake(cfg, sid2, lock_path=lock), GuardUnavailable)
    check("a v2 calibration that no longer hashes as LOCK v2 records refuses the intake (fail closed)",
          e is not None and "hashes to" in str(e), e)
    p.write_bytes(keep)
    prod = make_lock_v2(evals, [base], [], testing=False)
    e = raises(lambda: I.intake(cfg, sid2, lock_path=prod), GuardUnavailable)
    check("a production LOCK v2 that records no v2 calibration refuses the intake (its held rows could not be bound)",
          e is not None and "embed_calibration_v2_sha256" in str(e), e)


def test_ftp_box_table(cfg):
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import intake as I
    print("a record-server source over FTP (box table)")
    tray = {}
    rows = ["track_id,label_id,bbox_id,xmin,ymin,xmax,ymax,filename,tray_id"]
    for k in range(2):
        stem = "POROL_131801_2021Y07M2%dD_00H49M09S_img" % k
        tray["%s.jpg" % stem] = W.img_bytes(620 + k)
        rows.append("1,POROL,1,10,20,50,60,POROL/131801/%s,131801" % stem)
        rows.append("2,CHEAL,2,80,60,110,90,POROL/131801/%s,131801" % stem)
    rows.append("3,ACHMI,3,1,1,5,5,ACHMI/133801/ACHMI_img,133801")
    tz = W.zip_bytes(tray)
    gt = ("\n".join(rows) + "\n").encode()
    sums = "%s  ./jpegs/POROL/131801.zip\n%s  ./gt.csv\n" % (hashlib.sha512(tz).hexdigest(),
                                                            hashlib.sha512(gt).hexdigest())
    srv = W.FakeFtp({"checksums.sha512": sums.encode(), "gt.csv": gt, "jpegs/POROL/131801.zip": tz,
                     "jpegs/ACHMI/133801.zip": b"not fetched"})
    base = cfg.provider("mediatum")["base_url"]
    net = W.make_net({("GET", base + "/services/export/node/1717366"): (200, {"nodelist": [[{"attributes": {
        "title": "Trays", "license": "by, http://creativecommons.org/licenses/by/4.0"}}]]})},
        ftp={cfg.provider("mediatum")["ftp"]["host"]: srv})
    r = F.fetch(cfg, "mfwd_porol", net=net)
    check("the known item's FTP selection fetches the index and the target code's trays only", r["files"] == 2, r)
    ri = I.intake(cfg, "mediatum_1717366", guard=W.FakeGuard())
    rows = [json.loads(x) for x in (pathlib.Path(ri["dir"]) / "manifest.jsonl").read_text().splitlines()]
    ids = {int(ln.split()[0]) for row in rows for ln in pathlib.Path(row["label"]).read_text().splitlines()}
    check("both tray images are kept; EPPO codes map (the target and the reject class)", len(rows) == 2
          and ids == {2, 12}, (len(rows), ids))
    check("the tray is the capture group; the source is provenance-cleared (no hold)",
          all(x["capture_group"] == "131801" and x["hold_until"] is None and x["lab_group"] == "TUM" for x in rows),
          rows[0])
    dec = [json.loads(x) for x in (pathlib.Path(ri["dir"]) / "decisions.jsonl").read_text().splitlines()]
    check("box-table rows naming images that were not fetched are one counted decision",
          [d for d in dec if d["kind"] == "table_rows"][0]["count"] == 1)


def main():
    try:
        cfg = W.config()
        W.build_cache()
        from weed_optimizer_framework.tools.collect import fetch as F
        F.disk_free = lambda p: 10 ** 13
        z, eval_img, base_img = source_zip()
        F.fetch(cfg, "pags8", net=weedai_net(cfg, PAGS, z))
        test_fail_closed(cfg)
        bdir = test_batch(cfg, eval_img, base_img)
        test_registry(cfg, bdir)
        test_pending_and_licence(cfg)
        test_research_only(cfg)
        test_ftp_box_table(cfg)
        test_unlisted_and_clearance(cfg)
        test_copy_rule_and_keys(cfg)
        test_archive_escape()
        test_registry_guard(cfg)
        test_real_guard(cfg)
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
