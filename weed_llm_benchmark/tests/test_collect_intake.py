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
  * a person's licence override (licence_overrides): the fetch and the
    intake of an unresolved licence pass; rows, summary.json and the registry
    provenance record research_only (the override's) and the override, the
    licence and its class stay unresolved; a refused licence still closes;
    the owner's decision for the MFWD trays makes their rows research_only;
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
    LOCK that records none, refuses the intake;
  * D28-v2 (amendment 2026-10-03): each image the real GuardV2 refuses as a
    dHash copy of an evaluation image is weighed by its pair cosine (a
    stand-in descriptor for the calibration's embedder) before its copy is
    removed, in its decision and summary.json eval_hits: a flipped dev copy
    at 1.0 (the autopilot's D28: a leak), another picture 2 dHash bits from a
    dev image below 0.80 (D28: chance); a failing or foreign embedder, or no
    bound calibration, leaves the hit unweighed (D28: fail closed); a batch
    without a hit never calls the embedder. A hit is weighed against every
    evaluation image within the never-train radius and keeps the highest (a
    mirrored copy whose guard match is an unrelated neighbour: 1.0). The
    copies are removed even when the per-image loop fails. A batch whose
    summary does not weigh its hits gets intake/<batch>/eval_hits.json from
    rescore_eval_hits (re-derived from the staging blob, its own files
    untouched, no evaluation key or path in it), which D28 reads; a changed
    staging, a file that does not hash to its decision's sha256, or GuardV2
    deciding the re-derived image otherwise (reason or match) leaves the hit
    unweighed, the reason naming no evaluation image; changed decisions
    refuse.

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
    eh = sm.get("eval_hits") or {}
    p4 = [d for d in dec if d.get("reason") == "near_eval_v2"]
    check("D28-v2: the dHash copy is recorded in summary.json eval_hits, unweighed (no v2 calibration is bound to a "
          "stand-in guard), and its decision says why", eh.get("hits") == 1 and eh.get("scored") == 0
          and "no v2 embedding calibration" in (eh.get("why") or "") and (eh.get("per_source") or {}).get(SID, {})
          .get("pair_cos") == [] and len(p4) == 1 and p4[0].get("pair_cos") is None
          and "no v2 embedding calibration" in (p4[0].get("pair_cos_why") or ""), (eh, p4))
    d28_reads(sm, r["batch"])
    ev = [e for e in S.read() if e["source"] == SID]
    check("sources.jsonl: intaken, with the batch and the yield", ev[-1]["event"] == "intaken"
          and ev[-1]["batch"] == r["batch"] and ev[-1]["yield"]["target_boxes"] == 4)
    r2 = I.intake(cfg, SID, guard=W.FakeGuard(eval_paths=[eval_img], base_paths=[base_img]))
    check("a rerun of the same fetch is a no-op", r2["status"] == "already_intaken" and r2["batch"] == r["batch"]
          and len((batches_ledger()).read_text().splitlines()) == 1, r2)
    check("the work tree is removed after the commit", not list((intake_dir() / "work" / SID).glob("*/x")))
    return b


def run_d28(summaries, sidecars=None):
    """The autopilot's own D28 (inc_autopilot.diagnose_stream.d28, read-only)
    over {batch: summary.json} and {batch: eval_hits.json} (the D28-v2
    sidecars the snapshot ships beside them); None when group F's package does
    not load."""
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS
        from weed_optimizer_framework.tools.inc_autopilot import evidence as E
        from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS
        th = LS.load_thresholds()
    except Exception as e:  # noqa: BLE001 - group F's package is optional here
        print("  skip D28 interop: the autopilot's stream diagnoses do not load here (%s)" % e)
        return None

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

    arts = {"intake/%s/summary.json" % b: json.loads(json.dumps(sm)) for b, sm in summaries.items()}
    arts.update({"intake/%s/eval_hits.json" % b: json.loads(json.dumps(sd)) for b, sd in (sidecars or {}).items()})
    return DS.d28(DS.View(Ev(arts), {"sid": "s"}, th))


def d28_reads(sm, batch):
    """The autopilot's own D28 over this summary.json: the leak the collector
    measured (1 near-eval copy of 6 images the guard checked, which a stand-in
    guard without a LOCK leaves unweighed: D28-v2's one-hit rule, fail closed)
    must fire there too, and a summary without the leak fields would stay
    silent (the check can fail). summary.json carries both forms D28 may read:
    source_leak {eval_share, base_share} and the flat images and guard
    counts."""
    d = run_d28({batch: sm})
    if d is None:
        return
    check("the autopilot's D28 fires on this summary.json (the fields it reads are there)",
          d.get("fired") and SID in json.dumps(d.get("detail") or {}), d)
    bare = {k: v for k, v in sm.items() if k not in ("images", "guard", "source_leak", "decisions")}
    d0 = run_d28({batch: bare})
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
    check("... its provenance records research_only (false for a permissive licence) and no licence override",
          e["provenance"]["research_only"] is False and e["provenance"]["licence_override"] is None, e["provenance"])
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


def test_licence_override(cfg):
    from weed_optimizer_framework.tools.collect import Refusal, staging_dir
    from weed_optimizer_framework.tools.collect import intake as I
    from weed_optimizer_framework.tools.collect.config import CollectConfig
    print("a person's licence override: fetched and intaken, research-only, the licence kept as fetched")
    ref = "56565656-0000-0000-0000-000000000011"
    sid = "weedai_" + ref
    cats = [{"id": 1, "name": "weed: amaranthus palmeri"}]
    files = coco_files({"o0.jpg": W.img_bytes(680)}, cats,
                       [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 20, 40, 40]}])
    e = raises(lambda: fetch_simple(cfg, ref, files, licence="unknown"), Refusal)
    check("without an override an unresolved licence holds the fetch (R3)", e is not None
          and e.code == "licence_unresolved" and e.risk == "R3", e)
    ov = {"id": "research-only", "class": "research_only", "research_only": True,
          "decided_by": "human:owner@example.org", "decided_utc": "2026-09-30",
          "reason": "licence unresolved at harvest; accepted for research use only"}
    raw = json.loads(json.dumps(cfg.raw))
    raw["licence_overrides"] = dict(raw["licence_overrides"], **{sid: ov})
    cfg2 = CollectConfig(raw, cfg.path, cfg.sha256, cfg.funnel, cfg.eppo, cfg.eppo_record)
    fetch_simple(cfg2, ref, files, licence="unknown")
    r = I.intake(cfg2, sid, guard=W.FakeGuard())
    b = pathlib.Path(r["dir"])
    row = json.loads((b / "manifest.jsonl").read_text().splitlines()[0])
    sm = json.loads((b / "summary.json").read_text())
    src = json.loads((b / "sources.json").read_text())
    reg = json.loads(pathlib.Path(os.environ["COLLECT_REGISTRY"]).read_text())["datasets"][sid]
    check("with the override the fetch and the intake pass; the rows are research_only (the override's) and carry "
          "it; licence and licence_class stay as fetched (unresolved)",
          r["status"] == "intaken" and row["research_only"] is True and row["licence_override"] == ov
          and row["licence"] == "unresolved" and row["licence_class"] == "unresolved", row)
    check("... summary.json records research_only and the override",
          sm["research_only"] is True and sm["licence_override"] == ov, (sm.get("research_only"),
                                                                           sm.get("licence_override")))
    check("... sources.json's licence record keeps its class and carries the override",
          src["licence"]["class"] == "unresolved" and src["licence"]["override"] == ov, src["licence"])
    check("... the registry entry's provenance records research_only and the override (licence unresolved)",
          reg["license"] == "unresolved" and reg["provenance"]["license_class"] == "unresolved"
          and reg["provenance"]["research_only"] is True and reg["provenance"]["licence_override"] == ov,
          reg["provenance"])
    fj = staging_dir(sid) / "fetch.json"
    doc = json.loads(fj.read_text())
    doc["licence"] = dict(doc["licence"], id="all-rights-reserved", **{"class": "refused"})
    fj.write_text(json.dumps(doc))
    e = raises(lambda: I.intake(cfg2, sid, guard=W.FakeGuard()), Refusal)
    check("an override never rescues a refused licence: the intake closes", e is not None
          and e.code == "licence_refused" and e.action == "close", e)


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
    eh = json.loads((pathlib.Path(r["dir"]) / "summary.json").read_text()).get("eval_hits") or {}
    check("D28-v2: with a testing LOCK that binds no v2 calibration the flipped dev copy is recorded unweighed "
          "(D28 reads it by the one-hit rule)", eh.get("hits") == 1 and eh.get("scored") == 0
          and "no v2 embedding calibration" in (eh.get("why") or ""), eh)
    test_copy_scan_binding(cfg, evals, base, cats, anns)
    test_eval_hit_cosines(cfg, evals, base, cats)
    test_eval_hit_sidecar(cfg, evals, base, cats)
    test_eval_copies_removed_on_failure(cfg, evals, base, cats)
    test_eval_hit_all_matches(cfg, evals, base, cats)


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


class ContentEmbedder(object):
    """A stand-in for the copy scan's DINOv2 descriptor: the same picture, up
    to its 8 flips and rotations, gets the same unit vector (a flipped copy
    scores 1.0, as true flip copies score >= 0.896 with DINOv2), and any other
    picture an unrelated one (a seeded random vector), however close its
    dHash. fail: every call raises (no GPU, no weights)."""
    dim = 64

    def __init__(self, name="facebook/dinov2-base:cls", fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    def __call__(self, pils):
        import numpy as np
        from PIL import Image
        self.calls += 1
        if self.fail:
            raise RuntimeError("the descriptor model cannot be loaded")
        out = []
        for p in pils:
            a = np.asarray(p.convert("L").resize((8, 8), Image.BOX), dtype=np.uint8) // 32
            forms = [np.rot90(a, k) for k in range(4)] + [np.rot90(a.T, k) for k in range(4)]
            seed = int(hashlib.sha256(min(np.ascontiguousarray(f).tobytes() for f in forms)).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dim)
            out.append(v / np.linalg.norm(v))
        return np.stack(out).astype(np.float32)


def test_eval_hit_cosines(cfg, evals, base, cats):
    """D28-v2 (docs/CONTINUOUS_LOOP.md, amendment 2026-10-03): each image the
    real GuardV2 refuses as a dHash copy of an evaluation image is weighed by
    its pair cosine with the evaluation image it matched (the v2
    calibration's embedder, here a stand-in), recorded in its decision and in
    summary.json eval_hits, and only then removed; a hit that cannot be
    weighed is recorded so (D28: fail closed); a batch without a hit loads no
    model. The autopilot's D28 reads the records."""
    from PIL import Image
    from weed_optimizer_framework.tools.collect import intake as I
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    print("D28-v2: the dHash hits' pair cosines (intake)")
    check("the collector's dHash reasons are inc2.eval_hits' (one list)",
          tuple(I.DHASH_EVAL_REASONS) == tuple(EH.DHASH_HIT_REASONS))
    lock = make_lock_v2(evals, [base], [])
    sha = write_v2_calibration(lock.parent)
    lk = json.loads(lock.read_text())
    lk["embed_calibration_v2_sha256"] = sha
    lock.write_text(json.dumps(lk))

    def png(im):
        import io
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    with Image.open(evals["dev"][0]) as im:
        flipped = png(im.transpose(Image.FLIP_LEFT_RIGHT))
    W.grid_img(TMP / "eh" / "chance.png", 910, paint=(2,), fmt="PNG")     # the dev grid, one cell repainted
    chance = (TMP / "eh" / "chance.png").read_bytes()

    def source(ref, imgs):
        anns = [{"id": i, "image_id": i, "category_id": 1, "bbox": [10, 20, 40, 40]} for i in range(len(imgs))]
        return fetch_simple(cfg, ref, coco_files(imgs, cats, anns))

    def batch(r):
        b = pathlib.Path(r["dir"])
        dec = [json.loads(x) for x in (b / "decisions.jsonl").read_text().splitlines()]
        return b, json.loads((b / "summary.json").read_text()), [d for d in dec if d.get("reason") in
                                                                 I.DHASH_EVAL_REASONS]
    emb = ContentEmbedder()
    sid = source("e1e1e1e1-0000-0000-0000-000000000001", {"a_flip.png": flipped, "b_ok.png": W.img_bytes(941)})
    r = I.intake(cfg, sid, lock_path=lock, embedder=emb)
    b, sm, hit = batch(r)
    eh = sm.get("eval_hits") or {}
    row = (eh.get("per_source") or {}).get(sid) or {}
    check("a flipped dev copy (near_eval_variant) is weighed: pair cosine 1.0 with the dev image it matched, in its "
          "decision and in summary.json eval_hits (the calibration's threshold 0.91 and embedder recorded)",
          len(hit) == 1 and hit[0]["reason"] == "near_eval_variant" and hit[0].get("pair_cos") == 1.0
          and eh.get("hits") == 1 and eh.get("scored") == 1 and row.get("pair_cos") == [1.0]
          and eh.get("copy_threshold") == 0.91 and eh.get("embedder") == emb.name
          and (eh.get("calibration") or {}).get("sha256") == sha and eh.get("format") == EH.FORMAT, (hit, eh))
    kept = sorted(p.name for p in (b / "images").iterdir())
    check("... and only then removed: the batch holds the clean image alone",
          len(kept) == 1 and kept[0].endswith("b_ok.png"), kept)
    d = run_d28({r["batch"]: sm})
    if d is not None:
        dv = ((((d.get("detail") or {}).get("leaks") or [{}])[0].get("verdict") or {}).get("dhash") or {})
        check("the autopilot's D28 reads it: a hit at or above the copy threshold is a leak (not by the bare hit)",
              d.get("fired") and dv.get("copy_hits") == 1 and not dv.get("fail_closed"), d.get("summary"))
    sid = source("e1e1e1e1-0000-0000-0000-000000000002", {"a_chance.png": chance, "b_ok.png": W.img_bytes(942)})
    r = I.intake(cfg, sid, lock_path=lock, embedder=emb)
    b, sm, hit = batch(r)
    eh = sm.get("eval_hits") or {}
    check("a dHash near-copy that is another picture (the dev grid with one cell repainted, %s bits) is refused and "
          "weighed: pair cosine %s < 0.80" % ((hit[0].get("match") or {}).get("bits") if hit else None,
                                               hit[0].get("pair_cos") if hit else None),
          len(hit) == 1 and hit[0]["reason"] in I.DHASH_EVAL_REASONS and hit[0].get("pair_cos") is not None
          and hit[0]["pair_cos"] < 0.8 and eh.get("scored") == 1, (hit, eh))
    d = run_d28({r["batch"]: sm})
    if d is not None:
        check("... and the autopilot's D28 judges it chance: the source is not quarantined, and the summary states "
              "its hits, confirmed hits, max pair cos, P and verdict", not d.get("fired")
              and "%s: 1 dHash hit(s) in 2 images, 0 confirmed (pair cos >= 0.8)" % sid in d.get("summary", "")
              and "-> chance" in d.get("summary", ""), d.get("summary"))
    bad = ContentEmbedder(fail=True)
    sid = source("e1e1e1e1-0000-0000-0000-000000000003", {"a_flip.png": flipped, "b_ok.png": W.img_bytes(943)})
    r = I.intake(cfg, sid, lock_path=lock, embedder=bad)
    b, sm, hit = batch(r)
    eh = sm.get("eval_hits") or {}
    check("an embedder that fails never fails the intake: the hit is recorded unweighed with the reason, and its "
          "copy is removed", r["status"] == "intaken" and bad.calls >= 1 and eh.get("hits") == 1
          and eh.get("scored") == 0 and "cannot be described" in (eh.get("why") or "")
          and hit and hit[0].get("pair_cos") is None and len(list((b / "images").iterdir())) == 1, (eh, hit))
    d = run_d28({r["batch"]: sm})
    if d is not None:
        check("... and D28 reads an unweighed hit by the one-hit rule (fail closed)", d.get("fired") and (
            (((d["detail"]["leaks"][0].get("verdict") or {}).get("dhash") or {}).get("fail_closed"))), d.get("summary"))
    other = ContentEmbedder(name="another-model:cls")
    sid = source("e1e1e1e1-0000-0000-0000-000000000004", {"a_flip.png": flipped, "b_ok.png": W.img_bytes(944)})
    r = I.intake(cfg, sid, lock_path=lock, embedder=other)
    _b, sm, _hit = batch(r)
    eh = sm.get("eval_hits") or {}
    check("an embedder other than the calibration's is not used: unweighed, with the reason", eh.get("scored") == 0
          and other.calls == 0 and "is not the calibration's" in (eh.get("why") or ""), eh)
    never = ContentEmbedder(fail=True)
    sid = source("e1e1e1e1-0000-0000-0000-000000000005", {"b_ok.png": W.img_bytes(945)})
    r = I.intake(cfg, sid, lock_path=lock, embedder=never)
    _b, sm, _hit = batch(r)
    eh = sm.get("eval_hits") or {}
    check("a batch without a dHash hit records none and never calls the embedder (no model is loaded)",
          never.calls == 0 and eh.get("hits") == 0 and eh.get("per_source") == {}, eh)
    scored = EH.pair_cosines([{"key": "k", "image": str(TMP / "eh" / "missing.png"), "split": "dev",
                               "eval_key": "dev_0", "eval_image": str(evals["dev"][0])},
                              {"key": "k2", "image": str(evals["dev"][0]), "split": "dev", "eval_key": "nope"}],
                             ContentEmbedder())
    check("inc2.eval_hits.pair_cosines: an image that cannot be read, or an evaluation image neither held nor "
          "given, is unscored with the reason (never raises)", scored["k"]["pair_cos"] is None
          and "cannot be described" in scored["k"]["why"] and scored["k2"]["pair_cos"] is None
          and "neither" in scored["k2"]["why"], scored)


def _bound_lock(evals, base):
    """LOCK v2 over these evaluation images with a bound v2 calibration (copy
    threshold 0.91, embedder facebook/dinov2-base:cls)."""
    lock = make_lock_v2(evals, [base], [])
    sha = write_v2_calibration(lock.parent)
    lk = json.loads(lock.read_text())
    lk["embed_calibration_v2_sha256"] = sha
    lock.write_text(json.dumps(lk))
    return lock


def _png(im):
    import io
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _source(cfg, ref, imgs, cats):
    anns = [{"id": i, "image_id": i, "category_id": 1, "bbox": [10, 20, 40, 40]} for i in range(len(imgs))]
    return fetch_simple(cfg, ref, coco_files(imgs, cats, anns))


def test_eval_hit_sidecar(cfg, evals, base, cats):
    """D28-v2: an intake batch whose summary does not weigh its dHash hits (one
    committed before the amendment; here, one whose embedder failed) is
    weighed again into intake/<batch>/eval_hits.json by
    collect.intake.rescore_eval_hits: each hit image re-derived from the
    staging blob by its decision's rel, checked against its sha256 and
    GuardV2's decision, weighed and removed; the batch's own files are never
    rewritten; the autopilot's D28 judges the source by the sidecar."""
    from PIL import Image
    from weed_optimizer_framework.tools.collect import CollectError, intake_dir
    from weed_optimizer_framework.tools.collect import intake as I
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    from weed_optimizer_framework.tools.inc import common as C
    print("D28-v2: the sidecar of an intake batch committed before the amendment")
    lock = _bound_lock(evals, base)
    with Image.open(evals["dev"][0]) as im:
        flipped = _png(im.transpose(Image.FLIP_LEFT_RIGHT))
    W.grid_img(TMP / "eh" / "chance.png", 910, paint=(2,), fmt="PNG")
    chance = (TMP / "eh" / "chance.png").read_bytes()
    legacy = {}
    for name, img, ref in (("chance", chance, "e2e2e2e2-0000-0000-0000-000000000001"),
                           ("copy", flipped, "e2e2e2e2-0000-0000-0000-000000000002")):
        sid = _source(cfg, ref, {"a_hit.png": img, "b_ok.png": W.img_bytes(951 + len(legacy))}, cats)
        r = I.intake(cfg, sid, lock_path=lock, embedder=ContentEmbedder(fail=True))
        b = pathlib.Path(r["dir"])
        legacy[name] = (sid, r["batch"], b, json.loads((b / "summary.json").read_text()))
    sid, batch, b, sm = legacy["chance"]
    before = {f: C.sha256_file(b / f) for f in ("summary.json", "decisions.jsonl", "manifest.jsonl")}
    check("fixture: the batch's own record weighs none of its hit (the embedder failed), so it needs a sidecar",
          (sm.get("eval_hits") or {}).get("scored") == 0 and I.eval_hits_needed(sm), sm.get("eval_hits"))
    d0 = run_d28({batch: sm})
    if d0 is not None:
        check("  and D28 reads it by the one-hit rule: quarantined (fail closed)", d0.get("fired"), d0.get("summary"))
    bad = ContentEmbedder(fail=True)
    e = raises(lambda: I.rescore_eval_hits(cfg, batch, lock_path=lock, embedder=bad), CollectError)
    check("an embedder that fails weighs nothing: no sidecar is written (it would use up the batch's attempt), the "
          "call refuses so the platform's job fails and the batch stays due", e is not None and bad.calls >= 1
          and "could not be weighed" in str(e) and not (b / I.EVAL_HITS_NAME).exists(), e)
    emb = ContentEmbedder()
    res = I.rescore_eval_hits(cfg, batch, lock_path=lock, embedder=emb)
    side = json.loads((b / I.EVAL_HITS_NAME).read_text())
    rec = side.get("eval_hits") or {}
    row = (rec.get("per_source") or {}).get(sid) or {}
    pair = (side.get("pairs") or [{}])[0]
    check("rescore_eval_hits writes intake/<batch>/eval_hits.json: the hit re-derived from the staging blob and "
          "weighed (pair cos < 0.80, the calibration's threshold and embedder recorded)",
          res["status"] == "written" and side.get("format") == EH.SIDECAR_FORMAT and side.get("batch") == batch
          and rec.get("hits") == 1 and rec.get("scored") == 1 and row.get("pair_cos") and row["pair_cos"][0] < 0.8
          and rec.get("copy_threshold") == 0.91 and rec.get("embedder") == emb.name, (res, rec))
    dev_keys = [json.loads(x)["key"] for x in (lock.parent / "dev.jsonl").read_text().splitlines()]
    check("  the sidecar (which the snapshot ships) holds the hit key, its reason and the pair cosine: no evaluation "
          "path, key or pixels", pair.get("key") and pair.get("reason") in I.DHASH_EVAL_REASONS
          and "match" not in pair and str(evals["dev"][0]) not in json.dumps(side)
          and not any(k in json.dumps(side) for k in dev_keys), pair)
    after = {f: C.sha256_file(b / f) for f in before}
    work = intake_dir() / "work"
    left = sorted(str(p) for p in work.rglob("*") if p.is_file() and "eval_hits_" in str(p)) if work.is_dir() else []
    check("  the batch's own files are unchanged and the re-derived images are gone (its work directory removed)",
          after == before and not left and sorted(x.name for x in (b / "images").iterdir()) == [
              x for x in sorted(y.name for y in (b / "images").iterdir()) if "b_ok" in x], (after == before, left))
    d = run_d28({batch: sm}, {batch: side})
    if d is not None:
        check("the autopilot's D28 judges the source by the sidecar: chance, not quarantined, its numbers stated",
              not d.get("fired") and "%s: 1 dHash hit(s)" % sid in d.get("summary", "")
              and "-> chance" in d.get("summary", ""), d.get("summary"))
    check("a second run finds the sidecar (no rewrite); force writes it again",
          I.rescore_eval_hits(cfg, batch, lock_path=lock, embedder=emb)["status"] == "exists"
          and I.rescore_eval_hits(cfg, batch, lock_path=lock, embedder=emb, force=True)["status"] == "written")
    sid2, batch2, b2, sm2 = legacy["copy"]
    I.rescore_eval_hits(cfg, batch2, lock_path=lock, embedder=emb)
    side2 = json.loads((b2 / I.EVAL_HITS_NAME).read_text())
    d = run_d28({batch2: sm2}, {batch2: side2})
    if d is not None:
        dv = ((((d.get("detail") or {}).get("leaks") or [{}])[0].get("verdict") or {}).get("dhash") or {})
        check("a legacy batch holding a flipped dev copy: its sidecar weighs it at 1.0 and D28 quarantines by the "
              "copy threshold, not by the bare hit", d.get("fired") and dv.get("copy_hits") == 1
              and not dv.get("fail_closed"), d.get("summary"))
    edit_fetch(sid2, title="re-fetched")
    res = I.rescore_eval_hits(cfg, batch2, lock_path=lock, embedder=emb, force=True)
    side3 = json.loads((b2 / I.EVAL_HITS_NAME).read_text())
    check("when the staging now holds another fetch, the hit cannot be re-derived: recorded unweighed with the "
          "reason (D28: fail closed), never guessed", res["status"] == "written" and side3["eval_hits"]["scored"] == 0
          and "another fetch" in json.dumps(side3["pairs"]), side3["pairs"])
    d = run_d28({batch2: sm2}, {batch2: side3})
    if d is not None:
        check("  and D28 fails closed on it", d.get("fired"), d.get("summary"))
    sid4 = _source(cfg, "e2e2e2e2-0000-0000-0000-000000000004", {"a_hit.png": flipped, "b_ok.png": W.img_bytes(955)},
                   cats)
    r4 = I.intake(cfg, sid4, lock_path=lock, embedder=emb)
    check("a batch whose own record weighs its hits needs no sidecar", I.rescore_eval_hits(
        cfg, r4["batch"], lock_path=lock, embedder=emb)["status"] == "not_needed"
          and not (pathlib.Path(r4["dir"]) / I.EVAL_HITS_NAME).exists())
    dp = b / "decisions.jsonl"
    keep = dp.read_bytes()
    dp.write_bytes(keep + b"\n")
    e = raises(lambda: I.rescore_eval_hits(cfg, batch, lock_path=lock, embedder=emb, force=True), CollectError)
    check("decisions that no longer hash as the summary records refuse (nothing is weighed from them)",
          e is not None and "does not hash" in str(e), e)
    dp.write_bytes(keep)
    # the re-derived image must be the decision's own: the file at its rel hashes to the decision's sha256, and
    # GuardV2 refuses it again with the same reason and the same match; otherwise the hit stays unweighed with the
    # reason, and the sidecar (which the snapshot ships) names no evaluation image even then
    sm_keep = (b / "summary.json").read_bytes()
    eval_keys = sorted(json.loads(x)["key"] for f in lock.parent.glob("*.jsonl") if f.stem in ("dev", "test",
                                                                                              "imageweeds")
                       for x in f.read_text().splitlines() if x.strip())

    def tampered(edit):
        rows = [json.loads(x) for x in keep.decode().splitlines() if x.strip()]
        for r in rows:
            if r.get("kind") == "image" and r.get("reason") in I.DHASH_EVAL_REASONS:
                edit(r)
        dp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        s2 = json.loads(sm_keep)
        s2["decisions"]["sha256"] = C.sha256_file(dp)
        (b / "summary.json").write_text(json.dumps(s2))
        try:
            res = I.rescore_eval_hits(cfg, batch, lock_path=lock, embedder=emb, force=True)
            return res, json.loads((b / I.EVAL_HITS_NAME).read_text()), (b / "summary.json").read_text()
        finally:
            dp.write_bytes(keep)
            (b / "summary.json").write_bytes(sm_keep)
    hit_m = next(json.loads(x).get("match") or {} for x in keep.decode().splitlines()
                 if x.strip() and json.loads(x).get("reason") in I.DHASH_EVAL_REASONS)
    other_reason = {"near_eval_v2": "near_eval_variant", "near_eval_variant": "near_eval_v2"}
    other_key = next(k for k in eval_keys if k != hit_m.get("key"))
    for what, edit, says in (
            ("the decision's sha256 is not the file's", lambda r: r.update(sha256="0" * 64), "no longer hashes"),
            ("GuardV2 refuses it for another reason than the decision records",
             lambda r: r.update(reason=other_reason[r["reason"]]), "judges the re-derived image otherwise"),
            ("GuardV2 matches it to another evaluation image than the decision records",
             lambda r: r["match"].update(key=other_key), "another evaluation image than the recorded match")):
        res, side_t, sm_t = tampered(edit)
        whys = [p.get("pair_cos_why") or "" for p in side_t.get("pairs") or []]
        check("a hit whose re-derived image fails a check (%s) is left unweighed with the reason, never weighed"
              % what, res["status"] == "written" and side_t["eval_hits"]["scored"] == 0
              and any(says in w for w in whys), (res, whys))
        check("  and neither the sidecar nor the summary names an evaluation image (no split:key in the reason)",
              not any(k in json.dumps(side_t) or k in sm_t for k in eval_keys)
              and str(evals["dev"][0]) not in json.dumps(side_t), (whys, side_t["eval_hits"].get("why")))
    res = I.rescore_eval_hits(cfg, batch, lock_path=lock, embedder=emb, force=True)
    check("  with the batch's files as committed the hit is weighed again", res["status"] == "written"
          and res["scored"] == 1, res)


def test_eval_copies_removed_on_failure(cfg, evals, base, cats):
    """The copies of evaluation images an intake keeps until they are weighed
    never outlive the per-image loop: a failure anywhere in it (here, writing
    a later image's label) removes them before the error propagates."""
    from PIL import Image
    from weed_optimizer_framework.tools.collect import intake_dir
    from weed_optimizer_framework.tools.collect import intake as I
    print("D28-v2: a failure in the per-image loop leaves no copy of an evaluation image behind")
    lock = _bound_lock(evals, base)
    with Image.open(evals["dev"][0]) as im:
        flipped = _png(im.transpose(Image.FLIP_LEFT_RIGHT))
    sid = _source(cfg, "e3e3e3e3-0000-0000-0000-000000000001", {"a_flip.png": flipped, "b_ok.png": W.img_bytes(961)},
                  cats)
    real = I._label_text

    def boom(boxes):
        raise OSError("disk full (injected)")
    I._label_text = boom
    try:
        e = raises(lambda: I.intake(cfg, sid, lock_path=lock, embedder=ContentEmbedder()), OSError)
    finally:
        I._label_text = real
    open_batches = [p for p in intake_dir().iterdir() if p.is_dir() and (p / "images").is_dir()
                    and not (p / "summary.json").exists()]
    left = sorted(x.name for p in open_batches for x in (p / "images").iterdir())
    check("the injected failure propagates, and the uncommitted batch holds no copy of the evaluation image (the "
          "flipped dev copy was removed in the loop's finally)", e is not None and open_batches
          and not any("a_flip" in n for n in left), (e, left))
    r = I.intake(cfg, sid, lock_path=lock, embedder=ContentEmbedder())
    check("  the next intake redoes the batch whole", r["status"] == "intaken", r)


def test_eval_hit_all_matches(cfg, evals, base, cats):
    """A hit is weighed against every evaluation image within the never-train
    radius, not only the one GuardV2 names: when the guard's first match (by
    the image's own dHash) is an unrelated picture and a true copy sits within
    the radius under a flip, the hit keeps the copy's pair cosine."""
    import random
    from PIL import Image
    from weed_optimizer_framework.tools.collect import intake as I
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    print("D28-v2: the pair cosine is the highest over every evaluation image within the radius")

    def grid(seed, flip=False, paint=()):
        rng = random.Random(seed)
        g = [[rng.randrange(256) for _ in range(9)] for _ in range(8)]
        if flip:
            g = [list(reversed(row)) for row in g]
        for r in paint:
            g[r][8] = 255 if g[r][8] <= g[r][7] else 0
        im = Image.new("L", (9, 8))
        im.putdata([v for row in g for v in row])
        return im.resize((144, 128), Image.NEAREST).convert("RGB")
    e1 = TMP / "am" / "dev_e1.png"
    e2 = TMP / "am" / "test_e2.png"
    e1.parent.mkdir(parents=True, exist_ok=True)
    grid(970).save(e1, format="PNG")                         # the true original (dev)
    grid(970, flip=True, paint=(1, 4)).save(e2, format="PNG")   # another picture, a dHash neighbour of its mirror
    ev = {"dev": list(evals["dev"]) + [e1], "test": list(evals["test"]) + [e2], "imageweeds": list(evals["imageweeds"])}
    lock = _bound_lock(ev, base)
    hit = _png(grid(970, flip=True))                         # a mirrored copy of e1
    sid = _source(cfg, "e4e4e4e4-0000-0000-0000-000000000001", {"a_hit.png": hit, "b_ok.png": W.img_bytes(971)}, cats)
    emb = ContentEmbedder()
    r = I.intake(cfg, sid, lock_path=lock, embedder=emb)
    b = pathlib.Path(r["dir"])
    dec = [json.loads(x) for x in (b / "decisions.jsonl").read_text().splitlines()]
    h = [d for d in dec if d.get("reason") in I.DHASH_EVAL_REASONS]
    m = (h[0].get("match") or {}) if h else {}
    first = EH.pair_cosines([{"key": "k", "image": str(TMP / "am" / "hit.png"), "split": "test", "eval_key": "x",
                              "eval_image": str(e2)}], emb) if (TMP / "am" / "hit.png").write_bytes(hit) else {}
    check("fixture: GuardV2 names the unrelated neighbour (test, by the image's own dHash) as its match, and that "
          "pair alone scores below 0.80", len(h) == 1 and h[0]["reason"] == "near_eval_v2" and m.get("split") == "test"
          and (first.get("k") or {}).get("pair_cos") is not None and first["k"]["pair_cos"] < 0.8, (h, first))
    sm = json.loads((b / "summary.json").read_text())
    row = ((sm.get("eval_hits") or {}).get("per_source") or {}).get(sid) or {}
    check("the hit is weighed against every evaluation image within 6 bits under the 8 variants and keeps the "
          "highest: 1.0, the mirrored dev original (pair_cos_best names it)", row.get("pair_cos") == [1.0]
          and h[0].get("pair_cos") == 1.0 and (h[0].get("pair_cos_best") or [None])[0] == "dev", (row, h))
    d = run_d28({r["batch"]: sm})
    if d is not None:
        check("  so D28 quarantines it as a copy (with the guard's match alone it would have been chance)",
              d.get("fired"), d.get("summary"))


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
    ov = cfg.raw["licence_overrides"]["mediatum_1717366"]
    check("the owner's research-only decision for the MFWD trays applies: every row research_only with the "
          "override recorded, the licence as fetched", all(x["research_only"] is True and x["licence_override"] == ov
                                                           and x["licence"] == "cc-by-4.0" for x in rows), rows[0])


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
        test_licence_override(cfg)
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
