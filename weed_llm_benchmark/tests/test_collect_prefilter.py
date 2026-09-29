#!/usr/bin/env python3
"""The pre-download filter and the fetch pre-checks (docs/CONTINUOUS_LOOP.md
§3.1 "Refuses when", §6.3 L16, §7.2; group D acceptance "Prefilter").

Pinned:
  * the reject list applies to whole tokens of class names: a candidate with a
    carpetweed class is not rejected by "car", a sugar-beet class is not
    rejected by "bee", and a class list of objects only is rejected;
  * species-only names pass (binomials through the offline resolver; the
    alias table's join);
  * at least four class names equal to the legacy labels make a copy
    candidate (recorded, never downloaded); three do not;
  * a re-upload platform's candidate that declares a species with no public
    source outside the evaluation lab, or three or more targets, is a
    presumed derivative (lab group, h6_scan hold); two targets are not;
  * a provider without class lists is kept on its text only with box
    evidence; image-level candidates are held for a person (R3);
  * provenance clearance only for a cleared known item's own record: a
    re-upload whose title matches it, or a candidate record that claims
    clearance, is held for the copy scan; a title match of an evaluation-lab
    item takes that lab;
  * never fetched: a never-train slug of the trainer (read by parsing its
    source, equal to the trainer's own set; a source without the assignment
    refuses every fetch), a title of a never-train dataset, a (provider, ref)
    the config lists;
  * ranking by expected target boxes per GB (declared box counts first), a
    bonus per deficit class; a non-commercial copy is superseded by a
    permissive copy of the same dataset (P6);
  * the pre-check matrix: credentials (R3), licence unresolved (R3) and
    refused (close), an evaluation lab without the copy scan (R3) and with
    it, the per-source (R3), daily and envelope caps, attempts (close), the
    registry (quarantined, a source registered outside intake: close),
    another campaign's source (hold), and inside Slurm a provider not placed
    on the cluster (refuse).

Run:  python3 tests/test_collect_prefilter.py
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_prefilter_")
check, raises = W.check, W.raises


def cand(provider, ref, classes=None, **kw):
    c = {"provider": provider, "kind": provider, "ref": ref, "source_id": "%s_%s" % (provider, ref.replace("/", "__")),
         "title": kw.pop("title", ref), "description": kw.pop("description", ""), "keywords": [],
         "licence_text": kw.pop("licence_text", "CC BY 4.0"), "licence_evidence": None,
         "classes": None if classes is None else [{"id": i, "name": n, "hints": [], "boxes": kw.get("boxes", {}).get(n)}
                                                   for i, n in enumerate(classes)],
         "annotation": kw.pop("annotation", "boxes"), "annotation_evidence": [], "bytes": kw.pop("bytes", 1e9),
         "images": kw.pop("images", 500), "found_by": []}
    kw.pop("boxes", None)
    c.update(kw)
    return c


def setup():
    from weed_optimizer_framework.tools.collect import names as NM
    from weed_optimizer_framework.tools.collect.targets import Targets
    cfg = W.config()
    W.build_cache(W.CACHE_NAMES + ["Sugar beet", "Soybean"])
    nm = NM.load(cfg)
    return cfg, nm, Targets(cfg, nm)


def test_decide(cfg, nm, tg):
    from weed_optimizer_framework.tools import cwd12_species as S
    from weed_optimizer_framework.tools.collect import classmap as CM
    from weed_optimizer_framework.tools.collect import prefilter as PF
    print("decide")
    d = lambda c, **k: PF.decide(c, cfg, nm, tg, **k)  # noqa: E731
    r = d(cand("roboflow", "ws/a", ["Carpetweed", "car"]))
    check("a carpetweed class is not rejected by 'car' (whole tokens)",
          r["decision"]["status"] == "kept" and r["target_classes"] == ["Carpetweed"]
          and [x["reject_token"] for x in r["class_status"]] == [None, "car"], r["class_status"])
    check("'car' is a whole-token reject, 'Carpetweed' holds none", CM.reject_token("car", cfg) == "car"
          and CM.reject_token("Carpetweed", cfg) is None and CM.reject_token("Carpet weed", cfg) is None)
    check("'bee' never rejects a sugar-beet class", CM.reject_token("Sugar beet", cfg) is None
          and CM.reject_token("Sugarbeet", cfg) is None and CM.reject_token("bee", cfg) == "bee")
    r = d(cand("roboflow", "ws/b", ["Sugar beet", "Palmer Amaranth"]))
    check("a sugar-beet plus target class list is kept", r["decision"]["status"] == "kept"
          and r["target_classes"] == ["PalmerAmaranth"], r["decision"])
    r = d(cand("roboflow", "ws/c", ["car", "person", "drone"]))
    check("a class list of objects only is rejected on its reject words",
          r["decision"]["status"] == "rejected" and r["decision"]["reasons"][0]["code"] == "reject_token", r["decision"])
    r = d(cand("roboflow", "ws/d", ["Amaranthus palmeri", "Portulaca oleracea", "Chenopodium album"]))
    check("species-only names pass", r["decision"]["status"] == "kept"
          and r["target_classes"] == ["PalmerAmaranth", "Purslane"], r["target_classes"])
    legacy = list(S.CWD12_LEGACY_LABELS)
    check("the config's legacy labels are the reference dataset's legacy labels",
          cfg.raw["prefilter"]["legacy_labels"] == legacy)
    r = d(cand("roboflow", "ws/e", legacy[:4] + ["soil"]))
    check("four legacy labels make a copy candidate (recorded, not downloaded)",
          r["decision"]["status"] == "copy_candidate" and r["copy_candidate"]["legacy_equal"] == 4, r["decision"])
    r = d(cand("roboflow", "ws/f", legacy[:3] + ["soil"]))
    check("three legacy labels do not", r["decision"]["status"] != "copy_candidate", r["decision"])
    r = d(cand("roboflow", "ws/g", ["PricklySida", "soybean"]))
    check("a Roboflow candidate declaring an evaluation-lab-only species is a presumed derivative",
          r["lab_group"] == "LuLab" and r["lab_group_basis"] == "presumed_derivative" and r["hold_until"] == "h6_scan"
          and r["evaluation_lab"] and r["decision"]["status"] == "kept", (r["lab_group"], r["decision"]))
    r = d(cand("kaggle", "o/h", None, title="Weeds", description="Palmer amaranth, sicklepod and purslane with "
                                                                  "bounding boxes"))
    check("three targets named by a re-upload's text make a presumed derivative",
          r["lab_group_basis"] == "presumed_derivative", (r["lab_group"], r["text_targets"]))
    r = d(cand("kaggle", "o/i", None, title="Weeds", description="Palmer amaranth and sicklepod, bounding boxes"))
    check("two do not", r["lab_group"] is None and r["decision"]["status"] == "kept", (r["lab_group"], r["decision"]))
    r = d(cand("zenodo", "123", None, title="Field photos", description="Photographs of Palmer amaranth plants",
               annotation="unknown"))
    check("a text-only candidate without box evidence is rejected (no_box_evidence)",
          r["decision"]["reasons"][0]["code"] == "no_box_evidence", r["decision"])
    r = d(cand("zenodo", "124", None, title="Field photos", description="Palmer amaranth, YOLO labels", annotation="unknown"))
    check("... and kept with a box word", r["decision"]["status"] == "kept"
          and r["decision"]["reasons"][0]["code"] == "text_target" and "text:yolo" in r["annotation_evidence"],
          (r["decision"], r["annotation_evidence"]))
    r = d(cand("huggingface", "o/j", ["Palmer Amaranth"], annotation="image_level"))
    check("an image-level candidate is held for a person", r["decision"]["status"] == "held"
          and "image_level_only" in [x["code"] for x in r["decision"]["reasons"]], r["decision"])
    r = d(cand("weedai", "u1", ["weed: amaranthus palmeri"], title="CottonWeedDet12 mirror"))
    check("the reference dataset's title is never fetched", r["decision"]["reasons"][0]["code"] == "never_train")
    r = d(cand("weedai", "u2", ["weed: amaranthus tuberculatus"], title="ImageWeeds aerial 2021"))
    check("the exam's title is never fetched", r["decision"]["reasons"][0]["code"] == "never_train")
    r = d(cand("zenodo", "14861516", None, title="x", description="Palmer amaranth bounding boxes"))
    check("a (provider, ref) the config lists is never fetched", r["decision"]["reasons"][0]["code"] == "never_train"
          and "base v2" in r["decision"]["reasons"][0]["detail"], r["decision"])
    nt = PF.never_train_slugs()
    from weed_optimizer_framework.tools import mega_trainer as M
    check("the never-train slugs parsed from the trainer's source equal the trainer's set", nt == frozenset(M.NEVER_TRAIN_SLUGS),
          nt ^ frozenset(M.NEVER_TRAIN_SLUGS))
    c = cand("huggingface", "x/y", ["Palmer Amaranth"])
    c["source_id"] = sorted(nt)[0]
    r = PF.decide(c, cfg, nm, tg, never_train=nt)
    check("a never-train slug is rejected", r["decision"]["reasons"][0]["code"] == "never_train")
    bad = TMP / "no_assignment.py"
    bad.write_text("X = 1\n")
    from weed_optimizer_framework.tools.collect import ConfigError
    check("a trainer source without the assignment refuses every fetch (fail closed)",
          raises(lambda: PF.never_train_slugs(bad), ConfigError) is not None)


def test_clearance(cfg, nm, tg):
    """Provenance clearance (§3.2 [review]: only a primary release from a lab
    outside the evaluation labs) comes from the config for the known item's own
    record, never from a title match or from the candidate record."""
    from weed_optimizer_framework.tools.collect import prefilter as PF
    print("provenance clearance")
    pags = cfg.known_item("pags8")
    mfwd = cfg.known_item("mfwd_porol")
    d = lambda c: PF.decide(c, cfg, nm, tg)  # noqa: E731
    prim = d(dict(cand("weedai", pags["ref"], ["weed: amaranthus palmeri (BBCH10-12)"], title=pags["title"]),
                  source_id=pags["id"]))
    check("the known item's own record is provenance-cleared (no hold), lab declared",
          prim["known_item_role"] == "primary" and prim["provenance_cleared"] and prim["hold_until"] is None
          and prim["lab_group"] == "TAMU", (prim["known_item_role"], prim["hold_until"], prim["lab_group"]))
    for t in ("Palmer Amaranth Growth Stage (re-upload)", "MFWD weeds", "Moving Fields Weed Dataset, subset"):
        r = d(cand("roboflow", "someone/copy", ["Palmer Amaranth"], title=t))
        check("a re-upload whose title matches a cleared known item is not cleared: held h6_scan (%s)" % t,
              r["known_item_role"] == "match" and not r["provenance_cleared"] and r["hold_until"] == "h6_scan"
              and r["lab_group"] not in ("TAMU", "TUM"), (r["known_item_role"], r["hold_until"], r["lab_group"]))
    r = d(cand("weedai", "0000aaaa-0000-0000-0000-000000000000", ["weed: amaranthus palmeri"], title="plain",
               provenance_cleared=True, hold_until=None, lab_group="TAMU"))
    check("a candidate record claiming provenance_cleared (a stale candidates file) is not cleared",
          not r["provenance_cleared"] and r["hold_until"] == "h6_scan" and r["lab_group"] != "TAMU", r["hold_until"])
    r = d(cand("roboflow", "someone/no-classes", None, title="MFWD mirror", description="images", annotation="unknown"))
    check("... and a title match alone is not the known item: not kept as one",
          r["decision"]["status"] != "kept" or r["decision"]["reasons"][0]["code"] != "known_item", r["decision"])
    r = d(cand("roboflow", "someone/cwd3", ["Carpetweed"], title="CottonWeedDet3 copy"))
    check("a title match of an evaluation-lab item takes that lab (the cautious reading)",
          r["lab_group"] == "LuLab" and r["evaluation_lab"] and r["hold_until"] == "h6_scan", r["lab_group"])
    check("cleared_by_config: the MFWD record yes, another ref of that provider no, an evaluation lab never",
          PF.cleared_by_config(cfg, mfwd["id"], "mediatum", mfwd["ref"])
          and not PF.cleared_by_config(cfg, "mediatum_999", "mediatum", "999")
          and not PF.cleared_by_config(cfg, "kg_yuzhenlu__cottonweeddet3", "kaggle", "yuzhenlu/cottonweeddet3"))
    for t in ("3SeasonWeedDet10 (all years)", "project_agml three_season_weed_detection"):
        r = d(cand("huggingface", "x/tsw", ["Palmer Amaranth"], title=t))
        check("3SeasonWeedDet10 is never fetched under any title (%s)" % t,
              r["decision"]["status"] == "rejected" and r["decision"]["reasons"][0]["code"] == "never_train",
              r["decision"])


def test_rank(cfg, nm, tg):
    from weed_optimizer_framework.tools.collect import prefilter as PF
    print("rank and supersede")
    a = PF.decide(cand("roboflow", "ws/big", ["Palmer Amaranth", "soybean"], boxes={"Palmer Amaranth": 1000},
                       bytes=1e9), cfg, nm, tg)
    b = PF.decide(cand("roboflow", "ws/small", ["Palmer Amaranth", "soybean"], boxes={"Palmer Amaranth": 50},
                       bytes=1e9), cfg, nm, tg)
    c = PF.decide(cand("roboflow", "ws/def", ["Sicklepod", "soybean"], boxes={"Sicklepod": 50}, bytes=1e9),
                  cfg, nm, tg, deficit=["Sicklepod"])
    check("declared box counts are the estimate", a["estimate"]["basis"] == "declared_box_counts"
          and a["estimate"]["target_boxes"] == 1000.0, a["estimate"])
    check("a deficit class doubles the score (bonus 1.0 per deficit class)",
          c["estimate"]["score"] == 2 * b["estimate"]["score"], (b["estimate"], c["estimate"]))
    ranked = PF.rank([b, c, a])
    check("ranked by score", [x["source_id"] for x in ranked if x["rank"]] == [a["source_id"], c["source_id"],
                                                                             b["source_id"]])
    nc = PF.decide(cand("kaggle", "o/copy", None, title="Same Set", description="Palmer amaranth bounding boxes",
                        licence_text="CC BY-NC-SA 4.0"), cfg, nm, tg)
    pm = PF.decide(cand("zenodo", "999", None, title="Same Set", description="Palmer amaranth bounding boxes",
                        licence_text="cc-by-4.0"), cfg, nm, tg)
    PF.supersede([nc, pm], cfg)
    check("a non-commercial copy is superseded by the permissive copy (P6)",
          nc.get("superseded_by") == pm["source_id"] and not pm.get("superseded_by"), (nc.get("superseded_by"),))
    chk = PF.precheck(nc, cfg, ctx())
    check("... and its pre-check closes it", "superseded" in [f["code"] for f in chk["failures"]]
          and chk["action"] == "close", chk)
    a = PF.decide(cand("roboflow", "u1/weed-detection", ["Palmer Amaranth"], title="Weed Detection",
                       licence_text="CC BY-NC 4.0", images=812), cfg, nm, tg)
    b = PF.decide(cand("roboflow", "u2/weed-detection", ["Palmer Amaranth"], title="weed detection",
                       licence_text="CC BY 4.0", images=95), cfg, nm, tg)
    PF.supersede([a, b], cfg)
    check("two projects that only share a title (different image counts) are not copies: neither is closed",
          not a.get("superseded_by") and not b.get("superseded_by"), (a.get("superseded_by"), b.get("superseded_by")))
    pags = cfg.known_item("pags8")
    prim = PF.decide(dict(cand("weedai", pags["ref"], ["weed: amaranthus palmeri (BBCH10-12)"], title=pags["title"],
                               licence_text="CC BY-NC 4.0", images=614), source_id=pags["id"]), cfg, nm, tg)
    mt = PF.decide(cand("roboflow", "someone/pags8-reupload", ["Palmer Amaranth"], title=pags["title"],
                        licence_text="CC BY 4.0", images=614), cfg, nm, tg)
    PF.supersede([prim, mt], cfg)
    check("a record that only matches a known item's title never supersedes the item's own record",
          not prim.get("superseded_by"), prim.get("superseded_by"))
    rj = PF.decide(cand("roboflow", "u3/copy", ["car", "person"], title="Same Set", licence_text="CC BY 4.0",
                        images=500), cfg, nm, tg)
    nc2 = PF.decide(cand("kaggle", "o/copy2", None, title="Same Set", description="Palmer amaranth bounding boxes",
                         licence_text="CC BY-NC-SA 4.0", images=500), cfg, nm, tg)
    PF.supersede([rj, nc2], cfg)
    check("a copy the prefilter rejected never closes one that can be fetched",
          rj["decision"]["status"] == "rejected" and not nc2.get("superseded_by"), nc2.get("superseded_by"))


def ctx(**kw):
    base = {"state": {}, "registry": {}, "never_train": frozenset(), "creds": {}, "copy_scan": (False, None),
            "bytes_today": 0, "bytes_total": 0, "placement": None}
    base.update(kw)
    return base


def test_precheck(cfg, nm, tg):
    from weed_optimizer_framework.tools.collect import prefilter as PF
    print("pre-check matrix")
    ok = PF.decide(cand("roboflow", "ws/ok", ["Palmer Amaranth"]), cfg, nm, tg)
    r = PF.precheck(ok, cfg, ctx())
    check("a clean candidate passes (ok, no risk)", r["ok"] and r["risk"] is None and r["failures"] == [], r)
    codes = lambda r: [f["code"] for f in r["failures"]]  # noqa: E731
    r = PF.precheck(ok, cfg, ctx(creds={"roboflow": (False, "no key")}))
    check("missing credentials: hold, R3 (card X16)", codes(r) == ["credentials_missing"] and r["risk"] == "R3"
          and r["action"] == "hold", r)
    un = PF.decide(cand("roboflow", "ws/un", ["Palmer Amaranth"], licence_text=None), cfg, nm, tg)
    r = PF.precheck(un, cfg, ctx())
    check("an unresolved licence: hold, R3 (P6)", "licence_unresolved" in codes(r) and r["risk"] == "R3", r)
    rf = PF.decide(cand("roboflow", "ws/rf", ["Palmer Amaranth"], licence_text="All rights reserved"), cfg, nm, tg)
    check("a refused licence rejects the candidate and closes it", rf["decision"]["status"] == "rejected"
          and PF.precheck(rf, cfg, ctx())["action"] == "close")
    nc = PF.decide(cand("roboflow", "ws/nc", ["Palmer Amaranth"], licence_text="CC BY-NC 4.0"), cfg, nm, tg)
    check("a non-commercial licence is research-only and fetchable", nc["licence"]["research_only"]
          and PF.precheck(nc, cfg, ctx())["ok"])
    lu = PF.decide(dict(cand("kaggle", "yuzhenlu/cottonweeddet3", None, title="CottonWeedDet3",
                             description="carpetweed, morningglory and Palmer amaranth bounding boxes"),
                        source_id="kg_yuzhenlu__cottonweeddet3"), cfg, nm, tg)
    r = PF.precheck(lu, cfg, ctx())
    check("a declared evaluation-lab source without the copy scan: hold, R3 (P9)",
          lu["lab_group"] == "LuLab" and "copy_scan_pending" in codes(r) and r["risk"] == "R3", (lu["lab_group"], r))
    r = PF.precheck(lu, cfg, ctx(copy_scan=(True, {"path": "x"})))
    check("... and passes it once the copy scan is available", "copy_scan_pending" not in codes(r), r)
    big = dict(ok, bytes=60e9)
    r = PF.precheck(big, cfg, ctx())
    check("over the 50 GB per-source cap: hold, R3", "over_source_cap" in codes(r) and r["risk"] == "R3", r)
    appr = json.loads(json.dumps(cfg.raw))
    appr["budgets"]["approved_bytes"] = {ok["source_id"]: 80e9}
    from weed_optimizer_framework.tools.collect.config import CollectConfig
    cfg2 = CollectConfig(appr, cfg.path, cfg.sha256, cfg.funnel, cfg.eppo, cfg.eppo_record)
    check("... unless a person approved more for that source", "over_source_cap" not in codes(PF.precheck(big, cfg2, ctx())))
    r = PF.precheck(ok, cfg, ctx(bytes_today=50e9))
    check("the daily byte cap: hold, no person", "daily_bytes" in codes(r) and r["risk"] is None, r)
    r = PF.precheck(ok, cfg, ctx(bytes_total=200e9))
    check("the byte envelope: hold, R3", "byte_envelope" in codes(r) and r["risk"] == "R3", r)
    r = PF.precheck(ok, cfg, ctx(state={ok["source_id"]: {"status": "fetched", "failed_attempts": 3, "bytes": 0}}))
    check("three failed attempts close the source", "attempts_exhausted" in codes(r) and r["action"] == "close", r)
    r = PF.precheck(ok, cfg, ctx(registry={ok["source_id"]: {"status": "quarantined", "annotation": "intake_v1"}}))
    check("a quarantined source is closed", "quarantined" in codes(r) and r["action"] == "close", r)
    r = PF.precheck(ok, cfg, ctx(registry={ok["source_id"]: {"status": "downloaded", "annotation": "yolo"}}))
    check("a source registered outside intake is closed (never registered twice)",
          "registered_outside_intake" in codes(r), r)
    mh = PF.decide(dict(cand("mendeley", "d3n3mgjjbv/2", None, title="MH-Weed16"), source_id="mendeley_d3n3mgjjbv_v2"),
                   cfg, nm, tg)
    r = PF.precheck(mh, cfg, ctx())
    check("another campaign's source is held (the funnel owns MH-Weed16)", "owned_by_funnel" in codes(r)
          and r["action"] == "hold", r)
    os.environ["SLURM_JOB_ID"] = "123"
    try:
        r = PF.precheck(ok, cfg, ctx())
        check("inside Slurm, a provider not placed on the cluster is refused", "not_placed_on_cluster" in codes(r)
              and r["action"] == "refuse", r)
        pl = {"in_slurm": True, "providers": {"roboflow": {"placement": "cluster"}}}
        check("... and allowed once the probe placed it there", PF.precheck(ok, cfg, ctx(placement=pl))["ok"])
        gh = PF.decide(cand("github", "o/r", ["Palmer Amaranth"]), cfg, nm, tg)
        pl2 = {"in_slurm": True, "providers": {"github": {"placement": "cluster"}}}
        check("a lab-only provider is never placed on the cluster", "not_placed_on_cluster" in codes(
            PF.precheck(gh, cfg, ctx(placement=pl2))))
    finally:
        os.environ.pop("SLURM_JOB_ID", None)


def test_copy_scan(cfg):
    from weed_optimizer_framework.tools.collect import prefilter as PF
    print("copy-scan readiness (P9)")
    ready, ev = PF.copy_scan_ready(cfg)
    check("no calibration file: not ready", ready is False, ev)
    try:
        from weed_optimizer_framework.tools.inc2 import guard as G2
    except ImportError as e:
        print("  skip a passed calibration makes it ready: inc2.guard is not importable (%s)" % e)
        return
    # the calibration format is group A's (inc2.guard.load_calibration decides what passed); the wiring is
    # pinned here with a stand-in loader that passes a file whose record says ok
    real = G2.load_calibration

    def stand_in(path, embedder_name=None):
        doc = json.loads(pathlib.Path(path).read_text())
        if not (doc.get("calibration") or {}).get("ok"):
            raise G2.GuardError("not passed")
        return {"file": {"path": str(path)}, "cos_threshold": doc["calibration"]["cos_threshold"]}
    G2.load_calibration = stand_in
    try:
        p = W.TMP / "inc" / "splits" / "v2" / "leak" / "leak_calibration.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"calibration": {"ok": False, "cos_threshold": 0.9}}))
        check("a calibration the guard's loader refuses: not ready", PF.copy_scan_ready(cfg)[0] is False)
        p.write_text(json.dumps({"calibration": {"ok": True, "cos_threshold": 0.9}}))
        ready, ev = PF.copy_scan_ready(cfg)
        check("a calibration the guard's loader accepts: ready", ready is True and ev["path"] == str(p), ev)
    finally:
        G2.load_calibration = real


def main():
    try:
        cfg, nm, tg = setup()
        test_decide(cfg, nm, tg)
        test_clearance(cfg, nm, tg)
        test_rank(cfg, nm, tg)
        test_precheck(cfg, nm, tg)
        test_copy_scan(cfg)
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
