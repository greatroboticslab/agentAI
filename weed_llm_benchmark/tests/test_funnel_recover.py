#!/usr/bin/env python3
"""funnel/recover.py (runner docs/FUNNEL_AUDIT_RUNNER.md §4.18, §5.4.4;
contract docs/FUNNEL_AUDIT.md §7, §9.2, DEC-7). Mutation points M4, M8, M9.

On the synthetic recovery world of tests/funnel_labels_fixtures.recover_world
(Step 1 files in verify's formats, a census ledger, frames, an audit, the
labeller's identity record, H1-pre, class maps, a leak record, name status,
gold and one qualified kNN judge's scores):
  * mask(): the pixel box rounded outward holds exactly the image's mean RGB,
    everything else is unchanged, the PNG is lossless and content-addressed;
  * the never-train check refuses a masked copy of a planted near-eval image
    on its unmasked pool dHash although the masked dHash passes (M8), and
    recovery.json is written "refused"; an H6 copy also stops the run;
  * each policy on a planted case:
      R-V  verified and other_ok boxes kept, the conflict box masked;
      R-A  the Ragweed box keeps its source label; a named relative the probe
           calls Palmer amaranth stays the other class (the sibling guard,
           M4); a generic box the probe calls a target is masked;
      R-C  id 12 becomes the mapped class through an accepted card+geometry
           map, id 8 stays the other class; a card-only proposal is not
           accepted;
      R-T  a scientific synonym becomes its target;
      R-J  a no-name box the qualified judge calls Waterhemp is relabelled; a
           box of a failing stratum a judge calls a target is masked; a
           target-related box is masked, never relabelled;
  * a stratum below the box gate recovers nothing and says why;
  * a genus-unsure answer masks its box (the image then recovers nothing);
  * the identity check failing masks every Ragweed box under R-A;
  * H1 not supported recovers nothing under R-A;
  * a quarantined source contributes no image, even its clean ones (M9); a
    source without a licence is refused; a not-recoverable source and base
    B's images are left out;
  * the H10d hold-out: whole near-duplicate groups of at least
    min_group_images images, absent from the recovered pool, written to
    domain_dev.jsonl;
  * the step-1 labels are unchanged (hash check) and recovery.json records it;
  * the overlay labels are content-addressed and hold only kept boxes;
  * arms: U = B + recovered rows capped at |B| by a seeded draw stratified
    by pool; U_ctl the same keys with the control labels and unmasked
    images; CLASS_ctl and JUDGE_ctl the realloop steps' images with control
    labels; B's rows verbatim; a step missing from the experiment is
    recorded, not guessed.

Needs numpy and PIL (skips otherwise). No network, no GPU (the H6 detector
is injected).

Run:  python3 tests/test_funnel_recover.py
"""
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_labels_fixtures as FX  # noqa: E402

TMP = FX.setup("funnel_recover_")
FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return None


def main():
    missing = FX.have("numpy", "PIL")
    if missing:
        print("SKIP: %s not installed" % ", ".join(missing))
        SKIPS.append("recover")
        return
    import numpy as np
    from PIL import Image
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import recover as RC
    from weed_optimizer_framework.tools.funnel import (NeverTrainHit, RecoverError, read_json, read_jsonl,
                                                       write_json_atomic)
    fd = pathlib.Path(os.environ["INC_DIR"]) / "funnel"
    out = pathlib.Path(os.environ["INC_DIR"]) / "step1_r1"
    domain = FX.test_domain(TMP)
    prereg = D.load_prereg(fd / "prereg_v1.json")
    world = FX.recover_world(TMP, domain)
    ad = world["adapter"]
    K = world["classes"]
    no_copy = lambda images: []   # noqa: E731

    # ---------------------------------------------------------------- mask
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(20, dtype=np.uint8)[None, :] * 10
    arr[:, :, 1] = 50
    src = TMP / "mask_src.png"
    Image.fromarray(arr).save(src)
    mpath, msha = RC.mask(src, [(0, 0.5, 0.5, 0.33, 0.3)], TMP / "masks", key="m")
    got = np.asarray(Image.open(mpath).convert("RGB"))
    mean = np.rint(arr.reshape(-1, 3).astype(float).mean(axis=0)).astype(np.uint8)
    x0, x1 = int(np.floor((0.5 - 0.165) * 20)), int(np.ceil((0.5 + 0.165) * 20))
    y0, y1 = int(np.floor((0.5 - 0.15) * 10)), int(np.ceil((0.5 + 0.15) * 10))
    inside = (got[y0:y1, x0:x1] == mean).all()
    outside = got.copy()
    outside[y0:y1, x0:x1] = arr[y0:y1, x0:x1]
    check("mask: the rounded-out box holds the mean RGB exactly", inside and (x0, x1, y0, y1) == (6, 14, 3, 7),
          (x0, x1, y0, y1))
    check("mask: every other pixel is unchanged (lossless PNG)", np.array_equal(outside, arr))
    check("mask: content-addressed name and sha256", mpath.name == "m.%s.png" % msha[:16]
          and C.sha256_file(mpath) == msha)
    check("mask: the same input gives the same file", RC.mask(src, [(0, .5, .5, .33, .3)], TMP / "masks", key="m")
          == (mpath, msha))

    # ------------------------------------------------- the never-train guard
    label_sha = {k: C.sha256_file(im["label"]) for k, im in world["images"].items()}
    pols = ("R-A", "R-C", "R-T", "R-V", "R-J")
    err = raises(lambda: RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy), NeverTrainHit)
    rec = read_json(out / "recovery.json")
    check("M8: a masked copy of a near-eval image is refused on the unmasked check", err and rec["status"] == "refused"
          and any(h["key"] == "nd2_near" and h["which"] == "unmasked" for h in rec["guards"]["listed"]["never_train"]),
          (err, rec.get("guards")))
    masked_near = [p for p in (out / "images_masked").rglob("nd2_near.*.png")]
    mh = C.dhash(masked_near[0]) if masked_near else None
    check("... although the masked copy's dHash passes the index", masked_near and mh is not None
          and bin(mh ^ world["near_hash"]).count("1") > 6)
    ad._guard = world["guard_clean"]

    def one_copy(images):
        return [{"key": im["key"], "eval_split": "dev", "eval_key": "x"} for im in images if im["key"] == "p2_j1"]
    check("an H6 copy among the recovered images stops the run",
          raises(lambda: RC.run(prereg, domain, fd, out, pols, ad, detector=one_copy), NeverTrainHit))

    # --------------------------------------------------------------- a run
    doc = RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy, force=True)
    rows = {r["key"]: r for r in read_jsonl(out / "recovered_pool.jsonl")}
    check("recovery.json complete", doc["status"] == "complete" and doc["guards"]["never_train"]["hits"] == 0)

    def labels_of(key):
        return [tuple(round(x, 4) for x in b[1:]) + (b[0],) for b in C.read_yolo(rows[key]["label"])]

    def classes(key):
        return sorted(b[0] for b in C.read_yolo(rows[key]["label"]))
    v = rows.get("nd2_veto")
    check("R-V: verified and other_ok kept, the conflict masked", v and v["policy"] == "R-V" and v["pool"] == "VETO"
          and classes("nd2_veto") == sorted([K["W"], K["O"]]) and len(v["masked_boxes"]) == 1
          and v["masked_boxes"][0][0] == K["O"] and v["image"] != v["unmasked_image"], v)
    a1 = rows.get("nd_auth1")
    check("R-A: the Ragweed box keeps its source label", a1 and a1["policy"] == "R-A" and K["R"] in classes("nd_auth1"))
    check("M4: the named relative the probe calls a target stays the other class (sibling guard)",
          a1 and classes("nd_auth1").count(K["O"]) == 1 and len(a1["masked_boxes"]) == 1)
    check("R-A: a generic box the probe calls a target is masked",
          a1 and a1["masked_boxes"][0][1:3] == [0.5, 0.7])
    check("a stratum below the box gate recovers nothing", "nd_auth2" not in rows)
    g_bad = doc["gates"][world["strata"]["wh_bad"]]
    check("... and says why", not g_bad["passed"] and "box gate: 'label' lb 0.700" in g_bad["why"], g_bad["why"])
    check("a genus-unsure answer masks its box (nothing left to recover)", "nd_auth3" not in rows
          and "nd_auth4" in rows)
    c1 = rows.get("mh_class1")
    check("R-C: id 12 becomes the mapped class, id 8 stays the other class",
          c1 and c1["pool"] == "CLASS" and classes("mh_class1") == sorted([K["S"], K["O"]]))
    maps = {m["src_id"]: m for m in doc["class_maps"]}
    check("R-C: a card-only proposal is not accepted", maps["12"]["accepted"] and not maps["5"]["accepted"]
          and "card and the geometry" in maps["5"]["why"])
    check("R-T: a scientific synonym becomes its target", rows.get("xs_syn1") and classes("xs_syn1") == [K["W"]]
          and rows["xs_syn1"]["policy"] == "R-T")
    j1 = rows.get("p2_j1")
    check("R-J: the judged no-name box is relabelled; the failing-stratum box a judge calls a target is masked",
          j1 and j1["pool"] == "JUDGE" and classes("p2_j1") == [K["W"]] and len(j1["masked_boxes"]) == 1)
    rel = rows.get("p2_rel")
    check("R-J: a target-related box is masked, never relabelled", rel and classes("p2_rel") == [K["W"]]
          and len(rel["masked_boxes"]) == 1 and rel["masked_boxes"][0][1:3] == [0.3, 0.3])
    check("M9: a quarantined source contributes no image, even its clean ones",
          not any(k.startswith("qs_") for k in rows)
          and any(r.get("source") == FX.QS for r in doc["refusals"]))
    check("a source without a licence is refused", "nl_j" not in rows
          and any(r.get("source") == FX.NL and "licence" in r["why"] for r in doc["refusals"]))
    check("a not-recoverable source is left out", "nr_j" not in rows)
    check("base B's images are left out", "p2_base" not in rows)
    dd = read_json(out / "domain_dev.json")
    held = [r["key"] for r in read_jsonl(out / "domain_dev.jsonl")]
    an = dd["sources"].get(FX.AN, {})
    groups = {world["images"][k]["nd"] for k in held}
    check("H10d: whole near-dup groups of at least min_group_images images of the large source",
          an.get("images", 0) >= 20 and len(held) == an["images"]
          and all(sum(1 for k in world["images"] if world["images"][k]["nd"] == g) ==
                  sum(1 for k in held if world["images"][k]["nd"] == g) for g in groups), (an, len(held)))
    check("H10d: held-out images are absent from every recovered pool", not (set(held) & set(rows)))
    check("H10d: a small source has no hold-out and says why", "fewer than" in dd["sources"][FX.ND]["why"])
    dd_rows = [{"key": "g%d_%d" % (g, j), "source": FX.AN, "near_dup3": "n:g%d" % g, "image": "g%d_%d.png" % (g, j),
                "base": (g, j) == (0, 0)} for g in range(6) for j in range(10)]
    held_b = set()
    for seed in range(12):                     # every seeded start: the group holding a base image is never chosen
        d_b = RC.domain_dev([FX.AN], dd_rows, domain, seed_prefix="funnel/test/h10d/%d" % seed)
        held_b.update(d_b["keys"])
    check("H10d: a group holding a base image is never held out, and base images are never domain-dev rows",
          held_b and not any(k.startswith("g0_") for k in held_b), sorted(held_b)[:5])
    check("every other image of the large source is recovered",
          sum(1 for k in rows if k.startswith("an_j_")) == 44 - len(held))
    unchanged = all(C.sha256_file(world["images"][k]["label"]) == label_sha[k] for k in world["images"])
    check("the step-1 labels are unchanged", unchanged and doc["source_labels_unchanged"]["changed"] == 0
          and doc["source_labels_unchanged"]["checked"] > 0)
    r0 = rows["nd_auth1"]
    check("overlay labels are content-addressed under labels_overlay/",
          "labels_overlay" in r0["label"] and C.sha256_file(r0["label"])[:16] in r0["label"]
          and r0["label_sha256"] == C.sha256_file(r0["label"]))
    check("rows carry ctl labels, provenance group, lab and licence",
          r0["ctl_label"] == world["images"]["nd_auth1"]["label"] and r0["provenance_group"] == "NDSU"
          and r0["licence"] and r0["lab"] == domain.lab_of(FX.ND))
    check("counts by pool (the near-eval twin is recovered once the index is clean)",
          doc["counts"]["VETO"]["images"] == 2 and doc["counts"]["CLASS"]["images"] == 2
          and doc["counts"]["AUTH"]["images"] == 2, {k: v["images"] for k, v in doc["counts"].items()})
    again = RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy)
    check("a rerun on the same inputs is a no-op", again["recovered_pool"] == doc["recovered_pool"])
    check("R-J: a judge's target that is not the probe's predicted class (the gate's event) is not relabelled",
          "p2_jp" not in rows)
    pm = rows.get("p2_mask")
    check("R-J: a box only an unqualified judge calls a target is masked (any judge masks)",
          pm and classes("p2_mask") == [K["W"]] and len(pm["masked_boxes"]) == 1
          and pm["masked_boxes"][0][1:3] == [0.7, 0.7], pm)
    cdir = TMP / "lic_fd" / "cards"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "index.json").write_text(json.dumps({"cards": {
        "rf_a": [{"status": 401, "licence": "x"}, {"status": 200, "licence": "CC BY 4.0", "url": "u1",
                                                    "fetched_utc": "t1"}],
        "rf_b": [{"status": 200, "licence": None}]}}))
    fl = RC.fetched_licences(TMP / "lic_fd")
    check("fetched_licences: the first 200 answer with a licence, with its URL; none without one",
          fl == {"rf_a": "CC BY 4.0 (fetched by L11a from u1, t1)"}, fl)
    check("fetched_licences: no cards index, no licences", RC.fetched_licences(TMP / "no_such_fd") == {})
    check("recovery.json records the judges whose calls may mask", doc.get("mask_judges") == ["J-knn1", "J-knn2"],
          doc.get("mask_judges"))
    tn = list(domain.target_names)
    check("mask_judges_of: a judge with only target labels never masks (it names a target for every box)",
          RC.mask_judges_of({"J-a": tn, "J-b": tn + ["other"], "J-c": tn + ["non_object"], "J-d": []}, domain)
          == ["J-b", "J-c"])
    check("R-J: an H4-stratum box whose voters qualify against both attractor sets is relabelled",
          rows.get("p2_g4y") and classes("p2_g4y") == [K["W"]])
    check("a not-recoverable source is refused even with a licence (its licence is in the test config)",
          FX.NR in (domain.raw["sources"].get("licences") or {}) and "nr_j" not in rows)

    # a rerun with other inputs and no --force: refused before anything is written
    snap = {n: (out / n).read_bytes() for n in ("recovery.json", "recovered_pool.jsonl", "domain_dev.jsonl",
                                               "domain_dev.json")}
    jq_p = fd / "judge_qualification.json"
    jq_bytes = jq_p.read_bytes()
    jq2 = json.loads(jq_bytes)
    jq2["judges"]["J-knn2"]["by_type"]["other_named"]["qualified"] = False
    write_json_atomic(jq_p, jq2)
    err = raises(lambda: RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy), RecoverError)
    check("other inputs without --force are refused and nothing is written",
          err and "nothing was written" in err
          and all((out / n).read_bytes() == b for n, b in snap.items()), err)
    RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy, force=True)
    rows_n = {r["key"] for r in read_jsonl(out / "recovered_pool.jsonl")}
    check("R-J in an H4 stratum: voters not qualified against the independent attractors relabel nothing",
          "p2_g4y" not in rows_n and "p2_j1" in rows_n)
    jq_p.write_bytes(jq_bytes)
    RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy, force=True)
    snap = {n: (out / n).read_bytes() for n in ("recovery.json", "recovered_pool.jsonl")}
    ad._guard = world["guard_hit"]
    rq0 = read_json(fd / "rl_qualification.json")
    write_json_atomic(fd / "rl_qualification.json", dict(rq0, note="changed"))
    check("a never-train hit on a rerun without --force keeps the complete record",
          raises(lambda: RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy), NeverTrainHit)
          and all((out / n).read_bytes() == b for n, b in snap.items()))
    write_json_atomic(fd / "rl_qualification.json", rq0)
    ad._guard = world["guard_clean"]

    # the leak record: a failed calibration or an H6(b) incident stops F9 before anything is planned
    lk = read_json(fd / "leak_v1.json")
    for label, bad_leak in (("a failed copy-detector calibration", dict(lk, calibration={"ok": False})),
                            ("an H6(b) incident", dict(lk, h6b=dict(lk["h6b"], incident=True, base_copy=True))),
                            ("no H6(b) record", {k: v for k, v in lk.items() if k != "h6b"})):
        write_json_atomic(fd / "leak_v1.json", bad_leak)
        check("refused: %s" % label, raises(lambda: RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy,
                                                           force=True), RecoverError))
    write_json_atomic(fd / "leak_v1.json", lk)

    # --------------------------------------------------------------- gates
    au0 = read_json(fd / "audit_v1.json")
    th = RC._recovery_thresholds(prereg)
    uc = RC.unit_classes(read_json(fd / "class_maps.json"), read_json(fd / "name_status_v2.json"), domain)
    check("unit classes: the card map for the card-resolved source, the scientific synonym elsewhere",
          uc.get("c:%s|12" % FX.MH) == "Sicklepod" and uc.get("c:%s|0" % FX.XS) == "Waterhemp", uc)
    as_target = json.loads(json.dumps(au0))
    for e in as_target["strata"]:
        e["event"] = "target"                    # what the audit's per-stratum records estimate today
    seen = set()
    as_target["strata"] = [e for e in as_target["strata"] if not (e["stratum"] in seen or seen.add(e["stratum"]))]
    gt = RC.gates(as_target, {}, read_json(fd / "relation_geometry_v1.json"), {}, domain, thresholds=th, unit_class=uc)
    check("a gate is never read from the 'target' share: every stratum fails and names the missing estimate",
          not any(g["passed"] for g in gt.values())
          and "no 'label' estimate" in gt[world["strata"]["rag"]]["why"]
          and "no 'pred' estimate" in gt[world["strata"]["g1"]]["why"]
          and "no 'purity:Sicklepod' estimate" in gt[world["strata"]["mh"]]["why"], {s: g["why"] for s, g in gt.items()})

    def one(stratum_id, **mut):
        a = json.loads(json.dumps(au0))
        for e in a["strata"]:
            if e["stratum"] == stratum_id:
                for k, v in mut.items():
                    e[k] = v
        return RC.gates(a, {}, read_json(fd / "relation_geometry_v1.json"), {}, domain, thresholds=th,
                        unit_class=uc)[stratum_id]
    rag = world["strata"]["rag"]
    check("the gate takes the smaller bound over both unsure assignments",
          not one(rag, unsure={"as_no": {"estimate": 0.95, "interval": [0.80, 0.99]},
                               "as_yes": {"estimate": 0.97, "interval": [0.95, 0.99]}})["passed"])
    check("a precision not Rogan-Gladen corrected fails the gate",
          not one(rag, rogan_gladen={"applied": False, "flag": None})["passed"])
    check("a Rogan-Gladen flag fails the gate",
          not one(rag, rogan_gladen={"applied": True, "flag": "se_plus_sp_below_0.7"})["passed"])
    check("the passing gate passes (control)", one(rag)["passed"])
    check("level: species for every class, genus only for a genus-rank class",
          RC.level_ok({"level": "species"}, K["W"], domain)
          and not RC.level_ok({"level": "genus"}, K["W"], domain)
          and RC.level_ok({"level": "genus"}, domain.class_id("MorningGlory"), domain)
          and not RC.level_ok({"level": "plant"}, domain.class_id("MorningGlory"), domain))
    mh_unit = "G3/unit=c:%s|12" % FX.MH
    other_cls = RC._accepted_maps({"proposals": [{"source": FX.MH, "src_id": "12", "map_to": "Sicklepod",
                                                  "via": "card+geometry", "status": "proposed"}]},
                                  {mh_unit: {"class": "MorningGlory", "class_gate": True, "why": "passed"}},
                                  "supported", domain)
    check("a map is not accepted on gates read for another class",
          not other_cls[0]["accepted"] and "gates are for MorningGlory" in other_cls[0]["why"], other_cls)
    check("the audit holding two estimates of one event for a stratum is refused",
          raises(lambda: RC.gates(dict(au0, strata=au0["strata"] + [au0["strata"][1]]), {}, {}, {}, domain,
                                  thresholds=th, unit_class=uc), RecoverError))
    rq = read_json(fd / "rl_qualification.json")
    write_json_atomic(fd / "rl_qualification.json", {"identity": {}})
    RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy, force=True)
    rows_i = {r["key"] for r in read_jsonl(out / "recovered_pool.jsonl")}
    check("no identity record for a class the config checks before R-A: its boxes are masked",
          "nd_auth1" not in rows_i and "nd_auth4" in rows_i)
    write_json_atomic(fd / "rl_qualification.json", {"identity": {"Ragweed": {"pass": False, "before": ["R-A"]}}})
    RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy, force=True)
    rows2 = {r["key"] for r in read_jsonl(out / "recovered_pool.jsonl")}
    check("the identity check failing masks Ragweed under R-A", "nd_auth1" not in rows2 and "nd_auth4" in rows2)
    write_json_atomic(fd / "rl_qualification.json", rq)
    au = read_json(fd / "audit_v1.json")
    au2 = json.loads(json.dumps(au))
    au2["hypotheses"]["H1"]["verdict"] = "inconclusive"
    write_json_atomic(fd / "audit_v1.json", au2)
    RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy, force=True)
    rows3 = {r["key"]: r for r in read_jsonl(out / "recovered_pool.jsonl")}
    check("H1 not supported: nothing under R-A", not any(r["policy"] == "R-A" for r in rows3.values()))
    au2["valid"] = False
    write_json_atomic(fd / "audit_v1.json", au2)
    check("an invalid audit is refused", raises(lambda: RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy,
                                                               force=True), RecoverError))
    write_json_atomic(fd / "audit_v1.json", au)
    check("R-F without the chain record is refused",
          raises(lambda: RC.run(prereg, domain, fd, out, ("R-F",), ad, detector=no_copy, force=True), RecoverError))
    # R-F with a chain record: recovered as R-C (an accepted card map whose gates passed), else refused
    rf1 = FX.png(TMP / "refetch" / "mh_rf1.png", 42)
    rf2 = FX.png(TMP / "refetch" / "mh_rf2.png", 43)
    checks = {c: True for c in RC.CHAIN_CHECKS}
    chain = {"images": [
        {"key": "mh_rf1", "image": rf1, "sha256": C.sha256_file(rf1), "source": FX.MH,
         "label_boxes": [[K["S"], .5, .5, .2, .2], [K["O"], .2, .2, .1, .1]], "masked_boxes": [],
         "near_dup3": "n:mh_rf1", "dhash": 0x0123456789ABCDEF, "strata": [world["strata"]["mh"]], "checks": checks},
        {"key": "mh_rf2", "image": rf2, "sha256": C.sha256_file(rf2), "source": FX.MH,
         "label_boxes": [[domain.class_id("MorningGlory"), .5, .5, .2, .2]], "masked_boxes": [],
         "near_dup3": "n:mh_rf2", "dhash": 0x0FEDCBA987654321, "strata": [], "checks": checks},
        {"key": "mh_rf3", "image": rf1, "sha256": C.sha256_file(rf1), "source": FX.MH,
         "label_boxes": [[K["S"], .5, .5, .2, .2]], "masked_boxes": [], "near_dup3": "n:mh_rf3",
         "dhash": 0x1111, "strata": [], "checks": dict(checks, h6=False)}]}
    write_json_atomic(fd / RC.CHAIN_FILE, chain)
    out_rf = TMP / "r1_rf"
    drf = RC.run(prereg, domain, fd, out_rf, ("R-F",), ad, detector=no_copy, force=True)
    rf_rows = {r["key"]: r for r in read_jsonl(out_rf / "recovered_pool.jsonl")}
    check("R-F: a refetched image whose target class has an accepted card map is recovered (FETCH)",
          set(rf_rows) == {"mh_rf1"} and rf_rows["mh_rf1"]["pool"] == "FETCH", sorted(rf_rows))
    check("R-F: a class without an accepted map, or a failed chain check, is refused",
          any(r.get("key") == "mh_rf2" and "accepted card map" in r["why"] for r in drf["refusals"])
          and any(r.get("key") == "mh_rf3" and "failed" in r["why"] for r in drf["refusals"]), drf["refusals"])
    au_h = json.loads(json.dumps(au0))
    au_h["hypotheses"]["H3a"]["verdict"] = "inconclusive"
    write_json_atomic(fd / "audit_v1.json", au_h)
    drf2 = RC.run(prereg, domain, fd, out_rf, ("R-F",), ad, detector=no_copy, force=True)
    check("R-F: H3a not supported recovers nothing", drf2["recovered_pool"]["images"] == 0
          and any("H3a is inconclusive" in r["why"] for r in drf2["refusals"]))
    write_json_atomic(fd / "audit_v1.json", au0)
    check("a recovered row without a join label cannot enter a control arm",
          raises(lambda: RC._ctl(dict(rf_rows["mh_rf1"])), RecoverError))
    check("an unknown policy is refused",
          raises(lambda: RC.run(prereg, domain, fd, out, ("R-Z",), ad, detector=no_copy), RecoverError))
    doc = RC.run(prereg, domain, fd, out, pols, ad, detector=no_copy, force=True)
    rows = {r["key"]: r for r in read_jsonl(out / "recovered_pool.jsonl")}

    # ---------------------------------------------------------------- arms
    exp_dir = pathlib.Path(os.environ["INC_DIR"]) / "realloop_v2"
    (exp_dir / "manifests").mkdir(parents=True, exist_ok=True)
    cls_keys = sorted(k for k, r in rows.items() if r["pool"] == "CLASS")
    judge_keys = sorted(k for k, r in rows.items() if r["pool"] == "JUDGE")[:3]
    m1 = C.write_manifest(exp_dir / "manifests" / "rec_class_1.jsonl",
                          [{k2: rows[k][k2] for k2 in C.MANIFEST_KEYS} for k in cls_keys])
    exp = {"exp": "realloop_v2", "steps": [{"name": "REC-CLASS-1", "manifest": str(exp_dir / "manifests" /
                                                                                  "rec_class_1.jsonl"),
                                            "manifest_sha256": m1}],
           "step1": {"increment_sources": {"mode": "recovered",
                                           "overlay": {"recovery_sha256": C.sha256_file(out / "recovery.json")}}}}
    write_json_atomic(exp_dir / "exp.json", exp)
    base = [{"image": str(TMP / ("b%d.png" % i)), "label": str(TMP / ("b%d.txt" % i)), "sha256": "s%d" % i,
             "label_sha256": "l%d" % i, "source": "train_core", "session": "", "key": "base_%d" % i} for i in range(5)]
    bpath = C.INC_DIR / "step1" / "base_B.jsonl"
    C.write_manifest(bpath, base)
    arms = RC.arms(exp_dir, out, bpath, ad)
    A = arms["arms"]
    u = read_jsonl(out / "arms" / "U.jsonl")
    uc = read_jsonl(out / "arms" / "U_ctl.jsonl")
    check("U = B + recovered rows capped at |B|", A["U"]["from_base"] == 5 and A["U"]["from_recovered"] == 5
          and len(u) == 10, A["U"])
    rec_u = [r for r in u if not r["key"].startswith("base_")]
    sizes = {}
    for r in rows.values():
        sizes[r["pool"]] = sizes.get(r["pool"], 0) + 1
    quota = {p: 5.0 * n / len(rows) for p, n in sizes.items()}
    want = {p: int(q) for p, q in quota.items()}
    for p in sorted(sizes, key=lambda p: (-(quota[p] - want[p]), p))[:5 - sum(want.values())]:
        want[p] += 1
    want = {p: n for p, n in want.items() if n}
    check("U's recovered part is stratified by pool (largest remainder, ties by pool name)",
          A["U"]["by_pool"] == want, (A["U"]["by_pool"], want))
    check("U_ctl: the same keys with control labels and unmasked images",
          sorted(r["key"] for r in uc) == sorted(r["key"] for r in u)
          and all(r["label"] == rows[r["key"]]["ctl_label"] and r["image"] == rows[r["key"]]["unmasked_image"]
                  for r in uc if not r["key"].startswith("base_")))
    check("B's rows copied verbatim into every arm",
          all({k: r[k] for k in C.MANIFEST_KEYS} in u for r in base))
    cc = read_jsonl(out / "arms" / "CLASS_ctl.jsonl")
    check("CLASS_ctl = B + REC-CLASS-1's images with control labels",
          sorted(r["key"] for r in cc if not r["key"].startswith("base_")) == cls_keys
          and all(r["label"] == rows[r["key"]]["ctl_label"] for r in cc if not r["key"].startswith("base_")))
    check("a step missing from the experiment is recorded, not built", "JUDGE_ctl" in arms["not_built"]
          and "JUDGE_ctl" not in A)
    check("the U draw is seeded (a second build is identical)",
          RC.arms(exp_dir, out, bpath, ad)["arms"]["U"]["sha256"] == A["U"]["sha256"])
    del judge_keys, rec_u
    exp["step1"]["increment_sources"]["overlay"]["recovery_sha256"] = "0" * 64
    write_json_atomic(exp_dir / "exp.json", exp)
    from weed_optimizer_framework.tools.funnel import StaleInput
    check("arms refuse an experiment built from another recovery.json",
          raises(lambda: RC.arms(exp_dir, out, bpath, ad), StaleInput))
    exp["step1"] = {}
    write_json_atomic(exp_dir / "exp.json", exp)
    check("arms refuse an experiment that is not a recovered-mode build",
          raises(lambda: RC.arms(exp_dir, out, bpath, ad), RecoverError))


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "-"))
    sys.exit(1 if FAILURES else 0)
