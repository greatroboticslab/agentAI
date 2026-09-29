#!/usr/bin/env python3
"""The Step 1 adapter (docs/FUNNEL_AUDIT.md §3, §4.1, F3; runner §4.1-§4.6,
§5.2.1-§5.2.2).

What is pinned, and why:
  * adapters.load checks the interface: inc_step1 passes, a module lacking a
    function is refused;
  * on the REAL local Step 1 summaries and census_v0.json:
      - every Step 1 count contract §3 states is re-derived from the files and
        equals the number the contract text states (both read, never typed);
      - ledger_from_summaries gives S2 = no name + numeric + generic over the
        embedded boxes, and target_check kept / conflict / unknown and
        other_check conflict, each equal to the value the test reads from
        census_v0 (through name_status_v1) and admit_summary.json;
      - census_v0's rows are reproduced field by field from a synthetic
        crops.csv and pool_verdicts.npz in verify's formats (545,318 crops),
        through the census's own row aggregation;
  * on the synthetic Step 1 world (tests/funnel_world.py; skipped without
    sklearn): the census reconciles with the world's summaries; the verifier
    re-derivation reproduces pool_verdicts.npz and a flipped verdict is
    refused; every box and every pre-pool image has one ledger row, and the
    planted cases carry the right failed stages, first and sole causes and
    blockers; H5a finds the planted twin; KT2 leaves an unmatched copy box
    out; unit_keys gives a copy and its twin one provenance; a taxonomy cache
    missing a name refuses with the L12 message; a rerun with the same inputs
    is a no-op, with other inputs it refuses without --force, and after the
    sample lock it refuses with SampleLocked (also when census_v1.json is
    gone); a census that does not reconcile is written with ok false and
    refused; KT4/KT5 hold embedded boxes only; guard pairs carry disjointness
    keys (a copy its twin's provenance) and refuse a pair without its
    evaluation side; fail codes and blockers follow runner §4.1.

Run:  python3 tests/test_funnel_adapter.py
"""
import csv
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_adapter_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))


class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise AssertionError("the test touched the network")


socket.socket = _NoNet

import numpy as np  # noqa: E402

import funnel_world as FW  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.funnel import (AdapterError, SampleLocked, TaxonomyError,  # noqa: E402
                                                   read_json)
from weed_optimizer_framework.tools.funnel import adapters  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import ledger as L  # noqa: E402
from weed_optimizer_framework.tools.funnel.adapters import inc_step1 as A  # noqa: E402

FAILURES, SKIPS = [], []
LOCAL = ROOT / "results" / "framework" / "inc"
CONTRACT = ROOT.parent / "docs" / "FUNNEL_AUDIT.md"
CENSUS_V0 = LOCAL / "funnel" / "census_v0.json"


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return None


# ================================================================ interface
def test_interface():
    print("adapter interface")
    mod = adapters.load("inc_step1")
    check("inc_step1 passes the interface check", mod is A)
    check("the interface is the runner's", set(adapters.INTERFACE) == {
        "census", "ledger_from_summaries", "crop_table", "known_truth", "unit_keys", "guard_pairs", "pool_rows",
        "base_rows", "increment_rows", "eval_rows", "never_train_guard", "text_encoder", "step1_features",
        "j1_scores", "bioclip_embedder", "label_rows", "name_status_v1"})
    stub = types.ModuleType("stub_adapter")
    for f in adapters.INTERFACE:
        setattr(stub, f, lambda *a, **k: None)
    check("a complete stub passes", adapters.load(stub) is stub)
    del stub.guard_pairs
    msg = raises(lambda: adapters.load(stub), AdapterError)
    check("a stub lacking one function is refused, naming it", msg is not None and "guard_pairs" in msg, msg)
    check("a name that is no module is refused", raises(lambda: adapters.load("no_such_adapter"), AdapterError))
    check("a malformed name is refused", raises(lambda: adapters.load("../x"), AdapterError))
    check("the domain's adapter is inc_step1", adapters.for_domain(D.load("weed")) is A)
    print("name status v1")
    for name, want in (("", "no_name"), ("12", "numeric"), ("class3", "generic"), ("weed", "generic"),
                       ("Waterhemp", "target"), ("Redroot Pigweed", "target_related"), ("aphid", "not_a_plant"),
                       ("BroWeed", "named_other"), ("crop", "named_other")):
        check("name_status_v1(%r) = %s" % (name, want), A.name_status_v1(name) == want, A.name_status_v1(name))
    print("fail codes (runner §4.1, the first rule that applies)")
    tau = np.full(13, 0.5)
    sig = np.full(13, 0.6)
    for args, want in (((0, V.CONFLICT, 12, 0.7, 0.1), "argmax_other_confident"),
                       ((0, V.CONFLICT, 8, 0.7, 0.9), "argmax_target_confident"),
                       ((0, V.UNKNOWN, 8, 0.7, 0.1), "argmax_wrong"),
                       ((0, V.UNKNOWN, 0, 0.4, 0.9), "p_below_tau"),
                       ((0, V.UNKNOWN, 0, 0.7, 0.1), "cos_below_sigma"),
                       ((0, V.FAILED, -1, -1.0, -1.0), "failed"), ((0, V.VERIFIED, 0, 0.9, 0.9), None)):
        check("fail code %s" % want, A._fail_code(*args, tau=tau, sigma=sig) == want, A._fail_code(*args, tau, sig))
    print("blockers (runner §4.1: what keeps a verified box's image out)")
    check("the blocker priority is the runner's", A.BLOCKERS == ("other_called_target", "species_conflict",
                                                                 "species_unknown", "small_species_box", "failed"))
    for (lab, v), want in (((12, V.CONFLICT), "other_called_target"), ((0, V.CONFLICT), "species_conflict"),
                           ((0, V.UNKNOWN), "species_unknown"), ((0, V.SMALL), "small_species_box"),
                           ((0, V.FAILED), "failed"), ((12, V.FAILED), "failed"), ((12, V.SMALL), None),
                           ((12, V.OTHER_OK), None), ((0, V.VERIFIED), None)):
        check("a %s box judged %s blocks as %s" % ("OtherPlant" if lab == 12 else "target", v, want),
              A._blocker(lab, v) == want, A._blocker(lab, v))
    tau_nan = tau.copy()
    tau_nan[0] = np.nan
    check("a class without a threshold (NaN tau, never confident) fails on p, not on the prototype cosine",
          A._fail_code(0, V.UNKNOWN, 0, 0.9, 0.9, tau_nan, sig) == "p_below_tau",
          A._fail_code(0, V.UNKNOWN, 0, 0.9, 0.9, tau_nan, sig))


# ================================================================ real summaries
def test_real_summaries():
    print("contract §3 counts re-derived from the real local Step 1 files")
    cn = A.contract_numbers(CONTRACT)
    sc = A.stage_counts(LOCAL / "step1", CENSUS_V0)
    common = sorted(set(cn) & set(sc))
    bad = [(k, cn[k], sc[k]) for k in common if cn[k] != sc[k]]
    check("the contract states %d Step 1 counts; %d are re-derivable from the summaries and census_v0"
          % (len(cn), len(common)), len(common) >= 60, (len(cn), len(common)))
    check("every re-derived count equals the contract's", not bad, bad)
    rest = sorted(set(cn) - set(sc))
    check("the rest are the census-only counts (S7b v2, veto)",
          all(k.startswith("s7b_") or k.startswith("veto_") for k in rest), rest)

    print("ledger_from_summaries on the real local files")
    v0 = json.load(open(CENSUS_V0))
    admit = json.load(open(LOCAL / "step1" / "admit_summary.json"))
    out = TMP / "summ" / "funnel_ledger.json"
    led = A.ledger_from_summaries("weed", LOCAL / "step1", CENSUS_V0, out)
    check("the ledger validates", L.validate(L.load(out)) == [])
    check("derivation summaries, name status v1", led["derivation"] == "summaries"
          and led["name_status_version"] == "v1")
    idx = L.stage_index(led)
    un = sum(r["n"] for r in v0 if A.name_status_v1(r["src_name"]) in ("no_name", "numeric", "generic"))
    tot = sum(r["n"] for r in v0)
    ls = led["label_spaces"]
    got_un = sum(v["kinds"]["none"] + v["kinds"]["numeric"] + v["kinds"]["generic"] for v in ls.values())
    got_tot = sum(v["boxes"] for v in ls.values())
    check("S2 = %d / %d (read from census_v0 through name_status_v1)" % (un, tot),
          (got_un, got_tot) == (un, tot), (got_un, got_tot))
    tg = [r for r in v0 if r["label"] != "OtherPlant"]
    ot = [r for r in v0 if r["label"] == "OtherPlant"]
    want8 = (admit["boxes"]["verified"], sum(r["verdict"].get("conflict", 0) for r in tg),
             sum(r["verdict"].get("unknown", 0) for r in tg))
    got8 = (idx["S8"]["kept"], idx["S8"]["discarded"].get("conflict"), idx["S8"]["discarded"].get("unknown"))
    check("target_check kept / conflict / unknown = %s (admit_summary, census_v0)" % (want8,), got8 == want8, got8)
    want9 = sum(r["verdict"].get("conflict", 0) for r in ot)
    check("other_check conflict = %d (census_v0)" % want9, idx["S9"]["discarded"].get("conflict") == want9,
          idx["S9"]["discarded"])
    s1 = (sum(idx["S8"]["discarded"].values()) + idx["S9"]["discarded"]["conflict"], idx["S8"]["kept"])
    print("       S1 reject/accept = %d / %d" % s1)
    ps = json.load(open(LOCAL / "step1" / "pool_summary.json"))
    check("S3: the join keeps every pool box by class (pool_summary boxes_per_class)",
          idx["S7"]["kept_by_label"] == ps["boxes_per_class"])
    check("S5: the evidence criterion depends on the unaudited verifier",
          ("S12", "S8") in L.unaudited_dependencies(led))
    check("the fingerprint is the four identity inputs' hash", led["fingerprint"] == L.fingerprint(
        {k: C.sha256_file(LOCAL / "step1" / ("%s.json" % k))
         for k in ("admit_summary", "pool_summary", "select_summary", "calibration")}))
    check("domain scores: the reference q05 and the species-bearing sources' medians",
          led["domain_scores"]["reference"]["q05"] < 0.1 and len(led["domain_scores"]) >= 2)
    check("no verifier projection: reject_class is null", led["reject_class"] is None)
    proj = {"other_sample": {"taken": 50, "sources": {"a": 40, "b": 10}, "eligible_names": {"x": 90, "y": 10}}}
    led2 = A.ledger_from_summaries("weed", LOCAL / "step1", CENSUS_V0, TMP / "summ" / "l2.json",
                                   fit_info_projection=proj)
    check("with a projection, the reject class records its top cell and sample sources",
          led2["reject_class"]["eligible_top_cell"]["cell"] == "x"
          and led2["reject_class"]["eligible_top_cell"]["share"] == 0.9
          and led2["reject_class"]["sample_sources"] == {"a": 40, "b": 10}, led2["reject_class"])
    check("the S8b stage takes the projection's sample size", L.stage_index(led2)["S8b"]["kept"] == 50)
    s1 = TMP / "summ" / "step1"
    shutil.copytree(LOCAL / "step1", s1)
    (s1 / "verifier_fit_info.json").write_text(json.dumps(proj))
    led3 = A.ledger_from_summaries("weed", s1, CENSUS_V0, TMP / "summ" / "l3.json")
    check("without an explicit projection, step1/verifier_fit_info.json is read and recorded as an input",
          led3["reject_class"] == led2["reject_class"] and "verifier_fit_info" in led3["inputs"])


def test_census_v0_reproduction():
    print("census_v0 reproduced from a synthetic crops.csv and pool_verdicts.npz (verify's formats)")
    v0 = json.load(open(CENSUS_V0))
    d = TMP / "v0fixture"
    d.mkdir(parents=True)
    crops_path, npz_path = d / "crops.csv", d / "pool_verdicts.npz"
    verdict, pred, p = [], [], []
    sane = True
    with open(crops_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        n = 0
        for r in v0:
            lab = C.CLASS_NAMES.index(r["label"])
            vcount = {}
            for cname, cnt in sorted(r["pred_all"].items()):
                j = C.CLASS_NAMES.index(cname)
                conf = int(r["pred_confident"].get(cname, 0))
                for k in range(int(cnt)):
                    if k < conf:
                        v = V.VERIFIED if (lab < C.OTHER_PLANT and j == lab) else V.CONFLICT
                    else:
                        v = V.UNKNOWN if lab < C.OTHER_PLANT else V.OTHER_OK
                    vcount[v] = vcount.get(v, 0) + 1
                    wr.writerow([n, "pool", "%s__img%07d" % (V._sanitise(r["source"]), n), "/x.jpg", r["source"],
                                 r["source"], 0, "0.5", "0.5", "0.1", "0.1", 640, 480, lab, r["src_name"]])
                    verdict.append(V.VERDICT_CODES.index(v))
                    pred.append(j)
                    p.append(float(r["mean_p"][cname]))
                    n += 1
            sane = sane and vcount == r["verdict"]
    check("fixture: each census_v0 row's verdicts follow from its confident predictions", sane)
    meta = {"crops_sha256": C.sha256_file(crops_path), "verdict_codes": list(V.VERDICT_CODES)}
    np.savez(npz_path, meta=np.array(json.dumps(meta)), crop_id=np.arange(len(pred), dtype=np.int64),
             verdict=np.array(verdict, dtype=np.int8), pred=np.array(pred, dtype=np.int64),
             p=np.array(p, dtype=np.float64), cosine=np.full(len(pred), np.nan))
    rows = A.rows_from_step1_files(crops_path, npz_path)
    got = {(r["source"], r["src_name"], r["label"]): r for r in A.rows_v0_view(rows)}
    want = {(r["source"], r["src_name"], r["label"]): r for r in v0}
    check("one row per census_v0 row (%d)" % len(want), set(got) == set(want), set(got) ^ set(want))
    fields = ("n", "verdict", "pred_all", "pred_confident", "mean_p")
    bad = [k for k in want if k in got and any(got[k][f] != want[k][f] for f in fields)]
    check("every row equals census_v0 field by field (n, verdict, pred_all, pred_confident, mean_p)", not bad,
          [(k, {f: (got[k][f], want[k][f]) for f in fields if got[k][f] != want[k][f]}) for k in bad[:2]])
    check("the totals are census_v0's (%d crops)" % len(pred), sum(r["n"] for r in rows) == len(pred))
    meta["crops_sha256"] = "0" * 64
    np.savez(d / "stale.npz", meta=np.array(json.dumps(meta)), crop_id=np.arange(3), verdict=np.zeros(3, np.int8),
             pred=np.zeros(3, np.int64), p=np.zeros(3), cosine=np.zeros(3))
    check("verdicts made from another crops.csv are refused",
          raises(lambda: A.rows_from_step1_files(crops_path, d / "stale.npz"), AdapterError))


# ================================================================ the world
def ledger_rows(fd):
    return {r["id"]: r for r in L.iter_units(fd / "ledger.jsonl")}


def test_world():
    print("census on the synthetic Step 1 world")
    try:
        w = FW.build_world(TMP)
    except FW.WorldUnavailable as e:
        print("SKIP: %s" % e)
        SKIPS.append("world (%s)" % e)
        return
    fd = w.funnel_dir
    ps = json.load(open(w.step1 / "pool_summary.json"))
    names = sorted({v[0] for st in ps["per_slug"].values() for v in st["join"].values() if v[0]})
    tax = FW.build_taxonomy_cache(w, names)
    pre = fd / "prereg_v1.json"

    print("a missing taxonomy entry refuses before any Step 1 file is read")
    cache = json.load(open(tax))
    cache["resolved"].pop("maize")
    bad_tax = fd / "taxonomy_missing.json"
    bad_tax.write_text(json.dumps(cache))
    msg = raises(lambda: A.census(pre, "weed", fd, bad_tax, testing=True), TaxonomyError)
    check("the census refuses with the L12 message", msg is not None and "run fetch --what taxonomy (lever L12)" in msg,
          msg)
    check("nothing was written", not (fd / "census_v1.json").exists() and not (fd / "ledger.jsonl").exists())

    known = fd / "known_items_v1.json"
    known.write_text(json.dumps({"format": "funnel-known-items/1", "items": [
        {"name": "present one", "slug_patterns": ["^rf_zig-zag-lnodr__weed-detection-vanpe$"]},
        {"name": "missing one", "slug_patterns": ["^kg_nobody__nothing$"]}]}))
    c = A.census(pre, "weed", fd, tax, known_items=known, testing=True)
    check("the census reconciles with the world's summaries", c["reconciliation"]["ok"],
          [x for x in c["reconciliation"]["checks"] if not x["ok"]])
    check("the reconciliation covers verdicts, images, veto, per-slug drops and the sample",
          {"embedded_crops", "verdict_verified", "admitted_target_boxes", "small_boxes", "pool_images",
           "veto_lost_boxes", "veto_boxes_without_blocker", "s8b_taken"} <= {x["name"] for x in c["reconciliation"]["checks"]}
          and any(x["name"].startswith("dropped:") for x in c["reconciliation"]["checks"]))
    check("the re-derivation reproduced every pool verdict and argmax", c["rederivation"]["verdict_mismatch"] == 0
          and c["rederivation"]["argmax_mismatch"] == 0 and c["rederivation"]["crops"] == c["totals"]["embedded_crops"],
          c["rederivation"])
    check("the header records the inputs with sha256", len(c["inputs"]) > 20 and all(
        len(v["sha256"]) == 64 for v in c["inputs"].values()) and c["format"] == "funnel-census/1")
    check("H12: the known items are compared with the registry",
          c["h12"]["present"] == 1 and c["h12"]["total"] == 2, c["h12"])
    rows = ledger_rows(fd)
    meta = [json.loads(ln) for ln in open(w.step1 / "pool_meta.jsonl")]
    n_boxes = sum(len(m["boxes"]) for m in meta)
    n_drop = sum(sum(v.get("dropped", {}).values()) for v in ps["per_slug"].values())
    check("one ledger row per pool box (%d) and per pre-pool image (%d)" % (n_boxes, n_drop),
          sum(1 for r in rows.values() if r["unit"] == "box") == n_boxes
          and sum(1 for r in rows.values() if r["unit"] == "image") == n_drop)
    ids = [r["id"] for r in L.iter_units(fd / "ledger.jsonl")]
    check("ledger.jsonl is sorted by id", ids == sorted(ids))
    order = [s["id"] for s in D.load("weed").stages]
    probs = [(i, L.validate_unit_row(r, order, A.PATH_VOCAB)) for i, r in rows.items()]
    check("every row validates against the ledger's stages and the adapter's vocabulary",
          not [x for x in probs if x[1]], [x for x in probs if x[1]][:2])

    print("planted cases")
    P = w.planted
    veto = rows["b:%s#%d" % (P["vetoed"]["key"], P["vetoed"]["box"])]
    check("the vetoed verified box: S10 its only failed stage, sole cause, blocked by an OtherPlant box called "
          "a target", veto["path"]["S8"] == "verified" and veto["failed_stages"] == ["S10"]
          and veto["sole_cause"] == "S10" and veto["blockers"] == ["other_called_target"], veto)
    blk = rows["b:%s#%d" % (P["vetoed"]["key"], P["vetoed"]["blocker_box"])]
    check("its blocker is an OtherPlant box judged a conflict (S9)", blk["path"]["S9"] == "conflict"
          and blk["failed_stages"][0] == "S9", blk)
    ne = rows["b:%s#%d" % (P["veto_noevidence"]["key"], P["veto_noevidence"]["box"])]
    check("a box vetoed in an image of a non-evidenced source: two failed stages, no sole cause",
          ne["failed_stages"] == ["S10", "S12"] and ne["first_cause"] == "S10" and ne["sole_cause"] is None, ne)
    sm = rows["b:%s#%d" % (P["small_box"]["key"], P["small_box"]["box"])]
    check("the small box: S6 small, no crop, no probe values", sm["path"]["S6"] == "small"
          and "S6" in sm["failed_stages"] and sm["crop_id"] is None and sm["p"] is None, sm)
    hd = rows["b:%s#%d" % (P["hidden_target"]["key"], P["hidden_target"]["box"])]
    check("the numeric-named hidden target: an OtherPlant conflict called Sicklepod (S9), its source not "
          "evidenced (S12)", hd["label"] == C.OTHER_PLANT and hd["pred"] == C.CLASS_NAMES.index("Sicklepod")
          and hd["failed_stages"] == ["S9", "S12"] and hd["name_status_v2"] == "numeric", hd)
    wr = rows["b:%s#%d" % (P["wrong_ragweed"]["key"], P["wrong_ragweed"]["box"])]
    check("a rejected target-labelled box: S8 its sole cause, failure argmax_other_confident",
          wr["failed_stages"] == ["S8"] and wr["sole_cause"] == "S8" and wr["fail"] == "argmax_other_confident", wr)
    check("its census row counts the failure mode", any(
        r["source"] == wr["source"] and r["label"] == "Ragweed" and r["fail_modes"].get("argmax_other_confident")
        for r in c["rows"]))
    oac = [r["other_argmax_conflicts"] for r in c["rows"] if r["source"] == wr["source"] and r["label"] == "Ragweed"]
    check("... and the OtherPlant-argmax conflict detail (tau_other, second class)", oac and oac[0]
          and oac[0]["n"] >= 1 and oac[0]["tau_other"] is not None and oac[0]["second"], oac)
    dup = rows[P["exact_dup"]["dropped"]]
    check("the exact_dup twin: a pre-pool row, S3, twin_of the kept image", dup["path"]["S3"] == "exact_dup"
          and dup["twin_of"] == P["exact_dup"]["kept_key"] and dup["sole_cause"] == "S3", dup)
    check("H5a finds the planted twin's extra target box", c["h5a"]["dropped_twins_with_target_box_kept_lacks"] ==
          {"images": 1, "boxes": 1} and c["h5a"]["examples"][0]["id"] == P["exact_dup"]["dropped"], c["h5a"])
    for did, (split, ekey) in P["near_eval"].items():
        r = rows.get(did)
        check("near_eval pair %s: S4, split %s, the evaluation image named" % (did.split("|")[0], split),
              r is not None and r["path"]["S4"] == "near_eval" and r["near"]["split"] == split
              and r["near"]["eval"] == ekey and r["near"]["bits"] <= 6, r)
    cp = rows[P["copy"]["id"]]
    check("the copy of a train_core picture: S5, its twin named", cp["path"]["S5"] == "cwd12_copy"
          and cp["near"]["eval"] == P["copy"]["train_core_key"], cp)
    check("an image without boxes: S2 no_boxes", rows[P["no_boxes"]]["path"]["S2"] == "no_boxes")
    check("veto: one lost box, blocked by an OtherPlant box called a target; the contract's numbers are "
          "recorded, not asserted", c["veto"]["lost_boxes"] == 1 and c["veto"]["by_blocker"]["other_called_target"] == 1
          and c["veto"]["contract_check"]["recorded_not_asserted"] is True, c["veto"])

    print("guard pairs")
    with open(fd / "guard_pairs_v1.csv", newline="") as fh:
        gp = {r["pair_id"]: r for r in csv.DictReader(fh)}
    for did, (split, ekey) in P["near_eval"].items():
        pid = "p:%s|%s|%s" % (did[2:], split, ekey)
        check("the near_eval pair %s is listed with both sides' sha256" % split,
              pid in gp and gp[pid]["kind"] == "near_eval" and len(gp[pid]["sha_a"]) == 64
              and len(gp[pid]["sha_b"]) == 64, sorted(gp)[:4])
    dp = "p:%s|dup|%s" % (P["exact_dup"]["dropped"][2:], P["exact_dup"]["kept_key"])
    check("the exact_dup twin is listed; a byte copy's sha256 does not differ",
          dp in gp and gp[dp]["sha_differs"] == "0", gp.get(dp))
    cpid = "p:%s|train_core|%s" % (P["copy"]["id"][2:], P["copy"]["train_core_key"])
    check("the copy pair carries its twin's provenance, and every pair its lab and near-dup group (the G5 "
          "frame's disjointness keys)", cpid in gp and gp[cpid]["provenance"] == "prov:%s" % P["copy"]["train_core_key"]
          and all(r["lab"] and r["near_dup3"].startswith("n:") and r["provenance"].startswith("prov:")
                  for r in gp.values()), gp.get(cpid))
    split0 = sorted(P["near_eval"].values())[0][0]
    mp = C.manifest_path(split0)
    keep_m = mp.read_bytes()
    C.write_manifest(mp, [])
    msg = raises(lambda: A.guard_pairs(TMP / "gp_check.csv", fd / "ledger.jsonl"), AdapterError)
    check("a near_eval pair whose evaluation image no manifest lists refuses (never a pair without its "
          "evaluation side)", msg is not None and "no evaluation manifest" in msg, msg)
    mp.write_bytes(keep_m)
    os.unlink(mp)
    msg = raises(lambda: A.guard_pairs(TMP / "gp_check.csv", fd / "ledger.jsonl"), AdapterError)
    check("a missing evaluation manifest refuses", msg is not None and "missing" in msg, msg)
    mp.write_bytes(keep_m)

    print("name status, known truth and keys")
    ns = read_json(fd / "name_status_v2.json")
    check("name_status_v2.json is frozen and census records its sha256", ns["frozen"] is True
          and c["name_status_v2"]["sha256"] == C.sha256_file(fd / "name_status_v2.json"))
    st = {(n["source"], n["name"]): n["status_v2"] for n in ns["names"]}
    check("the world's names get their v2 statuses (no name, numeric, unresolvable, role, generic, related)",
          st[(FW.TUF, "")] == "no_name" and st[(FW.MH, "12")] == "numeric" and st[(FW.BQDOK, "BroWeed")] == "unresolvable"
          and st[(FW.LEOPARD, "crop")] == "role" and st[(FW.LEOPARD, "weed")] == "generic"
          and st[(FW.GREENHOUSE, "Redroot Pigweed")] == "target_related", st)
    kt = A.known_truth("weed", fd)
    core_n = sum(1 for r in csv.reader(open(w.step1 / "crops.csv")) if r[1] == "core")
    check("KT1 is every train_core crop (%d)" % core_n, len(kt["KT1"]) == core_n)
    copy_items = [it for it in kt["KT2"] if it["id"].startswith("t2:%s#" % P["copy"]["key"])]
    check("KT2 leaves the copy box without a train_core twin out",
          len(copy_items) == P["copy"]["boxes"] - 1
          and "t2:%s#%d" % (P["copy"]["key"], P["copy"]["unmatched_box"]) not in {i["id"] for i in copy_items},
          [i["id"] for i in copy_items])
    check("KT3 is the three_season copy", kt["KT3"] and all(i["source"] == FW.THREE for i in kt["KT3"]))
    n_kt4 = sum(1 for r in rows.values() if r["unit"] == "box" and r["source"] in (FW.WEEDCROP, FW.GREENHOUSE)
                and r["label"] < C.OTHER_PLANT and r["crop_id"] is not None)
    check("KT4 is every embedded target-labelled box of the authoritative sources (%d), claimed" % n_kt4,
          len(kt["KT4"]) == n_kt4 and all(i["claimed"] for i in kt["KT4"]))
    rp = [i for i in kt["KT5"] if i["truth_taxon"] == "Amaranthus retroflexus"]
    check("KT5 holds the Redroot pigweed boxes (card taxon Amaranthus retroflexus), claimed attractors",
          rp and all(i["truth_kind"] == "attractor" and i["claimed"] for i in rp), len(rp))
    check("every KT4 and KT5 item is an embedded crop (a judge's bank and the sheets need its crop)",
          all(i["crop_id"] is not None for k in ("KT4", "KT5") for i in kt[k]),
          [i["id"] for k in ("KT4", "KT5") for i in kt[k] if i["crop_id"] is None])
    check("the small Kochia box (an attractor's name, never embedded) is in no known-truth set",
          sm["kt"] == [] and sm["src_name"] == "Kochia", sm)
    raw_k = json.loads(D.load("weed").path.read_text())
    raw_k["attractors"] = [a for a in raw_k["attractors"] if a["taxon"] != "Bassia scoparia"]
    raw_k["known_truth"]["kt7"]["taxa"] = [t for t in raw_k["known_truth"]["kt7"]["taxa"] if t != "Bassia scoparia"]
    alt_k = TMP / "weed_no_kochia.json"
    alt_k.write_text(json.dumps(raw_k))
    msg = raises(lambda: A.known_truth(str(alt_k), fd), AdapterError)
    check("known truth refuses a census KT5 row whose class no longer resolves to an attractor (no KT5 item "
          "without its truth taxon)", msg is not None and "KT5" in msg, msg)
    check("KT6 is empty before the H3a exact part passes", kt.get("KT6") == [])
    check("census records the known-truth counts", c["known_truth_counts"].get("KT1") == core_n)
    t2 = copy_items[0]["id"]
    twin = "t1:%s#0" % P["copy"]["train_core_key"]
    uk = A.unit_keys([t2, twin], "weed", fd)
    check("unit_keys gives a copy and its twin the same provenance", uk[t2]["provenance"] == uk[twin]["provenance"]
          == "prov:%s" % P["copy"]["train_core_key"], uk)
    check("... and the same near_dup3 group (the copy is within 3 bits)",
          uk[t2]["near_dup3"] == uk[twin]["near_dup3"], uk)
    b = "b:%s#0" % P["vetoed"]["key"]
    uk2 = A.unit_keys([b, "d:%s" % P["copy"]["id"][2:], "c:%s|0" % FW.WEEDCROP], "weed", fd)
    check("pool, pre-pool and class units get source and lab (NDSU for the authoritative source)",
          uk2[b]["lab"] == "NDSU" and uk2[b]["source"] == FW.WEEDCROP and uk2["c:%s|0" % FW.WEEDCROP]["near_dup3"] is None,
          uk2)
    dcopy = "d:%s" % P["copy"]["id"][2:]
    check("a pre-pool reference copy shares its twin's provenance (runner §1.3), as its t2 item does",
          uk2[dcopy]["provenance"] == uk[t2]["provenance"] == "prov:%s" % P["copy"]["train_core_key"], uk2[dcopy])
    check("an unknown unit kind refuses", raises(lambda: A.unit_keys(["zz:1"], "weed", fd), AdapterError))
    raw_s = json.loads(D.load("weed").path.read_text())
    raw_s["sources"]["capture_stem_regex"] = {FW.MH: r"^(mh)_"}
    stem_cfg = TMP / "weed_stems.json"
    stem_cfg.write_text(json.dumps(raw_s))
    mk = P["hidden_target"]["key"]
    uk3 = A.unit_keys(["b:%s#0" % mk, "i:%s" % mk], str(stem_cfg), fd)
    check("with a capture-stem rule, unit_keys reads the pool image's file name, as the census rows do "
          "(one provenance group per capture stem)",
          uk3["b:%s#0" % mk]["provenance"] == uk3["i:%s" % mk]["provenance"] == "prov:%s|mh" % FW.MH, uk3)

    print("census-derived funnel ledger")
    fl = L.load(fd / "funnel_ledger.json")
    check("funnel_ledger.json is census-derived, name status v2, and validates",
          fl["derivation"] == "census" and fl["name_status_version"] == "v2")
    check("its label spaces count v2 kinds (BroWeed is generic, not named)",
          fl["label_spaces"][FW.BQDOK]["kinds"]["generic"] > 0)
    msg = raises(lambda: A.ledger_from_summaries("weed", w.step1, CENSUS_V0, fd / "funnel_ledger.json"),
                 AdapterError)
    check("a summaries ledger never replaces the census-derived one", msg is not None and "census-derived" in msg, msg)

    print("readers")
    check("pool_rows filters by source", {r["source"] for r in A.pool_rows([FW.TUF])} == {FW.TUF})
    check("base_rows is base_B.jsonl", len(A.base_rows()) == len(C.read_manifest(w.step1 / "base_B.jsonl")))
    check("eval_rows holds every evaluation split", set(A.eval_rows()) == set(C.EVAL_SPLITS))
    check("never_train_guard loads the index", A.never_train_guard().n > 0)
    keys = [r["key"] for r in A.pool_rows([FW.WEEDCROP])][:2]
    lr = A.label_rows(keys)
    check("label_rows reads the step1 labels", set(lr) == set(keys) and all(lr[k] for k in keys))
    check("label_rows refuses a key that is not a pool image",
          raises(lambda: A.label_rows(["no_such_key"]), AdapterError))
    exp = TMP / "inc" / "exp_x" / "manifests"
    exp.mkdir(parents=True)
    C.write_manifest(exp / "inc_01.jsonl", A.pool_rows([FW.TUF])[:2])
    check("increment_rows reads an experiment's manifests", list(A.increment_rows("exp_x")) == ["inc_01"])
    proj = A.verifier_fit_info_projection()
    check("the fit-info projection carries the OtherPlant sample's sources and names",
          proj["other_sample"]["taken"] > 0 and proj["other_sample"]["sources"] and proj["source"] == "derived")
    X, _info = A.step1_features()
    P_, cos = A.j1_scores(X[:5])
    check("j1_scores gives 13 probabilities per crop, summing to 1", P_.shape == (5, 13)
          and np.allclose(P_.sum(1), 1.0))
    sl = A.source_labels(FW.MH)
    check("source_labels keeps the source's own class ids and the image size",
          all(len(b) == 7 and b[5] > 0 for v in sl.values() for b in v) and any(b[0] == 12 for v in sl.values() for b in v))

    print("a census that does not reconcile is written with ok false, then refused")
    adp = w.step1 / "admit_summary.json"
    keep_ad = adp.read_bytes()
    ad2 = json.loads(keep_ad)
    ad2["boxes_small_not_embedded"] = int(ad2["boxes_small_not_embedded"]) + 1
    adp.write_text(json.dumps(ad2))
    msg = raises(lambda: A.census(pre, "weed", fd, tax, known_items=known, force=True, testing=True), AdapterError)
    bad = read_json(fd / "census_v1.json")
    check("a count that differs from admit_summary.json refuses, naming the check",
          msg is not None and "does not reconcile" in msg and "small_boxes" in msg, msg)
    check("... after writing census_v1.json with reconciliation.ok false and the failed check",
          bad["reconciliation"]["ok"] is False
          and [x["name"] for x in bad["reconciliation"]["checks"] if not x["ok"]] == ["small_boxes"],
          [x for x in bad["reconciliation"]["checks"] if not x["ok"]])
    adp.write_bytes(keep_ad)
    c = A.census(pre, "weed", fd, tax, known_items=known, force=True, testing=True)
    check("with the file restored the census reconciles again", c["reconciliation"]["ok"])

    print("reruns, refusals and the sample lock")
    before = (fd / "census_v1.json").read_bytes()
    c2 = A.census(pre, "weed", fd, tax, known_items=known, testing=True)
    check("a rerun with the same inputs is a no-op", (fd / "census_v1.json").read_bytes() == before
          and c2["built_utc"] == c["built_utc"])
    msg = raises(lambda: A.census(pre, "weed", fd, tax, testing=True), AdapterError)
    check("other inputs (no known items) refuse without --force", msg is not None and "--force" in msg, msg)
    npz = w.step1 / "pool_verdicts.npz"
    keep = npz.read_bytes()
    with np.load(npz, allow_pickle=False) as dd:
        arrs = {k: dd[k] for k in dd.files}
    arrs["verdict"] = arrs["verdict"].copy()
    arrs["verdict"][0] = (int(arrs["verdict"][0]) + 1) % 4
    np.savez(npz, **arrs)
    msg = raises(lambda: A.census(pre, "weed", fd, tax, known_items=known, force=True, testing=True), AdapterError)
    check("a flipped verdict in pool_verdicts.npz is refused", msg is not None and "disagree" in msg, msg)
    npz.write_bytes(keep)
    check("the census file of the refused run was not replaced", (fd / "census_v1.json").read_bytes() == before)
    raw = json.loads(D.load("weed").path.read_text())
    raw["stages"] = raw["stages"][:-1]
    alt = TMP / "weed_short.json"
    alt.write_text(json.dumps(raw))
    msg = raises(lambda: A.census(pre, str(alt), fd, tax, known_items=known, force=True, testing=True),
                 (AdapterError, D.DomainError))
    check("a config whose stages differ from the adapter's is refused", msg is not None, msg)
    D.append_amendment(pre, {"id": D.next_amendment_id(D.load_prereg(pre)), "kind": "sample_lock",
                             "date": "2026-09-28", "prereg_core_sha256": D.load_prereg(pre).core_sha256,
                             "sample_sha256": "0" * 64})
    c3 = A.census(pre, "weed", fd, tax, known_items=known, testing=True)
    check("after the sample lock, the same inputs are still a no-op", c3["built_utc"] == c["built_utc"])
    msg = raises(lambda: A.census(pre, "weed", fd, tax, force=True, testing=True), SampleLocked)
    check("after the sample lock, other inputs refuse with SampleLocked, even with --force", msg is not None, msg)
    ns_before = (fd / "name_status_v2.json").read_bytes()
    os.unlink(fd / "census_v1.json")
    msg = raises(lambda: A.census(pre, "weed", fd, tax, known_items=known, force=True, testing=True), SampleLocked)
    check("after the sample lock, a census without census_v1.json still refuses and name_status_v2.json (locked) "
          "is not rewritten", msg is not None and (fd / "name_status_v2.json").read_bytes() == ns_before, msg)


if __name__ == "__main__":
    try:
        test_interface()
        test_real_summaries()
        test_census_v0_reproduction()
        test_world()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)
