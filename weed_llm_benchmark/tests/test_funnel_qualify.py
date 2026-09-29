#!/usr/bin/env python3
"""funnel/qualify.py (runner docs/FUNNEL_AUDIT_RUNNER.md §4.15, §5.4.3, §7.2;
contract docs/FUNNEL_AUDIT.md §4.1, §4.3, DEC-1). Mutation points M2, M3, M6.

Disjointness and circularity:
  * shares finds each of the four kinds (source, near_dup3, provenance, lab)
    and nothing else; items of an independent set compare per observation;
  * a judge whose material holds a source of a stratum is not allowed there;
    a material recorded as a hash without its list file is refused;
  * qualifying the labeller for H1 on the set H1 tests (KT4) raises
    CircularityError; a set not claimed by the hypothesis passes;
  * the step-1 probe is refused as a judge on any set, and the zero-shot
    judge on the independent set (never_qualifies) -- M2.
Numbers:
  * Se and Sp on a hand-computed example (a sibling negative, an attractor
    negative, an unsure answer counted wrong);
  * est: Clopper-Pearson under 30, Wilson from 30 (estimate.py), n = 0 gives
    [0, 1];
  * phi on a hand-computed 2x2 table; NaN on an empty margin;
  * the identity check passes at 24 of 30 with no excluded answer and fails
    with one answer of the excluded taxon, and with fewer than n answers.
The labeller (rl):
  * reference-lab strata get scope KT7 alone; strata of another lab get every
    qualify_rl_on set they share nothing with;
  * a backend answering every counted sentinel right qualifies at species;
    KT4 (claimed by H1) sentinels answered wrong change nothing -- they are
    agreement only (M6); an unsure answer counts as wrong;
  * the primary backend per hypothesis is the one with the larger
    Se.lb + Sp.lb; a tie goes to RL-B; demotion to genus when species fails;
  * pair sentinels qualify RL-B for pairs; identity is recorded;
  * a gold row from G1 passed to rl is refused (M3).
Machine judges (judges):
  * on a synthetic known truth with score files: the disjoint kNN judge
    qualifies, the zero-shot judge is never scored on the independent set,
    the step-1 probe never qualifies, correlated judges are grouped, h7
    carries the kind-derived predictions; the file is locked (same inputs:
    no-op; other inputs: refused; after the sample lock: SampleLocked);
  * allowed_judges on judge_qualification.json entries (as strata passes
    them) uses the lab scope that leaves the stratum's lab out, reading the
    hashed lists from judge_material_v1.json; qualified_judges agrees.

Run:  python3 tests/test_funnel_qualify.py
"""
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_labels_fixtures as FX  # noqa: E402

TMP = FX.setup("funnel_qualify_")
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


def gold_row(item_id, unit_id, group, backend, answer, kt, truth, truth_kind, taxon, src, lab, answer_taxon="",
             answer_level="species", pair_truth="", nd="", prov=""):
    return {"item_id": item_id, "unit_id": unit_id, "unit": "box", "group": group, "stratum": "%s/x" % group,
            "labeller": "%s:m" % backend, "backend": backend, "option": "1", "answer": answer,
            "answer_level": answer_level, "answer_taxon": answer_taxon, "box_ok": "yes",
            "is_sentinel": "1" if group in ("sentinel", "pair_sentinel") else "0", "kt": kt,
            "truth": "" if truth is None else str(truth), "truth_kind": truth_kind, "truth_taxon": taxon or "",
            "pair_truth": pair_truth, "source": src, "lab": lab, "near_dup3": nd or "n:%s" % unit_id,
            "provenance": prov or "prov:%s" % unit_id}


def main():
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import qualify as Q
    from weed_optimizer_framework.tools.funnel import (CircularityError, QualifyError, SampleLocked,
                                                       write_csv_atomic, write_json_atomic)
    fd = pathlib.Path(os.environ["INC_DIR"]) / "funnel"
    domain = D.load("weed")
    prereg = D.load_prereg(fd / "prereg_v1.json")
    PAL, WH, RAG = domain.class_id("PalmerAmaranth"), domain.class_id("Waterhemp"), domain.class_id("Ragweed")

    # ---------------------------------------------------------- disjointness
    a = {"source": "s1", "near_dup3": "n:1", "provenance": "p:1", "lab": "L1"}
    for kind, val in (("source", "s1"), ("near_dup3", "n:1"), ("provenance", "p:1"), ("lab", "L1")):
        b = {"source": "s2", "near_dup3": "n:2", "provenance": "p:2", "lab": "L2"}
        b[kind] = val
        sh = Q.shares(a, b)
        check("shares finds %s only" % kind, sh[kind] == [val] and all(not v for k, v in sh.items() if k != kind), sh)
    check("disjoint keys share nothing", Q.disjoint(a, {"source": "s2", "near_dup3": "n:2", "provenance": "p:2",
                                                         "lab": "L2"}))
    check("empty values never share", Q.disjoint({"source": "", "lab": None}, {"source": "", "lab": None}))
    x1 = Q.item_keys({"id": "t7:10/1", "source": "kt7", "lab": "src:kt7"}, independent=True)
    x2 = Q.item_keys({"id": "t7:10/2", "source": "kt7", "lab": "src:kt7"}, independent=True)
    x3 = Q.item_keys({"id": "t7:11/1", "source": "kt7", "lab": "src:kt7"}, independent=True)
    check("independent items: one observation shares, two observations do not",
          not Q.disjoint(x1, x2) and Q.disjoint(x1, x3))
    mat = {"j_bank_src": {"sources": ["s1"], "labs": ["L9"], "near_dup3": [], "provenance": []},
           "j_other": {"sources": ["s3"], "labs": ["L3"], "near_dup3": [], "provenance": []}}
    check("a judge whose material holds the stratum's source is not allowed",
          Q.allowed_judges({"source": ["s1"], "lab": ["L1"], "near_dup3": [], "provenance": []}, mat) == ["j_other"])
    check("a hashed material without its list file is refused",
          raises(lambda: Q.allowed_judges(a, {"j": {"sources": [], "labs": [], "near_dup3": "ab" * 32,
                                                    "provenance": "cd" * 32}}), QualifyError))
    lu_keys = {"source": [FX.LU], "lab": [domain.lab_of(FX.LU)], "near_dup3": ["n:x"], "provenance": ["p:x"]}

    # ---------------------------------------------------------- circularity (M2)
    check("qualifying for H1 on KT4 raises CircularityError",
          raises(lambda: Q.check_circularity("H1", ["KT1", "KT4"], domain), CircularityError))
    check("qualifying for H1 on KT1, KT2, KT7 passes", Q.check_circularity("H1", ["KT1", "KT2", "KT7"], domain) is None)
    check("qualifying for H2b on KT5 raises", raises(lambda: Q.check_circularity("H2b", ["KT5"], domain),
                                                      CircularityError))
    check("M2: the zero-shot judge on the independent set raises",
          raises(lambda: Q.judge_may_use("J-zs", "zero_shot", "KT7", domain), CircularityError))
    check("M2: the step-1 probe raises on any set",
          raises(lambda: Q.judge_may_use("J1", "step1_probe", "KT2", domain), CircularityError))
    check("the disjoint kNN judge may use the independent set",
          Q.judge_may_use("J-knn2", "knn", "KT7", domain) is None)

    # ---------------------------------------------------------- numbers
    conf = Q.attractor_confusions(domain)
    items = [({"truth": PAL, "truth_kind": "target", "truth_taxon": "Amaranthus palmeri"}, "PalmerAmaranth"),
             ({"truth": PAL, "truth_kind": "target", "truth_taxon": "Amaranthus palmeri"}, "Waterhemp"),
             ({"truth": WH, "truth_kind": "target", "truth_taxon": "Amaranthus tuberculatus"}, "Waterhemp"),
             ({"truth": None, "truth_kind": "attractor", "truth_taxon": "Amaranthus retroflexus"}, "other"),
             ({"truth": None, "truth_kind": "attractor", "truth_taxon": "Amaranthus retroflexus"}, "PalmerAmaranth"),
             ({"truth": RAG, "truth_kind": "target", "truth_taxon": "Ambrosia artemisiifolia"}, None)]
    pairs = [t for it, ans in items for t in Q.trials(it, ans, "species", domain, conf)]
    se, sp = Q.se_sp(pairs)
    check("Se = 2/4 (an unsure answer is wrong)", (se["k"], se["n"]) == (2, 4), se)
    check("Sp = 3/5 (two sibling negatives, two attractor negatives, one of each wrong... )",
          (sp["k"], sp["n"]) == (3, 5), sp)
    check("est under 30 is Clopper-Pearson, from 30 Wilson",
          se["method"] == "clopper_pearson" and Q.est(29, 30)["method"] == "wilson")
    check("est(0, 0) is [0, 1]", Q.est(0, 0)["lb"] == 0.0 and Q.est(0, 0)["ub"] == 1.0)
    e24 = Q.est(40, 40)
    check("40 of 40 has a lower bound above 0.85 (a qualifiable sample)", e24["lb"] > 0.85, e24)
    ph = Q.phi([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0])
    check("phi on a hand-computed table (n11=2 n10=1 n01=1 n00=4): 7/15",
          abs(ph - (2 * 4 - 1 * 1) / 15.0) < 1e-12, ph)
    check("phi is NaN on an empty margin", Q.phi([0, 0, 0], [1, 0, 1]) != Q.phi([0, 0, 0], [1, 0, 1]))
    ic = domain.raw["identity_checks"][0]
    rows = [{"answer": ic["class"], "answer_taxon": ""}] * 24 + [{"answer": "other", "answer_taxon": ""}] * 6
    check("identity check passes at 24 of 30 with no excluded answer", Q.identity_check(rows, ic)["pass"])
    trif = [{"answer": "other", "answer_taxon": ic["excluded_taxa"][0]}]
    check("identity check fails with one excluded-taxon answer",
          not Q.identity_check(rows[:29] + trif, ic)["pass"])
    check("identity check fails with fewer than n answers", not Q.identity_check(rows[:25], ic)["pass"])

    # ---------------------------------------------------------- labeller (rl)
    # frames: an NDSU G2 stratum, a reference-lab G2 stratum, a G1 no-information stratum, G0
    from weed_optimizer_framework.tools.funnel import strata as ST
    frames = {"G2": [("b:nd_%d#0" % i, "G2/source=%s/label=Ragweed/fail=p_below_tau" % FX.ND, FX.ND) for i in range(3)]
              + [("b:lu_%d#0" % i, "G2/source=%s/label=Waterhemp/fail=argmax_wrong" % FX.LU, FX.LU) for i in range(3)],
              "G1": [("b:an_%d#0" % i, "G1/frame=noinfo/status=no_name/pred=Waterhemp", FX.AN) for i in range(3)]}
    groups = {}
    for g, rws in frames.items():
        out = [{"unit_id": u, "unit": "box", "stratum": s, "source": src, "image_key": u, "crop_id": "",
                "lab": domain.lab_of(src), "near_dup3": "n:%s" % u, "provenance": "prov:%s" % u, "label": "",
                "pred": "", "score": "", "allowed_judges": "", "extra": "{}"} for u, s, src in rws]
        p = fd / "frames_v1" / ("%s.csv" % g)
        sha = write_csv_atomic(p, ST.FRAME_FIELDS, out)
        groups[g] = {"file": {"path": str(p), "sha256": sha}, "N": len(out)}
    write_json_atomic(fd / "frames_v1.json", {"format": "funnel-frames/1", "groups": groups})
    grows = []
    n = [0]

    def add(backend, group, kt, truth, kind, taxon, answer, src, lab, answer_taxon="", unit=None, pair=""):
        n[0] += 1
        uid = unit or "u%d" % n[0]
        grows.append(gold_row("i%d" % n[0], uid, group, backend, answer, kt, truth, kind, taxon, src, lab,
                              answer_taxon=answer_taxon, pair_truth=pair))
    for be in ("RL-A", "RL-B"):
        for i in range(20):
            add(be, "sentinel", "KT1", PAL if i % 2 else WH, "target", "", "PalmerAmaranth" if i % 2 else "Waterhemp",
                FX.REF, domain.lab_of(FX.REF), unit="t1:core%d#0" % i)
        for i in range(10):
            add(be, "sentinel", "KT2", PAL, "target", "", "PalmerAmaranth", FX.LU if i == 0 else FX.COPY_SRC,
                domain.lab_of(FX.LU if i == 0 else FX.COPY_SRC), unit="t2:copy%d#0" % i)
        for i in range(20):
            add(be, "sentinel", "KT7", WH, "target", "Amaranthus tuberculatus", "Waterhemp", "kt7", "src:kt7",
                unit="t7:%d/1" % (100 + i))
        for i in range(20):
            add(be, "sentinel", "KT7", None, "attractor", "Amaranthus retroflexus", "other", "kt7", "src:kt7",
                answer_taxon="Amaranthus retroflexus", unit="t7:%d/1" % (200 + i))
        for i in range(10):
            add(be, "sentinel", "KT4", RAG, "target", "", "other", FX.ND, domain.lab_of(FX.ND),
                unit="b:kt4_%d#0" % i)
        for i in range(24):
            add(be, "identity", "KT1", RAG, "target", "", "Ragweed", FX.REF, domain.lab_of(FX.REF),
                unit="t1:rag%d#0" % i)
        for i in range(6):
            add(be, "identity", "KT1", RAG, "target", "", "other", FX.REF, domain.lab_of(FX.REF),
                unit="t1:ragx%d#0" % i)
        for i in range(40):
            add(be, "sentinel", "KT7", None, "attractor", "Bassia scoparia", "other", "kt7", "src:kt7",
                answer_taxon="Bassia scoparia", unit="t7:%d/1" % (400 + i))
    for i in range(80):
        add("RL-B", "pair_sentinel", "KT2", None, None, "", "same" if i < 40 else "different", FX.COPY_SRC,
            domain.lab_of(FX.COPY_SRC), unit="p:%s|c%d|%s|k%d" % (FX.COPY_SRC, i, FX.REF, i),
            pair="same" if i < 40 else "different")
    doc = Q.rl(prereg, domain, fd, rows=grows)
    sc = doc["strata_scopes"]
    nd_s = "G2/source=%s/label=Ragweed/fail=p_below_tau" % FX.ND
    lu_s = "G2/source=%s/label=Waterhemp/fail=argmax_wrong" % FX.LU
    check("reference-lab strata get scope KT7 alone", sc[lu_s] == "KT7", sc)
    check("an NDSU stratum is qualified on KT1, KT2 and KT7", sc[nd_s] == "KT1+KT2+KT7", sc)
    blk = doc["backends"]["RL-B"]["KT1+KT2+KT7"]["species"]
    check("all counted sentinels right: qualified at species", blk["qualified"] and blk["se"]["k"] == blk["se"]["n"])
    check("M6: KT4 sentinels answered wrong are not counted (Se n = 20 KT1 + 10 KT2 + 20 KT7)",
          blk["se"]["n"] == 50, blk["se"])
    check("KT4 is reported as agreement only", doc["agreement_only"]["RL-B"]["KT4"]["k"] == 0
          and doc["agreement_only"]["RL-B"]["KT4"]["n"] == 10)
    check("identity recorded per class (24 of 30, pass)", doc["identity"]["Ragweed"]["pass"]
          and doc["identity"]["Ragweed"]["n"] == 60)
    check("pairs qualify RL-B", doc["pairs"]["RL-B"]["qualified"])
    check("a tie goes to RL-B", doc["primary"]["H1"]["backend"] == "RL-B" and doc["primary"]["H1"]["level"] == "species",
          doc["primary"]["H1"])
    check("H1's scope is the NDSU strata's", doc["primary"]["H1"]["scope"] == "KT1+KT2+KT7", doc["primary"]["H1"])
    # RL-A worse on KT7 attractors -> RL-B primary by Se.lb + Sp.lb; then species failing -> genus
    worse = [dict(r, answer="PalmerAmaranth") if (r["backend"] == "RL-A" and r["truth_kind"] == "attractor"
                                                  and r["unit_id"] < "t7:204") else r for r in grows]
    d2 = Q.rl(prereg, domain, fd, rows=worse, force=True)
    check("the backend with the larger Se.lb + Sp.lb is primary", d2["primary"]["H1"]["backend"] == "RL-B")
    unsure = [dict(r, answer="unsure") if (r["kt"] == "KT7" and r["truth_kind"] == "target"
                                           and r["unit_id"] < "t7:110/") else r for r in grows]
    d3 = Q.rl(prereg, domain, fd, rows=unsure, force=True)
    b3 = d3["backends"]["RL-B"]["KT7"]["species"]
    check("unsure answers count as wrong", b3["se"]["k"] == b3["se"]["n"] - 10, b3["se"])
    genus = [dict(r, answer="Waterhemp") if (r["truth"] == str(PAL) and r["group"] == "sentinel") else r for r in grows]
    d4 = Q.rl(prereg, domain, fd, rows=genus, force=True)
    p4 = d4["primary"]["H1"]
    check("species fails, genus qualifies: the primary is demoted to genus",
          p4["level"] == "genus" and p4["demoted"] and not p4["qualified"], p4)
    bad = grows + [gold_row("g1x", "b:g1#0", "G1", "RL-B", "Waterhemp", "", WH, "target", "", FX.AN, "src:x")]
    check("M3: a gold row from G1 passed to rl is refused", raises(lambda: Q.rl(prereg, domain, fd, rows=bad,
                                                                               force=True), QualifyError))

    # a hypothesis whose strata exclude different sets: its scope (KT7) is no single stratum's scope
    fd2 = TMP / "inc" / "funnel_union"
    rws = [("b:ua_%d#0" % i, "G1/frame=noinfo/status=no_name/pred=Waterhemp", FX.AN, "n:t1:core0#0")
           for i in range(3)]
    rws += [("b:ub_%d#0" % i, "G1/frame=noinfo/status=numeric/pred=Waterhemp", FX.AN, "n:t2:copy1#0")
            for i in range(3)]
    out = [{"unit_id": u, "unit": "box", "stratum": s, "source": src, "image_key": u, "crop_id": "",
            "lab": domain.lab_of(src), "near_dup3": nd, "provenance": "prov:%s" % u, "label": "", "pred": "",
            "score": "", "allowed_judges": "", "extra": "{}"} for u, s, src, nd in rws]
    sha = write_csv_atomic(fd2 / "frames_v1" / "G1.csv", ST.FRAME_FIELDS, out)
    write_json_atomic(fd2 / "frames_v1.json", {"format": "funnel-frames/1", "groups": {
        "G1": {"file": {"path": str(fd2 / "frames_v1" / "G1.csv"), "sha256": sha}, "N": len(out)}}})
    du = Q.rl(prereg, domain, fd2, rows=grows)
    check("the two strata get different scopes (KT2+KT7, KT1+KT7)",
          sorted(du["strata_scopes"].values()) == ["KT1+KT7", "KT2+KT7"], du["strata_scopes"])
    check("a hypothesis spanning them is qualified in the scope they all allow (KT7), not left without a "
          "labeller", du["primary"]["H2a"]["scope"] == "KT7" and du["primary"]["H2a"]["backend"] == "RL-B"
          and "KT7" in du["backends"]["RL-B"], du["primary"]["H2a"])

    # per genus: species qualifies pooled, but not for the genus whose two species it confuses
    PUR = domain.class_id("Purslane")
    grow2, m = [], [0]

    def add2(kt, truth, kind, taxon, answer, answer_taxon=""):
        m[0] += 1
        grow2.append(gold_row("q%d" % m[0], "t7:%d/1" % (5000 + m[0]), "sentinel", "RL-B", answer, kt, truth, kind,
                              taxon, "kt7", "src:kt7", answer_taxon=answer_taxon))
    for i in range(100):
        add2("KT7", PUR, "target", "Portulaca oleracea", "Purslane")
    for i in range(40):
        add2("KT7", WH, "target", "Amaranthus tuberculatus", "PalmerAmaranth" if i < 8 else "Waterhemp")
    for i in range(40):
        add2("KT7", None, "attractor", "Bassia scoparia", "other", "Bassia scoparia")
    for i in range(100):
        add2("KT7", None, "attractor", "Erigeron canadensis", "other", "Erigeron canadensis")
    dg = Q.rl(prereg, domain, fd2, rows=grow2, force=True)
    pooled = dg["backends"]["RL-B"]["KT7"]["species"]
    amar = dg["by_genus"]["RL-B"]["KT7"]["Amaranthus"]["species"]
    check("per genus: pooled species qualifies, Amaranthus (8 of 40 swapped) does not",
          pooled["qualified"] and not amar["qualified"] and (amar["se"]["k"], amar["se"]["n"]) == (32, 40),
          (pooled["qualified"], amar))
    check("the primary names the genera its backend does not qualify for at species level",
          "Amaranthus" in (dg["primary"]["H2a"].get("genera_not_qualified_at_species") or []), dg["primary"]["H2a"])

    # J-vlm is not qualified for a type without attractor negatives of that type's set
    write_json_atomic(fd2 / Q.JUDGE_FILE, {"j1_errors": {"items": [r["unit_id"] for r in grows
                                                                    if r["kt"] == "KT7" and r["truth_kind"] == "target"
                                                                    and r["backend"] == "RL-B"]}})
    dv = Q.rl(prereg, domain, fd2, rows=grows, force=True)
    vb = dv["j_vlm"]["J-vlm"]["by_type"]
    check("J-vlm: other_named (independent attractor negatives) is scored; other_noinfo, with no claimed-set "
          "attractor sentinel answered, never qualifies",
          vb["other_named"]["n_attractor_negatives"] == 60 and vb["other_noinfo"]["n_attractor_negatives"] == 0
          and not vb["other_noinfo"]["qualified"] and vb["other_noinfo"]["se"]["n"] > 0, vb["other_noinfo"])

    # ---------------------------------------------------------- machine judges
    missing = FX.have("numpy")
    if missing:
        print("SKIP: numpy not installed (machine-judge qualification)")
        SKIPS.append("judges")
        return
    import numpy as np
    known = {k: [] for k in domain.kt_ids()}

    def it(uid, kt, crop_id, crop_set, truth, kind, taxon, src, sess=None, role=None):
        return {"id": uid, "kt": kt, "crop_id": crop_id, "crop_set": crop_set, "truth": truth, "truth_taxon": taxon,
                "truth_kind": kind, "claimed": kt in ("KT4", "KT5", "KT6"), "source": src,
                "lab": domain.lab_of(src), "near_dup3": "n:%s" % uid, "provenance": "prov:%s" % uid,
                "session": sess, "role": role}
    cid = 0
    j1_pred = {}
    for i in range(40):
        t = PAL if i % 2 else WH
        known["KT1"].append(it("t1:c%d#0" % i, "KT1", cid, "core", t, "target", "", FX.REF, sess="s%d" % (i % 3)))
        j1_pred[cid] = t
        cid += 1
    for i in range(30):
        known["KT2"].append(it("t2:p%d#0" % i, "KT2", cid, "copy", PAL, "target", "", FX.COPY_SRC))
        j1_pred[cid] = WH if i < 12 else PAL          # 12 step-1 probe errors to rescue
        cid += 1
    for i in range(6):                                # KT3: probe errors that are not rescue material (§4.3)
        known["KT3"].append(it("t2:k3_%d#0" % i, "KT3", cid, "copy", PAL, "target", "",
                               "project_agml__three_season_weed_detection"))
        j1_pred[cid] = WH
        cid += 1
    ledger = []
    for i in range(40):
        u = "b:k5_%d#0" % i
        known["KT5"].append(it(u, "KT5", cid, "pool", domain.other["id"], "attractor", "Amaranthus retroflexus", FX.ND2))
        ledger.append({"id": u, "unit": "box", "pred": PAL if i < 10 else domain.other["id"]})
        cid += 1
    for i in range(40):
        u = "t7:%d/1" % (300 + i)
        if i < 20:
            known["KT7"].append(it(u, "KT7", i, "kt7", WH, "target", "Amaranthus tuberculatus", "kt7", role="sentinel"))
        else:
            known["KT7"].append(it(u, "KT7", i, "kt7", domain.other["id"], "attractor", "Amaranthus retroflexus", "kt7",
                                   role="sentinel"))
    with open(fd / "ledger.jsonl", "w") as fh:
        for r in ledger:
            fh.write(json.dumps(r) + "\n")
    X = np.zeros((cid, domain.other["id"] + 1), dtype=np.float32)
    for c, t in j1_pred.items():
        X[c, t] = 1.0

    def probe(Xn):
        return np.asarray(Xn, dtype=np.float64), np.zeros_like(Xn)
    labels_t = list(domain.target_names)
    labels_o = labels_t + ["other"]
    labels_z = labels_o + ["non_object"]
    (fd / "judges").mkdir(parents=True, exist_ok=True)

    def save(judge, set_name, labels, calls):
        idx = np.array(sorted(calls), dtype=np.int64)
        top = np.array([labels.index(calls[i]) if calls[i] in labels else 0 for i in idx], dtype=np.int16)
        P = np.zeros((len(idx), len(labels)), dtype=np.float16)
        P[np.arange(len(idx)), top] = 1
        np.savez(fd / "judges" / ("%s__%s.npz" % (judge, set_name)), unit_index=idx, P=P, top=top,
                 meta=np.array(json.dumps({"judge": judge, "labels": labels, "set": set_name})))

    def truth_name(t):
        return domain.class_name(t["truth"]) if t["truth_kind"] == "target" else "other"
    crops_items = known["KT1"] + known["KT2"] + known["KT3"] + known["KT5"]
    shared_err = {t["crop_id"] for t in known["KT5"][:3]}      # both J-knn2 and J-zs err here

    def call(t):
        return "PalmerAmaranth" if t["crop_id"] in shared_err else truth_name(t)
    save("J-knn2", "crops", labels_o, {t["crop_id"]: call(t) for t in crops_items})
    save("J-knn2", "kt7", labels_o, {t["crop_id"]: truth_name(t) for t in known["KT7"]})
    save("J-knn1", "crops", labels_t, {t["crop_id"]: truth_name(t) if t["truth_kind"] == "target" else "PalmerAmaranth"
                                        for t in crops_items})
    save("J-knn1", "kt7", labels_t, {t["crop_id"]: "Waterhemp" for t in known["KT7"]})
    save("J-zs", "crops", labels_z, {t["crop_id"]: call(t) for t in crops_items})
    save("J-zs", "kt7", labels_z, {t["crop_id"]: truth_name(t) for t in known["KT7"]})
    save("J1", "kt7", labels_o, {t["crop_id"]: ("PalmerAmaranth" if t["truth_kind"] == "attractor" and t["crop_id"] < 25
                                                else truth_name(t)) for t in known["KT7"]})
    adapter = FX.FakeAdapter(known, {}, features=X, probe=probe)
    jq = Q.judges(prereg, domain, fd, adapter)
    J = jq["judges"]
    check("the disjoint kNN judge qualifies for every stratum type",
          all(J["J-knn2"]["by_type"][t]["qualified"] for t in Q.TYPES), {t: J["J-knn2"]["by_type"][t]["why"]
                                                                          for t in Q.TYPES})
    check("the zero-shot judge is never scored on the independent set",
          "KT7" in J["J-zs"]["never_qualifies_on"] and not J["J-zs"]["by_type"]["other_named"]["qualified"]
          and "KT7" not in J["J-zs"]["by_type"]["shifted_target"]["sets"]["positives"])
    check("the step-1 probe never qualifies", not any(b["qualified"] for b in J["J1"]["by_type"].values()))
    check("the reference-only kNN judge fails (it never says 'other')",
          not J["J-knn1"]["by_type"]["other_named"]["qualified"])
    check("rescue sets are the copy set and the independent set (KT2, KT7), not KT3",
          Q.rescue_set_ids(domain) == ["KT2", "KT7"], Q.rescue_set_ids(domain))
    check("rescue is measured on the probe's errors in KT2 and KT7 only (KT3's 6 errors are left out)",
          J["J-knn2"]["by_type"]["shifted_target"]["rescue"]["n"] == 12 + 5
          and J["J-knn2"]["by_type"]["shifted_target"]["rescue"]["k"] == 17, J["J-knn2"]["by_type"]["shifted_target"]["rescue"])
    ag = J["J-knn2"]["agreement_only"]
    check("agreement with the claimed labels is reported apart (KT5: 37 of 40)",
          (ag["KT5"]["k"], ag["KT5"]["n"]) == (37, 40) and ag["KT4"]["n"] == 0, ag)
    check("the step-1 probe's material is flagged: it judges nothing",
          J["J1"]["calibration_material"].get(Q.NEVER_JUDGES) is True)
    mats0 = Q.judge_materials(fd)
    via_lists = {j: Q.material_for(j, nd_probe, jq, mats0)[1] for j in jq["judges"]
                 for nd_probe in [{"source": [FX.ND], "lab": [domain.lab_of(FX.ND)], "near_dup3": ["n:q"],
                                   "provenance": ["p:q"]}]}
    check("allowed_judges never lists the step-1 probe (entries and material lists alike)",
          "J1" not in Q.allowed_judges({"source": [FX.ND], "lab": [domain.lab_of(FX.ND)], "near_dup3": ["n:q"],
                                        "provenance": ["p:q"]}, J)
          and "J1" not in Q.allowed_judges({"source": [FX.ND], "lab": [domain.lab_of(FX.ND)], "near_dup3": ["n:q"],
                                            "provenance": ["p:q"]}, via_lists))
    check("h7 predictions from the judges' kinds",
          jq["h7"]["J-zs"]["predicted"] == "fail" and jq["h7"]["J-knn1"]["predicted"] == "fail"
          and jq["h7"]["J-knn2"]["predicted"] == "qualify" and jq["h7"]["J-knn2"]["qualified"])
    check("J-zs and J-knn2 answer alike: one correlated group",
          any({"J-knn2", "J-zs"} <= set(g) for g in jq["correlated_groups"]), jq["phi"])
    check("the probe's errors are listed for the labeller's rescue", jq["j1_errors"]["n"] == 17)
    cm = J["J-knn2"]["calibration_material"]
    check("material: hashed near_dup3/provenance with a list file", isinstance(cm["near_dup3"], str)
          and cm["lists"]["path"].endswith(Q.MATERIAL_FILE))
    nd_keys = {"source": [FX.ND], "lab": [domain.lab_of(FX.ND)], "near_dup3": ["n:nd"], "provenance": ["p:nd"]}
    allowed_nd = Q.allowed_judges(nd_keys, J)
    check("allowed_judges on judge entries (as strata passes them): the NDSU-free scope admits J-knn2 on an "
          "NDSU stratum", "J-knn2" in allowed_nd and "not:NDSU" in J["J-knn2"]["by_lab_scope"], allowed_nd)
    allowed_lu = Q.allowed_judges(lu_keys, J)
    check("the reference-only kNN judge is never allowed on a reference-lab stratum (its bank)",
          "J-knn1" not in allowed_lu, allowed_lu)
    mats = Q.judge_materials(fd)
    qj = Q.qualified_judges("other_named", nd_keys, jq, mats)
    check("qualified_judges: J-knn2 in its NDSU-free scope (independent-set negatives)",
          ("J-knn2", "not:NDSU") in qj, qj)
    qn = Q.qualified_judges("other_noinfo", nd_keys, jq, mats)
    why = J["J-knn2"]["by_lab_scope"]["not:NDSU"]["by_type"]["other_noinfo"]["why"]
    check("with its only claimed-set negatives inside NDSU, the NDSU-free scope does not qualify other_noinfo",
          qn == [] and "no attractor negatives" in why, (qn, why))
    again = Q.judges(prereg, domain, fd, adapter)
    check("a rerun on the same inputs is a no-op", again == jq)
    save("J-knn2", "kt7", labels_o, {t["crop_id"]: "other" for t in known["KT7"]})
    check("other inputs are refused without --force", raises(lambda: Q.judges(prereg, domain, fd, adapter),
                                                           QualifyError))
    D.append_amendment(fd / "prereg_v1.json", {"id": D.next_amendment_id(prereg), "kind": "sample_lock",
                                               "date": "2026-09-28",
                                               "prereg_core_sha256": prereg.core_sha256, "sample_sha256": "0" * 64,
                                               "key_sha256": "0" * 64})
    locked = D.load_prereg(fd / "prereg_v1.json")
    check("after the sample lock the file is never rewritten",
          raises(lambda: Q.judges(locked, domain, fd, adapter, force=True), SampleLocked))
    import copy as _copy
    from weed_optimizer_framework.tools.funnel import StaleInput, file_record

    def with_frames_lock(sha):
        p2 = _copy.copy(locked)
        p2.raw = dict(locked.raw, amendments=[dict(a, frames_sha256=sha) for a in locked.raw["amendments"]])
        return p2
    check("rl refuses frames that are not the ones the sample lock records",
          raises(lambda: Q.rl(with_frames_lock("0" * 64), domain, fd, rows=grows, force=True), StaleInput))
    check("... and accepts the locked frames",
          Q.rl(with_frames_lock(file_record(fd / "frames_v1.json")["sha256"]), domain, fd, rows=grows,
               force=True)["primary"]["H1"]["backend"] == "RL-B")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "-"))
    sys.exit(1 if FAILURES else 0)
