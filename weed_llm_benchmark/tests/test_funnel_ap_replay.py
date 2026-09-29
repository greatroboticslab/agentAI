#!/usr/bin/env python3
"""The funnel audit's replay cases (docs/FUNNEL_AUDIT.md 8.9; runner 5.5.10).

Cases (every number about real data is read here from the pinned fixtures in
tests/fixtures/inc_replay/funnel/, never typed):
  R9     today's evidence (realloop_v1, the four Step 1 summaries, the ledger
         the Step 1 adapter derives from them, the claims C1 and C2): D17 fires
         and cites S1 = 11,488/2,049, S2 = 220,104/545,318 (name status v1),
         S3 Ragweed 27/3,029, S4, S5 (the evidence stage: 5 of 38 sources) and
         S7 (/steps/2/truth/verdict "helps"). L10 census is ranked first with
         its exact command; L12 and L11 (with L11a, the card fetch L11 reads)
         are proposed for the uninformative sources; no L13; the devil's-
         advocate pass is staged (OP_DA); card X11 is raised. R5, R6 and R8's
         own diagnoses are unchanged when the funnel ledger joins their
         evidence (apart from the rules version), and R9 holds under the
         test-blindness perturbation.
  R9_early  the Step 1 files without realloop_v1 (and no claim), next to
         pilot_v3 and R8's calibration-failed relevance: D19 fires and L10 is
         proposed before D4's L2; realloop_v1's GPU-hours are reported.
  R9b    claim C2 with only what existed on 2026-08-25 (the S1 gate's verdict,
         a reconstructed ledger): D17 and D19 fire and ask for an out-of-domain
         known-truth set.
  R10    an audited negative: D17 and D18 silent; the campaign moves the
         challenged C1 to tested_survives (funnel.claims, actor autopilot).
  R11    an audited positive: D18 fires; L13 covers only the Ragweed stratum
         (--policy R-A); the greenhouse OtherPlant -> Palmer stratum, whose
         source taxon is a relative, is a known confusion; the class map the
         card contradicts goes to L14; X11. An audit row on a guard or
         non-recoverable stage is refused, never L13; alone it escalates
         and moves no claim.
  R12    an invalid audit: D18 escalates calibration_overlap; D17 still fires;
         validate refuses any item that cites the audit.
  R13    the vehicles domain: the same code and thresholds fire D17 and D19;
         its exam name is refused by evidence, remote and validate (and the
         weed domain's lists do not know it); the weed domain's extra
         non-decision split (H10d domain dev) is blocked by evidence, remote,
         brain_plan and validate (mutation M10); the domain-free grep test
         runs.
  R14    the devil's advocate: the positive reply keeps 6 counter-arguments
         and 2 concessions and moves C1 to challenged; the sycophantic one
         keeps 0 and C1 stays challenged; a fabricated cite and a test leak are
         dropped, an echo is kept but not counted; a same-family reply moves
         no claim; a digest holding a blind marker is refused.
  R15    the KT7 incident of 2026-09-28 (the platform's own run): embed-judges
         embedded every crop and refused without funnel/kt7/crops_kt7.csv;
         the platform had never proposed the KT7 fetch, and the failed job
         was never run again. Now embed-judges (and qualify, draw, sheets,
         estimate) waits for the KT7 table, L11a fetches what the waiting step
         names (cards first, then kt7; known-items needs --sources and stays
         with a person), a job that ended FAILED lets the step run again once
         the file is on the cluster, and the third failed run stays with a
         person with the job's state and log path.
  funnel_negative_controls  pilot_v1-v3, b0_v1 and base_b_v1 with the funnel
         ledger and no claim: D17 is silent (no conclusion); a high-yield Step
         1 with out-of-domain known truth: D19 is silent.

Honest caveat: R9 reproduces the evidence the D17 design was written against
(contract 8.4, post hoc); only the prospective DA record (H11) and the first
non-weed campaign test generalisation.

Run:  python3 tests/test_funnel_ap_replay.py
"""
import copy
import hashlib
import json
import os
import pathlib
import posixpath
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_ap_replay_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.inc_autopilot import brain_plan as BP  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers as LV  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import validate as V  # noqa: E402

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
FIX = TESTS / "fixtures" / "inc_replay"
FF = FIX / "funnel"
FAILURES, SKIPS = [], []
FUNNEL_IDS = ("D17", "D18", "D19")
V1_ERA = ["pilot_v1", "b0_v1"]
WHOLE = ["pilot_v1", "pilot_v2", "b0_v1", "base_b_v1"]
WHOLE_V3 = WHOLE + ["pilot_v3"]
ADV, PLANNER = "vllm:glm-4.7-flash", "ollama:qwen3.8:27b"


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def skip(name, why):
    print("  skip %s: %s" % (name, why))
    SKIPS.append(name)


def jload(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def run(ev):
    diags = DG.detect(ev)
    return diags, DG.by_id(diags), LV.propose(diags, ev)


def cite_values(d):
    return [c.get("value") for c in d.get("cites") or []]


def has_cite(d, artifact, pointer, value):
    return any(c.get("artifact") == artifact and c.get("pointer") == pointer and c.get("value") == value
               for c in d.get("cites") or [])


def funnel_world(name, ledger="funnel_ledger_summaries.json", with_loop=True, extra=None):
    """A tree laid out like INC_DIR under TMP/<name>: realloop_v1 (unless
    with_loop is False), the four Step 1 summaries and the funnel ledger
    (under the name evidence.load_dir reads), plus `extra` {dest: fixture}."""
    root = TMP / name
    if root.exists():
        shutil.rmtree(str(root))
    (root / "step1").mkdir(parents=True)
    (root / "funnel").mkdir(parents=True)
    for f in ("admit_summary.json", "select_summary.json", "pool_summary.json", "calibration.json"):
        shutil.copyfile(FF / "step1" / f, root / "step1" / f)
    if with_loop:
        (root / "realloop_v1").mkdir()
        for f in ("exp.json", "report.json", "ledger.jsonl", "build_summary.json"):
            shutil.copyfile(FF / "realloop_v1" / f, root / "realloop_v1" / f)
    if ledger:
        shutil.copyfile(FF / ledger, root / "funnel" / "funnel_ledger.json")
    for dest, src in (extra or {}).items():
        (root / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FF / src, root / dest)
    return root


def real_numbers():
    """S1-S7's expected values, read from the pinned copies."""
    admit = jload(FF / "step1" / "admit_summary.json")
    pool = jload(FF / "step1" / "pool_summary.json")
    sel = jload(FF / "step1" / "select_summary.json")
    census = jload(FF / "census_v0.json")
    ps = admit["per_species"]
    targets = [k for k in ps if k != "OtherPlant"]
    rejected = sum(ps[k]["boxes"].get("conflict", 0) + ps[k]["boxes"].get("unknown", 0) for k in targets)
    other_conflict = ps["OtherPlant"]["boxes"]["conflict"]
    verified = admit["boxes"]["verified"]
    from weed_optimizer_framework.tools.inc import verify as VF
    unin, embedded, by_src = 0, 0, {}
    for r in census:
        embedded += r["n"]
        b = by_src.setdefault(r["source"], [0, 0])
        b[1] += r["n"]
        if r["label"] != "OtherPlant":
            continue
        st = VF.other_name_status(r["src_name"])
        if st in ("no_name", "generic"):         # v1: numeric names are generic ones whose key is digits
            unin += r["n"]
            b[0] += r["n"]
    s2_sources = sorted(s for s, (u, t) in by_src.items() if t and u / float(t) > 0.5)
    rag_v = ps["Ragweed"]["boxes"]["verified"]
    rag_j = pool["boxes_per_class"]["Ragweed"]
    q05 = sel["retrieval"]["train_core_image_score_q05_q50"][0]
    below = sorted(s for s, e in sel["retrieval"]["source_evidence"].items() if e["median"] < q05)
    inc_pool = sel["sources"]["increment_pool"]
    evidenced = [s for s in inc_pool if ((admit["per_slug"].get(s) or {}).get("boxes") or {}).get("verified", 0) >= 1]
    return {"rejected": rejected + other_conflict, "verified": verified, "unin": unin, "embedded": embedded,
            "embedded_admit": sum(admit["boxes"].values()), "s2_sources": s2_sources, "rag": (rag_v, rag_j),
            "below_q05": below, "q05": q05, "pool_sources": len(inc_pool), "evidenced": len(evidenced),
            "parts": {"conflict": sum(ps[k]["boxes"].get("conflict", 0) for k in targets),
                      "unknown": sum(ps[k]["boxes"].get("unknown", 0) for k in targets),
                      "other_conflict": other_conflict}}


def cluster_inc():
    """The cluster INC_DIR the fixtures' realloop_v1 names (its base manifest)."""
    m = jload(FF / "realloop_v1" / "exp.json")["base"]["manifest"]
    return posixpath.dirname(posixpath.dirname(posixpath.dirname(m)))


# ------------------------------------------------------------------------ R9
def test_r9():
    print("R9 today's evidence: D17 fires; L10 census first; L11 and L12 for the S2 sources; no L13; DA staged; X11")
    n = real_numbers()
    check("(the fixture) S1's parts are the contract's: 2,476 + 2,756 + 6,256 = 11,488 against 2,049 verified",
          (n["parts"]["conflict"], n["parts"]["unknown"], n["parts"]["other_conflict"], n["rejected"],
           n["verified"]) == (2476, 2756, 6256, 11488, 2049), n["parts"])
    root = funnel_world("r9")
    claims = jload(FF / "claims" / "claims_seed.json")
    ev = E.load_dir(root, "realloop_v1", exps=["realloop_v1"], claims=claims)
    diags, by, res = run(ev)
    d17 = by["D17"]
    sig = (d17.get("detail") or {}).get("signals") or {}
    check("D17 scarcity_conclusion_unaudited fires (warn)", d17["fired"] and d17["severity"] == "warn"
          and d17["name"] == "scarcity_conclusion_unaudited", d17["summary"])
    s1 = (sig.get("S1") or {}).get("detail") or {}
    check("S1 = %d/%d, read from admit_summary.json" % (n["rejected"], n["verified"]),
          sig["S1"]["holds"] is True and (s1.get("rejected"), s1.get("kept")) == (n["rejected"], n["verified"])
          and {n["parts"]["conflict"], n["parts"]["unknown"], n["parts"]["other_conflict"], n["verified"]}
          <= set(cite_values(sig["S1"])), s1)
    s2 = (sig.get("S2") or {}).get("detail") or {}
    check("S2 = %d/%d under name status v1 (census_v0 through verify.other_name_status), the version cited"
          % (n["unin"], n["embedded"]),
          sig["S2"]["holds"] is True and (s2.get("uninformative"), s2.get("boxes")) == (n["unin"], n["embedded"])
          and n["embedded"] == n["embedded_admit"] and (n["unin"], n["embedded"]) == (220104, 545318)
          and has_cite(sig["S2"], E.FUNNEL_LEDGER, "/name_status_version", "v1"), s2)
    s3 = (sig.get("S3") or {}).get("detail") or {}
    out = [(r["class"], r["verified"], r["joined"]) for r in s3.get("outliers") or []]
    check("S3: Ragweed %d/%d is the one class yield outlier (median %.3f)" % (n["rag"] + (s3.get("median") or 0,)),
          sig["S3"]["holds"] is True and out == [("Ragweed",) + n["rag"]] and n["rag"] == (27, 3029)
          and abs((s3.get("median") or 0) - 0.431) < 0.001, s3)
    s4 = (sig.get("S4") or {}).get("detail") or {}
    check("S4: every known-truth set lies in the reference domain; the sources below q05 are select_summary's",
          sig["S4"]["holds"] is True and all(s["in_domain"] for s in s4.get("sets") or [])
          and sorted(b["source"] for b in s4.get("below_q05") or []) == n["below_q05"]
          and len(n["below_q05"]) == 4 and s4.get("q05") == n["q05"], s4)
    pairs = [(p["stage"], p["depends_on"], p["in"], p["kept"]) for p in (sig["S5"]["detail"].get("pairs") or [])]
    check("S5: the evidence stage depends on the unaudited verifier (%d of %d sources kept)"
          % (n["evidenced"], n["pool_sources"]),
          sig["S5"]["holds"] is True and ("S12", "S8", n["pool_sources"], n["evidenced"]) in pairs
          and (n["evidenced"], n["pool_sources"]) == (5, 38), pairs)
    check("S7: realloop_v1 /steps/2/truth/verdict is 'helps' (an UNVERIFIED step) while no clean step helps",
          sig["S7"]["holds"] is True and has_cite(sig["S7"], "realloop_v1/report.json", "/steps/2/truth/verdict", "helps")
          and has_cite(sig["S7"], "realloop_v1/exp.json", "/steps/2/kind", "unverified"), sig["S7"]["summary"])
    check("D17 cites every held signal and every cite resolves",
          set(d17["detail"]["held"]) >= {"S1", "S2", "S3", "S4", "S5", "S7"}
          and all(ev.check_cite(c) for c in d17["cites"]), d17["detail"].get("held"))
    check("(C): realloop_v1 accepted nothing and its M was sized down; C1 and C2 are open negative claims",
          [f["form"] for f in d17["detail"]["conclusion"]["forms"]] == ["loop_negative", "sized_down", "claims"]
          and d17["detail"]["conclusion"]["forms"][2]["claims"] == ["C1", "C2"], d17["detail"]["conclusion"])
    props = res["proposals"]
    inc = cluster_inc()
    want = ["sbatch", "run_inc_funnel.sh", "census", "--prereg", inc + "/funnel/prereg_v1.json", "--out",
            inc + "/funnel/"]
    check("L10 is ranked first with the exact command: %s" % " ".join(want),
          props and props[0]["lever"] == "L10" and props[0]["argv"] == want
          and props[0]["policy_action"] == "inc_funnel_audit" and props[0]["risk"] == "R2"
          and set(props[0]["trigger"]) == {"D17", "D19"}, props[0]["argv"] if props else None)
    check("  it waits for L12 (census reads the taxonomy cache on the cluster), listed as deferred with then",
          (props[0].get("waits_for") or {}).get("lever") == "L12"
          and any(x["lever"] == "L10" and "after L12" in x["reason"] for x in res["deferred"]))
    lv = {p["lever"]: p for p in props}
    check("L12 and L11 (with L11a) are proposed for the S2 sources (the sources more than half uninformative)",
          {"L12", "L11a", "L11"} <= set(lv) and sorted(lv["L12"]["sources"]) == n["s2_sources"]
          and sorted(lv["L11"]["sources"]) == n["s2_sources"] and len(n["s2_sources"]) >= 1
          and lv["L12"]["argv"][:4] == ["python", "-m", "weed_optimizer_framework.tools.funnel", "fetch"]
          and lv["L11"]["argv"][:3] == ["sbatch", "run_inc_funnel.sh", "map"], sorted(lv))
    print("L11a: a cards index with refusals made under another domain config is fetched again, once per config")
    listing = {"files": {"funnel/taxonomy_cache.json": {"sha256": "a" * 64, "bytes": 1},
                         "funnel/cards/index.json": {"sha256": "b" * 64, "bytes": 1}}}
    (TMP / "r9_l11a").mkdir(exist_ok=True)
    root_c = funnel_world("r9_l11a_w", extra=None)
    (root_c / "funnel" / "files.json").write_text(json.dumps(listing))
    stale = {"index_domain_sha256": "c" * 64, "domain_sha256": "d" * 64, "refused": 7}
    for ctx_extra, want in (({}, None), ({"funnel_cards_stale": stale}, "d" * 12)):
        evc = E.load_dir(root_c, "realloop_v1", exps=["realloop_v1"], context=ctx_extra,
                         claims=jload(FF / "claims" / "claims_seed.json"))
        _dc, _bc, rc = run(evc)
        l11a = [p_ for p_ in rc["proposals"] if p_["lever"] == "L11a"]
        if want is None:
            check("  a current index on the cluster: L11a deferred",
                  not l11a and any(x["lever"] == "L11a" for x in rc["deferred"]), [p_["lever"] for p_ in rc["proposals"]])
        else:
            check("  a stale index: L11a proposed, keyed by the new config (a meta param, no command flag)",
                  len(l11a) == 1 and l11a[0]["params"].get("config") == want
                  and "--config" not in l11a[0]["argv"] and l11a[0]["argv"][:4] == ["python", "-m",
                  "weed_optimizer_framework.tools.funnel", "fetch"], l11a)
            ctx_done = dict(ctx_extra, lineage=[{"lever": "L11a", "status": "executed",
                                                 "params": {"what": "cards", "config": want}}])
            evd = E.load_dir(root_c, "realloop_v1", exps=["realloop_v1"], context=ctx_done,
                             claims=jload(FF / "claims" / "claims_seed.json"))
            _dd, _bd, rd = run(evd)
            check("  ... and not again once that config's fetch ran",
                  not [p_ for p_ in rd["proposals"] if p_["lever"] == "L11a"], [p_["lever"] for p_ in rd["proposals"]])
    check("no L13 anywhere (it runs only after D18)",
          "L13" not in lv and not [x for x in res["deferred"] if x["lever"] == "L13"]
          and "L13" not in d17["levers"], sorted(lv))
    check("the devil's-advocate pass is staged (OP_DA) and card X11 raised (S4 holds)",
          [o["op"] for o in res["operations"]] == ["OP_DA"] and "X11" in [c["lever"] for c in res["cards"]])
    check("D17 says what is missing: an out-of-domain known-truth set",
          "out-of-domain known-truth set" in (d17["detail"].get("needs") or ""), d17["detail"].get("needs"))
    check("every proposal is priced: L10 and L11 at run_inc_funnel.sh's walltime, the lab fetches at 0",
          lv["L10"]["est_gpu_hours"] == 8.0 and lv["L11"]["est_gpu_hours"] == 8.0
          and lv["L12"]["est_gpu_hours"] == 0.0, [(p["lever"], p["est_gpu_hours"]) for p in props])
    from weed_optimizer_framework.tools.inc_autopilot import executor as X
    from weed_optimizer_framework.tools.brain import policy as POL
    ok = []
    for p in props:
        row = POL.describe(p["policy_action"])
        pol, _meta = X.resolve_params(p["policy_action"], row, p["params"], p["argv"], p["est_gpu_hours"])
        good, _why = X.argv_check(X.render(p["policy_action"], pol), p["argv"])
        ok.append(good and POL._check_params(pol, row["param_bounds"])[0])
    check("the executor renders every proposal back to its argv, inside its policy row's bounds", all(ok), ok)

    print("R9: R1-R8 unchanged by the funnel ledger (their diagnoses; the rules version aside)")
    for name, exps, cur, step1 in (("r5", WHOLE, "pilot_v2", None), ("r6", WHOLE_V3, "pilot_v3", None),
                                   ("r8", WHOLE_V3, "pilot_v3", "r8")):
        a = legacy_world(name + "_a", exps, step1)
        b = legacy_world(name + "_b", exps, step1)
        shutil.copyfile(FF / "funnel_ledger_summaries.json", b / "funnel" / "funnel_ledger.json")
        da_, db_ = DG.detect(E.load_dir(a, cur, exps=exps)), DG.detect(E.load_dir(b, cur, exps=exps))
        # D14's summary counts the files the loader read (one more here); its verdict is compared
        keep = lambda ds: [d if d["id"] != "D14" else dict(d, summary=None) for d in ds   # noqa: E731
                           if d["id"] not in FUNNEL_IDS]
        check("%s: every diagnosis but D17-D19 is byte-identical with the ledger in the evidence" % name.upper(),
              json.dumps(keep(da_), sort_keys=True) == json.dumps(keep(db_), sort_keys=True))
        pa = LV.stable(LV.propose(da_, E.load_dir(a, cur, exps=exps)))["proposals"]
        pb = LV.stable(LV.propose(db_, E.load_dir(b, cur, exps=exps)))["proposals"]
        check("%s: its own proposals are unchanged, in order (L10 may join them)" % name.upper(),
              [p for p in pb if p["lever"] not in LV.FUNNEL_LEVERS] == pa, ([p["lever"] for p in pa],
                                                                           [p["lever"] for p in pb]))

    print("R9 under the test-blindness perturbation")
    b = funnel_world("r9_blind")
    for rel in ("realloop_v1/report.json", "realloop_v1/exp.json", "realloop_v1/build_summary.json",
                "funnel/funnel_ledger.json"):
        p = b / rel
        p.write_text(json.dumps(perturb_non_dev(jload(p)), indent=1))
    evb = E.load_dir(b, "realloop_v1", exps=["realloop_v1"], claims=claims)
    db_ = DG.detect(evb)
    check("the perturbed tree differs from the pinned one", (b / "realloop_v1" / "report.json").read_bytes()
          != (FF / "realloop_v1" / "report.json").read_bytes())
    check("diagnoses byte-identical", json.dumps(diags, sort_keys=True) == json.dumps(db_, sort_keys=True))
    check("proposals, cards, operations and argv byte-identical",
          json.dumps(LV.stable(res), sort_keys=True) == json.dumps(LV.stable(LV.propose(db_, evb)), sort_keys=True))
    return root, ev, diags, res


# ------------------------------------------------------------------------ R15
# The cluster's funnel files when embed-judges refused (2026-09-28): census,
# leak, the cards and the geometry match done, the DINOv2 shard written; no
# KT7 photos, no judges.
LIVE_0928 = ("funnel/taxonomy_cache.json", "funnel/census_v1.json", "funnel/leak_v1.json", "funnel/cards/index.json",
             "funnel/relation_geometry_v1.json", "funnel/emb_dinov2/emb_s000_of_001.npz")
KT7_FILES = ("funnel/kt7/kt7_items.jsonl", "funnel/kt7/crops_kt7.csv")
KT7_TABLE = "funnel/kt7/crops_kt7.csv"
# The campaign's lineage then (campaign._Run.lineage: the steps that ran).
LINEAGE_0928 = [{"lever": "L12", "status": "executed", "params": {"what": "taxonomy"}},
                {"lever": "L10", "status": "executed", "params": {"verb": "census"}},
                {"lever": "L11a", "status": "executed", "params": {"what": "cards"}},
                {"lever": "L10", "status": "executed", "params": {"verb": "leak"}},
                {"lever": "L11", "status": "executed", "params": {"part": "geometry"}}]


def _sha(p, salt=""):
    return hashlib.sha256((salt + p).encode()).hexdigest()


def _embed_run(n, status="failed", state="FAILED"):
    """A lineage record of an embed-judges run, as the ticker writes it: its job's
    final state (sacct) makes it 'failed', with the job's ids, states and log."""
    rec = {"lever": "L10", "status": status, "params": {"verb": "embed-judges"}, "proposal_id": "p%d" % n}
    if status == "failed":
        jid = str(47300000 + n)
        rec["job"] = {"ids": [jid], "states": {jid: state},
                      "log": "%s/funnel/logs/inc_funnel_embed-judges_%s.out" % (cluster_inc(), jid)}
    return rec


def test_r15():
    print("R15 the KT7 incident of 2026-09-28: embed-judges waits for the KT7 photos and L11a fetches them; a "
          "failed job lets the step run again once they are there; the third failure stays with a person")
    claims = jload(FF / "claims" / "claims_seed.json")
    root = funnel_world("r15")
    inc = cluster_inc()
    lab_out = posixpath.join(str(M.LAB_REPO / LV.protocol("funnel_lab_inc_dir")), "funnel/")
    menu = LV.load_menu()

    def evidence(files, lineage=(), lab=None):
        (root / "funnel" / "files.json").write_text(json.dumps(
            {"format": "funnel-files/1", "files": {p: {"sha256": _sha(p), "bytes": 1} for p in files}}))
        ctx = {"lineage": list(lineage)}
        if lab:
            ctx["funnel_lab"] = lab
        return E.load_dir(root, "realloop_v1", exps=["realloop_v1"], context=ctx, claims=claims)

    def props(res, lever):
        return [p for p in res["proposals"] if p["lever"] == lever]

    def deferred(res, lever):
        return [x["reason"] for x in res["deferred"] if x["lever"] == lever]

    # (1) the incident as it stood: embed-judges ran, its failure unrecorded
    ev = evidence(LIVE_0928, LINEAGE_0928 + [_embed_run(1, status="executed")])
    diags, by, res = run(ev)
    check("L10's next step is embed-judges (its output, funnel/judges/, is not on the cluster)",
          LV.funnel_next(ev, "L10")[1] == {"verb": "embed-judges"}, LV.funnel_next(ev, "L10"))
    w = LV._waits(ev, "L10", "embed-judges", menu)
    check("embed-judges waits for %s: then L11a, what kt7" % KT7_TABLE,
          (w or {}).get("lever") == "L11a" and w.get("what") == "kt7" and w.get("cluster_file") == KT7_TABLE, w)
    check("  the run that ran is not repeated while its failure is not recorded (a person's decision, as before)",
          not props(res, "L10") and any("was already run in this campaign (executed)" in r
                                        for r in deferred(res, "L10")), deferred(res, "L10"))
    l11a = props(res, "L11a")
    check("L11a is proposed with what=kt7, the fetch the waiting step needs (the cards are on the cluster)",
          len(l11a) == 1 and l11a[0]["params"].get("what") == "kt7"
          and l11a[0]["argv"] == ["python", "-m", "weed_optimizer_framework.tools.funnel", "fetch", "--prereg",
                                  lab_out + "prereg_v1.json", "--what", "kt7", "--out", lab_out]
          and (l11a[0].get("needed_by") or {}).get("verb") == "embed-judges" and l11a[0]["writes"] == KT7_TABLE,
          l11a and (l11a[0]["argv"], l11a[0].get("needed_by")))
    from weed_optimizer_framework.tools.inc_autopilot import executor as X
    from weed_optimizer_framework.tools.brain import policy as POL

    def executable(p):
        row = POL.describe(p["policy_action"])
        pol, _meta = X.resolve_params(p["policy_action"], row, p["params"], p["argv"], p["est_gpu_hours"])
        ok, _why = X.argv_check(X.render(p["policy_action"], pol), p["argv"])
        return ok and POL._check_params(pol, row["param_bounds"])[0]
    check("  the executor renders it back to its argv, inside inc_funnel_fetch's policy bounds",
          l11a and executable(l11a[0]))
    check("  its lineage key carries what (kt7 is a new step: the cards fetch that ran does not stand for it)",
          l11a and LV.stable(l11a[0])["params"] == {"what": "kt7", "est_gpu_hours": 0.0}, l11a and l11a[0]["params"])

    # (2) the job's final state recorded: FAILED
    ev = evidence(LIVE_0928, LINEAGE_0928 + [_embed_run(1)])
    diags, by, res = run(ev)
    l10 = props(res, "L10")
    want = ["sbatch", "run_inc_funnel.sh", "embed-judges", "--prereg", inc + "/funnel/prereg_v1.json", "--out",
            inc + "/funnel/"]
    check("its job ended FAILED: embed-judges is proposed again, with its exact command",
          len(l10) == 1 and l10[0]["argv"] == want and executable(l10[0]), l10 and l10[0]["argv"])
    check("  carrying waits_for L11a --what kt7, and listed as deferred behind it",
          l10 and (l10[0].get("waits_for") or {}).get("what") == "kt7"
          and any("then: L10 after L11a --what kt7 (%s is not on the cluster)" % KT7_TABLE in r
                  for r in deferred(res, "L10")), (l10 and l10[0].get("waits_for"), deferred(res, "L10")))
    check("  and L11a (what=kt7) is still the proposal that unblocks it",
          [p["params"]["what"] for p in props(res, "L11a")] == ["kt7"], [p["params"] for p in props(res, "L11a")])

    # (3) fetched on the lab, not yet on the cluster: the sync pushes it
    lab = {p: _sha(p, "lab") for p in KT7_FILES}
    ev = evidence(LIVE_0928, LINEAGE_0928 + [_embed_run(1), {"lever": "L11a", "status": "executed",
                                                             "params": {"what": "kt7"}}], lab=lab)
    diags, by, res = run(ev)
    check("KT7 fetched on the lab: L11a is not proposed again (the sync pushes it) and L10 still waits",
          not props(res, "L11a") and props(res, "L10") and props(res, "L10")[0].get("waits_for")
          and any("%s is on the lab and not yet on the cluster" % KT7_TABLE in r and "pushes it" in r
                  for r in deferred(res, "L11a")),
          (deferred(res, "L11a"), [p["lever"] for p in res["proposals"]]))
    check("  the funnel sync is due for both KT7 files", set(KT7_FILES) <= set(LV.funnel_sync_needed(ev)),
          LV.funnel_sync_needed(ev))

    # (4) on the cluster (check_manifest rebuilt the machine-local table)
    lab_same = {"funnel/kt7/kt7_items.jsonl": _sha("funnel/kt7/kt7_items.jsonl"), KT7_TABLE: _sha(KT7_TABLE, "lab")}
    ev = evidence(LIVE_0928 + KT7_FILES, LINEAGE_0928 + [_embed_run(1), {"lever": "L11a", "status": "executed",
                                                                         "params": {"what": "kt7"}}], lab=lab_same)
    diags, by, res = run(ev)
    l10 = props(res, "L10")
    check("the KT7 file appears on the cluster: embed-judges is proposed again, waiting for nothing",
          len(l10) == 1 and l10[0]["argv"] == want and not l10[0].get("waits_for")
          and not deferred(res, "L10"), (l10 and l10[0].get("waits_for"), deferred(res, "L10")))
    check("  L11a has nothing left to fetch", not props(res, "L11a") and deferred(res, "L11a"), deferred(res, "L11a"))
    check("  the machine-local table (each host's own image paths) does not keep the sync due",
          KT7_TABLE not in LV.funnel_sync_needed(ev), LV.funnel_sync_needed(ev))

    # (5) the bound: proposed again at most funnel_job_retries times after a failed job
    retries = LV.protocol("funnel_job_retries")
    runs = [_embed_run(i) for i in range(1, retries + 1)]
    diags, by, res = run(evidence(LIVE_0928 + KT7_FILES, LINEAGE_0928 + runs))
    check("after %d failed runs it is proposed again (%d retries)" % (retries, retries),
          retries == 2 and [p["argv"][2] for p in props(res, "L10")] == ["embed-judges"],
          [p["argv"] for p in props(res, "L10")])
    runs.append(_embed_run(retries + 1, state="TIMEOUT"))
    diags, by, res = run(evidence(LIVE_0928 + KT7_FILES, LINEAGE_0928 + runs))
    why = " ".join(deferred(res, "L10"))
    check("the third failure stays with a person: deferred with the job's state and its log path",
          not props(res, "L10") and "failed 3 times" in why and "ended TIMEOUT" in why
          and "%s/funnel/logs/inc_funnel_embed-judges_47300003.out" % inc in why and "a person" in why, why)

    # (6) the other steps that read the KT7 photos wait for them too
    ev = evidence(LIVE_0928)
    ok = {v: (LV._waits(ev, "L10", v, menu) or {}) for v in ("embed-judges", "qualify", "draw", "sheets", "estimate")}
    check("qualify, draw, sheets and estimate wait for the KT7 table as well (then L11a --what kt7)",
          all(w.get("lever") == "L11a" and w.get("what") == "kt7" and w.get("cluster_file") == KT7_TABLE
              for w in ok.values()), ok)
    w_est = LV._waits(evidence(LIVE_0928 + KT7_FILES), "L10", "estimate", menu) or {}
    check("  estimate then still waits for the devil's-advocate record (OP_DA)",
          w_est.get("lever") == "OP_DA" and w_est.get("cluster_file") == "funnel/prospective_da.json", w_est)
    check("  census and leak read no KT7 file (census waits only for the taxonomy cache)",
          LV._waits(ev, "L10", "leak", menu) is None
          and (LV._waits(evidence(()), "L10", "census", menu) or {}).get("lever") == "L12")

    # (7) the order of the fetches, and a fetch L11a's command cannot run
    files = tuple(p for p in LIVE_0928 if p != "funnel/cards/index.json")
    diags, by, res = run(evidence(files, LINEAGE_0928[:2] + LINEAGE_0928[3:] + [_embed_run(1)]))
    check("cards missing and KT7 needed: L11a fetches the cards first",
          [p["params"]["what"] for p in props(res, "L11a")] == ["cards"], [p["params"] for p in props(res, "L11a")])
    diags, by, res = run(evidence(files, LINEAGE_0928 + [_embed_run(1)]))
    check("  the cards fetch already ran (its index not pushed yet): L11a moves on to kt7",
          [p["params"]["what"] for p in props(res, "L11a")] == ["kt7"], [p["params"] for p in props(res, "L11a")])
    menu_h12 = copy.deepcopy(menu)
    menu_h12["levers"]["L10"]["preconditions"]["embed-judges"] = {
        "cluster_file": "funnel/known_items_v1.json", "then": "L11a", "what": "known-items", "why": "test"}
    ev = evidence(LIVE_0928 + KT7_FILES, LINEAGE_0928 + [_embed_run(1)])
    fw = LV.fetch_waits(ev, menu_h12)
    res = LV.propose(DG.detect(ev), ev, menu=menu_h12)
    check("a step waiting on the H12 list selects fetch --what known-items, which needs --sources the L11a command "
          "does not carry: deferred for a person, never run to a sure refusal",
          list(fw) == ["known-items"] and not props(res, "L11a")
          and any("--what known-items" in r and "--sources" in r for r in deferred(res, "L11a")), deferred(res, "L11a"))


def legacy_world(name, exps, step1):
    root = TMP / name
    if root.exists():
        shutil.rmtree(str(root))
    for e in exps:
        for f in E.EXP_FILES:
            p = FIX / e / f
            if p.is_file():
                (root / e / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, root / e / f)
    (root / "funnel").mkdir(parents=True, exist_ok=True)
    if step1 == "r8":
        (root / "step1").mkdir(exist_ok=True)
        for f in ("select_summary.json", "admit_summary.json"):
            shutil.copyfile(FIX / "step1_copies" / f, root / "step1" / f)
        shutil.copyfile(FIX / "synthetic" / "step1_real_calibration_failed" / "relevance.json",
                        root / "step1" / "relevance.json")
    return root


UNKNOWN_EXAM = "ood24"


def perturb_non_dev(obj):
    """Every number under a non-dev split changed (the known ones and every
    exam but dev under an "exams" dict), and an unknown exam added to each."""
    def go(x, under, parent=None):
        if isinstance(x, dict):
            out = {k: go(v, under or k in E.BLOCKED_SPLITS or (parent == "exams" and k != "dev"), k)
                   for k, v in x.items()}
            if parent == "exams":
                out[UNKNOWN_EXAM] = {"twelve": {"mean": 0.123456789, "n": 3}}
            return out
        if isinstance(x, list):
            return [go(v, under) for v in x]
        if under and isinstance(x, (int, float)) and not isinstance(x, bool):
            return -x + 0.5 if isinstance(x, float) else x + 11
        return x
    return go(obj, False)


# ------------------------------------------------------------------ R9_early
def test_r9_early():
    print("R9_early: the Step 1 files without realloop_v1: D19 fires and L10 comes before L2")
    root = legacy_world("r9_early", WHOLE_V3, "r8")
    for f in ("pool_summary.json", "calibration.json"):
        shutil.copyfile(FF / "step1" / f, root / "step1" / f)
    shutil.copyfile(FF / "funnel_ledger_summaries.json", root / "funnel" / "funnel_ledger.json")
    ev = E.load_dir(root, "pilot_v3", exps=WHOLE_V3)
    diags, by, res = run(ev)
    check("D19 filter_recall_unmeasured fires on S1 and S4 for the unaudited verifier stages",
          by["D19"]["fired"] and set(by["D19"]["detail"]["held"]) >= {"S1", "S4"}
          and set(by["D19"]["detail"]["stages"]) == {"S8", "S9"}, by["D19"]["summary"])
    check("D17 is silent: no negative conclusion yet (no loop, no claim)", not by["D17"]["fired"], by["D17"]["summary"])
    order = [p["lever"] for p in res["proposals"]]
    check("L10 is proposed before D4's L2 (DEC-4)", "L10" in order and "L2" in order
          and order.index("L10") < order.index("L2"), order)
    rep = jload(FF / "realloop_v1" / "report.json")
    print("  info L10 would have preceded realloop_v1's %.1f GPU-hours (read from its report)"
          % rep["gpu_hours_total"])
    check("(the report) realloop_v1 cost 26.2 GPU-hours", round(rep["gpu_hours_total"], 1) == 26.2,
          rep["gpu_hours_total"])


# ------------------------------------------------------------------------ R9b
def test_r9b():
    print("R9b: claim C2 with only the artifacts of 2026-08-25: D17 and D19 fire and ask for out-of-domain truth")
    fig = jload(ROOT.parent / "docs" / "poster" / "figures_data.json")["s1_gate_verdict_2026_08_25"]
    root = TMP / "r9b"
    (root / "funnel").mkdir(parents=True)
    shutil.copyfile(FF / "r9b" / "r9b_ledger.json", root / "funnel" / "funnel_ledger.json")
    ev = E.load_dir(root, "s1gate", exps=[], claims=jload(FF / "r9b" / "r9b_claims.json"))
    check("the evidence holds no Step 1 and no experiment: only the ledger and the claim",
          sorted(ev.artifacts) == [E.CLAIMS, E.FUNNEL_LEDGER], sorted(ev.artifacts))
    diags, by, res = run(ev)
    s1 = by["D17"]["detail"]["signals"]["S1"]
    total, passing = fig["audited_harvested_labelled_images"], fig["passing_bar_labelled_images"]
    check("S1 = %d/%d from the S1 gate's own counts (figures_data.json)" % (total - passing, passing),
          s1["holds"] is True and (s1["detail"]["rejected"], s1["detail"]["kept"]) == (total - passing, passing))
    check("D17 fires on C2 (an open scarcity claim) and D19 fires",
          by["D17"]["fired"] and by["D19"]["fired"]
          and by["D17"]["detail"]["conclusion"]["forms"] == [dict(by["D17"]["detail"]["conclusion"]["forms"][0])]
          and by["D17"]["detail"]["conclusion"]["forms"][0]["claims"] == ["C2"])
    check("both ask for an out-of-domain known-truth set before the claim can be cited",
          "out-of-domain known-truth set" in (by["D17"]["detail"].get("needs") or "")
          and "out-of-domain known-truth set" in (by["D19"]["detail"].get("needs") or "")
          and "out-of-domain known-truth set" in by["D19"]["summary"])


# ------------------------------------------------------------------ R10-R12
def _mini_run(tmp, claims_file=None):
    """A campaign _Run on a temporary lab tree (no cluster): its claims register
    and ledger are the real ones the ticker writes."""
    from weed_optimizer_framework.tools.inc_autopilot import campaign as C
    lab = tmp / "lab"
    paths = C.Paths(lab)
    paths.claims.parent.mkdir(parents=True, exist_ok=True)
    if claims_file:
        shutil.copyfile(claims_file, paths.claims)
    r = C._Run("funnelcamp", {"enabled": True}, paths, C._SshBudget(None), lambda: 1790000000.0,
               C._Log(), None, {}, None, None, None)
    r.st = C._blank_state("funnelcamp", r.cfg)
    r.st["exp"] = "realloop_v1"
    return r, paths


def test_r10():
    print("R10 audited negative: D17 and D18 silent; C1 moves to tested_survives")
    root = funnel_world("r10", extra={"funnel/audit_v1.json": "audits/r10_audit.json"})
    claims = jload(FF / "claims" / "claims_challenged.json")
    ev = E.load_dir(root, "realloop_v1", exps=["realloop_v1"], claims=claims)
    diags, by, res = run(ev)
    check("the audit is valid and made on this ledger (fingerprint)",
          ev.json(E.FUNNEL_AUDIT)["ledger_fingerprint"] == ev.json(E.FUNNEL_LEDGER)["fingerprint"])
    check("D17 silent: audited", not by["D17"]["fired"] and by["D17"]["summary"].startswith("audited"),
          by["D17"]["summary"])
    check("D18 silent (no stratum reaches FN lb 0.10), proposing C1 -> tested_survives",
          not by["D18"]["fired"] and [t["claim_id"] for t in by["D18"]["detail"]["claim_transitions"]] == ["C1"]
          and by["D18"]["detail"]["claim_transitions"][0]["audit_sha256"]
          == hashlib.sha256((root / "funnel" / "audit_v1.json").read_bytes()).hexdigest(),
          by["D18"]["summary"])
    check("nothing is proposed from D17-D19", not [p for p in res["proposals"] if set(p["trigger"]) & set(FUNNEL_IDS)])
    tmp = TMP / "r10_campaign"
    r, paths = _mini_run(tmp, FF / "claims" / "claims_challenged.json")
    r.ev = ev
    r._apply_d18(diags)
    reg = jload(paths.claims)
    c1 = [c for c in reg["claims"] if c["id"] == "C1"][0]
    led = [json.loads(x) for x in paths.ledger.read_text().splitlines()]
    check("the campaign applies it: C1 is tested_survives, by the autopilot, citing the audit's sha256",
          c1["status"] == "tested_survives" and c1["history"][-1]["by"] == M.AUTOPILOT_ACTOR
          and c1["history"][-1]["cites"][0]["value"] == by["D18"]["detail"]["claim_transitions"][0]["audit_sha256"],
          c1["history"][-1])
    check("  and the transition is a campaign ledger entry with its decided_by",
          any(e["event"] == "claim_transition" and e["decided_by"] == M.AUTOPILOT_ACTOR and e.get("to") ==
              "tested_survives" for e in led), [e["event"] for e in led])
    from weed_optimizer_framework.tools.funnel import claims as FC
    reg2 = jload(FF / "claims" / "claims_seed.json")
    try:
        FC.transition(reg2, "C1", "tested_survives", M.AUTOPILOT_ACTOR, "x",
                      [M.cite("funnel/audit_v1.json", "0" * 64, pointer="")], "autopilot",
                      {"audit_sha256": "0" * 64, "valid": True, "fingerprint_match": True, "d18_fired": False})
        moved = True
    except FC.ClaimsError:
        moved = False
    check("an open claim cannot skip the devil's advocate (open -> tested_survives is refused)", not moved)


def test_r11():
    print("R11 audited positive: D18 fires; L13 only for the Ragweed stratum; the sibling stratum guarded; L14")
    root = funnel_world("r11", extra={"funnel/audit_v1.json": "audits/r11_audit.json",
                                      "funnel/class_maps.json": "audits/r11_class_maps.json"})
    ev = E.load_dir(root, "realloop_v1", exps=["realloop_v1"], claims=jload(FF / "claims" / "claims_challenged.json"))
    diags, by, res = run(ev)
    d18 = by["D18"]
    check("D18 fires (warn) with L13, L14 and X11", d18["fired"] and d18["severity"] == "warn"
          and d18["levers"][:2] == ["L13", "L14"] and "X11" in d18["levers"], (d18["levers"], d18["summary"]))
    strata = [(s["stratum"], s["policy"]) for s in d18["detail"]["strata"]]
    check("L13 covers only weed_crop's rejected Ragweed stratum, by REC-AUTH (R-A)",
          strata == [("G2/source=project_agml__weed_crop_detection/label=Ragweed/fail=p_below_tau", "R-A")]
          and d18["detail"]["policy"] == "R-A", strata)
    conf = d18["detail"]["known_confusions"]
    check("the greenhouse OtherPlant -> Palmer stratum is a known confusion (A. retroflexus is a relative)",
          len(conf) == 1 and conf[0]["kind"] == "other_predicted_target" and "sibling guard" in conf[0]["why"], conf)
    inc = cluster_inc()
    l13 = [p for p in res["proposals"] if p["lever"] == "L13"]
    check("the L13 command", len(l13) == 1 and l13[0]["argv"] == [
        "sbatch", "run_inc_funnel.sh", "recover", "--prereg", inc + "/funnel/prereg_v1.json", "--audit",
        inc + "/funnel/audit_v1.json", "--maps", inc + "/funnel/class_maps.json", "--policy", "R-A", "--out",
        inc + "/step1_r1/"] and l13[0]["risk"] == "R3", l13[0]["argv"] if l13 else None)
    l14 = [p for p in res["proposals"] if p["lever"] == "L14"]
    check("the class map whose class count disagrees with its card goes to L14 (a lab hook, no command)",
          len(l14) == 1 and l14[0]["argv"] == [] and [q["src_id"] for q in l14[0]["queue"]] == ["2"]
          and "disagrees with the card" in l14[0]["queue"][0]["reason"], l14)
    check("X11 is raised", "X11" in [c["lever"] for c in res["cards"]])
    check("the recovered loop waits for the overlay (L2 with --increment-sources recovered deferred)",
          d18["detail"].get("then") == ["L2 with --increment-sources recovered"]
          and not [p for p in res["proposals"] if p["lever"] == "L2"])
    rec = TMP / "r11" / "funnel" / "recovery.json"
    rec.write_text(json.dumps({"format": "funnel-recovery/1", "status": "complete"}))
    ev2 = E.load_dir(root, "realloop_v1", exps=["realloop_v1"])
    d2, b2, r2 = run(ev2)
    l2 = [p for p in r2["proposals"] if p["lever"] == "L2"]
    want = ["python", "-m", "weed_optimizer_framework.tools.inc.realloop", "build", "--exp", "realloop_v2",
            "--base", inc + "/step1/base_B.jsonl", "--replay-mode", "full", "--recipes", "full",
            "--increment-sources", "recovered", "--step1-overlay", inc + "/step1_r1", "--size",
            str(jload(FF / "realloop_v1" / "exp.json")["increment_images"]), "--gate-flips-mode", "net"]
    check("with the overlay complete, D18 proposes realloop_v2 (L2 --increment-sources recovered, contract 9.1)",
          len(l2) == 1 and l2[0]["argv"] == want and "D18" in l2[0]["trigger"], l2[0]["argv"] if l2 else r2["deferred"])
    print("R11: an audit row on a guard or non-recoverable stage is never recovered")
    led = ev.json(E.FUNNEL_LEDGER)
    guard = [s["id"] for s in led["stages"] if s.get("guard") is True][0]
    notrec = [s["id"] for s in led["stages"] if s.get("recoverable") is not True and not s.get("guard")][0]
    aud = jload(FF / "audits" / "r11_audit.json")
    base_rows = copy.deepcopy(aud["d18_inputs"])
    big = max(r["recoverable_lb"] for r in base_rows) * 4
    bad_rows = [{"fn_lb": 0.9, "kind": "uninformative_label_space", "recoverable_lb": big,
                 "relative_of_prediction": False, "source_taxa": [], "stage": guard,
                 "stratum": "G5/source=rf_x/split=dev/bits=0-2"},
                {"fn_lb": 0.9, "kind": "target_rejected", "recoverable_lb": big, "relative_of_prediction": False,
                 "source_taxa": [], "stage": notrec, "stratum": "Gx/source=rf_y"}]
    root_g = funnel_world("r11_guard", extra={"funnel/class_maps.json": "audits/r11_class_maps.json"})
    (root_g / "funnel" / "audit_v1.json").write_text(json.dumps(dict(aud, d18_inputs=base_rows + bad_rows)))
    evg = E.load_dir(root_g, "realloop_v1", exps=["realloop_v1"],
                     claims=jload(FF / "claims" / "claims_challenged.json"))
    dg_, bg, rg = run(evg)
    check("with the Ragweed row: L13 still covers only the Ragweed stratum; the %s and %s rows are refused"
          % (guard, notrec),
          [s["stratum"] for s in bg["D18"]["detail"]["strata"]] == [base_rows[0]["stratum"]]
          and sorted(r["stage"] for r in bg["D18"]["detail"]["refused_stages"]) == sorted([guard, notrec])
          and bg["D18"]["detail"]["policy"] == "R-A", bg["D18"]["detail"].get("refused_stages"))
    (root_g / "funnel" / "audit_v1.json").write_text(json.dumps(dict(aud, d18_inputs=bad_rows)))
    evg2 = E.load_dir(root_g, "realloop_v1", exps=["realloop_v1"],
                      claims=jload(FF / "claims" / "claims_challenged.json"))
    dg2, bg2, rg2 = run(evg2)
    check("  alone they escalate, propose no L13 and move no claim",
          bg2["D18"]["fired"] and "OP_ESCALATE" in bg2["D18"]["levers"] and "L13" not in bg2["D18"]["levers"]
          and not [p for p in rg2["proposals"] if p["lever"] == "L13"]
          and bg2["D18"]["detail"]["claim_transitions"] == [], (bg2["D18"]["levers"], bg2["D18"]["summary"]))
    print("R11: a card map is proposed (R-C) only when H3a is supported, else the stratum goes to the judges")
    mh_row = {"fn_lb": 0.9, "kind": "uninformative_label_space", "recoverable_lb": big,
              "relative_of_prediction": False, "source_taxa": [], "stage": base_rows[0]["stage"],
              "stratum": "G3/unit=c:project_agml__mh_weed16_weed_detection|12"}
    mh_row["stage"] = [s["id"] for s in led["stages"] if s.get("recoverable") is True
                       and s.get("role") != "target_check"][0]
    for h3a, want_pol in ((None, "R-J"), ("inconclusive", "R-J"), ("supported", "R-C")):
        aud_h = dict(aud, d18_inputs=[mh_row])
        if h3a is not None:
            aud_h["hypotheses"] = {"H3a": {"verdict": h3a}}
        (root_g / "funnel" / "audit_v1.json").write_text(json.dumps(aud_h))
        evh = E.load_dir(root_g, "realloop_v1", exps=["realloop_v1"],
                         claims=jload(FF / "claims" / "claims_challenged.json"))
        _dh, bh, _rh = run(evh)
        st = bh["D18"]["detail"].get("strata") or []
        check("  mh_weed16 has a proposed card+geometry map; H3a %s -> %s" % (h3a, want_pol),
              len(st) == 1 and st[0]["policy"] == want_pol
              and (want_pol == "R-C" or st[0].get("why_not_card_map") == "H3a is %s" % h3a), st)
    from weed_optimizer_framework.tools.inc import realloop as RL
    got, real = {}, RL.build

    def fake(*a, **k):
        got["args"], got["kwargs"] = a, k
    RL.build = fake
    try:
        rc = RL.main(want[3:])
    finally:
        RL.build = real
    k = got.get("kwargs") or {}
    check("  realloop's own argparse reads it back (recovered, the overlay, M, full, full, net)",
          rc in (0, None) and got.get("args") == ("realloop_v2",) and k.get("increment_sources") == "recovered"
          and str(k.get("step1_overlay")).rstrip("/") == inc + "/step1_r1" and int(k.get("size")) == int(want[-3])
          and (k.get("replay_mode"), k.get("recipes"), k.get("gate_flips_mode")) == ("full", "full", "net"), got)


def test_r12():
    print("R12 an invalid audit: D18 escalates calibration_overlap; validate refuses items citing it")
    root = funnel_world("r12", extra={"funnel/audit_v1.json": "audits/r12_audit.json"})
    ev = E.load_dir(root, "realloop_v1", exps=["realloop_v1"], claims=jload(FF / "claims" / "claims_seed.json"))
    diags, by, res = run(ev)
    check("D18 fires crit as calibration_overlap and escalates",
          by["D18"]["fired"] and by["D18"]["severity"] == "crit" and "calibration_overlap" in by["D18"]["summary"]
          and by["D18"]["levers"] == ["OP_ESCALATE"], by["D18"]["summary"])
    check("D17 still fires: an invalid audit is no audit", by["D17"]["fired"]
          and "invalid" in by["D17"]["detail"]["audit"], by["D17"]["detail"]["audit"])
    cite = ev.cite(E.FUNNEL_AUDIT, "/d18_inputs/0/fn_lb")
    ok, why = V.resolve_cite(BP.dev_only(BP.artifacts_of(ev)), cite)
    check("validate.resolve_cite refuses a cite of the invalid audit", not ok and "valid" in why, why)
    pos = jload(FF / "da" / "da_positive.json")
    bad = copy.deepcopy(pos)
    bad["counter_arguments"][0]["evidence_cites"].append(cite)
    val = V.validate_da(bad, ev, ev.json(E.FUNNEL_LEDGER), BP.load_menu(), {"adversary": ADV, "planner": PLANNER},
                        diagnoses=diags, claims=jload(FF / "claims" / "claims_seed.json"))
    check("a devil's-advocate counter-argument citing it is dropped",
          val["ok"] and not val["counter_arguments"][0]["kept"]
          and any("valid" in r for r in val["counter_arguments"][0]["reasons"]), val["counter_arguments"][0])
    menu = BP.load_menu()
    item = {"lever": "L10", "params": {"verb": "census"}, "trigger": ["D17"], "rationale": "r", "falsifier": "f",
            "evidence_cites": [cite], "predicted": {"metric": "agreement", "direction": "up", "magnitude": None}}
    vp = V.validate({"ranked_menu": [item], "off_menu": []}, ev, menu, None, diagnoses=diags, exp="realloop_v1")
    check("a brain plan item citing it is dropped", not vp["menu"] and any(
        "valid" in r for d in vp["dropped"] for r in d["reasons"]), vp["dropped"])


# ------------------------------------------------------------------------ R13
VEH = FF / "vehicles" / "vehicles.json"


def test_r13():
    print("R13 the vehicles domain: the same code and thresholds fire D17 and D19; its exam is refused")
    from weed_optimizer_framework.tools.funnel import domain as FD
    from weed_optimizer_framework.tools.funnel import ledger as FL
    from weed_optimizer_framework.tools.inc_autopilot import remote as RM
    dom = FD.load(str(VEH))
    check("the vehicles config validates and names its exams", dom.name == "vehicles"
          and M.non_dev_exams(str(VEH)) == ("test", "night_exam", "rain_exam"))
    led = jload(FF / "vehicles" / "vehicles_ledger.json")
    check("its ledger validates (funnel.ledger)", FL.validate(led) == [], FL.validate(led))
    root = TMP / "r13"
    (root / "funnel").mkdir(parents=True)
    shutil.copyfile(FF / "vehicles" / "vehicles_ledger.json", root / "funnel" / "funnel_ledger.json")
    ev = E.load_dir(root, "roadrun", exps=[], claims=jload(FF / "vehicles" / "vehicles_claims.json"),
                    domain=str(VEH))
    diags, by, res = run(ev)
    held = by["D17"]["detail"].get("held") or []
    check("D17 fires on the vehicles claim with S1-S5 (the weed thresholds, thresholds.json)",
          by["D17"]["fired"] and set(held) >= {"S1", "S2", "S3", "S4", "S5"}, (by["D17"]["summary"], held))
    s3 = by["D17"]["detail"]["signals"]["S3"]["detail"]
    check("  S3 finds bus, S2 names the numeric-named and the no-name source",
          [r["class"] for r in s3["outliers"]] == ["bus"]
          and sorted(by["D17"]["detail"]["s2_sources"]) == ["dashcam_c", "fleet_b"])
    check("D19 fires, and says filter recall cannot be measured without out-of-domain truth",
          by["D19"]["fired"] and "out-of-domain known-truth set" in by["D19"]["summary"], by["D19"]["summary"])
    obj = {"final": [{"exams": {"dev": {"m": 1}, "night_exam": {"m": 2}}}], "night_exam": {"m": 3},
           "rain_exam": 4, "keep": 5}
    ev_v = E.from_texts({"funnel/funnel_ledger.json": json.dumps(led), "step1/admit_summary.json": json.dumps(obj)},
                        "roadrun", domain=str(VEH))
    ev_w = E.from_texts({"step1/admit_summary.json": json.dumps(obj)}, "roadrun")
    check("evidence refuses the vehicles exam name under its domain (and the weed list would not know it)",
          "night_exam" not in ev_v.json("step1/admit_summary.json") and "rain_exam" not in
          ev_v.json("step1/admit_summary.json") and "night_exam" in ev_w.json("step1/admit_summary.json"),
          ev_v.json("step1/admit_summary.json"))
    kept, dropped = RM.dev_only(obj, blocked=M.non_dev_exams(str(VEH)))
    check("remote.dev_only refuses it under the vehicles domain",
          "night_exam" not in kept and "/night_exam" in dropped and "night_exam" in RM.dev_only(obj)[0])
    rx = V.leak_text_re(str(VEH))
    check("validate's leak pattern refuses it under the vehicles domain (and the weed one does not)",
          rx.search("compare the night_exam numbers") is not None
          and V.LEAK_TEXT_RE.search("compare the night_exam numbers") is None
          and rx.search("the roadcam test set") is not None)
    rec = sorted(s["id"] for s in led["stages"] if s.get("recoverable") is True)
    reply = {"claim_id": "C1", "concessions": [], "stage_forecast": {s: 1.0 / len(rec) for s in rec},
             "counter_arguments": [{
                 "argument": "The night_exam scores show the bus class is fine.", "mechanism": "m",
                 "evidence_cites": [ev_v.cite(E.FUNNEL_LEDGER, "/stages/5/kept"),
                                    ev_v.cite("step1/admit_summary.json", "/keep")],
                 "lit_cites": [], "prediction": {"stage": "V5", "stratum": None, "metric": "fn_rate",
                                                 "direction": "above", "threshold": 0.1},
                 "cheapest_test": {"card": "X11"}, "falsifier": "f"}]}
    val = V.validate_da(reply, ev_v, led, BP.load_menu(), {"adversary": ADV, "planner": PLANNER}, domain=str(VEH),
                        claims=jload(FF / "vehicles" / "vehicles_claims.json"))
    check("validate_da drops a counter-argument naming the vehicles exam (test leak)",
          val["ok"] and not val["counter_arguments"][0]["kept"]
          and any("night_exam" in r for r in val["counter_arguments"][0]["reasons"]), val["counter_arguments"])
    print("R13: the weed domain's extra non-decision split (H10d domain dev) is blocked like an exam")
    from weed_optimizer_framework.tools.funnel import domain as FD2
    extra = list(FD2.load(M.DOMAIN).exam_splits().get("extra_non_decision") or [])
    check("(the config) the weed domain names an extra non-decision split", extra == ["domain_dev"], extra)
    split = extra[0] if extra else "domain_dev"
    check("  it is in model.non_dev_exams and every blocked list (evidence, remote, brain_plan)",
          split in M.non_dev_exams() and split in E.BLOCKED_SPLITS and split in RM.BLOCKED_SPLITS
          and split in BP.FORBIDDEN_EXAMS, (M.non_dev_exams(), E.BLOCKED_SPLITS))
    obj2 = {split: {"map50_95": 0.9}, "keep": 1, "rows": [{split: 2, "dev": 3}]}
    evd = E.from_texts({"step1/admit_summary.json": json.dumps(obj2)}, "realloop_v1")
    check("  evidence drops it at any depth", evd.json("step1/admit_summary.json") == {"keep": 1, "rows": [{"dev": 3}]},
          evd.json("step1/admit_summary.json"))
    check("  remote.dev_only drops it", RM.dev_only(obj2)[0] == {"keep": 1, "rows": [{"dev": 3}]})
    check("  brain_plan refuses a digest section carrying it", bool(BP.dev_leaks(obj2)))
    check("  validate's free-text leak pattern names it",
          V.LEAK_TEXT_RE.search("the %s numbers say otherwise" % split) is not None)
    mod = load_domain_free()
    problems = mod.scan()
    check("the domain-free grep test passes on the engine (tests/test_funnel_domain_free.py)", not problems,
          problems[:5])


def load_domain_free():
    import importlib.util
    spec = importlib.util.spec_from_file_location("funnel_domain_free", str(TESTS / "test_funnel_domain_free.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------ R14
def test_r14(root, ev, diags):
    print("R14 the devil's advocate: validation, claims, blindness")
    from weed_optimizer_framework.tools.funnel import claims as FC
    led = ev.json(E.FUNNEL_LEDGER)
    menu = BP.load_menu()
    pos = jload(FF / "da" / "da_positive.json")
    claims = jload(FF / "claims" / "claims_seed.json")
    job = {"schema": V.DA_REPLY_SCHEMA, "ok": True, "reply": pos, "model": "glm-4.7-flash",
           "model_used": "glm-4.7-flash", "digest_sha256": "d" * 64}

    def val(reply, models=None, domain=None, reg=None):
        return V.validate_da(reply, ev, led, menu, models or {"adversary": ADV, "planner": PLANNER},
                             diagnoses=diags, domain=domain, claims=reg if reg is not None else claims)
    v = val(job)
    c = v["counts"]
    check("the positive reply keeps 6 counter-arguments, all counted, and 2 concessions",
          v["ok"] and (c["kept"], c["counted"], c["concessions_kept"], c["dropped"]) == (6, 6, 2, 0),
          (c, [x["reasons"] for x in v["counter_arguments"] if not x["kept"]]))
    check("  from a model of another family, so it may move the claim", v["moves_claims"] is True
          and v["model"]["family"] == "glm" and v["model"]["planner_family"] == "qwen")
    reg = copy.deepcopy(claims)
    surv = sum(1 for x in v["counter_arguments"] if x["kept"] and x["counted"])
    FC.transition(reg, "C1", "challenged", BP.actor_for("adversary/glm-4.7-flash"), "surviving counter-arguments",
                  [x["evidence_cites"][0] for x in v["counter_arguments"]], "adversary",
                  {"valid": True, "surviving": surv, "model": ADV, "planner_model": PLANNER, "same_family": False})
    check("  C1 moves open -> challenged (actor adversary)", FC.get(reg, "C1")["status"] == "challenged")
    filings = V.da_filings(v, menu)
    check("  its tests are filed: menu levers L10 and L12, cards X11",
          sorted({f.get("lever") or f.get("card") for f in filings}) == ["L10", "L12", "X11"], filings)
    sy = val(jload(FF / "da" / "da_sycophantic.json"), reg=reg)
    check("the sycophantic reply keeps 0 (its concessions check nothing) and moves no claim",
          sy["ok"] and sy["counts"]["kept"] == 0 and sy["counts"]["concessions_kept"] == 0
          and sy["moves_claims"] is False and FC.get(reg, "C1")["status"] == "challenged", sy["counts"])
    fab = copy.deepcopy(pos)
    fab["counter_arguments"][0]["evidence_cites"][0]["value"] = [0.5, 0.9]
    vf = val(fab)
    check("a fabricated cite drops its counter-argument (5 kept)", vf["counts"]["kept"] == 5
          and not vf["counter_arguments"][0]["kept"], vf["counter_arguments"][0]["reasons"])
    leak = copy.deepcopy(pos)
    leak["counter_arguments"][1]["argument"] += " The ood23 exam shows it."
    vl = val(leak)
    check("a test leak drops its counter-argument", not vl["counter_arguments"][1]["kept"]
          and vl["counts"]["leaks"] >= 1, vl["counter_arguments"][1]["reasons"])
    echo = copy.deepcopy(pos)
    echo["counter_arguments"][4]["evidence_cites"] = [
        ev.cite("realloop_v1/report.json", "/steps/2/truth/verdict"), ev.cite(E.FUNNEL_LEDGER, "/stages/13/kept")]
    ve = val(echo)
    check("an echo (no artifact beyond D17's own cites) is kept but not counted",
          ve["counter_arguments"][4]["kept"] and not ve["counter_arguments"][4]["counted"]
          and ve["counts"]["counted"] == 5, ve["counter_arguments"][4])
    st = copy.deepcopy(pos)
    st["counter_arguments"][2]["prediction"]["stage"] = "S99"
    st["counter_arguments"][3]["cheapest_test"] = {"lever": "L13", "params": {"policy": "R-A"}}
    vs = val(st)
    check("a prediction on no ledger stage and an L13 test are dropped",
          not vs["counter_arguments"][2]["kept"] and not vs["counter_arguments"][3]["kept"])
    fc = copy.deepcopy(pos)
    fc["stage_forecast"]["S8"] += 0.1
    vfc = val(fc)
    check("an invalid stage forecast (sum 1.1) makes the whole reply invalid", not vfc["ok"]
          and "sums to" in (vfc["invalid_reason"] or ""), vfc["invalid_reason"])
    same = val(job, models={"adversary": "ollama:qwen3.8:27b", "planner": PLANNER})
    check("a same-family reply is recorded but moves no claim", same["ok"] and same["moves_claims"] is False
          and same["model"]["same_family"] is True, same["model"])
    reg2 = copy.deepcopy(claims)
    try:
        FC.transition(reg2, "C1", "challenged", "tier2:adversary/qwen", "x", [pos["counter_arguments"][0]
                                                                             ["evidence_cites"][0]], "adversary",
                      {"valid": True, "surviving": 6, "model": "ollama:qwen3.8:27b", "planner_model": PLANNER,
                       "same_family": True})
        moved = True
    except FC.ClaimsError:
        moved = False
    check("  funnel.claims refuses a same-family adversary's transition", not moved)
    fallback = val(job, models={"adversary": "ollama:gemma4", "planner": "ollama:gemma4"})
    check("both roles on their shared fallback (gemma4) is same-family: it moves nothing",
          fallback["moves_claims"] is False and fallback["model"]["same_family"] is True)

    print("R14: the DA digest is blind")
    contract = ROOT.parent / "docs" / "FUNNEL_AUDIT.md"
    prereg = ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json"
    d17_props = [p for p in LV.propose(diags, ev)["proposals"] if "D17" in p["trigger"]]
    marks = BP.blind_markers(prereg, contract, [FF / "da" / "da_positive.json"], d17_props)
    check("the blind markers: the contract's path and sha256, the prereg's path, sha256 and core sha256, the R14 "
          "fixture's sha256, and D17's proposals' ids and commands",
          "docs/FUNNEL_AUDIT.md" in marks and hashlib.sha256(contract.read_bytes()).hexdigest() in marks
          and "funnel/prereg_v1.json" in marks and hashlib.sha256((FF / "da" / "da_positive.json").read_bytes())
          .hexdigest() in marks and all(p["id"] in marks and " ".join(p["argv"]) in marks for p in d17_props),
          len(marks))
    dg = BP.build_da_digest(ev, claims, led, menu, campaign="c1", n=1, markers=marks, loop="realloop_v1",
                            created_utc="2026-09-28T00:00:00Z")
    txt = json.dumps(dg)
    check("a digest built with them holds none of them, and check_staged accepts it",
          not [m for m in marks if m in txt] and BP.check_staged(dg) == dg["prompt"]
          and dg["schema"] == BP.DA_DIGEST_SCHEMA and dg["claim_ids"] == ["C1", "C2"])
    check("  it shows the claims, the ledger's stage ids with their roles, and the recoverable stages",
          [s["id"] for s in dg["sections"]["stages"]] == [s["id"] for s in led["stages"]]
          and dg["sections"]["recoverable"] == sorted(pos["stage_forecast"], key=lambda s: [s["id"] for s in
                                                      led["stages"]].index(s))
          and "prereg" not in dg["sections"]["evidence"]["funnel/funnel_ledger.json"])
    evil = marks + [led["stages"][0]["filter"]]
    try:
        BP.build_da_digest(ev, claims, led, menu, campaign="c1", n=2, markers=evil, loop="realloop_v1")
        refused = False
    except ValueError as e:
        refused = "blind marker" in str(e)
    check("a digest that would hold a blind marker is refused", refused)
    tampered = copy.deepcopy(dg)
    tampered["sections"]["claims"][0]["text"] += " docs/FUNNEL_AUDIT.md"
    tampered["prompt"] = BP.render_da_prompt(tampered["sections"])
    tampered["prompt_sha256"] = hashlib.sha256(tampered["prompt"].encode("utf-8")).hexdigest()
    tampered["sha256"] = BP.da_digest_sha256(tampered)
    try:
        BP.check_staged(tampered)
        refused = False
    except ValueError as e:
        refused = "blind marker" in str(e)
    check("check_staged refuses a staged digest holding a marker (by its sha256), however consistently re-hashed",
          refused)


# ------------------------------------------------------------ negative controls
def test_negative_controls():
    print("funnel_negative_controls: pilots, b0_v1 and base_b_v1 with the ledger: D17 silent; high yield: D19 silent")
    for name, exps, cur in (("pilot_v1", V1_ERA, "pilot_v1"), ("pilot_v2", WHOLE, "pilot_v2"),
                            ("pilot_v3", WHOLE_V3, "pilot_v3"), ("b0_v1", ["b0_v1"], "b0_v1"),
                            ("base_b_v1", ["base_b_v1"], "base_b_v1")):
        root = legacy_world("nc_" + name, exps, None)
        (root / "step1").mkdir(exist_ok=True)
        for f in ("admit_summary.json", "select_summary.json", "pool_summary.json", "calibration.json"):
            shutil.copyfile(FF / "step1" / f, root / "step1" / f)
        shutil.copyfile(FF / "funnel_ledger_summaries.json", root / "funnel" / "funnel_ledger.json")
        ev = E.load_dir(root, cur, exps=exps)
        diags, by, res = run(ev)
        sig = DG.funnel_signals(ev, DG.load_thresholds())
        check("%s: the signals hold (S1, S4) but D17 is silent: no conclusion" % name,
              sig["S1"]["holds"] and sig["S4"]["holds"] and not by["D17"]["fired"]
              and "no negative conclusion" in by["D17"]["summary"], by["D17"]["summary"])
    root = TMP / "nc_highyield"
    (root / "funnel").mkdir(parents=True)
    shutil.copyfile(FF / "controls" / "highyield_ledger.json", root / "funnel" / "funnel_ledger.json")
    ev = E.load_dir(root, "highyield", exps=[])
    diags, by, res = run(ev)
    sig = by["D19"]["detail"]["signals"]
    check("high yield with out-of-domain known truth: S1 < 1, S4 false, D19 silent",
          sig["S1"]["holds"] is False and sig["S4"]["holds"] is False and not by["D19"]["fired"], by["D19"]["summary"])


def main():
    try:
        root, ev, diags, _res = test_r9()
        test_r9_early()
        test_r9b()
        test_r10()
        test_r11()
        test_r12()
        test_r13()
        test_r14(root, ev, diags)
        test_r15()
        test_negative_controls()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
