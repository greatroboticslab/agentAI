#!/usr/bin/env python3
"""Units of stream mode (docs/CONTINUOUS_LOOP.md 6): the lever menu against
the executor and the policy table, the cluster verbs' parsers and grammar,
the evidence allow-list, the budget windows and job settlement, the R0
records (Stage A, Stage C, the capacity decision), the measurement arms
(m832, s1024, and the box-quality arms y26l640, y26m640, l640: priced,
proposed once R0 is complete and only while missing, never the stream's arm),
E1 (2026-10-03: base v3's build L23V, the arms on splits v3 with cold_budget
priced from the budget, their agnostic rescore L23E; record only; L23V waits,
at most 12 h from when the stream first saw the list, for a person to lift
the quarantines D28 now judges chance, with one card), the dispositions, the replay gate's stream cases, the
config and the campaign dispatch, and the lab runner. No network, no GPU, no
ssh.

Run:  python3 tests/test_stream_ap_units.py
"""
import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import (AP, B, C, DS, E, LS, M, NAME, OWNER, POL, R, S, SR, X, World, check,  # noqa: E402
                                  seg_ledger)

FAILURES = W.FAILURES


def section(title):
    print(title)


# ------------------------------------------------------------------ the menu
PARAMS = {
    "L15": {"config": "/lab/x/collect/domains/weed.json", "classes": "PricklySida,Sicklepod",
            "out": "/lab/x/collect/candidates.json"},
    "L26": {"source": "mfwd/porol", "out": "/lab/x/intake/names/"},
    "LP": {}, "L16": {"source": "mfwd_porol", "max_bytes": 10737418240},
    "L16L": {"source": "gh:abc/def", "max_bytes": 1000, "out": "/lab/x/intake/staging/"},
    "L16R": {"source": "kg:yuzhenlu/cottonweeddet3", "max_bytes": 5180000000}, "L16I": {"source": "mfwd_porol"},
    "L16RL": {"source": "mediatum_1717366", "max_bytes": 1000, "out": "/lab/x/intake/staging/"},
    "L17": {"verb": "admit", "intake": "b0001"},
    "L18": {"pkg": "inc2", "stream": "weed_stream_v1", "k": 2, "exp": "weed_stream_v1_s003", "recipes": "r0,x1a"},
    "L19": {"pkg": "inc2", "exp": "weed_stream_v1_s001"}, "L20": {"pkg": "inc2", "stream": "weed_stream_v1"},
    "L21": {"pkg": "inc2", "stream": "weed_stream_v1", "to": "P_2"},
    "L22": {"pkg": "inc2", "stream": "weed_stream_v1", "m": 1526}, "L23": {"pkg": "inc2", "verb": "lock"},
    "L23B": {"pkg": "inc2", "exp": "b_v2", "manifest": LS.inc_path("splits/v2/base_v2.jsonl"), "seeds": "0,1,2,3,4",
             "arm": "n640", "role": "b_v2"},
    "L23N": {"pkg": "inc2", "exp": "b_v2_m832", "reference": "b_v2_m640"},
    "L23V": {"pkg": "inc2", "stream": "weed_stream_v1"},
    "L23E": {"pkg": "inc2", "exp": "e1_b_m640", "reference": "e1_a_m640"},
    "LV": {"pkg": "inc2", "module": "baseline", "verb": "canary-verdict", "exp": "canary_v2"},
    "LI": {"pkg": "inc2", "stream": "weed_stream_v1", "stage_b": "r0,x1a"},
    "LA": {"pkg": "inc2", "stream": "weed_stream_v1"},
    "LC": {"pkg": "inc2", "exp": "weed_stream_v1_m001"},
    "L24": {"pkg": "inc2", "source": "mfwd_porol", "cite": "D28"},
    "L25": {"pkg": "inc2", "exp": "pilot_v4", "from_exp": "pilot_v3", "recipes": "x1a,x1b"},
    "L27": {"pkg": "inc2", "stream": "weed_stream_v1", "from_pool": "P_1"},
    "L28": {"pkg": "inc2", "stream": "weed_stream_v1", "holdout": "tsw22", "m": 682},
    "LH": {"pkg": "inc2", "stream": "weed_stream_v1", "hold": "funnel_F9"}}


def t_menu():
    section("the stream lever menu against the executor and the policy table")
    menu = LS.load_menu()
    ids = LS.lever_ids(menu)
    check("every stream lever is exercised here", sorted(PARAMS) == sorted(ids) or set(PARAMS) - {"L16S"} ==
          set(ids) - {"L16S"}, sorted(set(ids) ^ set(PARAMS)))
    check("the policy table loads with no errors (the stream rows included)", POL.errors() == [], POL.errors())
    for lid in ids:
        r = LS.row(lid)
        desc = POL.describe(r["policy_action"])
        check("%s: its policy row %s exists with the lever's risk %s" % (lid, r["policy_action"], r["risk"]),
              desc.get("known") and desc.get("risk") == r["risk"], desc.get("reason") or desc.get("risk"))
        check("%s: no brain (tier2) may request it" % lid, "tier2" not in (desc.get("allowed_tiers") or []))
        if lid not in PARAMS:
            continue
        pp = LS.policy_params(lid, PARAMS[lid])
        argv = LS.render(lid, pp)
        ok, bad = LS.check_params(lid, pp)
        check("%s: its parameters are inside the policy row's bounds" % lid, ok, bad)
        rend = X.render(r["policy_action"], pp)
        chk = X.argv_check(rend, argv)
        back = X.params_from_argv(r["policy_action"], argv)
        check("%s: the executor re-renders the lever's argv token for token and reads its params back" % lid,
              chk[0] and back == pp, (chk, back, pp))
    check("an unknown protocol package is refused", _raises(lambda: LS.render("L18", dict(PARAMS["L18"], pkg="inc"))))
    for extra in ({"module": "pilot4", "verb": "verdict", "exp": "pilot_v4"}, {"module": "baseline",
                                                                                "verb": "capacity-verdict"}):
        pp = LS.policy_params("LV", dict(extra, pkg="inc2"))
        back = X.params_from_argv("inc_stream_verdict", LS.render("LV", pp))
        check("LV %s %s: rendered and read back by the executor" % (extra["module"], extra["verb"]), back == pp, back)
    un = dict(PARAMS["L23B"], exp="b0_tsw_v2", role="union", seeds="0,1,2")
    un.pop("manifest")
    un["union"] = ",".join(LS.inc_path(x) for x in ("splits/v2/train_core.jsonl", "splits/v2/tsw22.jsonl",
                                                    "splits/v2/tsw23.jsonl"))
    pp = LS.policy_params("L23B", un)
    ok, bad = LS.check_params("L23B", pp)
    back = X.params_from_argv("inc_build_baseline_v2", LS.render("L23B", pp))
    check("L23B with --union (B0 u tsw): inside the policy bounds and read back", ok and back == pp, (bad, back))
    try:
        from weed_optimizer_framework.tools.inc2 import stream as E2
        for lid in ("L18", "L20", "L21", "L22", "L27", "L28", "LI", "LA", "LC", "L19", "L24", "LH"):
            argv = LS.render(lid, LS.policy_params(lid, PARAMS[lid]))
            ns = E2.build_parser().parse_args(argv[3:])
            check("%s: group E's inc2.stream build_parser() reads its argv (%s)" % (lid, argv[3]), ns.cmd == argv[3], ns)
    except ImportError:
        W.SKIPS.append("inc2.stream absent (group E): the stream argv checked against the contract grammar only")
    try:
        from weed_optimizer_framework.tools.collect import __main__ as CM
        for lid in ("L15", "L26", "L16L", "L16RL", "L16", "L16I", "LP"):
            argv = LS.render(lid, LS.policy_params(lid, PARAMS[lid]))
            args = argv[argv.index("run_inc_collect.sh") + 1:] if "run_inc_collect.sh" in argv else argv[3:]
            ns = CM.build_parser().parse_args(args)
            check("%s: group D's collector parser reads its argv (%s)" % (lid, args[0]), ns.verb == args[0], ns)
            if "run_inc_collect.sh" in argv:
                check("  and run_inc_collect.sh takes the verb", args[0] in CM.JOB_VERBS, args[0])
    except ImportError:
        W.SKIPS.append("collect absent (group D)")
    try:
        from weed_optimizer_framework.tools.inc2 import step1_stream as S1
        for params in ({"verb": "admit", "intake": "i0001_zen_1"}, {"verb": "bootstrap"}, {"verb": "knowntruth"},
                       {"verb": "backfill"}, {"verb": "eval-hits"}):
            argv = LS.render("L17", LS.policy_params("L17", params))
            args = argv[argv.index("run_inc2_stream.sh") + 1:]
            ns = S1.build_parser().parse_args(args)
            check("L17 %s: group C's step1_stream parser reads its argv" % args[0], ns.verb == args[0], ns)
        pp = LS.policy_params("L17", {"verb": "eval-hits"})
        argv = LS.render("L17", pp)
        ok, bad = LS.check_params("L17", pp)
        req = SR.parse_submit("admit", argv[argv.index("run_inc2_stream.sh") + 1:])
        check("L17 eval-hits (D28-v2's sidecars): inside the policy row's bounds, read back by the executor, and "
              "accepted by stream-submit admit with no flags", ok and X.params_from_argv("inc_stream_admit", argv)
              == pp and req["verb"] == "eval-hits" and req["params"] == {}, (bad, argv, req))
        argv = LS.render("L17", LS.policy_params("L17", {"verb": "scan-holds", "hold": "h6_scan"}))
        check("L17 scan-holds: run_inc2_stream.sh's alias of serve-holds (the parser's own verb)",
              S1.build_parser().parse_args(["serve-holds"] + argv[argv.index("scan-holds") + 1:]).hold == ["h6_scan"])
    except ImportError:
        W.SKIPS.append("inc2.step1_stream absent (group C)")
    check("the fixed verb of a shared action cannot be overridden (L18 is 'build')",
          _raises(lambda: LS.policy_params("L18", dict(PARAMS["L18"], verb="fork"))))
    r = X.render("inc_build_segment", LS.policy_params("L22", PARAMS["L22"]), {"child_exp": "x_s001"})
    check("a stream build's remote line is stream-submit build with the child experiment",
          r["remote"][:2] == ["stream-submit", "build"] and "--child-exp" in r["remote"], r["remote"])
    r = X.render("inc_stream_commit", LS.policy_params("L19", PARAMS["L19"]), {"approval_id": "a1"})
    check("a login-node verb's remote line is stream-run <pkg>.stream VERB",
          r["remote"][:3] == ["stream-run", "inc2.stream", "commit"] and r["remote"][-2:] == ["--exp",
                                                                                          "weed_stream_v1_s001"],
          r["remote"])
    check("a lab lever renders local (L15, L26, L16L, L16RL, L16S)", all(X.render(
        LS.row(l)["policy_action"], LS.policy_params(l, PARAMS.get(l, {"source": "a"})))["local"]
        for l in ("L15", "L26", "L16L", "L16RL")) and X.render("inc_stream_sync", {"source": "a"})["local"])
    check("the gated R2 levers and the envelope levers are the contract's (and LI, the stream's creation)",
          LS.gated_r2() == ("L16", "L17", "L24")
          and set(LS.envelope_levers()) == {"L18", "L20", "L21", "L22", "L23B", "L23N", "L23V", "L23E", "L25", "L27",
                                            "L28", "LI"})
    check("the executor's gated actions cover L16 (fetch on the cluster or the lab, intake), L17 and L24",
          set(X.GATED_R2_ACTIONS.values()) == {"L16", "L17", "L24"})
    check("every envelope action of a stream lever is in approvals.ENVELOPE_ACTIONS",
          all(a in AP.ENVELOPE_ACTIONS for v in X.STREAM_ENVELOPE_LEVERS.values() for a in v))
    check("  and the splits build, a pre-check review and a funnel_F9 release are not",
          not {"inc_splits_build", "inc_stream_collect_review", "inc_stream_collect_review_lab",
               "inc_stream_release"} & set(AP.ENVELOPE_ACTIONS))
    check("experiment mode's envelope table is untouched", X.ENVELOPE_LEVERS == {
        "L1": ("inc_build_pilot",), "L2": ("inc_build_realloop",), "L5": ("inc_build_realloop",),
        "L8": ("inc_build_baseline",), "L9": ("inc_build_pilot",)})
    check("a stream build's child_exp must be one of its stream's (<stream>_s|m|c|b<NNN>)", _refused(
        {"policy_action": "inc_build_segment", "params": dict(LS.policy_params("L18", {"pkg": "inc2", "stream": "sx",
                                                                                     "k": 1, "exp": "sx_s001"}),
                                                                 est_gpu_hours=3.0),
         "child_exp": "other_s001"}))
    check("  and the segment --exp names", _refused(
        {"policy_action": "inc_build_segment", "params": dict(LS.policy_params("L18", {"pkg": "inc2", "stream": "sx",
                                                                                     "k": 1, "exp": "sx_s002"}),
                                                                 est_gpu_hours=3.0),
         "child_exp": "sx_s001"}))


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def _refused(req):
    try:
        X._resolve(X._normalize(req), POL.describe(req["policy_action"]),
                   X._new_result(X._normalize(req), W.AUTO, None, time.time()))
    except X.ExecError:
        return True
    return False


# ------------------------------------------------------------------ prices
def t_prices():
    section("prices (contract 5.6)")
    dom, th = LS.load_domain("weed"), LS.load_thresholds()
    at = {"pool_images": 7626, "M": 763}                  # the contract's 5.6 rows (N 7,626, M 763)
    est, d = LS.price("L18", {"pkg": "inc2", "stream": "s", "k": 4}, dom, at)
    exp_h = est - d["build_job_hours"]
    check("segment K=4, one recipe, truth on, N 7,626, M 763: %.1f GPU-h in 33-37 (+ the 4 h build job)" % exp_h,
          33.0 <= exp_h <= 37.5, d)
    check("  base 3 cold seeds 3.8-4.5, chain step 3.0, truth step 4.2-4.9",
          3.8 <= d["base"] <= 4.5 and 2.9 <= d["chain_per_step"] <= 3.05 and 4.2 <= d["truth_per_step"] <= 4.95, d)
    est2, d2 = LS.price("L18", {"pkg": "inc2", "stream": "s", "k": 4}, dom, dict(at, truth_every=2))
    check("truth every 2nd step costs less", est2 < est and d2["truth_steps"] == 2, d2)
    check("L-4: YOLO11n at 640 keeps the truth arm on every step", LS.truth_every(dom, th, 7626, 763) == 1)
    ev = LS.truth_every(dom, th, 7626, 763, ("r0",), 4.0)
    chain, truth = LS.step_hours(dom, 7626, 763, ("r0",), 4.0)
    b1, _ = LS.price("L23B", PARAMS["L23B"], dom, {"images": 6813})
    b2, _ = LS.price("L23B", dict(PARAMS["L23B"], exp="b_v2_s640", arm="s640", role="capacity"), dom, {"images": 6813})
    check("a capacity arm's baseline is priced by its cost factor (s640 at 2x n640)",
          1.9 * (b1 - 4.8) < b2 - 4.8 < 2.1 * (b1 - 4.8), (b1, b2))
    b3, d3 = LS.price("L23B", dict(PARAMS["L23B"], exp="b_v2_m832", seeds="0,1,2", arm="m832", role="capacity"), dom,
                      {"images": 6813})
    b4, d4 = LS.price("L23B", dict(PARAMS["L23B"], exp="b_v2_s1024", seeds="0,1,2", arm="s1024", role="capacity"), dom,
                      {"images": 6813})
    cold3 = LS._hours(3 * 6813, 100, LS.cost(dom, "cold_ms_per_image_epoch"))
    check("the measurement arms are priced by capacity.measure_arms: m832 x4.0 (%.1f GPU-h), s1024 x3.5 (%.1f GPU-h), "
          "3 seeds with finals and the build job" % (b3, b4),
          d3["factor"] == 4.0 and d4["factor"] == 3.5 and abs(b3 - (4.0 * cold3 + 0.8 + 4.0)) < 1e-3
          and abs(b4 - (3.5 * cold3 + 0.8 + 4.0)) < 1e-3, (d3, d4))
    check("  an arm in neither table is refused a price", _raises(lambda: LS.arm_factor(dom, "x9")))
    check("  the grid's est. factors (s640 2x, m640 3x) keep the truth arm on every step at N 7,626, M 763",
          LS.truth_every(dom, th, 7626, 763, ("r0",), 2.0) == 1 and LS.truth_every(dom, th, 7626, 763, ("r0",), 3.0) == 1)
    check("  a 4x arm (a measured rate can exceed the est. factors): one step with truth costs %.1f > 25 GPU-h -> "
          "every ceil(cost/25) = %d-th step" % (chain + truth, ev), ev == -(-(chain + truth) // 25) and ev > 1,
          (chain + truth, ev))
    m, dm = LS.price("L20", {"pkg": "inc2", "stream": "s"}, dom, {"pool_images": 9200})
    check("a milestone at 9.2K images: 8-9.5 GPU-h est. at 6-7 ms (here the 7 ms upper rate: %.2f) (+ build)"
          % (m - dm["build_job_hours"]), 8.0 <= m - dm["build_job_hours"] <= 9.8, dm)
    c, dc = LS.price("L28", {"pkg": "inc2", "stream": "s", "holdout": "tsw22", "m": 763}, dom, {"pool_images": 7626})
    check("Stage C with one recipe: 10.6-11.8 GPU-h est. (+ build)", 10.0 <= c - dc["build_job_hours"] <= 12.5, dc)


# ------------------------------------------------------------------ remote parsers and grammar
PROJECTS = """Your default charging project charge ID is cis240145p
Project: CIS240145P
  PI: someone
  Title: something
  Resource: Bridges-2 GPU
    Allocation: 20,000.00
    Balance: 10,529.12
    End Date: 2026-12-31
  Resource: Bridges-2 Regular Memory
    Allocation: 50,000.00
    Balance: 49,000.00
    End Date: 2026-12-31
"""
QUOTA = """Filesystem  Project      Used      Quota
/ocean      cis240145p   6.44T     7.00T
"""


def t_remote():
    section("the cluster's stream verbs")
    r = SR.parse_projects(PROJECTS)
    check("projects: each resource block's balance and end date", len(r) == 2 and r[0]["balance_su"] == 10529.12
          and r[0]["end_date"] == "2026-12-31" and "GPU" in r[0]["resource"], r)
    u, q = SR.parse_quota(QUOTA)
    check("the quota line: used and quota in GB (6.44T of 7.00T)", abs(u - 6440) < 1 and abs(q - 7000) < 1, (u, q))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="chain_", dir=str(W.TMP)))
    p = tmp / "ledger.jsonl"
    raw = b""
    for i in range(3):
        line = (json.dumps({"event": "e%d" % i, "prev_sha256": hashlib.sha256(raw).hexdigest()}) + "\n").encode()
        raw += line
    p.write_bytes(raw)
    check("the stream ledger's hash chain verifies", SR.chain_check(p)[0] is True)
    p.write_bytes(raw.replace(b'"e0"', b'"eX"'))
    check("  and an edited line breaks it", SR.chain_check(p)[0] is False, SR.chain_check(p))
    for kind, args, ok in (("collect", ["fetch", "--source", "a", "--max-bytes", "100"], True),
                           ("collect", ["fetch", "--source", "a"], False),
                           ("collect", ["fetch", "--source", "a/../b", "--max-bytes", "1"], False),
                           ("collect", ["rm", "-rf"], False),
                           ("admit", ["admit", "--intake", "b1"], True),
                           ("admit", ["scan-holds", "--hold", "funnel_F9"], True),
                           ("admit", ["scan-holds", "--hold", "x;y"], False),
                           ("build", ["inc2.stream", "build", "--stream", "s", "--k", "4"], True),
                           ("build", ["inc2.stream", "commit", "--exp", "e"], False),
                           ("build", ["inc2.splits", "lock"], True),
                           ("build", ["inc2.baseline", "build", "--exp", "b", "--manifest", "/x/a.jsonl", "--seeds",
                                      "0,1", "--arm", "n640", "--role", "b_v2"], True),
                           ("build", ["inc2.baseline", "build", "--exp", "b", "--manifest", "/x/a.jsonl", "--seeds",
                                      "0,1"], False),
                           ("build", ["inc2.baseline", "build", "--exp", "b", "--manifest", "/x/a.jsonl",
                                      "--union", "/x/a.jsonl,/x/b.jsonl", "--seeds", "0,1", "--arm", "n640",
                                      "--role", "union"], False),
                           ("build", ["inc2.baseline", "build", "--exp", "b", "--manifest", "/x/a.jsonl", "--seeds",
                                      "0,1", "--arm", "x9", "--role", "b_v2"], False),
                           ("build", ["inc2.baseline", "build", "--exp", "b_v2_m832", "--manifest", "/x/a.jsonl",
                                      "--seeds", "0,1,2", "--arm", "m832", "--role", "capacity"], True),
                           ("build", ["inc2.baseline", "build", "--exp", "b_v2_s1024", "--manifest", "/x/a.jsonl",
                                      "--seeds", "0,1,2", "--arm", "s1024", "--role", "capacity"], True),
                           ("build", ["inc2.baseline", "build", "--exp", "b", "--manifest", "/x/a.jsonl", "--seeds",
                                      "0,1,2", "--arm", "m1024", "--role", "capacity"], False),
                           ("build", ["inc2.stream", "init", "--stream", "s", "--stage-b", "r0,x1a"], True),
                           ("build", ["inc2.stream", "build", "--stream", "s", "--k", "4", "--exp", "s_s001"], True),
                           ("build", ["inc2.stream", "build", "--stream", "s", "--k", "4", "--arch", "yolo11s"], False),
                           ("build", ["inc2.stream", "build", "--stream", "s"], False),
                           ("build", ["inc2.stream", "build", "--stream", "s", "--k", "4", "--extra", "1"], False),
                           ("rm", [], False)):
        try:
            SR.parse_submit(kind, args)
            got = True
        except R.Refused:
            got = False
        check("stream-submit %s %s is %s" % (kind, " ".join(args), "admitted" if ok else "refused"), got == ok)
    w = World("remote_units")
    w._activate()
    rec = SR.stream_submit("collect", ["fetch", "--source", "a", "--max-bytes", "100"], {"trigger": "D20"}, dry_run=True)
    check("a dry run: sbatch --parsable --job-name -p GPU-shared <script>", rec.get("ok") and rec["sbatch_argv"][3:5]
          == ["-p", "GPU-shared"] and rec["sbatch_argv"][2] == "--job-name=inc_stream_collect_fetch_a", rec)
    w.qos = True
    rec = SR.stream_submit("collect", ["fetch", "--source", "a", "--max-bytes", "100"], {})
    check("an 'Invalid qos' refusal comes back as error_kind qos", rec.get("ok") is False and rec.get("error_kind") ==
          "qos", rec)
    w.qos = False
    w.squeue.append({"id": "9", "name": "inc_stream_collect_fetch_a", "state": "RUNNING"})
    rec = SR.stream_submit("collect", ["fetch", "--source", "a", "--max-bytes", "100"], {})
    check("  a second submission of the same job name is refused", rec.get("ok") is False and "already queued" in
          rec.get("error", ""), rec)
    rec = SR.stream_run("inc2.stream", "delete", ["--exp", "e"])
    check("stream-run refuses a verb outside its list", rec.get("ok") is False, rec)
    rec = SR.stream_run("inc2.splits", "lock", [])
    check("  and a module outside stream, baseline and pilot4", rec.get("ok") is False, rec)
    rec = SR.stream_run("inc2.baseline", "build", ["--exp", "x"])
    check("  and a build of a verdict module (builds are jobs)", rec.get("ok") is False, rec)
    rec = SR.stream_run("inc2.stream", "rollback", ["--stream", "s", "--to", "../x"])
    check("  and a rollback target that is not a pool id", rec.get("ok") is False, rec)
    # the real subprocess path of stream-run: a fake <pkg>.stream CLI
    SR._run = W._REAL_RUN
    SR.subprocess = subprocess
    fake = W.TMP / "fake_stream_cli.py"
    fake.write_text("import sys, json, os\nprint('noise')\nprint(json.dumps({'ok': True, 'argv': sys.argv[1:], "
                    "'who': os.environ.get('INCAP_DECIDED_BY')}))\n")
    os.environ["INCAP_STREAM_RUN"] = "%s %s" % (sys.executable, fake)
    try:
        rec = SR.stream_run("inc2.stream", "commit", ["--exp", "s_s001"], {"approval_id": "a1", "decided_by": OWNER})
    finally:
        os.environ.pop("INCAP_STREAM_RUN", None)
    check("stream-run runs the verb as a subprocess and reads its JSON result", rec.get("ok") and
          (rec.get("result") or {}).get("argv")[-3:] == ["commit", "--exp", "s_s001"], rec)
    check("  a person's approval reaches inc2.stream as INCAP_DECIDED_BY", (rec.get("result") or {}).get("who") == OWNER,
          rec.get("result"))
    busy = W.TMP / "busy_stream_cli.py"
    busy.write_text("import sys\nsys.stderr.write('[inc2.stream] BUSY: lease\\n')\nsys.exit(3)\n")
    os.environ["INCAP_STREAM_RUN"] = "%s %s" % (sys.executable, busy)
    try:
        rec = SR.stream_run("inc2.stream", "compare", ["--exp", "s_m001"], {})
    finally:
        os.environ.pop("INCAP_STREAM_RUN", None)
    check("  exit 3 (another writer holds stream.lease) is error_kind 'busy', a transient",
          rec.get("ok") is False and rec.get("error_kind") == "busy", rec)
    check("  and records who authorised it in the actions log",
          any(json.loads(x).get("approval_id") == "a1" for x in (w.inc / "_campaign" / "provenance" /
                                                                  "_actions.jsonl").read_text().splitlines()))
    hashes = SR.module_hashes()
    check("the module hashes cover the stream's own modules and data (S23)", {
        "tools/inc_autopilot/stream.py", "tools/inc_autopilot/diagnose_stream.py", "tools/inc_autopilot/levers_stream.py",
        "tools/inc_autopilot/stream_remote.py", "tools/inc_autopilot/stream_levers.json",
        "tools/inc_autopilot/stream_domains/weed.json"} <= set(hashes), sorted(hashes))
    check("remote.dispatch routes the stream verbs", R.dispatch(["stream-run", "inc2.stream", "delete", "--"])
          .get("verb") == "stream-run")


# ------------------------------------------------------------------ evidence
def t_evidence():
    section("the evidence allow-list (3.8)")
    for n in ("stream/weed_stream_v1/queue_summary.json", "stream/weed_stream_v1/ledger.jsonl",
              "stream/weed_stream_v1/dev_scores.json", "step1_stream/status.json", "intake/b0001/summary.json",
              "intake/sources.json", "intake/placement.json", "splits/v2/lock_status.json",
              "capacity/capacity_v1.json", "canary_v2/canary.json", "pilot_v4/stage_a.json",
              "intake/i0003_rf_x/eval_hits.json"):
        check("allowed: %s" % n, E.allowed(n))
    for n in ("stream/x/pool/P_1.jsonl", "intake/staging/a/b.jpg", "splits/v2/LOCK.json", "stream/x/scores/test.json",
              "step1_stream/queue/queue.jsonl", "capacity/capacity_v1_report.json", "capacity/capacity_v1_report.md",
              "stream/x/milestones/m001/research_log_entry.md", "stream/stage_a.json",
              "intake/i0003_rf_x/images/a.jpg", "step1_stream/eval_hits/b0000.json",
              "step1_stream/eval_hits/b0000/x.json", "intake/eval_hits.json"):
        check("refused: %s" % n, not E.allowed(n))
    check("stream, step1_stream and intake are not experiments", {"stream", "step1_stream", "intake"} <=
          set(E.RESERVED_DIRS) and not E.allowed("stream/exp.json"))
    led = "".join(json.dumps(x) + "\n" for x in ({"event": "commit", "exp": "s1", "accepted": ["a"]},
                                                  {"event": "milestone", "test": {"mean": 0.85}, "dev": {"mean": 0.8}}))
    ev = E.from_texts({"stream/s/ledger.jsonl": led}, "s")
    art = ev.json("stream/s/ledger.jsonl")
    check("a stream ledger is a list artifact, scrubbed of every non-decision exam",
          isinstance(art, list) and art[0]["event"] == "commit" and "test" not in art[1] and art[1]["dev"]["mean"] == 0.8,
          art)
    check("  and it is not an experiment's ledger", not ev.ledgers)


# ------------------------------------------------------------------ budget
def t_budget():
    section("the budget (6.6)")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="budget_", dir=str(W.TMP)))
    base = str(tmp)
    now = W.T0 + 5 * 86400.0                # 2026-10-06: every charge below is in October
    execs = [{"campaign": NAME, "action": "inc_stream_collect", "params": {"source": "a"}, "charged": True,
              "est_su": 4.0, "epoch": now - 3600, "ts": W.utc(now - 3600), "status": "executed", "run_id": "r1",
              "job_ids": ["701"]},
             {"campaign": NAME, "action": "inc_build_segment", "params": {"verb": "build", "stream": "s"},
              "child_exp": "s_s001", "charged": True, "est_su": 40.0, "epoch": now - 60, "ts": W.utc(now - 60),
              "status": "executed", "run_id": "r2", "job_ids": ["702"]},
             {"campaign": "weed_inc_v1", "action": "inc_build_realloop", "params": {"exp": "rl9"}, "charged": True,
              "est_su": 30.0, "epoch": now - 7200, "ts": W.utc(now - 7200), "status": "executed", "run_id": "r3"}]
    camp = {"name": NAME, "mode": "stream", "envelope_su": 1000.0, "daily_cap_su": 120.0, "window_cap_su": 350.0}
    st = B.state(camp, execs, {"su_envelope": 1500, "daily_cap": 120}, now, base_dir=base)
    check("the stream campaign's committed estimates: its collect job and its segment build", st["committed_su"] == 44.0,
          st)
    check("the monthly window counts this month's charges (44 of 350)", st["window_su"] == 44.0
          and st["window_remaining_su"] == 306.0, st)
    check("the domain cap sums every campaign's inc:* spend and commitments (44 + 30 of 1500)",
          st["domain_inc_su"] == 74.0 and st["domain_remaining_su"] == 1426.0, st)
    r = B.record_job_spend("701", NAME, {"state": "COMPLETED", "elapsed_s": 1800.0, "gpu_count": 1, "gpu_type": "v100-32"},
                           "inc_stream_collect", base_dir=base, ts=W.utc(now))
    st2 = B.state(camp, execs, {"su_envelope": 1500, "daily_cap_su": 120}, now, base_dir=base)
    check("a job settled from sacct releases its estimate and charges the measured SU (0.5)",
          r["ok"] and st2["committed_su"] == 40.0 and st2["spent_su"] == 0.5, (r, st2))
    B.record_report_spend({"exp": "s_s001", "done": True, "done_utc": W.utc(now), "gpu_hours": {"base": {"hours": 30.0,
                                                                                                     "runs": 3}}},
                          NAME, base_dir=base)
    st3 = B.state(camp, execs, None, now, base_dir=base)
    check("a stream build's estimate is released by its child experiment's report spend (child_exp)",
          st3["committed_su"] == 0.0 and st3["spent_su"] == 30.5, st3)
    ok, why = B.fits(dict(st3, window_remaining_su=5.0), 10.0, need_daily=True)
    check("fits refuses past this month's window", not ok and any("month" in x for x in why), why)
    ok, why = B.fits(dict(st3, domain_remaining_su=5.0), 10.0)
    check("  and past the domain's cap", not ok and any("domain" in x for x in why), why)
    exp_st = B.state({"name": "weedinc", "envelope_su": 300}, [], None, now, base_dir=base)
    check("an experiment campaign's budget state has no stream keys", not {"window_su", "domain_inc_su"} & set(exp_st))
    # the build job itself and a fork (a build job with no experiment) are settled from sacct too
    execs2 = execs + [{"campaign": NAME, "action": "inc_build_segment", "params": {"verb": "fork", "stream": "s", "m": 2},
                       "child_exp": None, "charged": True, "est_su": 4.0, "epoch": now - 30, "ts": W.utc(now - 30),
                       "status": "executed", "run_id": "r4", "job_ids": ["704"]}]
    before = B.state(camp, execs2, None, now, base_dir=base)["committed_su"]
    B.record_job_spend("704", NAME, {"state": "COMPLETED", "elapsed_s": 600.0, "gpu_count": 1, "gpu_type": "v100-32"},
                       "inc_build_segment", base_dir=base, ts=W.utc(now))
    after = B.state(camp, execs2, None, now, base_dir=base)
    check("a fork's estimate (a build job with no experiment) is released by its sacct settlement",
          before == 4.0 and after["committed_su"] == 0.0, (before, after["committed_su"]))
    check("  the domain's committed SU (what the allocation balance does not yet show) is reported apart from its "
          "spent SU (D27): weed_inc_v1's open 30 SU, beside 30.67 spent", after["domain_committed_su"] == 30.0
          and abs(after["domain_inc_su"] - (after["domain_committed_su"] + 30.666667)) < 1e-6, after)
    rates = B.partition_rates()
    check("su_rates.json declares every partition the loop uses (GPU-shared, GPU, RM-shared)",
          {"GPU-shared", "GPU", "RM-shared"} <= set(rates) and rates["GPU-shared"]["su_per_gpu_hour"] == 1.0, rates)


# ------------------------------------------------------------------ R0 records, dispositions
def t_records():
    section("dispositions and the R0 records")
    for verdict, pd, failed, tv, want in (("ACCEPT", 0.9, [], None, "accepted"), ("REJECT", 0.1, [], None, "data"),
                                          ("REJECT", 0.1, ["species"], "helps", "species"),
                                          ("REJECT", 0.5, ["flips"], None, "flips"),
                                          ("REJECT", 0.5, ["regression"], None, "recipe"),
                                          ("REJECT", 0.5, ["regression", "species"], None, "species"),
                                          ("HOLD", 0.5, [], None, "hold"), ("REJECT", 0.25, ["regression"], None, "data")):
        e = W.gate("s", verdict, pd, failed)
        e["decision"]["attribution"]["blame"] = "data"
        check("%s P_data %.2f failed %s truth %s -> %s" % (verdict, pd, failed, tv, want),
              DS.disposition(e, tv) == want, DS.disposition(e, tv))
    e = W.gate("s", "REJECT", 0.3, ["regression"], p_reject=0.35)
    check("the decision's own recorded p_reject is used (0.35)", DS.disposition(e) == "data")
    w = World("records")
    w.ready_r0()
    w.tick(2)
    st = w.state()
    check("Stage A READY as inc2.pilot4 verdict recorded it (stage_a.json): segment 1 runs r0 and x1a",
          st["stage_a"]["ready"] and st["stage_a"]["chosen"] == "x1a" and st["stage_a"]["recipes"] == "r0,x1a",
          st.get("stage_a"))
    check("  its record is written under the stream rules version", pathlib.Path(st["stage_a"]["path"]).is_file()
          and DS.rules_version() in st["stage_a"]["path"])
    check("Stage C as the stream read it (the ledger's feasibility read): M feasible", st["stage_c"]["feasible"],
          st.get("stage_c"))
    check("the capacity decision as capacity-verdict recorded it: n640 (no arm qualifies)",
          st["capacity"]["chosen"] == "n640" and st["capacity"]["chosen_exp"] == "b_v2", st.get("capacity"))
    run = W.S.StreamRun(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log, w.hooks,
                        lambda: W.RES, None, None, None)
    run.st, run.dom, run.th = w.state(), LS.load_domain("weed"), LS.load_thresholds()
    check("  R0 is READY for the TRAIN lane (nothing missing)", run._train_ready() == "", run._train_ready())
    w2 = World("records2")
    w2.capacity_choice = "s640"
    w2.stage_a_recipes = ["r0"]
    w2.stage_c_plan = {"m_feasible": False, "species_only_reject": True, "d33_prospective": ["PricklySida"],
                       "per_recipe": {"r0": {"verdict": "REJECT", "species_only_reject": True}}}
    w2.ready_r0()
    w2.tick(3)
    st2 = w2.state()
    check("an arm the decision chose (s640) is the stream's; its cost factor prices the segments",
          st2["capacity"]["chosen"] == "s640" and st2["capacity"]["arm"]["cost_factor"] == 2.0, st2.get("capacity"))
    check("no Stage A survivor: segment 1 runs r0 alone", st2["stage_a"]["recipes"] == "r0", st2.get("stage_a"))
    check("Stage C REJECTed on the species guard alone -> D33 prospectively, card X17 to the owner before R2",
          st2["stage_c"]["species_only_reject"] and any(c.get("lever") == "X17" for c in st2["cards"]), st2["stage_c"])
    pro = [e for e in w2.events("proposed") if e.get("lever") == "L18"]
    check("  (no queue yet: no segment is proposed)", not pro)
    w3 = World("records3")
    w3.canary_pass = False
    w3.ready_r0()
    w3.queue(4 * w3.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w3.t[0] - 2 * 86400.0))
    w3.tick(3)
    d = {x["id"]: x for x in json.loads(S.StreamPaths(str(w3.lab), "weed").diagnoses(NAME).read_text())["diagnoses"]}
    check("a failed canary (canary.json passed false) -> DCAN: TRAIN holds, a person reads it; no segment",
          d["DCAN"]["fired"] and "DCAN" in str(w3.lane("TRAIN").get("diag_hold"))
          and not [e for e in w3.events("proposed") if e.get("lever") == "L18"], (d["DCAN"]["summary"],
                                                                                  w3.lane("TRAIN")))


def _measure_world(tag, stage_c=True):
    """A world with every R0 prerequisite but the measurement arms' experiments
    (E1's arms, which require base v3, stay built: t_e1)."""
    w = World(tag)
    w.ready_r0(stage_c=stage_c)
    for b in w.dom["baselines"]["items"]:
        if b.get("measure") and not b.get("requires"):
            shutil.rmtree(str(w.inc / b["exp"]))
    return w


def t_measure():
    section("the measurement arms (m832, s1024 2026-09-30; y26l640, y26m640, l640 2026-10-01): proposed once R0 is "
            "complete, never the stream's arm")
    from weed_optimizer_framework.tools.inc2 import recipes as RC
    dom = LS.load_domain("weed")
    meas = [b for b in dom["baselines"]["items"] if b.get("measure") and not b.get("requires")]
    check("the domain's measurement baselines are inc2.recipes' measurement arms: capacity builds on base_v2, 3 seeds, "
          "not required; the decision's arms are L-4's grid",
          sorted(b["arm"] for b in meas) == sorted(RC.MEASURE_ARMS)
          and [b["exp"] for b in meas] == ["b_v2_m832", "b_v2_s1024", "b_v2_y26l640", "b_v2_y26m640", "b_v2_l640"]
          and all(b["role"] == "capacity" and b["seeds"] == "0,1,2" and not b.get("required")
                  and b["manifest"] == "splits/v2/base_v2.jsonl" for b in meas)
          and sorted(dom["capacity"]["arms"]) == sorted(RC.GRID_ARMS)
          and sorted(dom["capacity"]["measure_arms"]) == sorted(RC.MEASURE_ARMS),
          [(b["id"], b["arm"]) for b in meas])
    w0 = _measure_world("measure0", stage_c=False)
    w0.tick(2)
    mp = [e.get("child_exp") for e in w0.events("proposed") if e.get("lever") == "L23B"]
    check("before R0 is complete (Stage C not built) they are not proposed: MAINT takes L28 first",
          not mp and "L28" in [e.get("lever") for e in w0.events("proposed")],
          [e.get("lever") for e in w0.events("proposed")])
    w1 = _measure_world("measure1", stage_c=False)
    w1.stage_c_built(done=False)
    w1.tick(3)
    d1 = {x["id"]: x for x in json.loads(S.StreamPaths(str(w1.lab), "weed").diagnoses(NAME).read_text())["diagnoses"]}
    check("  nor while Stage C runs unread (nothing else of R0 due, MAINT idle): R0 completes first",
          not [e for e in w1.events("proposed") if e.get("lever") == "L23B"] and not d1["DR0"]["fired"]
          and w1.lane("MAINT").get("item") is None, ([e.get("lever") for e in w1.events("proposed")],
                                                     d1["DR0"]["summary"]))
    w = _measure_world("measure")
    cap0 = (w.inc / "capacity" / "capacity_v1.json").read_bytes()
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * 86400.0))
    w.tick(3)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23B"]
    first = pro[0] if pro else {}
    ptrs = sorted(c.get("pointer") for c in first.get("cites") or [])
    est, _d = LS.price("L23B", dict(PARAMS["L23B"], exp="b_v2_m832", seeds="0,1,2", arm="m832", role="capacity"),
                       dom, {"images": dom["increment"]["base_images"]})
    check("R0 complete and b_v2_m832 missing: DR0 proposes it (L23B --arm m832 --role capacity), priced x4.0",
          [e.get("child_exp") for e in pro] == ["b_v2_m832"]
          and first.get("argv", [])[-4:] == ["--arm", "m832", "--role", "capacity"]
          and abs(float(first.get("est_gpu_hours") or 0) - est) < 1e-6, [(e.get("child_exp"), e.get("argv", [])[-6:])
                                                                          for e in pro])
    check("  citing only the lock and its own state (a stage that changes with every segment would void the "
          "envelope grant)", ptrs == ["/stage/baselines/cap_m832", "/stage/lock"], first.get("cites"))
    run = W.S.StreamRun(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log, w.hooks,
                        lambda: W.RES, None, None, None)
    run.st, run.dom, run.th = w.state(), LS.load_domain("weed"), LS.load_thresholds()
    check("  the TRAIN lane does not wait for them: R0 READY with both missing, and the segment is cut (L18)",
          run._train_ready() == "" and any(e.get("lever") == "L18" for e in w.events("proposed")),
          (run._train_ready(), [e.get("lever") for e in w.events("proposed")]))
    # built and running: a done arm is rescored at its own resolution next (L23N, t_native)
    w.experiment("b_v2_m832", done=False)
    w.job_done("inc_build_b_v2_m832")
    w.tick(3)
    pro = [e.get("child_exp") for e in w.events("proposed") if e.get("lever") == "L23B"]
    check("once b_v2_m832 exists, b_v2_s1024 is proposed (--arm s1024)", pro == ["b_v2_m832", "b_v2_s1024"]
          and [e for e in w.events("proposed") if e.get("lever") == "L23B"][-1].get("argv", [])[-4:]
          == ["--arm", "s1024", "--role", "capacity"], pro)
    w.experiment("b_v2_s1024", done=False)
    w.job_done("inc_build_b_v2_s1024")
    # the box-quality arms (2026-10-01) follow, in the domain's order, each once its predecessor exists
    for b in meas[2:]:
        w.tick(3)
        last = [e for e in w.events("proposed") if e.get("lever") == "L23B"][-1]
        check("then %s is proposed (--arm %s)" % (b["exp"], b["arm"]),
              last.get("child_exp") == b["exp"] and last.get("argv", [])[-4:] == ["--arm", b["arm"], "--role", "capacity"],
              (last.get("child_exp"), last.get("argv", [])[-4:]))
        it = w.lane("MAINT").get("item") or {}
        if it.get("status") == "filed":
            why = [e.get("reasons") for e in w.events("filed") if e.get("lever") == "L23B"][-1]
            check("  past today's cap (120 SU) it is filed for a person, not run", "today's cap" in str(why), why)
            aid = it["approval_id"]
            res = AP.decide(w.domain, aid, "approve", OWNER, "test: past today's cap", w.clock(), root=str(w.lab))
            w.tick(2)
            it2 = w.lane("MAINT").get("item") or {}
            check("  a person approves it: it waits for today's cap with the approval open, no failed step",
                  isinstance(res, dict) and res.get("ok") and it2.get("approval_id") == aid
                  and it2.get("status") == "filed" and not int(w.lane("MAINT").get("fails") or 0)
                  and not [e for e in w.events("refused") if e.get("lever") == "L23B"], (it2.get("status"), res))
            w.advance(86400.0)
            w.tick(2)
            ex = [e for e in w.events("executed") if e.get("lever") == "L23B"]
            check("  and runs under that approval once the UTC day turns",
                  ex and ex[-1].get("child_exp") == b["exp"] and ex[-1].get("approval_id") == aid,
                  [(e.get("child_exp"), e.get("approval_id")) for e in ex])
        w.experiment(b["exp"], done=False)
        w.job_done("inc_build_%s" % b["exp"])
    w.tick(3)
    pro = [e.get("child_exp") for e in w.events("proposed") if e.get("lever") == "L23B"]
    d = {x["id"]: x for x in json.loads(S.StreamPaths(str(w.lab), "weed").diagnoses(NAME).read_text())["diagnoses"]}
    check("once all exist, none is proposed again (DR0 is silent)", pro == [b["exp"] for b in meas]
          and not d["DR0"]["fired"], (pro, d["DR0"]["summary"]))
    arms = [e for e in w.stream_ledger() if e.get("event") == "arm"]
    check("the stream's arm is unchanged throughout: the capacity decision's n640, one arm line, no LA, "
          "capacity_v1.json untouched",
          w.state()["capacity"]["chosen"] == "n640" and len(arms) == 1
          and not [e for e in w.events("proposed") if e.get("lever") in ("LA", "LV")]
          and (w.inc / "capacity" / "capacity_v1.json").read_bytes() == cap0, (w.state().get("capacity"), arms))
    wd = _measure_world("measure_data")
    wd.step1_status(one_time=("bootstrap", "knowntruth"))
    wd.tick(3)
    check("  nor while a DATA item of R0 is due (Stage C read, the L17 backfill not run): R0 completes first",
          not [e for e in wd.events("proposed") if e.get("lever") == "L23B"]
          and any(e.get("lever") == "L17" and "backfill" in (e.get("argv") or []) for e in wd.events("proposed")),
          [(e.get("lever"), (e.get("argv") or [])[-3:]) for e in wd.events("proposed")])
    # a measurement arm that fails: a card, never a stop of the stream (the TRAIN lane and the envelope run on)
    wf = _measure_world("measure_fail")
    wf.queue(4 * wf.M, boxes={"Purslane": 900}, oldest_utc=W.utc(wf.t[0] - 2 * 86400.0))
    wf.tick(3)
    wf.experiment("b_v2_m832", done=False, blocked={"base:s0": {"cause": {"kind": "failed_run"},
                                                               "error": "2 failed attempts (CUDA out of memory)"}})
    wf.job_done("inc_build_b_v2_m832")
    wf.tick(3)
    st = wf.state()
    h5 = [d for d in st.get("health") or [] if d.get("id") == "D5" and d.get("exp") == "b_v2_m832"]
    ex = [(e.get("child_exp"), e.get("basis")) for e in wf.events("executed") if e.get("lever") == "L23B"]
    check("a failed_run block of b_v2_m832 (not transient) pauses nothing: the campaign stays enabled, one card, its "
          "D5 carries no OP_PAUSE (no stop-loss refuses a grant), and b_v2_s1024 is still granted within the envelope",
          not st.get("paused") and wf.config().get("enabled") is True
          and [c["title"] for c in st.get("cards") or []] == ["Measurement arm b_v2_m832: D5"]
          and len(h5) == 1 and "OP_PAUSE" not in h5[0]["levers"] and h5[0]["detail"].get("record_only") is True
          and ex == [("b_v2_m832", "envelope"), ("b_v2_s1024", "envelope")]
          and wf.lane("TRAIN").get("item", {}).get("status") in ("executed", "running"),
          (st.get("paused"), [c["title"] for c in st.get("cards") or []], h5, ex))
    wf.advance(3 * 3600)
    wf.tick(3)
    st = wf.state()
    check("  its stale advance (D6, generation unchanged for over 2 h) is one more card, not a pause",
          not st.get("paused") and wf.config().get("enabled") is True
          and sorted(c["title"] for c in st.get("cards") or []) == ["Measurement arm b_v2_m832: D5",
                                                                     "Measurement arm b_v2_m832: D6"]
          and len(wf.events("measure_health")) == 2, [c["title"] for c in st.get("cards") or []])
    wf.experiment("%s_s001" % wf.sid, typ="chain", done=False,
                  blocked={"base:s0": {"cause": {"kind": "failed_run"}, "error": "2 failed attempts"}})
    wf.tick(2)
    st = wf.state()
    check("  the same block on a segment of the stream still pauses it (D5 -> OP_PAUSE)",
          "%s_s001 is not transient (D5)" % wf.sid in str((st.get("paused") or {}).get("reason"))
          and wf.config().get("enabled") is False, st.get("paused"))


def _native_world(tag, stage_c=True):
    """R0 complete (Stage C read; stage_c False: built and running unread) and both measurement arms done, their
    native-resolution scores missing."""
    w = World(tag)
    w.ready_r0(stage_c=stage_c)
    if not stage_c:
        w.stage_c_built(done=False)
    for b in w.dom["baselines"]["items"]:
        if b.get("measure"):
            (w.inc / b["exp"] / "native_rescore.json").unlink()
    return w


def _diags(w):
    return {x["id"]: x for x in json.loads(S.StreamPaths(str(w.lab), "weed").diagnoses(NAME).read_text())["diagnoses"]}


def t_native():
    section("the measurement arms read at their own resolution (L23N, 2026-10-01): one rescore per done arm, priced "
            "small; a failure is a card; a qualifying verdict is card X18, and nothing switches")
    dom = LS.load_domain("weed")
    est, det = LS.price("L23N", PARAMS["L23N"], dom, {"runs": 6})
    est0, _d = LS.price("L23N", PARAMS["L23N"], dom, {})
    b3, _d = LS.price("L23B", dict(PARAMS["L23B"], exp="b_v2_m832", seeds="0,1,2", arm="m832", role="capacity"), dom,
                      {"images": 6813})
    check("priced small: one scoring pass per final run, 6 runs (the arm's 3 and the reference's 3) x 0.25 GPU-h = "
          "%.2f, under a tenth of the arm's build (%.1f)" % (est, b3),
          est == 1.5 and est0 == 1.5 and det["runs"] == 6 and det["hours_per_run"] == 0.25 and est < b3 / 10, det)
    ok_args = ["inc2.baseline", "rescore-native", "--exp", "b_v2_m832", "--reference", "b_v2_m640"]
    for args, want in ((ok_args, True), (ok_args[:4], False), (ok_args + ["--exam", "test"], False),
                       (["inc2.baseline", "rescore-native", "--exp", "a;b", "--reference", "b_v2_m640"], False),
                       (["inc2.stream", "rescore-native", "--exp", "b_v2_m832", "--reference", "b_v2_m640"], False)):
        try:
            SR.parse_submit("build", args)
            got = True
        except R.Refused:
            got = False
        check("the cluster's build grammar %s %s" % ("admits" if want else "refuses", " ".join(args[1:])), got == want)
    req = SR.parse_submit("build", ok_args)
    check("  its job is run_inc2_build.sh under its own name (inc_build_native_<exp>), never the arm's build job's",
          req["script"] == "run_inc2_build.sh" and SR.job_name(req, {}) == "inc_build_native_b_v2_m832", req)
    check("the evidence allow-lists capacity/native_v1.json and <exp>/native_rescore.json, never the report",
          E.allowed("capacity/native_v1.json") and E.allowed("b_v2_m832/native_rescore.json")
          and not E.allowed("capacity/native_v1_report.json") and not E.allowed("b_v2_m832/native_v1.json"))
    wr = _native_world("native_running")
    for b in [x for x in wr.dom["baselines"]["items"] if x.get("measure")]:
        wr.experiment(b["exp"], done=False)
    wr.tick(3)
    check("an arm that is built but not done is not rescored (no L23N)",
          not [e for e in wr.events("proposed") if e.get("lever") == "L23N"], [e.get("lever") for e in wr.events("proposed")])
    wc = _native_world("native_stage_c", stage_c=False)
    wc.tick(3)
    check("  nor before R0 is complete (Stage C running unread, MAINT idle): no L23N, DR0 silent",
          not [e for e in wc.events("proposed") if e.get("lever") == "L23N"] and not _diags(wc)["DR0"]["fired"],
          ([e.get("lever") for e in wc.events("proposed")], _diags(wc)["DR0"]["summary"]))
    wd = _native_world("native_data")
    wd.step1_status(one_time=("bootstrap", "knowntruth"))
    wd.tick(3)
    check("  nor while a DATA item of R0 is due (the L17 backfill not run): R0 completes first",
          not [e for e in wd.events("proposed") if e.get("lever") == "L23N"]
          and any(e.get("lever") == "L17" for e in wd.events("proposed")),
          [(e.get("lever"), (e.get("argv") or [])[-2:]) for e in wd.events("proposed")])
    wk = World("native_recorded")
    wk.ready_r0()
    wk.tick(3)
    check("  nor once its record says complete, whatever the platform ran (a world whose arms were rescored)",
          not [e for e in wk.events("proposed") if e.get("lever") == "L23N"] and not _diags(wk)["DR0"]["fired"],
          _diags(wk)["DR0"]["summary"])

    w = _native_world("native")
    cap0 = (w.inc / "capacity" / "capacity_v1.json").read_bytes()
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * 86400.0))
    w.tick(3)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23N"]
    first = pro[0] if pro else {}
    ex = [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23N"]
    check("R0 complete, b_v2_m832 done without native scores: DR0 proposes its rescore once (L23N rescore-native "
          "--exp b_v2_m832 --reference b_v2_m640), priced 1.5 GPU-h and granted within the envelope",
          [(e.get("argv") or [])[-5:] for e in pro] == [["rescore-native", "--exp", "b_v2_m832", "--reference",
                                                         "b_v2_m640"]]
          and abs(float(first.get("est_gpu_hours") or 0) - 1.5) < 1e-9 and ex == ["envelope"],
          ([(e.get("argv") or [])[-5:] for e in pro], ex))
    check("  citing only the lock, the arm's status and its own native state",
          sorted(c.get("pointer") for c in first.get("cites") or []) == ["/stage/exp_status/b_v2_m832", "/stage/lock",
                                                                         "/stage/native/cap_m832"], first.get("cites"))
    check("  one GPU-shared job of run_inc2_build.sh, under its own name",
          [x["name"] for x in w.submits if "rescore-native" in x["argv"]] == ["inc_build_native_b_v2_m832"]
          and all("GPU-shared" in x["argv"] for x in w.submits if "rescore-native" in x["argv"]))
    check("  the TRAIN lane does not wait for it: the first segment is cut and run (L18)",
          any(e.get("lever") == "L18" for e in w.events("executed")), [e.get("lever") for e in w.events("executed")])
    w.tick(3)
    due = (((_diags(w)["DR0"].get("detail") or {}).get("due") or {}).get("MAINT")) or {}
    check("while its job runs it is not proposed again (the MAINT lane follows it; DR0 now calls for b_v2_s1024's, "
          "which waits for the lane)",
          len([e for e in w.events("proposed") if e.get("lever") == "L23N"]) == 1
          and ((w.lane("MAINT").get("item") or {}).get("lever")) == "L23N"
          and due.get("baseline") == "cap_s1024" and w.state()["stage"]["r0"].get("native_cap_m832") == "running",
          (w.lane("MAINT").get("item", {}).get("lever"), _diags(w)["DR0"]["summary"]))
    w.job_done("inc_build_native_b_v2_m832")
    w.tick(3)
    pro = [(e.get("argv") or [])[-3] for e in w.events("proposed") if e.get("lever") == "L23N"]
    check("its job done (the record not yet in the evidence): b_v2_m832 is not proposed again, b_v2_s1024 is, once",
          pro == ["b_v2_m832", "b_v2_s1024"] and w.state()["stage"]["r0"].get("native_cap_m832") == "done", pro)
    w.native_record("b_v2_m832")
    w.job_done("inc_build_native_b_v2_s1024")
    w.native_record("b_v2_s1024")
    meas = [b["exp"] for b in w.dom["baselines"]["items"] if b.get("measure") and b.get("native") is not False]
    for exp in meas[2:]:            # the box-quality arms (2026-10-01), read at 640, in the domain's order
        w.tick(3)
        pro = [(e.get("argv") or [])[-3] for e in w.events("proposed") if e.get("lever") == "L23N"]
        check("then %s's rescore is proposed, once" % exp, pro[-1] == exp and pro.count(exp) == 1, pro)
        w.job_done("inc_build_native_%s" % exp)
        w.native_record(exp)
    w.tick(4)
    pro = [(e.get("argv") or [])[-3] for e in w.events("proposed") if e.get("lever") == "L23N"]
    d = _diags(w)
    check("all rescored: none is proposed again (DR0 silent); the stream's arm, its one arm line and "
          "capacity_v1.json are unchanged, no LA",
          pro == meas and not d["DR0"]["fired"] and w.state()["capacity"]["chosen"] == "n640"
          and len([e for e in w.stream_ledger() if e.get("event") == "arm"]) == 1
          and (w.inc / "capacity" / "capacity_v1.json").read_bytes() == cap0
          and not [e for e in w.events("proposed") if e.get("lever") == "LA"], (pro, d["DR0"]["summary"]))

    wf = _native_world("native_fail")
    wf.tick(3)
    check("the DATA lane works beside it: discovery (L15) proposed and run in the same ticks",
          [(e.get("lane"), e.get("lever")) for e in wf.events("executed")][:2] == [("DATA", "L15"), ("MAINT", "L23N")],
          [(e.get("lane"), e.get("lever")) for e in wf.events("executed")])
    wf.job_done("inc_build_native_b_v2_m832", state="FAILED", refusal="[inc2.baseline] ERROR: refused")
    wf.tick(3)
    st = wf.state()
    pro = [(e.get("argv") or [])[-3] for e in wf.events("proposed") if e.get("lever") == "L23N"]
    cards = [c["title"] for c in st.get("cards") or []]
    check("its job FAILED: one card, never a pause; the MAINT lane is not held and counts no failed step; "
          "b_v2_m832's rescore stays failed and b_v2_s1024's runs next",
          not st.get("paused") and wf.config().get("enabled") is True
          and cards == ["Native-resolution rescore of b_v2_m832 failed (L23N)"]
          and not wf.lane("MAINT").get("hold") and int(wf.lane("MAINT").get("fails") or 0) == 0
          and not [k for k in st.get("step_failures") or {} if k.startswith("L23N")]
          and st["stage"]["r0"].get("native_cap_m832") == "failed" and pro == ["b_v2_m832", "b_v2_s1024"],
          (st.get("paused"), cards, wf.lane("MAINT"), pro))
    wf.job_done("inc_build_native_b_v2_s1024", state="FAILED")
    wf.tick(4)
    st = wf.state()
    pro = [(e.get("argv") or [])[-3] for e in wf.events("proposed") if e.get("lever") == "L23N"]
    check("  a second failed rescore is a second card, not a held lane (2 consecutive failed steps would hold it); "
          "neither is proposed again (the next arm's rescore runs next)",
          pro[:2] == ["b_v2_m832", "b_v2_s1024"] and pro.count("b_v2_m832") == 1 and pro.count("b_v2_s1024") == 1
          and pro[2:] in ([], ["b_v2_y26l640"]) and not wf.lane("MAINT").get("hold")
          and not st.get("paused") and len([c for c in st.get("cards") or [] if "Native-resolution" in c["title"]]) == 2,
          (pro, wf.lane("MAINT").get("hold")))

    # a submission whose outcome is unknown (the verb ran, its reply never came): followed by its job name
    wu = _native_world("native_uncertain")
    wu.lose_reply = "rescore-native"
    wu.tick(2)
    st = wu.state()
    it = wu.lane("MAINT").get("item") or {}
    unc = [e for e in wu.events("uncertain") if e.get("lever") == "L23N"]
    check("an L23N submission whose outcome is unknown pauses nothing: it is followed by its job name (MAINT runs it, "
          "uncertain, no job id), no card, its arm running",
          not st.get("paused") and wu.config().get("enabled") is True and it.get("lever") == "L23N"
          and it.get("status") == "running" and it.get("uncertain") is True and not it.get("job_ids")
          and unc and "job name" in unc[0].get("next", "") and not st.get("cards")
          and st["stage"]["r0"].get("native_cap_m832") == "running"
          and [x["name"] for x in wu.submits if "rescore-native" in x["argv"]] == ["inc_build_native_b_v2_m832"],
          (st.get("paused"), it, unc, st.get("cards")))
    wu.tick(3)
    it = wu.lane("MAINT").get("item") or {}
    check("  while a job of its name is queued it stays running: not failed, not lost, not proposed again",
          it.get("lever") == "L23N" and it.get("status") == "running" and not it.get("lost")
          and len([e for e in wu.events("proposed") if e.get("lever") == "L23N"]) == 1 and not wu.state().get("cards"),
          it)
    wu.job_done("inc_build_native_b_v2_m832")
    wu.native_record("b_v2_m832")
    wu.tick(2)
    st = wu.state()
    check("  its job gone and its record complete: done, the lane free for the next arm (b_v2_s1024's rescore)",
          st["stage"]["r0"].get("native_cap_m832") == "done" and not st.get("cards") and not st.get("paused")
          and any(e.get("lever") == "L23N" for e in wu.events("item_done"))
          and [(e.get("argv") or [])[-3] for e in wu.events("proposed") if e.get("lever") == "L23N"]
          == ["b_v2_m832", "b_v2_s1024"], (st["stage"]["r0"], wu.lane("MAINT").get("item")))
    wl = _native_world("native_uncertain_lost")
    wl.lose_reply = "rescore-native"
    wl.tick(2)
    wl.job_done("inc_build_native_b_v2_m832", state="FAILED")
    wl.tick(2)
    it = wl.lane("MAINT").get("item") or {}
    check("  its job gone without a record: lost for %d snapshots before anything is decided"
          % (S.BUILD_LOST_SNAPSHOTS - 1), it.get("lever") == "L23N" and it.get("lost") == S.BUILD_LOST_SNAPSHOTS - 1
          and not wl.state().get("cards"), it)
    wl.tick(1)
    st = wl.state()
    check("  and then failed, record only: one card, no pause, no held lane, no failed step; not proposed again",
          [c["title"] for c in st.get("cards") or []] == ["Native-resolution rescore of b_v2_m832 failed (L23N)"]
          and not st.get("paused") and not wl.lane("MAINT").get("hold")
          and int(wl.lane("MAINT").get("fails") or 0) == 0
          and st["stage"]["r0"].get("native_cap_m832") == "failed"
          and len([e for e in wl.events("proposed") if (e.get("argv") or [])[-3:-2] == ["b_v2_m832"]]) == 1,
          ([c["title"] for c in st.get("cards") or []], wl.lane("MAINT"), st["stage"]["r0"]))
    wo = _native_world("native_qos")
    wo.qos = True
    wo.tick(2)
    st = wo.state()
    check("an sbatch refused on qos is a platform defect, as for every lever (S21): the MAINT lane holds, a platform "
          "card, no pause; the rescore never ran, so it is not marked failed",
          "qos" in str(wo.lane("MAINT").get("hold")) and any(c.get("kind") == "platform" for c in st.get("cards") or [])
          and not st.get("paused") and st["stage"]["r0"].get("native_cap_m832") in (None, "missing"),
          (wo.lane("MAINT"), st.get("cards"), st["stage"]["r0"]))

    # the MAINT lane runs one item: a milestone compare and a 'hurts' rollback wait for a queued L23N job
    wm = _native_world("native_maint_wait")
    wm.tick(1)
    s1 = wm.segment(1, [("I2", "ACCEPT", 0.9, [], "helps", ["src_a"])])
    wm._commit(s1)
    m1 = wm.milestone_built(done=True)
    wm.compare_plan[m1] = {"verdict": "hurts", "perm_p": 0.004, "new_mean_dev": 0.781, "old_mean_dev": 0.812}
    wm.tick(3)
    check("a milestone compare that falls due while an L23N job is queued waits for it (DCMP due, no LC run, the "
          "MAINT lane still on L23N; queue time included)",
          _diags(wm)["DCMP"]["fired"] and not [r for r in wm.runs if "compare" in r]
          and not [e for e in wm.events("proposed") if e.get("lever") == "LC"]
          and (wm.lane("MAINT").get("item") or {}).get("lever") == "L23N",
          (_diags(wm)["DCMP"]["summary"], (wm.lane("MAINT").get("item") or {}).get("lever")))
    wm.job_done("inc_build_native_b_v2_m832")
    wm.native_record("b_v2_m832")
    wm.tick(3)
    order = [e.get("lever") for e in wm.events("proposed") if e.get("lane") == "MAINT"]
    check("  once its job ends the compare runs (DCMP precedes DR0); before the compare's 'hurts' is in the "
          "evidence, the lane takes b_v2_s1024's rescore, and the recommended rollback waits for its job",
          [r[-2:] for r in wm.runs if "compare" in r] == [["--exp", m1]] and order == ["L23N", "LC", "L23N"]
          and not [r for r in wm.runs if "rollback" in r]
          and (wm.lane("MAINT").get("item") or {}).get("lever") == "L23N", (order, [r[3:] for r in wm.runs]))
    wm.job_done("inc_build_native_b_v2_s1024")
    wm.native_record("b_v2_s1024")
    wm.tick(2)
    order = [e.get("lever") for e in wm.events("proposed") if e.get("lane") == "MAINT"]
    check("  that job ended, the rollback runs (L21 to P_0, within the envelope)",
          [r[-2:] for r in wm.runs if "rollback" in r] == [["--to", "P_0"]] and order == ["L23N", "LC", "L23N", "L21"],
          (order, [r[3:] for r in wm.runs]))

    wq = World("native_x18")
    wq.ready_r0()
    wq.native_verdict_file(qualifying=["b_v2_m832"])
    capq = (wq.inc / "capacity" / "capacity_v1.json").read_bytes()
    wq.tick(3)
    st = wq.state()
    x18 = [c for c in st.get("cards") or [] if c.get("lever") == "X18"]
    d = _diags(wq)
    check("a verdict in which b_v2_m832 qualifies is card X18 (R4, for a person), filed once: no pause, no hold, the "
          "stream's arm and capacity_v1.json unchanged, no LA",
          len(x18) == 1 and x18[0]["risk"] == "R4" and "b_v2_m832" in x18[0]["detail"] and d["DNAT"]["fired"]
          and d["DNAT"]["levers"] == ["X18"] and not st.get("paused")
          and not any(wq.lane(ln).get("hold") or wq.lane(ln).get("diag_hold") for ln in ("TRAIN", "DATA", "MAINT"))
          and st["capacity"]["chosen"] == "n640" and (wq.inc / "capacity" / "capacity_v1.json").read_bytes() == capq
          and not [e for e in wq.events("proposed") if e.get("lever") == "LA"], (x18, d["DNAT"]["summary"]))
    wq.tick(3)
    check("  and not filed again on the next ticks",
          len([c for c in wq.state().get("cards") or [] if c.get("lever") == "X18"]) == 1)
    wn = World("native_none")
    wn.ready_r0()
    wn.native_verdict_file(qualifying=())
    wn.tick(3)
    d = _diags(wn)
    check("a verdict in which no arm qualifies files no card (DNAT silent)",
          not d["DNAT"]["fired"] and not [c for c in wn.state().get("cards") or [] if c.get("lever") == "X18"],
          d["DNAT"]["summary"])
    pw = World("native_perturbed", perturbed=True)
    pw.ready_r0()
    pw.native_verdict_file(qualifying=["b_v2_m832"])
    pw.tick(2)
    check("test-blind: with every non-dev exam value of the cluster's files perturbed, DNAT reads the same",
          _diags(pw)["DNAT"]["summary"] == _diags(wq)["DNAT"]["summary"] and _diags(pw)["DNAT"]["fired"],
          (_diags(pw)["DNAT"]["summary"], _diags(wq)["DNAT"]["summary"]))


def t_formats():
    section("the other groups' formats, read as they write them")
    w = World("formats")
    w.ready_r0()
    w.tick()
    run = W.S.StreamRun(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log, w.hooks,
                        lambda: W.RES, None, None, None)
    run.st, run.dom, run.th = w.state(), LS.load_domain("weed"), LS.load_thresholds()
    # group D's candidates.json (collect-candidates/1)
    w.candidates([
        {"source_id": "zen:1", "id": "zen:1", "provider": "hf", "licence": {"id": "CC-BY-4.0", "class": "open"},
         "licence_ok": True, "target_classes": ["Purslane"], "bytes": 1e9, "expected_target_boxes": 500,
         "precheck": {"ok": True, "failures": []}, "decision": {"status": "kept"}},
        {"source_id": "rf:x", "id": "rf:x", "provider": "roboflow", "licence": {"id": None, "class": "unresolved"},
         "target_classes": ["Purslane"], "bytes": 1e8, "expected_target_boxes": 900,
         "precheck": {"ok": False, "risk": "R3", "action": "hold",
                      "failures": [{"code": "licence_unresolved", "action": "hold", "risk": "R3"}]}},
        {"source_id": "kg:y", "id": "kg:y", "provider": "kaggle", "licence": {"id": "MIT", "class": "open"},
         "target_classes": ["Purslane"], "bytes": 1e8, "expected_target_boxes": 900,
         "precheck": {"ok": False, "action": "close", "failures": [{"code": "never_train", "action": "close"}]}},
        {"source_id": "hf:z", "id": "hf:z", "provider": "hf", "licence": {"id": "MIT", "class": "open"},
         "target_classes": [], "bytes": 1e8, "decision": {"status": "pending_names"}, "names_pending": ["Zz"],
         "precheck": {"ok": False, "failures": [{"code": "names_pending", "action": "hold"}]}}])
    c = {x["id"]: x for x in run._candidates()}
    v = DS.View(E.from_texts({}, "x", context={"limits": {"gb_per_source": 50}}), run.dom, run.th)
    pre = {k: DS.precheck(v, x) for k, x in c.items()}
    check("a candidate the collector's pre-check passed is open", pre["zen:1"] == ([], []), pre["zen:1"])
    check("  an R3 hold of the collector's pre-check is a person's review", pre["rf:x"][1] and not pre["rf:x"][0],
          pre["rf:x"])
    check("  a close of the collector's pre-check refuses the source", pre["kg:y"][0], pre["kg:y"])
    check("  and pending names send it to L26 first", c["hf:z"]["names_unresolved"], c["hf:z"])
    nt, why = LS.never_train_slugs(run.dom)
    check("the never-train slugs are parsed from the trainer's source, never imported",
          nt and "cottonweeddet12" in nt and "weed_optimizer_framework.tools.mega_trainer" not in sys.modules, (why,))
    # group C's status.json: known truth and the holds past their deadline
    # live 2026-10-03: b0001, 284 of 284 correct, raised X11 under the Wilson-lower-bound rule
    w.step1_status(extra={"knowntruth": {"b0001": {"matched_verified": 284, "verified_correct": 284},
                                         "b0002": {"matched_verified": 60, "verified_correct": 58},
                                         "b0003": {"matched_verified": 284, "verified_correct": 276},
                                         "b0004": {"matched_verified": 284, "verified_correct": 277},
                                         "b0005": {"matched_verified": 29, "verified_correct": 20}},
                          "refit_triggers": {"precision_below": ["b0003"], "species_unknown_share": ["Eclipta"],
                                             "fired": True},
                          "holds_past_deadline": {"funnel_F9": 12}})
    ev = E.from_texts({"step1_stream/status.json": (w.inc / "step1_stream" / "status.json").read_text()}, "x",
                      context={"sid": "none"})
    by = DS.by_id(DS.detect(ev, run.dom, run.th, only=("DKT", "DHOLD")))
    s_kt = by["DKT"]["summary"]
    check("DKT: 276 of 284 correct (P = 0.0084 < 0.01) -> card X11; and the species trigger",
          by["DKT"]["fired"] and "X11" in by["DKT"]["levers"] and "batch b0003" in s_kt and "Eclipta" in s_kt, s_kt)
    check("  284 of 284 (no error), 58 of 60 (P = 0.12), 277 of 284 (P = 0.025) and 29 matched boxes do not fire",
          not any("batch %s" % b in s_kt for b in ("b0001", "b0002", "b0004", "b0005")), s_kt)
    w.step1_status(extra={"knowntruth": {"b0001": {"matched_verified": 284, "verified_correct": 284}},
                          "refit_triggers": {"precision_below": [], "species_unknown_share": [], "fired": False}})
    ev = E.from_texts({"step1_stream/status.json": (w.inc / "step1_stream" / "status.json").read_text()}, "x",
                      context={"sid": "none"})
    by = DS.by_id(DS.detect(ev, run.dom, run.th, only=("DKT",)))
    check("  a perfect batch alone is silent", not by["DKT"]["fired"], by["DKT"]["summary"])
    w.step1_status(extra={"knowntruth": {"b0003": {"matched_verified": 284, "verified_correct": 276}},
                          "refit_triggers": {"precision_below": ["b0003"], "species_unknown_share": ["Eclipta"],
                                             "fired": True},
                          "holds_past_deadline": {"funnel_F9": 12}})
    ev = E.from_texts({"step1_stream/status.json": (w.inc / "step1_stream" / "status.json").read_text()}, "x",
                      context={"sid": "none"})
    by = DS.by_id(DS.detect(ev, run.dom, run.th, only=("DKT", "DHOLD")))
    check("DHOLD reads step1_stream's holds past their deadline when no stream summary states them",
          by["DHOLD"]["fired"] and by["DHOLD"]["detail"]["holds"][0]["hold"] == "funnel_F9", by["DHOLD"]["summary"])
    # group D's intake summary and source ledger
    w.intake("i0001_zen_1", "zen:1", images=200, eval_share=0.0, base_share=0.25)
    w.sources([{"source": "zen:1", "event": "fetch_started"}, {"source": "zen:1", "event": "fetched", "bytes": 2.5e9,
                                                                "seconds": 3600.0, "su": 1.0},
               {"source": "zen:1", "event": "intaken", "batch": "i0001_zen_1", "seconds": 60.0}])
    w._activate()
    arts = SR.stream_summary(w.sid)["decision"]["artifacts"]          # the cluster verb itself
    fold = arts.get("intake/sources.json") or {}
    check("the source ledger is folded by the collector's own fold (status, bytes, batches)",
          (fold.get("zen:1") or {}).get("status") == "intaken" and (fold.get("zen:1") or {}).get("bytes") == 2.5e9
          and (fold.get("zen:1") or {}).get("batches") == ["i0001_zen_1"], fold.get("zen:1"))
    # a hold of the collector's (a licence, credentials, the copy scan) is released when the collector releases it
    w.sources([{"source": "zen:2", "event": "candidate"}, {"source": "zen:2", "event": "held",
                                                           "reason": "licence unresolved", "codes": ["licence_unresolved"]}])
    for _ in range(3):
        w.tick()
    s2 = (w.state()["sources"].get("zen:2") or {})
    check("a source the collector holds is held here too, by the collector", s2.get("status") == "held"
          and s2.get("held_by") == "collector", s2)
    w.sources([{"source": "zen:2", "event": "released", "reason": "a person resolved the licence"}])
    for _ in range(3):
        w.tick()
    s2 = (w.state()["sources"].get("zen:2") or {})
    check("  and collectable again once the collector releases it", s2.get("status") == "candidate"
          and not s2.get("held_by"), s2)
    # a person's licence override (the collect config's licence_overrides): collect.plan releases the hold in the
    # lab's ledger, which no snapshot folds, so a licence hold in this one is read as released
    w.sources([{"source": sid, "event": ev, "reason": "licence_unresolved", "codes": ["licence_unresolved"]}
               for sid in ("zen:4", "zen:5") for ev in ("candidate", "held")])
    for _ in range(3):
        w.tick()
    src = lambda sid: (w.state()["sources"].get(sid) or {})  # noqa: E731
    check("a licence hold of the collector's with no person's override is held here",
          src("zen:4").get("status") == "held" and src("zen:5").get("status") == "held", (src("zen:4"), src("zen:5")))
    ov = {"id": "research-only", "class": "research_only", "research_only": True, "decided_by": "human:owner",
          "decided_utc": "2026-09-30", "reason": "licence unresolved; accepted for research use only"}
    w.sources([{"source": "zen:6", "event": ev, "reason": "copy_scan_pending", "codes": ["copy_scan_pending"]}
               for ev in ("candidate", "held")])
    cc = json.loads(w.collect_cfg.read_text())
    cc["licence_overrides"] = {"zen:4": ov, "zen:6": ov}
    w.collect_cfg.write_text(json.dumps(cc))
    cc_sha = hashlib.sha256(w.collect_cfg.read_bytes()).hexdigest()
    for _ in range(3):
        w.tick()
    check("  once the collect config records a person's licence override for it, collectable again (released, "
          "recorded); a source without one stays held",
          src("zen:4").get("status") == "candidate" and not src("zen:4").get("held_by")
          and src("zen:5").get("status") == "held"
          and [e.get("source") for e in w.events("source_released")].count("zen:4") == 1,
          (src("zen:4"), src("zen:5"), [e.get("source") for e in w.events("source_released")]))
    check("  an override lifts a licence hold only: a source it names that the collector holds for the copy scan "
          "stays held, by the collector", src("zen:6").get("status") == "held"
          and src("zen:6").get("held_by") == "collector" and "zen:6" not in [e.get("source") for e in
                                                                            w.events("source_released")],
          src("zen:6"))
    rel = {e.get("source"): e for e in w.events("source_released")}
    check("  the release names the person's decision (by licence_overrides, decided_by, decided_utc, the licence, "
          "the collect config's sha256), not the collector; a release of the collector's own stays the collector's",
          rel["zen:4"].get("by") == "licence_overrides" and rel["zen:4"].get("decided_by") == "human:owner"
          and rel["zen:4"].get("decided_utc") == "2026-09-30" and rel["zen:4"].get("licence") == "research-only"
          and rel["zen:4"].get("research_only") is True and rel["zen:4"].get("collect_config_sha256") == cc_sha
          and rel["zen:2"].get("by") == "collector" and not str(rel["zen:2"].get("decided_by")).startswith("human:"),
          (rel.get("zen:4"), rel.get("zen:2")))
    w.candidates([{"source_id": "zen:4", "id": "zen:4", "provider": "hf",
                   "licence": {"id": "unresolved", "class": "unresolved"}, "licence_ok": True,
                   "licence_id": "research-only", "licence_class": "unresolved", "licence_override": ov,
                   "target_classes": ["Purslane"], "bytes": 1e8, "expected_target_boxes": 500,
                   "precheck": {"ok": True, "failures": []}, "decision": {"status": "kept"}}])
    c4 = {x["id"]: x for x in run._candidates()}["zen:4"]
    check("  and the plan's record of the override reads as a usable licence (no 'licence unknown' review)",
          c4["licence"] == "research-only" and c4["licence_ok"] is True and DS.precheck(v, c4) == ([], []),
          (c4["licence"], c4["licence_ok"], DS.precheck(v, c4)))
    # a candidates file plan wrote before the override: the licence unresolved, its pre-check holding
    # licence_unresolved for a person (R3); the collect config's override is read directly, as the fold reads it
    def stale(sid, cls="unresolved", lid="unresolved", code="licence_unresolved", action="hold"):
        return {"source_id": sid, "id": sid, "provider": "hf", "licence": {"id": lid, "class": cls},
                "licence_ok": None if cls == "unresolved" else False, "licence_id": lid, "licence_class": cls,
                "target_classes": ["Purslane"], "bytes": 1e8, "expected_target_boxes": 500,
                "precheck": {"ok": False, "risk": "R3" if action == "hold" else None, "action": action,
                             "failures": [dict({"code": code, "action": action}, **({"risk": "R3"}
                                                                                   if action == "hold" else {}))]},
                "decision": {"status": "kept"}}
    w.candidates([stale("zen:4"), stale("zen:5"),
                  stale("zen:7", cls="refused", lid="all-rights-reserved", code="licence_refused", action="close")])
    cc["licence_overrides"]["zen:7"] = ov
    w.collect_cfg.write_text(json.dumps(cc))
    cs = {x["id"]: x for x in run._candidates()}
    pre = {k: DS.precheck(v, cs[k]) for k in ("zen:4", "zen:5", "zen:7")}
    check("  a candidates file written before the override reads the collect config's override: a usable licence, "
          "the collector's licence_unresolved hold lifted, no review; without an override the source is still "
          "reviewed; a refused licence stays refused whatever an override says",
          cs["zen:4"]["licence"] == "research-only" and cs["zen:4"]["licence_ok"] is True
          and cs["zen:4"]["collector_precheck"] == {"refuse": [], "review": [], "wait": []} and pre["zen:4"] == ([], [])
          and "licence unknown" in pre["zen:5"][1] and "collector: licence_unresolved" in pre["zen:5"][1]
          and cs["zen:7"]["licence_ok"] is False and "licence not research-usable" in pre["zen:7"][0]
          and "collector: licence_refused" in pre["zen:7"][0], (cs["zen:4"], pre))
    w.sources([{"source": "zen:3", "event": "fetched", "bytes": 1e9, "complete": False, "remaining": 7}])
    fold = SR.stream_summary(w.sid)["decision"]["artifacts"].get("intake/sources.json") or {}
    check("the source fold carries the last fetch's completeness (a fetch stopped at a byte cap: its next shard)",
          (fold.get("zen:3") or {}).get("fetch_complete") is False and (fold.get("zen:3") or {}).get("fetch_remaining") == 7
          and (fold.get("zen:1") or {}).get("fetch_complete") is None, (fold.get("zen:3"), fold.get("zen:1")))
    got = run._collect_result({"tail": "log line\n[collect] fetch: {\"complete\": false, \"remaining\": 3}\n"})
    check("  and a lab fetch's closing line gives the same (the lab's ledger is folded by no snapshot)",
          got == {"complete": False, "remaining": 3}, got)
    ev = E.from_texts({"intake/i0001_zen_1/summary.json": json.dumps(arts["intake/i0001_zen_1/summary.json"])}, "x",
                      context={"sid": "none"})
    by = DS.by_id(DS.detect(ev, run.dom, run.th, only=("D28",)))
    check("D28 reads the intake summary's source_leak (25 % base copies >= 20 %)", by["D28"]["fired"]
          and by["D28"]["detail"]["leaks"][0]["source"] == "zen:1", by["D28"]["summary"])


# ------------------------------------------------------------------ the replay gate
def t_replay_gate():
    section("the replay gate's stream cases")
    w = World("gate", replay=False)
    base = dict({c: "pass" for c in X.REPLAY_REQUIRED}, R2="skip", R4b="skip")
    X.record_replay_result("pass", base, ctx=w.xctx)
    check("a pass without the stream cases still counts for experiment mode", X.replay_status(w.xctx)["passed"])
    check("  but not for a stream campaign", not X.stream_replay_status(w.xctx)["passed"],
          X.stream_replay_status(w.xctx)["reason"])
    try:
        X.record_replay_result("pass", dict(base, S3="fail"), ctx=w.xctx)
        check("a failing S-case can never be recorded as a pass", False)
    except ValueError:
        check("a failing S-case can never be recorded as a pass (it blocks every campaign's envelope)", True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gate_", dir=str(W.TMP)))

    def script(name, out, rc):
        (tmp / name).write_text("import sys\nprint(%r)\nsys.exit(%d)\n" % (out, rc))
    for n in ("f.py", "m.py", "d.py", "sm.py"):
        script(n, "0 failure(s), 0 skipped: none", 0)
    script("r.py", "0 failure(s), 2 skipped: R2, R4b on pilot_v2", 0)
    script("g.py", "ALL PASS", 0)
    cases_ok = "\n".join("case %s: pass" % c for c in X.STREAM_REPLAY_CASES) + "\n0 failure(s), 0 skipped: none"
    script("s.py", cases_ok, 0)
    scripts = {"replay": "r.py", "governance": "g.py", "funnel": "f.py", "funnel_mutations": "m.py", "domain_free": "d.py",
               "stream": "s.py", "stream_mutations": "sm.py"}
    rec = X.run_replay_tests(ctx=w.xctx, scripts=scripts, code_root=tmp)
    check("run_replay_tests records every stream case and the harness from their scripts",
          rec["status"] == "pass" and all(rec["cases"][c] == "pass" for c in X.STREAM_REPLAY_CASES)
          and rec["cases"][X.STREAM_MUTATION_CASE] == "pass", rec["cases"])
    check("  and a stream replay pass follows", X.stream_replay_status(w.xctx)["passed"])
    bad = cases_ok.replace("case S7: pass", "case S7: fail").replace("0 failure(s)", "1 failure(s)")
    script("s.py", bad, 1)
    rec = X.run_replay_tests(ctx=w.xctx, scripts=scripts, code_root=tmp)
    check("a failing S7 fails the whole record (every campaign's autonomy waits)", rec["status"] == "fail"
          and rec["cases"]["S7"] == "fail" and rec["cases"]["S6"] == "pass", rec["cases"].get("S7"))
    script("s.py", "crash", 1)
    rec = X.run_replay_tests(ctx=w.xctx, scripts=scripts, code_root=tmp)
    check("  a script that crashes fails every stream case", rec["status"] == "fail"
          and all(rec["cases"][c] == "fail" for c in X.STREAM_REPLAY_CASES))
    check("the stream scripts and the collector's config are governance files", {
        "tests/test_stream_ap_replay.py", "tests/test_stream_ap_mutations.py",
        "weed_optimizer_framework/tools/collect/domains/weed.json"} <= set(X.GOVERNANCE_FILES))
    check("the record-replay CLI runs the stream scripts too", X.REPLAY_SCRIPTS.get("stream") ==
          "tests/test_stream_ap_replay.py" and X.REPLAY_SCRIPTS.get("stream_mutations") ==
          "tests/test_stream_ap_mutations.py")


# ------------------------------------------------------------------ config and dispatch
def t_config():
    section("config, goal and dispatch")
    check("a stream campaign's goal is {'kind': 'continuous'}", C.check_goal({"kind": "continuous"}, "stream") ==
          {"kind": "continuous"})
    check("  and an experiment campaign's goal cannot be", _raises(lambda: C.check_goal({"kind": "continuous"})))
    check("experiment goals are checked as before", C.check_goal({"kind": "exp_done", "exp": "x"}) ==
          {"kind": "exp_done", "exp": "x"})
    w = World("config", autonomy="off", data_autonomy="off")
    check("configure_stream refuses a non-person", _raises(lambda: S.configure_stream(NAME, "round-scheduler:x",
                                                                                     cfg_hooks=w.hooks)))
    cfg = w.config()
    check("L-2's defaults: 1,000 SU to 2026-12-31, 350 monthly, 120 daily; data_autonomy off; the stream block",
          S.stream_config(cfg, NAME)["envelope_su"] == 1000.0 and S.stream_config(cfg, NAME)["window_cap_su"] == 350.0
          and S.stream_config(cfg, NAME)["daily_cap_su"] == 120.0 and cfg["data_autonomy"] == "off"
          and cfg["stream"]["sid"] == "weed_stream_v1", cfg)
    check("an experiment campaign cannot be turned into a stream one", _raises(lambda: _to_stream(w)))
    rc = S.main(["--config", str(w.cfg), "--lab-repo", str(w.lab), "enable", "--name", NAME, "--by", OWNER,
                 "--envelope-end-utc", "2027-03-31T23:59:59Z", "--collect-gb-daily", "30"])
    c2 = w.config()
    check("a person sets a new envelope end and the daily byte cap from the CLI (the envelope's end pauses the "
          "campaign until then)", rc == 0 and c2.get("envelope_end_utc") == "2027-03-31T23:59:59Z"
          and c2.get("collect_gb_daily") == 30.0, (rc, c2.get("envelope_end_utc"), c2.get("collect_gb_daily")))
    check("  a malformed end date is refused", _raises(lambda: S.configure_stream(
        NAME, OWNER, envelope_end_utc="31/12/2027", cfg_hooks=w.hooks, lab_repo=str(w.lab))))
    seen = []
    orig_run, orig_srun = C._Run, S.StreamRun

    class FakeRun(object):
        def __init__(self, name, *a, **k):
            self.name = name

        def go(self):
            seen.append(("exp", self.name))
            return {}

    class FakeStream(FakeRun):
        def go(self):
            seen.append(("stream", self.name))
            return {}
    C._Run, S.StreamRun = FakeRun, FakeStream
    try:
        hooks = C._local_cfg_hooks(W.TMP / "dispatch.json")
        (W.TMP / "dispatch.json").write_text(json.dumps({"campaigns": {"a_exp": {"enabled": True},
                                                                       "b_stream": {"mode": "stream", "enabled": True}}}))
        C.tick(slurm_sh=lambda s, t=60: {}, cfg_hooks=hooks, lab_repo=str(w.lab), resources=W.RES, clock=w.clock)
    finally:
        C._Run, S.StreamRun = orig_run, orig_srun
    check("campaign.tick hands mode 'stream' to StreamRun and everything else to _Run",
          sorted(seen) == [("exp", "a_exp"), ("stream", "b_stream")], seen)
    st = C.status(cfg_hooks=w.hooks, lab_repo=str(w.lab))
    check("campaign.status shows a stream campaign's lanes", (st.get(NAME) or {}).get("summary", {}).get("mode")
          == "stream", st.get(NAME))


def _to_stream(w):
    hooks = C._local_cfg_hooks(W.TMP / "to_stream.json")
    (W.TMP / "to_stream.json").write_text(json.dumps({"campaigns": {"old": {"mode": "experiment"}}}))
    S.configure_stream("old", OWNER, domain="weed", cfg_hooks=hooks, lab_repo=str(w.lab))


# ------------------------------------------------------------------ lab runner
def t_lab():
    section("the lab runner and the sync")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="lab_", dir=str(W.TMP)))
    spec = tmp / "j.spec.json"
    spec.write_text(json.dumps({"job": "j", "argv": [sys.executable, "-c", "print('hi')"], "result": str(tmp / "j.result.json"),
                                "timeout": 30}))
    rc = S.lab_run(str(spec))
    res = json.loads((tmp / "j.result.json").read_text())
    check("lab-run runs the argv and writes its result", rc == 0 and res["ok"] and "hi" in res["tail"], res)
    lab = tmp / "lab_inc"
    (lab / "intake" / "staging" / "srcA").mkdir(parents=True)
    (lab / "intake" / "staging" / "srcA" / "a.jpg").write_bytes(b"x")
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return W._Proc(0, "", "")
    r = S.lab_sync(lab, "srcA", "user@login", "user@data", runner=runner, cluster_inc="/ocean/x/inc")
    check("lab-sync rsyncs the source's staging and checks each file's sha256 on arrival",
          r["ok"] and calls[0][0] == "rsync" and "user@data:/ocean/x/inc/" in calls[0] and calls[1][0] == "ssh"
          and r["pushed"] == ["intake/staging/srcA/a.jpg"], (r, calls))
    check("  a source with no staging is refused", not S.lab_sync(lab, "nope", "t", runner=runner)["ok"])
    check("  each file's sha256 is the one checked on arrival", r["sha256"] == {
        "intake/staging/srcA/a.jpg": hashlib.sha256(b"x").hexdigest()}, r.get("sha256"))
    (lab / "intake" / "names").mkdir(parents=True)
    (lab / "intake" / "names" / "names_cache.json").write_text("{}")
    (lab / "intake" / "names" / "names_srcA.json").write_text("{}")
    calls.clear()
    r = S.lab_sync(lab, "srcA", "user@login", "user@data", runner=runner, cluster_inc="/ocean/x/inc", names=True)
    check("lab-sync --names pushes the collector's names layer and the source's names record (read offline on the "
          "cluster)", r["ok"] and r["pushed"] == ["intake/names/names_cache.json", "intake/names/names_srcA.json"], r)
    r = S.lab_sync(lab, "srcA", "user@login", "user@data", runner=runner, cluster_inc="/ocean/x/inc")
    check("  and a staging push carries them too", r["ok"] and "intake/names/names_cache.json" in r["pushed"]
          and "intake/staging/srcA/a.jpg" in r["pushed"], r["pushed"])
    check("  with no names layer, a names push is refused", not S.lab_sync(tmp / "empty", "srcA", "t",
                                                                           runner=runner, names=True)["ok"])


def t_lanes():
    section("lane mechanics: STOP first, items run elsewhere")
    w = World("lanes")
    w.ready_r0()
    w.queue(0)
    w.tick()
    run = W.S.StreamRun(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log, w.hooks,
                        lambda: W.RES, None, None, None)
    run.st, run.dom, run.th = w.state(), LS.load_domain("weed"), LS.load_thresholds()
    order = []
    stop_p = {"id": "stop1", "risk": "R2", "policy_action": "inc_stream_quarantine", "lever": "L24"}
    r3_p = {"id": "r3", "risk": "R3", "policy_action": "inc_build_segment", "lever": "L18"}
    run._ready = lambda: [("TRAIN", {"proposal": r3_p, "lever": "L18"}, None),
                          ("STOP", {"proposal": stop_p, "lever": "L24"}, None)]

    class Ssh(object):
        calls = 0

        def left(self):
            return self.calls < 1
    run.ssh = Ssh()
    orig_submit, orig_many = X.submit, X.submit_many

    def fake_submit(p, **kw):
        order.append(p["id"])
        run.ssh.calls += 1
        return {"status": "executed", "job_ids": ["1"]}

    def fake_many(ps, **kw):
        order.extend(p["id"] for p in ps)
        run.ssh.calls += 1
        return [{"status": "executed", "job_ids": ["2"]} for _ in ps]
    X.submit, X.submit_many = fake_submit, fake_many
    run._on_result = lambda ln, it, res: None
    try:
        run._submit_ready()
    finally:
        X.submit, X.submit_many = orig_submit, orig_many
    check("a ready STOP item (a quarantine) takes the tick's ssh before an R3 build (6.2: stop first)",
          order == ["stop1"], order)
    # an item a person ran from the INC page: the lane takes its outcome
    it = {"proposal": {"id": "pp", "policy_action": "inc_stream_admit", "follow": "job", "params": {"verb": "backfill"},
                       "trigger": ["DR0"]}, "lever": "L17", "status": "filed", "approval_id": "ap-x"}
    run.st["lanes"]["DATA"]["item"] = it
    got = []
    run._on_result = lambda ln, item, res: got.append((ln, res.get("status"), res.get("job_ids")))
    run._executed_elsewhere("DATA", it, {"execution": {"phase": "done", "executed_by": OWNER,
                                                       "outcome": {"status": "executed", "job_ids": ["99"]}}})
    check("an approved item a person ran elsewhere: the lane follows that run (its job ids)",
          got == [("DATA", "executed", ["99"])], got)
    # a job lever refused as "already executed" (proposed again before its
    # effect showed): nothing to follow, so the lane is freed, not failed
    run2 = W.S.StreamRun(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log, w.hooks,
                         lambda: W.RES, None, None, None)
    run2.st, run2.dom, run2.th = w.state(), LS.load_domain("weed"), LS.load_thresholds()
    jit = {"proposal": {"id": "dup", "policy_action": "inc_build_segment", "follow": "job", "params": {"verb": "fork"},
                        "trigger": ["D8S"]}, "lever": "L22", "status": "proposed"}
    run2.st["lanes"]["TRAIN"].update(item=jit, fails=0)
    run2._on_result("TRAIN", jit, {"status": "refused", "reasons": ["approval ap-1 was already executed"]})
    lane = run2.st["lanes"]["TRAIN"]
    check("  a job lever refused as already executed frees its lane (declined), never a failed step",
          lane.get("item") is None and not lane.get("fails") and "dup" in run2.st["declined"], lane)


def t_d28():
    section("D28's source rule (contract 6.4 as amended 2026-09-29: dHash copies, or embedding hits above what the "
            "per-image false-positive rate predicts; inc2.embed_calibration.source_verdict)")
    from weed_optimizer_framework.tools.inc2 import embed_calibration as EC
    from weed_optimizer_framework.tools.inc2 import guard as G2
    dom, th = LS.load_domain("weed"), LS.load_thresholds()
    check("the thresholds: source_alpha mirrors inc2's SOURCE_ALPHA; the 5 % never-train share rule is gone",
          LS.t(th, "D28", "source_alpha") == EC.SOURCE_ALPHA and "never_train_share" not in th["D28"])
    check("D28's reason names are the guard's (inc2.guard.REASONS)",
          set(DS.D28_DHASH_REASONS) | {DS.D28_EMBED_REASON} <= set(G2.REASONS))

    def summary(src, images, reasons, p_false=0.01, base_share=0.0):
        n_eval = sum(v for k, v in reasons.items() if k in DS.D28_DHASH_REASONS + (DS.D28_EMBED_REASON,))
        return json.dumps({"format": "collect-summary/1", "source": src, "batch": "b", "images": images,
                           "guard": dict(reasons), "yield": {"images_seen": images, "rejected": dict(reasons)},
                           "source_leak": {"eval_share": round(n_eval / float(images), 4), "base_share": base_share},
                           "copy_scan": {"checked": True, "p_false": p_false, "cos_threshold": 0.9}})

    def d28(texts, lock_p=None):
        if lock_p is not None:
            texts = dict(texts, **{"splits/v2/lock_status.json": json.dumps(
                {"splits_version": "v2", "locked": True, "embed_calibration_v2": {"p_false": lock_p}})})
        ev = E.from_texts(texts, "x", context={"sid": "none"})
        return DS.by_id(DS.detect(ev, dom, th, only=("D28",)))["D28"]
    d = d28({"intake/b0001_chance/summary.json": summary("zen:chance", 15, {"near_eval_embed": 1})})
    check("one embedding hit in a 15-image source (6.7 % of it, over the old 5 % share) is chance at a 1 % "
          "per-image rate: no leak", not d["fired"], d.get("summary"))
    d = d28({"intake/b0002_copy/summary.json": summary("zen:copy", 15, {"near_eval_variant": 1})})
    check("a source with a planted copy (a flip within 6 dHash bits of a dev image) and no pair-cosine record "
          "leaks (D28-v2's fail-closed rule): D28 -> L24",
          d["fired"] and "L24" in d["levers"] and d["detail"]["leaks"][0]["source"] == "zen:copy"
          and d["detail"]["leaks"][0]["verdict"]["dhash_hits"] == 1, d.get("summary"))
    d = d28({"intake/b0003_many/summary.json": summary("zen:many", 15, {"near_eval_embed": 5})})
    check("five embedding hits of 15 at 1 % are improbable (P < 0.001): a leak, with observed and expected hits",
          d["fired"] and d["detail"]["leaks"][0]["verdict"]["expected_false_hits"] == 0.15
          and d["detail"]["leaks"][0]["verdict"]["p_value"] < 0.001, d.get("detail"))
    d = d28({"intake/b0004_big/summary.json": summary("zen:big", 1915, {"near_eval_embed": 20})})
    check("a large same-domain source with hits within its rate is not a leak (805 of 1,915 was the old rule's "
          "failure)", not d["fired"], d.get("summary"))
    d = d28({"intake/b0005_base/summary.json": summary("zen:base", 20, {}, base_share=0.25)})
    check("the base-copy share rule is unchanged (25 % >= 20 %)", d["fired"], d.get("summary"))
    bare = json.loads(summary("zen:bare", 15, {"near_eval_v2": 1}))
    for k in ("images", "guard", "source_leak"):
        bare.pop(k)
    d = d28({"intake/b0007_bare/summary.json": json.dumps(bare)})
    check("a summary without the guard's flat fields and source_leak is not read (yield.rejected is not D28's input)",
          not d["fired"], d.get("summary"))
    old = json.loads(summary("zen:old", 15, {"near_eval_embed": 1}))
    old.pop("guard")
    d = d28({"intake/b0008_old/summary.json": json.dumps(old)})
    check("never-train refusals without their per-reason counts: a dHash copy cannot be ruled out (fail closed)",
          d["fired"] and d["detail"]["leaks"][0]["verdict"]["dhash_hits"] == 1, d.get("summary"))
    nop = json.loads(summary("zen:nop", 15, {"near_eval_embed": 1}))
    nop["copy_scan"] = {"checked": False}
    d = d28({"intake/b0006_nop/summary.json": json.dumps(nop)})
    check("without any per-image rate (no copy scan record, no LOCK calibration) an embedding hit flags (fail "
          "closed)", d["fired"], d.get("summary"))
    d = d28({"intake/b0006_nop/summary.json": json.dumps(nop)}, lock_p=0.01)
    check("  with the splits LOCK's v2 calibration rate it is chance again", not d["fired"], d.get("summary"))
    st = {"format": "inc2-step1-stream/status/1", "per_source": {
        "rf_chance": {"images_seen": 15, "near_eval_embed": 1},
        "rf_copy": {"images_seen": 15, "near_eval_embed": 0, "decision:near_eval_v2": 1},
        "rf_many": {"images_seen": 200, "near_eval_embed": 14}}}
    d = d28({"step1_stream/status.json": json.dumps(st)}, lock_p=0.01)
    got = sorted(h["source"] for h in (d.get("detail") or {}).get("leaks") or [])
    check("Step 1's per-source counts under the same rule: the chance hit is not a leak; the dHash copy and 14 of "
          "200 embedding hits are", got == ["rf_copy", "rf_many"], (got, d.get("summary")))


COPY_COS = 0.946384                  # the cluster's v2 calibration threshold (job 47260765)


def t_d28_v2():
    section("D28-v2 (docs/CONTINUOUS_LOOP.md, amendment 2026-10-03, pre-registered V3-1): a dHash hit is weighed by "
            "its pair cosine with the evaluation image it matched; a hit without one is a leak (fail closed)")
    from weed_optimizer_framework.tools.collect import intake as CI
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    dom, th = LS.load_domain("weed"), LS.load_thresholds()
    check("the thresholds, each with its why: confirm_cos 0.80, p_confirmed 6.29e-4 (the unconfirmed rate's "
          "one-sided 97.5 % bound), source_alpha 0.001",
          LS.t(th, "D28", "confirm_cos") == 0.8 and LS.t(th, "D28", "p_confirmed") == 0.000629
          and LS.t(th, "D28", "source_alpha") == 0.001
          and all(th["D28"][k].get("why") for k in ("confirm_cos", "p_confirmed", "source_alpha")), th["D28"])
    check("the dHash reasons are one list: D28's, inc2.eval_hits' and the collector's",
          tuple(DS.D28_DHASH_REASONS) == tuple(EH.DHASH_HIT_REASONS) == tuple(CI.DHASH_EVAL_REASONS))
    lock = {"splits/v2/lock_status.json": json.dumps({"splits_version": "v2", "locked": True, "embed_calibration_v2": {
        "p_false": 0.0105, "cos_threshold": COPY_COS}})}

    def intake(src, images, hits, pair_cos=None, record=True, copy_t=COPY_COS, embed=0):
        reasons = {"near_eval_variant": hits, "near_eval_embed": embed}
        doc = {"format": "collect-summary/1", "source": src, "batch": "b", "images": images, "guard": reasons,
               "source_leak": {"eval_share": round((hits + embed) / float(images), 4), "base_share": 0.0},
               "copy_scan": {"checked": True, "p_false": 0.0105, "cos_threshold": COPY_COS}}
        if record:
            keys = [{"key": "%s/%d" % (src, i), "source": src} for i in range(hits)]
            scored = {"%s/%d" % (src, i): {"pair_cos": c} for i, c in enumerate(pair_cos or [])}
            doc["eval_hits"] = EH.record(keys, scored, "facebook/dinov2-base:cls", copy_t)
        return {"intake/i_%s/summary.json" % src: json.dumps(doc)}

    def step1(rows):
        per = {}
        for src, (seen, hits, cos) in rows.items():
            r = {"images_seen": seen, "near_eval_embed": 0, "decision:near_eval_variant": hits,
                 "decision:pool": seen - hits}
            if cos is not None:
                r.update(eval_hits_scored=len(cos), eval_hit_pair_cos=sorted(cos, reverse=True),
                         eval_hit_copy_threshold=COPY_COS)
            per[src] = r
        return {"step1_stream/status.json": json.dumps({"format": "inc2-step1-stream/status/1", "per_source": per})}

    def d28(*parts, with_lock=True):
        texts = dict(lock) if with_lock else {}
        for p in parts:
            texts.update(p)
        ev = E.from_texts(texts, "x", context={"sid": "none"})
        return DS.by_id(DS.detect(ev, dom, th, only=("D28",)))["D28"]

    def dv(d, src):
        for h in ((d.get("detail") or {}).get("leaks") or []) + ((d.get("detail") or {}).get("cleared") or []):
            if h.get("source") == src:
                return (h.get("verdict") or {}).get("dhash") or h.get("dhash") or {}
        return {}

    # the 12 live sources of 2026-10-01: 49 hits in all, 1-23 per source, pair cos -0.05..0.66 (median about 0.1)
    # in the 11 Step 1 sources and 0.744 in the one intake source
    cos48 = [round(-0.05 + 0.71 * (i / 47.0) ** 2.2, 6) for i in range(48)]
    cos48 = [cos48[(7 * i) % 48] for i in range(48)]
    sizes = [20000, 9000, 6000, 5000, 4000, 3000, 2500, 2000, 1500, 800, 400]
    counts = [23, 6, 4, 3, 3, 2, 2, 2, 1, 1, 1]
    rows, k = {}, 0
    for i, (n, h) in enumerate(zip(sizes, counts)):
        rows["live_%02d" % i] = (n, h, cos48[k:k + h])
        k += h
    live = [step1(rows), intake("live_intake", 614, 1, [0.744])]
    d = d28(*live)
    cl = (d.get("detail") or {}).get("cleared") or []
    check("the 12 live-like sources (49 dHash hits, every pair cos < 0.80, max 0.744) do not quarantine: each is "
          "judged chance, 0 confirmed, P = 1", not d["fired"] and len(cl) == 12 and sum(c["dhash"]["hits"] for c in cl)
          == 49 and all(c["dhash"]["verdict"] == "chance" and c["dhash"]["confirmed"] == 0
                        and c["dhash"]["p_value"] == 1.0 for c in cl), (d.get("summary"), cl[:2]))
    check("  and the diagnosis states, for each source, its hits, confirmed hits, max pair cos, P and verdict",
          "live_00: 23 dHash hit(s) in 20000 images, 0 confirmed (pair cos >= 0.8), max pair cos %.3f, P = 1 -> chance"
          % max(rows["live_00"][2]) in d["summary"] and "live_intake: 1 dHash hit(s) in 614 images, 0 confirmed "
          "(pair cos >= 0.8), max pair cos 0.744, P = 1 -> chance" in d["summary"]
          and d["summary"].count("-> chance") == 12 and max(cos48) == 0.66 and sorted(cos48)[24] < 0.12,
          d.get("summary"))
    old = d28(step1({s: (n, h, None) for s, (n, h, _c) in rows.items()}),
              intake("live_intake", 614, 1, record=False))
    check("  without the pair cosines (the old rule's input) the same 12 quarantine: fail closed",
          old["fired"] and len(old["detail"]["leaks"]) == 12
          and all(h["verdict"]["dhash"]["fail_closed"] for h in old["detail"]["leaks"]), old.get("summary"))
    # a planted copy: one hit at pair cos 0.95, at or above the copy threshold
    d = d28(intake("planted", 614, 1, [0.95]))
    check("a planted hit at pair cos 0.95 (>= the v2 copy threshold 0.946384) quarantines: D28 -> L24",
          d["fired"] and "L24" in d["levers"] and dv(d, "planted").get("copy_hits") == 1
          and dv(d, "planted").get("verdict") == "leak"
          and "planted (1 dHash hit(s) in 614 images, 1 confirmed (pair cos >= 0.8), max pair cos 0.950" in d["summary"]
          and "at or above the copy threshold 0.946384" in d["summary"], d.get("summary"))
    d = d28(step1({"rf_planted": (5000, 2, [0.95, 0.10])}))
    check("  and so in Step 1's per-source counts", d["fired"] and dv(d, "rf_planted").get("copy_hits") == 1,
          d.get("summary"))
    d = d28(intake("one_confirmed", 1000, 1, [0.94]))
    check("one confirmed hit below the copy threshold (pair cos 0.94) in 1,000 images is chance (P = 0.47 >= 0.001): "
          "its image is dropped, its source is not quarantined", not d["fired"]
          and dv(d, "one_confirmed").get("confirmed") == 1 and abs(dv(d, "one_confirmed")["p_value"] - 0.467) < 0.01,
          d.get("summary"))
    # many confirmed hits: improbable by chance
    d = d28(intake("many", 2000, 9, [0.82, 0.83, 0.85, 0.86, 0.88, 0.9, 0.91, 0.93, 0.2]))
    check("8 confirmed hits (pair cos 0.82-0.93, all below the copy threshold) in 2,000 images, 1.26 expected: "
          "P < 0.001 -> a leak by the binomial rule", d["fired"] and dv(d, "many").get("confirmed") == 8
          and dv(d, "many").get("copy_hits") == 0 and dv(d, "many")["p_value"] < 0.001, d.get("summary"))
    d = d28(intake("few", 2000, 3, [0.82, 0.85, 0.9]))
    check("  3 confirmed in 2,000 is chance (P >= 0.001)", not d["fired"] and dv(d, "few").get("confirmed") == 3,
          d.get("summary"))
    # fail closed
    d = d28(intake("bare", 614, 1, record=False))
    check("a batch with a dHash hit and no pair-cosine record falls back to the one-hit rule: quarantined",
          d["fired"] and dv(d, "bare").get("fail_closed") and dv(d, "bare").get("verdict") == "leak (fail closed)",
          d.get("summary"))
    d = d28(intake("half", 614, 2, [0.1]))
    check("  so does one whose record weighs fewer hits than the guard counted (1 of 2)",
          d["fired"] and dv(d, "half").get("fail_closed") and "only 1 with a pair cosine" in d["summary"],
          d.get("summary"))
    d = d28(step1({"rf_old": (3000, 3, [0.1, 0.2])}))
    check("  and a Step 1 source with hits from a batch that recorded none (3 hits, 2 pair cosines)",
          d["fired"] and dv(d, "rf_old").get("fail_closed"), d.get("summary"))
    # an SIU-sized source
    cos95 = [round(-0.05 + 0.79 * i / 94.0, 6) for i in range(95)]
    d = d28(intake("siu_like", 200000, 95, cos95))
    check("an SIU-sized source (200,000 images) with 95 chance hits, all below 0.80, does not quarantine "
          "(about 126 confirmed hits would be expected by chance)", not d["fired"]
          and dv(d, "siu_like").get("confirmed") == 0 and dv(d, "siu_like").get("expected_confirmed") == 125.8,
          d.get("summary"))
    d = d28(intake("siu_some", 200000, 95, cos95[:92] + [0.81, 0.85, 0.9]))
    check("  nor with 3 of them confirmed (P = 1)", not d["fired"] and dv(d, "siu_some").get("confirmed") == 3,
          d.get("summary"))
    old = d28(intake("siu_like", 200000, 95, record=False))
    check("  under the old rule (no record) it would have been quarantined", old["fired"], old.get("summary"))
    # the copy threshold: the producer's record, else the LOCK's; with neither, confirm_cos stands in
    d = d28(intake("nothr", 1000, 1, [0.85], copy_t=None), with_lock=False)
    check("with no copy threshold known (no record, no LOCK), confirm_cos stands in: a hit at 0.85 is a leak "
          "(never less strict)", d["fired"] and dv(d, "nothr").get("copy_cos") == 0.8, d.get("summary"))
    d = d28(intake("nothr", 1000, 1, [0.85], copy_t=None))
    check("  with the LOCK's v2 threshold it is chance", not d["fired"] and dv(d, "nothr").get("copy_cos") == COPY_COS,
          d.get("summary"))
    check("an embedding hit keeps its own rule beside the dHash verdict (1 in 15 at the LOCK's rate: chance; 5 in "
          "15: a leak)", not d28(intake("e1", 15, 0, [], embed=1))["fired"]
          and d28(intake("e5", 15, 0, [], embed=5))["fired"])


def t_d28_v2_sources():
    section("D28-v2, round 2: a source is judged over all its intake batches; the boundaries; the sidecars of "
            "batches committed before the amendment; every source stated; DR0's L17 eval-hits")
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    dom, th = LS.load_domain("weed"), LS.load_thresholds()

    def lock(t=COPY_COS):
        return {"splits/v2/lock_status.json": json.dumps({"splits_version": "v2", "locked": True,
                                                          "embed_calibration_v2": {"p_false": 0.0105,
                                                                                   "cos_threshold": t}})}

    def summary(src, batch, images, hits, pair_cos=None, record=True, copy_t=COPY_COS):
        doc = {"format": "collect-summary/1", "source": src, "batch": batch, "images": images,
               "guard": {"near_eval_variant": hits, "near_eval_embed": 0},
               "source_leak": {"eval_share": round(hits / float(images), 4), "base_share": 0.0},
               "copy_scan": {"checked": True, "p_false": 0.0105, "cos_threshold": COPY_COS}}
        if record:
            keys = [{"key": "%s/%s/%d" % (src, batch, i), "source": src} for i in range(hits)]
            scored = {"%s/%s/%d" % (src, batch, i): {"pair_cos": c} for i, c in enumerate(pair_cos or [])}
            doc["eval_hits"] = EH.record(keys, scored, "facebook/dinov2-base:cls", copy_t)
        return {"intake/%s/summary.json" % batch: json.dumps(doc)}

    def sidecar(src, batch, hits, pair_cos, of=None):
        keys = [{"key": "%s/%s/%d" % (src, batch, i), "source": src} for i in range(hits)]
        scored = {"%s/%s/%d" % (src, batch, i): {"pair_cos": c} for i, c in enumerate(pair_cos)}
        rec = EH.record(keys, scored, "facebook/dinov2-base:cls", COPY_COS)
        return {"intake/%s/eval_hits.json" % batch: json.dumps(EH.sidecar(of or batch, rec, [], "intake",
                                                                           source=src))}

    def d28(*parts, t=COPY_COS, context=None, only="D28"):
        texts = dict(lock(t))
        for x in parts:
            texts.update(x)
        ev = E.from_texts(texts, "x", context=dict({"sid": "none"}, **(context or {})))
        return DS.by_id(DS.detect(ev, dom, th, only=(only,)))[only]

    def dv(d, src):
        for h in ((d.get("detail") or {}).get("leaks") or []) + ((d.get("detail") or {}).get("cleared") or []):
            if h.get("source") == src:
                return (h.get("verdict") or {}).get("dhash") or h.get("dhash") or {}
        return {}

    # 1. shards: one source in 5 batches of 1,000 images, 3, 3, 2, 2, 2 confirmed hits at 0.85
    counts = (3, 3, 2, 2, 2)
    shards = [summary("shardy", "i%04d_shardy" % (i + 1), 1000, k, [0.85] * k) for i, k in enumerate(counts)]
    d = d28(*shards)
    v = dv(d, "shardy")
    leak = next((h for h in (d.get("detail") or {}).get("leaks") or [] if h["source"] == "shardy"), {})
    check("a source split into 5 intake batches of 1,000 images with 3, 3, 2, 2, 2 confirmed hits (pair cos 0.85) "
          "is judged as one source: 12 confirmed in 5,000 images, 3.1 expected, P ~ 1.1e-4 < 0.001 -> a leak",
          d["fired"] and v.get("images") == 5000 and v.get("hits") == 12 and v.get("confirmed") == 12
          and 0.5e-4 < v.get("p_value", 1) < 2e-4 and leak.get("batches") == ["i%04d_shardy" % (i + 1)
                                                                               for i in range(5)]
          and len([c for c in d["cites"] if c.get("pointer") == "/source"]) == 5, (v, d.get("summary")))
    alone = [d28(x) for x in shards]
    check("  while each batch alone is chance (P 0.026 or 0.13): the per-batch reading let the shards escape",
          not any(x["fired"] for x in alone) and all(0.02 < dv(x, "shardy")["p_value"] < 0.2 for x in alone),
          [dv(x, "shardy").get("p_value") for x in alone])
    d = d28(*(shards[:4] + [summary("shardy", "i0005_shardy", 1000, 2, record=False)]))
    v = dv(d, "shardy")
    check("  one shard with hits and no pair-cosine record fails the whole source closed (only 10 of 12 weighed), "
          "and its row states the confirmed hits it has", d["fired"] and v.get("fail_closed") and v.get("scored") == 10
          and v.get("confirmed") == 10 and "only 10 with a pair cosine, 10 confirmed among them" in d["summary"],
          (v, d.get("summary")))
    two = [summary("dup2", "i0001_dup2", 1000, 1, [0.97], copy_t=0.95),
           summary("dup2", "i0002_dup2", 1000, 1, [0.1], copy_t=0.99)]
    d = d28(*two, t=0.999)
    check("  the copy threshold of a source is the lowest any of its batches recorded (0.95 of 0.95, 0.99 and the "
          "LOCK's 0.999): its hit at 0.97 is a copy", d["fired"] and dv(d, "dup2").get("copy_cos") == 0.95
          and dv(d, "dup2").get("copy_hits") == 1, d.get("summary"))
    # 4. the boundaries, exactly
    d = d28(summary("at080", "i0001_at080", 2000, 8, [0.8] * 8))
    check("8 hits exactly at confirm_cos 0.80 in 2,000 images are confirmed: P < 0.001 -> a leak (>=, not >)",
          d["fired"] and dv(d, "at080").get("confirmed") == 8 and dv(d, "at080")["p_value"] < 0.001, d.get("summary"))
    d = d28(summary("atcopy", "i0001_atcopy", 1000, 1, [COPY_COS]))
    check("one hit exactly at the copy threshold 0.946384 is a copy: a leak (>=, not >)",
          d["fired"] and dv(d, "atcopy").get("copy_hits") == 1 and dv(d, "atcopy").get("confirmed") == 1,
          d.get("summary"))
    d = d28(summary("lowt", "i0001_lowt", 1000, 1, [0.948], copy_t=0.95))
    check("the producer recorded 0.95, the LOCK 0.946384: a hit at 0.948 is a copy by the lower of the two (a leak)",
          d["fired"] and dv(d, "lowt").get("copy_cos") == COPY_COS and dv(d, "lowt").get("copy_hits") == 1,
          d.get("summary"))
    d = d28(summary("lowt2", "i0001_lowt2", 1000, 1, [0.948], copy_t=COPY_COS), t=0.95)
    check("  and the other way round (record 0.946384, LOCK 0.95): still a leak", d["fired"]
          and dv(d, "lowt2").get("copy_cos") == COPY_COS, d.get("summary"))
    # 2. the sidecar of a batch committed before the amendment
    old = summary("legacy", "i0003_legacy", 614, 1, record=False)
    d = d28(old)
    check("an intake batch committed before the amendment (no pair cosines) quarantines by the one-hit rule",
          d["fired"] and dv(d, "legacy").get("fail_closed"), d.get("summary"))
    d = d28(old, sidecar("legacy", "i0003_legacy", 1, [0.744]))
    check("  its sidecar intake/<batch>/eval_hits.json (collect.intake.rescore_eval_hits) weighs the hit again: "
          "pair cos 0.744 -> chance, not quarantined, and the summary states it",
          not d["fired"] and dv(d, "legacy").get("verdict") == "chance" and dv(d, "legacy").get("max_pair_cos") == 0.744
          and "legacy: 1 dHash hit(s) in 614 images, 0 confirmed (pair cos >= 0.8), max pair cos 0.744, P = 1 -> "
              "chance" in d["summary"], d.get("summary"))
    d = d28(old, sidecar("legacy", "i0003_legacy", 1, [0.97]))
    check("  a sidecar that finds a copy (0.97) quarantines, and D28 cites the sidecar", d["fired"]
          and dv(d, "legacy").get("copy_hits") == 1
          and any(c.get("artifact") == "intake/i0003_legacy/eval_hits.json" for c in d["cites"]), d["cites"][-3:])
    for why, side in (("another number of hits (2 for the guard's 1)",
                       sidecar("legacy", "i0003_legacy", 2, [0.1, 0.2])),
                      ("another batch", sidecar("legacy", "i0003_legacy", 1, [0.1], of="i0009_other"))):
        d = d28(old, side)
        check("  a sidecar that weighed %s is not used: fail closed" % why, d["fired"]
              and dv(d, "legacy").get("fail_closed"), d.get("summary"))
    rec_ok, _w = EH.usable_sidecar(json.loads(list(sidecar("s", "b", 2, [0.1, 0.2]).values())[0]), "b", {"s": 2})
    rec_bad, why_bad = EH.usable_sidecar({"format": "x"}, "b", {"s": 2})
    check("inc2.eval_hits.usable_sidecar: the batch and the per-source hit counts must match, the format must be "
          "the sidecar's", rec_ok is not None and rec_bad is None and "not an" in why_bad, why_bad)
    # 7. every source stated; the quarantined ones D28 now clears are named for a person
    many = [summary("src%02d" % i, "i%04d_src%02d" % (i + 1, i), 900, 1, [0.1 + 0.01 * i]) for i in range(25)]
    d = d28(*many)
    check("25 sources judged chance: the summary states every one of them, no 'and N more'",
          not d["fired"] and d["summary"].count("-> chance") == 25 and not re.search(r"and \d+ more", d["summary"])
          and len(d["detail"]["cleared"]) == 25, d.get("summary")[-300:])
    d = d28(old, sidecar("legacy", "i0003_legacy", 1, [0.744]),
            context={"sources": {"legacy": {"status": "quarantined", "cite": "D28"}}})
    check("a source quarantined under the one-hit rule that D28-v2 clears is named for a person (inc2.stream "
          "unquarantine): the quarantine is a person's to lift", not d["fired"]
          and d["detail"].get("cleared_quarantined") == ["legacy"] and "unquarantine" in d["summary"]
          and "legacy" in d["summary"].split("Quarantined, now judged chance")[-1], d.get("summary"))
    # DR0: the platform runs the sidecars (L17 eval-hits), once the one-time jobs are done
    stage = {"lock": True, "placement": True, "probe_ran": True, "baselines": {}, "exp_status": {},
             "step1_stream": {"bootstrap": True, "knowntruth": True, "backfill": True}}
    st_due = {"step1_stream/status.json": json.dumps({"format": "inc2-step1-stream/status/1", "per_source": {},
                                                      "eval_hits": {"due": ["b0000"], "sidecars": {}}})}
    d = d28(old, st_due, context={"stage": stage}, only="DR0")
    data = ((d.get("detail") or {}).get("due") or {}).get("DATA") or {}
    check("DR0 proposes L17 eval-hits when Step 1's b0000 and an intake batch hold hits no record weighs and no "
          "sidecar exists yet", d["fired"] and data.get("lever") == "L17" and data.get("verb") == "eval-hits"
          and "Step 1 batch b0000" in data.get("why", "") and "intake batch i0003_legacy" in data.get("why", ""),
          (data, d.get("summary")))
    st_done = {"step1_stream/status.json": json.dumps({"format": "inc2-step1-stream/status/1", "per_source": {},
                                                       "eval_hits": {"due": [], "sidecars": {"b0000": {
                                                           "used": True}}}})}
    d = d28(old, sidecar("legacy", "i0003_legacy", 1, [0.744]), st_done, context={"stage": stage}, only="DR0")
    data = ((d.get("detail") or {}).get("due") or {}).get("DATA")
    check("  and not once every such batch has its sidecar", data is None or data.get("verb") != "eval-hits",
          (data, d.get("summary")))
    st_old = {"step1_stream/status.json": json.dumps({"format": "inc2-step1-stream/status/1", "per_source": {
        "rf_old": {"images_seen": 3000, "decision:near_eval_variant": 2, "decision:pool": 2998}}})}
    d = d28(st_old, context={"stage": stage}, only="DR0")
    data = ((d.get("detail") or {}).get("due") or {}).get("DATA") or {}
    check("  and from a status.json written before the sidecars existed (no eval_hits section) whose source rows "
          "count dHash hits without pair cosines: eval-hits finds the batches itself", data.get("verb") == "eval-hits"
          and "rf_old" in data.get("why", ""), (data, d.get("summary")))
    d = d28(old, st_due, context={"stage": dict(stage, step1_stream={"bootstrap": True, "knowntruth": True,
                                                                      "backfill": False})}, only="DR0")
    data = ((d.get("detail") or {}).get("due") or {}).get("DATA") or {}
    check("  nor before Step 1's one-time jobs are done (backfill first)", data.get("verb") == "backfill", data)


def t_d28_v2_round3():
    section("D28-v2, round 3: one verdict per source across intake and Step 1; a source's batches combined "
            "(pair-cosine cap, embedding rate, base-copy share); DR0 never proposes a batch with a sidecar again")
    from weed_optimizer_framework.tools.inc2 import eval_hits as EH
    dom, th = LS.load_domain("weed"), LS.load_thresholds()

    def lock(p=0.0105):
        ec = {"cos_threshold": COPY_COS}
        if p is not None:
            ec["p_false"] = p
        return {"splits/v2/lock_status.json": json.dumps({"splits_version": "v2", "locked": True,
                                                          "embed_calibration_v2": ec})}

    def summary(src, batch, images, hits, pair_cos=None, rec_hits=None, embed=0, p=0.0105, base_share=0.0):
        """An intake summary: hits dHash hits the guard counted, a record weighing rec_hits (default hits) of
        them with pair_cos (None: no record), embed embedding hits judged under p (None: no copy-scan
        record), and its base-copy share."""
        doc = {"format": "collect-summary/1", "source": src, "batch": batch, "images": images,
               "guard": {"near_eval_variant": hits, "near_eval_embed": embed},
               "source_leak": {"eval_share": round((hits + embed) / float(images), 4), "base_share": base_share},
               "copy_scan": ({"checked": True, "p_false": p, "cos_threshold": COPY_COS} if p is not None
                             else {"checked": False})}
        if pair_cos is not None:
            n = len(pair_cos) if rec_hits is None else rec_hits
            keys = [{"key": "%s/%s/%d" % (src, batch, i), "source": src} for i in range(n)]
            doc["eval_hits"] = EH.record(keys, {"%s/%s/%d" % (src, batch, i): {"pair_cos": c}
                                                for i, c in enumerate(pair_cos)}, "facebook/dinov2-base:cls",
                                         COPY_COS)
        return {"intake/%s/summary.json" % batch: json.dumps(doc)}

    def sidecar(src, batch, hits, pair_cos, of=None):
        keys = [{"key": "%s/%s/%d" % (src, batch, i), "source": src} for i in range(hits)]
        rec = EH.record(keys, {"%s/%s/%d" % (src, batch, i): {"pair_cos": c} for i, c in enumerate(pair_cos)},
                        "facebook/dinov2-base:cls", COPY_COS)
        return {"intake/%s/eval_hits.json" % batch: json.dumps(EH.sidecar(of or batch, rec, [], "intake",
                                                                           source=src))}

    def status(per_source, due=None):
        doc = {"format": "inc2-step1-stream/status/1", "per_source": per_source}
        if due is not None:
            doc["eval_hits"] = {"due": due, "sidecars": {}}
        return {"step1_stream/status.json": json.dumps(doc)}

    def run(*parts, context=None, only="D28"):
        texts = dict(lock())
        for x in parts:
            texts.update(x)
        ev = E.from_texts(texts, "x", context=dict({"sid": "none"}, **(context or {})))
        return DS.by_id(DS.detect(ev, dom, th, only=(only,)))[only]

    def srcs(d, key):
        return [x["source"] if isinstance(x, dict) else x for x in (d.get("detail") or {}).get(key) or []]

    def leak_of(d, src):
        return next((h for h in (d.get("detail") or {}).get("leaks") or [] if h["source"] == src), {})

    quarantined = {"sources": {"S": {"status": "quarantined", "cite": "D28"},
                               "T": {"status": "quarantined", "cite": "D28"}}}
    # 1. N1: a source judged chance in one path and a leak in the other is a leak, never cleared
    s_chance = summary("S", "i0001_S", 100, 1, [0.3])
    s_leak = status({"S": {"images_seen": 100, "near_eval_embed": 20, "decision:pool": 80,
                           "decision:near_eval_embed": 20}})
    d = run(s_chance, s_leak, context=quarantined)
    check("a quarantined source whose intake dHash hit is chance (0.30) but whose Step 1 embedding hits leak (20 "
          "in 100) is a leak: not in detail.cleared nor cleared_quarantined, and no unquarantine is suggested",
          d["fired"] and srcs(d, "leaks") == ["S"] and srcs(d, "cleared") == [] and
          d["detail"].get("cleared_quarantined") == [] and "Quarantined, now judged chance" not in d["summary"],
          (d["detail"].get("cleared"), d["detail"].get("cleared_quarantined"), d.get("summary")))
    check("  its intake numbers are stated with the leak (requirement 5)",
          "its intake dHash hits alone: 1 dHash hit(s) in 100 images" in d["summary"]
          and (leak_of(d, "S").get("dhash_elsewhere") or [{}])[0].get("path") == "intake", d.get("summary"))
    d = run(summary("S", "i0001_S", 1000, 1, [0.97]), status({"S": {
        "images_seen": 500, "decision:near_eval_variant": 1, "decision:pool": 499, "eval_hit_pair_cos": [0.2],
        "eval_hit_copy_threshold": COPY_COS}}), context=quarantined)
    check("  and the other way round (an intake copy at 0.97, a chance Step 1 hit at 0.2): a leak, not cleared, "
          "its Step 1 numbers stated with the leak", d["fired"] and srcs(d, "leaks") == ["S"]
          and srcs(d, "cleared") == [] and d["detail"].get("cleared_quarantined") == []
          and "its Step 1 dHash hits alone: 1 dHash hit(s) in 500 images" in d["summary"],
          (d["detail"].get("cleared"), d.get("summary")))
    t_both = (summary("T", "i0001_T", 500, 1, [0.1]), status({"T": {
        "images_seen": 500, "near_eval_embed": 0, "decision:near_eval_variant": 1, "decision:pool": 499,
        "eval_hit_pair_cos": [0.2], "eval_hit_copy_threshold": COPY_COS}}))
    d = run(*t_both, context=quarantined)
    row = ((d.get("detail") or {}).get("cleared") or [{}])[0]
    check("a source judged chance in both intake and Step 1 is listed once (cleared, cleared_quarantined) and "
          "stated once with each path's numbers", not d["fired"] and srcs(d, "cleared") == ["T"]
          and d["detail"].get("cleared_quarantined") == ["T"] and d["summary"].count("T: ") == 1
          and "(intake) and 1 dHash hit(s) in 500 images" in d["summary"] and "(Step 1)" in d["summary"]
          and sorted(row.get("dhash_by_path") or {}) == ["Step 1", "intake"]
          and d["summary"].split("Quarantined, now judged chance")[-1].count("T") == 1, d.get("summary"))
    # 2. m8: a batch's pair cosines count at most its own hits
    d = run(summary("cap", "i0001_cap", 1000, 1, [0.1, 0.1], rec_hits=2),
            summary("cap", "i0002_cap", 1000, 1, None))
    v = (leak_of(d, "cap").get("verdict") or {}).get("dhash") or {}
    check("a batch whose record weighs more cosines (2) than its guard counted hits (1) lends none to another batch "
          "of the source that weighs none: 1 of 2 weighed, fail closed", d["fired"] and v.get("fail_closed")
          and v.get("scored") == 1 and v.get("hits") == 2, (v, d.get("summary")))
    # 3. m11: the embedding rule's rate is the lowest of the batches with embedding hits
    d = run(summary("emb", "i0001_emb", 100, 0, embed=1, p=0.0105), summary("emb", "i0002_emb", 100, 0, embed=2,
                                                                            p=0.0001))
    v = leak_of(d, "emb").get("verdict") or {}
    check("a source's embedding hits are judged at the lowest rate any of its batches was judged under (3 hits in "
          "200 at 1e-4: P ~ 1.3e-6, a leak; at 0.0105 it would be chance)", d["fired"] and v.get("p_false") == 0.0001
          and v.get("hits") == 3 and v.get("p_value", 1) < 1e-4, (v, d.get("summary")))
    # 4. m12: the base-copy share is the highest of any batch, not the last one read
    d = run(summary("base", "i0001_base", 400, 0, base_share=0.25), summary("base", "i0002_base", 400, 0,
                                                                            base_share=0.0))
    check("a source with 25 % base copies in its first batch and none in its second leaks by the 20 % rule (the "
          "highest share, not the last batch's)", d["fired"] and leak_of(d, "base").get("base_copy_share") == 0.25,
          d.get("summary"))
    # 5. m13: one batch with embedding hits and no rate fails the embedding rule closed
    d = run(summary("nop", "i0001_nop", 1000, 0, embed=1, p=0.0105), summary("nop", "i0002_nop", 1000, 0, embed=1,
                                                                             p=None), lock(None))
    v = leak_of(d, "nop").get("verdict") or {}
    check("one batch with embedding hits judged under no rate (no copy-scan record, no LOCK rate) fails the source's "
          "embedding rule closed, though 2 hits in 2,000 at the other batch's 0.0105 are chance",
          d["fired"] and v.get("p_false") is None and any("fail closed" in w for w in v.get("why") or []),
          (v, d.get("summary")))
    d = run(summary("nop", "i0001_nop", 1000, 0, embed=1, p=0.0105), summary("nop", "i0002_nop", 1000, 0, embed=1,
                                                                             p=0.0105), lock(None))
    check("  and with both rates recorded, the same hits are chance", not d["fired"], d.get("summary"))
    # 6. m14: a batch whose sidecar exists is not proposed again, whatever the sidecar weighed
    stage = {"lock": True, "placement": True, "probe_ran": True, "baselines": {}, "exp_status": {},
             "step1_stream": {"bootstrap": True, "knowntruth": True, "backfill": True}}
    old = summary("legacy", "i0003_legacy", 614, 1, None)
    for why, side in (("weighed none of its hit (partial)", sidecar("legacy", "i0003_legacy", 1, [])),
                      ("weighed another number of hits (unusable)", sidecar("legacy", "i0003_legacy", 2, [0.1, 0.2])),
                      ("names another batch (unusable)", sidecar("legacy", "i0003_legacy", 1, [0.1], of="i0009_x"))):
        d = run(old, side, status({}, due=[]), context={"stage": stage}, only="DR0")
        data = ((d.get("detail") or {}).get("due") or {}).get("DATA")
        d2 = run(old, side)
        check("an intake batch whose sidecar %s is not proposed for eval-hits again (one attempt per batch; a person "
              "re-runs it with --force), and D28 fails it closed" % why,
              (data is None or data.get("verb") != "eval-hits") and d2["fired"]
              and ((leak_of(d2, "legacy").get("verdict") or {}).get("dhash") or {}).get("fail_closed"),
              (data, d2.get("summary")))
    d = run(old, status({}, due=[]), context={"stage": stage}, only="DR0")
    data = ((d.get("detail") or {}).get("due") or {}).get("DATA") or {}
    check("  while the same batch without a sidecar is proposed", data.get("verb") == "eval-hits"
          and "intake batch i0003_legacy" in data.get("why", ""), data)


def _e1_items(dom):
    return [b for b in dom["baselines"]["items"] if b.get("requires") == "base3"]


def _e1_world(tag):
    """R0 complete, every other measurement arm built and rescored; splits v3 (E1's base), E1's arms and their
    agnostic rescore missing."""
    w = World(tag)
    w.ready_r0()
    (w.inc / "splits" / "v3" / "summary.json").unlink()
    for b in _e1_items(w.dom):
        shutil.rmtree(str(w.inc / b["exp"]))
    return w


def t_e1():
    section("E1 (2026-10-03): base v3's build (L23V), the arms on splits v3 with cold_budget (L23B, role baseline, "
            "priced from the budget), their agnostic rescore and verdict (L23E); record only")
    dom = LS.load_domain("weed")
    e1 = _e1_items(dom)
    check("the domain's E1 arms: e1_a on splits/v3/base_v2_weed.jsonl then e1_b on base_v3_weed.jsonl, m640, role "
          "baseline, 3 seeds, measure (record only), requires base3, never read at a native resolution, the budget "
          "1.2M image-epochs at 14.2 ms",
          [b["id"] for b in e1] == ["e1_a", "e1_b"]
          and [b["manifest"] for b in e1] == ["splits/v3/base_v2_weed.jsonl", "splits/v3/base_v3_weed.jsonl"]
          and all(b["arm"] == "m640" and b["role"] == "baseline" and b["seeds"] == "0,1,2" and b["measure"]
                  and b["native"] is False and not b.get("required")
                  and b["budget"] == {"image_epochs": 1200000, "ms_per_image_epoch": 14.2} for b in e1)
          and dom["e1"]["arms"] == {"A": "e1_a", "B": "e1_b"}, [(b["id"], b.get("manifest")) for b in e1])
    pa = {"pkg": "inc2", "exp": "e1_a_m640", "manifest": LS.inc_path("splits/v3/base_v2_weed.jsonl"),
          "seeds": "0,1,2", "arm": "m640", "role": "baseline"}
    est_a, det = LS.price("L23B", pa, dom, {"images": 6811, "budget": e1[0]["budget"]})
    est_b, _d = LS.price("L23B", dict(pa, exp="e1_b_m640"), dom, {"images": 30000, "budget": e1[1]["budget"]})
    want = 3 * 1.2e6 * 14.2 / 3.6e6 + LS.cost(dom, "finals_hours") + LS.cost(dom, "build_job_hours")
    old, _d = LS.price("L23B", pa, dom, {"images": 30000})
    check("an E1 arm is priced from its budget, whatever N: 3 x 1.2M x 14.2 ms + finals + the build job = %.2f GPU-h "
          "for 6,811 and for 30,000 images (the 100-epoch basis would say %.1f at 30,000)" % (est_a, old),
          abs(est_a - want) < 1e-6 and abs(est_b - want) < 1e-6 and det["estimator"] == "budget" and old > est_b, det)
    ev, dv = LS.price("L23V", PARAMS["L23V"], dom, {})
    ee, de = LS.price("L23E", PARAMS["L23E"], dom, {"runs": 6})
    okv, badv = LS.check_params("L23V", LS.policy_params("L23V", dict(PARAMS["L23V"], est_gpu_hours=ev)))
    check("L23V is one build job priced at its own walltime (%.1f GPU-h: run_inc2_build.sh's 12 h limit, not the other "
          "build jobs' %.1f), which its policy row admits; L23E six scoring passes (%.2f GPU-h)"
          % (ev, LS.cost(dom, "build_job_hours"), ee),
          ev == LS.cost(dom, "base3_job_hours") == 12.0 and dv["estimator"] == "base3_job" and okv and ee == 1.5,
          (dv, de, badv))
    ok, bad = LS.check_params("L23B", LS.policy_params("L23B", pa))
    back = X.params_from_argv("inc_build_baseline_v2", LS.render("L23B", LS.policy_params("L23B", pa)))
    check("the role enum admits 'baseline' (the policy row and the executor read L23B --role baseline back)",
          ok and back == LS.policy_params("L23B", pa), (bad, back))
    for args, want_ok, name in (
            (["inc2.base3", "build", "--stream", "weed_stream_v1"], True, "inc_build_base3_v3"),
            (["inc2.base3", "build"], False, None),
            (["inc2.base3", "build", "--stream", "weed_stream_v1", "--exp", "x"], False, None),
            (["inc2.baseline", "rescore-agnostic", "--exp", "e1_b_m640", "--reference", "e1_a_m640"], True,
             "inc_build_agnostic_e1_b_m640"),
            (["inc2.baseline", "rescore-agnostic", "--exp", "e1_b_m640"], False, None),
            (["inc2.baseline", "build", "--exp", "e1_a_m640", "--manifest", pa["manifest"], "--seeds", "0,1,2",
              "--arm", "m640", "--role", "baseline"], True, "inc_build_e1_a_m640")):
        try:
            req = SR.parse_submit("build", args)
            got, jn = True, SR.job_name(req, {})
        except R.Refused:
            got, jn = False, None
        check("the cluster's build grammar %s %s%s" % ("admits" if want_ok else "refuses", " ".join(args[:2]),
                                                       " (job %s)" % name if name else ""),
              got == want_ok and (name is None or jn == name), (got, jn))
    check("the evidence allow-lists splits/v3/summary.json, <exp>/agnostic_rescore.json and capacity/e1_v1.json, "
          "never the report, a manifest or the holdout lists",
          E.allowed("splits/v3/summary.json") and E.allowed("e1_b_m640/agnostic_rescore.json")
          and E.allowed("capacity/e1_v1.json") and not E.allowed("capacity/e1_v1_report.json")
          and not E.allowed("splits/v3/base_v3_weed.jsonl") and not E.allowed("splits/v3/test_v1/a.jsonl"))
    w = _e1_world("e1")
    cap0 = (w.inc / "capacity" / "capacity_v1.json").read_bytes()
    w.tick(3)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23V"]
    ex = [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23V"]
    first = pro[0] if pro else {}
    check("R0 complete, the other measurement arms built and rescored, splits v3 missing: DR0 proposes the base v3 "
          "build once (L23V inc2.base3 build --stream SID), within the envelope, citing only the lock and /stage/base3; "
          "no E1 arm before it",
          len(pro) == 1 and (first.get("argv") or [])[-4:] == ["weed_optimizer_framework.tools.inc2.base3", "build",
                                                                 "--stream", w.sid] and ex == ["envelope"]
          and sorted(c.get("pointer") for c in first.get("cites") or []) == ["/stage/base3", "/stage/lock"]
          and not [e for e in w.events("proposed") if e.get("lever") == "L23B"],
          ([(e.get("argv") or [])[-4:] for e in pro], ex, first.get("cites")))
    check("  one GPU-shared run_inc2_build.sh job under its own name, inc_build_base3_v3",
          [x["name"] for x in w.submits if "inc2.base3" in x["argv"]] == ["inc_build_base3_v3"]
          and all("GPU-shared" in x["argv"] for x in w.submits if "inc2.base3" in x["argv"]))
    w.tick(3)
    check("  while it runs it is not proposed again, and no E1 arm is (they require splits v3)",
          len([e for e in w.events("proposed") if e.get("lever") == "L23V"]) == 1
          and w.state()["stage"]["r0"].get("base3") == "running"
          and not [e for e in w.events("proposed") if e.get("lever") == "L23B"])
    w.job_done("inc_build_base3_v3")
    w.base3_summary()
    w.tick(3)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23B"]
    first = pro[0] if pro else {}
    check("splits v3 complete: E1-A is proposed (L23B --manifest splits/v3/base_v2_weed.jsonl --arm m640 --role "
          "baseline), priced from the budget (%s GPU-h), within the envelope" % first.get("est_gpu_hours"),
          [e.get("child_exp") for e in pro] == ["e1_a_m640"]
          and (first.get("argv") or [])[-8:] == ["--manifest", pa["manifest"], "--seeds", "0,1,2", "--arm", "m640",
                                                  "--role", "baseline"]
          and abs(float(first.get("est_gpu_hours") or 0) - want) < 1e-6
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23B"] == ["envelope"]
          and "/stage/base3" in [c.get("pointer") for c in first.get("cites") or []],
          ([(e.get("child_exp"), (e.get("argv") or [])[-8:]) for e in pro], first.get("est_gpu_hours")))
    w.experiment("e1_a_m640", done=False)
    w.job_done("inc_build_e1_a_m640")
    w.tick(3)
    pro = [e.get("child_exp") for e in w.events("proposed") if e.get("lever") == "L23B"]
    check("then E1-B (base_v3_weed.jsonl), once", pro == ["e1_a_m640", "e1_b_m640"], pro)
    w.experiment("e1_b_m640", done=False)
    w.job_done("inc_build_e1_b_m640")
    w.tick(3)
    check("  while the arms run: no rescore of either (no L23N: native false; no L23E: not done)",
          not [e for e in w.events("proposed") if e.get("lever") in ("L23N", "L23E")])
    w.experiment("e1_a_m640", done=True)
    w.tick(3)
    check("  E1-A done, E1-B still running: no L23E (it needs both arms done)",
          not [e for e in w.events("proposed") if e.get("lever") in ("L23N", "L23E")])
    w.experiment("e1_b_m640", done=True)
    w.tick(3)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23E"]
    first = pro[0] if pro else {}
    check("both done: their agnostic rescore and E1's verdict, once (L23E rescore-agnostic --exp e1_b_m640 "
          "--reference e1_a_m640, 1.5 GPU-h), within the envelope; still no L23N for them",
          [(e.get("argv") or [])[-5:] for e in pro] == [["rescore-agnostic", "--exp", "e1_b_m640", "--reference",
                                                         "e1_a_m640"]]
          and abs(float(first.get("est_gpu_hours") or 0) - 1.5) < 1e-9
          and [e.get("basis") for e in w.events("executed") if e.get("lever") == "L23E"] == ["envelope"]
          and [x["name"] for x in w.submits if "rescore-agnostic" in x["argv"]] == ["inc_build_agnostic_e1_b_m640"]
          and not [e for e in w.events("proposed") if e.get("lever") == "L23N"],
          ([(e.get("argv") or [])[-5:] for e in pro], first.get("est_gpu_hours")))
    w.job_done("inc_build_agnostic_e1_b_m640")
    w.agnostic_record("e1_b_m640")
    w.tick(3)
    d = _diags(w)
    check("its record complete: nothing of E1 is proposed again (DR0 silent); the stream's arm and capacity_v1.json "
          "are unchanged, no LA",
          len([e for e in w.events("proposed") if e.get("lever") == "L23E"]) == 1 and not d["DR0"]["fired"]
          and w.state()["capacity"]["chosen"] == "n640"
          and (w.inc / "capacity" / "capacity_v1.json").read_bytes() == cap0
          and not [e for e in w.events("proposed") if e.get("lever") == "LA"], d["DR0"]["summary"])
    wf = _e1_world("e1_fail")
    wf.tick(3)
    wf.job_done("inc_build_base3_v3", state="FAILED", refusal="[inc2.base3] ERROR: refused")
    wf.tick(3)
    st = wf.state()
    cards = [c["title"] for c in st.get("cards") or []]
    check("the base v3 build FAILED: one card, never a pause or a held lane, it stays failed (not proposed again) and "
          "no E1 arm is built",
          not st.get("paused") and wf.config().get("enabled") is True
          and cards == ["Base v3 build (splits v3, E1) failed (L23V)"]
          and len([e for e in wf.events("proposed") if e.get("lever") == "L23V"]) == 1
          and st["stage"]["r0"].get("base3") == "failed" and not wf.lane("MAINT").get("hold")
          and not int(wf.lane("MAINT").get("fails") or 0)
          and not [e for e in wf.events("proposed") if e.get("lever") == "L23B"], (cards, st["stage"]["r0"]))
    for tag, status in (("e1_wall", "over_walltime"), ("e1_nosum", None)):
        ww = _e1_world(tag)
        ww.tick(3)
        ww.job_done("inc_build_base3_v3")
        if status:
            ww.base3_summary(status=status)
        ww.tick(3)
        d = _diags(ww)
        check("the base v3 job ended %s: splits v3 is not done, so no E1 arm is built and L23V is not proposed again "
              "(DR0 silent)" % ("with summary.json status over_walltime" if status else "without a summary.json"),
              not [e for e in ww.events("proposed") if e.get("lever") == "L23B"]
              and len([e for e in ww.events("proposed") if e.get("lever") == "L23V"]) == 1
              and not d["DR0"]["fired"], ([e.get("lever") for e in ww.events("proposed")], d["DR0"]["summary"]))
    wu = _e1_world("e1_uncertain")
    wu.lose_reply = "inc2.base3"
    wu.tick(2)
    it = wu.lane("MAINT").get("item") or {}
    check("an L23V submission whose outcome is unknown pauses nothing: it is followed by its job name",
          not wu.state().get("paused") and it.get("lever") == "L23V" and it.get("status") == "running"
          and it.get("uncertain"), (it.get("lever"), it.get("status"), wu.state().get("paused")))
    wu.job_done("inc_build_base3_v3")
    wu.base3_summary()
    wu.tick(3)
    check("  and done once splits/v3/summary.json says complete (E1-A follows)",
          any(e.get("lever") == "L23V" for e in wu.events("item_done"))
          and [e.get("child_exp") for e in wu.events("proposed") if e.get("lever") == "L23B"] == ["e1_a_m640"],
          [e.get("lever") for e in wu.events("proposed")])


def _lift_world(tag, quarantined=("src_lift",)):
    """_e1_world plus an intake batch of src_lift whose one dHash hit D28-v2 judges chance (pair cos 0.31), and
    the stream's queue summary quarantining the given sources."""
    w = _e1_world(tag)
    w.intake("i0007_src_lift", "src_lift", images=900, reasons={"near_eval_variant": 1}, pair_cos=[0.31])
    w.queue(0, extra={"quarantined_sources": {x: {"cite": "D28", "seq": 3, "utc": "2026-10-01T00:00:00Z"}
                                              for x in quarantined}})
    return w


def t_e1_lift_wait():
    section("E1 (2026-10-03): L23V waits, at most 12 h from when the stream first saw the list, for a person to lift "
            "the quarantines D28 now judges chance; one card names them and the exact command")
    th = LS.load_thresholds()
    check("the bound is pre-registered with its why: D28.lift_wait_hours = 12",
          LS.t(th, "D28", "lift_wait_hours") == 12 and th["D28"]["lift_wait_hours"].get("why"))
    w = _lift_world("lift")
    w.tick(3)
    d = _diags(w)
    st = w.state()
    lw = (st["stage"]["r0"] or {}).get("lift_wait") or {}
    cards = [c for c in st.get("cards") or [] if c.get("kind") == "quarantine_lift"]
    wait = (d["DR0"].get("detail") or {}).get("lift_wait") or {}
    cmd = ("python -m weed_optimizer_framework.tools.inc2.stream unquarantine --source src_lift --stream %s "
           "--decided-by human:<id>" % w.sid)
    check("D28 judges src_lift's hit chance and the stream still quarantines it (lift_pending): no L23V; the stream "
          "keeps when it first saw the list; DR0 says why and until when",
          d["D28"]["detail"].get("lift_pending") == ["src_lift"]
          and not [e for e in w.events("proposed") if e.get("lever") == "L23V"]
          and lw.get("sources") == ["src_lift"] and lw.get("first_seen_utc") in (W.utc(W.T0), W.utc(W.T0 + W.TICK))
          and wait.get("first_seen_utc") == lw["first_seen_utc"]
          and wait.get("until_utc") == W.utc(S._secs(lw["first_seen_utc"]) + 12 * 3600) and wait.get("commands") == [cmd]
          and d["DR0"]["fired"] and "waits until" in d["DR0"]["summary"]
          and not w.lane("MAINT").get("item"), (d["D28"]["detail"].get("lift_pending"), lw, wait, d["DR0"]["summary"]))
    check("  one card for a person, naming the source and the exact command",
          len(cards) == 1 and "src_lift" in cards[0]["title"] and cmd in cards[0]["detail"]
          and len([e for e in w.events("lift_wait")]) == 1, [c.get("title") for c in st.get("cards") or []])
    first = lw.get("first_seen_utc")
    w.tick(6)
    st = w.state()
    check("  an hour later: still no L23V, still one card, the same first-seen time",
          not [e for e in w.events("proposed") if e.get("lever") == "L23V"]
          and len([c for c in st.get("cards") or [] if c.get("kind") == "quarantine_lift"]) == 1
          and st["stage"]["r0"]["lift_wait"]["first_seen_utc"] == first, st["stage"]["r0"].get("lift_wait"))
    w.queue(0, extra={"quarantined_sources": {}})
    # the platform's own record of the source still says quarantined (L24 set it; a person's unquarantine on the
    # cluster does not reach it): D28 still names it, but the stream no longer quarantines it
    sp = S.StreamPaths(str(w.lab), w.domain).state(NAME)
    stj = json.loads(sp.read_text())
    stj.setdefault("sources", {})["src_lift"] = {"status": "quarantined", "cite": "D28"}
    sp.write_text(json.dumps(stj))
    w.tick(2)
    pro = [e for e in w.events("proposed") if e.get("lever") == "L23V"]
    first = pro[0] if pro else {}
    d = _diags(w)
    check("a person lifted it (the stream's quarantine no longer lists it, though the platform's source record still "
          "says quarantined and D28 still names it): L23V is proposed, citing only the lock and /stage/base3, and the "
          "wait ends",
          len(pro) == 1 and sorted(c.get("pointer") for c in first.get("cites") or []) == ["/stage/base3", "/stage/lock"]
          and d["D28"]["detail"].get("cleared_quarantined") == ["src_lift"]
          and d["D28"]["detail"].get("lift_pending") == []
          and not w.state()["stage"]["r0"].get("lift_wait") and len(w.events("lift_wait_ended")) == 1,
          ([e.get("lever") for e in w.events("proposed")], w.state()["stage"]["r0"].get("lift_wait"),
           d["D28"]["detail"].get("cleared_quarantined")))
    w2 = _lift_world("lift_timeout")
    w2.tick(3)
    w2.advance(12 * 3600 - 3 * W.TICK)
    w2.tick(1)
    early = [e for e in w2.events("proposed") if e.get("lever") == "L23V"]
    w2.tick(2)
    pro = [e for e in w2.events("proposed") if e.get("lever") == "L23V"]
    check("not lifted: no L23V before 12 h from the first sight; at 12 h L23V is proposed with the quarantine as it "
          "stands (the card stays one)", not early and len(pro) == 1
          and len([c for c in w2.state().get("cards") or [] if c.get("kind") == "quarantine_lift"]) == 1,
          ([e.get("utc") for e in pro], w2.state()["stage"]["r0"].get("lift_wait")))
    w3 = _e1_world("lift_none")
    w3.intake("i0007_src_lift", "src_lift", images=900, reasons={"near_eval_variant": 1}, pair_cos=[0.31])
    w3.tick(3)
    check("a source judged chance that the stream does not quarantine: no wait, L23V as before",
          len([e for e in w3.events("proposed") if e.get("lever") == "L23V"]) == 1
          and not [c for c in w3.state().get("cards") or [] if c.get("kind") == "quarantine_lift"])
    w4 = _lift_world("lift_leak")
    w4.intake("i0007_src_lift", "src_lift", images=900, reasons={"near_eval_variant": 1}, pair_cos=[0.97])
    w4.tick(3)
    d4 = _diags(w4)
    check("a quarantined source whose hit D28 judges a copy (pair cos 0.97) is a leak, not a lift: no wait, L23V as "
          "before", d4["D28"]["detail"].get("lift_pending") == []
          and len([e for e in w4.events("proposed") if e.get("lever") == "L23V"]) == 1,
          (d4["D28"]["detail"].get("lift_pending"), d4["D28"]["summary"][:200]))


def _l23v(w):
    return [e for e in w.events("proposed") if e.get("lever") == "L23V"]


def _lift_cards(w):
    return [c for c in w.state().get("cards") or [] if c.get("kind") == "quarantine_lift"]


def _lw(w):
    return ((w.state().get("stage") or {}).get("r0") or {}).get("lift_wait")


def t_e1_lift_faults():
    section("E1 lift wait (2026-10-03, review fixes): a D28 that cannot judge never ends the wait or restarts its "
            "clock; the clock starts when L23V is first deferred; the card keeps the alarm; only D28's quarantines")
    real = DS.d28

    def boom(v):
        raise RuntimeError("a sidecar half written")
    # 1. D28 raises on a tick while the wait runs: the recorded wait goes on
    w = _lift_world("lf_unknown")
    w.tick(3)
    first = (_lw(w) or {}).get("first_seen_utc")
    DS.d28 = boom
    try:
        w.tick(2)
    finally:
        DS.d28 = real
    d = _diags(w)
    check("D28 raises while L23V waits: no L23V; DR0 keeps waiting on the recorded sources and first deferral; one "
          "card, no end of the wait",
          first and not _l23v(w) and d["D28"].get("unknown") and (_lw(w) or {}).get("first_seen_utc") == first
          and (_lw(w) or {}).get("sources") == ["src_lift"] and "waits until" in d["DR0"]["summary"]
          and len(_lift_cards(w)) == 1 and not w.events("lift_wait_ended"),
          (_lw(w), d["DR0"]["summary"][:300], [e.get("lever") for e in w.events("proposed")]))
    # 2. the queue summary does not state the stream's quarantine: unknown, the wait goes on
    w.queue(0, extra={"quarantined_sources": None})
    w.tick(2)
    d = _diags(w)
    check("  the queue summary does not state quarantined_sources: D28's lift_pending is unknown (None), never an "
          "empty list, and the wait goes on", d["D28"]["detail"].get("lift_pending") is None and not _l23v(w)
          and (_lw(w) or {}).get("first_seen_utc") == first and not w.events("lift_wait_ended"),
          (d["D28"]["detail"].get("lift_pending"), _lw(w)))
    # 3. a tick whose snapshot cannot be read neither ends the wait nor restarts its clock
    w.queue(0, extra={"quarantined_sources": {"src_lift": {"cite": "D28", "seq": 3, "utc": "2026-10-01T00:00:00Z"}}})
    w.tick(1)
    real_ev = E.from_snapshot

    def bad(*a, **k):
        raise E.EvidenceError("an unreadable snapshot")
    E.from_snapshot = bad
    try:
        w.tick(1)
    finally:
        E.from_snapshot = real_ev
    w.tick(2)
    check("  an evidence error tick: the wait keeps its first deferral (no second wait, no second card)",
          w.events("evidence_error") and (_lw(w) or {}).get("first_seen_utc") == first
          and len(w.events("lift_wait")) == 1 and len(_lift_cards(w)) == 1 and not w.events("lift_wait_ended")
          and not _l23v(w), (_lw(w), len(_lift_cards(w))))
    # 4. 12 h after the first deferral: L23V as it stands, the wait marked expired and never started again
    w.advance(S._secs(first) + 12 * 3600 + W.TICK - w.t[0])
    w.tick(1)
    pro = _l23v(w)
    check("  12 h after the first deferral (not after the error tick): L23V is proposed with the quarantine as it "
          "stands, the wait recorded as expired, once",
          len(pro) == 1 and len(w.events("lift_wait_expired")) == 1, ([e.get("utc") for e in pro], _lw(w)))
    w.tick(3)
    check("  once base v3 runs the wait ends (base v3 is running); no second wait, no second L23V",
          len(_l23v(w)) == 1 and not _lw(w) and len(w.events("lift_wait")) == 1
          and [e.get("reasons") for e in w.events("lift_wait_ended")] == [["base v3 is running"]],
          (_lw(w), [e.get("reasons") for e in w.events("lift_wait_ended")]))
    # 5. D28 cannot judge from the first tick L23V is due: a bounded wait on the unknown, ended once D28 judges
    w2 = _e1_world("lf_unknown_first")
    DS.d28 = boom
    try:
        w2.tick(3)
    finally:
        DS.d28 = real
    c2 = _lift_cards(w2)
    check("D28 raises from the first tick L23V is due (no wait recorded): L23V waits on the unknown, bounded, with "
          "one card saying D28 cannot judge", not _l23v(w2) and (_lw(w2) or {}).get("sources") is None
          and (_lw(w2) or {}).get("first_seen_utc") and len(c2) == 1 and "cannot judge" in c2[0]["title"],
          (_lw(w2), [c.get("title") for c in c2]))
    w2.tick(2)
    check("  D28 judges again (nothing to lift): the wait ends (lifted) and L23V is proposed",
          len(_l23v(w2)) == 1 and len(w2.events("lift_wait_ended")) == 1 and not _lw(w2), (_lw(w2),))
    # 6. the clock starts when L23V is first deferred, not when the list is first seen
    w3 = _lift_world("lf_clock")
    w3.intake("i0009_src_other", "src_other", images=500, reasons={"near_eval_variant": 1})
    w3.tick(3)
    w3.advance(13 * 3600)
    w3.tick(2)
    d = _diags(w3)
    before = (_lw(w3), list(_lift_cards(w3)), list(_l23v(w3)))
    w3.intake("i0009_src_other", "src_other", images=500, reasons={"near_eval_variant": 1}, pair_cos=[0.2])
    w3.tick(2)
    lw3 = _lw(w3) or {}
    check("D28 lists src_lift for 13 h while DR0 has DATA work due (eval-hits): no wait and no card yet; once DATA "
          "clears, the wait starts then (12 h from that tick) with its card, and no L23V",
          d["D28"]["detail"].get("lift_pending") == ["src_lift"] and before == (None, [], [])
          and lw3.get("first_seen_utc") and S._secs(lw3["first_seen_utc"]) >= W.T0 + 13 * 3600
          and len(_lift_cards(w3)) == 1 and not _l23v(w3), (before, lw3))
    # 7. the card keeps the alarm: an escalation the same tick stays the current card; the lift card alone warns
    for tag, quar, extra in (("lf_alarm_leak", ("src_lift", "src_leak"), True), ("lf_alarm", ("src_lift",), False)):
        w4 = _lift_world(tag, quarantined=quar)
        if extra:
            w4.intake("i0008_src_leak", "src_leak", images=900, reasons={"near_eval_variant": 1}, pair_cos=[0.97])
        w4.tick(4)
        st4 = w4.state()
        want = "escalation" if extra else "quarantine_lift"
        check("  %s: the current card is %s and the page's alarm warns" % (
            "a leak (D28 crit) and a lift wait" if extra else "a lift wait alone", want),
            (st4.get("card") or {}).get("kind") == want and S.summary(w4.config(), st4)["alarm"] == "warn"
            and len(_lift_cards(w4)) == 1, ((st4.get("card") or {}).get("kind"), S.summary(w4.config(), st4)["alarm"]))
    det = (_lift_cards(w4) or [{}])[0].get("detail") or ""
    check("  the card says how to keep a quarantine (do nothing) and offers no keep decision nothing records",
          "to keep a quarantine, do nothing" in det and "decides to keep" not in det, det[:300])
    # 8. only D28's quarantines are reconsidered
    w5 = _e1_world("lf_cite")
    w5.intake("i0007_src_lift", "src_lift", images=900, reasons={"near_eval_variant": 1}, pair_cos=[0.31])
    w5.queue(0, extra={"quarantined_sources": {"src_lift": {"cite": "D31", "seq": 3, "utc": "2026-10-01T00:00:00Z"}}})
    w5.tick(3)
    d5 = _diags(w5)
    check("a source the stream quarantined on D31's word (not D28's), its hit now judged chance: no wait, L23V as "
          "before", d5["D28"]["detail"].get("lift_pending") == [] and len(_l23v(w5)) == 1 and not _lift_cards(w5),
          (d5["D28"]["detail"].get("lift_pending"), [e.get("lever") for e in w5.events("proposed")]))


def t_e1_cut_order():
    section("E1 (2026-10-03, review fix): a segment's cut (L18) and the base v3 build (L23V) are never submitted while "
            "the other's outcome is unknown to it")
    w = _e1_world("order")
    w.queue(4 * w.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w.t[0] - 2 * 86400.0))
    w.tick(4)
    seg = [e for e in w.events("executed") if e.get("lever") == "L18"]
    child = (seg[0] if seg else {}).get("child_exp")
    waits = [e for e in w.events("waiting") if e.get("lever") == "L23V"]
    check("both due: the segment (L18, TRAIN first) is submitted; L23V is proposed but waits while the segment's cut "
          "is not in the evidence",
          len(seg) == 1 and len(_l23v(w)) == 1 and not [e for e in w.events("executed") if e.get("lever") == "L23V"]
          and waits and child in waits[0]["reasons"][0] and "not in the evidence" in waits[0]["reasons"][0],
          ([e.get("lever") for e in w.events("executed")], [e.get("reasons") for e in waits]))
    n = int(child.rsplit("_s", 1)[1])
    w.segment(n, [("I1", "ACCEPT", 0.9, [], "helps", ["src_a"])], done=False)
    w.tick(3)
    check("  the segment's build line in the evidence: base v3 is submitted (it reads the cut rows as in flight)",
          [e.get("lever") for e in w.events("executed") if e.get("lever") == "L23V"] == ["L23V"]
          and w.state()["stage"]["r0"].get("base3") == "running",
          [e.get("lever") for e in w.events("executed")])
    w2 = _e1_world("order_rev")
    w2.tick(3)
    check("base v3 submitted first (nothing to cut)", w2.state()["stage"]["r0"].get("base3") == "running"
          and [e.get("lever") for e in w2.events("executed")].count("L23V") == 1)
    w2.queue(4 * w2.M, boxes={"Purslane": 900}, oldest_utc=W.utc(w2.t[0] - 2 * 86400.0))
    w2.tick(3)
    nt = [e for e in w2.events("not_taken") if e.get("lever") == "L18" and "base v3" in str(e.get("reasons"))]
    check("  then a segment is due: it is not taken while base v3 runs (a cut now could take rows it holds out)",
          not [e for e in w2.events("proposed") if e.get("lever") == "L18"] and nt, [e.get("reasons") for e in nt])
    w2.job_done("inc_build_base3_v3")
    w2.base3_summary()
    w2.tick(4)
    check("  base v3 done (its test lists exist, the cutter refuses their rows): the segment is cut",
          [e.get("lever") for e in w2.events("executed")].count("L18") == 1,
          [e.get("lever") for e in w2.events("executed")])


def main():
    for fn in (t_menu, t_prices, t_remote, t_evidence, t_budget, t_records, t_measure, t_native, t_e1, t_e1_lift_wait, t_e1_lift_faults, t_e1_cut_order,
               t_formats,
               t_replay_gate,
               t_config, t_lab, t_lanes, t_d28, t_d28_v2, t_d28_v2_sources, t_d28_v2_round3):
        try:
            fn()
        except Exception as e:
            import traceback
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:]))
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
