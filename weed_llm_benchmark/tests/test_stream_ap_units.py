#!/usr/bin/env python3
"""Units of stream mode (docs/CONTINUOUS_LOOP.md 6): the lever menu against
the executor and the policy table, the cluster verbs' parsers and grammar,
the evidence allow-list, the budget windows and job settlement, the R0
records (Stage A, Stage C, the capacity decision), the measurement arms
(m832, s1024, and the box-quality arms y26l640, y26m640, l640: priced,
proposed once R0 is complete and only while missing, never the stream's arm), the dispositions, the replay gate's stream cases, the
config and the campaign dispatch, and the lab runner. No network, no GPU, no
ssh.

Run:  python3 tests/test_stream_ap_units.py
"""
import copy
import hashlib
import json
import os
import pathlib
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
                       {"verb": "backfill"}):
            argv = LS.render("L17", LS.policy_params("L17", params))
            args = argv[argv.index("run_inc2_stream.sh") + 1:]
            ns = S1.build_parser().parse_args(args)
            check("L17 %s: group C's step1_stream parser reads its argv" % args[0], ns.verb == args[0], ns)
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
          and set(LS.envelope_levers()) == {"L18", "L20", "L21", "L22", "L23B", "L23N", "L25", "L27", "L28", "LI"})
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
              "capacity/capacity_v1.json", "canary_v2/canary.json", "pilot_v4/stage_a.json"):
        check("allowed: %s" % n, E.allowed(n))
    for n in ("stream/x/pool/P_1.jsonl", "intake/staging/a/b.jpg", "splits/v2/LOCK.json", "stream/x/scores/test.json",
              "step1_stream/queue/queue.jsonl", "capacity/capacity_v1_report.json", "capacity/capacity_v1_report.md",
              "stream/x/milestones/m001/research_log_entry.md", "stream/stage_a.json"):
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
    """A world with every R0 prerequisite but the measurement arms' experiments."""
    w = World(tag)
    w.ready_r0(stage_c=stage_c)
    for b in w.dom["baselines"]["items"]:
        if b.get("measure"):
            shutil.rmtree(str(w.inc / b["exp"]))
    return w


def t_measure():
    section("the measurement arms (m832, s1024 2026-09-30; y26l640, y26m640, l640 2026-10-01): proposed once R0 is "
            "complete, never the stream's arm")
    from weed_optimizer_framework.tools.inc2 import recipes as RC
    dom = LS.load_domain("weed")
    meas = [b for b in dom["baselines"]["items"] if b.get("measure")]
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
    meas = [b["exp"] for b in w.dom["baselines"]["items"] if b.get("measure")]
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
    w.step1_status(extra={"knowntruth": {"b0003": {"matched_verified": 60, "verified_correct": 58,
                                                   "verified_precision_wilson_lb": 0.887}},
                          "refit_triggers": {"precision_lb_below": ["b0003"], "species_unknown_share": ["Eclipta"],
                                             "fired": True},
                          "holds_past_deadline": {"funnel_F9": 12}})
    ev = E.from_texts({"step1_stream/status.json": (w.inc / "step1_stream" / "status.json").read_text()}, "x",
                      context={"sid": "none"})
    by = DS.by_id(DS.detect(ev, run.dom, run.th, only=("DKT", "DHOLD")))
    check("DKT: 58 of 60 verified correct, Wilson lower bound under 0.99 -> card X11; and the species trigger",
          by["DKT"]["fired"] and "X11" in by["DKT"]["levers"] and "Eclipta" in by["DKT"]["summary"], by["DKT"]["summary"])
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
    check("a source with a planted copy (a flip within 6 dHash bits of a dev image) leaks: D28 -> L24",
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


def main():
    for fn in (t_menu, t_prices, t_remote, t_evidence, t_budget, t_records, t_measure, t_native, t_formats,
               t_replay_gate,
               t_config, t_lab, t_lanes, t_d28):
        try:
            fn()
        except Exception as e:
            import traceback
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:]))
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
