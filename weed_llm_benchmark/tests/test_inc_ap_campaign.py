#!/usr/bin/env python3
"""The INC campaign ticker (inc_autopilot/campaign.py): docs/INC_AUTOPILOT.md,
component 6 and step (g)8.

No cluster, no ssh, no Mongo: a simulated clock and a fake `slurm_sh` that
answers the executor's scripts. The fake runs the real remote.py verbs
(campaign-snapshot, snapshot, status, step1) on a temporary INC_DIR laid out
from the replay fixtures (tests/fixtures/inc_replay: pilot_v1), with the
driver's advance, the report writer, squeue and sbatch replaced by the fake
cluster. pilot_v2 and its finished report are made from pilot_v1's files
(full replay; chains that track the truth arm, so D1 is silent and D4 is
ready on it).

Pinned:
  * the full cycle: pilot_v1 done -> REPORT -> DIAGNOSE (D1 fires) ->
    prospective D4 written -> L1 proposed (the manual pilot_v2 command) ->
    autonomy off: an approval card, nothing runs, no second approval across
    ticks -> a person approves -> executed exactly once (decided_by the
    person) -> RUN pilot_v2 (building, built, advanced) -> REPORT (outcome
    scored, spend recorded) -> DIAGNOSE -> prospective D4 ready -> goal met
    -> COMPLETE with a card;
  * autonomy 'envelope' with a replay pass: the L1 build runs without a
    person, exactly once, across ticks and across a restart that lost the
    state write after the executor ran; without a replay pass it waits;
  * a finished child with no goal: L4 runs directly (R2), WAIT_JOB, then
    lineage (open issue 1) keeps it from running twice and the campaign
    completes with a residual card, never idle;
  * stop-loss after 2 consecutive failed steps; a pause (by a person, or a
    config paused_reason) is honoured: no ssh, no execution; health stops
    (D7 code drift) and an automatic unblock of a transient block (D5 -> L7);
  * test blindness: evidence is built from the decision part only
    (display_only never reaches evidence.from_snapshot), and perturbing every
    test / ood / imageweeds value on the cluster leaves every decision
    (campaign ledger, approvals, diagnoses) identical (metamorphic);
  * one ssh per tick, every tick; the ssh budget refuses a second call;
  * the scheduler hook: every CAMPAIGN_EVERY-th tick, on its own thread; a
    ticker that raises never reaches the loop, and tick() never raises;
  * the brain: staged at DIAGNOSE, submitted through the executor
    (inc_plan_submit), pulled with the snapshot (inc_plan_pull), validated
    and merged (a bad cite dropped, an off-menu card recorded);
  * the open issues: lineage from executions (1), validate's L5 through
    levers.loop_params and levers.applied (2), the build job's own SU in
    levers.price (3), the inc_label_audit line reference (4), ledger reads
    that carry through_sha256 and recover from a prefix_mismatch (5).

Run:  python3 tests/test_inc_ap_campaign.py
"""
import base64
import copy
import gzip
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import sys
import tempfile
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_campaign_"))
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ["INC_DIR"] = str(TMP / "default_inc")      # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
os.environ.pop("INCAP_MAX_FILE_BYTES", None)
os.environ.pop("INCAP_MAX_LEDGER_BYTES", None)
sys.path.insert(0, str(PKG_ROOT))

from weed_optimizer_framework.tools.brain import approvals as AP  # noqa: E402
from weed_optimizer_framework.tools.brain import policy as POL  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import brain_plan as BP  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import campaign as C  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import executor as X  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers as LV  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as R  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import validate as V  # noqa: E402

FIX = PKG_ROOT / "tests" / "fixtures" / "inc_replay"
FAILURES = []
OWNER = "human:owner@example.org"
AUTO = M.AUTOPILOT_ACTOR
NAME = "weedinc1"
T0 = 1790000000.0
TICK = 600.0
NON_DEV = ("test", "ood22", "ood23", "imageweeds")
SEG_RE = re.compile(r'echo "INCAP_SEG (\d+)"; (.*?); echo "INCAP_SEG_END', re.S)
GOAL_D4 = {"kind": "diagnosis", "id": "D4", "name": "decision_slot_ready"}
REPLAY_PASS = dict({c: "pass" for c in X.REPLAY_REQUIRED}, R2="skip", R4b="skip")
RESOURCES = {"mongo_ok": True, "cluster_reachable": True}


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


class QuietLog(object):
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))

    def error(self, m):
        self.lines.append(("error", m))


# --- the fixtures, and pilot_v2 made from them ------------------------------------------
def _fixture_text(rel):
    return (FIX / rel).read_text(encoding="utf-8")


def _v2(text):
    return text.replace("pilot_v1", "pilot_v2")


def pilot_v2_files():
    """pilot_v2 from pilot_v1: full replay, initialised later; its gate entries
    no longer flag the recipe (P_recipe 0.9, so D1 is silent) and its full
    chain agrees with the truth arm on 6 of 7 steps (so D4 is ready)."""
    exp = json.loads(_v2(_fixture_text("pilot_v1/exp.json")))
    exp["replay_mode"] = "full"
    exp["initialised_utc"] = "2026-09-27T05:55:39Z"
    lines = []
    for ln in _v2(_fixture_text("pilot_v1/ledger.jsonl")).splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        if e.get("type") == "gate" and isinstance(e.get("decision"), dict):
            e["decision"]["p_recipe"] = 0.9
        lines.append(json.dumps(e, sort_keys=True))
    rep = json.loads(_v2(_fixture_text("pilot_v1/report.json")))
    rep["replay_mode"] = "full"
    rep["agreement"] = {"full": {"agree": 6, "compared": 7, "rate": 6 / 7.0},
                        "freeze": {"agree": 3, "compared": 7, "rate": 3 / 7.0},
                        "lora": {"agree": 3, "compared": 7, "rate": 3 / 7.0}}
    for st in rep.get("steps") or []:
        for c in (st.get("chains") or {}).values():
            if isinstance(c, dict) and "p_recipe" in c:
                c["p_recipe"] = 0.9
    rep["done_utc"] = "2026-09-27T07:49:08Z"
    return {"exp.json": json.dumps(exp, sort_keys=True),
            "build_summary.json": _v2(_fixture_text("pilot_v1/build_summary.json")),
            "ledger": lines, "report": rep}


def perturb(obj, under=False):
    """Every value under a test / ood / imageweeds key changed, and every
    'production' flag flipped (report.py derives it from test)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "production" and isinstance(v, bool):
                out[k] = not v
            else:
                out[k] = perturb(v, under or k in NON_DEV)
        return out
    if isinstance(obj, list):
        return [perturb(v, under) for v in obj]
    if not under:
        return obj
    if isinstance(obj, bool):
        return not obj
    if isinstance(obj, int):
        return obj + 3
    if isinstance(obj, float):
        return obj * 0.37 + 0.11
    if isinstance(obj, str):
        return obj + "_x"
    return obj


# --- the fake cluster ---------------------------------------------------------------------
class World(object):
    """A temporary lab tree and INC_DIR, the fake cluster behind slurm_sh, and a
    simulated clock. The remote.py verbs run for real on the INC_DIR."""

    def __init__(self, tag, perturbed=False, autonomy="off", goal=GOAL_D4, brain=False):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="w_%s_" % tag, dir=str(TMP)))
        self.lab = self.dir / "lab"
        self.inc = self.dir / "inc"
        self.lab.mkdir()
        self.inc.mkdir()
        self.cfg = self.dir / "round_scheduler.json"
        self.hooks = C._local_cfg_hooks(self.cfg)
        self.t = [T0]
        self.perturbed = perturbed
        self.squeue = []                     # [{"id", "name", "state"}]
        self.tick_calls = []                 # per tick: [script]
        self.verbs = []                      # every verb argv (or ("plan", ...))
        self.submits = []
        self.advances = []
        self.plan_inputs = {}
        self.plan_replies = {}
        self.mode = "ok"                     # ok | refuse_submit | raise
        self.prov_on_submit = True           # the fake writes a "running" provenance at submit
        self.refused_ssh = []                # ticks whose ssh budget refused a second call
        self.drift = False
        self.next_job = 7000
        self.reports = {}
        self.log = QuietLog()
        self.xctx = X.Context(lab_repo=str(self.lab), resources=RESOURCES, clock=self.clock)
        self._put_pilot_v1()
        C.configure(NAME, OWNER, enable=True, exps=["pilot_v1"], current="pilot_v1",
                    autonomy=autonomy, brain=brain, cfg_hooks=self.hooks, lab_repo=str(self.lab),
                    clock=self.clock)
        C.set_goal(NAME, goal, OWNER, cfg_hooks=self.hooks, lab_repo=str(self.lab), clock=self.clock)

    def clock(self):
        return self.t[0]

    # ---- the INC_DIR
    def _w(self, rel, text, mtime=None):
        p = self.inc / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.perturbed and (rel.endswith(".json")):
            text = json.dumps(perturb(json.loads(text)), sort_keys=True)
        p.write_text(text, encoding="utf-8")
        if mtime is not None:
            os.utime(str(p), (mtime, mtime))

    def _ledger(self, exp, lines):
        if self.perturbed:
            lines = [json.dumps(perturb(json.loads(x)), sort_keys=True) for x in lines]
        p = self.inc / exp / "ledger.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(x + "\n" for x in lines), encoding="utf-8")

    def _state(self, exp, done, generation, blocked=None, mtime=None):
        st = {"exp": exp, "type": "chain", "done": done, "generation": generation,
              "done_utc": "2026-09-27T07:49:08Z" if done else None, "runs": {},
              "blocked": blocked or {}, "unblocks": [], "chains": {}, "submissions": []}
        self._w("%s/state.json" % exp, json.dumps(st, sort_keys=True), mtime)

    def _put_pilot_v1(self):
        for n in ("exp.json", "build_summary.json"):
            self._w("pilot_v1/" + n, _fixture_text("pilot_v1/" + n))
        self._ledger("pilot_v1", [x for x in _fixture_text("pilot_v1/ledger.jsonl").splitlines() if x.strip()])
        self._state("pilot_v1", True, 99, mtime=T0 - 7200)
        self._w("pilot_v1/report.json", _fixture_text("pilot_v1/report.json"), mtime=T0 - 3600)

    def build(self, exp="pilot_v2", n_ledger=10):
        """The build job ran: the experiment exists and runs (driver init)."""
        f = pilot_v2_files()
        self._w(exp + "/exp.json", f["exp.json"])
        self._w(exp + "/build_summary.json", f["build_summary.json"])
        self._ledger(exp, f["ledger"][:n_ledger])
        self._state(exp, False, 1, mtime=self.t[0] - 60)
        self._provenance(exp, "advanced")
        self.squeue = [j for j in self.squeue if j["name"] != "inc_build_%s" % exp]
        self.squeue.append({"id": "7999_0", "name": "inc_%s_0001" % exp, "state": "RUNNING"})

    def finish(self, exp="pilot_v2"):
        f = pilot_v2_files()
        self._ledger(exp, f["ledger"])
        self._state(exp, True, 40, mtime=self.t[0] - 30)
        self.reports[exp] = f["report"]
        rp = self.inc / exp / "report.json"
        if rp.exists():
            rp.unlink()
        self.squeue = [j for j in self.squeue if not j["name"].startswith("inc_%s_" % exp)]

    def _provenance(self, exp, status, refusal=None):
        rec = {"exp": exp, "attempts": [{"job_id": str(self.next_job), "status": status,
                                        "refusal": refusal, "started_utc": "x"}]}
        self._w("_campaign/provenance/%s.json" % exp, json.dumps(rec))

    # ---- slurm_sh
    def __call__(self, script, timeout=60):
        (self.tick_calls[-1] if self.tick_calls else self.tick_calls.append([]) or self.tick_calls[-1]) \
            .append(script)
        if self.mode == "raise":
            raise OSError("ssh: connect to host bridges2.psc.edu: timed out")
        out = ["Welcome to Bridges-2"]
        for m in SEG_RE.finditer(script):
            i, argv = int(m.group(1)), shlex.split(m.group(2))
            out.append("INCAP_SEG %d" % i)
            rec = self._answer(argv)
            out.append("login profile noise")
            out.append("INCAP " + json.dumps(rec, default=str))
            out.append("INCAP_SEG_END %d %d" % (i, 0 if rec.get("ok") else 1))
        return {"ok": True, "stdout": "\n".join(out), "stderr": "", "returncode": 0}

    def _answer(self, argv):
        if argv[:3] == ["python", "-u", "-c"]:
            code, args = argv[3], argv[4:]
            if code == X._PLAN_SUBMIT_PY:
                return self._plan_submit(args)
            if code == X._PLAN_PULL_PY:
                self.verbs.append(("plan-pull", args[0]))
                reply = self.plan_replies.get(args[0])
                return {"verb": "plan-pull", "ok": True, "output": args[0],
                        "exists": reply is not None, "reply": reply}
            return {"ok": False, "error": "unknown inline code"}
        args = argv[4:]                       # python -u -m MODULE ARGS
        self.verbs.append(args)
        if args[0] == "submit":
            return self._submit(args)
        self._activate()
        return R.dispatch(args)

    def _plan_submit(self, args):
        inp, out, want, model, name, repo, payload = args
        raw = gzip.decompress(base64.b64decode(payload))
        self.verbs.append(("plan-submit", inp))
        if hashlib.sha256(raw).hexdigest() != want:
            return {"verb": "plan-submit", "ok": False, "error": "sha mismatch"}
        self.plan_inputs[out] = json.loads(raw.decode("utf-8"))
        self.next_job += 1
        return {"verb": "plan-submit", "ok": True, "job_id": str(self.next_job), "input": inp,
                "output": out}

    def _submit(self, argv):
        self.submits.append(argv)
        rec = {"verb": "submit", "ok": True, "argv": argv}
        if self.mode == "refuse_submit":
            rec.update(ok=False, error="[inc.pilot] ERROR pilot_v2 is already built at /x; an "
                                       "experiment is built once", error_kind="refused")
            return rec
        self.next_job += 1
        rec["job_id"] = str(self.next_job)
        args = argv[argv.index("--") + 1:]
        if "--exp" in args:
            exp = args[args.index("--exp") + 1]
            self.squeue.append({"id": str(self.next_job), "name": "inc_build_%s" % exp,
                                "state": "PENDING"})
            if self.prov_on_submit:
                self._provenance(exp, "running")
        elif args and args[0] == "--trusted":
            self.squeue.append({"id": str(self.next_job), "name": "inc_audit", "state": "PENDING"})
        return rec

    # ---- the remote verbs the fake replaces
    def _activate(self):
        world = self
        R.inc_dir = lambda: world.inc
        R.advance = self._advance
        R.report = self._report
        R.squeue_jobs = lambda prefix="inc_": {"ok": True, "jobs": [
            dict(j, elapsed="0:01", submit="x") for j in world.squeue if j["name"].startswith(prefix)]}

    def _advance(self, exp, backend=None):
        self.advances.append(exp)
        rec = R.base_record("advance")
        rec["exp"] = exp
        if self.drift:
            return R.fail(rec, "the outer package copy /o (what the INC jobs import) differs from "
                               "the git-tracked copy /n in ['inc/gate.py']", "DriftError", "code_drift")
        p = self.inc / exp / "state.json"
        st = json.loads(p.read_text())
        if not st.get("done"):
            m = p.stat().st_mtime
            st["generation"] = int(st.get("generation") or 0) + 1
            p.write_text(json.dumps(st, sort_keys=True))
            os.utime(str(p), (m, m))
        rec["result"] = {"locked": False, "passes": 1, "submitted": 0, "job_ids": [],
                         "done": bool(st.get("done")), "lines": []}
        return rec

    def _report(self, exp):
        rec = R.base_record("report")
        rec["exp"] = exp
        rep = self.reports.get(exp)
        if rep is None:
            return R.fail(rec, "no report for %s" % exp, error_kind="other")
        st_m = (self.inc / exp / "state.json").stat().st_mtime
        self._w("%s/report.json" % exp, json.dumps(rep, sort_keys=True), mtime=st_m + 10)
        rec.update(done=True, files={})
        return rec

    # ---- driving
    def tick(self, advance=True, fn=None):
        """One campaign tick through this fake cluster (or `fn`, a wrapper
        around it). Every tick must attempt at most one ssh: a second call the
        ticker's budget refused is a failure (the order in _step never makes
        one), recorded here for every tick of every scenario."""
        self.tick_calls.append([])
        out = C.tick(slurm_sh=fn or self, cfg_hooks=self.hooks, log=self.log, clock=self.clock,
                     lab_repo=str(self.lab), resources=RESOURCES)
        if (out or {}).get("ssh_refused"):
            self.refused_ssh.append(len(self.tick_calls))
            check("tick %d of %s attempted no second ssh (ssh_refused == 0)"
                  % (len(self.tick_calls), self.dir.name), False, out)
        if (out or {}).get("ok") is False:
            check("tick %d of %s ran (tick() reported no failure)" % (len(self.tick_calls), self.dir.name),
                  False, out)
        if advance:
            self.t[0] += TICK
        return out

    def put_step1(self):
        """Step 1 with a relevance.json made for its select build (levers:
        relevance_state 'matching'), so D4 ready makes an L2 buildable."""
        self._w("step1/select_summary.json", json.dumps(
            {"sizes": {"base_B": 3927},
             "outputs": {"increment_pool.jsonl": {"sha256": "a" * 64},
                         "base_B.jsonl": {"sha256": "b" * 64}}}, sort_keys=True))
        self._w("step1/relevance.json", json.dumps(
            {"format": "inc.relevance/1", "params": {"min_crops": 20, "calibration_percentile": 5.0, "tau_min": 0.5},
             "calibration": {"tau": 0.62, "check": {"ok": True, "tau_min": 0.5}},
             "inputs": {"increment_pool": {"sha256": "a" * 64}}, "increment_pool": {"sources": {}}},
            sort_keys=True))

    def state(self):
        return C.load_state(C.Paths(str(self.lab)), NAME)

    def ledger(self):
        p = C.Paths(str(self.lab)).ledger
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def events(self, name):
        return [r for r in self.ledger() if r.get("event") == name]

    def approvals(self):
        return AP.state("weed", root=str(self.lab))

    def config(self):
        return (self.hooks[0]().get("campaigns") or {}).get(NAME) or {}

    def replay_pass(self):
        return X.record_replay_result("pass", dict(REPLAY_PASS), ctx=self.xctx)


def submits_of(world, builder=None):
    return [a for a in world.submits if builder is None or (len(a) > 1 and a[1] == builder)]


def max_calls(world):
    return max([len(c) for c in world.tick_calls] or [0])


# ===========================================================================================
def section_cycle_with_approval():
    print("full cycle, autonomy off: approval card, a person approves, executed once")
    w = World("cycle")
    w.tick()
    st = w.state()
    item = (st.get("item") or {}).get("proposal") or {}
    check("tick 1: pilot_v1 done -> reported -> diagnosed -> GATE with L1",
          st["phase"] == "GATE" and item.get("lever") == "L1", (st["phase"], item.get("lever")))
    check("L1's argv is the manual command (inc.pilot build --exp pilot_v2 --replay-mode full)",
          item.get("argv") == ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build",
                               "--exp", "pilot_v2", "--replay-mode", "full"], item.get("argv"))
    diag = w.events("diagnosed")
    check("D1 fired on pilot_v1 and is in the ledger with its cites",
          diag and any(t["id"] == "D1" and t["cites"] for t in diag[-1]["trigger"]))
    pro = w.events("prospective_d4")
    path = C.Paths(str(w.lab)).replay / DG.prospective_name("pilot_v1", DG.rules_version())
    check("prospective D4 on pilot_v1 is written with its sha256 in the ledger, before the proposal",
          pro and path.is_file() and pro[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
          and pro[0]["ready"] is False
          and w.ledger().index(pro[0]) < w.ledger().index(w.events("proposed")[0]), pro)
    check("  named for its pilot and the rules version it was decided under, which the ledger records",
          pro and pro[0]["rules_version"] == DG.rules_version() and pro[0]["superseded"] == []
          and path.name == "prospective_d4_pilot_v1__%s.json" % DG.rules_version()
          and json.loads(path.read_text())["rules_version"] == DG.rules_version(), pro and pro[0].get("path"))
    nt = w.events("not_taken")
    check("L4 (D3) is recorded as not taken: one item in flight", any(r.get("lever") == "L4" for r in nt))
    check("R4 cards X1 and X4 are recorded", {r.get("lever") for r in w.events("card")} >= {"X1", "X4"})
    check("tick 1 made one ssh (the snapshot)", len(w.tick_calls[0]) == 1 and w.verbs[0][0] == "campaign-snapshot")
    w.tick()
    items = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]
    check("tick 2: filed for approval as the autopilot, no ssh",
          len(items) == 1 and items[0]["status"] == "pending" and items[0]["requested_by"] == AUTO
          and len(w.tick_calls[1]) == 0 and w.state()["phase"] == "GATE", (items, w.tick_calls[1:2]))
    check("the approval card is up", (w.state().get("card") or {}).get("kind") == "approval")
    check("the approval carries the lever, trigger and cites",
          items and items[0]["context"]["lever"] == "L1" and "D1" in items[0]["context"]["trigger"]
          and items[0]["context"]["cites"])
    n_exec = len(X.executions(w.xctx, raw=True))
    for _ in range(2):
        w.tick()
    items = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]
    check("waiting ticks file nothing twice and run nothing", len(items) == 1 and not w.submits
          and len(w.events("filed")) == 1, (len(items), w.submits))
    check("waiting for a person adds no execution-log line per tick",
          len(X.executions(w.xctx, raw=True)) == n_exec)
    AP.decide("weed", items[0]["id"], "approve", OWNER, "go", w.t[0], root=str(w.lab))
    w.tick()
    ex = w.events("executed")
    check("after approval: executed once, authorised as the person",
          len(submits_of(w)) == 1 and ex and ex[0]["decided_by"] == OWNER
          and ex[0]["approval_id"] == items[0]["id"] and ex[0]["job_ids"]
          and ex[0]["child_exp"] == "pilot_v2" and ex[0]["parent_exp"] == "pilot_v1", ex)
    check("the executed entry carries trigger diagnoses with their cites",
          ex and any(t["id"] == "D1" and t["cites"] for t in ex[0]["trigger"]))
    st = w.state()
    check("RUN on pilot_v2, building", st["phase"] == "RUN" and st["exp"] == "pilot_v2"
          and (st.get("building") or {}).get("exp") == "pilot_v2", (st["phase"], st["exp"]))
    run = C._Run(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log,
                 w.hooks, RESOURCES, None, None, None)
    run.st = st
    lin = run.lineage()
    check("open issue 1: lineage from executions shows the L1 build in flight",
          any(r["lever"] == "L1" and r["parent_exp"] == "pilot_v1" and r["child_exp"] == "pilot_v2"
              and r["status"] == "in_flight" for r in lin), lin)
    for _ in range(2):
        w.tick()
    check("more ticks: still one submission", len(submits_of(w)) == 1)
    w.build()
    w.tick()
    check("built: the ledger says so, RUN advances pilot_v2",
          w.events("built") and w.state()["phase"] == "RUN" and "pilot_v2" in w.advances,
          [r["event"] for r in w.ledger()[-4:]])
    snap_args = [v for v in w.verbs if isinstance(v, list) and v[0] == "campaign-snapshot"][-1]
    check("the snapshot advances only the live experiment (pilot_v1 is frozen)",
          "--advance" in snap_args and "pilot_v1" not in snap_args, snap_args)
    w.tick()
    snap_args = [v for v in w.verbs if isinstance(v, list) and v[0] == "campaign-snapshot"][-1]
    lf = [snap_args[i + 1] for i, a in enumerate(snap_args) if a == "--ledger-from"]
    check("open issue 5: the next read asks from its line with the through_sha256",
          lf and re.match(r"^pilot_v2=10:[0-9a-f]{64}$", lf[0]), lf)
    w.finish()
    w.tick()
    st = w.state()
    rep = w.events("reported")
    check("pilot_v2 finished: reported with its spend charged (launched by the campaign)",
          any(r["exp"] == "pilot_v2" and (r.get("spend") or {}).get("ok") for r in rep), rep)
    check("pilot_v1 (not launched by the campaign) is reported but not charged",
          any(r["exp"] == "pilot_v1" and (r.get("spend") or {}).get("ok") is None for r in rep), rep)
    out = w.events("outcome")
    check("the L1 outcome is scored against its prediction",
          out and out[0]["lever"] == "L1" and out[0]["child_exp"] == "pilot_v2"
          and out[0]["verdict"] in ("better", "worse", "within_noise", "insufficient"), out)
    pro2 = [r for r in w.events("prospective_d4") if r["exp"] == "pilot_v2"]
    check("prospective D4 on pilot_v2: ready, full replay, recipe full",
          pro2 and pro2[0]["ready"] is True and pro2[0]["replay_mode"] == "full"
          and pro2[0]["recipes"] == ["full"], pro2)
    check("goal D4 decision_slot_ready met: COMPLETE with a card",
          st["phase"] == "COMPLETE" and (st.get("card") or {}).get("kind") == "goal", (st["phase"], st.get("card")))
    comp = w.events("complete")
    check("the prospective record precedes the completion",
          comp and pro2 and w.ledger().index(pro2[0]) < w.ledger().index(comp[-1]))
    for _ in range(2):
        w.tick()
    check("COMPLETE: further ticks make no ssh and change nothing",
          all(len(c) == 0 for c in w.tick_calls[-2:]) and w.state()["phase"] == "COMPLETE")
    check("one ssh per tick over the whole cycle", max_calls(w) <= 1, [len(c) for c in w.tick_calls])
    keys = ("decided_by", "trigger", "parent_exp", "child_exp", "approval_id", "job_ids")
    check("every campaign-ledger entry carries decided_by, trigger, parent_exp, child_exp, "
          "approval_id and job_ids", all(all(k in r for k in keys) for r in w.ledger()))
    status = json.loads(C.Paths(str(w.lab)).status.read_text())
    check("the status heartbeat names the phase and the card",
          status["campaigns"][NAME]["phase"] == "COMPLETE"
          and status["campaigns"][NAME]["card"]["kind"] == "goal", status["campaigns"].get(NAME))
    return w


def section_denial():
    print("a person denies the proposal: it is not proposed again; the next one comes")
    w = World("deny")
    w.tick()
    w.tick()
    items = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]
    AP.decide("weed", items[0]["id"], "deny", OWNER, "not now", w.t[0], root=str(w.lab))
    w.tick()
    den = w.events("denied")
    check("the denial is in the ledger, decided by the person",
          den and den[0]["decided_by"] == OWNER and den[0]["approval_id"] == items[0]["id"], den)
    st = w.state()
    it = (st.get("item") or {}).get("proposal") or {}
    check("the same tick re-diagnoses (the denial used no ssh): L1 is not proposed again, L4 (the "
          "next) is",
          it.get("lever") == "L4" and not w.submits
          and len([i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]) == 1,
          (it.get("lever"), w.submits))
    check("the declined L1 is recorded as not taken",
          any(r.get("lever") == "L1" and "declined" in " ".join(r.get("reasons") or [])
              for r in w.events("not_taken")))
    w.tick()
    check("L4 (R2) runs directly", len(submits_of(w, "audit")) == 1 and w.state()["phase"] == "WAIT_JOB")


def section_envelope_and_restart():
    print("autonomy envelope: executed without a person, exactly once, across a restart")
    w = World("env", autonomy="envelope")
    w.tick()
    check("tick 1 proposes L1", ((w.state().get("item") or {}).get("proposal") or {}).get("lever") == "L1")
    w.tick()
    it = w.state().get("item") or {}
    check("no replay pass: the build waits for a person",
          it.get("status") == "filed" and not w.submits
          and any("replay tests" in r for r in it.get("last_reasons") or []), it.get("last_reasons"))
    w.replay_pass()
    real_save = C.save_state
    calls = {"n": 0}

    def flaky(paths, name, st):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full (simulated crash after the executor ran)")
        return real_save(paths, name, st)
    C.save_state = flaky
    try:
        w.tick()
    finally:
        C.save_state = real_save
    check("with a replay pass: the envelope grant runs it (one submit)", len(submits_of(w, "build")) == 1,
          w.submits)
    st = w.state()
    check("the state write was lost: the item is still at GATE on disk",
          st["phase"] == "GATE" and st.get("item") is not None, st["phase"])
    w.tick()
    st = w.state()
    check("after the 'restart' the execution is recovered, not run again",
          len(submits_of(w, "build")) == 1 and st["phase"] == "RUN" and st["exp"] == "pilot_v2"
          and w.events("recovered"), (len(w.submits), st["phase"], st["exp"]))
    item = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]
    check("the approval is an envelope grant by the autopilot, run as the owner",
          item and item[0]["status"] == "approved" and item[0].get("decision_basis") == "envelope"
          and item[0]["decided_by"] == AUTO, item)
    ex = [r for r in X.executions(w.xctx) if r.get("action") == "inc_build_pilot"
          and r.get("status") == "executed"]
    check("executions: one run, authorised as the person who granted the envelope",
          len(ex) == 1 and ex[0]["authorized_as"] == OWNER and ex[0]["est_su"] == 42.544, ex)
    for _ in range(3):
        w.tick()
    check("three more ticks: still one build submission", len(submits_of(w, "build")) == 1)
    check("one ssh per tick", max_calls(w) <= 1, [len(c) for c in w.tick_calls])
    return w


def section_child_without_goal():
    print("a finished child with no goal: L4 runs directly, lineage stops a repeat, residual card")
    w = World("nogoal", autonomy="envelope", goal=None)
    w.replay_pass()
    w.tick()
    w.tick()
    check("L1 ran under the envelope", len(submits_of(w, "build")) == 1)
    w.build()
    w.tick()
    w.finish()
    w.replay_pass()
    w.tick()
    st = w.state()
    item = (st.get("item") or {}).get("proposal") or {}
    check("pilot_v2 diagnosed: D3 -> L4 on pilot_v2 proposed (D4 ready but L2 deferred)",
          st["phase"] == "GATE" and item.get("lever") == "L4" and item.get("parent_exp") == "pilot_v2",
          (st["phase"], item.get("lever"), item.get("parent_exp")))
    w.tick()
    st = w.state()
    check("L4 is R2: run directly, then WAIT_JOB on its job",
          len(submits_of(w, "audit")) == 1 and st["phase"] == "WAIT_JOB" and st.get("wait_jobs"),
          (st["phase"], w.submits))
    w.tick()
    check("the audit job is queued: still waiting", w.state()["phase"] == "WAIT_JOB")
    w.squeue = [j for j in w.squeue if j["name"] != "inc_audit"]
    w.tick()
    st = w.state()
    check("job gone: re-diagnosed; L4 already applied (lineage) so nothing is left: COMPLETE",
          st["phase"] == "COMPLETE" and (st.get("card") or {}).get("kind") == "residual"
          and len(submits_of(w, "audit")) == 1, (st["phase"], st.get("card")))
    check("the residual card names the deferred levers",
          "L2" in ((st.get("card") or {}).get("detail") or ""), st.get("card"))
    check("one ssh per tick", max_calls(w) <= 1, [len(c) for c in w.tick_calls])


def section_stop_loss_and_pause():
    print("stop-loss after 2 consecutive failed steps; pause honoured")
    w = World("stop", autonomy="envelope")
    w.mode = "refuse_submit"
    w.replay_pass()
    for _ in range(2):
        w.tick()
    st = w.state()
    check("first refused build: one failed step, back to DIAGNOSE", st["fails"] == 1
          and st["phase"] == "DIAGNOSE" and not st.get("paused"), (st["fails"], st["phase"]))
    w.tick()
    it = (w.state().get("item") or {}).get("proposal") or {}
    check("re-diagnosed with the refusal in the context: L1 again, as a new proposal (attempt 1)",
          it.get("lever") == "L1" and (w.state().get("item") or {}).get("attempt") == 1,
          w.state().get("item"))
    check("the refusal reached the diagnoses (DREF)", any(d["id"] == "DREF" for d in w.state()["diagnoses"]))
    w.replay_pass()
    w.tick()
    st = w.state()
    check("second refused build: paused by the stop-loss",
          st["fails"] >= 2 and (st.get("paused") or {}).get("reason", "").startswith("stop-loss")
          and w.config().get("enabled") is False and w.config().get("paused_reason"),
          (st["fails"], st.get("paused"), w.config()))
    n = len(w.submits)
    w.mode = "ok"
    for _ in range(2):
        w.tick()
    check("paused: no ssh, no submission", len(w.submits) == n and all(len(c) == 0 for c in w.tick_calls[-2:]))

    print("the ticker's own pause, then a person enables it again")
    C.configure(NAME, OWNER, enable=True, cfg_hooks=w.hooks, lab_repo=str(w.lab), clock=w.clock)
    st = w.state()
    st["paused"] = {"reason": "stop-loss (a state a racing tick wrote back)", "utc": C._utc(w.t[0] - 5)}
    C.save_state(C.Paths(str(w.lab)), NAME, st)
    w.tick()
    check("an enable after the pause clears the state's pause too (resumed_utc)",
          not w.state().get("paused") and w.events("resumed"), w.state().get("paused"))

    print("a build that fails on the cluster: back to the parent, a new proposal")
    w3 = World("buildfail", autonomy="envelope")
    w3.replay_pass()
    w3.tick()
    w3.tick()
    first = [r for r in w3.events("executed")][0]["proposal_id"]
    w3._provenance("pilot_v2", "build_failed", refusal="[inc.pilot] ERROR missing manifest")
    w3.squeue = [j for j in w3.squeue if j["name"] != "inc_build_pilot_v2"]
    w3.tick()
    st = w3.state()
    check("build_failed: a failed step, back to pilot_v1, the child dropped",
          st["fails"] == 1 and st["exp"] == "pilot_v1" and st["phase"] == "DIAGNOSE"
          and "pilot_v2" not in st["exps"] and w3.events("build_failed"), (st["fails"], st["exp"], st["phase"]))
    run = C._Run(NAME, w3.config(), C.Paths(str(w3.lab)), C._SshBudget(None), w3.clock, w3.log,
                 w3.hooks, RESOURCES, None, None, None)
    run.st = st
    check("the failed build's lineage record is 'failed' (levers.py ignores it)",
          any(r["lever"] == "L1" and r["status"] == "failed" for r in run.lineage()), run.lineage())
    w3.replay_pass()
    w3.tick()
    it = w3.state().get("item") or {}
    check("re-proposed as a new proposal (attempt 1, a new id), not the old execution adopted",
          it.get("attempt") == 1 and (it.get("proposal") or {}).get("id") != first
          and not w3.events("recovered"), it.get("attempt"))
    check("the refusal names the builder that failed (not the placeholder)",
          (st.get("refusals") or [{}])[-1].get("builder") == "inc_build_pilot", st.get("refusals"))
    w3.prov_on_submit = False            # the new job is PENDING: only the old attempt's record exists
    w3.tick()
    check("and it runs again once", len(submits_of(w3, "build")) == 2)
    w3.tick()
    st = w3.state()
    check("the earlier attempt's build_failed provenance does not fail the new build while it is queued",
          st["phase"] == "RUN" and (st.get("building") or {}).get("exp") == "pilot_v2"
          and len(w3.events("build_failed")) == 1, (st["phase"], st.get("building")))
    w3.build()
    w3.tick()
    check("  and the new build is confirmed by the snapshot", w3.events("built"), w3.state()["phase"])

    print("a build refused with a refusal the identical build meets again (levers.json retry false): declined, "
          "DREF escalates, no second submission")
    w5 = World("noretry", autonomy="envelope")
    w5.replay_pass()
    w5.tick()
    w5.tick()
    n_builds = len(submits_of(w5, "build"))
    draw = "[inc.realloop] ERROR: 4 increments of 287 images need 1148, the draw found 1140"
    w5._provenance("pilot_v2", "build_failed", refusal=draw)
    w5.squeue = [j for j in w5.squeue if j["name"] != "inc_build_pilot_v2"]
    w5.tick()
    st = w5.state()
    key = [k for k in st.get("declined") or []]
    check("the refused build is a failed step and its request is declined for the campaign (the ledger says why)",
          n_builds == 1 and st["fails"] == 1 and len(key) == 1
          and any("meets again" in " ".join(e.get("reasons") or []) for e in w5.events("not_taken")),
          (n_builds, st["fails"], key))
    w5.replay_pass()
    for _ in range(3):
        w5.tick()
    st = w5.state()
    dref = [d for d in st.get("diagnoses") or [] if d["id"] == "DREF"]
    check("re-diagnosed: DREF fires crit with OP_ESCALATE (retry false) and the ledger records the escalation",
          dref and dref[0]["severity"] == "crit" and dref[0]["levers"] == ["OP_ESCALATE"]
          and dref[0]["detail"]["refusals"][-1]["retry"] is False and w5.events("escalated"),
          dref and dref[0].get("levers"))
    it = ((st.get("item") or {}).get("proposal") or {}).get("lever")
    check("the same request is not proposed or submitted again (not_taken: declined or refused earlier), the "
          "campaign goes on with what else it may do, and the stop-loss never fires",
          len(submits_of(w5, "build")) == n_builds and it != "L1"
          and any("declined or refused earlier" in " ".join(e.get("reasons") or []) and e.get("lever") == "L1"
                  for e in w5.events("not_taken"))
          and not st.get("paused") and not [e for e in w5.events("paused")],
          (len(submits_of(w5, "build")), st["phase"], it, st.get("paused")))
    run = C._Run(NAME, w5.config(), C.Paths(str(w5.lab)), C._SshBudget(None), w5.clock, w5.log,
                 w5.hooks, RESOURCES, None, None, None)
    run.st = copy.deepcopy(st)
    run._declined_now = [{"lever": "L1", "parent_exp": "pilot_v1"}]
    run._complete_residual({"deferred": []}, [])
    check("with nothing else left, the residual card names the declined request",
          "not proposed again: L1 on pilot_v1" in run.st["card"]["detail"], run.st["card"]["detail"])

    print("a person switches the current experiment while a tick races the state write")
    w4 = World("switch")
    w4.tick()
    stale = w4.state()                                   # what a running tick holds
    C.configure(NAME, OWNER, current="b0_v1", cfg_hooks=w4.hooks, lab_repo=str(w4.lab), clock=w4.clock)
    C.save_state(C.Paths(str(w4.lab)), NAME, stale)      # the racing tick writes the old state back
    w4._w("b0_v1/exp.json", _fixture_text("b0_v1/exp.json"))
    w4.tick()
    st = w4.state()
    check("the switch is applied from the config request, once, despite the race",
          st["exp"] == "b0_v1" and w4.events("switched") and w4.config().get("current_exp") == "b0_v1",
          (st["exp"], w4.config().get("current_exp")))
    w4.tick()
    check("and not applied twice", len(w4.events("switched")) == 1)
    check("the config carries the live view for the page",
          (w4.config().get("state") or {}).get("exp") == "b0_v1"
          and (w4.config().get("state") or {}).get("phase") == w4.state()["phase"], w4.config().get("state"))

    print("pause by a person, then enable")
    w2 = World("pause")
    w2.tick()
    C.pause(NAME, "hold for the review", OWNER, cfg_hooks=w2.hooks, lab_repo=str(w2.lab), clock=w2.clock)
    before = copy.deepcopy(w2.state())
    for _ in range(2):
        w2.tick()
    after = w2.state()
    check("paused by a person: nothing filed, no ssh, the item stays",
          not w2.approvals() and all(len(c) == 0 for c in w2.tick_calls[-2:])
          and after["item"]["proposal"]["id"] == before["item"]["proposal"]["id"])
    cfg = w2.hooks[0]()
    cfg["campaigns"][NAME]["enabled"] = True            # enabled by hand, the reason still set
    w2.hooks[1](cfg)
    w2.tick()
    check("a config paused_reason alone is honoured too", not w2.approvals())
    C.configure(NAME, OWNER, enable=True, cfg_hooks=w2.hooks, lab_repo=str(w2.lab), clock=w2.clock)
    w2.tick()
    check("enabled again: the waiting item is filed", len(w2.approvals()) == 1)


def section_health():
    print("health: D7 code drift pauses; D5 transient block -> L7 unblock, then RUN")
    w = World("health", autonomy="envelope")
    w.replay_pass()
    w.tick()
    w.tick()
    w.build()
    w.tick()
    blocked = {"chain:full": {"utc": "x", "error": "score read failed 3 times",
                              "cause": {"kind": "transient"}}}
    w._state("pilot_v2", False, 5, blocked=blocked)
    w.tick()
    st = w.state()
    it = (st.get("item") or {}).get("proposal") or {}
    check("D5 fired with a transient block: L7 proposed, to resume RUN",
          it.get("lever") == "L7" and st["item"].get("resume") == "RUN"
          and it.get("params", {}).get("unit") == "chain:full", st.get("item"))
    w.tick()
    un = [v for v in w.verbs if isinstance(v, list) and v[0] == "unblock"]
    check("L7 ran directly (R2) and the campaign is back in RUN",
          len(un) == 1 and "--unit" in un[0] and w.state()["phase"] == "RUN", (un, w.state()["phase"]))
    w._state("pilot_v2", False, 6)
    w.drift = True
    w.tick()
    st = w.state()
    check("the advance refused on code drift: D7 pauses the campaign",
          (st.get("paused") or {}).get("reason", "").startswith("code drift") and
          any(c.get("lever") == "X5" for c in st.get("cards") or []), st.get("paused"))
    w3 = World("block", autonomy="envelope")
    w3.replay_pass()
    w3.tick()
    w3.tick()
    w3.build()
    w3.tick()
    w3._state("pilot_v2", False, 5, blocked={"truth": {"utc": "x", "error": "2 failed attempts",
                                                       "cause": {"kind": "failed_run"}}})
    w3.tick()
    check("a block that is not transient pauses (D5 OP_PAUSE)",
          "not transient" in ((w3.state().get("paused") or {}).get("reason") or ""), w3.state().get("paused"))


def _normalise(obj, w):
    txt = json.dumps(obj, sort_keys=True, default=str)
    txt = txt.replace(str(w.lab), "<lab>").replace(str(w.inc), "<inc>").replace(str(w.dir), "<dir>")
    obj = json.loads(txt)

    def strip(x):
        if isinstance(x, dict):
            # provenance hashes of raw files (they include the non-dev bytes), and the
            # executor's random run ids and wall-clock replay stamps: not decisions
            return {k: strip(v) for k, v in x.items()
                    if "sha256" not in k and k not in ("provenance", "files", "run_id", "recorded_utc")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(obj)


def _scenario(tag, perturbed, seen):
    orig = C.E.from_snapshot

    def spy(record, exp, context=None, ledger_prefix=None):
        seen.append(copy.deepcopy(record))
        return orig(record, exp, context=context, ledger_prefix=ledger_prefix)
    C.E.from_snapshot = spy
    try:
        w = World(tag, perturbed=perturbed, autonomy="envelope")
        w.replay_pass()
        w.tick()
        w.tick()
        w.build()
        w.tick()
        w.finish()
        w.replay_pass()
        w.tick()
    finally:
        C.E.from_snapshot = orig
    return w


def section_test_blindness():
    print("test blindness: decision inputs are dev only; perturbing non-dev values changes nothing")
    seen_a, seen_b = [], []
    a = _scenario("blind_a", False, seen_a)
    b = _scenario("blind_b", True, seen_b)
    check("the perturbed cluster really differs (report.json bytes)",
          (a.inc / "pilot_v1" / "report.json").read_bytes() != (b.inc / "pilot_v1" / "report.json").read_bytes())
    leaks = []
    for rec in seen_a + seen_b:
        txt = json.dumps(rec)
        if "display_only" in txt:
            leaks.append("display_only")
        for sub in (rec.get("experiments") or {}).values():
            dec = ((sub or {}).get("snapshot") or {}).get("decision")
            leaks += BP.dev_leaks(dec)
    check("evidence.from_snapshot never received display_only or a non-dev exam key",
          seen_a and not leaks, leaks[:5])
    la = [_normalise({k: v for k, v in r.items() if k != "utc"}, a) for r in a.ledger()]
    lb = [_normalise({k: v for k, v in r.items() if k != "utc"}, b) for r in b.ledger()]
    diff = [(x, y) for x, y in zip(la, lb) if x != y]
    check("metamorphic: the campaign ledger is identical", len(la) == len(lb) and not diff,
          diff[:1] or (len(la), len(lb)))
    sa, sb = a.state(), b.state()
    for k in ("phase", "exp", "diagnoses", "item", "prospective", "cards", "declined"):
        check("metamorphic: state %s is identical" % k,
              _normalise(sa.get(k), a) == _normalise(sb.get(k), b), (sa.get(k), sb.get(k)))
    aa = sorted(_normalise({k: v for k, v in i.items() if k != "id"}, a) for i in a.approvals().values())
    ab = sorted(_normalise({k: v for k, v in i.items() if k != "id"}, b) for i in b.approvals().values())
    check("metamorphic: the approvals are identical", json.dumps(aa, sort_keys=True) == json.dumps(ab, sort_keys=True))
    check("one ssh per tick (both runs)", max_calls(a) <= 1 and max_calls(b) <= 1)


def section_ssh_budget():
    print("one ssh per tick")
    n = {"calls": 0}

    def fn(script, timeout=60):
        n["calls"] += 1
        return {"ok": True, "stdout": "", "stderr": "", "returncode": 0}
    b = C._SshBudget(fn)
    b("echo 1", 10)
    raised = False
    try:
        b("echo 2", 10)
    except RuntimeError:
        raised = True
    check("the budget refuses a second call without calling through", raised and n["calls"] == 1
          and b.refused == 1)
    w = World("ssh_fail")
    w.mode = "raise"
    w.tick()
    w.tick()
    st = w.state()
    check("an ssh that fails: no decision, counted, no crash", st["snapshot_failures"] == 2
          and st["phase"] == "RUN" and max_calls(w) <= 1, (st["snapshot_failures"], st["phase"]))
    w.mode = "ok"
    w.tick()
    check("it recovers on the next tick", w.state()["phase"] == "GATE" and w.events("snapshot_recovered"))


def _import_scheduler():
    try:
        import fastapi  # noqa: F401
    except ImportError:                      # python without fastapi: a stub is enough here
        fa = types.ModuleType("fastapi")

        class APIRouter(object):
            def get(self, *a, **k):
                return lambda f: f

            def post(self, *a, **k):
                return lambda f: f
        fa.APIRouter, fa.Request = APIRouter, object
        resp = types.ModuleType("fastapi.responses")
        resp.JSONResponse = dict
        fa.responses = resp
        sys.modules["fastapi"], sys.modules["fastapi.responses"] = fa, resp
    from weed_optimizer_framework.tools import round_scheduler as RS
    return RS


class _Stop(BaseException):
    pass


def section_scheduler_hook():
    print("the scheduler hook: every 5th tick, on its own thread, never raising into the loop")
    RS = _import_scheduler()
    log = QuietLog()
    RS._CTX.update({"log": log, "slurm_sh": None, "db": None})
    real_tick = C.tick
    ran = []

    def boom(**kw):
        ran.append(kw)
        raise RuntimeError("ticker exploded")
    C.tick = boom
    try:
        RS._CAMPAIGN.update(ticks=0, thread=None)
        started = [RS._campaign_tick({"campaigns": {NAME: {}}}) for _ in range(RS.CAMPAIGN_EVERY)]
        th = started[-1]
        if th is not None:
            th.join(10)
        check("the 5th call starts one thread, the others none",
              started[:-1] == [None] * (RS.CAMPAIGN_EVERY - 1) and th is not None and len(ran) == 1)
        check("the hook passes lab_repo (None: the dashboard's tree is the package's own)",
              ran and "lab_repo" in ran[0] and ran[0]["lab_repo"] is None, ran[:1])
        other = str(TMP / "dashboard_tree")
        RS._CTX["repo"] = other
        RS._CAMPAIGN.update(ticks=0, thread=None)
        started = [RS._campaign_tick({"campaigns": {NAME: {}}}) for _ in range(RS.CAMPAIGN_EVERY)]
        if started[-1] is not None:
            started[-1].join(10)
        RS._CTX.pop("repo", None)
        check("  and the dashboard's REPO_ROOT when it is another tree (where its approval queue and "
              "the /inc page read)", len(ran) == 2 and ran[1].get("lab_repo") == other, ran[1:])
        check("campaign.lab_repo_arg: None for no repo or model.LAB_REPO, else the repo",
              C.lab_repo_arg(None) is None and C.lab_repo_arg(str(M.LAB_REPO)) is None
              and C.lab_repo_arg(other) == other)
        del ran[1:]
        check("the ticker's exception was logged, not raised",
              any("INC campaign tick failed" in m for _, m in log.lines), log.lines)
        check("no campaigns configured: nothing starts",
              all(RS._campaign_tick({"domains": {}}) is None for _ in range(RS.CAMPAIGN_EVERY)))
        # the loop itself keeps running
        RS._CAMPAIGN.update(ticks=0, thread=None)
        n = {"sleep": 0}
        real = (RS.time.sleep, RS._cfg, RS._heartbeat)

        def sleep(_s):
            n["sleep"] += 1
            if n["sleep"] > 2 * RS.CAMPAIGN_EVERY + 1:
                raise _Stop()
        RS.time.sleep = sleep
        RS._cfg = lambda: {"domains": {}, "campaigns": {NAME: {"enabled": True}}}
        RS._heartbeat = lambda c, d: None
        try:
            try:
                RS._loop()
            except _Stop:
                pass
        finally:
            RS.time.sleep, RS._cfg, RS._heartbeat = real
        th = RS._CAMPAIGN.get("thread")
        if th is not None:
            th.join(10)
        check("the loop ran %d ticks with a ticker that raises every time" % (2 * RS.CAMPAIGN_EVERY),
              n["sleep"] == 2 * RS.CAMPAIGN_EVERY + 2 and len(ran) == 3, (n, len(ran)))
    finally:
        C.tick = real_tick
    bad = C.tick(slurm_sh=None, cfg_hooks=(lambda: (_ for _ in ()).throw(ValueError("unreadable")),
                                           lambda c: None),
                 log=QuietLog(), lab_repo=str(TMP / "hooklab"))
    check("tick() returns a failure, never raises, on an unreadable config",
          bad.get("ok") is False and "unreadable" in bad.get("error", ""), bad)


def section_brain():
    print("the brain: staged, submitted through the executor, pulled, validated, merged")
    w = World("brain", brain=True)
    w.tick()
    st = w.state()
    b = st.get("brain") or {}
    inp = C.Paths(str(w.lab)).plan_input(NAME, 1)
    check("tick 1: a dev-only digest is staged for pilot_v1", b.get("status") == "staged"
          and inp.is_file() and not BP.dev_leaks(json.loads(inp.read_text())["sections"]["evidence"]), b)
    w.tick()
    b = w.state().get("brain") or {}
    subs = [v for v in w.verbs if isinstance(v, tuple) and v[0] == "plan-submit"]
    check("tick 2: L1 filed (no ssh), then the plan job submitted (the tick's one ssh)",
          b.get("status") == "submitted" and len(subs) == 1 and len(w.tick_calls[1]) == 1
          and (w.state().get("item") or {}).get("status") == "filed", (b, subs))
    ex = [r for r in X.executions(w.xctx) if r.get("action") == "inc_plan_submit"]
    check("the plan job is charged to the envelope (2 H100 GPU-hours = 4 SU)",
          ex and ex[-1]["status"] == "executed" and ex[-1]["est_su"] == 4.0, ex[-1:])
    out_path = BP.plan_paths(NAME, 1)["output"]
    digest = w.plan_inputs.get(out_path) or {}
    check("the cluster received the staged digest unchanged",
          digest.get("sha256") == b.get("digest_sha256") and BP.digest_sha256(digest) == digest.get("sha256"))
    w.tick()
    pulls = [v for v in w.verbs if isinstance(v, tuple) and v[0] == "plan-pull"]
    check("tick 3: the reply is pulled with the snapshot, in one ssh; not back yet",
          len(pulls) == 1 and len(w.tick_calls[2]) == 1 and w.state()["brain"]["status"] == "submitted")
    d3 = next(d for d in w.state()["diagnoses"] if d["id"] == "D3")
    w.plan_replies[out_path] = {
        "schema": BP.REPLY_SCHEMA, "ok": True, "digest_sha256": digest["sha256"],
        "model": "qwen3.8:27b", "finished_utc": C._utc(w.t[0]),
        "plan": {"ranked_menu": [
            {"lever": "L4", "params": {"exp": "pilot_v1"}, "evidence_cites": [d3["cites"][0]],
             "rationale": "the planted relabel was not attributed to labels",
             "predicted": {"metric": "agreement", "direction": "none", "magnitude": None},
             "falsifier": "the audit finds Bswap at baseline", "trigger": ["D3"]},
            {"lever": "L1", "params": {}, "evidence_cites": [dict(d3["cites"][0], value="not it")],
             "rationale": "x", "predicted": {"metric": "agreement", "direction": "up", "magnitude": None},
             "falsifier": "y"}],
            "off_menu": [{"hypothesis": "the replay sample is too small to protect the base",
                          "why_menu_insufficient": "no lever changes the sampler",
                          "required_change": "a sampler change in driver.py (pinned)",
                          "cheapest_test": "one chain with a larger replay sample",
                          "control": "pilot_v1", "success_criterion": "agreement >= 5/7",
                          "title": "Replay sampler"}],
            "stop_recommendation": None}}
    w.tick()
    b = w.state().get("brain") or {}
    merged = w.events("brain_merged")
    check("tick 4: merged; the bad cite dropped; the brain's L4 matched the deterministic one",
          b.get("status") == "merged" and merged and b["counts"]["dropped"] == 1
          and b["counts"]["valid"] == 1 and not b.get("filed"), (b, merged[-1:]))
    check("the off-menu idea is an R4 card, never queued",
          any(c.get("proposed_by", "").startswith("tier2:") for c in w.state().get("cards") or [])
          and not [i for i in w.approvals().values() if i.get("risk") == "R4"])
    check("one ssh per tick", max_calls(w) <= 1, [len(c) for c in w.tick_calls])


def _empty_reply(w, n):
    out = BP.plan_paths(NAME, n)["output"]
    return out, {"schema": BP.REPLY_SCHEMA, "ok": True,
                 "digest_sha256": w.plan_inputs[out]["sha256"], "model": "qwen3.8:27b",
                 "finished_utc": C._utc(w.t[0]),
                 "plan": {"ranked_menu": [], "off_menu": [], "stop_recommendation": None}}


def _to_brain_wait(tag):
    """Brain on, no goal: pilot_v1 -> L1 (envelope) -> pilot_v2 -> L4 -> WAIT_JOB -> nothing
    deterministic left while plan 2 (for pilot_v2) is still due."""
    w = World(tag, autonomy="envelope", goal=None, brain=True)
    for i in range(3):
        w.replay_pass()
        w.tick()
    w.build()
    w.tick()
    out, rep = _empty_reply(w, 1)
    w.plan_replies[out] = rep
    w.tick()
    w.finish()
    for _ in range(8):
        w.replay_pass()
        w.tick()
        if w.state()["phase"] == "WAIT_JOB":
            w.squeue = [j for j in w.squeue if j["name"] != "inc_audit"]
        if w.state()["phase"] == "BRAIN_WAIT":
            break
    return w


def section_brain_wait_and_adoption():
    print("BRAIN_WAIT: nothing deterministic left while a plan is due; then COMPLETE")
    w = _to_brain_wait("bwait")
    st = w.state()
    check("nothing deterministic to propose on pilot_v2 and plan 2 is due: BRAIN_WAIT",
          st["phase"] == "BRAIN_WAIT" and (st.get("brain") or {}).get("n") == 2
          and st["brain"]["status"] == "submitted", (st["phase"], st.get("brain")))
    w.tick()
    check("waiting: one pull per tick, still BRAIN_WAIT", w.state()["phase"] == "BRAIN_WAIT"
          and len(w.tick_calls[-1]) == 1)
    out, rep = _empty_reply(w, 2)
    w.plan_replies[out] = rep
    w.tick()
    st = w.state()
    check("the plan came back with nothing to file: COMPLETE with a residual card",
          st["phase"] == "COMPLETE" and (st.get("card") or {}).get("kind") == "residual"
          and st["brain"]["status"] == "merged", (st["phase"], st.get("card")))
    w2 = _to_brain_wait("btimeout")
    w2.t[0] += BP.PLAN_TIMEOUT_S + TICK
    w2.tick()
    st = w2.state()
    check("no reply past the deadline: the plan times out and the campaign completes",
          st["brain"]["status"] == "timeout" and st["phase"] == "COMPLETE"
          and w2.events("brain_timeout"), (st["brain"].get("status"), st["phase"]))

    w5 = World("bstale", brain=True)
    w5.tick()
    check("a plan is staged", (w5.state().get("brain") or {}).get("status") == "staged")
    w5.t[0] += BP.PLAN_TIMEOUT_S + TICK
    w5.tick()
    check("a staged plan never submitted within its deadline is given up, not sent late",
          w5.state()["brain"]["status"] == "timeout"
          and not [v for v in w5.verbs if isinstance(v, tuple) and v[0] == "plan-submit"],
          w5.state().get("brain"))

    print("a person approves a tier2 item instead: it is adopted, the pending item superseded")
    w3 = World("adopt")
    w3.tick()
    w3.tick()
    check("L1 is pending approval", (w3.state().get("item") or {}).get("status") == "filed")
    ev = E.load_dir(FIX, "pilot_v1", exps=["pilot_v1"])
    l4 = [p for p in LV.propose(DG.detect(ev), ev)["proposals"] if p["lever"] == "L4"][0]
    l4 = dict(l4, proposed_by="tier2:qwen3.8_27b")
    r = X.submit(l4, actor="tier2:qwen3.8_27b", campaign={"name": NAME}, ctx=w3.xctx)
    check("the tier2 L4 is filed for a person", r["status"] == "filed", r.get("reasons"))
    AP.decide("weed", r["approval_id"], "approve", OWNER, "yes", w3.t[0], root=str(w3.lab))
    w3.tick()
    st = w3.state()
    check("adopted: executed as the person, the pending L1 superseded, WAIT_JOB on the audit",
          len(submits_of(w3, "audit")) == 1 and st["phase"] == "WAIT_JOB" and w3.events("superseded")
          and w3.events("executed")[-1]["decided_by"] == OWNER, (st["phase"], w3.submits))
    check("one ssh per tick", max_calls(w3) <= 1 and max_calls(w) <= 1 and max_calls(w2) <= 1)


def section_open_issue_fixes():
    print("open issues 2, 3 and 4 (issues 1 and 5 are checked in the cycle)")
    ev = E.load_dir(FIX, "pilot_v1", exps=["pilot_v1"])
    est, info, _c = LV.price("L1", {"exp": "pilot_v2"}, ev, "pilot_v1")
    base, _i, _c2 = LV.estimate_pilot_rebuild(ev, "pilot_v1", "full")
    check("issue 3: levers.price adds the build job's 4 h (run_inc_build.sh #SBATCH --time)",
          info.get("build_job_hours") == 4.0 and abs(est - (base + 4.0)) < 1e-9
          and info.get("experiment_hours") == base, info)
    # issue 2: a brain L5 through levers.loop_params and levers.applied
    inc = M.CLUSTER_INC_DIR
    loop = {"exp": "realloop_v1", "type": "chain", "builder": "inc.realloop build",
            "initialised_utc": "2026-10-01T00:00:00Z", "replay_mode": "full", "truth": False,
            "recipes": {"full": {"epochs": 30}}, "increment_images": 393,
            "base": {"manifest": inc + "/realloop_v1/manifests/B.jsonl",
                     "source_manifest": inc + "/step1/base_B.jsonl"},
            "steps": [{"name": "V1", "kind": "verified", "clean": True}]}
    texts = {n: (FIX / n).read_text() for n in ("pilot_v1/exp.json", "pilot_v1/report.json")}
    texts["realloop_v1/exp.json"] = json.dumps(loop)
    texts["realloop_v1/build_summary.json"] = json.dumps({"n_verified": 6})
    texts["step1/select_summary.json"] = json.dumps({"sizes": {"base_B": 3927}})
    menu = BP.load_menu()

    def mat(bp, context=None):
        t = dict(texts)
        ev2 = E.from_texts(t, "realloop_v1", context=context)
        ctx = {"ev": ev2, "lmenu": V._levers_menu(menu), "exp": "realloop_v1", "diags": {}}
        return V.materialise("L5", menu["L5"], bp, ctx)
    m = mat({"size": 800})
    p = m["params"]
    check("issue 2: L5's --base is Step 1's source manifest (not the experiment's copy)",
          p.get("base") == inc + "/step1/base_B.jsonl", p)
    check("issue 2: --n-verified from build_summary, --no-truth kept from the parent, --size the brain's",
          p.get("n_verified") == 6 and p.get("no_truth") == 1 and p.get("size") == 800
          and p.get("replay_mode") == "full" and p.get("recipes") == "full", p)
    check("issue 2/3: the L5 price includes the build job", (m["estimate"] or {}).get("build_job_hours") == 4.0)
    deferred = False
    try:
        mat({"size": 800}, context={"lineage": [{"lever": "L5", "parent_exp": "realloop_v1",
                                                 "child_exp": "realloop_v2", "status": "in_flight"}]})
    except LV.Defer as e:
        deferred = "already applied" in str(e)
    check("issue 2: an L5 already in flight on that parent (lineage) is deferred", deferred)
    refused = False
    try:
        mat({"size": 800, "base": inc + "/realloop_v1/manifests/B.jsonl"})
    except V.Refuse as e:
        refused = "not the plan's to choose" in str(e)
    check("issue 2: a brain base that is not the parent's source manifest is refused", refused)
    row = POL.describe("inc_label_audit")
    lines = (PKG_ROOT / "run_inc_audit.sh").read_text().splitlines()
    check("issue 4: the policy row cites run_inc_audit.sh:30-36, and those lines are the command",
          "run_inc_audit.sh:30-36" in row.get("description", "")
          and lines[29].strip().startswith("#   REPO=") and "--out" in lines[35], (lines[29], lines[35]))
    # issue 5, the rewritten-prefix half
    w = World("prefix", autonomy="envelope")
    w.replay_pass()
    w.tick()
    w.tick()
    w.build(n_ledger=10)
    w.tick()
    f = pilot_v2_files()["ledger"]
    rewritten = [json.dumps(dict(json.loads(f[0]), note="rewritten"), sort_keys=True)] + f[1:12]
    w._ledger("pilot_v2", rewritten)
    w.tick()
    mirror = [json.loads(x) for x in C.Paths(str(w.lab)).mirror("pilot_v2").read_text().splitlines()]
    check("issue 5: a rewritten prefix is detected and the lab copy re-read from line 0",
          w.events("ledger_rewritten") and [m["entry"] for m in mirror] == [json.loads(x) for x in rewritten]
          and w.state()["ledger"]["pilot_v2"]["next_line"] == 12,
          (len(mirror), w.state()["ledger"].get("pilot_v2")))


def section_cli():
    print("the CLI: enable, set-goal, pause, status")
    import contextlib
    import io
    d = TMP / "cli"
    d.mkdir()
    cfg, lab = str(d / "rs.json"), str(d / "lab")

    def run(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = C.main(["--config", cfg, "--lab-repo", lab] + list(argv))
        return rc, json.loads(buf.getvalue())
    rc, out = run("enable", "--name", "c1", "--by", OWNER, "--exp", "pilot_v1", "--current", "pilot_v1",
                  "--autonomy", "envelope", "--envelope-su", "120")
    check("enable: enabled, autonomy granted by the person, envelope set",
          rc == 0 and out["enabled"] is True and out["autonomy_granted_by"] == OWNER
          and out["envelope_su"] == 120.0, out)
    rc, out = run("set-goal", "--name", "c1", "--by", OWNER, "--diagnosis", "D4:decision_slot_ready")
    check("set-goal", rc == 0 and out["goal"] == GOAL_D4, out)
    rc, out = run("enable", "--name", "c1", "--by", "tier2:qwen")
    check("a non-person cannot configure a campaign", rc == 1 and "person" in out["error"], out)
    rc, out = run("set-goal", "--name", "c1", "--by", OWNER, "--diagnosis", "D99:x")
    check("an unknown diagnosis is not a goal", rc == 1, out)
    rc, out = run("pause", "--name", "c1", "--by", OWNER, "--reason", "maintenance")
    check("pause", rc == 0 and out["enabled"] is False and "maintenance" in out["paused_reason"], out)
    rc, out = run("status")
    s1 = out.get("c1", {}).get("summary", {})
    check("status: the campaign is paused (alarm crit) with the person's card",
          rc == 0 and s1.get("alarm") == "crit" and (s1.get("card") or {}).get("kind") == "paused", s1)
    rows = [json.loads(x) for x in (pathlib.Path(lab) / "results/framework/_brain/weed/inc/inc_campaign.jsonl")
            .read_text().splitlines()]
    check("every configuration change is in the campaign ledger, decided by the person",
          [r["event"] for r in rows] == ["configured", "goal_set", "paused"]
          and all(r["decided_by"] == OWNER for r in rows), [r["event"] for r in rows])


# --- transport outcomes, stale approvals, resume, goal, build SU, the D4 guard ---------------
class _Crash(BaseException):
    """The lab process dying between the ssh and the execution log's outcome line."""


def _through(w, match, reply, times=1):
    """slurm_sh: the first `times` calls whose script contains `match` go to the
    cluster (the fake runs them) and come back as `reply(result)`; the rest are
    answered normally."""
    n = {"k": 0}

    def fn(script, timeout=60):
        if match in script and n["k"] < times:
            n["k"] += 1
            return reply(w(script, timeout))
        return w(script, timeout)
    return fn


def _timeout(_res):
    # dashboard_server._shell on subprocess.TimeoutExpired: whatever the cluster
    # printed is lost.
    return {"ok": False, "stdout": "", "stderr": "TIMEOUT", "returncode": -1}


def _dropped(res):
    # the connection drops after the segment started: no INCAP line, no SEG_END
    lines = [x for x in res["stdout"].splitlines() if x.startswith(("Welcome", "INCAP_SEG "))]
    return {"ok": False, "stdout": "\n".join(lines), "stderr": "Connection to bridges2.psc.edu closed by "
                                                            "remote host.", "returncode": 255}


def _crash(_res):
    raise _Crash()


def _build_execs(w):
    return [r for r in X.executions(w.xctx) if r.get("action") == "inc_build_pilot"
            and r.get("status") in ("executed", "failed", "started")]


def section_parse_remote():
    print("executor: which calls certainly never reached the cluster")
    connect = "ssh: connect to host bridges2.psc.edu port 22: Connection timed out"
    cases = [
        ("a timeout (the dashboard's _shell drops the output)",
         {"ok": False, "stdout": "", "stderr": "TIMEOUT", "returncode": -1}, True),
        ("ssh never connected", {"ok": False, "stdout": "", "stderr": connect, "returncode": 255}, False),
        ("a hook that raised on connecting", {"ok": False, "stdout": "",
                                               "stderr": "OSError: " + connect, "returncode": -2}, False),
        ("login throttling before the session", {"ok": False, "stdout": "", "returncode": 255,
                                                  "stderr": "Connection closed by 128.182.108.57 port 22"}, False),
        ("a connection closed by the host during the session",
         {"ok": False, "stdout": "", "stderr": "Connection to bridges2.psc.edu closed by remote host.",
          "returncode": 255}, True),
        ("output came, then a connect-like error", {"ok": False, "stdout": "Welcome", "stderr": connect,
                                                    "returncode": 255}, True),
        ("a hook exception after the call (a decode error)",
         {"ok": False, "stdout": "", "stderr": "UnicodeDecodeError: 'utf-8' codec", "returncode": -2}, True),
        ("the remote preamble failed", {"ok": True, "stdout": "INCAP_PREAMBLE_FAILED\n", "stderr": "",
                                        "returncode": 0}, False),
        ("a call the hook did not send", {"ok": False, "stdout": "", "stderr": "NotSent: budget",
                                          "returncode": -3, "not_sent": True}, False),
    ]
    for label, res, may in cases:
        seg = X.parse_remote(res, 1)[0]
        check("%s: may_have_run is %s" % (label, may),
              seg["may_have_run"] is may and not seg["known"] and not seg["ok"], seg)
    seg = X.parse_remote({"ok": True, "returncode": 0, "stderr": "",
                          "stdout": "INCAP_SEG 0\nINCAP {\"ok\": false, \"error\": \"refused\"}\n"
                                    "INCAP_SEG_END 0 1"}, 1)[0]
    check("a verb that answered with a failure did not run (known outcome)",
          seg["known"] and seg["may_have_run"] is False, seg)
    budget = C._SshBudget(lambda s, t=60: {"ok": True, "stdout": "", "stderr": "", "returncode": 0})
    budget("x", 1)
    ctx = X.Context(slurm_sh=budget)
    r = X._call_slurm(ctx, "y", 1)
    check("the ticker's ssh budget refusing a second call is NotSent: never reached the cluster",
          r.get("not_sent") is True and X.never_reached(r) and budget.refused == 1, r)


def section_approvals_release():
    print("approvals: a claim whose call never reached the cluster is released")
    d = str(TMP / "rel")
    r = AP.propose("weed", "inc_build_pilot", {"exp": "x"}, "R3", AUTO, "why", 1.0, root=d)
    iid = r["item"]["id"]
    AP.decide("weed", iid, "approve", OWNER, "ok", 2.0, root=d)
    check("released without a started execution is refused",
          not AP.record_executed("weed", iid, "released", AUTO, 3.0, root=d)["ok"])
    AP.record_executed("weed", iid, "started", AUTO, 3.0, root=d)
    rel = AP.record_executed("weed", iid, "released", AUTO, 4.0, outcome={"status": "failed"}, root=d)
    it = AP.state("weed", root=d)[iid]
    check("released: approved, no execution, the attempt kept, awaiting execution again",
          rel["ok"] and it["status"] == "approved" and it.get("execution") is None
          and len(it.get("released") or []) == 1
          and [i["id"] for i in AP.awaiting_execution("weed", root=d)] == [iid], it)
    AP.record_executed("weed", iid, "started", AUTO, 5.0, root=d)
    AP.record_executed("weed", iid, "done", AUTO, 6.0, root=d)
    check("  it can then run once; done closes it for good",
          not AP.record_executed("weed", iid, "started", AUTO, 7.0, root=d)["ok"]
          and not AP.record_executed("weed", iid, "released", AUTO, 7.0, root=d)["ok"])


def section_transport_outcomes():
    print("transport: a timeout after sbatch is an unknown outcome; the build is followed, never resubmitted")
    w = World("to_appr")
    w.tick()
    w.tick()
    own = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"][0]
    AP.decide("weed", own["id"], "approve", OWNER, "go", w.t[0], root=str(w.lab))
    w.tick(fn=_through(w, "submit", _timeout))
    st = w.state()
    ex = _build_execs(w)
    check("the timed-out submit is charged as an outcome that is unknown (not 'never started')",
          len(ex) == 1 and ex[0]["status"] == "failed" and ex[0]["charged"]
          and ex[0]["remote"]["may_have_run"] is True
          and "outcome is unknown" in " ".join(ex[0].get("reasons") or []), ex)
    check("the campaign follows pilot_v2 (RUN, building, uncertain): no failed step, no pause",
          st["phase"] == "RUN" and st["exp"] == "pilot_v2" and (st.get("building") or {}).get("uncertain")
          and st.get("fails") == 0 and not st.get("paused") and w.events("uncertain"),
          (st["phase"], st["exp"], st.get("building"), st.get("paused")))
    aps = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]
    check("the approval is closed (it may have run) and no new approval is filed",
          len(aps) == 1 and (aps[0].get("execution") or {}).get("phase") == "failed", aps)
    b = X.budget_now({"name": NAME}, w.xctx)
    check("the envelope holds its estimate", b["committed_su"] == 42.544, b["committed_su"])
    for _ in range(3):
        w.tick()
    st = w.state()
    check("the queued inc_build_pilot_v2 keeps it building; one submission ever",
          len(submits_of(w, "build")) == 1 and st["phase"] == "RUN"
          and (st.get("building") or {}).get("exp") == "pilot_v2", (w.submits, st["phase"]))
    w.build()
    w.tick()
    st = w.state()
    run = C._Run(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log,
                 w.hooks, RESOURCES, None, None, None)
    run.st = st
    check("the snapshot confirms it: built, the card cleared, its lineage 'executed'",
          w.events("built") and not st.get("building") and not st.get("card")
          and any(r["lever"] == "L1" and r["status"] == "executed" for r in run.lineage()),
          (st.get("card"), run.lineage()))

    w2 = World("to_env", autonomy="envelope")
    w2.replay_pass()
    w2.tick()
    w2.tick(fn=_through(w2, "submit", _timeout))
    for _ in range(4):
        w2.replay_pass()
        w2.tick()
    check("envelope: a timeout after sbatch, then 4 ticks: one submission, one job in the queue",
          len(submits_of(w2, "build")) == 1
          and len([j for j in w2.squeue if j["name"] == "inc_build_pilot_v2"]) == 1
          and w2.state()["exp"] == "pilot_v2", (w2.submits, w2.squeue))

    print("transport: an unknown outcome that never happened is found lost and proposed again")
    w3 = World("to_lost", autonomy="envelope")
    w3.replay_pass()
    w3.tick()

    def lost(script, timeout=60):                  # the ssh times out before sbatch ran
        w3.tick_calls[-1].append(script)
        return {"ok": False, "stdout": "", "stderr": "TIMEOUT", "returncode": -1}
    w3.replay_pass()
    w3.tick(fn=lost)
    first = (w3.events("uncertain") or [{}])[0].get("proposal_id")
    for _ in range(BUILD_LOST_TICKS):
        w3.replay_pass()
        w3.tick()
    st = w3.state()
    bf = w3.events("build_failed")
    check("nothing queued and no provenance for 3 snapshots: lost, back to pilot_v1, not a failed step",
          bf and bf[0]["kind"] == "lost" and st["exp"] == "pilot_v1" and st.get("fails") == 0
          and not st.get("paused") and "pilot_v2" not in st["exps"], (bf, st["exp"], st.get("fails")))
    for _ in range(3):
        w3.replay_pass()
        w3.tick()
        if submits_of(w3, "build"):
            break
    pro = [r for r in w3.events("proposed") if r.get("lever") == "L1"]
    check("L1 is proposed again under a new id (attempt 1) and runs once",
          len(pro) == 2 and pro[1]["proposal_id"] != first and pro[1]["attempt"] == 1
          and len(submits_of(w3, "build")) == 1, (pro, w3.submits))

    print("transport: a connection dropped mid-verb on an R2 job pauses; re-enabled, it observes")
    w4 = World("drop_r2")
    w4.tick()
    w4.tick()
    l1 = [i for i in w4.approvals().values() if i.get("action") == "inc_build_pilot"][0]
    AP.decide("weed", l1["id"], "deny", OWNER, "not now", w4.t[0], root=str(w4.lab))
    w4.tick()
    check("L4 is the item after the denial", ((w4.state().get("item") or {}).get("proposal") or {})
          .get("lever") == "L4")
    w4.tick(fn=_through(w4, "submit", _dropped))
    st = w4.state()
    check("the audit's outcome is unknown: paused for a person, phase DIAGNOSE (not GATE with no item)",
          (st.get("paused") or {}).get("reason", "").startswith("the outcome of L4")
          and st["phase"] == "DIAGNOSE" and st.get("item") is None, (st["phase"], st.get("paused")))
    C.configure(NAME, OWNER, enable=True, cfg_hooks=w4.hooks, lab_repo=str(w4.lab), clock=w4.clock)
    w4.tick()
    st = w4.state()
    check("enabled again: the next tick observes (one ssh) and diagnoses; L4 is not run twice",
          len(w4.tick_calls[-1]) == 1 and len(submits_of(w4, "audit")) == 1
          and st["phase"] == "COMPLETE" and (st.get("card") or {}).get("kind") == "residual",
          (len(w4.tick_calls[-1]), st["phase"], st.get("card")))

    print("transport: ssh never connected on an approved build: the approval is not used up")
    w5 = World("blip")
    w5.tick()
    w5.tick()
    own = [i for i in w5.approvals().values() if i.get("action") == "inc_build_pilot"][0]
    pid = (w5.state()["item"]["proposal"] or {}).get("id")
    AP.decide("weed", own["id"], "approve", OWNER, "go", w5.t[0], root=str(w5.lab))
    w5.mode = "raise"
    w5.tick()
    w5.mode = "ok"
    ap = w5.approvals()[own["id"]]
    st = w5.state()
    ex = _build_execs(w5)
    check("the claim is released: approved, no execution, one released attempt",
          ap["status"] == "approved" and ap.get("execution") is None and len(ap.get("released") or []) == 1, ap)
    check("not charged, not a failed step, the same item kept (same proposal id)",
          len(ex) == 1 and not ex[0]["charged"] and "never started" in " ".join(ex[0]["reasons"])
          and st.get("fails") == 0 and (st["item"]["proposal"] or {}).get("id") == pid
          and st.get("transport_failures") == 1 and w5.events("transport_failed"),
          (ex, st.get("fails"), st.get("item")))
    w5.tick()
    execd = w5.events("executed")
    check("the next tick runs the same approval once, as the person; no second approval",
          len(submits_of(w5, "build")) == 1 and execd and execd[0]["approval_id"] == own["id"]
          and execd[0]["decided_by"] == OWNER
          and len([i for i in w5.approvals().values() if i.get("action") == "inc_build_pilot"]) == 1,
          (w5.submits, execd))

    print("transport: the lab dies between the ssh and the outcome line; the build is followed")
    w6 = World("crash", autonomy="envelope")
    w6.replay_pass()
    w6.tick()
    w6.replay_pass()
    try:
        w6.tick(fn=_through(w6, "submit", _crash))
    except _Crash:
        w6.t[0] += TICK
    check("only the started record was written", [r["status"] for r in _build_execs(w6)] == ["started"])
    w6.replay_pass()
    w6.tick()
    st = w6.state()
    check("recovered as an unknown outcome: RUN on pilot_v2, not paused, not run again",
          w6.events("recovered") and st["phase"] == "RUN" and st["exp"] == "pilot_v2"
          and not st.get("paused") and len(submits_of(w6, "build")) == 1, (st["phase"], st.get("paused")))


BUILD_LOST_TICKS = C.BUILD_LOST_SNAPSHOTS


def section_stale_approvals():
    print("a stale approval never runs a lever twice on a parent")
    w = World("stale")
    w.tick()
    w.tick()
    own = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"][0]

    def brain_l1(w, child, pid):
        r = AP.propose("weed", "inc_build_pilot", {"exp": child, "replay_mode": "full", "est_gpu_hours": 42.544},
                       "R3", "tier2:qwen3", "brain L1", w.t[0], est_su=42.544, root=str(w.lab),
                       context={"campaign": NAME, "proposal_id": pid, "lever": "L1", "trigger": ["D1"],
                                "cites": [], "argv": ["python", "-m", "weed_optimizer_framework.tools.inc.pilot",
                                                      "build", "--exp", child, "--replay-mode", "full"],
                                "parent_exp": "pilot_v1", "child_exp": child, "proposed_by": "tier2:qwen3"})
        return r["item"]["id"]
    b1 = brain_l1(w, "pilot_v2", "brainprop1")
    AP.decide("weed", b1, "approve", OWNER, "the brain's", w.t[0], root=str(w.lab))
    w.tick()
    st = w.state()
    check("the brain's L1 is adopted and run; the ticker's own L1 is superseded (recorded)",
          len(submits_of(w, "build")) == 1 and w.events("superseded")
          and any(s.get("approval_id") == own["id"] for s in st.get("superseded") or []), st.get("superseded"))
    w.build()
    w.tick()
    w.finish()
    w.tick()
    AP.decide("weed", own["id"], "approve", OWNER, "stale card", w.t[0], root=str(w.lab))
    for _ in range(2):
        w.tick()
    st = w.state()
    ref = w.events("adopt_refused")
    check("the stale approval of the same lever on the same parent is refused, not run",
          len(submits_of(w, "build")) == 1 and ref and ref[0]["approval_id"] == own["id"]
          and "already applied" in ref[0]["reasons"][0] and own["id"] in (st.get("adopt_skip") or [])
          and len(w.events("executed")) == 1, (w.submits, ref))
    check("  with a card saying so", "not run" in ((st.get("card") or {}).get("title") or ""), st.get("card"))

    w2 = World("renamed")
    w2.tick()
    w2.tick()
    own2 = [i for i in w2.approvals().values() if i.get("action") == "inc_build_pilot"][0]
    b2 = brain_l1(w2, "pilot_v2b", "brainprop2")
    AP.decide("weed", b2, "approve", OWNER, "renamed", w2.t[0], root=str(w2.lab))
    w2.tick()
    AP.decide("weed", own2["id"], "approve", OWNER, "stale card", w2.t[0], root=str(w2.lab))
    for _ in range(2):
        w2.tick()
    check("while pilot_v2b runs, the stale approval is not run",
          [a[a.index("--") + 1:] for a in submits_of(w2, "build")]
          == [["pilot", "build", "--exp", "pilot_v2b", "--replay-mode", "full"]], w2.submits)
    # the next adopting phase (DIAGNOSE, GATE, COMPLETE) looks at it, with an ssh to spare
    run = C._Run(NAME, w2.config(), C.Paths(str(w2.lab)), C._SshBudget(w2), w2.clock, w2.log,
                 w2.hooks, RESOURCES, None, None, None)
    run.st = w2.state()
    w2.tick_calls.append([])
    ran = run._adopt_approved()
    check("a renamed duplicate (pilot_v2b ran; the stale pilot_v2 approval) is refused, no ssh",
          not ran and run.ssh.calls == 0
          and any(r["approval_id"] == own2["id"] and "already applied" in r["reasons"][0]
                  for r in w2.events("adopt_refused")), w2.events("adopt_refused"))

    w3 = World("page_l1")
    w3.tick()
    w3.tick()
    own3 = [i for i in w3.approvals().values() if i.get("action") == "inc_build_pilot"][0]
    ctx = X.Context(slurm_sh=w3, lab_repo=str(w3.lab), resources=RESOURCES, clock=w3.clock)
    w3.tick_calls.append([])
    r = X.submit({"policy_action": "inc_build_pilot", "lever": "L1", "parent_exp": "pilot_v1",
                  "trigger": ["D1"], "params": {"exp": "pilot_v2c", "replay_mode": "full",
                                                "est_gpu_hours": 42.544}},
                 actor=OWNER, campaign={"name": NAME}, ctx=ctx)
    check("a person runs an L1 of pilot_v1 from the page (pilot_v2c)", r["status"] == "executed", r["reasons"])
    AP.decide("weed", own3["id"], "approve", OWNER, "stale card", w3.t[0], root=str(w3.lab))
    w3.tick()
    st = w3.state()
    nt = [x for x in w3.events("not_taken") if x.get("approval_id") == own3["id"]]
    check("the ticker's own approved L1 is then dropped, not run a second time on pilot_v1; the "
          "same tick diagnoses again (L1 deferred, L4 next)",
          len(submits_of(w3, "build")) == 1 and nt and "already applied" in nt[0]["reasons"][0]
          and ((st.get("item") or {}).get("proposal") or {}).get("lever") == "L4",
          (w3.submits, (st.get("item") or {}).get("proposal", {}).get("lever"), nt))


def section_gate_and_brain_wait():
    print("a GATE with no item observes and diagnoses again; brain proposals wait in COMPLETE")
    w = World("gate_empty")
    w.tick()
    st = w.state()
    st["item"] = None
    C.save_state(C.Paths(str(w.lab)), NAME, st)
    w.tick()
    st = w.state()
    check("GATE with no item: DIAGNOSE observes (one ssh) and the lever is proposed again",
          w.events("gate_empty") and len(w.tick_calls[-1]) == 1 and st["phase"] == "GATE"
          and ((st.get("item") or {}).get("proposal") or {}).get("lever") == "L1"
          and len(w.events("proposed")) == 2, (st["phase"], len(w.tick_calls[-1])))
    w.tick()
    ap = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"][0]
    run = C._Run(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log,
                 w.hooks, RESOURCES, None, None, None)
    run.st = w.state()
    run.st["item"] = None
    run.st["brain"] = {"n": 1, "exp": "pilot_v1", "status": "merged",
                       "filed": [{"lever": "L1", "status": "filed", "approval_id": ap["id"]}]}
    run._brain_wait_step()
    check("brain proposals filed for a person: COMPLETE with an approval card naming them",
          run.st["phase"] == "COMPLETE" and run.st["card"]["kind"] == "approval"
          and run.st["card"].get("approval_ids") == [ap["id"]], run.st.get("card"))
    run._refresh_waiting_card()
    check("  the card stays while one is pending", run.st["card"]["kind"] == "approval")
    AP.decide("weed", ap["id"], "deny", OWNER, "no", w.t[0], root=str(w.lab))
    run._refresh_waiting_card()
    check("  once a person decided them all, a residual card replaces it",
          run.st["card"]["kind"] == "residual", run.st.get("card"))


def section_resume_and_goal():
    print("a stop-loss pause lifted by any enable of the config it was written to")
    w = World("resume_cfg")
    w.tick()
    r = C._Run(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log,
               w.hooks, RESOURCES, None, None, None)
    r.st = w.state()
    r._pause("stop-loss: probe")
    C.save_state(C.Paths(str(w.lab)), NAME, r.st)
    check("the pause was written to the config", (w.state().get("paused") or {}).get("config_written") is True
          and w.config().get("enabled") is False)
    cfg = w.hooks[0]()
    c = cfg["campaigns"][NAME]
    c["enabled"] = True
    c.pop("paused_reason", None)                   # a write that is not configure(): no resumed_utc
    w.hooks[1](cfg)
    w.tick()
    st = w.state()
    check("enabled with the reason cleared: resumed on the next tick, which acts",
          not st.get("paused") and w.events("resumed") and len(w.approvals()) == 1
          and C.status(NAME, w.hooks, str(w.lab))[NAME]["summary"]["alarm"] != "crit",
          (st.get("paused"), w.approvals()))

    w2 = World("resume_unwritten")
    w2.tick()

    def bad_save(c):
        raise OSError("disk full")
    r = C._Run(NAME, w2.config(), C.Paths(str(w2.lab)), C._SshBudget(None), w2.clock, w2.log,
               (w2.hooks[0], bad_save), RESOURCES, None, None, None)
    r.st = w2.state()
    r._pause("stop-loss: unwritten")
    C.save_state(C.Paths(str(w2.lab)), NAME, r.st)
    w2.tick()
    check("a pause the config never received is not lifted by the config's old enabled flag",
          (w2.state().get("paused") or {}).get("reason") == "stop-loss: unwritten"
          and len(w2.tick_calls[-1]) == 0, w2.state().get("paused"))
    C.configure(NAME, OWNER, enable=True, cfg_hooks=w2.hooks, lab_repo=str(w2.lab), clock=w2.clock)
    w2.tick()
    check("  a person's enable (resumed_utc) lifts it", not w2.state().get("paused"))

    print("a goal the ticker cannot check: autonomy off, a card, and no envelope grant")
    w3 = World("goal_bad", autonomy="envelope", goal=None)
    cfg = w3.hooks[0]()
    cfg["campaigns"][NAME]["goal"] = {"text": "realloop decision ready", "metric": "agreement",
                                      "target": 0.71, "direction": "up", "exam": "dev"}
    w3.hooks[1](cfg)
    w3.replay_pass()
    w3.tick()
    check("the card says the goal cannot be checked",
          (w3.state().get("card") or {}).get("title") == "The campaign goal cannot be checked"
          and len(w3.events("goal_invalid")) == 1, w3.state().get("card"))
    w3.replay_pass()
    w3.tick()
    it = w3.state().get("item") or {}
    check("the envelope build waits for a person (autonomy reads off)",
          not w3.submits and it.get("status") == "filed"
          and any("autonomy is off" in x for x in it.get("last_reasons") or []), it.get("last_reasons"))
    refused = False
    try:
        C.configure(NAME, OWNER, autonomy="envelope", cfg_hooks=w3.hooks, lab_repo=str(w3.lab),
                    clock=w3.clock)
    except ValueError as e:
        refused = "goal" in str(e)
    check("configure refuses autonomy 'envelope' while the goal cannot be checked", refused)
    C.set_goal(NAME, GOAL_D4, OWNER, cfg_hooks=w3.hooks, lab_repo=str(w3.lab), clock=w3.clock)
    w3.replay_pass()
    w3.tick()
    check("with a goal it can check, the envelope build runs", len(submits_of(w3, "build")) == 1,
          (w3.state().get("item") or {}).get("last_reasons"))


def section_build_su():
    print("the build job's own SU reaches 'spent' when its experiment reports")
    rep_hours = sum(float(v["hours"]) for v in pilot_v2_files()["report"]["gpu_hours"].values())
    for measured in (False, True):
        w = World("build_su_%d" % measured)
        w.tick()
        w.tick()
        own = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"][0]
        AP.decide("weed", own["id"], "approve", OWNER, "go", w.t[0], root=str(w.lab))
        w.tick()
        b = X.budget_now({"name": NAME}, w.xctx)
        check("building: the estimate (runs + the 4 h build job) is committed",
              b["committed_su"] == 42.544 and b["spent_su"] == 0.0, b["committed_su"])
        w.build()
        if measured:
            job = w.events("executed")[0]["job_ids"][0]
            w._w("_campaign/provenance/pilot_v2.json", json.dumps(
                {"exp": "pilot_v2", "attempts": [{"job_id": job, "status": "advanced",
                                                  "started_utc": "2026-09-27T06:00:00Z",
                                                  "updated_utc": "2026-09-27T06:30:00Z"}]}))
        w.tick()
        w.finish()
        w.tick()
        b = X.budget_now({"name": NAME}, w.xctx)
        want = rep_hours + (0.5 if measured else 4.0)
        rep = [r for r in w.events("reported") if r["exp"] == "pilot_v2"]
        bj = ((rep[0].get("spend") or {}).get("build_job") or {}) if rep else {}
        check("reported (%s): committed released, spent = the report's units + the build job"
              % ("provenance-measured 0.5 h" if measured else "walltime 4 h"),
              b["committed_su"] == 0.0 and abs(b["spent_su"] - want) < 1e-6 and bj.get("ok")
              and bj.get("measured") is measured, (b["spent_su"], want, bj))


def section_rules_version():
    print("rules version: one prospective record per (pilot, rules version); a change re-diagnoses, runs nothing "
          "twice")
    w = World("rules", autonomy="envelope", goal=None)
    paths = C.Paths(str(w.lab))
    ver = DG.rules_version()
    w.replay_pass()
    w.tick()
    w.tick()
    w.build()
    w.tick()
    w.finish()
    w.replay_pass()
    w.tick()
    st = w.state()
    first = paths.replay / DG.prospective_name("pilot_v2", ver)
    check("pilot_v2 diagnosed under rules %s: its record is named for them, ready, and the state keeps the version"
          % ver, first.is_file() and json.loads(first.read_text())["ready"] is True
          and st["prospective"]["pilot_v2"]["rules_version"] == ver and st["rules_version"] == ver,
          (sorted(p.name for p in paths.replay.glob("*.json")), st.get("rules_version")))
    first_sha = hashlib.sha256(first.read_bytes()).hexdigest()
    w.tick()
    check("L4 (R2) runs: WAIT_JOB on the audit", w.state()["phase"] == "WAIT_JOB" and len(submits_of(w, "audit")) == 1)
    real = DG.rules_version
    new_ver = "0123456789ab"
    DG.rules_version = lambda: new_ver                  # the deployed rules change while the job is out
    try:
        w.tick()
        check("the audit job is still queued: WAIT_JOB, no record of the new rules yet",
              w.state()["phase"] == "WAIT_JOB" and not (paths.replay / DG.prospective_name("pilot_v2", new_ver))
              .exists())
        w.squeue = [j for j in w.squeue if j["name"] != "inc_audit"]
        w.tick()
        st = w.state()
        second = paths.replay / DG.prospective_name("pilot_v2", new_ver)
        pro = [r for r in w.events("prospective_d4") if r.get("exp") == "pilot_v2"]
        check("the R2 job finished: DIAGNOSE re-diagnosed pilot_v2 and wrote its record under the new rules, "
              "beside the old one",
              second.is_file() and first.is_file() and json.loads(second.read_text())["rules_version"] == new_ver
              and st["prospective"]["pilot_v2"]["rules_version"] == new_ver and st["rules_version"] == new_ver,
              sorted(p.name for p in paths.replay.glob("*.json")))
        check("  the ledger entry names the superseded record and its sha256; the state keeps it in history",
              pro and pro[-1]["rules_version"] == new_ver
              and [(x["path"], x["sha256"], x["rules_version"]) for x in pro[-1]["superseded"]]
              == [(str(first), first_sha, ver)]
              and [h.get("sha256") for h in st["prospective"]["pilot_v2"]["history"]] == [first_sha],
              pro and pro[-1].get("superseded"))
        check("  the earlier record is untouched", hashlib.sha256(first.read_bytes()).hexdigest() == first_sha)
        check("  nothing that ran ran again (one L1 build, one audit), and the campaign completes",
              len(submits_of(w, "build")) == 1 and len(submits_of(w, "audit")) == 1 and st["phase"] == "COMPLETE",
              (st["phase"], w.submits))
        n_pro = len(w.events("prospective_d4"))
        w.tick()
        check("COMPLETE under the rules it was diagnosed with: no re-diagnosis, no new record",
              w.state()["phase"] == "COMPLETE" and len(w.events("prospective_d4")) == n_pro
              and not w.events("rules_changed"))
    finally:
        DG.rules_version = real

    print("  the deploy: a COMPLETE campaign whose pilot record predates versioned records (ready false)")
    w = World("legacy", autonomy="envelope", goal=None)
    paths = C.Paths(str(w.lab))
    w.replay_pass()
    w.tick()
    w.tick()
    w.build()
    w.tick()
    w.finish()
    w.replay_pass()
    w.tick()
    w.tick()
    w.squeue = [j for j in w.squeue if j["name"] != "inc_audit"]
    w.tick()
    st = w.state()
    check("the campaign completed (L4 applied, L2 deferred: no Step 1)", st["phase"] == "COMPLETE", st["phase"])
    cur = paths.replay / DG.prospective_name("pilot_v2", ver)
    rec = json.loads(cur.read_text())
    cur.unlink()
    legacy = paths.replay / "prospective_d4_pilot_v2.json"
    for k in ("rules_version", "rules_files"):
        rec.pop(k, None)
    rec.update(ready=False, blocked_by="D1", replay_mode=None, recipes=None, gate_flips_mode=None)
    legacy.write_text(json.dumps(rec, indent=1, sort_keys=True))
    legacy_sha = hashlib.sha256(legacy.read_bytes()).hexdigest()
    st["prospective"] = {"pilot_v2": {"path": str(legacy), "sha256": legacy_sha, "ready": False,
                                      "utc": "2026-09-27T15:00:00Z"}}
    st.pop("rules_version", None)
    C.save_state(paths, NAME, st)
    w.put_step1()
    run = C._Run(NAME, w.config(), paths, C._SshBudget(None), w.clock, w.log, w.hooks, RESOURCES, None, None,
                 None)
    run.st = C.load_state(paths, NAME)
    check("before the deploy's DIAGNOSE, an L2 from pilot_v2 is held: its only record is not of the current rules",
          "under the current rules version" in run._d4_unfrozen({"lever": "L2", "parent_exp": "pilot_v2"},
                                                                E.load_dir(w.inc, "pilot_v2", exps=["pilot_v2"])))
    n_audit, n_build = len(submits_of(w, "audit")), len(submits_of(w, "build"))
    w.replay_pass()
    w.tick()
    st = w.state()
    item = (st.get("item") or {}).get("proposal") or {}
    pro = [r for r in w.events("prospective_d4") if r.get("exp") == "pilot_v2"]
    changed = w.events("rules_changed")
    check("the next tick sees the rules change (none recorded -> %s) and DIAGNOSEs pilot_v2 again" % ver,
          changed and changed[-1]["rules_version"] == ver and changed[-1]["previous_rules_version"] is None,
          changed)
    check("  the record of the current rules is written (ready), superseding the unversioned one by its sha256",
          cur.is_file() and json.loads(cur.read_text())["ready"] is True and pro and pro[-1]["rules_version"] == ver
          and [(x["path"], x["sha256"]) for x in pro[-1]["superseded"]] == [(str(legacy), legacy_sha)],
          pro and pro[-1].get("superseded"))
    check("  the unversioned record stays, byte for byte", legacy.is_file()
          and hashlib.sha256(legacy.read_bytes()).hexdigest() == legacy_sha)
    check("  and only then is L2 proposed (relevance present), after the record in the ledger",
          item.get("lever") == "L2" and item.get("parent_exp") == "pilot_v2" and "--relevance" in item.get("argv")
          and w.ledger().index(pro[-1]) < w.ledger().index([r for r in w.events("proposed")
                                                            if r.get("lever") == "L2"][0]),
          (item.get("lever"), item.get("argv")))
    check("  nothing that ran ran again", (len(submits_of(w, "audit")), len(submits_of(w, "build")))
          == (n_audit, n_build), w.submits)
    w.replay_pass()
    w.tick()
    loops = [a for a in w.submits if "--exp" in a and "realloop_v1" in a]
    check("the L2 runs once under the envelope: RUN realloop_v1", len(loops) == 1
          and w.state()["exp"] == "realloop_v1" and w.state()["phase"] == "RUN", (w.state()["phase"], w.submits))
    check("one ssh per tick", max_calls(w) <= 1, [len(c) for c in w.tick_calls])


def section_d4_guard():
    print("no real loop from a pilot's D4 decision before its prospective record")
    w = World("l2guard", autonomy="envelope", goal=None)
    w.put_step1()
    w.replay_pass()
    w.tick()
    w.replay_pass()
    w.tick()
    w.build()
    w.tick()
    w.finish()
    real = C.DG.prospective_d4

    def fail(*a, **k):
        raise OSError("read-only file system (simulated)")
    C.DG.prospective_d4 = fail
    try:
        w.replay_pass()
        w.tick()
    finally:
        C.DG.prospective_d4 = real
    st = w.state()
    nt = [r for r in w.events("not_taken") if r.get("lever") == "L2"]
    check("the record could not be written: L2 is not taken for that reason, never proposed",
          w.events("prospective_failed") and nt
          and all("not frozen in a prospective record" in " ".join(r.get("reasons") or []) for r in nt)
          and not [r for r in w.events("proposed") if r.get("lever") in ("L2", "L6")]
          and "pilot_v2" not in (st.get("prospective") or {}), (nt, st.get("prospective")))
    for _ in range(6):
        w.replay_pass()
        w.tick()
        if w.state()["phase"] == "WAIT_JOB":
            w.squeue = [j for j in w.squeue if not j["name"].startswith("inc_audit")]
        if [r for r in w.events("proposed") if r.get("lever") == "L2"]:
            break
    led = w.ledger()
    pro = [r for r in led if r.get("event") == "prospective_d4" and r.get("exp") == "pilot_v2"]
    l2 = [r for r in led if r.get("event") == "proposed" and r.get("lever") == "L2"]
    check("once the record is written, L2 is proposed after it",
          pro and l2 and led.index(pro[0]) < led.index(l2[0]), ([r["event"] for r in led[-8:]], pro, l2))

    ev = E.load_dir(FIX, "pilot_v1", exps=["pilot_v1"])
    run = C._Run(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log,
                 w.hooks, RESOURCES, None, None, None)
    run.st = C._blank_state(NAME, w.config())
    l6 = {"lever": "L6", "trigger": ["D11"], "parent_exp": "pilot_v1"}
    check("L6 from D11 on a pilot is held too (keyed on the build, not the trigger)",
          "not frozen" in run._d4_unfrozen(l6, ev))
    ver = DG.rules_version()
    replay = C.Paths(str(w.lab)).replay
    replay.mkdir(parents=True, exist_ok=True)

    def frozen(name, **rec):
        body = dict({"format": DG.PROSPECTIVE_FORMAT, "exp": "pilot_v1", "rules_version": ver, "ready": True,
                     "outcome": "decision_slot_ready", "replay_mode": "full", "recipes": ["full"]}, **rec)
        path = replay / name
        path.write_text(json.dumps(body, sort_keys=True))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    path, sha = frozen(DG.prospective_name("pilot_v1", ver))
    run.st["prospective"] = {"pilot_v1": {"sha256": sha, "path": str(path), "rules_version": ver, "ready": True}}
    check("  and released by a READY prospective record of the pilot under the current rules version",
          run._d4_unfrozen(l6, ev) == "", run._d4_unfrozen(l6, ev))
    run.st["prospective"] = {"pilot_v1": {"sha256": sha, "path": str(path), "rules_version": "0" * 12,
                                          "ready": True}}
    check("  not by the ticker's record of another rules version",
          "under the current rules version" in run._d4_unfrozen(l6, ev), run._d4_unfrozen(l6, ev))
    path, sha = frozen(DG.prospective_name("pilot_v1", ver), ready=False, outcome="decision_slot_ready",
                       blocked_by="D1", replay_mode=None, recipes=None)
    run.st["prospective"] = {"pilot_v1": {"sha256": sha, "path": str(path), "rules_version": ver, "ready": False}}
    check("  not by a record of the current rules that is not ready (D4 blocked by D1)",
          "as not ready" in run._d4_unfrozen(l6, ev) and "D1" in run._d4_unfrozen(l6, ev), run._d4_unfrozen(l6, ev))
    path, sha2 = frozen(DG.prospective_name("pilot_v1", ver))
    check("  not by a file the ticker did not write (its sha256 is not the state's)",
          "not the one the ticker wrote" in run._d4_unfrozen(l6, ev), run._d4_unfrozen(l6, ev))
    path.unlink()
    legacy, lsha = frozen("prospective_d4_pilot_v1.json", rules_version=None)
    run.st["prospective"] = {"pilot_v1": {"sha256": lsha, "path": str(legacy), "ready": True}}
    check("  not by a ready record written before records were versioned (prospective_d4_pilot_v1.json)",
          "under the current rules version" in run._d4_unfrozen(l6, ev), run._d4_unfrozen(l6, ev))
    rec_, _p, why = DG.current_prospective(replay, "pilot_v1")
    check("  current_prospective: no record under the current rules, and it names the older one",
          rec_ is None and "prospective_d4_pilot_v1.json" in why, why)
    legacy.unlink()
    check("an L2 with no evidence to tell a pilot from a loop waits",
          run._d4_unfrozen({"lever": "L2", "parent_exp": "pilot_v1"}, None) != "")
    inc = M.CLUSTER_INC_DIR
    loop = {"exp": "realloop_v1", "type": "chain", "builder": "inc.realloop build",
            "initialised_utc": "2026-10-01T00:00:00Z", "replay_mode": "sample",
            "recipes": {"full": {"epochs": 30}},
            "base": {"source_manifest": inc + "/step1/base_B.jsonl"}, "steps": []}
    ev2 = E.from_texts({"pilot_v1/exp.json": (FIX / "pilot_v1/exp.json").read_text(),
                        "realloop_v1/exp.json": json.dumps(loop)}, "realloop_v1")
    run.st["prospective"] = {}
    check("an L2 rebuilding a real loop (from D1) needs no pilot record",
          run._d4_unfrozen({"lever": "L2", "trigger": ["D1"], "parent_exp": "realloop_v1"}, ev2) == "")
    l2 = {"lever": "L2", "policy_action": "inc_build_realloop", "risk": "R3", "parent_exp": "pilot_v1",
          "child_exp": "realloop_v1", "trigger": ["D4"], "proposed_by": "tier2:qwen3",
          "params": {"exp": "realloop_v1", "replay_mode": "full", "recipes": "full"}}
    refuse, wait = run._adoption_check(l2, ev)
    check("an approved L2 on a pilot with no prospective record waits (not refused for good)",
          refuse == "" and "not frozen" in wait, (refuse, wait))
    # a brain plan carrying that L2: never filed for a person before the record exists
    run.ev = ev
    run.st["brain"] = {"n": 1, "exp": "pilot_v1", "status": "submitted"}
    submitted = []
    saved = (C.V.validate, C.BP.merge, C.OC.record_validation, C.X.submit)
    C.V.validate = lambda *a, **k: {"counts": {}, "proposed_by": "tier2:qwen3"}
    C.BP.merge = lambda *a, **k: {"proposals": [dict(l2)], "cards": []}
    C.OC.record_validation = lambda *a, **k: None
    C.X.submit = lambda p, **k: submitted.append(p) or {"status": "filed", "approval_id": "ap-x"}
    try:
        run._merge_brain({"reply": {"model": "qwen3"}})
    finally:
        C.V.validate, C.BP.merge, C.OC.record_validation, C.X.submit = saved
    check("a brain L2 from a pilot's decision is not filed before the prospective record",
          not submitted and run.st["brain"]["filed"][0]["status"] == "not_taken"
          and "not frozen" in run.st["brain"]["filed"][0]["reasons"][0], run.st["brain"].get("filed"))


def _to_filed_l2(autonomy):
    """A World whose campaign holds its own L2 on pilot_v2 at GATE, filed and
    waiting: for a person (autonomy off), or (envelope) because the envelope
    rule does not hold (no replay pass recorded for this code)."""
    w = World("inflight_%s" % autonomy, autonomy=autonomy, goal=None)
    w.put_step1()
    if autonomy == "envelope":
        w.replay_pass()
    w.tick()
    if autonomy == "envelope":
        w.replay_pass()
    w.tick()
    if autonomy == "off":
        l1 = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]
        AP.decide("weed", l1[0]["id"], "approve", OWNER, "go", w.t[0], root=str(w.lab))
        w.tick()
    w.build()
    w.tick()
    w.finish()
    for _ in range(10):
        st = w.state()
        item = (st.get("item") or {}).get("proposal") or {}
        if item.get("lever") == "L2" and (st.get("item") or {}).get("status") == "proposed" \
                and autonomy == "envelope":
            try:
                w.xctx.replay_result.unlink()      # the envelope rule no longer holds: L2 is filed
            except FileNotFoundError:
                pass
        if item.get("lever") == "L2" and (st.get("item") or {}).get("status") == "filed":
            break
        if autonomy == "envelope" and item.get("lever") != "L2":
            w.replay_pass()
        w.tick()
        if w.state()["phase"] == "WAIT_JOB":
            w.squeue = [j for j in w.squeue if not j["name"].startswith("inc_audit")]
    return w


def section_rules_change_in_flight():
    print("a rules change while the campaign's own L2 waits at GATE: dropped, D4 frozen again under the new "
          "rules, and only then run (or never, when the new rules' decision is not ready)")
    real_ver, real_th = DG.rules_version, DG.load_thresholds
    new_ver = "0123456789ab"

    def strict():
        th = real_th()
        th["D4"]["ready_rate"]["value"] = {"num": 7, "den": 7}
        return th
    for autonomy in ("off", "envelope"):
        for ready_after in (True, False):
            tag = "%s, the new rules' decision %s" % (autonomy, "ready" if ready_after else "not ready")
            w = _to_filed_l2(autonomy)
            paths = C.Paths(str(w.lab))
            st = w.state()
            item = st.get("item") or {}
            old_ver = real_ver()
            check("%s: the own L2 on pilot_v2 is filed and waiting, from a READY record of rules %s"
                  % (tag, old_ver), (item.get("proposal") or {}).get("lever") == "L2"
                  and item.get("status") == "filed"
                  and st["prospective"]["pilot_v2"]["rules_version"] == old_ver
                  and st["prospective"]["pilot_v2"]["ready"] is True, (st["phase"], item.get("status")))
            aid = item.get("approval_id")
            n_loops = len([a for a in w.submits if "realloop_v1" in a])
            DG.rules_version = lambda: new_ver
            if not ready_after:
                DG.load_thresholds = strict
            try:
                if autonomy == "off":
                    AP.decide("weed", aid, "approve", OWNER, "go", w.t[0], root=str(w.lab))
                else:
                    w.replay_pass()
                n_led = len(w.ledger())
                for _ in range(4):
                    w.tick()
                    if w.state()["phase"] == "WAIT_JOB":
                        w.squeue = [j for j in w.squeue if not j["name"].startswith("inc_audit")]
                led = w.ledger()[n_led:]
                ev_names = [r.get("event") for r in led]
                dropped = [r for r in led if r.get("event") == "not_taken" and r.get("lever") == "L2"
                           and "rules version" in " ".join(r.get("reasons") or [])]
                pro = [r for r in led if r.get("event") == "prospective_d4" and r.get("exp") == "pilot_v2"
                       and r.get("rules_version") == new_ver]
                ran = [r for r in led if r.get("event") == "executed" and r.get("lever") == "L2"]
                second = paths.replay / DG.prospective_name("pilot_v2", new_ver)
                check("  %s: the waiting L2 is dropped first (not_taken, naming the rules versions)" % tag,
                      dropped and led.index(dropped[0]) == min(i for i, r in enumerate(led)
                                                               if r.get("event") in ("not_taken", "executed",
                                                                                     "prospective_d4")),
                      ev_names)
                check("  %s: D4 on pilot_v2 is frozen under the new rules (%s), superseding the old record"
                      % (tag, "ready" if ready_after else "not ready"),
                      pro and second.is_file() and json.loads(second.read_text())["ready"] is ready_after
                      and pro[0]["superseded"] and pro[0]["superseded"][0]["rules_version"] == old_ver,
                      (pro, sorted(x.name for x in paths.replay.glob("*.json"))))
                loops = [a for a in w.submits if "realloop_v1" in a]
                if ready_after:
                    check("  %s: the L2 runs once, and only after the new rules' record" % tag,
                          ran and pro and len(loops) == n_loops + 1 and led.index(pro[0]) < led.index(ran[0])
                          and ran[0].get("approval_id") == aid, (ev_names, len(loops)))
                else:
                    check("  %s: no real loop is built: nothing runs the approved L2 under the new rules" % tag,
                          not ran and len(loops) == n_loops
                          and (w.approvals().get(aid) or {}).get("execution") is None, (ev_names, len(loops)))
                check("  %s: one ssh per tick" % tag, max_calls(w) <= 1, [len(c) for c in w.tick_calls])
            finally:
                DG.rules_version, DG.load_thresholds = real_ver, real_th

    print("  an approved L2 whose build does not follow the READY record of the current rules is refused")
    w = World("differs", autonomy="off", goal=None)
    ev = E.load_dir(FIX, "pilot_v1", exps=["pilot_v1"])
    run = C._Run(NAME, w.config(), C.Paths(str(w.lab)), C._SshBudget(None), w.clock, w.log,
                 w.hooks, RESOURCES, None, None, None)
    run.st = C._blank_state(NAME, w.config())
    ver = DG.rules_version()
    replay = C.Paths(str(w.lab)).replay
    replay.mkdir(parents=True, exist_ok=True)
    path = replay / DG.prospective_name("pilot_v1", ver)
    path.write_text(json.dumps({"format": DG.PROSPECTIVE_FORMAT, "exp": "pilot_v1", "rules_version": ver,
                                "ready": True, "outcome": "decision_slot_ready", "replay_mode": "full",
                                "recipes": ["full"], "gate_flips_mode": "net"}, sort_keys=True))
    run.st["prospective"] = {"pilot_v1": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "path": str(path), "rules_version": ver, "ready": True}}
    l2 = {"lever": "L2", "policy_action": "inc_build_realloop", "risk": "R3", "parent_exp": "pilot_v1",
          "child_exp": "realloop_v1", "trigger": ["D4"], "proposed_by": "tier2:qwen3",
          "params": {"exp": "realloop_v1", "replay_mode": "full", "recipes": "full", "gate_flips_mode": "net"}}
    check("  the build that follows it (full, full, net) passes", run._adoption_check(l2, ev) == ("", ""),
          run._adoption_check(l2, ev))
    for key, val, what in (("recipes", "freeze", "recipes"), ("replay_mode", "sample", "replay_mode"),
                           ("gate_flips_mode", None, "gate_flips_mode")):
        bad = copy.deepcopy(l2)
        if val is None:
            bad["params"].pop(key)             # the builders' default (negative), not the record's net
        else:
            bad["params"][key] = val
        refuse, wait = run._adoption_check(bad, ev)
        check("  another %s is refused for good, not held" % what,
              wait == "" and "does not follow D4's decision" in refuse and what in refuse, (refuse, wait))


def main():
    section_cli()
    section_parse_remote()
    section_approvals_release()
    section_cycle_with_approval()
    section_denial()
    section_envelope_and_restart()
    section_child_without_goal()
    section_stop_loss_and_pause()
    section_health()
    section_test_blindness()
    section_ssh_budget()
    section_scheduler_hook()
    section_brain()
    section_brain_wait_and_adoption()
    section_open_issue_fixes()
    section_transport_outcomes()
    section_stale_approvals()
    section_gate_and_brain_wait()
    section_resume_and_goal()
    section_build_su()
    section_d4_guard()
    section_rules_version()
    section_rules_change_in_flight()
    print("\n%d failure(s)" % len(FAILURES))
    if FAILURES:
        for f in FAILURES:
            print("  - %s" % f)
    else:
        print("ALL PASS")
    return 1 if FAILURES else 0


def test_campaign():
    """pytest entry point: the whole script, failing on any failed check."""
    del FAILURES[:]
    assert main() == 0, FAILURES


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    sys.exit(rc)
