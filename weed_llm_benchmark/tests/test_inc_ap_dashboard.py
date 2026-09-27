#!/usr/bin/env python3
"""The INC campaign surface on the dashboard (docs/INC_AUTOPILOT.md (e)).

Every file the routes read here is written by the real ticker: each test
world runs inc_autopilot.campaign.tick() against a fake cluster that answers
the executor's script with the real remote.py verbs (campaign-snapshot,
snapshot, status) on a temporary INC_DIR laid out from the replay fixtures,
with only the driver's advance, the report writer, squeue and sbatch
replaced. The lab tree is model.LAB_REPO (LAB_REPO is set before import), so
the ticker, the executor and the dashboard all use campaign.Paths(None), as
on the lab.

What is asserted:
  * the routes read the ticker's own files: its state (phase, current
    experiment after a build moved it to the child), latest_snapshot.json,
    the snapshot history, frozen.json, the ledger copy, diagnoses.json and
    campaign_status.json; no GET reaches the cluster;
  * the step x chain grid reproduces the report's agreement and each cell
    cites its ledger line; the display-only final table survives freezing;
  * the diagnoses shown are the ticker's record, or a preview labelled as
    such, built from the ticker's own inputs (its lineage for this campaign
    only) and never alarmed on; perturbing the display-only table moves
    nothing;
  * the ledger is the ticker's copy: a rewritten ledger on the cluster
    (prefix_mismatch) replaces the lab's lines;
  * health: a stop-loss pause in the config or only in the ticker's state is
    crit, and a page resume ends it for the ticker too; a stale tick is crit;
    an approval card is warn; a recorded crit diagnosis on the current
    experiment is crit; a tree the ticker does not write is crit;
  * POST /api/inc/campaign goes through campaign.configure / pause /
    set_goal: goals the ticker reads (check_goal), the scheduler's lock,
    the campaign ledger, administrators only;
  * POST /api/inc/action/{verb} as human:<person> through the executor; a
    manual snapshot is kept in the ticker's format; a real loop is refused
    while D4's decision on the latest pilot is not frozen;
  * the served page: no backslash, no inline handler, parses, no direct
    cancel button;
  * in the real dashboard app: routes behind the auth middleware,
    /api/health/inc exempt, X-User ignored for INC writes, INC run and build
    jobs refused by /api/cancel_job, other INC jobs need cluster access and
    are recorded, the job-log globs anchored, array ids in _batch_sacct.

Run:  python3 tests/test_inc_ap_dashboard.py
"""
import base64
import copy
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_dash_"))
HOME = TMP / "home"
HOME.mkdir()
LAB = TMP / "lab"
LAB.mkdir()
os.environ["HOME"] = str(HOME)                     # ~/.round_scheduler.json, ~/.dash_admins
os.environ["INC_DIR"] = str(TMP / "inc")            # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "clusterrepo")
os.environ["REPO_ROOT"] = str(LAB)                  # the dashboard's lab repo
os.environ["LAB_REPO"] = str(LAB)                   # model.LAB_REPO: where the ticker writes
(TMP / "dashpass").write_text("test-pass\n")
os.environ["DASHPASS_FILE"] = str(TMP / "dashpass")
os.environ["DASH_USER"] = "tester"
os.environ["ROBOFLOW_KEY_FILE"] = str(TMP / "no_roboflow_key")
for k in ("CLUSTER_SSH", "INC_JOB_SCRIPT", "BRAIN_SU_LEDGER_DIR", "INCAP_MAX_FILE_BYTES",
          "INCAP_MAX_LEDGER_BYTES"):
    os.environ.pop(k, None)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI                                          # noqa: E402
from fastapi.testclient import TestClient                            # noqa: E402

from weed_optimizer_framework.tools import inc_dashboard as IDB      # noqa: E402
from weed_optimizer_framework.tools import round_scheduler as RS     # noqa: E402
from weed_optimizer_framework.tools.brain import approvals as AP     # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import campaign as C    # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG   # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E    # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import executor as EX   # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as RM     # noqa: E402

assert pathlib.Path(M.LAB_REPO).resolve() == LAB.resolve(), M.LAB_REPO
FIX = ROOT / "tests" / "fixtures" / "inc_replay"
INC = pathlib.Path(os.environ["INC_DIR"])
PATHS = C.Paths(None)
ADMIN, MEMBER, STRANGER = "boss@lab.org", "member@lab.org", "stranger@lab.org"
OWNER = "human:owner@example.org"
NAME = "c1"
NON_DEV = ("test", "ood22", "ood23", "imageweeds")
SEG_RE = re.compile(r'echo "INCAP_SEG (\d+)"; (.*?); echo "INCAP_SEG_END', re.S)
RESOURCES = {"mongo_ok": True, "cluster_reachable": True}
FIXTURE_LEDGER = [json.loads(x) for x in (FIX / "pilot_v1" / "ledger.jsonl").read_text().splitlines()
                  if x.strip()]
N_LEDGER = len(FIXTURE_LEDGER)


class QuietLog(object):
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))

    def error(self, m):
        self.lines.append(("error", m))


# ------------------------------------------------------------ the fake cluster and the ticker
class World(object):
    """A clean lab tree and INC_DIR, the fake cluster behind slurm_sh, the
    real ticker. `running` experiments are pilot_v1's files with a state that
    is not done, so the campaign stays in RUN and observes every tick."""

    def __init__(self, current="pilot_v1", exps=("b0_v1", "pilot_v1"), running=(), goal=None):
        for d in (LAB, INC):
            shutil.rmtree(str(d), ignore_errors=True)
            d.mkdir(parents=True)
        pathlib.Path(RS._CFG_FILE).write_text(json.dumps({"domains": {}}))
        IDB._FILE_CACHE.clear()
        IDB._MIRRORS.clear()
        IDB._DIAG_CACHE.clear()
        IDB._REPLAY.clear()
        self.t0 = time.time()
        self.squeue, self.submits, self.verbs, self.calls = [], [], [], []
        self.drift, self.next_job = False, 7000
        self.log = QuietLog()
        for sh in ROOT.glob("run_inc_*.sh"):          # levers.py reads their #SBATCH --time
            shutil.copy(str(sh), str(LAB / sh.name))
        self._copy("pilot_v1", done=True)
        self._copy("b0_v1", state=False)
        for e in running:
            self._copy(e, done=False, src="pilot_v1", report=False)
        C.configure(NAME, OWNER, enable=True, exps=list(exps), current=current, clock=self.clock)
        if goal is not None:
            C.set_goal(NAME, goal, OWNER, clock=self.clock)

    @staticmethod
    def clock():
        """Wall-clock time: the page's own writes (a resume, a pause) are stamped
        with it too, and the order of the two is what is tested."""
        return time.time()

    def _copy(self, exp, done=True, src=None, report=True, state=True):
        src = src or exp
        dst = INC / exp
        dst.mkdir(parents=True, exist_ok=True)
        for p in (FIX / src).iterdir():
            if p.name == "report.json" and not report:
                continue
            text = p.read_text()
            if src != exp:
                text = text.replace(src, exp)
            (dst / p.name).write_text(text)
        if state:
            st = {"exp": exp, "type": "chain", "done": done, "generation": 5, "runs": {},
                  "done_utc": "2026-09-27T07:49:08Z" if done else None, "blocked": {},
                  "chains": {}, "submissions": [], "unblocks": []}
            (dst / "state.json").write_text(json.dumps(st))
            os.utime(str(dst / "state.json"), (self.t0 - 7200, self.t0 - 7200))
            if (dst / "report.json").exists():
                os.utime(str(dst / "report.json"), (self.t0 - 3600, self.t0 - 3600))

    # ---- slurm_sh
    def __call__(self, script, timeout=60):
        self.calls.append(script)
        out = ["Welcome to Bridges-2"]
        for m in SEG_RE.finditer(script):
            i, argv = int(m.group(1)), shlex.split(m.group(2))
            out.append("INCAP_SEG %d" % i)
            rec = self._answer(argv)
            out.append("INCAP " + json.dumps(rec, default=str))
            out.append("INCAP_SEG_END %d %d" % (i, 0 if rec.get("ok") else 1))
        return {"ok": True, "stdout": "\n".join(out), "stderr": "", "returncode": 0}

    def _answer(self, argv):
        if argv[:3] == ["python", "-u", "-c"]:
            return {"ok": False, "error": "no inline code on this fake cluster"}
        args = argv[4:]                       # python -u -m MODULE ARGS
        self.verbs.append(args)
        if args[0] == "submit":
            self.submits.append(args)
            self.next_job += 1
            a = args[args.index("--") + 1:] if "--" in args else args
            if "--exp" in a:
                self.squeue.append({"id": str(self.next_job), "name": "inc_build_%s" % a[a.index("--exp") + 1],
                                    "state": "PENDING"})
            return {"verb": "submit", "ok": True, "argv": args, "job_id": str(self.next_job)}
        self._activate()
        return RM.dispatch(args)

    def _activate(self):
        world = self
        RM.advance = self._advance
        RM.report = self._report
        RM.squeue_jobs = lambda prefix="inc_": {"ok": True, "jobs": [
            dict(j, elapsed="0:01", submit="x") for j in world.squeue if j["name"].startswith(prefix)]}

    def _advance(self, exp, backend=None):
        rec = RM.base_record("advance")
        rec["exp"] = exp
        if self.drift:
            return RM.fail(rec, "the outer package copy /o (what the INC jobs import) differs from "
                                "the git-tracked copy /n in ['inc/gate.py']", "DriftError", "code_drift")
        p = INC / exp / "state.json"
        st = json.loads(p.read_text())
        if not st.get("done"):
            m = p.stat().st_mtime
            st["generation"] = int(st.get("generation") or 0) + 1
            p.write_text(json.dumps(st))
            os.utime(str(p), (m, m))
        rec["result"] = {"locked": False, "passes": 1, "submitted": 0, "job_ids": [],
                         "done": bool(st.get("done")), "lines": []}
        return rec

    def _report(self, exp):
        rec = RM.base_record("report")
        rec["exp"] = exp
        return RM.fail(rec, "no report writer on this fake cluster", error_kind="other")

    def tick(self):
        out = C.tick(slurm_sh=self, log=self.log, clock=self.clock, resources=RESOURCES)
        IDB._DIAG_CACHE.clear()
        return out

    def state(self):
        return C.load_state(PATHS, NAME)


def _ctx(slurm=None):
    return {"repo": str(LAB), "log": None, "db": _DB(),
            "actor_of": lambda req: (req.headers.get("x-test-user") or "") if req is not None else "",
            "is_admin": lambda a: a == ADMIN,
            "can_use_cluster": lambda a: a in (ADMIN, MEMBER),
            "slurm_sh": slurm if slurm is not None else _Recorder(),
            "log_action": lambda action, res: LOGGED.append((action, res))}


class _Recorder(object):
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        raise AssertionError("slurm_sh was called: %r" % (a,))


class _DB(object):
    @staticmethod
    def available():
        return True


LOGGED = []


def _client(slurm=None):
    IDB._CTX.clear()
    app = FastAPI()
    IDB.mount(app, _ctx(slurm))
    return TestClient(app)


def _as(who):
    return {"x-test-user": who} if who else {}


def _cfg():
    return json.loads(pathlib.Path(RS._CFG_FILE).read_text())["campaigns"][NAME]


def _ledger_events(event):
    p = PATHS.ledger
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip() and
            json.loads(x).get("event") == event] if p.exists() else []


def _stable(obj):
    """A diagnoses answer without the fields that change per call."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("id", "created_utc", "snapshot_utc",
                                                                  "snapshot_file")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(obj), sort_keys=True)


class Base(unittest.TestCase):
    world_kw = {}
    ticks = 1

    def setUp(self):
        self.w = World(**self.world_kw)
        for _ in range(self.ticks):
            self.w.tick()
        self.c = _client()

    def get(self, path, **kw):
        r = self.c.get(path, **kw)
        self.assertLess(r.status_code, 500, r.text[:500])
        return r.json()


# ------------------------------------------------------------ the ticker's files
class TestTickerFiles(Base):
    """After one real tick on a finished pilot_v1: REPORT, DIAGNOSE, L1 proposed (GATE)."""

    def test_the_ticker_wrote_what_the_page_reads(self):
        st = self.w.state()
        self.assertEqual((st["phase"], st["exp"]), ("GATE", "pilot_v1"))
        for p in (PATHS.latest(NAME), PATHS.diagnoses(NAME), PATHS.status, PATHS.mirror("pilot_v1"),
                  PATHS.frozen("pilot_v1")):
            self.assertTrue(p.is_file(), p)
        self.assertTrue(IDB._history_files("pilot_v1"), "a snapshot history file in the ticker's format")

    def test_campaign_comes_from_the_ticker_state(self):
        d = self.get("/api/inc/campaign")
        self.assertTrue(d["available"])
        c = d["campaigns"][0]
        self.assertEqual((c["name"], c["phase"], c["exp"], c["current_source"]),
                         (NAME, "GATE", "pilot_v1", "ticker state"))
        self.assertEqual(c["item"]["lever"], "L1")
        self.assertEqual(c["exps"], ["b0_v1", "pilot_v1"])
        self.assertIsNotNone(c["last_tick_utc"])
        self.assertIsNotNone(d["heartbeat"]["utc"], "campaign_status.json is read")
        self.assertIsNone(d["tree_mismatch"])
        self.assertIn({"id": "D4", "name": "decision_slot_ready"}, d["goal_options"])
        self.assertFalse(d["replay"]["passed"], "no replay result is recorded in the temp tree")

    def test_snapshot_of_a_frozen_experiment_keeps_the_display_table(self):
        d = self.get("/api/inc/pilot_v1/snapshot")
        self.assertTrue(d["available"], d.get("reason"))
        self.assertIn(d["kind"], ("latest", "history"))
        f = d["display_only"]
        self.assertEqual(f["label"], IDB.DISPLAY_ONLY_LABEL)
        self.assertIn("test", f["rows"][0]["exams"], "the page shows every exam of the final table")
        rest = {k: v for k, v in d.items() if k != "display_only"}
        self.assertFalse(re.search(r'"(%s)"\s*:' % "|".join(NON_DEV), json.dumps(rest)))
        # with the latest payload and every history file gone, frozen.json (decision only)
        # is what is left: the record is still usable and the table is gone, not faked
        PATHS.latest(NAME).unlink()
        for p in IDB._history_files("pilot_v1"):
            p.unlink()
        d = self.get("/api/inc/pilot_v1/snapshot")
        self.assertTrue(d["available"], d.get("reason"))
        self.assertEqual(d["kind"], "frozen")
        self.assertIsNone(d["display_only"])

    def test_grid_matches_the_report(self):
        d = self.get("/api/inc/pilot_v1/snapshot")
        g = d["grid"]
        self.assertEqual(g["chains"], ["full", "freeze", "lora"])
        self.assertEqual(len(g["rows"]), 7)
        self.assertEqual(g["source"], "ledger")
        rep = json.loads((FIX / "pilot_v1" / "report.json").read_text())
        for ch in g["chains"]:
            self.assertEqual(g["agreement"][ch]["agree"], rep["agreement"][ch]["agree"], ch)
            self.assertEqual(g["agreement"][ch]["compared"], rep["agreement"][ch]["compared"], ch)
        bswap = [r for r in g["rows"] if r["step"] == "Bswap"][0]
        self.assertFalse(bswap["clean"])
        cell = g["rows"][0]["chains"]["full"]
        line = cell["cite"]["line"]
        self.assertEqual(cell["cite"]["ledger_url"], "/api/inc/pilot_v1/ledger?line=%d" % line)
        self.assertEqual(FIXTURE_LEDGER[line - 1]["decision"]["verdict"], cell["verdict"])
        self.assertEqual(d["ledger"]["lines_held"], N_LEDGER)
        self.assertTrue(d["ledger"]["verified"])
        self.assertIn("ticker's copy", d["ledger"]["source"])
        self.assertEqual(d["evidence"]["source"], "ticker")

    def test_diagnoses_are_the_ticker_record(self):
        d = self.get("/api/inc/pilot_v1/diagnoses")
        self.assertTrue(d["available"], d.get("reason"))
        self.assertEqual(d["source"], "ticker")
        self.assertIn("campaigns/%s/diagnoses.json" % NAME, d["recorded_in"])
        rec = json.loads(PATHS.diagnoses(NAME).read_text())
        self.assertEqual(sorted(x["id"] for x in d["diagnoses"]), sorted(x["id"] for x in rec["diagnoses"]))
        by = {x["id"]: x for x in d["diagnoses"]}
        self.assertTrue(by["D1"]["fired"])
        self.assertTrue(d["diagnoses"][0]["fired"], "fired diagnoses come first")
        led = [c for c in by["D1"]["cites"] if c.get("line")]
        self.assertTrue(led and all(c["ledger_url"].startswith("/api/inc/pilot_v1/ledger?line=") for c in led))
        argvs = [" ".join(p["argv"]) for p in d["computed"]["proposals"]]
        self.assertTrue(any(a.endswith("inc.pilot build --exp pilot_v2 --replay-mode full") for a in argvs),
                        argvs)
        self.assertNotIn("_ev", d)

    def test_diagnoses_ignore_the_display_only_table(self):
        """Test blindness of this surface: perturb every value the display-only
        part holds (test and every other exam) wherever it is cached, and the
        diagnoses and their preview do not move."""
        base = _stable(self.get("/api/inc/pilot_v1/diagnoses"))

        def bump(x, under=False):
            if isinstance(x, dict):
                return {k: bump(v, under or k == "display_only") for k, v in x.items()}
            if isinstance(x, list):
                return [bump(v, under) for v in x]
            return x + 0.123 if under and isinstance(x, float) else x
        before = json.dumps(self.get("/api/inc/pilot_v1/snapshot")["display_only"]["rows"])
        n = 0
        for p in [PATHS.latest(NAME)] + IDB._history_files("pilot_v1"):
            text = p.read_text()
            new = json.dumps(bump(json.loads(text)))
            n += new != json.dumps(json.loads(text))
            p.write_text(new)
        self.assertGreaterEqual(n, 2, "every cached copy of the display-only table was perturbed")
        IDB._DIAG_CACHE.clear()
        IDB._FILE_CACHE.clear()
        self.assertEqual(_stable(self.get("/api/inc/pilot_v1/diagnoses")), base)
        after = self.get("/api/inc/pilot_v1/snapshot")["display_only"]
        self.assertNotEqual(json.dumps(after["rows"]), before)

    def test_ledger_window_and_tail(self):
        d = self.get("/api/inc/pilot_v1/ledger?line=25&context=2")
        self.assertTrue(d["available"])
        self.assertEqual([r["line"] for r in d["rows"]], [23, 24, 25, 26, 27])
        self.assertEqual(d["rows"][2]["entry"]["id"], FIXTURE_LEDGER[24]["id"])
        self.assertEqual(d["missing"], [])
        t = self.get("/api/inc/pilot_v1/ledger?tail=3")
        self.assertEqual([r["line"] for r in t["rows"]], [N_LEDGER - 2, N_LEDGER - 1, N_LEDGER])

    def test_bad_names_are_refused(self):
        for bad in ("a.b", "step1", "_campaign"):
            r = self.c.get("/api/inc/%s/snapshot" % bad)
            self.assertEqual(r.status_code, 400, bad)
        self.assertFalse(self.get("/api/inc/nothing_here/snapshot")["available"])
        self.assertFalse(self.get("/api/inc/nothing_here/diagnoses")["available"])

    def test_lineage_nodes_and_proposals(self):
        d = self.get("/api/inc/lineage?campaign=%s" % NAME)
        names = {n["exp"]: n for n in d["nodes"]}
        self.assertEqual(names["pilot_v1"]["builder"], "inc.pilot build")
        self.assertEqual(d["edges"], [], "a proposal is not an edge until it runs")
        p = self.get("/api/inc/proposals")
        self.assertEqual((p["campaign"], p["exp"]), (NAME, "pilot_v1"))
        self.assertEqual(p["in_flight"]["lever"], "L1")
        self.assertIn("X1", [c.get("lever") for c in p["cards"]], "the ticker's R4 cards")
        x1 = [c for c in p["cards"] if c.get("lever") == "X1"][0]
        self.assertTrue(x1.get("required_change"), "completed from the lever menu")
        self.assertTrue(x1["source"].startswith("ticker"))

    def test_track_record_and_replay(self):
        self.assertFalse(self.get("/api/inc/track")["available"])
        with open(str(PATHS.track_events), "w") as fh:
            fh.write(json.dumps({"kind": "outcome", "lever": "L1", "proposed_by": M.AUTOPILOT_ACTOR,
                                 "verdict": "better", "correct": True, "child_exp": "pilot_v2"}) + "\n")
        self.assertEqual(self.get("/api/inc/track")["levers"]["L1"]["correct"], 1)
        d = self.get("/api/inc/replay")
        self.assertIn("no replay result", d["replay"]["reason"])
        name = DG.prospective_name("pilot_v1", DG.rules_version())
        pro = [x for x in d["prospective"] if x["file"] == name]
        self.assertTrue(pro, "the ticker's prospective D4 record on pilot_v1, named for the rules version")
        self.assertIsNone(pro[0]["compare"])
        self.assertTrue(pro[0]["current_rules"])
        self.assertFalse(pro[0]["superseded"])
        self.assertEqual((pro[0]["rules_version"], d["rules_version"]), (DG.rules_version(), DG.rules_version()))
        old = PATHS.replay / "prospective_d4_pilot_v1.json"
        old.write_text(json.dumps({"exp": "pilot_v1", "ready": False, "outcome": "decision_slot_ready",
                                   "replay_mode": None, "recipes": None, "decided_utc": "2026-09-01T00:00:00Z"}))
        try:
            hist = [x for x in self.get("/api/inc/replay")["prospective"] if x["file"] == old.name]
            self.assertTrue(hist and hist[0]["superseded"] and not hist[0]["current_rules"])
            self.assertIsNone(hist[0]["compare"], "a superseded record is never compared with a real loop")
        finally:
            old.unlink()


class TestAfterApproval(Base):
    """A person approves L1; the next tick executes it and the ticker moves to pilot_v2."""

    def setUp(self):
        super().setUp()
        self.w.tick()                                   # files the approval (no ssh)
        items = [i for i in AP.state("weed", root=str(LAB)).values() if i.get("action") == "inc_build_pilot"]
        self.assertEqual(len(items), 1)
        self.aid = items[0]["id"]

    def test_an_approval_card_is_warn(self):
        v = IDB.verdict()
        self.assertEqual(v["level"], "warn", v["reason"])
        self.assertIn("approval", v["reason"])
        p = self.get("/api/inc/proposals")
        self.assertEqual([i["lever"] for i in p["filed"]["deterministic"]], ["L1"])
        self.assertEqual(p["n_pending"], 1)

    def test_current_experiment_follows_the_ticker_to_the_child(self):
        AP.decide("weed", self.aid, "approve", OWNER, "go", time.time(), root=str(LAB))
        self.w.tick()
        st = self.w.state()
        self.assertEqual((st["exp"], st["phase"]), ("pilot_v2", "RUN"))
        d = self.get("/api/inc/campaign")
        c = d["campaigns"][0]
        self.assertEqual((c["exp"], c["current_exp"], c["current_source"], c["phase"]),
                         ("pilot_v2", "pilot_v2", "ticker state", "RUN"))
        self.assertEqual(c["building"]["exp"], "pilot_v2")
        self.assertEqual(self.get("/api/inc/proposals")["exp"], "pilot_v2")
        h = [r for r in IDB.verdict()["campaigns"] if r["campaign"] == NAME][0]
        self.assertEqual(h["current_exp"], "pilot_v2")
        lin = self.get("/api/inc/lineage?campaign=%s" % NAME)
        e = [x for x in lin["edges"] if x["child"] == "pilot_v2"][0]
        self.assertEqual((e["parent"], e["lever"], e["approval_ids"]), ("pilot_v1", "L1", [self.aid]))
        self.assertIn("D1", e["trigger"])


class TestRunningCampaign(Base):
    """A campaign whose experiment runs: it observes every tick and decides nothing."""
    world_kw = {"current": "run_a", "exps": ("pilot_v1", "run_a"), "running": ("run_a",)}

    def test_running_phase_and_a_labelled_preview(self):
        st = self.w.state()
        self.assertEqual((st["phase"], st["exp"]), ("RUN", "run_a"))
        self.assertEqual(IDB.verdict()["level"], "ok", IDB.verdict()["reason"])
        d = self.get("/api/inc/run_a/diagnoses?campaign=%s" % NAME)
        self.assertTrue(d["available"], d.get("reason"))
        self.assertEqual(d["source"], "lab")
        self.assertIn("preview", d["note"])
        self.assertIn("never alarmed on", d["note"])

    def test_the_preview_uses_the_ticker_context_for_this_campaign_only(self):
        EX._log(EX.Context(), {"ts": "2026-09-27T01:00:00Z", "campaign": "other", "action": "inc_build_pilot",
                               "status": "executed", "lever": "L1", "parent_exp": "run_a",
                               "child_exp": "run_b", "params": {"exp": "run_b"}})
        EX._log(EX.Context(), {"ts": "2026-09-27T01:00:00Z", "campaign": NAME, "action": "inc_build_pilot",
                               "status": "executed", "lever": "L1", "parent_exp": "pilot_v1",
                               "child_exp": "pilot_v9", "params": {"exp": "pilot_v9"}})
        ev, info = IDB._evidence("run_a", NAME)
        self.assertEqual(info["source"], "ticker")
        ctx = ev.json(E.CONTEXT)
        self.assertEqual([(r["child_exp"], r["status"]) for r in ctx["lineage"]], [("pilot_v9", "executed")])
        for k in ("refusals", "outcomes", "budget"):
            self.assertIn(k, ctx, k)
        self.assertIn("projected_su", ctx["budget"])
        run = IDB._run_for(IDB._pick(IDB._cfg_read()[0], NAME), self.w.state())
        self.assertEqual(ctx["lineage"], run.lineage())

    def test_a_rewritten_ledger_replaces_the_lab_copy(self):
        self.assertEqual(self.get("/api/inc/run_a/ledger?line=1")["rows"][0]["entry"]["id"],
                         FIXTURE_LEDGER[0]["id"].replace("pilot_v1", "run_a"))
        p = INC / "run_a" / "ledger.jsonl"
        lines = p.read_text().splitlines(True)
        first = json.loads(lines[0])
        first["id"] = "rewritten-line-1"
        p.write_text(json.dumps(first) + "\n" + "".join(lines[1:]))
        self.w.tick()
        self.assertTrue(any(r.get("exp") == "run_a" for r in _ledger_events("ledger_rewritten")),
                        "the ticker saw the prefix_mismatch")
        d = self.get("/api/inc/run_a/ledger?line=1")
        self.assertEqual(d["rows"][0]["entry"]["id"], "rewritten-line-1")
        self.assertTrue(d["verified"], d["notes"])

    def test_a_stop_loss_is_crit_and_a_page_resume_ends_it_for_the_ticker(self):
        self.w.drift = True
        self.w.tick()
        st = self.w.state()
        self.assertTrue(st["paused"] and "D7" in st["paused"]["reason"], st.get("paused"))
        self.assertIn("D7", _cfg()["paused_reason"])
        v = IDB.verdict()
        self.assertEqual(v["level"], "crit")
        self.assertIn("D7", v["reason"])
        self.w.drift = False
        time.sleep(1.1)                     # the person resumes after the stop-loss
        r = self.c.post("/api/inc/campaign", json={"campaign": NAME, "resume": True}, headers=_as(ADMIN))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["changes"]["enabled"], [False, True])
        cfg = C.campaign_config(_cfg(), NAME)
        run = IDB._run_for(cfg, self.w.state())
        self.assertIsNone(run._paused_reason(), "the ticker's own reading: no longer paused")
        self.assertEqual(self.w.state()["fails"], 0)
        v = IDB.verdict()
        self.assertNotEqual(v["level"], "crit", "the D7 of before the resume was answered by it: %s"
                            % v["reason"])
        time.sleep(1.1)                     # the next observation, a second later
        self.w.tick()
        self.assertEqual(self.w.state()["health"], [], "the next observation: no drift")
        self.assertEqual(IDB.verdict()["level"], "ok", IDB.verdict()["reason"])
        self.assertTrue(any(r["decided_by"] == "human:" + ADMIN for r in _ledger_events("configured")),
                        "the resume is in the campaign ledger, in the person's name")

    def test_a_pause_held_only_in_the_state_is_crit(self):
        st = self.w.state()
        st["paused"] = {"reason": "stop-loss: 2 consecutive failed campaign steps",
                        "utc": C._utc(time.time() + 60), "by": C.AUTO}
        C.save_state(PATHS, NAME, st)
        self.assertTrue(_cfg()["enabled"])
        v = IDB.verdict()
        self.assertEqual(v["level"], "crit")
        self.assertIn("2 consecutive failed", v["reason"])
        d = self.get("/api/inc/campaign")["campaigns"][0]
        self.assertIn("2 consecutive failed", d["paused_reason"])
        self.assertEqual(self.c.post("/api/inc/campaign", json={"campaign": NAME, "resume": True},
                                     headers=_as(ADMIN)).status_code, 200)
        self.assertIsNone(self.w.state()["paused"])
        self.assertNotEqual(IDB.verdict()["level"], "crit")

    def test_a_stale_tick_is_crit(self):
        self.assertEqual(IDB.verdict()["level"], "ok")
        v = IDB.verdict(now=time.time() + 4 * 3600)
        self.assertEqual(v["level"], "crit")
        self.assertIn("last ticked", v["reason"])

    def test_a_recorded_crit_diagnosis_on_the_current_experiment_is_crit(self):
        C._write_json(PATHS.diagnoses(NAME), {"utc": C._utc(time.time() + 60), "exp": "run_a", "diagnoses": [
            {"id": "DREF", "name": "builder_refusal", "fired": True, "severity": "crit", "summary": "unmapped",
             "cites": [{"artifact": "campaign/context.json", "pointer": "/refusals/0", "value": "x"}],
             "levers": [], "exp": "run_a"}]})
        v = IDB.verdict()
        self.assertEqual(v["level"], "crit")
        self.assertIn("DREF", v["reason"])
        C._write_json(PATHS.diagnoses(NAME), {"utc": C._utc(time.time() + 60), "exp": "pilot_v1",
                                              "diagnoses": json.loads(PATHS.diagnoses(NAME).read_text())
                                              ["diagnoses"]})
        self.assertEqual(IDB.verdict()["level"], "ok", "a diagnosis of another experiment is history")

    def test_health_never_computes_a_diagnosis(self):
        real = IDB._compute_diagnoses
        IDB._compute_diagnoses = lambda *a, **k: (_ for _ in ()).throw(AssertionError("computed"))
        try:
            self.assertEqual(IDB.verdict()["level"], "ok")
        finally:
            IDB._compute_diagnoses = real


class TestHealthMisc(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.c = _client()

    def test_enabled_but_never_ticked(self):
        v = IDB.verdict()
        self.assertEqual(v["level"], "warn", v["reason"])
        self.assertIn("not ticked yet", v["reason"])
        self.assertEqual(IDB.verdict(now=time.time() + 3 * 3600)["level"], "crit")
        c = self.c.get("/api/inc/campaign").json()["campaigns"][0]
        self.assertEqual((c["phase"], c["exp"]), ("RUN", "pilot_v1"), "configure() made the state")

    def test_no_campaign_is_ok(self):
        pathlib.Path(RS._CFG_FILE).write_text(json.dumps({"domains": {}}))
        self.assertEqual(IDB.verdict()["level"], "ok")
        d = self.c.get("/api/inc/campaign").json()
        self.assertEqual(d["campaigns"], [])
        self.assertIn("no INC campaign", d["reason"])

    def test_unreadable_config_is_crit(self):
        pathlib.Path(RS._CFG_FILE).write_text("{not json")
        v = IDB.verdict()
        self.assertEqual(v["level"], "crit")
        self.assertIn("cannot be read", v["reason"])

    def test_autonomy_without_a_replay_pass_warns(self):
        C.configure(NAME, OWNER, autonomy="envelope")
        self.w.tick()
        v = IDB.verdict()
        self.assertEqual(v["level"], "warn")
        self.assertIn("replay pass", v["reason"])

    def test_a_tree_the_ticker_does_not_write_is_crit(self):
        other = TMP / "other_tree"
        other.mkdir(exist_ok=True)
        IDB._CTX["repo"] = str(other)
        try:
            v = IDB.verdict()
            self.assertEqual(v["level"], "crit")
            self.assertIn("the ticker writes under", v["reason"])
        finally:
            IDB._CTX["repo"] = str(LAB)

    def test_the_unauthenticated_alarm_names_no_person(self):
        self.w.tick()
        C.pause(NAME, "looking at D3 with owner@example.org", "human:" + ADMIN)
        r = self.c.get("/api/health/inc")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["level"], "crit")
        self.assertNotIn("@", json.dumps(r.json()))
        self.assertIn("<person>", r.json()["reason"])


class TestNoSshFromReads(Base):
    def test_no_get_reaches_the_cluster(self):
        trip = _Recorder()
        self.c = _client(trip)
        real = EX._call_slurm

        def boom(*a, **k):
            raise AssertionError("the executor tried to reach the cluster from a read")
        EX._call_slurm = boom
        try:
            for path in ("/api/inc/campaign", "/api/inc/lineage", "/api/inc/pilot_v1/snapshot",
                         "/api/inc/pilot_v1/diagnoses", "/api/inc/pilot_v1/ledger?line=3",
                         "/api/inc/proposals", "/api/inc/track", "/api/inc/replay", "/api/health/inc",
                         "/inc", "/api/inc/b0_v1/snapshot", "/api/inc/b0_v1/diagnoses"):
                r = self.c.get(path)
                self.assertNotIn(r.status_code, (500,), (path, r.text[:300]))
        finally:
            EX._call_slurm = real
        self.assertEqual(trip.calls, [])

    def test_every_route_is_read_only_except_the_two_posts(self):
        import ast
        src = pathlib.Path(IDB.__file__).read_text()
        writes = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    f = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(f, ast.Attribute) and f.attr in ("post", "put", "patch", "delete"):
                        writes.append(node.name)
        self.assertEqual(sorted(writes), ["api_inc_action", "api_inc_campaign_set"])


# ------------------------------------------------------------ writes
class TestCampaignPost(Base):
    def post(self, body, who=ADMIN):
        return self.c.post("/api/inc/campaign", json=body, headers=_as(who))

    def test_unidentified_caller_is_refused(self):
        r = self.post({"campaign": NAME, "pause": "x"}, who="")
        self.assertEqual(r.status_code, 401)
        self.assertIsNone(_cfg().get("paused_reason"))

    def test_no_identity_hook_refuses(self):
        IDB._CTX.pop("actor_of")
        r = self.post({"campaign": NAME, "pause": "x"})
        self.assertEqual(r.status_code, 403)
        self.assertIn("no identity hook", r.json()["reason"])

    def test_a_member_is_not_an_administrator(self):
        r = self.post({"campaign": NAME, "enabled": False}, who=MEMBER)
        self.assertEqual(r.status_code, 403)
        self.assertTrue(_cfg()["enabled"])

    def test_pause_goes_to_config_state_and_campaign_ledger(self):
        r = self.post({"campaign": NAME, "pause": "looking at D3"})
        self.assertEqual(r.status_code, 200, r.text)
        why = _cfg()["paused_reason"]
        self.assertIn("human:" + ADMIN, why)
        self.assertIn("looking at D3", why)
        self.assertFalse(_cfg()["enabled"])
        self.assertEqual(self.w.state()["paused"]["by"], "human:" + ADMIN)
        self.assertTrue(any(e["decided_by"] == "human:" + ADMIN for e in _ledger_events("paused")))
        self.assertEqual(IDB.verdict()["level"], "crit")
        self.assertEqual(self.post({"campaign": NAME, "resume": True}).status_code, 200)
        self.assertIsNone(_cfg()["paused_reason"])
        self.assertIsNone(self.w.state()["paused"])
        self.assertTrue(any(a == "inc_campaign_config" for a, _ in LOGGED))
        self.assertEqual(self.post({"campaign": NAME, "pause": "x", "resume": True}).status_code, 400)

    def test_a_pause_needs_a_reason(self):
        self.assertEqual(self.post({"campaign": NAME, "pause": "  "}).status_code, 400)

    def test_goals_are_the_ones_the_ticker_reads(self):
        g = {"kind": "diagnosis", "id": "D4", "name": "decision_slot_ready"}
        r = self.post({"campaign": NAME, "goal": g})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(C.check_goal(_cfg()["goal"]), g)
        self.assertTrue(any(e.get("goal") == g and e["decided_by"] == "human:" + ADMIN
                            for e in _ledger_events("goal_set")))
        r = self.post({"campaign": NAME, "goal": {"kind": "exp_done", "exp": "pilot_v2"}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(_cfg()["goal"], {"kind": "exp_done", "exp": "pilot_v2"})
        for bad in ("agreement >= 5/7 on dev", {"metric": "twelve", "exam": "dev"},
                    {"kind": "diagnosis", "id": "D99", "name": "x"},
                    {"kind": "exp_done", "exp": "test"}, {"kind": "exp_done", "exp": "ood23_run"}):
            self.assertEqual(self.post({"campaign": NAME, "goal": bad}).status_code, 400, bad)
        self.assertEqual(self.post({"campaign": NAME, "goal": None}).status_code, 200)
        self.assertIsNone(_cfg()["goal"])

    def test_envelope_is_bounded_by_the_domain(self):
        self.assertEqual(self.post({"campaign": NAME, "envelope_su": 10 ** 6}).status_code, 400)
        self.assertEqual(self.post({"campaign": NAME, "envelope_su": -1}).status_code, 400)
        self.assertEqual(self.post({"campaign": NAME, "envelope_su": "300"}).status_code, 400)
        r = self.post({"campaign": NAME, "envelope_su": 250, "daily_cap_su": 50})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((_cfg()["envelope_su"], _cfg()["daily_cap_su"]), (250.0, 50.0))

    def test_autonomy_is_granted_in_a_person_s_name(self):
        r = self.post({"campaign": NAME, "autonomy": "envelope"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((_cfg()["autonomy"], _cfg()["autonomy_granted_by"]), ("envelope", "human:" + ADMIN))
        self.assertIn("replay pass", r.json()["warning"])
        self.assertEqual(self.post({"campaign": NAME, "autonomy": "off"}).status_code, 200)
        self.assertEqual(_cfg()["autonomy"], "off")
        self.assertEqual(self.post({"campaign": NAME, "autonomy": "always"}).status_code, 400)
        IDB._CTX["is_admin"] = lambda a: a in (ADMIN, "admin")
        r = self.post({"campaign": NAME, "autonomy": "envelope"}, who="admin")
        self.assertEqual(r.status_code, 400, "the shared operator login is not a person")
        self.assertIn("person", r.json()["reason"])

    def test_create_and_unknown(self):
        self.assertEqual(self.post({"campaign": "c2", "enabled": True}).status_code, 404)
        self.assertEqual(self.post({"campaign": "c2", "create": True}).status_code, 400)
        r = self.post({"campaign": "c2", "create": True, "exp": "pilot_v2"})
        self.assertEqual(r.status_code, 200, r.text)
        c2 = json.loads(pathlib.Path(RS._CFG_FILE).read_text())["campaigns"]["c2"]
        self.assertEqual((c2["enabled"], c2["current_exp"], c2["exps"]), (False, "pilot_v2", ["pilot_v2"]))
        self.assertEqual(C.load_state(PATHS, "c2")["exp"], "pilot_v2")
        self.assertEqual(self.post({"campaign": "../x"}).status_code, 400)

    def test_an_unreadable_config_is_not_overwritten(self):
        pathlib.Path(RS._CFG_FILE).write_text("{truncated")
        r = self.post({"campaign": NAME, "pause": "x"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(pathlib.Path(RS._CFG_FILE).read_text(), "{truncated")

    def test_the_write_waits_for_the_scheduler_lock(self):
        got = RS._LOCK.acquire(timeout=5)
        self.assertTrue(got)
        res = {}

        def post():
            res["r"] = self.post({"campaign": NAME, "pause": "while the scheduler holds its lock"})
        th = threading.Thread(target=post)
        try:
            th.start()
            time.sleep(0.8)
            self.assertTrue(th.is_alive(), "the write did not wait for round_scheduler._LOCK")
            self.assertIsNone(_cfg().get("paused_reason"))
            # the scheduler's own write under its lock is kept by the page's write
            c = json.loads(pathlib.Path(RS._CFG_FILE).read_text())
            c["domains"]["weed"] = {"round": {"job_id": "123"}}
            RS._save_cfg(c)
        finally:
            RS._LOCK.release()
        th.join(30)
        self.assertEqual(res["r"].status_code, 200, res["r"].text)
        c = json.loads(pathlib.Path(RS._CFG_FILE).read_text())
        self.assertEqual(c["domains"]["weed"]["round"]["job_id"], "123")
        self.assertIn("scheduler holds", c["campaigns"][NAME]["paused_reason"])


class TestActionPost(Base):
    def setUp(self):
        super().setUp()
        self.calls = []
        self._real = (EX.submit, EX.execute_approved)

        def fake_submit(request, actor=None, campaign=None, ctx=None):
            self.calls.append(("submit", request, actor, campaign, ctx))
            return {"status": "executed", "ok": True, "job_ids": ["4242"], "reasons": []}

        def fake_exec(item_id, campaign=None, ctx=None, invoked_by=None, quiet_repeat=False, note=None):
            self.calls.append(("approved", item_id, invoked_by, campaign, ctx, note))
            return {"status": "refused", "ok": False, "action": "inc_build_pilot",
                    "reasons": ["approval %s is pending, not approved" % item_id]}
        EX.submit, EX.execute_approved = fake_submit, fake_exec

    def tearDown(self):
        EX.submit, EX.execute_approved = self._real

    def post(self, verb, body, who=MEMBER):
        return self.c.post("/api/inc/action/%s" % verb, json=body, headers=_as(who))

    def test_the_executor_is_called_as_the_person(self):
        r = self.post("advance", {"campaign": NAME, "params": {"exp": "pilot_v1"}, "actor": "human:evil@x"})
        self.assertEqual(r.status_code, 200, r.text)
        kind, req, actor, camp, ctx = self.calls[0]
        self.assertEqual(actor, "human:" + MEMBER, "the body cannot name the actor")
        self.assertEqual(req["policy_action"], "inc_advance")
        self.assertEqual(req["params"], {"exp": "pilot_v1"})
        self.assertEqual(camp["name"], NAME)
        self.assertIsNone(camp["paused_reason"])
        self.assertEqual(pathlib.Path(ctx.lab_repo).resolve(), LAB.resolve())
        self.assertTrue(callable(ctx.slurm_sh), "a write gets the cluster hook")
        self.assertEqual(r.json()["job_ids"], ["4242"])
        self.assertTrue(any(a == "inc_advance" and "Submitted batch job 4242" in res["msg"] for a, res in LOGGED))

    def test_policy_action_ids_and_lever_fields_pass_through(self):
        argv = ["inc.pilot", "build", "--exp", "pilot_v2", "--replay-mode", "full"]
        r = self.post("inc_build_pilot", {"campaign": NAME, "argv": argv, "lever": "L1", "trigger": ["D1"],
                                          "est_gpu_hours": 38.5, "parent_exp": "pilot_v1",
                                          "proposed_by": "tier2:someone"})
        self.assertEqual(r.status_code, 200)
        req = self.calls[0][1]
        self.assertEqual((req["policy_action"], req["argv"], req["lever"], req["est_gpu_hours"]),
                         ("inc_build_pilot", argv, "L1", 38.5))
        self.assertNotIn("proposed_by", req, "a person cannot file under someone else's name")

    def test_execute_approved_runs_as_the_person(self):
        r = self.post("execute-approved", {"campaign": NAME, "approval_id": "ap-1-2"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.calls[0][:3], ("approved", "ap-1-2", "human:" + MEMBER))
        self.assertEqual(self.post("execute-approved", {"campaign": NAME}).status_code, 400)

    def test_access_is_required(self):
        self.assertEqual(self.post("advance", {"params": {"exp": "pilot_v1"}}, who=STRANGER).status_code, 403)
        self.assertEqual(self.post("advance", {"params": {"exp": "pilot_v1"}}, who="").status_code, 401)
        self.assertEqual(self.calls, [])

    def test_unknown_verb_and_campaign(self):
        self.assertEqual(self.post("rm-rf", {}).status_code, 400)
        self.assertEqual(self.post("advance", {"campaign": "nope"}).status_code, 404)
        self.assertEqual(self.calls, [])

    def test_a_real_loop_waits_for_the_prospective_record(self):
        ver = DG.rules_version()
        rec = PATHS.replay / DG.prospective_name("pilot_v1", ver)
        self.assertTrue(rec.is_file(), "the ticker froze D4 on pilot_v1 under the current rules version")
        body = {"campaign": NAME, "params": {"exp": "loop_v1", "replay_mode": "sample", "recipes": "full"}}
        r = self.post("build-realloop", body)
        self.assertEqual(r.status_code, 409, "D4 on pilot_v1 is frozen as not ready: no real loop from it")
        self.assertIn("as not ready", r.json()["reasons"][0])
        self.assertEqual(self.calls, [])
        frozen = json.loads(rec.read_text())
        rec.write_text(json.dumps(dict(frozen, ready=True, outcome="decision_slot_ready", blocked_by=None,
                                       replay_mode="sample", recipes=["full"], gate_flips_mode="negative")))
        r = self.post("build-realloop", body)
        self.assertEqual(r.status_code, 409, "a READY file the ticker did not write (its state's sha256 differs)")
        self.assertIn("not the one the ticker wrote", r.json()["reasons"][0])
        self._ticker_wrote(rec)
        self.assertEqual(self.post("build-realloop", body).status_code, 200, "a READY record of the current rules")
        self.assertEqual(len(self.calls), 1)
        other = dict(body, params=dict(body["params"], recipes="freeze"))
        r = self.post("build-realloop", other)
        self.assertEqual(r.status_code, 409, "a build that does not follow the frozen decision")
        self.assertIn("does not follow D4's decision on pilot_v1", r.json()["reasons"][0])
        self.assertIn('recipes ["freeze"], the record\'s ["full"]', r.json()["reasons"][0])
        r = self.post("build-realloop", dict(other, params=dict(body["params"], gate_flips_mode="net")))
        self.assertEqual(r.status_code, 409, "another gate than the record's")
        self.assertIn("gate_flips_mode", r.json()["reasons"][0])
        self.assertEqual(len(self.calls), 1)
        rec.unlink()
        legacy = PATHS.replay / "prospective_d4_pilot_v1.json"
        legacy.write_text(json.dumps(dict(frozen, ready=True, rules_version=None)))
        r = self.post("build-realloop", body)
        self.assertEqual(r.status_code, 409, "a record of other rules (the unversioned name) does not count")
        self.assertIn(rec.name, r.json()["reasons"][0])
        self.assertIn("prospective_d4_pilot_v1.json", r.json()["reasons"][0])
        self.assertEqual(len(self.calls), 1)
        r = self.post("inc_build_realloop", dict(body, acknowledge_non_prospective="R4b is already void"))
        self.assertEqual(r.status_code, 200)
        self.assertIn("acknowledged non-prospective by human:%s: R4b is already void" % MEMBER,
                      self.calls[-1][1]["reason"])

    def _ticker_wrote(self, rec):
        """The campaign state names `rec` as the record the ticker wrote (its sha256)."""
        st = C.load_state(PATHS, NAME)
        st.setdefault("prospective", {})["pilot_v1"] = dict(
            (st.get("prospective") or {}).get("pilot_v1") or {}, path=str(rec), rules_version=DG.rules_version(),
            sha256=hashlib.sha256(rec.read_bytes()).hexdigest(), ready=True)
        C.save_state(PATHS, NAME, st)

    def test_execute_approved_of_a_real_loop_is_guarded_too(self):
        params = {"exp": "loop_v1", "replay_mode": "full", "recipes": "full"}
        r = AP.propose("weed", "inc_build_realloop", params, "R3", "tier2:qwen3", "a brain's L2", time.time(),
                       root=str(LAB), context={"campaign": NAME, "lever": "L2", "parent_exp": "pilot_v1",
                                               "proposal_id": "p-loop"})
        aid = r["item"]["id"]
        body = {"campaign": NAME, "approval_id": aid}
        r = self.post("execute-approved", body)
        self.assertEqual(r.status_code, 409, "D4 on pilot_v1 is frozen as not ready: the approval does not run")
        self.assertIn("as not ready", r.json()["reasons"][0])
        self.assertEqual(self.calls, [])
        rec = PATHS.replay / DG.prospective_name("pilot_v1", DG.rules_version())
        rec.write_text(json.dumps(dict(json.loads(rec.read_text()), ready=True, outcome="decision_slot_ready",
                                       blocked_by=None, replay_mode="full", recipes=["freeze"],
                                       gate_flips_mode="negative")))
        self._ticker_wrote(rec)
        r = self.post("execute-approved", body)
        self.assertEqual(r.status_code, 409, "the approved build does not follow the READY record (freeze)")
        self.assertIn("does not follow D4's decision", r.json()["reasons"][0])
        self.assertEqual(self.calls, [])
        r = self.post("execute-approved", dict(body, acknowledge_non_prospective="a person's choice of recipe"))
        self.assertEqual(self.calls[-1][:3], ("approved", aid, "human:" + MEMBER))
        self.assertEqual(self.calls[-1][5], "acknowledged non-prospective by human:%s: a person's choice of recipe"
                         % MEMBER, "the executor logs the acknowledgement with the execution")
        rec.write_text(json.dumps(dict(json.loads(rec.read_text()), recipes=["full"])))
        self._ticker_wrote(rec)
        self.post("execute-approved", body)
        self.assertEqual(len(self.calls), 2, "a build that follows the READY record runs without one")
        self.assertIsNone(self.calls[-1][5])
        pilot = AP.propose("weed", "inc_build_pilot", {"exp": "pilot_v9"}, "R3", "tier2:qwen3", "L1",
                           time.time() + 1, root=str(LAB), context={"campaign": NAME})["item"]["id"]
        self.post("execute-approved", {"campaign": NAME, "approval_id": pilot})
        self.assertEqual(self.calls[-1][1], pilot, "an approval of another action is not a real loop's")


class TestActionEndToEnd(Base):
    """Actions through the real executor with the fake cluster."""

    def test_a_human_advance_runs_through_the_executor(self):
        self.c = _client(self.w)
        r = self.c.post("/api/inc/action/advance", json={"campaign": NAME, "params": {"exp": "pilot_v1"}},
                        headers=_as(MEMBER))
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual((d["status"], d["authorized_as"], d["policy_action"]),
                         ("executed", "human:" + MEMBER, "inc_advance"))
        self.assertIn("remote advance --exp pilot_v1", self.w.calls[-1].replace("'", ""))
        execs = EX.executions(EX.Context())
        self.assertEqual((execs[-1]["actor"], execs[-1]["status"]), ("human:" + MEMBER, "executed"))

    def test_a_manual_snapshot_is_kept_in_the_ticker_format(self):
        self.c = _client(self.w)
        time.sleep(1.1)                     # a record of a later second than the tick's
        r = self.c.post("/api/inc/action/snapshot", json={"campaign": NAME, "params": {"exp": "b0_v1"}},
                        headers=_as(MEMBER))
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertTrue(d["kept_as"].endswith(".manual.json"), d.get("kept_as"))
        self.assertNotIn("payload", d["remote"], "the record is read back from the lab, not returned")
        obj = json.loads((PATHS.snapshots / "b0_v1" / d["kept_as"]).read_text())
        self.assertEqual((obj["by"], obj["record"]["verb"], obj["record"]["experiments"]["b0_v1"]["snapshot"]["verb"]),
                         ("human:" + MEMBER, "campaign-snapshot", "snapshot"))
        s = self.get("/api/inc/b0_v1/snapshot")
        self.assertEqual((s["file"], s["by"]), ("snapshots/b0_v1/" + d["kept_as"], "human:" + MEMBER))

    def test_a_non_email_login_is_refused_by_the_policy(self):
        self.c = _client(self.w)
        IDB._CTX["can_use_cluster"] = lambda a: True
        r = self.c.post("/api/inc/action/advance", json={"campaign": NAME, "params": {"exp": "pilot_v1"}},
                        headers=_as("user:bob"))
        self.assertEqual(r.status_code, 403)
        self.assertIn("not a recognised actor", " ".join(r.json()["reasons"]))


# ------------------------------------------------------------ the page
class TestPage(unittest.TestCase):
    def _script(self):
        page = IDB._PAGE
        return page[page.index("<script>") + 8:page.rindex("</script>")]

    def test_page_is_served_and_english(self):
        c = _client()
        r = c.get("/inc")
        self.assertEqual(r.status_code, 200)
        self.assertIn("INC campaign", r.text)
        self.assertIn('name="viewport"', r.text)
        self.assertFalse(re.search(r"[一-鿿]", r.text), "the product is English only")

    def test_page_reads_every_route_and_wires_approvals(self):
        js = self._script()
        for needle in ('"/api/inc/campaign"', '"/api/inc/lineage"', '"/snapshot"', '"/diagnoses"',
                       "/ledger?", '"/api/inc/proposals"', '"/api/inc/track"', '"/api/inc/replay"',
                       '"/api/inc/action/"', '"/api/brain/" + DOMAIN + "/approvals/"', '"/api/inc/action/cancel"',
                       "data-decide", "a decision needs a reason", "execute-approved",
                       'kind: "diagnosis"', 'kind: "exp_done"', "S.current = c ? c.exp"):
            self.assertIn(needle, js, needle)
        for card in ("Campaign", "Lineage", "Current experiment", "Diagnoses", "Proposals",
                     "Human research cards (R4)", "Track record", "Replay and prospective tests",
                     "display only"):
            self.assertIn(card, IDB._PAGE, card)

    def test_no_direct_scancel_from_the_page(self):
        js = self._script()
        self.assertNotIn("/api/cancel_job", js)
        self.assertNotIn("data-cancel-job", js)

    def test_served_script_has_no_backslash_and_no_inline_handler(self):
        self.assertNotIn(chr(92), self._script())
        self.assertNotIn("onclick=", IDB._PAGE)

    def test_served_script_parses(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("no node on this machine")
        path = TMP / "inc_page.js"
        path.write_text(self._script())
        r = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[:500])


# ------------------------------------------------------------ the real app
class TestRealDashboard(unittest.TestCase):
    """Mounted in dashboard_server, behind its auth middleware."""

    @classmethod
    def setUpClass(cls):
        cls.w = World()
        cls.w.tick()
        from weed_optimizer_framework.tools import dashboard_server as D
        cls.D = D
        cls.c = TestClient(D.app)

    def auth(self, **extra):
        h = {"Authorization": "Basic " + base64.b64encode(b"tester:test-pass").decode()}
        h.update(extra)
        return h

    def test_routes_are_mounted(self):
        paths = {getattr(r, "path", "") for r in self.D.app.routes}
        for p in ("/inc", "/api/inc/campaign", "/api/inc/lineage", "/api/inc/{exp}/snapshot",
                  "/api/inc/{exp}/diagnoses", "/api/inc/{exp}/ledger", "/api/inc/replay",
                  "/api/inc/action/{verb}", "/api/health/inc"):
            self.assertIn(p, paths, p)

    def test_the_real_mount_reads_the_ticker_tree(self):
        self.assertEqual(IDB._tree_mismatch(), "")
        self.assertIsNone(IDB._lab_repo_arg(), "REPO is model.LAB_REPO: campaign.Paths(None)")
        d = self.c.get("/api/inc/campaign", headers=self.auth()).json()
        self.assertEqual(d["campaigns"][0]["phase"], "GATE")

    def test_writes_and_reads_need_a_login(self):
        self.assertEqual(self.c.post("/api/inc/campaign", json={"campaign": NAME, "pause": "x"}).status_code, 401)
        self.assertEqual(self.c.post("/api/inc/action/advance", json={}).status_code, 401)
        self.assertEqual(self.c.get("/api/inc/campaign").status_code, 401)
        self.assertEqual(self.c.get("/inc").status_code, 401)
        r = self.c.get("/api/inc/campaign", headers=self.auth())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["available"])

    def test_health_is_exempt_like_the_other_alarms(self):
        r = self.c.get("/api/health/inc")
        self.assertIn(r.status_code, (200, 503))
        self.assertIn(r.json()["level"], ("ok", "warn", "crit"))

    def test_the_page_carries_the_site_nav(self):
        r = self.c.get("/inc", headers=self.auth())
        self.assertEqual(r.status_code, 200)
        self.assertIn('id="_appnav"', r.text)

    def test_the_operator_login_maps_to_a_policy_actor_and_x_user_is_ignored(self):
        """Basic-auth operator -> actor 'admin' -> human:admin; an X-User header
        naming someone else changes nothing on the INC routes."""
        calls = []
        real = EX.submit
        EX.submit = lambda request, actor=None, campaign=None, ctx=None: (
            calls.append(actor) or {"status": "filed", "ok": True, "approval_id": "ap-x"})
        try:
            r = self.c.post("/api/inc/action/advance", json={"campaign": NAME, "params": {"exp": "pilot_v1"}},
                            headers=self.auth(**{"X-User": "someone.else@lab.org"}))
        finally:
            EX.submit = real
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(calls, ["human:admin"])
        r = self.c.post("/api/inc/campaign", json={"campaign": NAME, "autonomy": "envelope"},
                        headers=self.auth(**{"X-User": "someone.else@lab.org"}))
        self.assertEqual(r.status_code, 400, "autonomy is not granted under a claimed name")
        self.assertNotEqual(_cfg().get("autonomy_granted_by"), "human:someone.else@lab.org")

    def test_safe_prefixes_cover_inc_jobs(self):
        src = pathlib.Path(self.D.__file__).read_text()
        block = src[src.index("SAFE_PREFIXES = ("):]
        block = block[:block.index(")") + 1]
        self.assertIn('"inc_"', block)

    def _cancel(self, name, can_use=True):
        seen = []
        real = (self.D._slurm, self.D._can_use_cluster, self.D._log_action)

        def slurm(cmd, timeout=15):
            seen.append(cmd)
            if cmd[0] == "squeue":
                return {"ok": True, "stdout": name + "\n", "stderr": ""}
            return {"ok": True, "stdout": "", "stderr": ""}
        logged = []
        self.D._slurm = slurm
        self.D._can_use_cluster = lambda a: can_use
        self.D._log_action = lambda action, res: logged.append((action, res))
        try:
            r = self.c.post("/api/cancel_job/4242", headers=self.auth(**{"X-User": "someone.else@lab.org"}))
        finally:
            self.D._slurm, self.D._can_use_cluster, self.D._log_action = real
        return r, [c for c in seen if c[0] == "scancel"], logged

    def test_inc_runs_and_builds_are_not_cancelled_outside_the_executor(self):
        for name in ("inc_pilot_v1_0007", "inc_build_pilot_v2", "inc_audit_x_0001"):
            r, sc, _l = self._cancel(name)
            self.assertEqual(r.status_code, 409, name)
            self.assertIn("/api/inc/action/cancel", r.json()["msg"])
            self.assertEqual(sc, [], name)

    def test_other_inc_jobs_need_cluster_access_and_are_recorded(self):
        r, sc, logged = self._cancel("inc_plan", can_use=False)
        self.assertEqual((r.status_code, sc), (403, []))
        r, sc, logged = self._cancel("inc_audit_pilot_v1")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(sc, [["scancel", "4242"]])
        self.assertEqual(logged[0][0], "cancel_job")
        self.assertEqual(logged[0][1]["actor"], "admin", "X-User is not trusted here")
        r, sc, logged = self._cancel("brain_hrv_1")
        self.assertEqual((r.status_code, len(sc)), (200, 1), "other prefixes keep their behaviour")

    def test_job_log_finds_inc_logs_and_only_the_right_one(self):
        src = pathlib.Path(self.D.__file__).read_text()
        fn = src[src.index("def api_job_log("):src.index("def api_recent_jobs(")]
        self.assertIn("grep -E '_{jobid}(_[0-9]+)?[.]out$'", fn)
        self.assertIn("{inc}/_campaign/plans/logs/", fn)
        root = pathlib.Path(os.environ["REPO_ROOT"]) / "results" / "framework" / "inc"
        for rel, text in (("pilot_x/logs/inc_pilot_x_0001_5551_3.out", "hello from task 3\n"),
                          ("pilot_12345/logs/inc_pilot_12345_0001_999_0.out", "array 999 of pilot_12345\n"),
                          ("run_b/logs/inc_run_b_0002_12345_1.out", "array 12345 task 1\n")):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        for jid, want in (("5551", "hello from task 3"), ("12345", "array 12345 task 1"),
                          ("999", "array 999 of pilot_12345"), ("5551_3", "hello from task 3")):
            r = self.c.get("/api/job_log/%s" % jid, headers=self.auth())
            self.assertEqual(r.status_code, 200, (jid, r.text[:300]))
            self.assertIn(want, r.json()["content"], jid)
        self.assertEqual(self.c.get("/api/job_log/555", headers=self.auth()).status_code, 404,
                         "job 555 must not open job 5551's log")
        self.assertEqual(self.c.get("/api/job_log/0001", headers=self.auth()).status_code, 400,
                         "a zero-padded id would match every array's step suffix")

    def test_batch_sacct_resolves_array_jobs(self):
        out = ("123_0|COMPLETED\n123_0.batch|COMPLETED\n123_1|RUNNING\n"
               "124|FAILED\n124.batch|FAILED\n"
               "125_[2-5%2]|PENDING\n125_0|COMPLETED\n"
               "126_0|COMPLETED\n126_1|COMPLETED\n"
               "127_0|COMPLETED\n127_1|FAILED\n"
               "128_0|COMPLETED\n128_1|REVOKED\n")
        real = self.D._slurm
        self.D._slurm = lambda cmd, timeout=15: {"ok": True, "stdout": out, "stderr": ""}
        try:
            m = self.D._batch_sacct(["123", "124", "125", "126", "127", "128"])
        finally:
            self.D._slurm = real
        self.assertEqual((m["123"], m["124"], m["125"], m["126"], m["127"]),
                         ("running", "failed", "running", "succeeded", "failed"))
        self.assertEqual(m["123_0"], "succeeded")
        self.assertNotIn("128", m, "a task in an unknown state leaves the array unresolved")

    def test_linked_from_supervision_and_project_pages(self):
        from weed_optimizer_framework.tools.brain import api as A
        self.assertIn('href="/inc"', A._PAGE)
        src = pathlib.Path(self.D.__file__).read_text()
        fn = src[src.index("def agent_generic("):]
        fn = fn[:fn.index("\n@app.")]
        self.assertIn('href="/inc"', fn)


def main():
    try:
        prog = unittest.main(exit=False, verbosity=2)
        ok = prog.result.wasSuccessful()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%s" % ("ALL PASS" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
