#!/usr/bin/env python3
"""The simulated world of the stream-mode tests (docs/CONTINUOUS_LOOP.md 6.8).

No cluster, no ssh, no GPU, no network: a simulated clock, a temporary lab
tree and cluster INC_DIR, and a fake `slurm_sh` that runs the REAL remote.py /
stream_remote.py verbs on that INC_DIR, with only the cluster's commands
replaced (sbatch, squeue, sacct, projects, the quota tool, a pinned driver's
advance and report, and the login-node verbs, which small fakes of group E's
inc2.stream CLI and group B's verdicts answer).

The world writes what the other groups' code writes on the cluster, in their
formats: group E's hash-chained stream ledger (init, arm, cut, build, commit
with Protocol v3 steps and the dispositions of inc2.stream.dispose itself,
milestone build and compare, feasibility, rollback, bisect) and its
queue_summary.json; group B's capacity/capacity_v1.json, <exp>/canary.json and
pilot_v4/stage_a.json; group C's step1_stream/status.json; group D's intake
batch summary.json, intake/sources.jsonl events and placement.json; the
splits lock; and pinned-driver experiments (exp.json, state.json,
report.json, ledger.jsonl, runs/<run>/scores/dev.json).

Imported by tests/test_stream_ap_replay.py, test_stream_ap_units.py and
test_stream_ap_identity.py. Run directly it runs a short smoke tick.
"""
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
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="stream_ap_"))
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("INC_DIR", str(TMP / "default_inc"))
os.environ.setdefault("REPO", str(TMP / "repo"))
for _k in ("INCAP_MAX_FILE_BYTES", "INCAP_MAX_LEDGER_BYTES", "CLUSTER_SSH", "INC_JOB_SCRIPT"):
    os.environ.pop(_k, None)
sys.path.insert(0, str(PKG_ROOT))

from weed_optimizer_framework.tools.brain import approvals as AP  # noqa: E402
from weed_optimizer_framework.tools.brain import policy as POL  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import budget as B  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import campaign as C  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import executor as X  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as R  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream as S  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream_remote as SR  # noqa: E402

FIX = PKG_ROOT / "tests" / "fixtures" / "inc_replay"
OWNER = "human:owner@example.org"
AUTO = M.AUTOPILOT_ACTOR
NAME = "stream1"
T0 = 1790812800.0                     # 2026-10-01T00:00:00Z
TICK = 600.0
DAY = 86400.0
RES = {"mongo_ok": True, "cluster_reachable": True}
SEG_RE = re.compile(r'echo "INCAP_SEG (\d+)"; (.*?); echo "INCAP_SEG_END', re.S)
FAILURES, SKIPS = [], []
CASES = {}
_CUR = {"case": None, "start": 0}


def check(name, cond, detail=""):
    tag = ("[%s] " % _CUR["case"]) if _CUR["case"] else ""
    if cond:
        print("  ok   %s%s" % (tag, name))
    else:
        print("  FAIL %s%s %s" % (tag, name, str(detail)[:1500]))
        FAILURES.append(tag + name)
    return bool(cond)


def run_case(cid, fn):
    """Run one S-case; it passes when it added no failure and did not raise."""
    print("%s" % cid)
    _CUR.update(case=cid, start=len(FAILURES))
    try:
        fn()
    except Exception as e:                                   # a crashed case is a failed case
        import traceback
        check("%s ran without raising" % cid, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-1500:]))
    ok = len(FAILURES) == _CUR["start"]
    CASES[cid] = "pass" if ok else "fail"
    print("case %s: %s" % (cid, CASES[cid]))
    _CUR.update(case=None)
    return ok


def closing():
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


def utc(t):
    return S._utc(t)


NON_DEV = tuple(M.non_dev_exams())


def perturb(obj, under=False):
    """Every value under a non-decision exam key changed (test blindness)."""
    if isinstance(obj, dict):
        return {k: perturb(v, under or k in NON_DEV) for k, v in obj.items()}
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


def all_pass_cases():
    c = dict({k: "pass" for k in X.REPLAY_REQUIRED}, R2="skip", R4b="skip")
    c.update({k: "pass" for k in X.STREAM_REPLAY_CASES})
    c[X.STREAM_MUTATION_CASE] = "pass"
    return c


class QuietLog(object):
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warning(self, m):
        self.lines.append(("warning", m))

    def error(self, m):
        self.lines.append(("error", m))


class FakeRunner(object):
    """The lab runner of a test: launches are recorded; a test sets results."""

    def __init__(self):
        self.launched, self.results = [], {}
        self.auto_sync = True

    def launch(self, job, argv, timeout=0):
        self.launched.append({"job": job, "argv": list(argv)})
        return {"job": job}

    def poll(self, job):
        if job not in self.results and str(job).startswith("sync_") and self.auto_sync:
            # a lab -> cluster push succeeds at once unless a test says otherwise
            self.results[job] = {"ok": True, "rc": 0, "seconds": 1.0}
        return self.results.get(job)

    def finish(self, ok=True, rc=0, **kw):
        for l in self.launched:
            if l["job"] not in self.results:
                self.results[l["job"]] = dict({"ok": ok, "rc": rc, "seconds": 1.0}, **kw)


class _Proc(object):
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


# ------------------------------------------------------------------ the world
GUARDS = ("flips", "regression", "species")


def _dispose(v3, truth, p_reject=0.25):
    """Group E's disposition rule (inc2.stream.dispose), used by the fake of its
    commit so that the dispositions the ledger records are the real rule's."""
    try:
        from weed_optimizer_framework.tools.inc2 import stream as E2
        return E2.dispose(v3, truth, p_reject)
    except ImportError:                                   # pragma: no cover - group E absent
        if v3["verdict"] == "ACCEPT":
            return "accepted"
        if v3["p_data"] <= p_reject and truth != "helps":
            return "data"
        g = v3["guards"]
        for k, d in (("species", "species"), ("flips", "flips"), ("regression", "recipe")):
            if not g[k]:
                return d
        return "hold" if v3["verdict"] == "HOLD" else "truth_helps"


def commit_fields(n, exp, rows, chosen="r0"):
    """The fields of inc2.stream's commit line for a segment whose chosen chain
    decided `rows` [(increment, verdict, P_data, failed guards, truth verdict,
    sources[, species failed])]: the v3 steps, the dispositions by group E's
    rule, d30 and d33 as group E computes them."""
    steps, disp = [], {}
    for k, r in enumerate(rows, 1):
        inc, verdict, pd, failed, tv = r[:5]
        sp = list(r[6]) if len(r) > 6 and r[6] else (["Rare"] if "species" in failed else [])
        v3 = {"verdict": verdict, "p_data": pd, "p_recipe": 0.0, "inc": 0.816, "cand_mean": 0.815,
              "null_mean": 0.814, "null_sd": 0.001, "guards": {g: g not in failed for g in GUARDS},
              "species_failed": sp, "v3_applied": True, "blame": "recipe"}
        d = _dispose(v3, tv, 0.25)
        disp[inc] = d
        steps.append({"k": k, "increment": inc, "v3": v3, "truth": tv, "disposition": d,
                      "pinned": {"verdict": verdict}})
    rej = [x for x in steps if x["v3"]["verdict"] == "REJECT"]
    rec = [x for x in rej if x["disposition"] == "recipe"]
    spc = {}
    for x in rej:
        for s_ in x["v3"]["species_failed"]:
            spc[s_] = spc.get(s_, 0) + 1
    return {"segment": n, "exp": exp, "chosen": chosen, "recipe": chosen,
            "accepted": [i for i, d in disp.items() if d == "accepted"], "steps": {chosen: steps},
            "dispositions": disp, "stale_base": False, "pool": {"name": "P_%d" % n},
            "d30": {"rejects": len(rej), "recipe_caused": len(rec), "fires": bool(rej) and 2 * len(rec) >= len(rej)},
            "d33": {"rejects": len(rej), "species_fail_counts": spc,
                    "species": sorted(s_ for s_, c in spc.items() if 2 * c >= len(rej))}, "discordant": []}


def ledger_text(events, sid="x"):
    """A hash-chained stream ledger (inc2.stream's form) of these events."""
    raw = b""
    for i, e in enumerate(events):
        rec = dict(e, sid=sid, seq=i, prev_sha256=hashlib.sha256(raw).hexdigest(),
                   utc=e.get("utc") or "2026-10-01T00:00:00Z", by=e.get("by") or "platform")
        raw += (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8")
    return raw.decode("utf-8")


class World(object):
    def __init__(self, tag, autonomy="envelope", data_autonomy="on", replay=True, perturbed=False,
                 domain="weed", stream_domain=None, floors=(1.0, 1.0), collect_config=None, sid=None,
                 protocol_v3=True):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="w_%s_" % tag, dir=str(TMP)))
        self.lab = self.dir / "lab"
        self.inc = self.dir / "inc"
        self.scripts = self.dir / "scripts"
        for d in (self.lab, self.inc, self.scripts):
            d.mkdir(parents=True)
        for n in ("run_inc_collect.sh", "run_inc2_stream.sh", "run_inc2_build.sh", "run_inc2_job.sh", "run_inc_job.sh"):
            (self.scripts / n).write_text("#!/bin/bash\n")
        self.cfg = self.dir / "round_scheduler.json"
        self.hooks = C._local_cfg_hooks(self.cfg)
        self.t = [T0]
        self.perturbed = perturbed
        self.domain = domain
        self.squeue, self.sacct, self.submits, self.runs, self.advances = [], {}, [], [], []
        self.tick_calls, self.verbs, self.ticks_out = [], [], []
        self.next_job = 5000
        self.qos = False
        self.busy_runs = 0
        self.projects = [{"resource": "Bridges-2 GPU", "allocation_su": 20000.0, "balance_su": 10529.0,
                          "end_date": "2026-12-31"}]
        self.quota_rec = {"ok": True, "used_gb": 675.0, "quota_gb": 7000.0, "free_gb": 6325.0}
        self.code_override = None
        self.log = QuietLog()
        self.runner = FakeRunner()
        self.floors = floors
        self._patch_thresholds()
        self.xctx = X.Context(lab_repo=str(self.lab), resources=RES, clock=self.clock, domain=domain)
        S.LAB_RUNNER = lambda run, w=self: w.runner
        # the collector's config (group D's file, a governance file): known items
        self.collect_cfg = self.lab / "collect" / "domains" / ("%s.json" % domain)
        self.collect_cfg.parent.mkdir(parents=True, exist_ok=True)
        self.collect_cfg.write_text(json.dumps(collect_config or {"format": "collect-domain/1", "known_items": [],
                                                                   "placement": {"lab_only": ["github"]}}))
        S.configure_stream(NAME, OWNER, domain=domain, enable=True, autonomy=autonomy, data_autonomy=data_autonomy,
                           protocol_v3_accepted=protocol_v3, stream_domain=stream_domain,
                           collect_config=str(self.collect_cfg), cfg_hooks=self.hooks, lab_repo=str(self.lab),
                           clock=self.clock)
        if sid:
            def ch(c):
                c["stream"]["sid"] = sid
                return c
            C._update_config(self.hooks, NAME, ch)
        self.dom = LS.load_domain(stream_domain or domain)
        self.sid = sid or self.dom["sid"]
        self.M = int(self.dom["increment"]["M"])
        # the fake of group E's stream (inc2.stream): its ledger, its summary
        self.es = {"pool": "P_0", "next_seg": 1, "next_ms": 1, "segments": {}, "chosen_recipe": None,
                   "stage_b": None, "arm": None, "rollback_pending": [], "accepted_since": 0, "segments_since": 0,
                   "first_accept_utc": None, "ms_in_flight": None, "boundary": None, "uncommitted_done": [],
                   "forked_to": None, "bisect_arms": 0,
                   "milestones": {"0": {"exp": self._arm_exp(self.dom["capacity"]["default_arm"]), "pool": "P_0",
                                        "state": "external", "verdict": "baseline"}},
                   "q": {"Q": 0, "boxes": {}, "oldest_utc": None, "consumed": {}, "pool_images": None,
                         "held": {}, "extra": {}}}
        # what the fakes of groups B and E decide when asked
        self.compare_plan, self.stage_c_plan, self.rollback_suspect = {}, None, ["I2"]
        self.capacity_choice, self.canary_pass, self.stage_a_recipes = self.dom["capacity"]["default_arm"], True, None
        if replay:
            self.replay_pass()

    def _arm_exp(self, arm):
        return (((self.dom.get("capacity") or {}).get("arms") or {}).get(arm) or {}).get("exp")

    # ---- plumbing
    def clock(self):
        return self.t[0]

    def advance(self, secs):
        self.t[0] += secs

    def _patch_thresholds(self):
        orig = LS.load_thresholds.__wrapped__ if hasattr(LS.load_thresholds, "__wrapped__") else LS.load_thresholds
        floors = self.floors

        def patched(path=None):
            th = orig(path)
            if floors is not None:
                th["D21"]["floor_gb"]["value"], th["D21"]["floor_su"]["value"] = floors
            return th
        patched.__wrapped__ = orig
        LS.load_thresholds = patched

    def replay_pass(self, cases=None):
        return X.record_replay_result("pass", cases or all_pass_cases(), ctx=self.xctx)

    def config(self):
        return (self.hooks[0]().get("campaigns") or {}).get(NAME) or {}

    def set_config(self, **kw):
        def ch(c):
            c.update(kw)
            return c
        C._update_config(self.hooks, NAME, ch)

    def state(self):
        return S.load_state(S.StreamPaths(str(self.lab), self.domain), NAME)

    def ledger(self):
        p = S.StreamPaths(str(self.lab), self.domain).ledger
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def events(self, name):
        return [r for r in self.ledger() if r.get("event") == name and r.get("campaign") == NAME]

    def approvals(self):
        return AP.state(self.domain, root=str(self.lab))

    def executions(self):
        return X.executions(self.xctx)

    def lane(self, ln):
        return (self.state() or {}).get("lanes", {}).get(ln) or {}

    # ---- the cluster's files
    def _w(self, rel, obj, mtime=None):
        p = self.inc / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.perturbed and isinstance(obj, (dict, list)):
            obj = perturb(obj)
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True))
        if mtime is not None:
            os.utime(str(p), (mtime, mtime))
        return p

    def lock(self, locked=True, manifests=("train_core", "tsw22", "tsw23", "base_v2", "dev", "test", "imageweeds")):
        p = self.inc / "splits" / "v2" / "LOCK.json"
        if locked:
            self._w("splits/v2/LOCK.json", {"splits_version": "v2", "manifests": {m: {"sha256": "a" * 64}
                                                                               for m in manifests},
                                            "h6_status": {"base_b": "clean"}})
        elif p.exists():
            p.unlink()

    # -- group E: the stream ledger and its summary (inc2.stream)
    def stream_event(self, event, **kw):
        """One hash-chained line of inc2.stream's ledger: seq, prev_sha256 (of
        the file's bytes before it), utc, sid, by."""
        p = self.inc / "stream" / self.sid / "ledger.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = p.read_bytes() if p.exists() else b""
        n = raw.count(b"\n")
        rec = dict(kw, event=event, sid=self.sid, by=kw.get("by") or "platform", utc=kw.get("utc") or utc(self.t[0]),
                   seq=n, prev_sha256=hashlib.sha256(raw).hexdigest())
        if self.perturbed:
            rec = perturb(rec)
        with open(str(p), "ab") as fh:
            fh.write((json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))
        return rec

    def stream_ledger(self):
        p = self.inc / "stream" / self.sid / "ledger.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def queue(self, Q, boxes=None, oldest_utc=None, consumed=None, pool_images=None, pool=None, held=None,
              extra=None, probe=None):
        """The eligible queue as inc2.stream's summary states it (and the rest
        of the summary from the fake stream's state). held: {hold: {rows,
        past_deadline}}."""
        q = self.es["q"]
        q.update(Q=Q, boxes=boxes or {}, oldest_utc=oldest_utc, consumed=consumed or {}, held=held or {},
                 extra=extra or {}, probe=probe)
        if pool_images is not None:
            q["pool_images"] = pool_images
        if pool:
            self.es["pool"] = pool
        self._summary()

    def _summary(self):
        es, q = self.es, self.es["q"]
        M_ = self.M
        age = None
        if q.get("oldest_utc"):
            age = (self.t[0] - S._secs(q["oldest_utc"])) / DAY
        ms = es["milestones"]
        good = [m for k, m in sorted(ms.items(), key=lambda kv: int(kv[0])) if m.get("verdict") != "hurts"
                and m.get("state") in ("external", "compared")]
        days = None
        if es["first_accept_utc"]:
            days = (self.t[0] - S._secs(es["first_accept_utc"])) / DAY
        due = []
        if es["accepted_since"] >= 4:
            due.append("accepted_increments")
        if es["segments_since"] >= 3:
            due.append("segments")
        if days is not None and days >= 30:
            due.append("days_with_accepted")
        if isinstance(es["boundary"], dict) and es["boundary"].get("fires"):
            due.append("boundary_check")
        probe = q.get("probe")
        if probe is None and q["Q"] >= M_:
            probe = {"exact_fill": True, "approximate": False}
        doc = {"format": "inc2-stream/queue-summary/1", "sid": self.sid, "stream_version": 1,
               "generated_utc": utc(self.t[0]), "testing": False, "M": M_, "K_max": 4,
               "arm": es["arm"], "stage_b": es["stage_b"], "chosen_recipe": es["chosen_recipe"],
               "forked_to": es["forked_to"],
               "pool": {"name": es["pool"], "current": es["pool"], "sha256": "p" * 64,
                        "images": q.get("pool_images") or int(self.dom["increment"]["base_images"]),
                        "n_images": q.get("pool_images") or int(self.dom["increment"]["base_images"])},
               "eligible": {"images": q["Q"], "target_boxes": q["boxes"], "oldest_utc": q["oldest_utc"],
                            "oldest_age_days": age, "by_source": {}},
               "held": {h: int((x or {}).get("rows") or 0) for h, x in (q.get("held") or {}).items()},
               "consumed_last": q["consumed"],
               "queue": {"eligible_images": q["Q"],
                         "held_past_deadline": {h: int((x or {}).get("past_deadline") or 0)
                                                for h, x in (q.get("held") or {}).items()
                                                if (x or {}).get("past_deadline")}},
               "cut": {"ready": q["Q"] >= 4 * M_, "k": max(0, min(4, q["Q"] // M_)) if M_ else 0, "probe": probe,
                       "last_refusal": None, "train_idle": True},
               "next_segment": "%s_s%03d" % (self.sid, es["next_seg"]),
               "uncommitted_done": list(es["uncommitted_done"]),
               "milestones": {"records": {k: dict(m) for k, m in ms.items()}, "in_flight": es["ms_in_flight"],
                              "last_good": good[-1]["exp"] if good else None,
                              "accepted_since": es["accepted_since"], "segments_since": es["segments_since"],
                              "days_since_first_accepted": days, "due": bool(due), "due_reasons": due,
                              "next": "%s_m%03d" % (self.sid, es["next_ms"])},
               "boundary_check": es["boundary"],
               "rollback_pending": list(es["rollback_pending"]),
               "ledger": {"events": len(self.stream_ledger())}}
        doc.update(q.get("extra") or {})
        self._w("stream/%s/queue_summary.json" % self.sid, doc)

    def stream_init(self, stage_b=("r0", "x1a")):
        self.es["stage_b"] = list(stage_b)
        self.stream_event("init", M=self.M, K_max=4, stage_b=list(stage_b), arm={"id": "n640"})
        self.stream_event("prospective", sha256="f" * 64)
        self._summary()

    def choose_arm(self):
        cap = json.loads((self.inc / "capacity" / "capacity_v1.json").read_text())
        self.es["arm"] = {"id": cap["chosen_arm"], "source": "capacity decision"}
        self.es["milestones"]["0"]["exp"] = cap["chosen_exp"]
        self.stream_event("arm", arm=self.es["arm"], milestone0=cap["chosen_exp"])
        self._summary()

    def segment(self, n, rows, done=True, chosen="r0", recipes=None, truth=True, listed=False, final=None, dev=None):
        """A segment inc2.stream built (its cut and build lines) and the pinned
        driver's experiment. rows: [(increment, verdict, P_data, failed guards,
        truth verdict, sources[, species failed])], read by the fake commit."""
        exp = "%s_s%03d" % (self.sid, n)
        if exp not in self.es["segments"]:
            for r in rows:
                self.stream_event("cut", increment=r[0], segment=n, sources={s_: 1 for s_ in r[5]})
            self.stream_event("build", segment=n, exp=exp, base_pool=self.es["pool"], increments=[r[0] for r in rows],
                              recipes=list(recipes or [chosen]), truth=bool(truth))
        self.es["segments"][exp] = {"n": n, "rows": rows, "chosen": chosen}
        self.es["next_seg"] = max(self.es["next_seg"], n + 1)
        if listed and done and exp not in self.es["uncommitted_done"]:
            self.es["uncommitted_done"].append(exp)
        self.experiment(exp, typ="chain", done=done, final=final, dev=dev, steps=[
            {"name": r[0], "clean": False, "manifest": "%s/stream/%s/increments/%s.jsonl" % (M.CLUSTER_INC_DIR, self.sid, r[0])}
            for r in rows])
        self.job_done("inc_build_%s" % exp)
        self._summary()
        return exp

    def _commit(self, exp):
        seg = self.es["segments"].get(exp)
        if seg is None:
            return 1, "", "[inc2.stream] ERROR: %s is not a segment of stream %s" % (exp, self.sid)
        fields = commit_fields(seg["n"], exp, seg["rows"], seg["chosen"])
        accepted, pool = fields["accepted"], fields["pool"]["name"]
        self.stream_event("commit", **fields)
        es = self.es
        es["pool"] = pool
        es["segments_since"] += 1
        es["accepted_since"] += len(accepted)
        if accepted and not es["first_accept_utc"]:
            es["first_accept_utc"] = utc(self.t[0])
        es["chosen_recipe"] = seg["chosen"]
        es["uncommitted_done"] = [x for x in es["uncommitted_done"] if x != exp]
        self._summary()
        return 0, json.dumps({"committed": exp}), ""

    def milestone_built(self, done=True):
        es = self.es
        n = es["next_ms"]
        exp = "%s_m%03d" % (self.sid, n)
        self.stream_event("milestone", phase="build", n=n, exp=exp, pool=es["pool"])
        es["milestones"][str(n)] = {"exp": exp, "pool": es["pool"], "state": "built", "verdict": None}
        es["ms_in_flight"] = exp
        es["next_ms"] = n + 1
        self.experiment(exp, typ="baseline", done=done)
        self.job_done("inc_build_%s" % exp)
        self._summary()
        return exp

    def _compare(self, exp):
        es = self.es
        ms = next(((k, m) for k, m in es["milestones"].items() if m.get("exp") == exp and k != "0"), None)
        if ms is not None:
            k, m = ms
            plan = dict({"verdict": "helps", "perm_p": 0.6, "new_mean_dev": 0.816, "old_mean_dev": 0.812,
                         "species_failed": []}, **(self.compare_plan.get(exp) or {}))
            good = [x for kk, x in sorted(es["milestones"].items(), key=lambda kv: int(kv[0]))
                    if kk != k and x.get("verdict") != "hurts" and x.get("state") in ("external", "compared")]
            hurts = plan["verdict"] == "hurts"
            to_pool = good[-1]["pool"] if hurts and good else None
            self.stream_event("milestone", phase="compare", n=int(k), exp=exp, compared_with=good[-1]["exp"],
                              verdict=plan["verdict"], perm_p=plan["perm_p"], truth_p=0.5,
                              new_mean_dev=plan["new_mean_dev"], old_mean_dev=plan["old_mean_dev"],
                              species_failed=plan["species_failed"], rollback_recommended=hurts, to_pool=to_pool)
            m.update(state="compared", verdict=plan["verdict"])
            es["ms_in_flight"] = None
            if hurts:
                es["rollback_pending"].append({"milestone": exp, "to_pool": to_pool})
            else:
                es["accepted_since"], es["segments_since"], es["first_accept_utc"] = 0, 0, None
            self._summary()
            return 0, json.dumps({"verdict": plan["verdict"]}), ""
        if exp == "%s_c001" % self.sid:
            res = self.stage_c_plan or {"m_feasible": True, "species_only_reject": False, "d33_prospective": [],
                                        "per_recipe": {"r0": {"verdict": "ACCEPT"}, "x1a": {"verdict": "ACCEPT"}}}
            self.stream_event("feasibility", phase="read", exp=exp, result=res)
            self._summary()
            return 0, json.dumps(res), ""
        for e in self.stream_ledger():
            if e.get("event") == "bisect" and e.get("phase") == "build" and exp in (e.get("arms") or {}).values():
                inc = next(i for i, x in e["arms"].items() if x == exp)
                self.stream_event("bisect", phase="decide", rollback=e.get("rollback"),
                                  rollback_utc=e.get("rollback_utc"), decisions={inc: "helps"})
                return 0, "{}", ""
        return 1, "", "[inc2.stream] ERROR: %s is neither a milestone, the feasibility chain nor a bisect arm" % exp

    def stage_c_built(self, done=True):
        exp = "%s_c001" % self.sid
        self.stream_event("feasibility", phase="build", exp=exp, holdout="tsw22", m=self.M)
        self.experiment(exp, typ="chain", done=done)
        self.job_done("inc_build_%s" % exp)
        return exp

    def bisect_built(self, done=True):
        rb = [e for e in self.stream_ledger() if e.get("event") == "rollback"][-1]
        arms = {}
        for inc in rb.get("suspect") or []:
            self.es["bisect_arms"] += 1
            arms[inc] = "%s_b%03d" % (self.sid, self.es["bisect_arms"])
        self.stream_event("bisect", phase="build", rollback=1, rollback_utc=rb["utc"], arms=arms, **{"from": rb["to"]})
        for exp in arms.values():
            self.experiment(exp, typ="baseline", done=done)
        self.job_done("inc_build_%s" % sorted(arms.values())[0])
        return arms

    # -- group C: step1_stream/status.json
    def step1_status(self, one_time=("bootstrap", "knowntruth", "backfill"), per_source=None, extra=None):
        st = {"format": "inc2-step1-stream/status/1", "built_utc": utc(self.t[0]), "stream_version": 1,
              "one_time": {k: ("2026-09-29T00:00:00Z" if k in one_time else None)
                           for k in ("bootstrap", "backfill", "knowntruth")},
              "versions": {"verifier": "v1", "reference": "v1", "splits_lock_sha256": "l" * 64, "embedder": "bioclip2"},
              "pending": {"intake_batches": {}}, "batches": {"committed": [], "in_progress": [], "by_kind": {}},
              "queue": {"rows": 0}, "admission": {}, "holds": {}, "holds_past_deadline": {}, "refused": {},
              "knowntruth": {}, "refit_triggers": {"precision_lb_below": [], "species_unknown_share": [],
                                                   "fired": False},
              "human_queue": {"rows": 0},
              "per_source": {k: dict({"images_seen": 0, "near_eval_embed": 0, "target_boxes_admitted": 0}, **v)
                             for k, v in (per_source or {}).items()}}
        st.update(extra or {})
        self._w("step1_stream/status.json", st)

    # -- group D: the intake batch summary, the source ledger, the probe
    def intake(self, batch, source, images=100, eval_share=0.0, base_share=0.0, target_boxes=None, reasons=None):
        tb = images * 2 if target_boxes is None else target_boxes
        self._w("intake/%s/summary.json" % batch, {
            "format": "collect-summary/1", "source": source, "batch": batch, "rows": images,
            # the flat fields collect.intake writes for D28: the images the guard checked and its refusals
            "images": images + sum((reasons or {}).values()), "guard": dict(reasons or {}),
            "yield": {"images_seen": images + sum((reasons or {}).values()), "images_kept": images,
                      "target_images": images if tb else 0, "target_boxes": tb, "rejected": reasons or {}},
            "source_leak": {"eval_share": eval_share, "base_share": base_share,
                            "fires": eval_share >= 0.05 or base_share >= 0.2},
            "zero_yield": tb == 0, "zero_yield_reasons": (reasons or {}) if tb == 0 else None})

    def sources(self, rows):
        """Events of the collector's source ledger (collect.state: source,
        event, ts, bytes, su ...)."""
        p = self.inc / "intake" / "sources.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), "a") as fh:
            for r in rows:
                fh.write(json.dumps(dict({"ts": utc(self.t[0]), "format": "collect-source-event/1"}, **r),
                                    sort_keys=True) + "\n")

    def placement(self, providers):
        self._w("intake/placement.json", {"in_slurm": True, "providers": {
            k: {"status": v, "reachable": v == "pass", "placement": "cluster" if v == "pass" else "lab"}
            for k, v in providers.items()}})

    def candidates(self, rows, recall=None):
        p = self.lab / "results" / "framework" / "inc" / "collect" / "candidates.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {"format": "collect-candidates/1", "candidates": rows}
        if recall is not None:
            doc["recall"] = recall
        p.write_text(json.dumps(doc))

    # -- group B: the R0 verdicts
    def capacity_file(self, chosen=None):
        chosen = chosen or self.capacity_choice
        arms = self.dom["capacity"]["arms"]
        self._w("capacity/capacity_v1.json", {
            "format": "inc2-capacity/1", "chosen_arm": chosen, "chosen_exp": arms[chosen]["exp"],
            "qualifying": [] if chosen == self.dom["capacity"]["default_arm"] else [arms[chosen]["exp"]],
            "arms": {a["exp"]: {"arm": k, "mean": 0.81, "sd": 0.004, "seeds": [0, 1, 2]} for k, a in arms.items()},
            "truth_every": 1, "n_images": int(self.dom["increment"]["base_images"]), "m": self.M,
            "step_cost": {"gpu_h": [10.0, 14.0], "estimate": True}})

    def canary_file(self, passed=None):
        b = next((x for x in self.dom["baselines"]["items"] if x.get("verdict") == "canary-verdict"), None)
        if b is None:
            return
        ok = self.canary_pass if passed is None else passed
        self._w("%s/canary.json" % b["exp"], {"format": "inc2-canary/1", "exp": b["exp"], "passed": ok,
                                               "within_one_sd": ok, "sidecar_ok": True, "production": True,
                                               "canary_dev": 0.806, "reference_dev": {"mean": 0.8082, "sd": 0.0063}})

    def stage_a_file(self, status="READY", recipes=None):
        sa = self.dom["stage_a"]
        rec = list(recipes or self.stage_a_recipes or ["r0", "x1a"])
        self._w("%s/%s" % (sa["exp"], sa.get("record") or "stage_a.json"), {
            "format": "inc2-stage-a/1", "exp": sa["exp"], "status": status, "segment1_recipes": rec,
            "survivors": rec[1:], "best_survivor": rec[1] if len(rec) > 1 else None,
            "pending": {} if status == "READY" else {"experiment_done": True}})

    def experiment(self, exp, typ="baseline", done=True, gen=10, final=None, ledger=None, dev=None, hours=None,
                   blocked=None, extra_report=None, steps=None):
        """A pinned-driver experiment on the cluster (built; done or running)."""
        if steps is None and ledger:
            steps = [{"name": e["step"], "clean": False, "manifest": "%s/%s/manifests/%s.jsonl"
                      % (M.CLUSTER_INC_DIR, exp, e["step"])} for e in ledger if e.get("type") == "gate"]
        self._w("%s/exp.json" % exp, {"exp": exp, "type": typ, "initialised_utc": utc(self.t[0] - 3600),
                                      "seeds": [0, 1, 2], "steps": steps or [],
                                      "base": {"manifest": "%s/%s/manifests/base.jsonl" % (M.CLUSTER_INC_DIR, exp)}})
        self._w("%s/state.json" % exp, {"exp": exp, "type": typ, "done": done, "generation": gen, "runs": {},
                                        "blocked": blocked or {}, "unblocks": [], "chains": {}, "submissions": []},
                mtime=self.t[0] - 120)
        if ledger is not None:
            lines = [json.dumps(perturb(e) if self.perturbed else e, sort_keys=True) for e in ledger]
            (self.inc / exp / "ledger.jsonl").write_text("".join(x + "\n" for x in lines))
        if done:
            rep = {"exp": exp, "type": typ, "done": True, "done_utc": utc(self.t[0] - 60),
                   "gpu_hours": {"base": {"hours": hours or 4.0, "runs": 3}}, "final": final or []}
            rep.update(extra_report or {})
            self._w("%s/report.json" % exp, rep, mtime=self.t[0])
        for run, v in (dev or {}).items():
            self._w("%s/runs/%s/scores/dev.json" % (exp, run),
                    {"exam": "dev", "map50_95": v, "n_gt": {}, "production": True})
        for j in self.squeue:
            if j["name"] == "inc_build_%s" % exp:
                # the build job ended once it built the experiment: sacct knows it
                self.sacct.setdefault(j["id"], {"state": "COMPLETED", "elapsed_s": 1800.0, "gpu_count": 1,
                                                "gpu_type": "v100-32"})
        self.squeue = [j for j in self.squeue if j["name"] != "inc_build_%s" % exp]

    def final_row(self, model, dev_mean, dev_sd=0.005, n=3, test_mean=0.85):
        return {"model": model, "runs": [], "exams": {"dev": {"twelve": {"mean": dev_mean, "sd": dev_sd, "n": n}},
                                                      "test": {"twelve": {"mean": test_mean, "sd": 0.01, "n": n}},
                                                      "imageweeds": {"twelve": {"mean": 0.05, "sd": 0.01, "n": n}}}}

    # ---- the fake cluster behind slurm_sh
    def __call__(self, script, timeout=60):
        if not self.tick_calls:
            self.tick_calls.append([])
        self.tick_calls[-1].append(script)
        out = ["Welcome"]
        for m in SEG_RE.finditer(script):
            i, argv = int(m.group(1)), shlex.split(m.group(2))
            out.append("INCAP_SEG %d" % i)
            rec = self._answer(argv)
            out.append("INCAP " + json.dumps(rec, default=str))
            out.append("INCAP_SEG_END %d %d" % (i, 0 if rec.get("ok") else 1))
        return {"ok": True, "stdout": "\n".join(out), "stderr": "", "returncode": 0}

    def _answer(self, argv):
        args = argv[4:]                              # python -u -m MODULE ARGS
        self.verbs.append(args)
        self._activate()
        if args and args[0] == "submit":             # a v1 builder (L4's label audit): recorded, a job id
            self.submits.append({"argv": list(args), "name": "inc_audit", "env": {}})
            self.next_job += 1
            self.squeue.append({"id": str(self.next_job), "name": "inc_audit_x", "state": "PENDING"})
            return {"verb": "submit", "ok": True, "job_id": str(self.next_job), "argv": args}
        return R.dispatch(args)

    def _activate(self):
        w = self
        R.inc_dir = lambda: w.inc
        R.script_dir = lambda: w.scripts
        R.squeue_jobs = lambda prefix="inc_": {"ok": True, "jobs": [dict(j, elapsed="0:01", submit="x")
                                                                    for j in w.squeue if j["name"].startswith(prefix)]}
        R.advance = self._advance
        R.report = self._report
        SR.projects = lambda: {"ok": True, "resources": copy.deepcopy(w.projects)}
        SR.quota = lambda largest=False: dict(w.quota_rec, **({"largest": [{"dir": "downloads", "gb": 300.0}]}
                                                              if largest else {}))
        SR.sacct = lambda ids: {"ok": True, "jobs": {str(j): w.sacct[str(j)] for j in ids if str(j) in w.sacct}}
        SR.subprocess = types.SimpleNamespace(run=self._subprocess_run, TimeoutExpired=subprocess.TimeoutExpired)
        SR._run = self._run
        if self.code_override is not None:
            # the cluster's copy (called without a root) differs; the lab reads its own tree
            SR.module_hashes = lambda root=None, w=w: dict(w.code_override) if root is None else _REAL_MODULE_HASHES(root)
        else:
            SR.module_hashes = _REAL_MODULE_HASHES

    def _subprocess_run(self, argv, **kw):
        if argv and argv[0] == "sbatch":
            name = next(a.split("=", 1)[1] for a in argv if a.startswith("--job-name="))
            self.submits.append({"argv": list(argv), "name": name, "env": {k: v for k, v in (kw.get("env") or {}).items()
                                                                            if k.startswith("INCAP_")}})
            if self.qos:
                return _Proc(1, "", "sbatch: error: Batch job submission failed: Invalid qos specification")
            self.next_job += 1
            self.squeue.append({"id": str(self.next_job), "name": name, "state": "PENDING"})
            return _Proc(0, "%d\n" % self.next_job, "")
        raise AssertionError("the fake cluster ran %r" % (argv,))

    def _run(self, argv, timeout=120, stdin=None, env=None):
        """The login-node verbs: a fake of group E's inc2.stream CLI (commit,
        compare, choose-arm, rollback, quarantine, release) and of group B's
        verdicts (inc2.baseline canary-verdict / capacity-verdict, inc2.pilot4
        verdict), writing the files those write."""
        self.runs.append(list(argv))
        self.run_env = {k: v for k, v in (env or {}).items() if k.startswith("INCAP_")}
        module = argv[2].rsplit(".", 1)[-1] if len(argv) > 2 else None
        verb = argv[3] if len(argv) > 3 else None
        flags = dict(zip(argv[4::2], argv[5::2]))
        if self.busy_runs > 0 and module == "stream":
            self.busy_runs -= 1
            return 3, "", "[inc2.stream] BUSY: stream.lease is held by another writer"
        if module == "stream":
            if verb == "commit":
                return self._commit(flags.get("--exp"))
            if verb == "compare":
                return self._compare(flags.get("--exp"))
            if verb == "choose-arm":
                self.choose_arm()
                return 0, "{}", ""
            if verb == "rollback":
                to = flags.get("--to")
                pend = [x for x in self.es["rollback_pending"] if x["to_pool"] == to]
                self.stream_event("rollback", to=to, suspect=list(self.rollback_suspect),
                                  milestone=pend[-1]["milestone"] if pend else None, envelope=bool(pend),
                                  **{"from": self.es["pool"]})
                self.es["rollback_pending"] = [x for x in self.es["rollback_pending"] if x["to_pool"] != to]
                self.es["pool"] = to
                self._summary()
                return 0, json.dumps({"ok": True}), ""
            if verb in ("quarantine", "release"):
                self.stream_event(verb, source=flags.get("--source"), hold=flags.get("--hold"),
                                  cite=flags.get("--cite"))
                return 0, json.dumps({"ok": True}), ""
        if module == "baseline" and verb == "capacity-verdict":
            self.capacity_file()
            return 0, "{}", ""
        if module == "baseline" and verb == "canary-verdict":
            self.canary_file()
            return 0, "{}", ""
        if module == "pilot4" and verb == "verdict":
            self.stage_a_file()
            return 0, "{}", ""
        return 1, "", "unknown verb %r %r" % (module, verb)

    def _advance(self, exp, backend=None):
        self.advances.append({"exp": exp, "job_script": os.environ.get("INC_JOB_SCRIPT")})
        rec = R.base_record("advance")
        rec["exp"] = exp
        rec["result"] = {"locked": False, "passes": 1, "submitted": 0, "job_ids": [], "done": False, "lines": []}
        return rec

    def _report(self, exp):
        rec = R.base_record("report")
        rec["exp"] = exp
        rec.update(done=True, files={})
        return rec

    # ---- driving
    def tick(self, n=1, advance=True):
        out = None
        for _ in range(n):
            self.tick_calls.append([])
            out = C.tick(slurm_sh=self, cfg_hooks=self.hooks, log=self.log, clock=self.clock, lab_repo=str(self.lab),
                         resources=RES)
            self.ticks_out.append(out)
            if (out or {}).get("ssh_refused"):
                check("tick %d attempted no second ssh" % len(self.tick_calls), False, out)
            if (out or {}).get("ok") is False:
                check("tick %d ran" % len(self.tick_calls), False, out)
            res = ((out or {}).get("campaigns") or {}).get(NAME) or {}
            if res.get("error"):
                st = self.state() or {}
                print("      tick error: %s\n%s" % (res.get("error"), ((st.get("last_error") or {}).get("trace") or "")[-1200:]))
            if advance:
                self.advance(TICK)
        return out

    def max_calls(self):
        return max([len(c) for c in self.tick_calls] or [0])

    def job_done(self, name_prefix, state="COMPLETED", elapsed=1800.0, refusal=None):
        """Every queued job whose name starts with the prefix ends (sacct state;
        `refusal`: the refusal line stream_remote reads off a data job's log)."""
        keep = []
        for j in self.squeue:
            if j["name"].startswith(name_prefix):
                self.sacct[j["id"]] = {"state": state, "elapsed_s": elapsed, "gpu_count": 1, "gpu_type": "v100-32"}
                if refusal:
                    self.sacct[j["id"]]["refusal"] = refusal
            else:
                keep.append(j)
        self.squeue = keep

    def ready_r0(self, stage_c=True, bootstrap=True):
        """Every R0/R1 prerequisite in place on the cluster, as the other
        groups' code writes it: the lock; the baselines done; the canary's,
        the capacity grid's and Stage A's verdicts (READY: r0 and x1a); the
        stream created with its arm adopted; Stage C built and read (M
        feasible); the placement; Step 1's one-time jobs."""
        self.lock()
        if bootstrap:
            self.step1_status()
        self.placement({"hf": "pass", "ftp": "pass", "weedai": "pass", "kaggle": "pass", "roboflow": "pass",
                        "mendeley_zenodo": "pass"})
        for b in self.dom["baselines"]["items"]:
            self.experiment(b["exp"], final=[self.final_row("base base_v2", 0.812, 0.002, 5)])
        self.canary_file()
        self.capacity_file()
        sa = self.dom["stage_a"]
        self.experiment(sa["exp"], typ="chain")
        self.stage_a_file()
        self.stream_init(self.stage_a_recipes or ("r0", "x1a"))
        self.choose_arm()
        if stage_c:
            self.stage_c_built()
            self._compare("%s_c001" % self.sid)


def gate(step, verdict, p_data, failed=(), sources=("src_a",), k=1, species_failed=None, chain="r0", p_recipe=0.0,
         p_reject=0.25, cm=0.815, nm=0.814, cs=0.002, ns=0.001):
    guards = {g: {"passed": g not in failed} for g in ("flips", "regression", "species")}
    if "species" in failed:
        guards["species"]["failed"] = list(species_failed or ["Rare"])
    return {"type": "gate", "id": "gate/%s/%d" % (chain, k), "step": step, "k": k, "chain": chain, "clean": False,
            "sources": list(sources),
            "decision": {"verdict": verdict, "p_data": p_data, "p_recipe": p_recipe, "guards": guards,
                         "attribution": {"blame": "recipe", "species_failed": list(species_failed or
                                                                                 (["Rare"] if "species" in failed else []))},
                         "config": {"p_reject": p_reject}, "cand_mean": cm, "null_mean": nm, "cand_sd": cs,
                         "null_sd": ns, "inc": 0.816}}


def truth(step, verdict, k=1):
    return {"type": "truth", "id": "truth/%d" % k, "step": step, "k": k,
            "detail": {"verdict": verdict, "p": 0.5, "with_mean": 0.8, "without_mean": 0.8}}


def seg_ledger(rows, chain="r0"):
    """[(step, verdict, P_data, failed guards, truth verdict, sources)] -> gate and truth entries."""
    out = []
    for i, r in enumerate(rows):
        step, verdict, pd, failed, tv, srcs = r[:6]
        sp = r[6] if len(r) > 6 else None
        if tv:
            out.append(truth(step, tv, i + 1))
        out.append(gate(step, verdict, pd, failed, srcs, i + 1, sp, chain))
    return out


_REAL_MODULE_HASHES = SR.module_hashes
_REAL_RUN = SR._run


def smoke():
    w = World("smoke")
    w.tick(2)
    print("ticks:", [((o or {}).get("campaigns") or {}).get(NAME) for o in w.ticks_out])
    print("verbs:", [v[0] for v in w.verbs])
    return 0


if __name__ == "__main__":
    sys.exit(smoke())
