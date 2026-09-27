#!/usr/bin/env python3
"""The INC campaign end to end, from pilot_v2's real record to the real loop
(docs/INC_AUTOPILOT.md: D15, D16, L9 and the gate carried through).

inc_autopilot.campaign.tick() runs on a simulated clock against a fake
`slurm_sh`. The fake answers the executor's scripts by running the real
remote.py verbs (campaign-snapshot, snapshot, status, step1) on a temporary
INC_DIR; the driver's advance, the report writer, squeue and sbatch are the
fake cluster's. The INC_DIR is laid out from the replay fixtures
(tests/fixtures/inc_replay: pilot_v1, pilot_v2 and base_b_v1, byte for byte,
each with a finished state.json) and a small synthetic Step 1 without
relevance.json (one source with no domain evidence). The children the
campaign builds are synthetic: pilot_v3 is pilot_v2's definition pinned to
protocol v2 (exp.json gate block, state.json gate_pin, ledger gate_pin/0,
every decision's config flips_mode net) whose full chain agrees with the
truth arm on 6 of 7 steps, with report.json v2_check 'supported' (or
'refuted' by a non-clean acceptance in the second scenario).

Pinned:
  * the cycle: pilot_v2 finished -> D15 fires (the five flips-only REJECTs,
    pooled over chains, cited), D1 fires unblocked with card X1 (three
    truth-helps misses failed another guard too) -> L9 proposed first, argv
    exactly
    'python -m weed_optimizer_framework.tools.inc.pilot build --exp pilot_v3
    --replay-mode full --gate-flips-mode net' -> without a replay pass it
    waits for a person; with one, envelope autonomy grants it (the
    autopilot's grant, run as the person who enabled the envelope) ->
    executed exactly once -> RUN pilot_v3 (building, built, advanced) ->
    finished: REPORT (spend charged, outcome scored) -> DIAGNOSE: D16 info
    (supported), D4 decision_slot_ready (full 6/7), prospective_d4 JSON
    (ready, full replay, recipe full, gate net) written before any L2 ->
    D2 fires on Step 1 without relevance.json -> L3 executed directly (R2)
    -> WAIT_JOB -> the job leaves the queue and relevance.json exists ->
    DIAGNOSE: D2 silent, L2 proposed with --relevance, --gate-flips-mode net
    and D4's replay mode and recipe -> envelope autonomy -> executed once ->
    RUN realloop_v1;
  * the same cycle with relevance.json already present: at pilot_v3's
    DIAGNOSE the prospective record and the L2 proposal land in the same
    tick, the record first (the ordering inside one DIAGNOSE, not a tick
    boundary, keeps the record ahead of the build);
  * the refuted branch: v2_check 'refuted' -> D16 crit, card X7, D4
    blocked_by D16, the prospective record not ready; L3 still runs, and no
    L2 is ever proposed, filed or run; the campaign completes with a
    residual card;
  * budget exhaustion: an envelope that fits L9's estimate (0.5 SU spare)
    but not what pilot_v3 really cost (its runs at 1.3x the measured rates:
    more than the estimate's experiment_hours) -> the next charge is refused
    on budget and the campaign pauses; later ticks make no ssh and submit
    nothing;
  * one ssh per tick in every tick of every scenario (the ticker's budget
    never refuses a call).

Run:  python3 tests/test_inc_ap_e2e.py
"""
import copy
import csv
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_e2e_"))
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ["INC_DIR"] = str(TMP / "default_inc")      # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
os.environ.pop("INCAP_MAX_FILE_BYTES", None)
os.environ.pop("INCAP_MAX_LEDGER_BYTES", None)
sys.path.insert(0, str(PKG_ROOT))

from weed_optimizer_framework.tools.brain import approvals as AP  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import campaign as C  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import executor as X  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers as LV  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import remote as R  # noqa: E402

FIX = PKG_ROOT / "tests" / "fixtures" / "inc_replay"
FAILURES = []
OWNER = "human:owner@example.org"
AUTO = M.AUTOPILOT_ACTOR
NAME = "weedinc_e2e"
T0 = 1790000000.0
TICK = 600.0
SEG_RE = re.compile(r'echo "INCAP_SEG (\d+)"; (.*?); echo "INCAP_SEG_END', re.S)
REPLAY_PASS = dict({c: "pass" for c in X.REPLAY_REQUIRED}, R2="skip", R4b="skip")
RESOURCES = {"mongo_ok": True, "cluster_reachable": True}
EXPS = ["pilot_v1", "pilot_v2", "base_b_v1"]
GOAL = {"kind": "exp_done", "exp": "realloop_v1"}
L9_ARGV = ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build", "--exp", "pilot_v3",
           "--replay-mode", "full", "--gate-flips-mode", "net"]
OCEAN_INC = "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark/results/framework/inc"
OFF_DOMAIN = "fvossel__csgo_player_detection"
IN_DOMAIN = "francesco__grass_weeds"


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


def _fix(rel):
    return (FIX / rel).read_text(encoding="utf-8")


# --- the synthetic child: pilot_v2's definition pinned to protocol v2 --------------------
# Verdicts per (chain, step) under v2. The full chain agrees with the truth arm on 6 of 7
# (all but I4, which the truth arm calls neutral), freeze and lora on 5 of 7; every
# truth-helps step (I2, I3, I5) is ACCEPTed in every chain, so D1 and D15 are silent.
V3_VERDICTS = {
    "full": {"I1": "REJECT", "I2": "ACCEPT", "Bswap": "REJECT", "I3": "ACCEPT", "Breal": "REJECT",
             "I4": "ACCEPT", "I5": "ACCEPT"},
    "freeze": {"I1": "ACCEPT", "I2": "ACCEPT", "Bswap": "REJECT", "I3": "ACCEPT", "Breal": "REJECT",
               "I4": "REJECT", "I5": "ACCEPT"},
    "lora": {"I1": "ACCEPT", "I2": "ACCEPT", "Bswap": "REJECT", "I3": "ACCEPT", "Breal": "REJECT",
             "I4": "REJECT", "I5": "ACCEPT"},
}
EQ = {"ACCEPT": "helps", "HOLD": "neutral", "REJECT": "hurts"}


def v3_verdicts(outcome):
    v = copy.deepcopy(V3_VERDICTS)
    if outcome == "refuted":
        v["lora"]["Breal"] = "ACCEPT"            # v2 accepts a step not marked clean: refuted
    return v


def child_files(name, outcome, cost_scale=1.0):
    """exp.json, build_summary.json, the whole ledger and report.json of `name`,
    pilot_v2 rebuilt with --gate-flips-mode net."""
    exp = json.loads(_fix("pilot_v2/exp.json").replace("pilot_v2", name))
    exp.update(initialised_utc="2026-09-28T02:00:00Z", gate={"flips_mode": "net"})
    bs = json.loads(_fix("pilot_v2/build_summary.json").replace("pilot_v2", name))
    bs["gate"] = {"flips_mode": "net"}
    verdicts = v3_verdicts(outcome)
    truth, lines, cfg_net = {}, [], None
    for ln in _fix("pilot_v2/ledger.jsonl").replace("pilot_v2", name).splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        if e.get("type") == "truth":
            truth[e["step"]] = e["detail"]["verdict"]
        if e.get("type") == "gate":
            d = e["decision"]
            d["config"]["flips_mode"] = "net"
            cfg_net = d["config"]
            v = verdicts[e["chain"]][e["step"]]
            d["verdict"] = v
            if v == "ACCEPT":
                for g in d["guards"].values():
                    g["passed"] = True
                d["p_data"] = max(d["p_data"], 0.889)
            if e["step"] == "Bswap":
                d["attribution"]["class_vs_loc"] = "labels"
        lines.append(e)
        if e.get("type") == "code_pin":
            lines.append({"id": "gate_pin/0", "type": "gate_pin", "exp": name, "block": {"flips_mode": "net"},
                          "config": None})
    for e in lines:
        if e.get("type") == "gate_pin":
            e["config"] = dict(cfg_net)
    rep = json.loads(_fix("pilot_v2/report.json").replace("pilot_v2", name))
    rep.update(done=True, done_utc="2026-09-28T09:00:00Z",
               gate={"flips_mode": "net", "pinned": True, "block": {"flips_mode": "net"}, "config": dict(cfg_net),
                     "ledger_checked": {}, "ledger_mismatches": [], "exp_json_changes": {}})
    agree = {r: 0 for r in verdicts}
    for st in rep["steps"]:
        for r, c in st["chains"].items():
            c["verdict"] = verdicts[r][st["step"]]
            c["agree"] = EQ[c["verdict"]] == st["truth"]["verdict"]
            c["guards"] = {g: True for g in c["guards"]} if c["verdict"] == "ACCEPT" else c["guards"]
            if st["step"] == "Bswap":
                c["attribution"]["class_vs_loc"] = "labels"
            agree[r] += c["agree"]
    rep["agreement"] = {r: {"agree": a, "compared": 7, "rate": a / 7.0} for r, a in agree.items()}
    for r, ch in rep["chains"].items():
        ch["bswap"]["attributed_to_labels"] = True
        ch["accepted"] = [s for s, v in verdicts[r].items() if v == "ACCEPT"]
    bad = [{"chain": r, "step": s, "v1_counterfactual": "REJECT"} for r in sorted(verdicts)
           for s, v in verdicts[r].items() if v == "ACCEPT" and s in ("Bswap", "Breal")]
    a2 = sum(agree.values())
    rep["v2_check"] = {"final": True, "truth_arm": True, "outcome": outcome, "compared": 21, "agree_v2": a2,
                       "agree_v1_counterfactual": 8, "discordant": [], "non_clean_accepted": bad}
    for u in rep["gpu_hours"].values():
        u["hours"] = u["hours"] * cost_scale
    rep["gpu_hours_total"] = sum(u["hours"] for u in rep["gpu_hours"].values())
    return {"exp.json": json.dumps(exp, sort_keys=True), "build_summary.json": json.dumps(bs, sort_keys=True),
            "ledger": [json.dumps(e, sort_keys=True) for e in lines], "report": rep}


# --- Step 1: one source with domain evidence, one without, no relevance.json -------------
def step1_files():
    inc_rows = [{"key": "g%02d" % i, "source": IN_DOMAIN} for i in range(10)]
    inc_rows += [{"key": "c%02d" % i, "source": OFF_DOMAIN} for i in range(20)]
    base_rows = [{"key": "b%02d" % i, "source": IN_DOMAIN} for i in range(5)]
    csv_rows = ([{"key": r["key"], "status": "selected" if i < 6 else "below_gate", "species_boxes": 3,
                  "other_boxes": 0} for i, r in enumerate(inc_rows[:10])]
                + [{"key": r["key"], "status": "no_evidence", "species_boxes": 0, "other_boxes": 2}
                   for r in inc_rows[10:]]
                + [{"key": r["key"], "status": "selected", "species_boxes": 4, "other_boxes": 0} for r in base_rows])

    def jl(rows):
        return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    inc_t, base_t = jl(inc_rows), jl(base_rows)
    sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()  # noqa: E731
    sel = {"sizes": {"base_B": 3927, "increment_pool": len(inc_rows)},
           "sources": {"increment_pool": {IN_DOMAIN: 10, OFF_DOMAIN: 20}},
           "retrieval": {"source_evidence": {IN_DOMAIN: {"median": 0.4, "species_crops": 30}}},
           "outputs": {"increment_pool.jsonl": {"sha256": sha(inc_t)}, "base_selected.jsonl": {"sha256": sha(base_t)},
                       "base_B.jsonl": {"sha256": "b" * 64}}}
    admit = {"per_slug": {IN_DOMAIN: {"boxes": {"verified": 25, "other_ok": 5}},
                          OFF_DOMAIN: {"boxes": {"other_ok": 40, "small": 3}}}}
    rel = {"format": "inc.relevance/1", "params": {"min_crops": 20, "calibration_percentile": 5.0, "tau_min": 0.5},
           "calibration": {"tau": 0.62, "check": {"ok": True, "tau_min": 0.5}},
           "inputs": {"increment_pool": {"sha256": sha(inc_t)}, "base_selected": {"sha256": sha(base_t)}},
           "increment_pool": {"sources": {IN_DOMAIN: {"status": "pass", "images": 10,
                                                      "top_set_share": {"leaf_disease": 0.0}},
                                          OFF_DOMAIN: {"status": "fail", "images": 20,
                                                       "top_set_share": {"leaf_disease": 0.0}}}}}
    return {"increment_pool.jsonl": inc_t, "base_selected.jsonl": base_t, "csv": csv_rows,
            "select_summary.json": json.dumps(sel, sort_keys=True),
            "admit_summary.json": json.dumps(admit, sort_keys=True), "relevance": json.dumps(rel, sort_keys=True)}


# --- the fake cluster ---------------------------------------------------------------------
class World(object):
    """A lab tree, an INC_DIR laid out from the fixtures, the fake cluster behind
    slurm_sh and a simulated clock. The remote.py verbs run for real on the INC_DIR."""

    def __init__(self, tag, envelope_su=300.0, outcome="supported", cost_scale=1.0):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="w_%s_" % tag, dir=str(TMP)))
        self.lab, self.inc = self.dir / "lab", self.dir / "inc"
        self.lab.mkdir()
        self.inc.mkdir()
        self.hooks = C._local_cfg_hooks(self.dir / "round_scheduler.json")
        self.t = [T0]
        self.outcome, self.cost_scale = outcome, cost_scale
        self.squeue, self.tick_calls, self.verbs, self.submits, self.advances = [], [], [], [], []
        self.reports = {}
        self.next_job = 7000
        self.log = QuietLog()
        self.xctx = X.Context(lab_repo=str(self.lab), resources=RESOURCES, clock=self.clock)
        for e in EXPS:
            self._put_fixture(e)
        self._put_step1()
        C.configure(NAME, OWNER, enable=True, exps=EXPS, current="pilot_v2", autonomy="envelope",
                    envelope_su=envelope_su, cfg_hooks=self.hooks, lab_repo=str(self.lab), clock=self.clock)
        C.set_goal(NAME, GOAL, OWNER, cfg_hooks=self.hooks, lab_repo=str(self.lab), clock=self.clock)

    def clock(self):
        return self.t[0]

    # ---- the INC_DIR
    def _w(self, rel, text, mtime=None):
        p = self.inc / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        if mtime is not None:
            os.utime(str(p), (mtime, mtime))

    def _ledger(self, exp, lines):
        self._w("%s/ledger.jsonl" % exp, "".join(x + "\n" for x in lines))

    def _state(self, exp, done, generation, mtime=None, gate_pin=None):
        st = {"exp": exp, "type": "chain", "done": done, "generation": generation,
              "done_utc": "2026-09-27T07:49:08Z" if done else None, "runs": {}, "blocked": {}, "unblocks": [],
              "chains": {}, "submissions": []}
        if gate_pin is not None:
            st["gate_pin"] = gate_pin
        self._w("%s/state.json" % exp, json.dumps(st, sort_keys=True), mtime)

    def _put_fixture(self, exp):
        """The pinned fixture files, byte for byte, and a finished state.json older
        than the report (report 'current')."""
        for n in ("exp.json", "build_summary.json", "ledger.jsonl"):
            if (FIX / exp / n).is_file():
                self._w("%s/%s" % (exp, n), _fix("%s/%s" % (exp, n)))
        self._state(exp, True, 99, mtime=T0 - 7200)
        self._w("%s/report.json" % exp, _fix("%s/report.json" % exp), mtime=T0 - 3600)

    def _put_step1(self):
        s1 = step1_files()
        self.step1 = s1
        for n in ("select_summary.json", "admit_summary.json", "increment_pool.jsonl", "base_selected.jsonl"):
            self._w("step1/" + n, s1[n])
        p = self.inc / "step1" / "select_clusters.csv"
        with open(str(p), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["key", "status", "species_boxes", "other_boxes"])
            wr.writeheader()
            wr.writerows(s1["csv"])

    def build(self, exp="pilot_v3", n_ledger=12):
        """The build job ran: the experiment exists and runs (driver init)."""
        f = child_files(exp, self.outcome, self.cost_scale)
        self._w(exp + "/exp.json", f["exp.json"])
        self._w(exp + "/build_summary.json", f["build_summary.json"])
        self._ledger(exp, f["ledger"][:n_ledger])
        pin = [json.loads(x) for x in f["ledger"] if json.loads(x).get("type") == "gate_pin"][0]
        self._state(exp, False, 1, mtime=self.t[0] - 60, gate_pin={"block": pin["block"], "config": pin["config"]})
        self._provenance(exp, "advanced")
        self.squeue = [j for j in self.squeue if j["name"] != "inc_build_%s" % exp]
        self.squeue.append({"id": "7999_0", "name": "inc_%s_0001" % exp, "state": "RUNNING"})

    def finish(self, exp="pilot_v3"):
        f = child_files(exp, self.outcome, self.cost_scale)
        self._ledger(exp, f["ledger"])
        pin = [json.loads(x) for x in f["ledger"] if json.loads(x).get("type") == "gate_pin"][0]
        self._state(exp, True, 40, mtime=self.t[0] - 30, gate_pin={"block": pin["block"], "config": pin["config"]})
        self.reports[exp] = f["report"]
        rp = self.inc / exp / "report.json"
        if rp.exists():
            rp.unlink()
        self.squeue = [j for j in self.squeue if not j["name"].startswith("inc_%s_" % exp)]

    def relevance_done(self):
        """The relevance job left the queue and wrote relevance.json for this select build."""
        self.squeue = [j for j in self.squeue if j["name"] != "inc_relevance"]
        self._w("step1/relevance.json", self.step1["relevance"])

    def _provenance(self, exp, status, refusal=None):
        rec = {"exp": exp, "attempts": [{"job_id": str(self.next_job), "status": status, "refusal": refusal,
                                         "started_utc": "x"}]}
        self._w("_campaign/provenance/%s.json" % exp, json.dumps(rec))

    # ---- slurm_sh
    def __call__(self, script, timeout=60):
        self.tick_calls[-1].append(script)
        out = ["Welcome to Bridges-2"]
        for m in SEG_RE.finditer(script):
            i, argv = int(m.group(1)), shlex.split(m.group(2))
            out.append("INCAP_SEG %d" % i)
            rec = self._answer(argv)
            out.append("INCAP " + json.dumps(rec, default=str))
            out.append("INCAP_SEG_END %d %d" % (i, 0 if rec.get("ok") else 1))
        return {"ok": True, "stdout": "\n".join(out), "stderr": "", "returncode": 0}

    def _answer(self, argv):
        args = argv[4:]                       # python -u -m MODULE ARGS
        self.verbs.append(args)
        if args[0] == "submit":
            return self._submit(args)
        self._activate()
        return R.dispatch(args)

    def _submit(self, argv):
        self.submits.append(argv)
        self.next_job += 1
        rec = {"verb": "submit", "ok": True, "argv": argv, "job_id": str(self.next_job)}
        args = argv[argv.index("--") + 1:]
        if "--exp" in args:
            exp = args[args.index("--exp") + 1]
            self.squeue.append({"id": str(self.next_job), "name": "inc_build_%s" % exp, "state": "PENDING"})
            self._provenance(exp, "running")
        elif argv[1] == "relevance":
            self.squeue.append({"id": str(self.next_job), "name": "inc_relevance", "state": "PENDING"})
        elif argv[1] == "audit":
            self.squeue.append({"id": str(self.next_job), "name": "inc_audit", "state": "PENDING"})
        return rec

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
    def tick(self):
        """One campaign tick through the fake cluster; every tick attempts at most
        one ssh (a second call the ticker's budget refused is a failure)."""
        self.tick_calls.append([])
        out = C.tick(slurm_sh=self, cfg_hooks=self.hooks, log=self.log, clock=self.clock,
                     lab_repo=str(self.lab), resources=RESOURCES)
        n = len(self.tick_calls)
        if (out or {}).get("ssh_refused") or len(self.tick_calls[-1]) > 1:
            check("tick %d of %s made at most one ssh" % (n, self.dir.name), False, out)
        if (out or {}).get("ok") is False:
            check("tick %d of %s ran" % (n, self.dir.name), False, out)
        err = ((out or {}).get("campaigns") or {}).get(NAME, {}).get("error")
        if err:
            check("tick %d of %s raised nothing in the campaign" % (n, self.dir.name), False, err)
        self.t[0] += TICK
        return out

    def state(self):
        return C.load_state(C.Paths(str(self.lab)), NAME)

    def ledger(self):
        p = C.Paths(str(self.lab)).ledger
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def events(self, name, **match):
        return [r for r in self.ledger() if r.get("event") == name
                and all(r.get(k) == v for k, v in match.items())]

    def diagnoses(self):
        return json.loads(C.Paths(str(self.lab)).diagnoses(NAME).read_text())

    def approvals(self):
        return AP.state("weed", root=str(self.lab))

    def config(self):
        return (self.hooks[0]().get("campaigns") or {}).get(NAME) or {}

    def replay_pass(self):
        return X.record_replay_result("pass", dict(REPLAY_PASS), ctx=self.xctx)

    def item(self):
        return ((self.state() or {}).get("item") or {}).get("proposal") or {}

    def builds(self, exp=None):
        out = []
        for a in self.submits:
            args = a[a.index("--") + 1:]
            if a[1] == "build" and (exp is None or ("--exp" in args and args[args.index("--exp") + 1] == exp)):
                out.append(args)
        return out


def one_ssh_everywhere(w):
    return max([len(c) for c in w.tick_calls] or [0]) <= 1


def run_to_pilot_v3(w, log_checks=True):
    """Ticks from pilot_v2's finished record to pilot_v3 finished, reported and
    diagnosed. Returns nothing; the scenario checks what it needs."""
    w.tick()                                              # snapshot: pilot_v2 -> D15 -> L9 at GATE
    if log_checks:
        st = w.state()
        by = {d["id"]: d for d in w.diagnoses()["diagnoses"]}
        d15, d1 = by["D15"], by["D1"]
        pairs = sorted((p["chain"], p["step"], p["line"]) for p in d15["detail"]["pairs"])
        check("tick 1: pilot_v2 reported and diagnosed: D15 fires with the five flips-only REJECTs as cites",
              d15["fired"] and pairs == [("freeze", "I2", 15), ("freeze", "I3", 22), ("full", "I2", 13),
                                         ("full", "I3", 19), ("lora", "I3", 23)]
              and {13, 15, 19, 22, 23} <= {c.get("line") for c in d15["cites"]}, pairs)
        check("tick 1: D1 fires unblocked with X1 (freeze I5 failed the regression guard too, lora I2 and I5 the "
              "species guard)", d1["fired"] and d1["levers"] == ["X1"] and "blocked_by" not in d1["detail"]
              and not d15["detail"]["blocks_d1"], d1["levers"])
        check("tick 1: the item at GATE is L9 with exactly the pre-registered command",
              st["phase"] == "GATE" and w.item().get("lever") == "L9" and w.item().get("argv") == L9_ARGV,
              (st["phase"], w.item().get("lever"), w.item().get("argv")))
        check("tick 1: L9 outranks D2's L3 and D3's L4 (recorded as not taken)",
              {r.get("lever") for r in w.events("not_taken")} >= {"L3", "L4"},
              [(r.get("lever"), r.get("reasons")) for r in w.events("not_taken")])
        check("tick 1: X1 (D1) and X4 (D3b) are recorded as R4 cards, none of them queued",
              {"X1", "X4"} <= {c["lever"] for c in st.get("cards") or []}
              and not [i for i in w.approvals().values() if i.get("risk") == "R4"],
              [c["lever"] for c in st.get("cards") or []])
        diag = w.events("diagnosed")
        check("tick 1: the campaign ledger records D15 with its cites",
              diag and any(t["id"] == "D15" and t["cites"] for t in diag[-1]["trigger"]))
    w.tick()                                              # no replay pass: filed, waits for a person
    if log_checks:
        it = w.state().get("item") or {}
        check("tick 2: without a replay pass the L9 build waits for a person (filed, nothing run, no ssh)",
              it.get("status") == "filed" and not w.submits and not w.tick_calls[-1]
              and any("replay tests" in r for r in it.get("last_reasons") or []), it.get("last_reasons"))
    w.replay_pass()
    w.tick()                                              # envelope grant: executed
    if log_checks:
        items = [i for i in w.approvals().values() if i.get("action") == "inc_build_pilot"]
        ex = [r for r in X.executions(w.xctx) if r.get("action") == "inc_build_pilot" and r.get("status") == "executed"]
        check("tick 3: with a replay pass, envelope autonomy runs L9: one build submission, exactly its command",
              len(w.builds()) == 1 and w.builds()[0] == ["pilot", "build", "--exp", "pilot_v3", "--replay-mode",
                                                          "full", "--gate-flips-mode", "net"], w.submits)
        check("tick 3: the grant is the autopilot's (basis envelope), run as the person who enabled it",
              items and items[0]["status"] == "approved" and items[0].get("decision_basis") == "envelope"
              and items[0]["decided_by"] == AUTO and ex and ex[0]["authorized_as"] == OWNER
              and ex[0]["lever"] == "L9", (items and items[0].get("decision_basis"), ex[:1]))
        st = w.state()
        check("tick 3: RUN on pilot_v3, building", st["phase"] == "RUN" and st["exp"] == "pilot_v3"
              and (st.get("building") or {}).get("exp") == "pilot_v3", (st["phase"], st["exp"]))
    w.tick()                                              # building: waits
    w.build("pilot_v3")
    w.tick()                                              # built: RUN advances
    if log_checks:
        check("pilot_v3 built and advanced, still one build submission",
              w.events("built", child_exp="pilot_v3") and "pilot_v3" in w.advances and len(w.builds()) == 1)
    w.tick()
    w.finish("pilot_v3")
    w.tick()                                              # done: REPORT -> DIAGNOSE


def section_cycle():
    print("the cycle: pilot_v2 -> D15 -> L9 -> pilot_v3 (v2 supported) -> L3 -> L2 on the v2 gate")
    w = World("cycle")
    run_to_pilot_v3(w)
    st = w.state()
    rep = [r for r in w.events("reported") if r["exp"] == "pilot_v3"]
    check("pilot_v3 finished: reported with its spend charged (launched by the campaign)",
          rep and (rep[0].get("spend") or {}).get("ok"), rep)
    out = [r for r in w.events("outcome") if r.get("child_exp") == "pilot_v3"]
    check("the L9 outcome is scored against its prediction (agreement up)",
          out and out[0]["lever"] == "L9" and out[0]["verdict"] in ("better", "worse", "within_noise",
                                                                     "insufficient"), out)
    by = {d["id"]: d for d in w.diagnoses()["diagnoses"]}
    check("DIAGNOSE on pilot_v3: D16 fires info (v2 supported), no lever",
          by["D16"]["fired"] and by["D16"]["severity"] == "info" and by["D16"]["levers"] == []
          and by["D16"]["detail"]["outcome"] == "supported", by["D16"]["summary"])
    check("DIAGNOSE on pilot_v3: D4 decision_slot_ready, full 6/7, gate net, v2 supported",
          by["D4"]["fired"] and by["D4"]["name"] == "decision_slot_ready"
          and by["D4"]["detail"]["recipes"] == ["full"] and by["D4"]["detail"]["rates"]["full"] == "6/7"
          and by["D4"]["detail"]["gate"] == {"flips_mode": "net", "v2_check": "supported"}, by["D4"]["summary"])
    check("DIAGNOSE on pilot_v3: D1 and D15 silent", not by["D1"]["fired"] and not by["D15"]["fired"])
    pro = w.events("prospective_d4", exp="pilot_v3")
    path = C.Paths(str(w.lab)).replay / C.DG.prospective_name("pilot_v3", C.DG.rules_version())
    rec = json.loads(path.read_text()) if path.is_file() else {}
    check("prospective_d4 on pilot_v3 is written (ready, full replay, recipe full, gate net, v2 supported), its "
          "sha256 in the ledger",
          pro and rec.get("ready") is True and rec.get("replay_mode") == "full" and rec.get("recipes") == ["full"]
          and rec.get("gate_flips_mode") == "net" and rec.get("v2_check") == "supported"
          and pro[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), rec)
    check("D2 fires on Step 1 without relevance.json (the off-domain source named); the item is L3",
          by["D2"]["fired"] and OFF_DOMAIN in [s["source"] for s in by["D2"]["detail"]["sources"]]
          and w.item().get("lever") == "L3" and st["phase"] == "GATE", (by["D2"]["summary"], w.item().get("lever")))
    check("no L2 is proposed while relevance.json is missing",
          not [r for r in w.events("proposed") if r.get("lever") == "L2"])
    w.tick()
    st = w.state()
    rel_runs = [a for a in w.submits if a[1] == "relevance"]
    check("L3 (R2) runs directly, once: WAIT_JOB on the relevance job",
          len(rel_runs) == 1 and st["phase"] == "WAIT_JOB" and st.get("wait_jobs"), (st["phase"], w.submits))
    w.tick()
    check("the relevance job is still queued: still WAIT_JOB", w.state()["phase"] == "WAIT_JOB")
    w.relevance_done()
    w.tick()
    st = w.state()
    it = w.item()
    by = {d["id"]: d for d in w.diagnoses()["diagnoses"]}
    want = ["python", "-m", "weed_optimizer_framework.tools.inc.realloop", "build", "--exp", "realloop_v1",
            "--base", OCEAN_INC + "/step1/base_B.jsonl", "--replay-mode", "full", "--recipes", "full",
            "--relevance", OCEAN_INC + "/step1/relevance.json", "--gate-flips-mode", "net"]
    check("relevance present: D2 silent, and the item is L2 with --relevance, --gate-flips-mode net and D4's "
          "replay mode and recipe", not by["D2"]["fired"] and it.get("lever") == "L2" and it.get("argv") == want,
          (by["D2"]["summary"], it.get("lever"), it.get("argv")))
    first_l2 = [i for i, r in enumerate(w.ledger()) if r.get("event") == "proposed" and r.get("lever") == "L2"]
    check("the prospective record of pilot_v3 precedes the first L2 in the campaign ledger",
          first_l2 and w.ledger().index(pro[0]) < first_l2[0], (first_l2, pro))
    w.tick()
    st = w.state()
    loops = w.builds("realloop_v1")
    ex = [r for r in X.executions(w.xctx) if r.get("action") == "inc_build_realloop" and r.get("status") == "executed"]
    check("L2 runs under the envelope, once, carrying the v2 gate: RUN realloop_v1",
          len(loops) == 1 and loops[0][-2:] == ["--gate-flips-mode", "net"] and ex and ex[0]["lever"] == "L2"
          and ex[0]["authorized_as"] == OWNER and st["phase"] == "RUN" and st["exp"] == "realloop_v1",
          (loops, st["phase"], st["exp"]))
    for _ in range(2):
        w.tick()
    check("more ticks: one L9 and one L2 submission in all", len(w.builds("pilot_v3")) == 1
          and len(w.builds("realloop_v1")) == 1, w.submits)
    check("the lineage: L9 on pilot_v2 -> pilot_v3, L2 on pilot_v3 -> realloop_v1",
          [(r["lever"], r["parent_exp"], r["child_exp"]) for r in w.events("executed") if r.get("lever") in
           ("L9", "L2")] == [("L9", "pilot_v2", "pilot_v3"), ("L2", "pilot_v3", "realloop_v1")],
          [(r.get("lever"), r.get("parent_exp"), r.get("child_exp")) for r in w.events("executed")])
    check("one ssh per tick over the whole cycle", one_ssh_everywhere(w), [len(c) for c in w.tick_calls])
    keys = ("decided_by", "trigger", "parent_exp", "child_exp", "approval_id", "job_ids")
    check("every campaign-ledger entry carries its provenance fields",
          all(all(k in r for k in keys) for r in w.ledger()))


def section_same_tick():
    print("the prospective record and L2 in one DIAGNOSE: relevance.json is there before pilot_v3 finishes")
    w = World("same_tick")
    w.relevance_done()                                    # Step 1 already has its relevance.json
    run_to_pilot_v3(w, log_checks=False)
    led = w.ledger()
    pro = [i for i, r in enumerate(led) if r.get("event") == "prospective_d4" and r.get("exp") == "pilot_v3"]
    l2 = [i for i, r in enumerate(led) if r.get("event") == "proposed" and r.get("lever") == "L2"]
    diag = [i for i, r in enumerate(led) if r.get("event") == "diagnosed" and r.get("exp") == "pilot_v3"]
    check("pilot_v3 is diagnosed once; the record and the first L2 proposal are written in that same tick",
          len(pro) == 1 and l2 and diag and led[pro[0]]["tick"] == led[l2[0]]["tick"] == led[diag[-1]]["tick"],
          [(led[i]["event"], led[i]["tick"]) for i in pro + l2[:1] + diag[-1:]])
    check("... and the record comes first in the ledger", pro and l2 and pro[0] < l2[0], (pro, l2))
    check("the L2 is D4's (full replay, recipe full), with --relevance and --gate-flips-mode net",
          l2 and led[l2[0]]["argv"][-6:] == ["--recipes", "full", "--relevance", OCEAN_INC + "/step1/relevance.json",
                                             "--gate-flips-mode", "net"], l2 and led[l2[0]]["argv"])
    check("L3 is never proposed (relevance.json matches the select build)",
          not [r for r in led if r.get("event") == "proposed" and r.get("lever") == "L3"])
    check("one ssh per tick", one_ssh_everywhere(w), [len(c) for c in w.tick_calls])


def section_refuted():
    print("the refuted branch: v2_check refuted -> D16 crit, card X7, no L2")
    w = World("refuted", outcome="refuted")
    run_to_pilot_v3(w, log_checks=False)
    check("L9 ran once and pilot_v3 finished", len(w.builds("pilot_v3")) == 1
          and w.events("reported", exp="pilot_v3"))
    by = {d["id"]: d for d in w.diagnoses()["diagnoses"]}
    check("D16 fires crit (refuted by lora's Breal ACCEPT), lever X7",
          by["D16"]["fired"] and by["D16"]["severity"] == "crit" and by["D16"]["levers"] == ["X7"]
          and "Breal" in by["D16"]["summary"], by["D16"]["summary"])
    check("D4 is blocked_by D16: no L2 lever", by["D4"]["fired"] and by["D4"]["levers"] == []
          and by["D4"]["detail"]["blocked_by"]["id"] == "D16", by["D4"]["summary"])
    st = w.state()
    check("card X7 is recorded (R4, never queued)", any(c["lever"] == "X7" and c["risk"] == "R4"
                                                          for c in st.get("cards") or [])
          and not [i for i in w.approvals().values() if i.get("risk") == "R4"])
    rec = json.loads((C.Paths(str(w.lab)).replay / C.DG.prospective_name("pilot_v3", C.DG.rules_version()))
                     .read_text())
    check("the prospective record is not ready, blocked by D16", rec["ready"] is False
          and rec["blocked_by"] == "D16" and rec["recipes"] is None, rec)
    for _ in range(3):
        w.tick()
        if w.state()["phase"] == "WAIT_JOB":
            w.relevance_done()
    for _ in range(2):
        w.tick()
    st = w.state()
    check("L3 ran (relevance is independent of the gate); no L2 was ever proposed, filed or run",
          [a[1] for a in w.submits].count("relevance") == 1
          and not [r for r in w.ledger() if r.get("lever") == "L2" and r.get("event") in ("proposed", "filed",
                                                                                          "executed")]
          and not w.builds("realloop_v1")
          and not [i for i in w.approvals().values() if i.get("action") == "inc_build_realloop"], w.submits)
    detail = (st.get("card") or {}).get("detail") or ""
    menu = LV.load_menu()
    known = set(menu["levers"]) | set(menu["cards"]) | set(menu["operations"])
    named = re.findall(r"(?:Deferred: |; )(\S+) \((D\d+b?)\): ", detail)
    check("the campaign completes with a residual card naming the withheld L2 (D16), every deferred entry a "
          "menu lever", st["phase"] == "COMPLETE" and (st.get("card") or {}).get("kind") == "residual"
          and ("L2", "D16") in named and all(lv in known for lv, _d in named), (st["phase"], named, detail))
    check("one ssh per tick", one_ssh_everywhere(w), [len(c) for c in w.tick_calls])


def section_budget():
    print("budget exhaustion: an envelope that fits L9's estimate but not what pilot_v3 cost")
    ev = E.load_dir(FIX, "pilot_v2", exps=EXPS)
    est, info, _c = LV.price("L9", {"exp": "pilot_v3", "replay_mode": "full"}, ev, "pilot_v2")
    w = World("budget", envelope_su=est + 0.5, cost_scale=1.3)
    run_to_pilot_v3(w, log_checks=False)
    check("L9 (%.3f SU) fitted the %.3f SU envelope and ran once" % (est, est + 0.5), len(w.builds("pilot_v3")) == 1)
    rep = [r for r in w.events("reported") if r["exp"] == "pilot_v3"]
    spent = (rep[0].get("spend") or {}).get("su") if rep else None
    check("pilot_v3's real spend (its runs at 1.3x the rates it was priced from) is charged and is more than the "
          "estimate's experiment_hours (%.3f SU)" % info["experiment_hours"],
          spent is not None and spent > info["experiment_hours"], (spent, info["experiment_hours"]))
    check("with its build job (%.1f SU) that is more than the %.3f SU envelope"
          % (info["build_job_hours"], est + 0.5),
          spent is not None and spent + info["build_job_hours"] > est + 0.5, (spent, info["build_job_hours"]))
    n_sub = len(w.submits)
    w.tick()
    st = w.state()
    why = (st.get("paused") or {}).get("reason") or ""
    check("the next charge is refused on budget and the campaign pauses (config and state)",
          "budget" in why and w.config().get("enabled") is False and w.config().get("paused_reason")
          and len(w.submits) == n_sub, (why, w.submits[n_sub:]))
    before = len(w.submits)
    for _ in range(2):
        w.tick()
    check("paused: later ticks make no ssh and submit nothing",
          len(w.submits) == before and all(len(c) == 0 for c in w.tick_calls[-2:]))
    check("one ssh per tick", one_ssh_everywhere(w), [len(c) for c in w.tick_calls])


def main():
    try:
        section_cycle()
        section_same_tick()
        section_refuted()
        section_budget()
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
