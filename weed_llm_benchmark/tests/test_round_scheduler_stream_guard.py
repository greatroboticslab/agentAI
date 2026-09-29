#!/usr/bin/env python3
"""The old paths are closed while a stream campaign owns a domain
(docs/CONTINUOUS_LOOP.md 8, "No mega_trainer training" and "The old harvest
cannot feed the pool"; scenario S28).

  * round_scheduler.old_path_refusal refuses the domain's collect, filter and
    train steps while a campaigns.<name> block with mode "stream" names the
    domain (enabled or paused); another domain, or no stream campaign, is
    untouched;
  * round_scheduler._advance (the real function, with a fake round ledger and
    no cluster) records the step as refused, pauses the domain and submits
    nothing;
  * the dashboard's harvest route is given the same check with AUTO_SYNC=1
    (old_path_refusal(..., auto_sync=True)); wiring it into dashboard_server.py
    is outside group F's files (see the build notes);
  * roboflow_sync.cmd_sync_newest_slugs, the real function, skips a registry
    entry with status "intake" (it takes only status "downloaded"), so an
    intake source never reaches a public Roboflow project; it imports no
    roboflow client when nothing is pending.

Run:  python3 tests/test_round_scheduler_stream_guard.py
"""
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAILURES = []


def _check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1200]))
        FAILURES.append(name)
    return bool(cond)


class _Log(object):
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(m)

    warning = error = info


class _DB(object):
    ROUND_STEPS = ["collect", "label", "filter", "train", "eval"]

    def get_current_round(self, domain):
        return {"round_num": 7, "steps": {}}

    def get_domain_config(self, domain):
        return {}

    def step_fields(self, dcfg, step):
        return []


def run_checks(check=_check):
    from weed_optimizer_framework.tools import round_scheduler as RS
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="rs_stream_guard_"))
    old_file, old_ctx = RS._CFG_FILE, dict(RS._CTX)
    ok = True
    try:
        RS._CFG_FILE = str(tmp / "round_scheduler.json")
        cfg = {"domains": {"weed": {"enabled": True}, "corn": {"enabled": True}},
               "campaigns": {"weed_inc_v1": {"mode": "experiment"},
                             "weed_stream_v1": {"mode": "stream", "domain": "weed", "enabled": False,
                                                "paused_reason": "paused by a person"}}}
        pathlib.Path(RS._CFG_FILE).write_text(json.dumps(cfg))
        for step in ("collect", "filter", "train"):
            ok &= check("a stream campaign owns weed (even paused): the old %s step is refused" % step,
                        "weed_stream_v1" in RS.old_path_refusal("weed", step))
        ok &= check("  another domain is untouched", RS.old_path_refusal("corn", "train") == "")
        ok &= check("  an experiment-mode campaign owns nothing", RS.stream_owner("weed", {"campaigns": {
            "weed_inc_v1": {"mode": "experiment", "domain": "weed"}}}) is None)
        ok &= check("the dashboard's harvest with AUTO_SYNC=1 gets a refusal to show (public Roboflow push)",
                    "AUTO_SYNC=1" in RS.old_path_refusal("weed", "harvest", auto_sync=True))
        ok &= check("  and a harvest without it outside the closed steps is not refused",
                    RS.old_path_refusal("weed", "harvest") == "")
        # the real _advance
        recs, submits, log = [], [], _Log()
        RS._CTX.clear()
        RS._CTX.update({"db": _DB(), "log": log, "slurm_sh": lambda *a, **k: {"ok": False},
                        "record_step": lambda domain, step, status, **kw: recs.append((domain, step, status, kw))
                        or {"ok": True}})
        RS._STATE.clear()
        real_submit = RS._submit
        RS._submit = lambda cmd: submits.append(cmd) or ("123", "Submitted batch job 123")
        try:
            dcfg = {"enabled": True}
            RS._advance("weed", dcfg)
        finally:
            RS._submit = real_submit
        ok &= check("_advance on a stream-owned domain: the next step (collect) is recorded refused",
                    recs and recs[-1][1] == "collect" and recs[-1][2] == "refused"
                    and "stream campaign" in str(recs[-1][3].get("detail")), recs)
        ok &= check("  nothing is submitted", not submits, submits)
        saved = json.loads(pathlib.Path(RS._CFG_FILE).read_text())
        ok &= check("  and the domain is paused with the reason on disk",
                    saved["domains"]["weed"].get("enabled") is False
                    and "stream campaign" in str(saved["domains"]["weed"].get("paused_reason")), saved["domains"]["weed"])
        cfg["campaigns"].pop("weed_stream_v1")
        pathlib.Path(RS._CFG_FILE).write_text(json.dumps(cfg))
        ok &= check("with no stream campaign the old path is not refused by this guard",
                    RS.old_path_refusal("weed", "collect") == "")
    finally:
        RS._CFG_FILE = old_file
        RS._CTX.clear()
        RS._CTX.update(old_ctx)
        RS._STATE.clear()
    # roboflow_sync's real function skips status "intake"
    from weed_optimizer_framework.tools import roboflow_sync as RF
    repo = tmp / "repo"
    (repo / "results" / "framework").mkdir(parents=True)
    ds_dir = tmp / "downloads" / "stream_intake_src"
    ds_dir.mkdir(parents=True)
    (repo / "results" / "framework" / "dataset_registry.json").write_text(json.dumps({"datasets": {
        "stream_intake_src": {"status": "intake", "annotation": "intake_v1", "local_path": str(ds_dir),
                              "class_names": ["Purslane"], "domain": "weed"}}}))
    old_env = os.environ.get("REPO_ROOT")
    os.environ["REPO_ROOT"] = str(repo)
    had_rf = "roboflow" in sys.modules
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            RF.cmd_sync_newest_slugs(types.SimpleNamespace(project=None, folder=None, cap_per_slug=0))
    finally:
        if old_env is None:
            os.environ.pop("REPO_ROOT", None)
        else:
            os.environ["REPO_ROOT"] = old_env
    out = buf.getvalue()
    ok &= check("roboflow_sync.cmd_sync_newest_slugs (real) skips an 'intake' registry entry: pending sync 0",
                "pending sync: 0" in out, out[-400:])
    ok &= check("  and imports no Roboflow client", had_rf or "roboflow" not in sys.modules)
    return bool(ok)


def main():
    print("round scheduler: the stream guard")
    run_checks(_check)
    print("\n%d failure(s), 0 skipped: none" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
