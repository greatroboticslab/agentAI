#!/usr/bin/env python3
"""A lab job killed from outside is noticed, not followed for ever.

Live, 2026-10-03: a deploy restarted the dashboard (KillMode=control-group),
which killed the detached lab-run of the zenodo_15808623 fetch. LabRunner.poll
read only the result file, which a killed lab-run never writes, so the DATA
lane's L16L item stayed 'running' with nothing behind it and blocked the
lane. poll now returns a failure ('lost') when no lab-run of the spec is
alive and no result was written; launch records the child's pid.

Pinned, with real processes (no mocks of the process table):
  * a running job polls None (alive), and its spec records the pid;
  * the same job killed (its whole session) polls a lost failure;
  * a job that finished normally polls its own result, never 'lost';
  * a spec written before pids were recorded, with no process naming it,
    polls lost; with a live process naming it, polls None;
  * in the stream world, a lost lab fetch frees the lane and is proposed
    again; it counts neither as the source's failed attempt nor toward the
    lane's stop-loss, until the same step is lost LOST_RUNS_MAX times in a row
    (then it counts as a failure, so a job that is always killed still ends);
  * a person reopens a closed source (`stream reopen`): the next tick makes
    it a candidate with its failures reset, once per stamp, recorded with who
    and why; a stamp older than the closure does nothing; a reopen needs a
    person and a reason. (2026-10-03: the third 'failed attempt' that closed
    zenodo_15808623 was the restart's kill.)
"""
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import NAME, OWNER, S, World, check  # noqa: E402

ROOT = pathlib.Path(tempfile.mkdtemp(prefix="stream_ap_lab_lost_"))
REPO = str(pathlib.Path(__file__).resolve().parents[1])


def wait(pred, secs=20.0):
    t0 = time.time()
    while time.time() - t0 < secs:
        if pred():
            return True
        time.sleep(0.2)
    return False


def test_real_processes():
    print("LabRunner.poll with real processes")
    run = S.LabRunner(ROOT / "jobs", cwd=REPO)
    run.launch("sleeper", [sys.executable, "-c", "import time; time.sleep(120)"], timeout=600)
    spec = S._read_json(ROOT / "jobs" / "sleeper.spec.json")
    check("launch records the lab-run's pid in the spec", isinstance(spec.get("pid"), int), spec)
    check("a running job polls None", wait(lambda: S._cmdline(spec["pid"]) not in (None, "")) and
          run.poll("sleeper") is None, run.poll("sleeper"))
    os.killpg(spec["pid"], signal.SIGKILL)
    check("killed from outside (its whole session), it polls a lost failure with no result file",
          wait(lambda: (run.poll("sleeper") or {}).get("lost") is True)
          and not (ROOT / "jobs" / "sleeper.result.json").exists(), run.poll("sleeper"))
    res = run.poll("sleeper")
    check("  the failure says the process is gone", res.get("ok") is False and "gone" in res.get("error", ""), res)

    run.launch("quick", [sys.executable, "-c", "print('hi')"], timeout=60)
    check("a job that finishes normally polls its own result (ok, rc 0), never lost",
          wait(lambda: run.poll("quick") is not None) and run.poll("quick").get("ok") is True
          and not run.poll("quick").get("lost"), run.poll("quick"))

    legacy = ROOT / "jobs" / "legacy.spec.json"
    S._write_json(legacy, {"job": "legacy", "argv": ["true"], "result": str(ROOT / "jobs" / "legacy.result.json"),
                           "timeout": 60, "cwd": REPO})
    check("a spec with no pid and no process naming it polls lost", (run.poll("legacy") or {}).get("lost") is True,
          run.poll("legacy"))
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "lab-run", "--spec", str(legacy)],
                              start_new_session=True)
    try:
        check("  with a live lab-run process naming that spec, it polls None",
              wait(lambda: run.poll("legacy") is None), run.poll("legacy"))
    finally:
        holder.kill()
        holder.wait()


CAND = {"id": "gh_lost", "provider": "github", "licence": "MIT", "target_classes": ["Purslane"],
        "bytes": 1e9, "expected_target_boxes": 900}


def fetches(w):
    return [x for x in w.runner.launched if "fetch" in x["argv"] and "gh_lost" in x["argv"]]


def lose_last(w):
    job = fetches(w)[-1]["job"]
    w.runner.results[job] = {"ok": False, "rc": None, "lost": True, "job": job,
                             "error": "the lab process of %s is gone and wrote no result" % job}


def test_world():
    print("in the stream world, a lost lab fetch is proposed again and charged to nobody")
    w = World("lab_lost", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.placement({"hf": "pass", "ftp": "fail"})
    w.candidates([dict(CAND)])
    w.tick(2)
    check("the lab fetch is launched", len(fetches(w)) == 1, [x["argv"][-6:] for x in w.runner.launched])
    lose_last(w)
    w.tick(3)
    src = ((w.state() or {}).get("sources") or {}).get("gh_lost") or {}
    lost = [e for e in w.events("failed") if e.get("lever") == "L16L" and e.get("lost")]
    check("the lost fetch is recorded (lost) and the fetch is launched again",
          lost and "gone" in str(lost[-1].get("reasons")) and len(fetches(w)) == 2,
          (len(fetches(w)), lost[-1:] if lost else None))
    check("  it is not the source's failed attempt, nor a step toward the lane's stop-loss, nor a collection "
          "attempt toward the 4th-attempt pause (the retry's own start counts 1)",
          int(src.get("failures") or 0) == 0 and int(w.lane("DATA").get("fails") or 0) == 0
          and not w.lane("DATA").get("hold") and int(src.get("attempts") or 0) == 1
          and not (w.state() or {}).get("paused"), (src, w.lane("DATA")))
    lose_last(w)
    w.tick(3)
    check("a second loss in a row is still charged to nobody, and the fetch runs a third time",
          len(fetches(w)) == 3 and int((((w.state() or {}).get("sources") or {}).get("gh_lost") or {})
                                       .get("failures") or 0) == 0, len(fetches(w)))
    lose_last(w)
    w.tick(3)
    src = ((w.state() or {}).get("sources") or {}).get("gh_lost") or {}
    check("the third loss of the same step in a row counts as a failure (a job that is always killed still ends)",
          int(src.get("failures") or 0) == 1 and int(w.lane("DATA").get("fails") or 0) >= 1,
          (src, w.lane("DATA").get("fails")))


def test_reopen():
    print("a person reopens a closed source")
    w = World("reopen_src", floors=(5.0, 10.0))
    w.ready_r0()
    w.tick(1)
    paths = S.StreamPaths(str(w.lab), w.domain)
    st = S._read_json(paths.state(NAME))
    st.setdefault("sources", {})["gh_closed"] = {"status": "closed", "failures": 3, "attempts": 3,
                                                 "closed_reason": "3 failed attempts (7.5)",
                                                 "closed_utc": W.utc(w.clock())}
    S._write_json(paths.state(NAME), st)
    for kw, frag in ((dict(by="platform"), "person"), (dict(by=OWNER, why=" "), "why"),
                     (dict(by=OWNER, why="x", src="../etc"), "source id")):
        try:
            S.configure_stream(NAME, kw["by"], cfg_hooks=w.hooks, lab_repo=str(w.lab),
                               reopen_source=kw.get("src", "gh_closed"), reopen_why=kw.get("why", "ok"))
            got = None
        except ValueError as e:
            got = str(e)
        check("  a reopen is refused without %s" % frag, got is not None and frag in got, got)
    w.advance(60)
    S.configure_stream(NAME, OWNER, cfg_hooks=w.hooks, lab_repo=str(w.lab), reopen_source="gh_closed",
                       reopen_why="its third failure was a restart's kill", clock=w.clock)
    w.tick(1)
    src = ((w.state() or {}).get("sources") or {}).get("gh_closed") or {}
    ev = [e for e in w.events("source_reopened") if e.get("source") == "gh_closed"]
    check("the next tick makes it a candidate with failures and attempts reset, recorded with who and why",
          src.get("status") == "candidate" and src.get("failures") == 0 and src.get("attempts") == 0 and ev
          and ev[-1].get("decided_by") == OWNER and "restart" in str(ev[-1].get("reasons")), (src, ev))
    w.tick(2)
    check("  once per stamp", len([e for e in w.events("source_reopened") if e.get("source") == "gh_closed"]) == 1)
    st = S._read_json(paths.state(NAME))
    st["sources"]["gh_closed"].update(status="closed", failures=3, closed_utc=W.utc(w.clock() + 3600))
    S._write_json(paths.state(NAME), st)
    w.tick(1)
    check("a stamp older than a later closure does nothing",
          (((w.state() or {}).get("sources") or {}).get("gh_closed") or {}).get("status") == "closed")


def main():
    try:
        test_real_processes()
        test_world()
        test_reopen()
    finally:
        import shutil
        shutil.rmtree(ROOT, ignore_errors=True)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
