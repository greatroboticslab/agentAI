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
  * in the stream world, a lost lab fetch is a failed step (not a stall):
    the lane is freed and the fetch is proposed again.
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
from test_stream_ap_world import S, World, check  # noqa: E402

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


def test_world():
    print("in the stream world, a lost lab fetch is a failed step, then proposed again")
    w = World("lab_lost", floors=(5.0, 10.0))
    w.ready_r0()
    w.queue(0)
    w.placement({"hf": "pass", "ftp": "fail"})
    w.candidates([{"id": "gh_lost", "provider": "github", "licence": "MIT", "target_classes": ["Purslane"],
                   "bytes": 1e9, "expected_target_boxes": 900}])
    w.tick(2)
    first = [x for x in w.runner.launched if "fetch" in x["argv"] and "gh_lost" in x["argv"]]
    check("the lab fetch is launched", len(first) == 1, [x["argv"][-6:] for x in w.runner.launched])
    w.runner.results[first[0]["job"]] = {"ok": False, "rc": None, "lost": True, "job": first[0]["job"],
                                         "error": "the lab process of %s is gone and wrote no result" % first[0]["job"]}
    w.tick(3)
    again = [x for x in w.runner.launched if "fetch" in x["argv"] and "gh_lost" in x["argv"]]
    failed = [e for e in w.events("failed") if e.get("lever") == "L16L"]
    check("the lost fetch is recorded failed (no stall) and the fetch is launched again",
          failed and "gone" in str(failed[-1].get("reasons")) and len(again) == 2
          and not w.lane("DATA").get("hold"), (len(again), failed[-1:] if failed else None, w.lane("DATA").get("hold")))


def main():
    try:
        test_real_processes()
        test_world()
    finally:
        import shutil
        shutil.rmtree(ROOT, ignore_errors=True)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
