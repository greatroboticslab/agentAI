#!/usr/bin/env python3
"""The supervision layer has to say when it is in a state whose output cannot be reported.

Five failures, each of which happened in this project's own record and none of
which raised anything at the time:

  * nine campaign reviews produced by a 4.7 GB model on the lab's 3060, every
    record looking exactly like a correctly-wired shadow reviewer;
  * a completed step with no review after it, found by hand two weeks later;
  * verdicts answering a rubric that had since been edited;
  * a benchmark arm standing on 78 of 149 cases read as though it stood on 149;
  * verdicts written after the last scoring run, so the published numbers
    described a corpus that had moved.

Run:  python tests/test_supervision_health.py
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import supervision_health as SH  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def _brain(tmp, rec):
    d = os.path.join(tmp, "_brain", "weed")
    os.makedirs(os.path.join(d, "reviews"), exist_ok=True)
    if rec is not None:
        for p in (os.path.join(d, "latest_review.json"),
                  os.path.join(d, "reviews", "1_r15_train.json")):
            with open(p, "w") as fh:
                json.dump(rec, fh)
    return d


def _bench(tmp, arms, dev_n=4, rubric="r1", score_after=True):
    root = os.path.join(tmp, "supervision_bench")
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    with open(os.path.join(root, "split.json"), "w") as fh:
        json.dump({"dev": ["c%d" % i for i in range(dev_n)], "test": []}, fh)
    for arm, (n, rub) in arms.items():
        d = os.path.join(root, "verdicts", arm)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            with open(os.path.join(d, "c%d_r0.json" % i), "w") as fh:
                json.dump({"case_id": "c%d" % i, "rubric_sha256": rub}, fh)
    p = os.path.join(root, "results", "run.json")
    with open(p, "w") as fh:
        json.dump({}, fh)
    if score_after:
        os.utime(p, (2 ** 31, 2 ** 31))          # scored long after every verdict
    else:
        os.utime(p, (1, 1))                       # scored long before them
    return root


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="suphealth"))
    try:
        LAB = {"ts": 1000.0, "model": "qwen2.5-coder:7b",
               "endpoint": "http://127.0.0.1:11434/v1", "tier": "fast",
               "place": "lab", "authoritative": False, "mode": "shadow",
               "applied": False,
               "not_authoritative_why": "endpoint is the lab loopback"}
        CLU = dict(LAB, model="qwen3.8:27b", endpoint="http://gpu012:8100/v1",
                   place="cluster", authoritative=True, not_authoritative_why="")

        # 1. the 2026-09-04 wiring
        v = SH.verdict(brain_dir=_brain(str(tmp / "a"), LAB), now=1100.0,
                       review_enabled=True)
        check("a lab-placed reviewer is crit", v["level"] == "crit", v["reason"])
        check("and the reason names the endpoint", "127.0.0.1" in v["reason"])

        # 2. a record from before the fields existed claims nothing
        old = {"ts": 1000.0, "model": "x", "endpoint": "y", "applied": False}
        v = SH.verdict(brain_dir=_brain(str(tmp / "b"), old), now=1100.0,
                       review_enabled=True)
        check("a pre-field record is not treated as authoritative",
              v["level"] == "crit")
        check("and says the record predates the fields",
              "predates" in v["reason"])

        # 3. correctly wired
        v = SH.verdict(brain_dir=_brain(str(tmp / "c"), CLU), now=1100.0,
                       review_enabled=True, last_step_ts=1050.0)
        check("a cluster-placed, current reviewer is ok", v["level"] == "ok",
              v["reason"])

        # 4. the reviewer switched off is a normal state, not an alarm
        v = SH.verdict(brain_dir=_brain(str(tmp / "d"), None), now=1100.0,
                       review_enabled=False)
        check("a reviewer that is off is ok", v["level"] == "ok", v["reason"])

        # 5. a step finished and nothing reviewed it
        v = SH.verdict(brain_dir=_brain(str(tmp / "e"), CLU), now=99000.0,
                       review_enabled=True, last_step_ts=90000.0)
        check("a step with no review after it is crit", v["level"] == "crit",
              v["reason"])
        check("and the reason says so", "nothing followed" in v["reason"])

        # 6. a partial arm cannot read as complete
        root = _bench(str(tmp / "f"), {"L2@big": (4, "r1"), "L2@small": (2, "r1")})
        v = SH.verdict(brain_dir=_brain(str(tmp / "f2"), CLU), bench_root=root,
                       now=1100.0, review_enabled=True, last_step_ts=1050.0)
        check("a partial arm is warn", v["level"] == "warn", v["reason"])
        check("and it is named with its count", "L2@small 2 of 4" in v["reason"])

        # 7. two rubrics in one directory
        root = _bench(str(tmp / "g"), {"a": (4, "r1"), "b": (4, "r2")})
        v = SH.verdict(brain_dir=_brain(str(tmp / "g2"), CLU), bench_root=root,
                       now=1100.0, review_enabled=True, last_step_ts=1050.0)
        drift = [c for c in v["checks"] if c["check"] == "rubric_drift"]
        check("two rubric hashes raise rubric drift",
              drift and drift[0]["level"] == "warn")

        # 8. verdicts newer than the last scoring run
        root = _bench(str(tmp / "h"), {"a": (4, "r1")}, score_after=False)
        v = SH.verdict(brain_dir=_brain(str(tmp / "h2"), CLU), bench_root=root,
                       now=1100.0, review_enabled=True, last_step_ts=1050.0)
        stale = [c for c in v["checks"] if c["check"] == "score_staleness"]
        check("verdicts written after the last score are warn",
              stale and stale[0]["level"] == "warn")

        # 9. the heartbeat writes `domains` as a mapping; the API renders it as
        # a list. live_verdict must read either -- the first version assumed the
        # list, iterated the mapping's keys and called .get() on a string, and
        # the alarm's own reader was the thing that broke.
        for shape in ({"weed": {"enabled": False}},
                      [{"domain": "weed", "enabled": False}]):
            fw = tmp / ("hb_%s" % type(shape).__name__) / "results" / "framework"
            os.makedirs(str(fw), exist_ok=True)
            with open(os.path.join(str(fw), "scheduler_status.json"), "w") as fh:
                json.dump({"domains": shape}, fh)
            SH._CTX["repo"] = str(fw.parents[1])
            try:
                v = SH.live_verdict(now=1.0)
                check("live_verdict reads a %s domains block" % type(shape).__name__,
                      v["level"] in ("ok", "warn", "crit"))
            except Exception as e:
                check("live_verdict reads a %s domains block" % type(shape).__name__,
                      False, repr(e))
        SH._CTX.clear()

        # 10. an absent corpus must not read as a complete one
        empty = tmp / "nocorpus" / "supervision_bench"
        os.makedirs(str(empty), exist_ok=True)
        v = SH.verdict(brain_dir=_brain(str(tmp / "i"), CLU), bench_root=str(empty),
                       now=1100.0, review_enabled=True, last_step_ts=1050.0)
        comp = [c for c in v["checks"] if c["check"] == "arm_completeness"][0]
        check("an absent corpus says so rather than reading as complete",
              comp.get("present") is False and "no benchmark corpus" in comp["detail"])

        # 11. never raises, whatever it is pointed at
        try:
            v = SH.verdict(brain_dir="/nonexistent/nope",
                           bench_root="/nonexistent/nope", now=1.0)
            check("a missing tree does not raise", v["level"] in ("ok", "warn", "crit"))
        except Exception as e:
            check("a missing tree does not raise", False, repr(e))
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)

    print("%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
