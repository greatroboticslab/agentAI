#!/usr/bin/env python3
"""Resume must reuse an answer only while the question is identical.

A walltime-truncated benchmark leaves most of an arm committed on disk. Re-asking
the model those cases costs GPU hours for answers already written, so `run(...,
resume=True)` reads them back. That is only sound while the committed record
answers the same question the scorer is about to ask: same case, arm, repeat and
model, and the same bundle and rubric hashes. These pin each of those guards, and
pin that a reused run says so in its own result rather than reading as fresh.

Run:  python tests/test_bench_resume.py
"""
import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import bench  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def _case():
    return {"case_id": "c1", "bundle_sha256": "b" * 8, "ok": True}


def _rec(**over):
    rec = {"case_id": "c1", "arm": "L2", "repeat": 0, "model": "m1",
           "verdict": {"fires": True}, "meta": {"model": "m1"},
           "bundle_sha256": "b" * 8, "rubric_sha256": bench.rubric_sha256(),
           "ts": 0.0}
    rec.update(over)
    return rec


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="benchresume"))
    try:
        p = tmp / "v.json"
        case = _case()

        p.write_text(json.dumps(_rec()))
        got = bench._reusable_verdict(p, case, "L2", "m1", 0)
        check("a matching record is reused", got is not None)
        check("reuse is marked on the record",
              bool(got and got.get("reused_from_disk")))

        for label, over, ask in (
                ("another case", {"case_id": "c2"}, ("L2", "m1", 0)),
                ("another arm", {"arm": "L3"}, ("L2", "m1", 0)),
                ("another repeat", {"repeat": 1}, ("L2", "m1", 0)),
                ("another model", {"model": "m2"}, ("L2", "m1", 0)),
                ("a re-cut bundle", {"bundle_sha256": "z" * 8}, ("L2", "m1", 0)),
                ("an edited rubric", {"rubric_sha256": "z" * 8}, ("L2", "m1", 0))):
            p.write_text(json.dumps(_rec(**over)))
            check("%s is re-run, not reused" % label,
                  bench._reusable_verdict(p, case, *ask) is None)

        p.write_text("{not json")
        check("an unreadable verdict is re-run, not reused",
              bench._reusable_verdict(p, case, "L2", "m1", 0) is None)
        p.write_text(json.dumps({"case_id": "c1", "arm": "L2", "repeat": 0,
                                 "model": "m1"}))
        check("a record with no verdict object is re-run, not reused",
              bench._reusable_verdict(p, case, "L2", "m1", 0) is None)

        # Resume off is the default, so a plain run can never silently read an
        # old answer back: this is what keeps "we measured it" true.
        check("resume defaults to off",
              bench.run.__defaults__[-1] is False)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)

    print("%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
