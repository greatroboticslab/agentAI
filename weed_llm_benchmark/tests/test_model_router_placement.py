#!/usr/bin/env python3
"""The placement rule, pinned in a test instead of in a memory file.

Every model that analyses, reviews, decides or plans runs on the cluster; the
lab's single 3060 hosts the tiny guide only. Between 2026-09-04 and 09-11 that
was violated for a week — nine campaign reviews came off a 4.7 GB model on the
lab box while 458 GB of verified weights sat on the cluster — and nothing failed,
because the rule lived in prose and the wiring lived in a dict nobody re-read.

These pin the two things that made it silent:
  * an unknown role handed back a usable lab model, so a caller that did not
    check `ok` got the 3B model for work the role table never approved;
  * a judgement resolved onto the lab was indistinguishable, in the returned
    dict, from one resolved onto the cluster.

Run:  python tests/test_model_router_placement.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import model_router as M  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def main():
    r = M.resolve("nope-not-a-role")
    check("an unknown role is not ok", r["ok"] is False)
    check("an unknown role hands back no model", r["model"] is None,
          "got %r" % (r["model"],))
    check("an unknown role is never authoritative", r["authoritative"] is False)

    for role in ("planner", "deep_review", "hard_reasoning"):
        r = M.resolve(role)
        check("%s is defined" % role, r["ok"] is True, r.get("error", ""))
        check("%s runs on the cluster" % role, r.get("place") == "cluster",
              "place=%r" % r.get("place"))
        check("%s is authoritative" % role, r.get("authoritative") is True)
        check("%s is async" % role, r.get("is_async") is True)

    # The rule's teeth: a judgement that came off the lab is a draft, and the
    # returned dict has to say so. These two roles answer a live web request, so
    # they are allowed to run small — they are not allowed to look authoritative.
    for role in ("interactive_plan", "analysis_summary"):
        r = M.resolve(role)
        check("%s is marked a judgement" % role, r.get("judgement") is True)
        if r.get("place") == "lab":
            check("%s on the lab is not authoritative" % role,
                  r.get("authoritative") is False)

    # No role may quietly become authoritative by being placed on the lab.
    for row in M.role_table():
        if row.get("judgement") and row.get("place") != "cluster":
            check("%s: lab judgement is not authoritative in the table" % row["role"],
                  row.get("authoritative") is False)

    # Nothing in the table may point at the lab's synchronous ollama endpoint for
    # a cluster role: a cluster brain is an sbatch job, not an HTTP call.
    for row in M.role_table():
        if row.get("place") == "cluster":
            check("%s does not name a lab endpoint" % row["role"],
                  "127.0.0.1" not in str(row.get("default", ""))
                  and "localhost" not in str(row.get("default", "")))

    print("%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
