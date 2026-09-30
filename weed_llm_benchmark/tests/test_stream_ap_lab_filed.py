#!/usr/bin/env python3
"""A lab item the stream filed for a person runs once the person decides.

The stream autopilot's _ready skips lab actions, and _run_lab_items took
proposed items only, so a lab fetch (L16L) filed for a person (here: the
D21 floors are placeholders) never ran after its approval: the DATA lane
stood still from 2026-09-29 to 2026-09-30 with the approval recorded.
_run_lab_items now treats a filed lab item as _ready treats a filed cluster
item.

Pinned, in the stream world of test_stream_ap_world:
  * floors null: the lab fetch of a lab-only source is filed for a person
    and nothing is launched;
  * the person approves: the next tick launches the fetch on the lab
    (detached), the lane runs it, and the ledger records it executed;
  * in another world the person denies: the next tick records the denial,
    clears the lane, and launches nothing.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import AP, OWNER, World, check  # noqa: E402

CAND = {"id": "gh_src", "provider": "github", "licence": "MIT", "target_classes": ["Purslane"],
        "bytes": 1e9, "expected_target_boxes": 900}


def filed_world(tag):
    w = World(tag, floors=(None, None))
    w.ready_r0()
    w.queue(0)
    w.placement({"hf": "pass", "ftp": "fail"})
    w.candidates([dict(CAND)])
    w.tick(2)
    return w


def lab_fetches(w):
    return [x for x in w.runner.launched if "fetch" in x["argv"] and "gh_src" in x["argv"]]


def test_approve():
    print("a filed lab fetch runs after a person approves it")
    w = filed_world("labfiled_ok")
    it = w.lane("DATA").get("item") or {}
    check("floors null: the lab fetch (L16L) is filed for a person, and nothing is launched",
          it.get("lever") == "L16L" and it.get("status") == "filed" and it.get("approval_id")
          and not lab_fetches(w), (it.get("lever"), it.get("status"), it.get("approval_id")))
    res = AP.decide(w.domain, it.get("approval_id"), "approve", OWNER, "test: a person approves the fetch",
                    w.clock(), root=str(w.lab))
    check("  the person's approval is recorded", isinstance(res, dict) and res.get("ok"), res)
    w.tick()
    it2 = w.lane("DATA").get("item") or {}
    ex = [e for e in w.events("executed") if e.get("lever") == "L16L"]
    check("the next tick launches the fetch on the lab (detached), and the lane runs it",
          len(lab_fetches(w)) == 1 and it2.get("status") == "running" and it2.get("lab_job"),
          (it2.get("status"), [x["argv"] for x in w.runner.launched]))
    check("  the ledger records it executed under the approval",
          ex and ex[-1].get("approval_id") == it.get("approval_id"), ex[-1:])
    w.tick()
    check("  and it is launched once, not again on the next tick", len(lab_fetches(w)) == 1,
          [x["argv"] for x in w.runner.launched])


def test_deny():
    print("a filed lab fetch a person denies is cleared, never launched")
    w = filed_world("labfiled_no")
    it = w.lane("DATA").get("item") or {}
    res = AP.decide(w.domain, it.get("approval_id"), "deny", OWNER, "test: a person denies the fetch",
                    w.clock(), root=str(w.lab))
    check("  the person's denial is recorded", isinstance(res, dict) and res.get("ok"), res)
    w.tick()
    it2 = w.lane("DATA").get("item") or {}
    den = [e for e in w.events("denied") if e.get("approval_id") == it.get("approval_id")]
    check("the next tick records the denial and clears the lane; nothing is launched",
          den and it2.get("approval_id") != it.get("approval_id") and not lab_fetches(w),
          (den, it2.get("status"), it2.get("approval_id")))


def main():
    test_approve()
    test_deny()
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
