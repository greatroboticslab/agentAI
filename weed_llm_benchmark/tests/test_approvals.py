"""Tests for brain/approvals.py -- the queue that makes R3 mean something.

Without a queue, R3 collapses into R4 in practice: an action nobody can ever
request is indistinguishable from one nobody may take. These assert the three
rules that make the queue a governance record rather than a to-do list.
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import approvals as A   # noqa: E402


class Base(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="approvals_")
        self.t = 1788700000.0

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def propose(self, **over):
        kw = {"domain": "weed", "action": "roboflow_generate_versions",
              "params": {"n": 1}, "risk": "R3",
              "requested_by": "tier1:deepseek-v4-flash",
              "reason": "the source pool changed", "ts": self.t,
              "est_su": 12.0, "root": self.root}
        kw.update(over)
        return A.propose(**kw)


class TestOnlyAPersonDecides(Base):

    def test_an_agent_cannot_decide_its_own_request(self):
        item = self.propose()["item"]
        r = A.decide("weed", item["id"], "approve", "tier1:deepseek-v4-flash",
                     "looks fine to me", self.t + 1, root=self.root)
        self.assertFalse(r["ok"])
        self.assertIn("no ceiling", r["reason"])
        self.assertEqual(A.state("weed", root=self.root)[item["id"]]["status"],
                         "pending")

    def test_no_tier_may_decide(self):
        item = self.propose()["item"]
        for actor in ("tier0:gemma4", "tier2:claude", "round-scheduler", "", None):
            r = A.decide("weed", item["id"], "approve", actor, "why",
                         self.t + 1, root=self.root)
            self.assertFalse(r["ok"], actor)

    def test_a_person_can(self):
        item = self.propose()["item"]
        r = A.decide("weed", item["id"], "approve", "human:harry",
                     "the pool really did change", self.t + 1, root=self.root)
        self.assertTrue(r["ok"], r.get("reason"))
        self.assertEqual(r["item"]["status"], "approved")
        self.assertEqual(r["item"]["decided_by"], "human:harry")


class TestR4IsNeverQueued(Base):

    def test_an_irreversible_action_is_refused_at_proposal(self):
        r = self.propose(risk="R4", action="roboflow_delete_junk_apply")
        self.assertFalse(r["ok"])
        self.assertIn("never queued", r["reason"])

    def test_and_nothing_is_written(self):
        self.propose(risk="R4")
        self.assertEqual(A.read("weed", root=self.root), [])

    def test_an_unknown_risk_is_refused(self):
        self.assertFalse(self.propose(risk="R9")["ok"])
        self.assertFalse(self.propose(risk="")["ok"])


class TestTheRecord(Base):

    def test_a_request_with_no_reason_is_refused(self):
        self.assertFalse(self.propose(reason="  ")["ok"])

    def test_a_request_with_no_requester_is_refused(self):
        self.assertFalse(self.propose(requested_by="")["ok"])

    def test_a_decision_with_no_reason_is_refused(self):
        item = self.propose()["item"]
        r = A.decide("weed", item["id"], "deny", "human:harry", "", self.t + 1,
                     root=self.root)
        self.assertFalse(r["ok"])

    def test_a_second_decision_is_refused_and_recorded(self):
        """'Approved twice by two people' must be readable, not silently lost."""
        item = self.propose()["item"]
        A.decide("weed", item["id"], "approve", "human:harry", "yes",
                 self.t + 1, root=self.root)
        r = A.decide("weed", item["id"], "deny", "human:someone", "actually no",
                     self.t + 2, root=self.root)
        self.assertFalse(r["ok"])
        self.assertIn("already approved", r["reason"])
        st = A.state("weed", root=self.root)[item["id"]]
        self.assertEqual(st["status"], "approved")
        self.assertEqual(len(st["attempts"]), 1)
        self.assertEqual(st["attempts"][0]["decided_by"], "human:someone")

    def test_the_log_is_append_only(self):
        item = self.propose()["item"]
        A.decide("weed", item["id"], "deny", "human:harry", "not now",
                 self.t + 1, root=self.root)
        recs = A.read("weed", root=self.root)
        self.assertEqual([r["kind"] for r in recs], ["request", "decision"])

    def test_a_torn_final_line_costs_that_line_only(self):
        item = self.propose()["item"]
        with open(A.path("weed", root=self.root), "a") as fh:
            fh.write('{"kind": "decision", "id": "x"')
        self.assertEqual(len(A.read("weed", root=self.root)), 1)
        self.assertEqual(A.state("weed", root=self.root)[item["id"]]["status"],
                         "pending")

    def test_pending_lists_only_what_still_needs_a_person(self):
        a = self.propose()["item"]
        b = self.propose(action="clean_train_d", ts=self.t + 5)["item"]
        A.decide("weed", a["id"], "approve", "human:harry", "ok", self.t + 6,
                 root=self.root)
        ids = [i["id"] for i in A.pending("weed", root=self.root)]
        self.assertEqual(ids, [b["id"]])

    def test_a_decision_on_an_unknown_item_is_refused(self):
        r = A.decide("weed", "ap-nope", "approve", "human:harry", "why",
                     self.t, root=self.root)
        self.assertFalse(r["ok"])
        self.assertIn("no such", r["reason"])

    def test_an_empty_queue_reads_as_empty_not_as_an_error(self):
        self.assertEqual(A.read("weed", root=self.root), [])
        self.assertEqual(A.state("weed", root=self.root), {})
        self.assertEqual(A.pending("weed", root=self.root), [])


class TestAgainstThePolicyTable(Base):

    def test_every_risk_the_queue_accepts_is_one_policy_declares(self):
        from weed_optimizer_framework.tools.brain import policy
        declared = {row.get("risk") for row in policy._table()["actions"].values()}
        self.assertTrue(declared <= {"R0", "R1", "R2", "R3", "R4"})
        for risk in declared - {"R4"}:
            self.assertTrue(self.propose(risk=risk, ts=self.t + hash(risk) % 100)["ok"],
                            risk)

    def test_the_queue_refuses_exactly_what_the_gate_refuses_for_a_tier(self):
        """An R4 an agent may not request is also one it may not queue."""
        from weed_optimizer_framework.tools.brain import policy
        d = policy.authorize("tier1:x", "roboflow_delete_junk_apply", {}, None, None)
        self.assertFalse(d["allowed"])
        self.assertFalse(self.propose(risk="R4")["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
