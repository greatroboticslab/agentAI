"""Tests for brain/api.py -- the read-only supervision surface.

The properties asserted here are the ones that decide whether the page can be
trusted: a missing store never reads as an empty one, an unwired tier never reads
as a configured one, a broken chain still shows its own rows, and nothing here
can write.
"""

import ast
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import api as A     # noqa: E402
from weed_optimizer_framework.tools import db                 # noqa: E402


class Base(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="brain_api_")
        A._CTX.clear()
        A._CTX.update({"repo": self.root, "db": db, "log": None})
        self.bdir = os.path.join(self.root, "results", "framework", "_brain", "weed")
        os.makedirs(self.bdir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestMissingIsNotEmpty(Base):
    """The failure this layer was built after was a silence mistaken for health."""

    def test_each_store_reports_missing_rather_than_empty(self):
        for fn, needle in ((A.api_brain_signals, "evidence bundle"),
                           (A.api_brain_plans, "plans"),
                           (A.api_brain_experiments, "experiments"),
                           (A.api_brain_su, "SU ledger")):
            r = fn("weed")
            self.assertFalse(r["available"], fn.__name__)
            self.assertIn(needle, r["reason"], fn.__name__)
            self.assertEqual(r["rows"], [])

    def test_an_empty_store_is_available_with_no_rows(self):
        open(os.path.join(self.bdir, "experiments.jsonl"), "w").close()
        r = A.api_brain_experiments("weed")
        self.assertTrue(r["available"])
        self.assertEqual(r["rows"], [])

    def test_zero_spend_is_only_reported_when_a_ledger_exists(self):
        """`total` on a missing ledger returns 0 SU, which is true of an empty
        file and false of an absent one."""
        self.assertFalse(A.api_brain_su("weed")["available"])
        open(os.path.join(self.bdir, "su_ledger.jsonl"), "w").close()
        r = A.api_brain_su("weed")
        self.assertTrue(r["available"])
        self.assertEqual(r["total"]["su"], 0.0)


class TestContent(Base):

    def test_an_unwired_tier_is_null_and_not_a_default(self):
        r = A.api_brain_roles("weed")
        self.assertTrue(r["available"])
        for tier, model in r["tiers"].items():
            self.assertTrue(model is None or isinstance(model, str), tier)
        self.assertIn("policy", r)

    def test_roles_exposes_the_sealed_lever_menu_with_controls(self):
        r = A.api_brain_roles("weed")
        self.assertTrue(r["lever_menu"])
        for lever in r["lever_menu"]:
            self.assertTrue(lever["control"], lever["id"])
            self.assertIn(lever["risk"], ("R1", "R2", "R3"))

    def test_policy_answers_what_the_layer_may_do(self):
        r = A.api_brain_policy("weed")
        self.assertTrue(r["available"])
        self.assertGreater(r["n_actions"], 25)
        risks = {row["risk"] for row in r["rows"]}
        self.assertTrue(risks <= {"R0", "R1", "R2", "R3", "R4"})
        self.assertIn("R4", risks, "a destructive action should exist and be labelled")

    def test_a_torn_final_line_does_not_hide_the_whole_file(self):
        """A walltime-killed writer leaves one; losing the file would lose the run."""
        p = os.path.join(self.bdir, "experiments.jsonl")
        with open(p, "w") as fh:
            fh.write(json.dumps({"id": "a", "ts": 1}) + "\n")
            fh.write('{"id": "b", "ts": 2')
        r = A.api_brain_experiments("weed")
        self.assertTrue(r["available"])
        self.assertEqual([row["id"] for row in r["rows"]], ["a"])

    def test_signals_report_fired_and_unknown_separately(self):
        bundle = {"case_id": "live", "sha256": "abc", "sections": {
            "sacct": [{"artifact_id": "s.txt", "JobID": "1", "State": "TIMEOUT",
                       "Elapsed": "12:00:18", "Timelimit": "12:00:00",
                       "raw": "1|rndtrain|TIMEOUT|12:00:18|12:00:00",
                       "lines": [[3, "1|rndtrain|TIMEOUT|12:00:18|12:00:00"]]}]}}
        with open(os.path.join(self.bdir, "latest_bundle.json"), "w") as fh:
            json.dump(bundle, fh)
        r = A.api_brain_signals("weed")
        self.assertTrue(r["available"])
        self.assertGreaterEqual(r["n_fired"], 1)
        self.assertGreater(r["n_unknown"], 0,
                           "a check that cannot run must be counted, not dropped")
        self.assertEqual(r["bundle_sha256"], "abc")

    def test_timeline_names_the_sources_it_could_not_read(self):
        r = A.api_brain_timeline("weed")
        self.assertTrue(r["available"])
        self.assertTrue(r["sources_unavailable"],
                        "an unreadable source must be named, not omitted")


class TestSafety(Base):

    def test_a_domain_id_cannot_escape_its_directory(self):
        for hostile in ("../../etc", "weed/../../..", "we/ed", "weed\x00"):
            r = A.api_brain_signals(hostile)
            self.assertFalse(r["available"])
            self.assertNotIn("..", r.get("path", ""))

    def test_there_is_exactly_one_write_route_and_it_is_the_approval(self):
        """Pinning the exception is stronger than forbidding all writes.

        Corrections and actions have their own owners -- the single-writer
        channel and the policy gate -- and nothing here may take those over. An
        approval is different in kind: it is a PERSON ruling on a queued request,
        and there is nowhere else on the platform for them to do it. So the rule
        is not "no writes"; it is "this write and no other", which is a claim a
        test can actually hold.
        """
        src = pathlib.Path(A.__file__).read_text()
        tree = ast.parse(src)
        routes = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    f = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(f, ast.Attribute) and f.attr in ("get", "post",
                                                                   "put", "patch",
                                                                   "delete"):
                        routes.append((f.attr, node.name))
        writes = [n for verb, n in routes if verb != "get"]
        self.assertEqual(writes, ["api_brain_approvals_decide"],
                         "the only write on this surface is a person deciding "
                         "an approval; everything else belongs to its own owner")

    def test_the_module_opens_no_file_for_writing(self):
        src = pathlib.Path(A.__file__).read_text()
        self.assertFalse(re.search(r"open\([^)]*[\"'][wa]", src))

    def test_the_routes_are_not_exempt_from_authentication(self):
        """These expose campaign state; the middleware must cover them."""
        ds = (pathlib.Path(A.__file__).resolve().parents[1] / "dashboard_server.py").read_text()
        block = ds[ds.index("_AUTH_EXEMPT_PATHS = {"):]
        block = block[:block.index("}") + 1]
        self.assertNotIn("/api/brain/", block)

    def test_the_surface_is_mounted_by_the_dashboard(self):
        ds = (pathlib.Path(A.__file__).resolve().parents[1] / "dashboard_server.py").read_text()
        self.assertIn("from .brain import api as _brain_api", ds)
        self.assertIn("_brain_api.mount(app,", ds)

    def test_a_broken_chain_still_shows_its_rows(self):
        """A reader looking at a tampered chain needs both what it says and that
        it cannot be trusted; returning nothing would hide the tampering too."""
        r = A.api_brain_corrections("weed")
        self.assertTrue(r["available"])
        self.assertIn("chain_ok", r)
        self.assertIn("rows", r)


class TestPage(Base):
    """The page exists so the demo can be driven from a browser and nothing else."""

    def test_the_page_renders_for_a_domain(self):
        r = A.page_supervision("weed")
        body = r.body.decode("utf-8")
        self.assertIn("Supervision", body)
        self.assertIn('const DOMAIN = "weed"', body)

    def test_the_page_reads_every_route_it_has(self):
        body = A.page_supervision("weed").body.decode("utf-8")
        for key in ("signals", "corrections", "su", "roles", "plans",
                    "experiments", "policy", "timeline"):
            self.assertIn('"%s"' % key, body, key)

    def test_a_hostile_domain_cannot_break_out_of_the_page(self):
        for hostile in ("../../etc", '"><script>alert(1)</script>', "we ed"):
            body = A.page_supervision(hostile).body.decode("utf-8")
            self.assertNotIn("<script>alert", body)
            self.assertNotIn("..", body.split("const DOMAIN")[1][:80])

    def test_the_page_says_when_a_store_is_missing_rather_than_drawing_it_empty(self):
        body = A.page_supervision("weed").body.decode("utf-8")
        self.assertIn("if(!data.available)", body)
        self.assertIn("Not available", body)


class TestApprovals(Base):
    """The one write on this surface, and why it is not a second writer.

    An approval is a PERSON deciding, and there is nowhere else on the platform
    for them to do it. The single-writer rule protects the correction chain from
    a second automated author; it was never about keeping people out of their own
    governance queue. What matters is that the author cannot be forged.
    """

    def test_an_empty_queue_says_so_rather_than_showing_nothing(self):
        r = A.api_brain_approvals("weed")
        self.assertFalse(r["available"])
        self.assertIn("queued for approval", r["reason"])

    def test_a_decision_is_refused_when_the_caller_cannot_be_identified(self):
        A._CTX.pop("actor_of", None)
        r = A.api_brain_approvals_decide("weed", "ap-1", {"decision": "approve",
                                                          "reason": "ok"}, None)
        self.assertFalse(r["ok"])
        self.assertIn("no author", r["reason"])

    def test_the_body_cannot_name_the_decider(self):
        """A decision whose author the body could name is one anyone can
        attribute to anyone."""
        from weed_optimizer_framework.tools.brain import approvals as AP
        AP.propose("weed", "clean_train_d", {}, "R3", "tier1:x", "because",
                   1788700000.0, root=self.root)
        item = list(AP.state("weed", root=self.root))[0]
        A._CTX["actor_of"] = lambda req: "harry"
        A._CTX["can_manage"] = lambda actor, dom: True
        r = A.api_brain_approvals_decide("weed", item,
                                         {"decision": "approve", "reason": "yes",
                                          "decided_by": "human:someone_else"}, None)
        self.assertTrue(r["ok"], r.get("reason"))
        self.assertEqual(r["item"]["decided_by"], "human:harry")

    def test_a_caller_who_does_not_manage_the_agent_is_refused(self):
        A._CTX["actor_of"] = lambda req: "stranger"
        A._CTX["can_manage"] = lambda actor, dom: False
        r = A.api_brain_approvals_decide("weed", "ap-1",
                                         {"decision": "approve", "reason": "x"}, None)
        self.assertFalse(r["ok"])
        self.assertIn("do not manage", r["reason"])

    def test_the_queue_shows_up_after_something_is_queued(self):
        from weed_optimizer_framework.tools.brain import approvals as AP
        AP.propose("weed", "clean_train_d", {}, "R3", "tier1:x", "because",
                   1788700000.0, root=self.root)
        r = A.api_brain_approvals("weed")
        self.assertTrue(r["available"])
        self.assertEqual(r["n_pending"], 1)

    def test_the_page_renders_the_queue(self):
        body = A.page_supervision("weed").body.decode("utf-8")
        self.assertIn('"approvals"', body)
        self.assertIn("Waiting on a person", body)


class TestServedScript(Base):
    """Check the script the browser receives, not the one in the file.

    The page is a Python string. An escaped quote written for the browser is
    collapsed by Python before the browser sees it, so a handler that parsed
    perfectly as source arrived as `decide(''+id+'','approve')` and threw
    "unexpected token: string literal" on load, taking the whole page with it.
    A syntax check on the file would have passed. This checks what is served.
    """

    def _served_script(self):
        page = A._PAGE
        return page[page.index("<script>") + 8:page.rindex("</script>")]

    def test_the_served_script_carries_no_backslash_escapes(self):
        js = self._served_script()
        self.assertNotIn("\\", js,
                         "a backslash in this string was written for the browser "
                         "and will be eaten by Python first")

    def test_the_page_uses_no_inline_event_handlers(self):
        js = self._served_script()
        self.assertNotIn("onclick=", js,
                         "inline handlers need quotes inside quotes inside a "
                         "Python string; use a data attribute and a listener")

    def test_the_decision_controls_are_reachable_from_the_markup(self):
        js = self._served_script()
        for needle in ('data-act="approve"', 'data-act="deny"',
                       'button[data-act]', "a decision needs a reason"):
            self.assertIn(needle, js, needle)

    def test_the_served_script_parses(self):
        """Balanced quotes and braces, as a cheap stand-in for a parser."""
        js = self._served_script()
        for pair in ("{}", "()", "[]"):
            self.assertEqual(js.count(pair[0]), js.count(pair[1]), pair)
        for q in ("'", '"'):
            self.assertEqual(js.count(q) % 2, 0, "unbalanced %s" % q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
