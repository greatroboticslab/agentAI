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

    def test_the_module_declares_no_write_routes(self):
        src = pathlib.Path(A.__file__).read_text()
        tree = ast.parse(src)
        verbs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    f = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(f, ast.Attribute):
                        verbs.add(f.attr)
        self.assertEqual(verbs, {"get"},
                         "writes belong to the single-writer channel and the "
                         "policy gate, not to this surface")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
