"""The benchmark actually reaches a model, end to end.

Every other test in this family checks a piece: the client can POST, the renderer
builds a prompt, the validator resolves a quote. None of them checked that
`bench --model-entry ...` resolves an object bench can CALL. It did not: the
documented entry was a factory function, `load_model_entry` returned it
unchanged, and bench then invoked it as `factory(prompt, model_id, num_ctx)`.
Every call raised, every review scored as no detection, and the run finished with
a complete results table of zeros. A two-hour window on a large model would have
been spent measuring nothing, and the table would have looked like a result.

So this rehearses the exact command the head-to-head runs, against a local server
standing in for the deployed model.
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import bench          # noqa: E402
from weed_optimizer_framework.tools.brain import corpus         # noqa: E402

ENTRY = "weed_optimizer_framework.tools.brain.supervisor:OpenAICompatClient"
SACCT_LINE = "44727703|rndtrain|TIMEOUT|12:00:18|12:00:00"

VERDICT = {"verdict": "issue",
           "findings": [{"signal": "walltime_bound", "quote": SACCT_LINE,
                         "diagnosis": "the run was killed at the wall",
                         "severity": "crit"}],
           "corrections": [], "escalate": {"to": "none", "reason": ""},
           "confidence": 0.9}


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Handler.seen.append(json.loads(self.rfile.read(n).decode("utf-8")))
        body = json.dumps({
            "model": "stand-in",
            "choices": [{"message": {"content": json.dumps(VERDICT)}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestBenchReachesAModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:%d/v1" % cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        _Handler.seen = []
        self.tmp = tempfile.mkdtemp(prefix="bench_model_path_")
        self.root = pathlib.Path(self.tmp) / "bench"
        (self.root / "cases").mkdir(parents=True)
        (self.root / "rubric.md").write_text("rubric\n")
        for cid, incident, state, elapsed in (("inc-1", True, "TIMEOUT", "12:00:18"),
                                              ("hc-1", False, "COMPLETED", "04:10:00")):
            d = self.root / "cases" / cid
            d.mkdir()
            (d / "truth.json").write_text(json.dumps({
                "date": "2026-01-01", "incident": incident, "class": "config",
                "signals_expected": ["walltime_bound"] if incident else []}))
            line = SACCT_LINE if incident else SACCT_LINE.replace("TIMEOUT", "COMPLETED")
            (d / "bundle.json").write_text(json.dumps({
                "case_id": cid, "sha256": "sha-" + cid,
                "sections": {"sacct": [{"artifact_id": "s.txt", "JobID": "44727703",
                                        "State": state, "Elapsed": elapsed,
                                        "Timelimit": "12:00:00", "raw": line,
                                        "lines": [[3, line]]}]}}))
        (self.root / "split.json").write_text(json.dumps({
            "dev": ["inc-1"], "test": ["hc-1"], "rule": "fixture",
            "sha256": corpus.corpus_digest(self.root, ["inc-1", "hc-1"])}))
        self.saved = {k: os.environ.get(k) for k in
                      ("BRAIN_ENDPOINT", "BRAIN_MODEL", "BRAIN_SU_PER_HOUR")}
        os.environ.update({"BRAIN_ENDPOINT": self.base, "BRAIN_MODEL": "stand-in",
                           "BRAIN_SU_PER_HOUR": "16"})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _run(self):
        return bench.main(["run", "--root", str(self.root), "--arms", "A0p,L2",
                           "--split", "all", "--repeats", "1", "--no-write",
                           "--model-entry", ENTRY, "--model", "stand-in",
                           "--num-ctx", "32768"])

    def test_the_run_reaches_the_model_once_per_case(self):
        rc = self._run()
        self.assertEqual(rc, 0, "the run reported an error")
        self.assertEqual(len(_Handler.seen), 2,
                         "bench's entry point never reached the model")

    def test_the_prompt_carries_the_evidence_and_the_instructions(self):
        self._run()
        # Cases are not sent in a guaranteed order, so find the incident's own
        # request rather than assuming it came first.
        prompts = [r["messages"][0]["content"] for r in _Handler.seen]
        incident = [p for p in prompts if "case: inc-1" in p]
        self.assertEqual(len(incident), 1, "the incident case was not reviewed")
        self.assertIn(SACCT_LINE, incident[0], "the model was not shown the evidence")
        self.assertIn("verbatim", incident[0], "the reviewer instructions were missing")
        self.assertEqual(_Handler.seen[0]["response_format"], {"type": "json_object"})

    def test_the_benchmark_asks_the_same_question_production_asks(self):
        """The renderer was already shared; the instructions were not.

        bench carried its own copy of the reviewer instructions, so the two paths
        agreed on what the model is shown and disagreed on what it is asked --
        which makes every benchmark number a statement about a question nothing
        in production ever puts to a model.
        """
        from weed_optimizer_framework.tools.brain import supervisor as S
        self.assertEqual(bench.PROMPT_TASK, S.load_prompt())

    def test_a_tier1_escalation_is_a_valid_shape(self):
        """escalation.py's E1 family routes there; a shape check that rejected
        tier1 would mark a correct verdict malformed."""
        from weed_optimizer_framework.tools.brain import supervisor as S
        v = dict(VERDICT, escalate={"to": "tier1", "reason": "cheap review"})
        ok, problems = S.check_shape(v)
        self.assertTrue(ok, problems)

    def test_a_factory_entry_point_is_refused_rather_than_silently_scoring_zero(self):
        from weed_optimizer_framework.tools.brain import supervisor as S
        with self.assertRaises(TypeError):
            S.default_client("prompt", "model", 32768)


if __name__ == "__main__":
    unittest.main(verbosity=2)
