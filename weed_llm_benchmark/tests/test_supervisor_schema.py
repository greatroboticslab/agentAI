"""Tests for brain/supervisor.py -- the seam between a bundle and a model.

The properties asserted here are the ones that decide whether a measured score
means anything: the live prompt is the benchmark's prompt, an overflowing prompt
is refused rather than truncated, a provider failure is a recorded outcome rather
than an exception, and citation checking is delegated rather than re-implemented.
"""

import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import supervisor as S  # noqa: E402
from weed_optimizer_framework.tools.brain import bench as B       # noqa: E402


# Quotes are deliberately over the pre-registered 20-character floor: below it
# the validator refuses them as accidental matches, which is a property of the
# citation rule and not something a fixture should route around.
GOOD_VERDICT = {
    "verdict": "issue",
    "findings": [{"signal": "walltime_bound", "quote": "rndtrain|State TIMEOUT|12:00:18",
                  "diagnosis": "the job hit its wall", "severity": "crit"}],
    "corrections": [{"action": "round_train", "params": {"epochs": 40},
                     "risk": "R2", "reason": "shorten the recipe",
                     "quote": "rndtrain|State TIMEOUT|12:00:18"}],
    "escalate": {"to": "none", "reason": ""},
    "confidence": 0.8,
}

BUNDLE = {
    "case_id": "t-1",
    "sections": {
        "sacct": [{"artifact_id": "sacct_1.txt", "JobID": "44727703",
                   "State": "TIMEOUT", "Elapsed": "12:00:18",
                   "raw": "44727703|rndtrain|State TIMEOUT|12:00:18",
                   "lines": [[3, "44727703|rndtrain|State TIMEOUT|12:00:18"]]}],
        "out_tail": {"artifact_id": "train.out", "sha256": "abc",
                     "lines": [[811, "epoch 24/60"], [812, "slurmstepd: CANCELLED"]]},
    },
}


class _Handler(BaseHTTPRequestHandler):
    reply_text = json.dumps(GOOD_VERDICT)
    status = 200
    seen = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Handler.seen.append(json.loads(self.rfile.read(n).decode("utf-8")))
        if _Handler.status != 200:
            self.send_response(_Handler.status)
            self.end_headers()
            self.wfile.write(b"upstream is unhappy")
            return
        body = json.dumps({
            "model": "fake-model-served",
            "choices": [{"message": {"content": _Handler.reply_text}}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 56},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class ServerCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d/v1" % cls.port

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        _Handler.seen = []
        _Handler.status = 200
        _Handler.reply_text = json.dumps(GOOD_VERDICT)

    def client(self, **kw):
        kw.setdefault("endpoint", self.base)
        kw.setdefault("model", "fake-model")
        kw.setdefault("su_per_hour", 8.0)
        return S.OpenAICompatClient(**kw)


class TestShape(unittest.TestCase):

    def test_a_well_formed_verdict_passes(self):
        ok, problems = S.check_shape(GOOD_VERDICT)
        self.assertTrue(ok, problems)

    def test_a_finding_without_a_quote_is_malformed(self):
        v = json.loads(json.dumps(GOOD_VERDICT))
        v["findings"][0]["quote"] = "  "
        ok, problems = S.check_shape(v)
        self.assertFalse(ok)
        self.assertTrue(any("no quote" in p for p in problems))

    def test_a_correction_without_a_quote_is_malformed(self):
        v = json.loads(json.dumps(GOOD_VERDICT))
        del v["corrections"][0]["quote"]
        ok, problems = S.check_shape(v)
        self.assertFalse(ok)

    def test_bad_verdict_word_severity_confidence_and_escalation(self):
        for mutate, needle in (
                (lambda v: v.update(verdict="probably"), "verdict is"),
                (lambda v: v["findings"][0].update(severity="fatal"), "severity"),
                (lambda v: v.update(confidence=1.7), "outside 0..1"),
                (lambda v: v.update(confidence="high"), "not a number"),
                (lambda v: v.update(escalate={"to": "god"}), "escalate.to"),
                (lambda v: v.update(findings="none"), "findings is not a list")):
            v = json.loads(json.dumps(GOOD_VERDICT))
            mutate(v)
            ok, problems = S.check_shape(v)
            self.assertFalse(ok, needle)
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_a_non_object_reply_is_malformed(self):
        for bad in ("[]", None, 3, [GOOD_VERDICT]):
            ok, _ = S.check_shape(bad)
            self.assertFalse(ok)

    def test_shape_check_does_not_judge_truth(self):
        """A well-formed verdict that is completely wrong still passes shape."""
        v = {"verdict": "ok", "findings": [], "corrections": [],
             "escalate": {"to": "none", "reason": ""}, "confidence": 1.0}
        self.assertTrue(S.check_shape(v)[0])


class TestJsonExtraction(unittest.TestCase):

    def test_plain_object(self):
        self.assertEqual(S.json_from_text('{"a": 1}'), {"a": 1})

    def test_fenced_and_prefixed(self):
        text = 'Sure! Here is my review:\n```json\n{"a": 1, "b": [2,3]}\n```\nDone.'
        self.assertEqual(S.json_from_text(text), {"a": 1, "b": [2, 3]})

    def test_braces_inside_strings_do_not_confuse_it(self):
        self.assertEqual(S.json_from_text('{"q": "a } brace \\" and {"}')["q"],
                         'a } brace " and {')

    def test_no_object_returns_none(self):
        self.assertIsNone(S.json_from_text("I could not comply."))
        self.assertIsNone(S.json_from_text(""))


class TestPromptIsTheBenchmarkPrompt(unittest.TestCase):

    def test_build_prompt_delegates_to_the_benchmark_renderer(self):
        """If these ever diverge, every benchmark number describes a prompt that
        production does not send."""
        task = S.load_prompt()
        view = {"case_id": "t-1", "sections": BUNDLE["sections"],
                "scalars_only": False}
        self.assertEqual(S.build_prompt(BUNDLE), B.render_prompt(view, task=task))

    def test_the_prompt_carries_the_citable_evidence(self):
        p = S.build_prompt(BUNDLE)
        self.assertIn("slurmstepd: CANCELLED", p)
        self.assertIn("   811\t", p)

    def test_the_instructions_are_read_from_disk_and_demand_verbatim_quotes(self):
        text = S.load_prompt()
        self.assertIn("verbatim", text)
        self.assertIn("discarded", text)

    def test_a_missing_prompt_file_is_an_error_not_a_default(self):
        with self.assertRaises(Exception):
            S.load_prompt("/nonexistent/supervisor.txt")


class TestClient(ServerCase):

    def test_a_call_returns_text_and_telemetry(self):
        out = self.client()("hello", None, 32768)
        self.assertEqual(out["error"], "")
        self.assertEqual(out["tokens_in"], 1234)
        self.assertEqual(out["tokens_out"], 56)
        self.assertEqual(out["model_used"], "fake-model-served")
        self.assertGreaterEqual(out["latency_s"], 0.0)
        self.assertEqual(out["su_source"], "endpoint_rate")

    def test_json_mode_is_requested(self):
        self.client()("hello", None, 32768)
        self.assertEqual(_Handler.seen[-1]["response_format"], {"type": "json_object"})

    def test_an_overflowing_prompt_is_refused_and_never_sent(self):
        """The defect class under study is a model answering about the half of
        the evidence that survived truncation. The reviewer must not commit it."""
        huge = "x" * 200000
        out = self.client()(huge, None, 1024)
        self.assertTrue(out["context_overflow"])
        self.assertIn("refused", out["error"])
        self.assertEqual(_Handler.seen, [], "the overflowing prompt was sent")

    def test_an_http_error_is_a_result_not_an_exception(self):
        _Handler.status = 500
        out = self.client()("hello", None, 32768)
        self.assertIn("HTTP 500", out["error"])
        self.assertEqual(out["text"], "")

    def test_an_unreachable_endpoint_is_a_result_not_an_exception(self):
        out = S.OpenAICompatClient(endpoint="http://127.0.0.1:1/v1",
                                   model="m", timeout_s=2)("hello", None, 32768)
        self.assertTrue(out["error"])

    def test_no_endpoint_configured_is_reported(self):
        out = S.OpenAICompatClient(endpoint="", model="m")("hello", None, 32768)
        self.assertEqual(out["error"], "no endpoint configured")

    def test_su_is_zero_and_labelled_when_no_rate_is_declared(self):
        out = self.client(su_per_hour="")("hello", None, 32768)
        self.assertEqual(out["su"], 0.0)
        self.assertEqual(out["su_source"], "no rate declared")

    def test_the_client_matches_the_signature_bench_resolves(self):
        entry, why = B.load_model_entry(
            "weed_optimizer_framework.tools.brain.supervisor:default_client")
        self.assertTrue(callable(entry), why)


class TestHeartbeat(unittest.TestCase):

    def _write(self, root, role, **over):
        d = os.path.join(root, "results", "framework", "_endpoints")
        os.makedirs(d, exist_ok=True)
        hb = {"host": "127.0.0.1", "port": 8123, "heartbeat_ts": 1000.0}
        hb.update(over)
        with open(os.path.join(d, "%s.json" % role), "w") as fh:
            json.dump(hb, fh)

    def test_a_fresh_heartbeat_yields_an_endpoint(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "deep")
            url, why = S.endpoint_from_heartbeat("deep", root, now=1010.0)
            self.assertEqual(url, "http://127.0.0.1:8123/v1")
            self.assertEqual(why, "")

    def test_a_stale_heartbeat_is_treated_as_gone(self):
        """The job holding that port may already have hit its walltime."""
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "deep")
            url, why = S.endpoint_from_heartbeat("deep", root, now=1000.0 + 10000)
            self.assertIsNone(url)
            self.assertIn("treated as gone", why)

    def test_a_missing_or_malformed_heartbeat_says_so(self):
        with tempfile.TemporaryDirectory() as root:
            url, why = S.endpoint_from_heartbeat("deep", root, now=1000.0)
            self.assertIsNone(url)
            self.assertIn("no heartbeat", why)
            self._write(root, "noport", port=None)
            self.assertIsNone(S.endpoint_from_heartbeat("noport", root, now=1000.0)[0])
            self._write(root, "nots", heartbeat_ts="soon")
            self.assertIsNone(S.endpoint_from_heartbeat("nots", root, now=1000.0)[0])


class TestReview(ServerCase):

    def test_a_review_resolves_its_citation_and_reports_telemetry(self):
        rec = S.review(BUNDLE, client=self.client(), num_ctx=32768)
        self.assertTrue(rec["ok"], rec.get("reason"))
        self.assertEqual(rec["rejected_unverifiable_count"], 0)
        self.assertEqual(rec["citations"]["accepted"], 1)
        self.assertEqual(rec["tokens_in"], 1234)
        self.assertTrue(rec["prompt_sha256"])
        self.assertTrue(rec["bundle_sha256"])

    def test_an_invented_quote_is_rejected_and_counted(self):
        v = json.loads(json.dumps(GOOD_VERDICT))
        v["findings"][0]["quote"] = "rndtrain|State FLOURISHED|after 3 fortnights"
        _Handler.reply_text = json.dumps(v)
        rec = S.review(BUNDLE, client=self.client(), num_ctx=32768)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["rejected_unverifiable_count"], 1)
        self.assertEqual(rec["accepted_findings"], [])

    def test_a_malformed_verdict_is_reported_not_scored(self):
        _Handler.reply_text = json.dumps({"verdict": "maybe"})
        rec = S.review(BUNDLE, client=self.client(), num_ctx=32768)
        self.assertFalse(rec["ok"])
        self.assertIn("malformed", rec["reason"])
        self.assertIsNotNone(rec["verdict"])

    def test_a_reply_with_no_json_is_reported(self):
        _Handler.reply_text = "I am unable to review this."
        rec = S.review(BUNDLE, client=self.client(), num_ctx=32768)
        self.assertFalse(rec["ok"])
        self.assertIn("no JSON object", rec["reason"])
        self.assertIn("unable", rec["reply_excerpt"])

    def test_a_provider_outage_is_a_failed_review_not_a_crash(self):
        _Handler.status = 503
        rec = S.review(BUNDLE, client=self.client(), num_ctx=32768)
        self.assertFalse(rec["ok"])
        self.assertIn("HTTP 503", rec["reason"])

    def test_a_review_that_would_overflow_is_refused_before_the_call(self):
        rec = S.review(BUNDLE, client=self.client(), num_ctx=8)
        self.assertFalse(rec["ok"])
        self.assertIn("refused", rec["reason"])
        self.assertEqual(_Handler.seen, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
