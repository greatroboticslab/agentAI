"""The supervisor call path: one bundle in, one schema-checked verdict out (v3.29.0).

Why this exists
---------------
The tiers above the deterministic checks are language models reached over an
OpenAI-compatible endpoint. Which model is behind that endpoint is a deployment
question -- a vLLM server on four H100s, an ollama window on eight, a small model
on one V100 -- and none of it should reach the reviewing code. This module is the
whole seam: endpoint discovery, one call, and the telemetry that makes the call
comparable to every other call.

Two properties are load-bearing for the measurement.

* **The live prompt is built by the benchmark's renderer.** `review()` hands the
  bundle to `bench.render_prompt` through the same view shape the L2/L3 arms use.
  If production built its prompt one way and the benchmark another, the benchmark
  would be measuring a prompt nothing in production ever sends, and a model's
  score would not transfer. Constructing them in one place is what makes the
  head-to-head numbers mean anything about the deployed layer.
* **A prompt that does not fit is reported, never sent.** `tokens_in` is
  estimated before the call and checked against `num_ctx`; an overflowing prompt
  returns a refusal. Silent truncation -- a model answering confidently about the
  half of the evidence that survived the context window -- is the defect class
  this whole campaign is about, and the reviewer must not commit it itself.

Citation checking is delegated to `citations.validate_verdict`, which already
counts the three failure modes separately (no quote, unresolvable quote, resolved
but not load-bearing). Nothing here re-implements it: a second copy of that logic
would drift from the one the benchmark scores against, and the two would disagree
about the same reply.

The callable contract
---------------------
`OpenAICompatClient` is `(prompt, model_id, num_ctx) -> {text, tokens_in,
tokens_out, latency_s, su}`, which is exactly what `bench.py --model-entry`
resolves and calls. The benchmark and the live path therefore drive the same
client, and a bug in transport shows up in both rather than in only the one
nobody is watching.

SU accounting here is a per-call estimate from a declared rate
(`BRAIN_SU_PER_HOUR`), labelled `su_source: "endpoint_rate"`. It is not the
billing record; the ledger reconciles against `sacct` and owns that number.
"""

import json
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_TIMEOUT_S = 600
DEFAULT_NUM_CTX = 32768
DEFAULT_TEMPERATURE = 0.3
HEARTBEAT_MAX_AGE_S = 180.0     # WP4 gate: a window is "configured" only while fresh
CHARS_PER_TOKEN = 3.6           # conservative; the same ratio bench.py estimates with

PROMPT_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "prompts", "supervisor.txt")

VERDICT_KEYS = ("verdict", "findings", "corrections", "escalate", "confidence")


# --- helpers -----------------------------------------------------------------

def sha256_text(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def estimate_tokens(text):
    """Character-based token estimate. Deliberately an over-estimate.

    A cheap over-estimate that refuses a borderline prompt is the safe error; an
    under-estimate lets a prompt through that the server then truncates without
    telling anyone.
    """
    return int(len(str(text or "")) / CHARS_PER_TOKEN) + 1


def load_prompt(path=None):
    """The reviewer instructions, from disk. Missing file is an error, not a default.

    A prompt silently falling back to a built-in string would mean two campaigns
    ran against different instructions while reporting the same prompt name.
    """
    p = path or os.environ.get("BRAIN_SUPERVISOR_PROMPT") or PROMPT_FILENAME
    with open(p, "rb") as fh:
        return fh.read().decode("utf-8")


def endpoint_from_heartbeat(role, root=".", now=None):
    """Endpoint for a deep window, or None with the reason it is not usable.

    Returns `(base_url, reason)`. A heartbeat older than HEARTBEAT_MAX_AGE_S is
    treated as gone: the job holding that port may already have hit its walltime,
    and reporting a dead window as configured is how a tick ends up waiting on a
    reply that will never come.
    """
    path = os.path.join(root, "results", "framework", "_endpoints", "%s.json" % role)
    try:
        with open(path, "rb") as fh:
            hb = json.loads(fh.read().decode("utf-8"))
    except Exception as e:
        return None, "no heartbeat for role %r (%s)" % (role, e)
    host = str(hb.get("host") or "127.0.0.1")
    port = hb.get("port")
    beat = hb.get("heartbeat_ts")
    if not port:
        return None, "heartbeat for %r declares no port" % (role,)
    try:
        age = (time.time() if now is None else now) - float(beat)
    except Exception:
        return None, "heartbeat for %r has no usable timestamp" % (role,)
    if age > HEARTBEAT_MAX_AGE_S:
        return None, ("heartbeat for %r is %.0f s old (limit %.0f); the window is "
                      "treated as gone" % (role, age, HEARTBEAT_MAX_AGE_S))
    return "http://%s:%s/v1" % (host, port), ""


# --- the callable bench.py resolves -----------------------------------------

class OpenAICompatClient(object):
    """One chat completion against any OpenAI-compatible server.

    Configured from the environment so the same entry point serves a vLLM server,
    an ollama window and a small per-call job without a code change:
    `BRAIN_ENDPOINT` (base URL ending in /v1), `BRAIN_MODEL`, `BRAIN_TEMPERATURE`,
    `BRAIN_TIMEOUT_S`, `BRAIN_SU_PER_HOUR`, `BRAIN_JSON_MODE`.
    """

    def __init__(self, endpoint=None, model=None, temperature=None, timeout_s=None,
                 su_per_hour=None, json_mode=None, opener=None):
        self.endpoint = (endpoint or os.environ.get("BRAIN_ENDPOINT") or "").rstrip("/")
        self.model = model or os.environ.get("BRAIN_MODEL") or ""
        self.temperature = float(temperature if temperature is not None
                                 else os.environ.get("BRAIN_TEMPERATURE",
                                                     DEFAULT_TEMPERATURE))
        self.timeout_s = float(timeout_s if timeout_s is not None
                               else os.environ.get("BRAIN_TIMEOUT_S", DEFAULT_TIMEOUT_S))
        rate = su_per_hour if su_per_hour is not None else os.environ.get("BRAIN_SU_PER_HOUR")
        self.su_per_hour = float(rate) if rate not in (None, "") else None
        jm = json_mode if json_mode is not None else os.environ.get("BRAIN_JSON_MODE", "1")
        self.json_mode = str(jm).strip().lower() not in ("0", "false", "no", "")
        self._opener = opener or urllib.request.urlopen

    def _post(self, url, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with self._opener(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def __call__(self, prompt, model_id=None, num_ctx=DEFAULT_NUM_CTX):
        """(prompt, model_id, num_ctx) -> {text, tokens_in, tokens_out, latency_s, su}.

        A refusal is returned as a result with an `error`, never raised: a
        provider outage is a real supervision outcome and the harness records it
        as a failed review rather than losing the row.
        """
        model = str(model_id or self.model or "")
        out = {"text": "", "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
               "su": 0.0, "su_source": "endpoint_rate", "model_requested": model,
               "model_used": "", "endpoint_used": self.endpoint,
               "prompt_sha256": sha256_text(prompt), "error": ""}
        est = estimate_tokens(prompt)
        out["tokens_in_estimated"] = est
        if not self.endpoint:
            out["error"] = "no endpoint configured"
            return out
        if est > int(num_ctx or DEFAULT_NUM_CTX):
            # Refuse rather than let the server truncate: an answer about the
            # surviving half of the evidence is worse than no answer.
            out["error"] = ("prompt is about %d tokens against num_ctx %d; refused "
                            "rather than truncated" % (est, int(num_ctx)))
            out["context_overflow"] = True
            return out
        payload = {"model": model, "temperature": self.temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            body = self._post(self.endpoint + "/chat/completions", payload)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            out["latency_s"] = round(time.time() - t0, 3)
            out["error"] = "HTTP %s: %s" % (e.code, detail or e.reason)
            return out
        except Exception as e:
            out["latency_s"] = round(time.time() - t0, 3)
            out["error"] = "%s: %s" % (type(e).__name__, e)
            return out
        out["latency_s"] = round(time.time() - t0, 3)
        try:
            out["text"] = str(body["choices"][0]["message"]["content"] or "")
        except Exception:
            out["error"] = "reply carried no message content"
        usage = body.get("usage") or {}
        out["tokens_in"] = int(usage.get("prompt_tokens") or 0)
        out["tokens_out"] = int(usage.get("completion_tokens") or 0)
        out["model_used"] = str(body.get("model") or model)
        if self.su_per_hour is not None:
            out["su"] = round(self.su_per_hour * out["latency_s"] / 3600.0, 6)
        else:
            out["su"] = 0.0
            out["su_source"] = "no rate declared"
        return out


def default_client():
    """Entry point for `bench.py --model-entry ...brain.supervisor:default_client`."""
    return OpenAICompatClient()


# --- shape checking ----------------------------------------------------------

def check_shape(obj):
    """Structural check of one verdict. Returns (ok, problems[]).

    This checks the SHAPE only. Whether a finding is true is decided against the
    evidence by `citations.validate_verdict` and against truth by the benchmark;
    conflating the three is how a well-formed wrong answer scores as a right one.
    """
    problems = []
    if not isinstance(obj, dict):
        return False, ["reply is not a JSON object"]
    v = obj.get("verdict")
    if v not in ("ok", "issue"):
        problems.append("verdict is %r, expected 'ok' or 'issue'" % (v,))
    for key in ("findings", "corrections"):
        if not isinstance(obj.get(key, []), list):
            problems.append("%s is not a list" % key)
    for i, f in enumerate(obj.get("findings") or []):
        if not isinstance(f, dict):
            problems.append("findings[%d] is not an object" % i)
            continue
        if not str(f.get("quote") or "").strip():
            problems.append("findings[%d] carries no quote" % i)
        if f.get("severity") not in ("info", "warn", "crit", None):
            problems.append("findings[%d] severity is %r" % (i, f.get("severity")))
    for i, c in enumerate(obj.get("corrections") or []):
        if not isinstance(c, dict):
            problems.append("corrections[%d] is not an object" % i)
            continue
        if not str(c.get("action") or "").strip():
            problems.append("corrections[%d] names no action" % i)
        if not str(c.get("quote") or "").strip():
            problems.append("corrections[%d] carries no quote" % i)
    esc = obj.get("escalate")
    if esc is not None:
        if not isinstance(esc, dict):
            problems.append("escalate is not an object")
        elif esc.get("to") not in ("none", "tier2", "human", None):
            problems.append("escalate.to is %r" % (esc.get("to"),))
    conf = obj.get("confidence")
    if conf is not None:
        try:
            c = float(conf)
            if not (0.0 <= c <= 1.0):
                problems.append("confidence %r is outside 0..1" % (conf,))
        except Exception:
            problems.append("confidence %r is not a number" % (conf,))
    return (not problems), problems


def json_from_text(text):
    """First balanced JSON object in a reply, or None.

    Models wrap JSON in prose and fences even under a JSON mode; treating that as
    a refusal would measure the harness rather than the model.
    """
    s = str(text or "")
    start = s.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None


# --- one review --------------------------------------------------------------

def build_prompt(bundle, task=None, case_id=None):
    """The review prompt for one bundle, rendered by the benchmark's renderer.

    Importing `bench.render_prompt` here is deliberate. A production prompt built
    by different code than the evaluated one would make every benchmark number a
    statement about a prompt that is never sent.
    """
    from . import bench                      # local import: bench is large
    view = {"case_id": case_id or bundle.get("case_id") or bundle.get("domain") or "live",
            "sections": bundle.get("sections") or {}, "scalars_only": False}
    return bench.render_prompt(view, task=task or load_prompt())


def review(bundle, client=None, num_ctx=DEFAULT_NUM_CTX, case_id=None,
           load_bearing_lines=None):
    """One review of one bundle: prompt, call, shape check, citation check.

    Returns the verdict alongside the telemetry the ledger records. `ok` false
    always carries a reason, and a review that could not run is reported as such
    rather than as a model that found nothing.
    """
    from . import citations
    client = client or default_client()
    record = {"ok": False, "reason": "", "verdict": None, "shape_problems": [],
              "bundle_sha256": bundle.get("sha256") or sha256_text(
                  json.dumps(bundle.get("sections") or {}, sort_keys=True)),
              "num_ctx": int(num_ctx), "rejected_unverifiable_count": 0}
    try:
        prompt = build_prompt(bundle, case_id=case_id)
    except Exception as e:
        record["reason"] = "could not build the prompt: %s: %s" % (type(e).__name__, e)
        return record
    record["prompt_sha256"] = sha256_text(prompt)
    record["prompt_chars"] = len(prompt)

    res = client(prompt, None, num_ctx)
    for k in ("tokens_in", "tokens_out", "latency_s", "su", "su_source",
              "model_requested", "model_used", "endpoint_used", "tokens_in_estimated",
              "context_overflow"):
        if k in res:
            record[k] = res[k]
    if res.get("error"):
        record["reason"] = res["error"]
        return record

    obj = json_from_text(res.get("text"))
    if obj is None:
        record["reason"] = "no JSON object in the reply"
        record["reply_excerpt"] = str(res.get("text") or "")[:400]
        return record
    shaped, problems = check_shape(obj)
    record["verdict"] = obj
    record["shape_problems"] = problems
    if not shaped:
        record["reason"] = "verdict is malformed: %s" % "; ".join(problems)
        return record

    checked = citations.validate_verdict(bundle, obj, load_bearing_lines)
    record["citations"] = checked.get("stats") or {}
    record["accepted_findings"] = checked.get("findings") or []
    record["rejected_findings"] = checked.get("rejected") or []
    record["rejected_unverifiable_count"] = len(record["rejected_findings"])
    record["ok"] = True
    return record


# --- CLI ---------------------------------------------------------------------

def _main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        # A hand-rolled dispatcher still owes a person `--help`; printing
        # "unknown command '--help'" is how a working tool looks broken.
        print("usage: supervisor.py {review|prompt|endpoint|shape} ...", file=sys.stderr)
        return 0 if len(argv) > 1 else 2
    cmd = argv[1]
    if cmd == "endpoint":
        url, why = endpoint_from_heartbeat(argv[2] if len(argv) > 2 else "deep",
                                           argv[3] if len(argv) > 3 else ".")
        print(json.dumps({"endpoint": url, "reason": why}, indent=2))
        return 0 if url else 1
    if cmd == "prompt":
        with open(argv[2], "rb") as fh:
            bundle = json.loads(fh.read().decode("utf-8"))
        sys.stdout.write(build_prompt(bundle))
        return 0
    if cmd == "shape":
        obj = json.loads(sys.stdin.read())
        ok, problems = check_shape(obj)
        print(json.dumps({"ok": ok, "problems": problems}, indent=2))
        return 0 if ok else 1
    if cmd == "review":
        with open(argv[2], "rb") as fh:
            bundle = json.loads(fh.read().decode("utf-8"))
        ctx = int(os.environ.get("BRAIN_NUM_CTX", DEFAULT_NUM_CTX))
        out = review(bundle, num_ctx=ctx)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out.get("ok") else 1
    print("unknown command %r" % (cmd,), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
