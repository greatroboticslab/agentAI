#!/usr/bin/env python3
"""The INC research brain: corpus, digest, reply, validation, outcome (docs/INC_AUTOPILOT.md (c)).

The brain is advisory. What makes it safe to run at all is pinned here with a
mock model client, so no GPU, network or cluster is involved:

* the literature corpus (docs/literature) is consistent, every passage line
  can be quoted back verbatim, header lines and paraphrases are refused, the
  seed papers the protocol cites are present with the levers they inform, and
  BM25 retrieval is deterministic;
* the digest carries only dev: it is byte-identical when every test / ood22 /
  ood23 / imageweeds value in the evidence changes, it refuses to render when
  a non-dev address reaches it, and every value it shows can be cited back
  exactly from its `_at` address;
* the digest fits its context instead of failing (the live failure of
  2026-09-27 22:38Z: 46,923 tokens against num_ctx 49152 minus the reserve):
  num_ctx is chosen from its size, at most MAX_NUM_CTX, which the plan job's
  80 GB GPU holds even at twice the assumed KV cache per token; built from the
  whole campaign's real evidence (the frozen fixtures, and realloop_v1 when
  local) it is over the old budget, stages untrimmed, and at the old num_ctx is
  trimmed in TRIM_ORDER with every cut recorded, every fired diagnosis cite
  kept and every shown value citable, byte-identical under a campaign-wide
  non-dev perturbation; the minimal digest (every cut made) still holds every
  never-cut section whole; only a minimal digest that does not fit is refused;
* validation drops a bad cite, out-of-bounds or undeclared params, any test
  leak (text, exam keys in cite values, split-manifest paths), an off-menu
  lever in the ranked list, an unknown lever, a paraphrased or content-free
  quote, a lever whose preconditions do not hold (only_after, requires, the
  D4 gate on real-loop builds) and a duplicate; keeps the rest of the plan;
  and turns an off-menu idea into an R4 card draft through
  planner.make_experiment (which approvals refuses to queue);
* a valid item is materialised as levers.py would build it: priced from the
  evidence (never from the brain, never 0 when unknown), its command rendered
  and authorised for the tier2 actor, with parent_exp and a trigger only the
  brain names and supports; filed through the executor, it charges the
  campaign envelope;
* a plan that times out, fails, lands late, answers another digest or arrives
  garbled leaves the deterministic proposals exactly as they were, and a
  brain item equal to a deterministic one becomes a note on it;
* the cluster job refuses a staged digest edited after staging, however
  consistently it was re-hashed;
* outcome.py scores a child experiment's report against the prediction per
  chain with a sign test over paired steps (real pilot_v1 -> pilot_v2 when
  that report is present), the dev_twelve noise floor from B0's dev seed
  spread, and folds it into the track record;
* run_inc_plan.sh keeps the run_llm_review.sh pattern (per-job port, warm-up,
  planner role, 2 h) without the outer-copy rsync, its path check holds when
  the /ocean path runs through a symlink, it turns flash attention on, and it
  logs the model's own KV layout against the assumed one before warm-up.

Evidence is read from tests/fixtures/inc_replay/<exp> when present, else from
results/framework/inc/<exp> (read only). Cases whose evidence is not local
(pilot_v2's report, step1/relevance.json) are skipped until the fixture lands.

Run:  python3 tests/test_inc_ap_brain.py
"""
import calendar
import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A policy table of our own, so the governance checks below do not depend on
# whichever inc_* rows the live table carries today.
_TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_brain_"))
_POLICY = json.loads((ROOT / "weed_optimizer_framework/tools/brain/policy_actions.json")
                     .read_text(encoding="utf-8"))
_LIVE_ACTIONS = dict(_POLICY["actions"])
_EXP_PAT = "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
_POLICY["actions"] = {
    "inc_build_pilot": {
        "template": "sbatch run_inc_build.sh pilot build --exp {exp} --replay-mode {replay_mode}",
        "param_bounds": {"exp": {"type": "str", "pattern": _EXP_PAT},
                         "replay_mode": {"type": "enum", "value_type": "str",
                                         "values": ["sample", "full"]},
                         "est_gpu_hours": {"type": "float", "min": 0.0, "max": 200.0}},
        "risk": "R3", "reversible": True,
        "est_su": {"gpu_type": "v100-32", "gpu_count": 1, "hours_param": "est_gpu_hours",
                   "why": "test row"},
        "dry_run_variant": None, "allowed_tiers": ["round-scheduler", "tier2", "human"],
        "description": "test row: a pilot build"},
}
# The live rows of the other actions a brain item can render to, when the live
# table has them (their cases skip otherwise).
for _a in ("inc_label_audit", "inc_unblock_transient", "inc_build_realloop",
           "inc_build_baseline", "inc_relevance_build"):
    if _a in _LIVE_ACTIONS:
        _POLICY["actions"][_a] = _LIVE_ACTIONS[_a]
(_TMP / "policy_actions.json").write_text(json.dumps(_POLICY), encoding="utf-8")
os.environ["BRAIN_POLICY_ACTIONS"] = str(_TMP / "policy_actions.json")

from weed_optimizer_framework.tools.brain import approvals, policy  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import brain_plan as BP  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import corpus as C  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers as L  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import model as M  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import outcome as O  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import validate as V  # noqa: E402

FAILURES = []
SKIPS = []
FIXTURES = ROOT / "tests" / "fixtures" / "inc_replay"
RESULTS = ROOT / "results" / "framework" / "inc"
MODEL = "qwen3.8:27b"


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def skip(name, why):
    print("  skip %s (%s)" % (name, why))
    SKIPS.append(name)


def raises(fn, exc, contains=None):
    try:
        fn()
    except exc as e:
        return contains is None or contains in str(e)
    except Exception:
        return False
    return False


def exp_dir(exp):
    """Where an experiment's files are read from: the frozen fixture, else results/."""
    for base in (FIXTURES, RESULTS):
        if (base / exp / "exp.json").is_file():
            return base
    return None


def report_path(exp):
    """The experiment's report.json: the fixture's, else results/ (read only)."""
    for base in (FIXTURES, RESULTS):
        if (base / exp / "report.json").is_file():
            return base / exp / "report.json"
    return None


def inc_root(exps):
    """A temp INC tree holding copies of `exps` (fixture or results), so a test
    can perturb it; None if any is missing."""
    root = pathlib.Path(tempfile.mkdtemp(dir=str(_TMP), prefix="inc_"))
    for e in exps:
        base = exp_dir(e)
        if base is None:
            return None
        (root / e).mkdir()
        for f in (base / e).iterdir():
            if f.is_file() and f.suffix in (".json", ".jsonl"):
                (root / e / f.name).write_bytes(f.read_bytes())
    return root


# A lever menu in levers.json's shape, self-contained so the tests pin the
# brain's behaviour and not the current menu's wording.
MENU_RAW = {
    "_meta": {"version": "test"},
    "levers": {
        "L1": {"title": "Rebuild the pilot with full replay", "kind": "build",
               "policy_action": "inc_build_pilot", "risk": "R3",
               "argv": ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build",
                        "--exp", "{exp}", "--replay-mode", "full"],
               "param_bounds": {"exp": {"type": "str", "pattern": _EXP_PAT},
                                "replay_mode": {"type": "enum", "value_type": "str",
                                                "values": ["full"]},
                                "est_gpu_hours": {"type": "float", "min": 0.0, "max": 200.0}},
               "fixed": {"replay_mode": "full"},
               "control": "the parent pilot with sample replay",
               "success": "agreement >= 5/7 and a truth-helps step ACCEPTed",
               "falsifier": "agreement stays below 5/7",
               "predicted": {"metric": "agreement", "direction": "up"},
               "lit": [{"paper_id": "ibrahim2024"}]},
        "L4": {"title": "Label audit", "kind": "job", "policy_action": "inc_label_audit",
               "risk": "R2", "param_bounds": {"exp": {"type": "str", "pattern": _EXP_PAT}},
               "control": "held-out trusted folds", "success": "Bswap above baseline"},
        "L9": {"title": "A lever with no bounds anywhere", "kind": "job",
               "policy_action": "inc_nothing", "risk": "R2"},
    },
    "cards": {
        "X1": {"title": "LR re-warming", "risk": "R4",
               "hypothesis": "replay alone does not stop forgetting"},
    },
}
MENU = BP.normalise_menu(MENU_RAW)

D1 = {"id": "D1", "name": "recipe_forgets", "fired": True, "severity": "crit",
      "summary": "the cheap recipe degrades the incumbent: p_recipe 0.0 at ledger line 4",
      "cites": [{"artifact": "pilot_v1/ledger.jsonl", "line": 4, "pointer": "/decision/p_recipe",
                 "value": 0.0}],
      "levers": ["L1"], "exp": "pilot_v1"}
D4 = {"id": "D4", "name": "no_recipe_tracks_truth", "fired": False, "severity": "info",
      "summary": "silent", "cites": [], "levers": [], "exp": "pilot_v1"}

QUOTE = "Every re-warm raises loss even with no distribution shift."
GOOD_L1 = {"lever": "L1", "params": {"exp": "pilot_v3"}, "trigger": ["D1"],
           "rationale": "20 of 21 gate entries blame the recipe: the null arm forgets the pool.",
           "evidence_cites": [
               {"artifact": "pilot_v1/report.json", "pointer": "/agreement/full/rate",
                "value": 0.42857142857142855},
               {"artifact": "pilot_v1/ledger.jsonl", "line": 4, "pointer": "/decision/p_recipe",
                "value": 0.0}],
           "lit_cites": [{"paper_id": "2403.08763", "line": 11, "quote": QUOTE}],
           "predicted": {"metric": "agreement", "direction": "up", "magnitude": 0.14},
           "falsifier": "the best chain still agrees on 3 of 7 steps or fewer"}
GOOD_OFF = {"hypothesis": "Replay alone leaves the cheap recipe forgetting; the LR must re-warm.",
            "why_menu_insufficient": "lr0, warmup and epochs are pinned PROTOCOL constants.",
            "required_change": "a new protocol version of the incremental recipe",
            "cheapest_test": "one pilot on the same bins with full replay and a re-warmed LR",
            "control": "the latest full-replay pilot",
            "success_criterion": "agreement at least 5/7 with a truth-helps step accepted",
            "lit_cites": [{"paper_id": "ibrahim2024", "line": 11, "quote": QUOTE}]}


def variant(base, **changes):
    item = copy.deepcopy(base)
    for k, v in changes.items():
        if v is None:
            item.pop(k, None)
        else:
            item[k] = v
    return item


PLAN = {
    "ranked_menu": [
        GOOD_L1,
        variant(GOOD_L1, evidence_cites=[dict(GOOD_L1["evidence_cites"][0], value=0.4286)]),
        variant(GOOD_L1, params={"exp": "pilot v3!"}),
        variant(GOOD_L1, params={"exp": "pilot_v3", "replay_mode": "sample"}),
        variant(GOOD_L1, rationale="This should raise the test mAP of the full chain."),
        variant(GOOD_L1, evidence_cites=[{"artifact": "pilot_v1/report.json",
                                          "pointer": "/final/0/exams/test/twelve/mean",
                                          "value": 0.7734860673019547}]),
        variant(GOOD_L1, lever="X1"),
        variant(GOOD_L1, lever="L99"),
        variant(GOOD_L1, lit_cites=[{"paper_id": "2403.08763", "line": 11,
                                     "quote": "Every re-warm increases loss even without shift."}]),
        variant(GOOD_L1, lit_cites=[{"paper_id": "2403.08763", "line": 4,
                                     "quote": "Simple and Scalable Strategies to Continually"}]),
        variant(GOOD_L1, predicted=None),
        variant(GOOD_L1, evidence_cites=[{"artifact": "pilot_v1/report.json", "pointer": "/done",
                                          "value": 1}]),
        variant(GOOD_L1, lever="L9", params={}),
        variant(GOOD_L1, evidence_cites=[]),
        variant(GOOD_L1, params={"exp": "pilot_v3", "recipes": "full"}),
        GOOD_L1,
        variant(GOOD_L1, lit_cites=[{"paper_id": "2403.08763", "line": 11,
                                     "quote": "RESULT [continual]: At 4"}]),
    ],
    "off_menu": [
        GOOD_OFF,
        variant(GOOD_OFF, control=None),
        variant(GOOD_OFF, hypothesis="The ood22 exam would show the drift."),
    ],
    "stop_recommendation": {"stop": False, "reason": "the goal is not met yet"},
}


class MockClient(object):
    """The supervisor.OpenAICompatClient callable shape, with a canned reply."""

    def __init__(self, text="", error="", exc=None):
        self.text, self.error, self.exc, self.calls = text, error, exc, []

    def __call__(self, prompt, model_id=None, num_ctx=None):
        self.calls.append((len(prompt), model_id, num_ctx))
        if self.exc:
            raise self.exc
        return {"text": self.text, "error": self.error, "tokens_in": len(prompt) // 3,
                "tokens_out": len(self.text) // 3, "latency_s": 0.01, "model_used": model_id}


# --- corpus -----------------------------------------------------------------------

def test_corpus():
    print("literature corpus")
    root = C.default_dir()
    check("corpus: docs/literature is found from the package", (root / "index.json").is_file(),
          str(root))
    problems = C.check_corpus(root)
    check("corpus: files and index agree", problems == [], problems[:3])
    corpus = C.Corpus(root)
    idx = corpus.index
    check("corpus: 128 papers, 432 passages from the 144 curated note entries",
          (idx["n_papers"], idx["n_passages"], idx["source_notes"]["entries"]) == (128, 432, 144),
          (idx["n_papers"], idx["n_passages"], idx["source_notes"]["entries"]))
    heads = [(root / p["file"]).read_text(encoding="utf-8").splitlines()[:8]
             for p in corpus.papers.values()]
    check("corpus: every file says its lines are notes, not verbatim paper text",
          all(any("NOT verbatim" in ln for ln in h) for h in heads))
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    check("corpus: README states the not-verbatim provenance", "not verbatim" in readme.lower())

    # Every passage resolves verbatim: its whole text, and a slice from inside it.
    bad = []
    for pid, line, kind, topic, text, _ in corpus.docs:
        body = C.PASSAGE_RE.match(text).group(3)
        mid = body[len(body) // 4: len(body) // 4 + 40]
        for q in (body, mid):
            ok, why = corpus.check_quote(pid, line, q)
            if not ok:
                bad.append((pid, line, why))
    check("corpus: every passage (whole and a 40-char slice) resolves as a quote", bad == [],
          bad[:3])

    seeds = {"2103.03098": ["L5", "D8"], "2403.08763": ["L1"], "2412.06712": ["L1"],
             "2206.14486": ["L3"], "2306.09683": ["L3", "L4"], "2304.07193": ["L3"],
             "2508.10104": ["L3"], "2503.04688": ["L1"], "1911.00068": ["L4"],
             "2103.14749": ["L4"]}
    missing = [(p, lv) for p, lvs in seeds.items() for lv in lvs
               if lv not in (corpus.papers.get(p) or {}).get("levers", [])]
    check("corpus: the protocol's seed papers are present with the levers they inform",
          missing == [], missing)
    check("corpus: aliases used by levers.json resolve (ibrahim2024 -> 2403.08763)",
          corpus.canonical_id("ibrahim2024") == "2403.08763"
          and corpus.canonical_id("time2024") == "2412.06712"
          and corpus.canonical_id("lwf2016") is None)
    check("corpus: a quote through an alias resolves", corpus.check_quote("ibrahim2024", 11,
                                                                         QUOTE)[0])

    check("corpus: a header line is not citable",
          not corpus.check_quote("2403.08763", 4, "Simple and Scalable Strategies")[0])
    check("corpus: a quote under 20 characters is refused",
          not corpus.check_quote("2403.08763", 11, "Every re-warm")[0])
    empty = [(pid, line, q) for pid, line, kind, topic, text, _ in corpus.docs[:60]
             for q in ("%s [%s]: " % (kind, topic),
                       "%s [%s]: %s" % (kind, topic, text.split(": ", 1)[1][:5]))]
    passed = [(pid, line, q) for pid, line, q in empty if corpus.check_quote(pid, line, q)[0]]
    check("corpus: a quote that is only the 'KIND [topic]: ' label (plus a few characters) is refused",
          passed == [], passed[:3])
    tr = [(pid, line) for pid, line, _, _, t, _ in corpus.docs if t.endswith(C.TRUNCATED_MARK)]
    check("corpus: the truncation mark alone is not a quote",
          tr and not any(corpus.check_quote(pid, line, C.TRUNCATED_MARK)[0] for pid, line in tr))
    check("corpus: passage kinds are known (a US line is the project's note)",
          (corpus.passage("2403.08763", 11) or {}).get("kind") == "RESULT"
          and (corpus.passage("2403.08763", 12) or {}).get("kind") == "US"
          and corpus.passage("2403.08763", 4) is None)
    src = root / idx["source_notes"]["path"]
    check("corpus: the source notes are committed beside the corpus with the sha256 the index "
          "records, and they rebuild it byte for byte",
          src.is_file() and C.check_source(root) == [], C.check_source(root))
    check("corpus: a paraphrase (one word changed) is refused",
          not corpus.check_quote("2403.08763", 11, QUOTE.replace("raises", "increases"))[0])
    check("corpus: an unknown paper is refused", not corpus.check_quote("9999.99999", 11, QUOTE)[0])
    check("corpus: a line past the end is refused",
          not corpus.check_quote("2403.08763", 999, QUOTE)[0])
    truncated = [t for _, _, _, _, t, _ in corpus.docs if t.endswith(C.TRUNCATED_MARK)]
    check("corpus: fields cut at collection are marked, not passed off as whole",
          len(truncated) > 0 and all(len(t) > 380 for t in truncated), len(truncated))

    q = "replay forgetting continual fine-tuning incremental warm-start learning rate"
    hits = corpus.search(q, k=8)
    check("corpus: BM25 finds continual-learning passages for a forgetting query",
          len(hits) == 8 and sum(h["topic"] == "continual" for h in hits) >= 5,
          [(h["paper_id"], h["topic"]) for h in hits])
    check("corpus: search is deterministic", hits == corpus.search(q, k=8))
    lh = corpus.search("label noise detection boxes", k=6, levers=["L4"])
    check("corpus: a lever filter ranks papers informing it first",
          lh and all("L4" in h["levers"] for h in lh), [(h["paper_id"], h["levers"]) for h in lh])
    check("corpus: an empty query returns nothing", corpus.search("the of and", k=5) == [])


def test_corpus_build():
    print("corpus build from curated notes")
    notes = _TMP / "notes.md"
    long_us = "This applies to the incremental loop " + "because the pool repeats " * 20
    notes.write_text("\n".join([
        "==================== TOPIC continual ====================",
        "- [OK|yes|adopt] A Replay Paper (TMLR 2024, 2024) https://arxiv.org/abs/2403.08763",
        "    RESULT: Replay of 5% matches union training.",
        "    US: " + long_us.strip(),
        "    CORRECTION: Confirmed against Table 2.",
        "SYNTHESIS: not a paper",
        "(1) also not a paper",
        "==================== TOPIC weeds ====================",
        "- [OK|yes|consider] A Weed Dataset (WeedSet) (Data in Brief, 2023) https://example.org/x",
        "    RESULT: 3,000 images of weeds.",
        "    US: Useful as an exam.",
        "    CORRECTION: Counts confirmed.",
        "- [OK|yes|adopt] A Replay Paper (arXiv 2403.08763 (v2), 2024) https://arxiv.org/html/2403.08763",
        "    RESULT: Seen again from the weeds sweep.",
        "    US: Applies to weeds.",
        "    CORRECTION: Same paper.",
    ]) + "\n", encoding="utf-8")
    out = _TMP / "lit"
    idx = C.build(notes, out, aliases={"replay2024": "2403.08763"})
    check("build: duplicate arXiv ids merge into one paper file", idx["n_papers"] == 2,
          idx["n_papers"])
    rp = C.Corpus(out)
    meta = rp.papers["2403.08763"]
    check("build: the merged paper carries both topics and the union of their levers",
          meta["topics"] == ["continual", "weeds"] and meta["levers"] == ["L1", "L2", "X1", "L3"],
          (meta["topics"], meta["levers"]))
    check("build: synthesis blocks are not passages", idx["n_passages"] == 9, idx["n_passages"])
    lines = (out / "2403.08763.md").read_text(encoding="utf-8").splitlines()
    check("build: a field cut near 400 characters is marked truncated",
          any(ln.startswith("US [continual]:") and ln.endswith(C.TRUNCATED_MARK) for ln in lines))
    check("build: a non-arXiv paper gets a title slug id",
          "weed-dataset-weedset" in rp.papers, sorted(rp.papers))
    check("build: the passage index points at passage lines",
          all(C.PASSAGE_RE.match(lines[p["line"] - 1]) for p in meta["passages"]))
    files = lambda d: {f.relative_to(d).as_posix(): f.read_bytes()   # noqa: E731
                       for f in sorted(d.rglob("*")) if f.is_file()}
    before = files(out)
    C.build(notes, out, aliases={"replay2024": "2403.08763"})
    check("build: a rebuild is byte-identical", before == files(out))
    src = out / C.SOURCE_FILE
    text = src.read_text(encoding="utf-8") if src.is_file() else ""
    check("build: the source extract is written, without the synthesis blocks, with its sha256 "
          "in the index",
          "SYNTHESIS" not in text and "also not a paper" not in text and "A Replay Paper" in text
          and idx["source_notes"]["path"] == C.SOURCE_FILE
          and idx["source_notes"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest(),
          idx["source_notes"])
    out2 = _TMP / "lit_from_extract"
    C.build(src, out2, aliases={"replay2024": "2403.08763"})
    check("build: building from the extract gives the same files as from the full notes",
          files(out2) == files(out))
    tampered = _TMP / "lit_tampered"
    shutil.copytree(str(out), str(tampered))
    ts = tampered / C.SOURCE_FILE
    ts.write_text(ts.read_text(encoding="utf-8").replace("3,000 images", "4,000 images"),
                  encoding="utf-8")
    check("build: an edited source is caught by its sha256",
          any("sha256" in p for p in C.check_source(tampered)), C.check_source(tampered))
    idx_t = json.loads((tampered / "index.json").read_text(encoding="utf-8"))
    idx_t["source_notes"]["sha256"] = hashlib.sha256(ts.read_bytes()).hexdigest()
    (tampered / "index.json").write_text(json.dumps(idx_t, indent=1, sort_keys=True) + "\n",
                                         encoding="utf-8")
    check("build: an edited source whose hash was updated is caught by the rebuild",
          any("differs" in p for p in C.check_source(tampered)), C.check_source(tampered))
    check("build: an alias to a paper not in the notes fails the build",
          raises(lambda: C.build(notes, _TMP / "lit2", aliases={"x": "1111.11111"}), ValueError,
                 "alias"))


# --- digest -----------------------------------------------------------------------

def _leaves(obj, ptr=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_at":
                continue
            yield from _leaves(v, "%s/%s" % (ptr, k.replace("~", "~0").replace("/", "~1")))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, "%s/%d" % (ptr, i))
    else:
        yield ptr, obj


def _digest(root, exp="pilot_v1", diags=(D1, D4), **kw):
    arts = BP.load_artifacts(root, [exp] + list(kw.pop("parents", [])))
    return arts, BP.build_digest(arts, exp, list(diags), MENU, corpus=C.Corpus(),
                                 created_utc="2026-09-27T00:00:00Z", **kw)


def test_digest():
    print("dev-only digest")
    root = inc_root(["pilot_v1"])
    if root is None:
        skip("digest cases", "pilot_v1 evidence not local")
        return None, None, None
    arts, dg = _digest(root)
    secs = {k: v for k, v in dg["sections"].items() if k != "literature"}
    check("digest: no non-dev exam key, stamp or path in any evidence section",
          BP.dev_leaks(secs) == [], BP.dev_leaks(secs)[:3])
    ev = dg["sections"]["evidence"]["pilot_v1"]
    check("digest: the per-step table, final dev rows and gate entries are there",
          len(ev.get("steps", [])) == 7 and len(ev.get("final_dev", [])) == 5
          and len(ev.get("gate_entries", [])) == 21,
          sorted(ev))
    check("digest: final rows show dev only",
          all(set(r["exams"]) == {"dev"} for r in ev["final_dev"]))
    check("digest: the prompt names no non-dev exam as a key",
          not re.search(r'"(test|ood22|ood23|imageweeds)":', dg["prompt"]))
    check("digest: fired and silent diagnoses are separated",
          [d["id"] for d in dg["sections"]["diagnoses"]["fired"]] == ["D1"]
          and [d["id"] for d in dg["sections"]["diagnoses"]["silent"]] == ["D4"])
    menu_ids = [r["id"] for r in dg["sections"]["menu"]]
    check("digest: the menu shows levers with bounds and cards without",
          menu_ids == ["L1", "L4", "L9", "X1"]
          and isinstance(dg["sections"]["menu"][0]["param_bounds"], dict)
          and "param_bounds" not in dg["sections"]["menu"][3], menu_ids)
    check("digest: a lever's literature alias is linked to its corpus id",
          dg["sections"]["menu"][0]["lit"][0].get("corpus_id") == "2403.08763")
    lit = dg["sections"]["literature"]
    corpus = C.Corpus()
    check("digest: literature passages are corpus lines, L1-informing papers first (D1 -> L1)",
          lit and all(corpus.line(h["paper_id"], h["line"]) == h["text"] for h in lit)
          and "L1" in corpus.papers[lit[0]["paper_id"]]["levers"], [h["paper_id"] for h in lit])
    tr = dg["trimmed"]
    check("digest: num_ctx is chosen from its size with the reply room free, nothing trimmed",
          dg["num_ctx"] == BP.choose_num_ctx(dg["tokens_estimated"])
          and tr["cuts"] == [] and tr["tokens_before"] == tr["tokens_after"] == dg["tokens_estimated"]
          and tr["num_ctx_from"] == "digest size" and "trimmed" not in dg["sections"]
          and "### TRIMMED" not in dg["prompt"]
          and dg["tokens_estimated"] * BP.CTX_HEADROOM + BP.REPLY_ROOM_TOKENS <= dg["num_ctx"],
          (dg["tokens_estimated"], dg["num_ctx"], tr))

    # Every value the digest shows is citable back exactly from its address.
    bad, n = [], 0
    for sec in ev.values():
        for row in sec:
            at = row["_at"]
            for ptr, value in _leaves(row):
                n += 1
                c = {"artifact": at["artifact"], "pointer": at["pointer"] + ptr, "value": value}
                if "line" in at:
                    c["line"] = at["line"]
                ok, why = V.resolve_cite(arts, c)
                if not ok:
                    bad.append((c["artifact"], c["pointer"], why))
    check("digest: all %d shown values resolve from their _at address" % n,
          n > 300 and bad == [], bad[:3])
    for d in dg["sections"]["diagnoses"]["fired"]:
        check("digest: diagnosis %s's cites resolve" % d["id"],
              all(V.resolve_cite(arts, c)[0] for c in d["cites"]))

    # Metamorphic: change every non-dev exam value; the digest must not move.
    rep_path = root / "pilot_v1" / "report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    changed = 0
    for row in rep.get("final", []):
        for exam in ("test", "ood22", "ood23", "imageweeds"):
            block = row.get("exams", {}).get(exam)
            if isinstance(block, dict):
                for arm in block.values():
                    if isinstance(arm, dict) and isinstance(arm.get("mean"), float):
                        arm["mean"] = 1.0 - arm["mean"]
                        arm["sd"] = 0.123
                        changed += 1
    root2 = inc_root(["pilot_v1"])
    (root2 / "pilot_v1" / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    _, dg2 = _digest(root2)
    check("digest: %d non-dev values changed, digest bytes identical" % changed,
          changed >= 20 and dg2["sha256"] == dg["sha256"] and dg2["prompt"] == dg["prompt"],
          (changed, dg2["sha256"][:12], dg["sha256"][:12]))

    leak = dict(D1, cites=[{"artifact": "pilot_v1/report.json",
                            "pointer": "/final/0/exams/test/twelve/mean", "value": 0.77}])
    check("digest: a diagnosis citing a test address stops the digest",
          raises(lambda: BP.build_digest(arts, "pilot_v1", [leak], MENU, corpus=None),
                 ValueError, "non-dev"))
    stamp = dict(D1, cites=[dict(D1["cites"][0], value={"exam": "ood23", "map50_95": 0.1})])
    check("digest: a score stamp of a non-dev exam stops the digest",
          raises(lambda: BP.build_digest(arts, "pilot_v1", [stamp], MENU, corpus=None),
                 ValueError, "non-dev"))
    # pilot_v1 alone: about 19K tokens in full, about 16K at its minimum (its
    # 21 per-species gate rows are never cut).
    n_small = 23000
    b_small = n_small - BP.REPLY_RESERVE_TOKENS
    _, small = _digest(root, num_ctx=n_small)
    cuts = small["trimmed"]["cuts"]
    stages = [c["stage"] for c in cuts]
    check("digest: a small fixed context trims in TRIM_ORDER (no older experiments: the "
          "literature first, down to its floor) and says so",
          small["num_ctx"] == n_small and small["tokens_estimated"] <= b_small
          and stages and stages[0] == "literature"
          and stages == sorted(stages, key=BP.TRIM_ORDER.index)
          and len(small["sections"]["literature"]) >= BP.LIT_FLOOR_K
          and small["sections"]["trimmed"]["cut"] == [c["cut"] for c in cuts]
          and small["trimmed"]["tokens_before"] > b_small >= small["trimmed"]["tokens_after"],
          (stages, small["tokens_estimated"]))
    check("digest: a context too small for the minimal digest refuses",
          raises(lambda: _digest(root, num_ctx=7000), ValueError, "minimum"))
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose as D
        from weed_optimizer_framework.tools.inc_autopilot import evidence as E
    except ImportError as exc:
        skip("digest: real diagnoses", "diagnose.py not importable: %s" % exc)
    else:
        ev = E.load_dir(root, "pilot_v1", exps=["pilot_v1"])
        ds = D.detect(ev)
        real = BP.build_digest(ev, "pilot_v1", ds, MENU, corpus=C.Corpus())
        ra = BP.artifacts_of(ev)
        fired = [d for d in ds if d["fired"]]
        bad = [(d["id"], c.get("artifact"), c.get("pointer")) for d in fired for c in d["cites"]
               if not V.resolve_cite(ra, c)[0]]
        check("digest: diagnose.detect's fired diagnoses (%s) render, and all their cites "
              "resolve for the validator" % ",".join(d["id"] for d in fired),
              fired and bad == [] and len(real["sections"]["diagnoses"]["fired"]) == len(fired),
              bad[:3])
    out = _TMP / "cli_digest.json"
    cli = subprocess.run([sys.executable, "-m", "weed_optimizer_framework.tools.inc_autopilot.brain_plan",
                          "digest", "--inc-dir", str(root), "--exp", "pilot_v1", "--out", str(out),
                          "--campaign", "c1", "--n", "2"], cwd=str(ROOT), capture_output=True,
                         text=True)
    got = json.loads(out.read_text()) if out.is_file() else {}
    check("digest: the CLI writes a staged digest", cli.returncode == 0
          and got.get("schema") == BP.DIGEST_SCHEMA and got.get("n") == 2, cli.stderr[-300:])
    s1 = RESULTS / "step1" / "select_summary.json"
    if s1.is_file():
        root3 = inc_root(["pilot_v1"])
        (root3 / "step1").mkdir()
        for f in ("select_summary.json", "admit_summary.json"):
            if (RESULTS / "step1" / f).is_file():
                (root3 / "step1" / f).write_bytes((RESULTS / "step1" / f).read_bytes())
        _, dg3 = _digest(root3)
        check("digest: Step 1 aggregates appear when the snapshot has them",
              "select_summary" in dg3["sections"]["evidence"].get("step1", {}))
    else:
        skip("digest: Step 1 aggregates", "step1/select_summary.json not local")
    if (RESULTS / "step1" / "relevance.json").is_file():
        root4 = inc_root(["pilot_v1"])
        (root4 / "step1").mkdir()
        (root4 / "step1" / "relevance.json").write_bytes(
            (RESULTS / "step1" / "relevance.json").read_bytes())
        _, dg4 = _digest(root4)
        rel = dg4["sections"]["evidence"].get("step1", {}).get("relevance", [{}])[0]
        check("digest: relevance statuses per source reach the brain",
              bool(((rel.get("increment_pool") or {}).get("sources"))))
    else:
        skip("digest: relevance statuses", "skip-until-fixture: step1/relevance.json not pulled")
    return arts, dg, root


# --- fitting the context ----------------------------------------------------------

# Every experiment of the campaign so far, oldest first. The live failure
# (2026-09-27 22:38Z, after realloop_v1) was a digest of about 46,923 tokens
# against the fixed num_ctx 49152 minus the 6000-token reply reserve.
CAMPAIGN_EXPS = ("b0_v1", "pilot_v1", "pilot_v2", "base_b_v1", "pilot_v3")
OLD_NUM_CTX = 49152
OLD_BUDGET = OLD_NUM_CTX - 6000
LINEAGE = [
    {"lever": "L1", "action": "inc_build_pilot", "parent_exp": "pilot_v1", "child_exp": "pilot_v2",
     "status": "executed", "approval_id": "ap-1", "proposal_id": "pr-1", "ts": "2026-09-27T03:00:00Z"},
    {"lever": "L8", "action": "inc_build_baseline", "parent_exp": "pilot_v2",
     "child_exp": "base_b_v1", "status": "executed", "approval_id": "ap-2", "proposal_id": "pr-2",
     "ts": "2026-09-27T09:00:00Z"},
    {"lever": "L9", "action": "inc_build_pilot", "parent_exp": "pilot_v2", "child_exp": "pilot_v3",
     "status": "executed", "approval_id": "ap-3", "proposal_id": "pr-3", "ts": "2026-09-27T11:00:00Z"},
    {"lever": "L2", "action": "inc_build_realloop", "parent_exp": "pilot_v3",
     "child_exp": "realloop_v1", "status": "executed", "approval_id": "ap-4", "proposal_id": "pr-4",
     "ts": "2026-09-27T14:00:00Z"}]


def _perturb_non_dev(obj, hot=False, parent=None):
    """(copy, n): every number under a non-dev exam (a key naming one, a non-dev
    entry of an "exams" dict, a score stamp of one) changed; n = values changed."""
    if isinstance(obj, dict):
        hot = hot or (isinstance(obj.get("exam"), str) and obj["exam"] != "dev")
        out, n = {}, 0
        for k, v in obj.items():
            out[k], m = _perturb_non_dev(v, hot or k in BP.FORBIDDEN_EXAMS
                                         or (parent == "exams" and k != "dev"), k)
            n += m
        return out, n
    if isinstance(obj, list):
        out, n = [], 0
        for v in obj:
            v2, m = _perturb_non_dev(v, hot, parent)
            out.append(v2)
            n += m
        return out, n
    if hot and isinstance(obj, float):
        return round(1.0 - obj + 0.0123, 9), 1
    if hot and isinstance(obj, int) and not isinstance(obj, bool):
        return obj + 7, 1
    return obj, 0


def _perturb_tree(root):
    """Change every non-dev exam value in every artifact under `root`; the count changed."""
    n = 0
    for p in sorted(root.rglob("*")):
        if p.suffix == ".json":
            obj, m = _perturb_non_dev(json.loads(p.read_text(encoding="utf-8")))
            p.write_text(json.dumps(obj), encoding="utf-8")
            n += m
        elif p.suffix == ".jsonl":
            lines = []
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    obj, m = _perturb_non_dev(json.loads(ln))
                    ln, n = json.dumps(obj), n + m
                lines.append(ln)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def _campaign_tree(exps):
    """A temp INC tree of `exps` plus the frozen Step 1 copies; None if any is not local."""
    root = inc_root(exps)
    if root is None:
        return None
    (root / "step1").mkdir()
    for f in sorted((FIXTURES / "step1_copies").glob("*.json")):
        (root / "step1" / f.name).write_bytes(f.read_bytes())
    return root


def _campaign_digest(root, exp, parents, num_ctx=None, diags=None):
    from weed_optimizer_framework.tools.inc_autopilot import diagnose as D
    ev = E.load_dir(root, exp, exps=list(parents) + [exp])
    ds = D.detect(ev) if diags is None else diags
    dg = BP.build_digest(ev, exp, ds, BP.load_menu(), campaign="c1", n=3, corpus=C.Corpus(),
                         parents=list(parents), num_ctx=num_ctx, lineage=LINEAGE,
                         track={"levers": {"L1": {"scored": 1, "correct": 1, "contradicted": 0,
                                                  "insufficient": 0, "accuracy": 1.0}}},
                         budget={"envelope_su": 300, "spent_su": 180.5, "committed_su": 20.0,
                                 "remaining_su": 99.5},
                         residuals=["L5 deferred: " + "no larger size is registered for this "
                                    "loop; the sizing rule needs a Step 1 pool report " * 3],
                         created_utc="2026-09-27T22:38:00Z")
    return ev, ds, dg


def _shown_values_resolve(dg, arts):
    """[(artifact, pointer, why)] of every evidence value that does not resolve from its _at."""
    bad = []
    for block in dg["sections"]["evidence"].values():
        for sec in block.values():
            for row in sec:
                at = row["_at"]
                for ptr, value in _leaves(row):
                    c = {"artifact": at["artifact"], "pointer": at["pointer"] + ptr, "value": value}
                    if "line" in at:
                        c["line"] = at["line"]
                    ok, why = V.resolve_cite(arts, c)
                    if not ok:
                        bad.append((c["artifact"], c["pointer"], why))
    return bad


def test_num_ctx():
    print("num_ctx: chosen from the digest's size, capped by what the job can hold")
    twice = BP.KV_SIZING_FACTOR * BP.KV_BYTES_PER_TOKEN_ASSUMED
    check("num_ctx: the cap is at most the planner's context and fits the job's 80 GB GPU "
          "under the assumed layout; the planner's full context would not",
          BP.MAX_NUM_CTX <= BP.PLANNER_CONTEXT_LENGTH
          and BP.gpu_mem_needed_gb(BP.MAX_NUM_CTX) <= BP.JOB_GPU_MEM_GB
          < BP.gpu_mem_needed_gb(BP.PLANNER_CONTEXT_LENGTH),
          (BP.gpu_mem_needed_gb(BP.MAX_NUM_CTX), BP.gpu_mem_needed_gb(BP.PLANNER_CONTEXT_LENGTH)))
    check("num_ctx: the layout is an assumption, so the cap is the largest step that still fits "
          "at twice the assumed KV per token (%.1f GB; the next step %.1f GB; 131072 %.1f GB)"
          % (BP.gpu_mem_needed_gb(BP.MAX_NUM_CTX, twice),
             BP.gpu_mem_needed_gb(BP.MAX_NUM_CTX + BP.NUM_CTX_STEP, twice),
             BP.gpu_mem_needed_gb(131072, twice)),
          BP.KV_SIZING_FACTOR >= 2 and BP.MAX_NUM_CTX % BP.NUM_CTX_STEP == 0
          and BP.gpu_mem_needed_gb(BP.MAX_NUM_CTX, twice) <= BP.JOB_GPU_MEM_GB
          < BP.gpu_mem_needed_gb(BP.MAX_NUM_CTX + BP.NUM_CTX_STEP, twice))
    # The job's layout line reads the model's own layout (ollama /api/show model_info).
    dense = {"general.architecture": "qwen3", "qwen3.block_count": 64,
             "qwen3.attention.head_count": 64, "qwen3.attention.head_count_kv": 8,
             "qwen3.attention.key_length": 128, "qwen3.attention.value_length": 128,
             "qwen3.embedding_length": 5120}
    wide = dict(dense, **{"qwen3.attention.head_count_kv": 16})
    hybrid = {"general.architecture": "qwen3next", "qwen3next.block_count": 64,
              "qwen3next.attention.head_count": 16, "qwen3next.attention.head_count_kv": 4,
              "qwen3next.attention.key_length": 256, "qwen3next.attention.value_length": 256,
              "qwen3next.full_attention_interval": 4}
    per_layer = {"general.architecture": "g", "g.block_count": 4, "g.attention.head_count": 8,
                 "g.embedding_length": 1024, "g.attention.head_count_kv": [2, 0, 2, 4]}
    got = {k: BP.kv_layout(v)[0] for k, v in (("dense", dense), ("wide", wide), ("hybrid", hybrid),
                                              ("per_layer", per_layer), ("none", {}))}
    check("layout: KV bytes per token from model_info (a dense 32B's = the assumed figure; 16 KV "
          "heads = twice; a 1-in-4 hybrid = a quarter of its layers; per-layer head counts; "
          "unreadable = None)",
          got == {"dense": BP.KV_BYTES_PER_TOKEN_ASSUMED, "wide": twice,
                  "hybrid": 16 * 4 * 512 * 2, "per_layer": 8 * 256 * 2, "none": None}, got)
    line_ok = BP.layout_line(dense, BP.MAX_NUM_CTX)
    line_over = BP.layout_line(wide, 131072)
    check("layout: the job's line compares the model's layout with the assumed one and flags a "
          "num_ctx its GPU cannot hold",
          "1.00x" in line_ok and "OVER THE GPU" not in line_ok and "2.00x" in line_over
          and "OVER THE GPU" in line_over and "unknown" in BP.layout_line({}, 49152),
          (line_ok, line_over))
    sizes = [0, 1, 9000, 26748, 34373, 46923, 60000, 80000, BP.AUTO_BUDGET_TOKENS]
    got = [BP.choose_num_ctx(t) for t in sizes]
    check("num_ctx: never below the old fixed 49152, never above the cap, a multiple of 8192, "
          "non-decreasing in the digest's size",
          all(BP.DEFAULT_NUM_CTX <= c <= BP.MAX_NUM_CTX and c % BP.NUM_CTX_STEP == 0 for c in got)
          and got == sorted(got) and got[0] == BP.DEFAULT_NUM_CTX, got)
    check("num_ctx: covers CTX_HEADROOM x the estimate plus the reply room, up to the auto budget",
          all(t * BP.CTX_HEADROOM + BP.REPLY_ROOM_TOKENS <= c for t, c in zip(sizes, got)
              if t <= BP.AUTO_BUDGET_TOKENS)
          and BP.choose_num_ctx(BP.AUTO_BUDGET_TOKENS) == BP.MAX_NUM_CTX
          and (BP.AUTO_BUDGET_TOKENS + 1) * BP.CTX_HEADROOM + BP.REPLY_ROOM_TOKENS > BP.MAX_NUM_CTX,
          list(zip(sizes, got)))
    check("num_ctx: the live failure's 46,923-token digest gets a window that holds it",
          BP.choose_num_ctx(46923) >= 46923 * BP.CTX_HEADROOM + BP.REPLY_ROOM_TOKENS
          and BP.choose_num_ctx(46923) > OLD_NUM_CTX, BP.choose_num_ctx(46923))
    check("num_ctx: a caller's num_ctx over the cap is refused, not trimmed to",
          raises(lambda: BP.build_digest({}, "x", [], MENU, num_ctx=BP.MAX_NUM_CTX + 1),
                 ValueError, "MAX_NUM_CTX"))


def _trim_case(exp, parents):
    """The whole campaign's real evidence: the full digest exceeds the old budget;
    it stages untrimmed with a larger num_ctx, trims to fit the old num_ctx in
    TRIM_ORDER, keeps every fired diagnosis cite, stays test-blind, and refuses
    only when its minimum does not fit."""
    tag = "trim %s (+%d older)" % (exp, len(parents))
    root = _campaign_tree(list(parents) + [exp])
    if root is None:
        skip(tag, "%s evidence not local" % exp)
        return
    ev, ds, full = _campaign_digest(root, exp, parents)
    arts = BP.artifacts_of(ev)
    fired = [d for d in ds if d.get("fired")]
    check("%s: the real campaign digest (%d est. tokens) is over the old budget %d"
          % (tag, full["tokens_estimated"], OLD_BUDGET), full["tokens_estimated"] > OLD_BUDGET)
    check("%s: chosen num_ctx holds it untrimmed (%d)" % (tag, full["num_ctx"]),
          full["trimmed"]["cuts"] == [] and OLD_NUM_CTX < full["num_ctx"] <= BP.MAX_NUM_CTX
          and full["num_ctx"] == BP.choose_num_ctx(full["tokens_estimated"])
          and all("steps" in full["sections"]["evidence"][p] for p in parents
                  if (arts.get("%s/report.json" % p) or {}).get("steps")), full["trimmed"])

    _, _, dg = _campaign_digest(root, exp, parents, num_ctx=OLD_NUM_CTX, diags=ds)
    tr = dg["trimmed"]
    cuts = tr["cuts"]
    stages = [c["stage"] for c in cuts]
    check("%s: at the old num_ctx it is trimmed to fit, not refused (%d -> %d tokens, %d cuts)"
          % (tag, tr["tokens_before"], tr["tokens_after"], len(cuts)),
          dg["num_ctx"] == OLD_NUM_CTX and tr["num_ctx_from"] == "caller"
          and tr["budget_tokens"] == OLD_BUDGET
          and tr["tokens_before"] == full["tokens_estimated"] > OLD_BUDGET
          >= tr["tokens_after"] == dg["tokens_estimated"], tr)
    check("%s: the cuts follow TRIM_ORDER, the oldest experiment's per-step table first"
          % tag, stages and stages == sorted(stages, key=BP.TRIM_ORDER.index)
          and stages[0] == "parent_steps" and cuts[0]["cut"].startswith("evidence/pilot_v1/steps"),
          [c["cut"] for c in cuts])
    check("%s: each cut records the token estimate before and after it, chained" % tag,
          cuts[0]["tokens_before"] == tr["tokens_before"]
          and cuts[-1]["tokens_after"] == tr["tokens_after"]
          and all(a["tokens_after"] == b["tokens_before"] for a, b in zip(cuts, cuts[1:]))
          and all(c["tokens_after"] < c["tokens_before"] for c in cuts), cuts)
    ev_full, ev_cut = full["sections"]["evidence"], dg["sections"]["evidence"]
    gone = [c["cut"].split(":")[0] for c in cuts if c["stage"] == "parent_steps"]
    named = {c["cut"].split(":")[0].split("/")[1] for c in cuts if c["cut"].startswith("evidence/")}
    check("%s: what the record says was cut is gone, and no evidence it does not name changed"
          % tag, gone and all(g.split("/")[2] not in ev_cut[g.split("/")[1]] for g in gone)
          and all(ev_cut.get(p) == ev_full[p] for p in ev_full if p not in named), gone)
    check("%s: the prompt names every cut in its TRIMMED section" % tag,
          dg["sections"]["trimmed"] == {"note": BP.TRIM_NOTE, "cut": [c["cut"] for c in cuts]}
          and "### TRIMMED" in dg["prompt"] and BP.check_staged(dg) == dg["prompt"])
    check("%s: the current experiment's dev tables, the menu with its track record, the "
          "budget and the lineage are kept whole" % tag,
          all(ev_cut[exp].get(k) == ev_full[exp].get(k) and ev_full[exp].get(k)
              for k in ("summary", "chains", "steps", "final_dev", "gate_entries"))
          and dg["sections"]["menu"] == full["sections"]["menu"]
          and dg["sections"]["budget"] == full["sections"]["budget"]
          and dg["sections"]["lineage"] == full["sections"]["lineage"])
    cites_in = {json.dumps(c, sort_keys=True) for d in dg["sections"]["diagnoses"]["fired"]
                for c in d["cites"]}
    want = [json.dumps(json.loads(json.dumps(c)), sort_keys=True) for d in fired for c in d["cites"]]
    unresolved = [(d["id"], c.get("artifact"), c.get("pointer")) for d in fired
                  for c in d["cites"] if not V.resolve_cite(arts, c)[0]]
    check("%s: every cite of the %d fired diagnoses (%s; %d cites) is kept and resolves"
          % (tag, len(fired), ",".join(d["id"] for d in fired), len(want)),
          fired and all(w in cites_in for w in want)
          and dg["sections"]["diagnoses"]["fired"] == full["sections"]["diagnoses"]["fired"]
          and unresolved == [], unresolved[:3])
    bad = _shown_values_resolve(dg, arts)
    check("%s: every value the trimmed digest shows still resolves from its _at" % tag,
          bad == [], bad[:3])
    secs = {k: v for k, v in dg["sections"].items() if k != "literature"}
    check("%s: no non-dev exam key, stamp or path in any trimmed section" % tag,
          BP.dev_leaks(secs) == [] and not re.search(r'"(test|ood22|ood23|imageweeds)":',
                                                      dg["prompt"]), BP.dev_leaks(secs)[:3])

    # The minimal digest: every cut made. Its size is what the refusal reports;
    # num_ctx = that size plus the reserve is the smallest that holds it.
    small = [None]

    def _too_small():
        try:
            _campaign_digest(root, exp, parents, num_ctx=8000, diags=ds)
        except ValueError as e:
            small[0] = str(e)
            return True
        return False

    m = re.search(r"digest is about (\d+) tokens even trimmed to its minimum",
                  small[0] or "") if _too_small() else None
    if m is None:
        check("%s: the minimal digest's size is reported by the refusal" % tag, False, small[0])
        return
    n_min = int(m.group(1)) + BP.REPLY_RESERVE_TOKENS
    _, _, mn = _campaign_digest(root, exp, parents, num_ctx=n_min, diags=ds)
    check("%s: the smallest num_ctx that holds the minimal digest is %d; one less refuses"
          % (tag, n_min),
          mn["tokens_estimated"] == n_min - BP.REPLY_RESERVE_TOKENS
          and raises(lambda: _campaign_digest(root, exp, parents, num_ctx=n_min - 1, diags=ds),
                     ValueError, "even trimmed to its minimum"),
          (mn["tokens_estimated"], len(mn["trimmed"]["cuts"])))
    ms, fs = mn["sections"], full["sections"]
    mstages = [c["stage"] for c in mn["trimmed"]["cuts"]]
    check("%s: the minimal digest made cuts in every stage of TRIM_ORDER, down to the "
          "residuals (%d cuts)" % (tag, len(mstages)),
          set(mstages) == set(BP.TRIM_ORDER)
          and any(c["cut"].startswith("lineage:") for c in mn["trimmed"]["cuts"])
          and any(c["cut"].startswith("residuals:") for c in mn["trimmed"]["cuts"]),
          [c["cut"][:50] for c in mn["trimmed"]["cuts"]])
    check("%s: minimal digest, fired diagnoses: exactly the untrimmed ones, every cite (%d)"
          % (tag, len(want)),
          ms["diagnoses"]["fired"] == fs["diagnoses"]["fired"]
          and all(w in {json.dumps(c, sort_keys=True) for d in ms["diagnoses"]["fired"]
                        for c in d["cites"]} for w in want))
    kept = ("summary", "chains", "steps", "final_dev", "gate_entries", "exp", "state")
    check("%s: minimal digest, the current experiment's dev tables are unchanged (%s)"
          % (tag, ", ".join(k for k in kept if k in fs["evidence"][exp])),
          all(ms["evidence"][exp].get(k) == fs["evidence"][exp].get(k) for k in kept)
          and all(fs["evidence"][exp].get(k) for k in ("summary", "chains", "steps", "final_dev",
                                                       "gate_entries", "exp"))
          and set(BP.CURRENT_KEPT) == set(kept)
          and set(ms["evidence"][exp]) == {k for k in kept if k in fs["evidence"][exp]},
          sorted(ms["evidence"][exp]))
    check("%s: minimal digest, the menu with its track record, the budget and the "
          "deterministic proposals are unchanged" % tag,
          ms["menu"] == fs["menu"] and ms["budget"] == fs["budget"] and fs["budget"]
          and ms["deterministic"] == fs["deterministic"])
    check("%s: minimal digest, the lineage is its summary (%s), every record kept"
          % (tag, ", ".join(BP.LINEAGE_SUMMARY_KEYS)),
          fs["lineage"] and ms["lineage"] == [{k: r.get(k) for k in BP.LINEAGE_SUMMARY_KEYS}
                                              for r in fs["lineage"]]
          and all(r["child_exp"] for r in ms["lineage"]), ms["lineage"][:1])
    check("%s: minimal digest, %d literature passages, the top-ranked ones"
          % (tag, BP.LIT_FLOOR_K),
          BP.LIT_FLOOR_K > 0 and len(fs["literature"]) > BP.LIT_FLOOR_K
          and ms["literature"] == fs["literature"][:BP.LIT_FLOOR_K])
    want_lines = {p: {k: v for k, v in fs["evidence"][p]["summary"][0].items()
                      if k in BP.PARENT_LINE_KEYS or k == "_at"}
                  for p in parents if (fs["evidence"].get(p) or {}).get("summary")}
    check("%s: minimal digest, every older experiment with a report keeps its one-line "
          "summary (%d)" % (tag, len(want_lines)),
          want_lines and len(want_lines) == len([p for p in parents if "%s/report.json" % p in arts])
          and all(ms["evidence"].get(p) == {"summary": [line]} for p, line in want_lines.items()),
          {p: ms["evidence"].get(p) for p in parents})
    unresolved_min = [(d["id"], c.get("pointer")) for d in ms["diagnoses"]["fired"]
                      for c in d["cites"] if not V.resolve_cite(arts, c)[0]]
    check("%s: minimal digest, every fired cite and every shown value resolves, no non-dev "
          "exam anywhere, and the staged prompt is the rendering of its sections" % tag,
          unresolved_min == [] and _shown_values_resolve(mn, arts) == []
          and BP.dev_leaks({k: v for k, v in ms.items() if k != "literature"}) == []
          and BP.check_staged(mn) == mn["prompt"] and "### TRIMMED" in mn["prompt"],
          unresolved_min[:3])

    # Squeezed to where the minimal digest's record shows the last older
    # experiment cut to its line: every older experiment is its one-line
    # summary, never removed, and nothing after parent_tables was cut.
    pt = [c for c in mn["trimmed"]["cuts"] if c["stage"] == "parent_tables"]
    n_sq = (pt[-1]["tokens_after"] if pt else n_min) + BP.REPLY_RESERVE_TOKENS
    _, _, tight = _campaign_digest(root, exp, parents, num_ctx=n_sq, diags=ds)
    lines = {p: tight["sections"]["evidence"].get(p) for p in parents}
    check("%s: squeezed to %d, older experiments are cut to their one-line summaries, "
          "never removed, and nothing later in TRIM_ORDER is cut" % (tag, n_sq),
          pt and [c["stage"] for c in tight["trimmed"]["cuts"]]
          == ["parent_steps"] * len([c for c in mn["trimmed"]["cuts"] if c["stage"] == "parent_steps"])
          + ["parent_tables"] * len(pt)
          and all(set(b) <= {"summary"} for b in lines.values())
          and all(set(b["summary"][0]) <= set(BP.PARENT_LINE_KEYS) | {"_at"}
                  for b in lines.values() if b)
          and all(lines[p] for p in parents
                  if "%s/report.json" % p in arts)
          and tight["sections"]["literature"] == full["sections"]["literature"]
          and _shown_values_resolve(tight, arts) == []
          and tight["tokens_estimated"] <= n_sq - BP.REPLY_RESERVE_TOKENS,
          ([c["stage"] for c in tight["trimmed"]["cuts"]], {p: sorted(b) for p, b in lines.items()}))

    # Test-blind: every non-dev value in every artifact changes; nothing the
    # digest holds or cuts may move.
    root2 = _campaign_tree(list(parents) + [exp])
    changed = _perturb_tree(root2)
    _, _, full2 = _campaign_digest(root2, exp, parents)
    _, _, dg2 = _campaign_digest(root2, exp, parents, num_ctx=OLD_NUM_CTX)
    _, _, mn2 = _campaign_digest(root2, exp, parents, num_ctx=n_min)
    check("%s: %d non-dev values changed across the campaign; the full, the trimmed and the "
          "minimal digests are byte-identical, cuts included" % (tag, changed),
          changed > 100 and full2["sha256"] == full["sha256"] and dg2["sha256"] == dg["sha256"]
          and dg2["prompt"] == dg["prompt"] and dg2["trimmed"] == dg["trimmed"]
          and full2["num_ctx"] == full["num_ctx"]
          and mn2["sha256"] == mn["sha256"] and mn2["trimmed"] == mn["trimmed"],
          (changed, dg2["sha256"][:12], dg["sha256"][:12]))

    # The second limit: the ssh line that stages the digest (executor._plan_segment).
    from weed_optimizer_framework.tools.inc_autopilot import campaign as CP
    from weed_optimizer_framework.tools.inc_autopilot import executor as X
    import shlex
    import types
    xctx = types.SimpleNamespace(campaign_dir=pathlib.Path(tempfile.mkdtemp(dir=str(_TMP))))
    CP._write_json(X.plan_input_path(xctx, "c1", 3), full)
    seg = X._plan_segment(xctx, "inc_plan_submit",
                          {"campaign": "c1", "n": 3, "digest_sha256": full["sha256"], "model": ""})
    shipped = shlex.split(seg)[-1]
    check("%s: the untrimmed digest stages through the executor's ssh line; staged_chars is "
          "exactly what it ships (%d of %d)" % (tag, len(shipped), X.PLAN_MAX_STAGED_CHARS),
          len(shipped) == BP.staged_chars(full) <= BP.STAGED_BUDGET_CHARS
          and BP.STAGED_MAX_CHARS == X.PLAN_MAX_STAGED_CHARS
          and full["trimmed"]["staged_chars_before"] >= len(shipped) - 64,
          (len(shipped), BP.staged_chars(full)))
    saved = BP.STAGED_BUDGET_CHARS
    squeeze = BP.staged_chars(full) - 6000
    try:
        BP.STAGED_BUDGET_CHARS = squeeze
        _, _, tdg = _campaign_digest(root, exp, parents, diags=ds)
        BP.STAGED_BUDGET_CHARS = 5000
        refused = raises(lambda: _campaign_digest(root, exp, parents, diags=ds), ValueError,
                         "ssh command line")
    finally:
        BP.STAGED_BUDGET_CHARS = saved
    tc = tdg["trimmed"]["cuts"]
    check("%s: a digest within its context but over the staging line is trimmed for the "
          "line (over: transport), in TRIM_ORDER, and then stages" % tag,
          tc and all(c["over"] == "transport" for c in tc)
          and tc[0]["cut"].startswith("evidence/pilot_v1/steps")
          and [c["stage"] for c in tc] == sorted((c["stage"] for c in tc), key=BP.TRIM_ORDER.index)
          and all(a["staged_chars_before"] > b["staged_chars_before"] for a, b in zip(tc, tc[1:]))
          and BP.staged_chars(tdg) <= squeeze
          and tdg["num_ctx"] == BP.choose_num_ctx(tdg["tokens_estimated"])
          and tdg["sections"]["diagnoses"]["fired"] == full["sections"]["diagnoses"]["fired"],
          [(c["cut"][:40], c.get("staged_chars_before")) for c in tc])
    check("%s: a minimal digest over the staging line refuses" % tag, refused)

    # Refusal only when even the minimal digest does not fit.
    check("%s: a num_ctx too small for the minimal digest refuses" % tag,
          raises(lambda: _campaign_digest(root, exp, parents, num_ctx=12000, diags=ds),
                 ValueError, "even trimmed to its minimum"))
    huge = dict(fired[0], id="DX", summary="x" * int((BP.AUTO_BUDGET_TOKENS + 1000) * 2.76))
    check("%s: a minimal digest over the largest context the job can hold refuses" % tag,
          raises(lambda: _campaign_digest(root, exp, parents, diags=ds + [huge]),
                 ValueError, "MAX_NUM_CTX"))


def test_digest_trim():
    print("fitting the digest to its context (the whole campaign's evidence)")
    # The fixtures alone (sha-pinned): pilot_v3 with every older experiment.
    _trim_case("pilot_v3", CAMPAIGN_EXPS[:-1])
    # The live failure's experiment, realloop_v1 (results/, read only), with all five.
    _trim_case("realloop_v1", CAMPAIGN_EXPS)


# --- reply, validation, merge --------------------------------------------------

def test_parse():
    print("reply parsing")
    body = json.dumps({"ranked_menu": [GOOD_L1], "off_menu": [], "stop_recommendation": None})
    for label, text in (("bare", body), ("fenced", "```json\n%s\n```" % body),
                        ("think block", "<think>let me see {\"x\": 1}</think>\n" + body),
                        ("prose around", "Here is the plan:\n%s\nThanks." % body)):
        plan, probs = BP.parse_reply(text)
        check("parse: %s JSON is read" % label,
              plan is not None and plan["ranked_menu"][0]["lever"] == "L1", probs)
    plan, probs = BP.parse_reply('{"ranked_menu": {"lever": "L1"}, "extra": 1}')
    check("parse: a non-list section is emptied and reported",
          plan["ranked_menu"] == [] and len(probs) == 2, probs)
    check("parse: no JSON -> no plan", BP.parse_reply("I cannot help.")[0] is None)
    check("parse: empty -> no plan", BP.parse_reply("")[0] is None)


def test_validate(arts, root):
    print("validation of a brain plan (mock client)")
    corpus = C.Corpus()
    # levers.price's figure: the rebuild's runs plus the build job's own walltime.
    est = L.price("L1", {"exp": "pilot_v3"}, E.load_dir(root, "pilot_v1", exps=["pilot_v1"]), "pilot_v1")[0]
    val = V.validate({"schema": BP.REPLY_SCHEMA, "model": MODEL, "plan": PLAN,
                      "digest_sha256": "abc"}, arts, MENU, corpus, diagnoses=[D1, D4])
    reasons = {(d["section"], d["index"]): " | ".join(d["reasons"]) for d in val["dropped"]}
    check("validate: exactly the good menu item survives", len(val["menu"]) == 1,
          [p["lever"] for p in val["menu"]])
    p = val["menu"][0] if val["menu"] else {}
    check("validate: the survivor is a tier2 proposal the policy accepts as an actor",
          p.get("proposed_by") == "tier2:qwen3.8_27b"
          and policy._ACTOR_RE.fullmatch(p.get("proposed_by", "")) is not None, p.get("proposed_by"))
    check("validate: params carry the lever's fixed replay_mode and the autopilot's price",
          p.get("params") == {"exp": "pilot_v3", "est_gpu_hours": est, "replay_mode": "full"},
          (p.get("params"), est))
    check("validate: the price is levers.price's (rebuild plus build job) from pilot_v1's measured GPU "
          "time (%.3f GPU-h), with its cites" % est,
          p.get("est_gpu_hours") == est and est > 0
          and (p.get("estimate") or {}).get("estimator") == "pilot_rebuild"
          and (p.get("estimate") or {}).get("cites"), p.get("estimate"))
    check("validate: the proposal carries lever, action, risk, trigger, cites, lit, prediction",
          (p.get("lever"), p.get("policy_action"), p.get("risk"), p.get("trigger"), p.get("basis"))
          == ("L1", "inc_build_pilot", "R3", ["D1"], "diagnosis") and len(p.get("cites", [])) == 2
          and p.get("lit") == [{"paper_id": "2403.08763", "line": 11, "quote": QUOTE,
                                "kind": "RESULT"}]
          and p.get("predicted", {}).get("direction") == "up",
          {k: p.get(k) for k in ("lever", "policy_action", "risk", "trigger", "basis", "lit")})
    check("validate: the command is rendered, with the parent and child experiments",
          p.get("argv") == ["python", "-m", "weed_optimizer_framework.tools.inc.pilot", "build",
                            "--exp", "pilot_v3", "--replay-mode", "full"]
          and p.get("parent_exp") == "pilot_v1" and p.get("child_exp") == "pilot_v3",
          (p.get("argv"), p.get("parent_exp"), p.get("child_exp")))
    auth = policy.authorize(p.get("proposed_by"), "inc_build_pilot",
                            dict(p.get("params", {}), replay_mode="full"))
    check("validate: filing it as tier2 needs approval (R3), it is not direct",
          auth.get("allowed") and auth.get("needs_approval")
          and (p.get("policy_check") or {}).get("needs_approval") is True, auth)
    expect = {1: "holds", 2: "does not match the required pattern",
              3: "contradicts the lever's fixed value", 4: "test leak", 5: "test leak",
              6: "off-menu", 7: "not on the menu", 8: "verbatim", 9: "not a passage line",
              10: "predicted", 11: "holds True", 12: "no bounds", 13: "no evidence cite",
              14: "takes no param 'recipes'", 15: "duplicate of ranked item 1",
              16: "passage's own text"}
    for i, needle in sorted(expect.items()):
        got = reasons.get(("ranked_menu", i), "")
        check("validate: ranked item %d dropped (%s)" % (i, needle), needle in got, got)
    check("validate: a bad cite drops only its item (plan survives)",
          ("ranked_menu", 1) in reasons and len(val["menu"]) == 1)
    check("validate: one R4 card from the off-menu idea", len(val["cards"]) == 1,
          len(val["cards"]))
    card = val["cards"][0] if val["cards"] else {}
    exp = card.get("experiment") or {}
    check("validate: the card has model.card's shape (as the deterministic cards)",
          set(M.card("X", "t", [], [])) <= set(card) and card.get("proposed_by")
          == "tier2:qwen3.8_27b", sorted(card))
    check("validate: the card's draft is planner.make_experiment's, at R4",
          card.get("risk") == "R4" and exp.get("risk") == "R4"
          and exp.get("control") == GOOD_OFF["control"]
          and exp.get("success_criterion") == GOOD_OFF["success_criterion"]
          and exp.get("recipe", "").startswith("off-menu: "), exp)
    check("validate: the card's alias quote is stored under the corpus id",
          card.get("lit") == [{"paper_id": "2403.08763", "line": 11, "quote": QUOTE,
                               "kind": "RESULT"}], card.get("lit"))
    check("validate: an off-menu idea without a control is refused by make_experiment",
          "names no control" in reasons.get(("off_menu", 1), ""), reasons.get(("off_menu", 1)))
    check("validate: an off-menu idea naming ood22 is refused",
          "test leak" in reasons.get(("off_menu", 2), ""))
    q = approvals.propose("weed", "inc_card", {}, "R4", card.get("proposed_by"), "card", 1,
                          base_dir=str(_TMP / "ap"))
    check("validate: an R4 card cannot be queued (approvals refuses R4)", q.get("ok") is False, q)
    c = val["counts"]
    check("validate: counts add up",
          (c["items"], c["valid"], c["cards"], c["dropped"]) == (20, 1, 1, 18)
          and c["leaks"] == 3 and c["cites_failed"] == 2 and c["lit_failed"] == 3
          and c["duplicates"] == 1, c)
    check("validate: the stop recommendation is kept",
          val["stop_recommendation"] == {"stop": False, "reason": "the goal is not met yet"})
    none = V.validate({"schema": BP.REPLY_SCHEMA, "model": MODEL, "plan": None}, arts, MENU,
                      corpus)
    check("validate: no plan -> ok False, nothing filed", none["ok"] is False and not none["menu"])
    nocorp = V.validate({"ranked_menu": [GOOD_L1]}, arts, MENU, None, model=MODEL,
                        diagnoses=[D1])
    check("validate: without a corpus a literature quote cannot pass",
          not nocorp["menu"] and "no literature corpus" in " ".join(nocorp["dropped"][0]["reasons"]))
    real = BP.load_menu()
    if real:
        rows = [r for r in real.values() if r["menu"] == "menu"]
        check("validate: every menu lever of the live levers.json has proposer bounds",
              rows and all(BP.lever_bounds(r)[0] is not None for r in rows),
              [r["id"] for r in rows if BP.lever_bounds(r)[0] is None])
        check("validate: every X card of the live levers.json is off-menu",
              all(real[x]["menu"] == "off_menu" for x in real if x.startswith("X")))
    else:
        skip("validate: live levers.json", "levers.json not present")
    return val


def _item(lever, params, cites=None, **kw):
    it = {"lever": lever, "params": params, "rationale": "see the cited evidence",
          "evidence_cites": cites if cites is not None else [GOOD_L1["evidence_cites"][0]],
          "predicted": {"metric": "agreement", "direction": "none", "magnitude": None},
          "falsifier": "the cited value does not move"}
    it.update(kw)
    return it


def _val(items, arts, menu=None, diagnoses=(D1, D4), **kw):
    return V.validate({"schema": BP.REPLY_SCHEMA, "model": MODEL, "digest_sha256": "d",
                       "plan": {"ranked_menu": list(items), "off_menu": list(kw.pop("off", []))}},
                      arts, MENU if menu is None else menu, C.Corpus(), diagnoses=list(diagnoses),
                      **kw)


def _why(v, i=0):
    return " | ".join(v["dropped"][i]["reasons"]) if len(v["dropped"]) > i else ""


def test_validate_governance(arts, root):
    print("validation: price, preconditions, filing, triggers and leaks")
    ev = E.load_dir(root, "pilot_v1", exps=["pilot_v1"])
    est = L.price("L1", {"exp": "pilot_v3"}, ev, "pilot_v1")[0]

    # The price is the autopilot's, whatever the brain says.
    for label, given in (("no price", None), ("a price of 0.001", 0.001), ("a price of 0", 0.0)):
        params = {"exp": "pilot_v3"}
        if given is not None:
            params["est_gpu_hours"] = given
        v = _val([variant(GOOD_L1, params=params)], arts)
        p = v["menu"][0] if v["menu"] else {}
        check("price: a brain L1 with %s is priced from the evidence (%.3f GPU-h), its own figure "
              "kept aside" % (label, est),
              p.get("est_gpu_hours") == est and p.get("params", {}).get("est_gpu_hours") == est
              and p.get("brain_est_gpu_hours") == given, (p.get("est_gpu_hours"), _why(v)))
    no_rep = {k: x for k, x in arts.items() if k != "pilot_v1/report.json"}
    v = _val([variant(GOOD_L1, evidence_cites=[GOOD_L1["evidence_cites"][1]])], no_rep)
    check("price: a build the evidence cannot price is dropped, never priced 0",
          not v["menu"] and v["counts"]["unpriced"] == 1 and "deferred" in _why(v), _why(v))

    # Filed through the executor, the priced brain build charges the envelope.
    try:
        from weed_optimizer_framework.tools.inc_autopilot import budget as B
        from weed_optimizer_framework.tools.inc_autopilot import executor as X
    except ImportError as exc:
        skip("price: executor charges the envelope", "executor not importable: %s" % exc)
    else:
        v = _val([GOOD_L1], arts)
        prop = v["menu"][0]
        lab = _TMP / "lab_budget"
        calls = []

        def slurm_sh(script, timeout=None):
            calls.append(script)
            return {"ok": True, "returncode": 0, "stderr": "",
                    "stdout": "INCAP_SEG 0\nINCAP " + json.dumps({"ok": True, "job_id": "123"})
                              + "\nINCAP_SEG_END 0 0\n"}
        ctx = X.Context(slurm_sh=slurm_sh, resources={"mongo_ok": True, "cluster_reachable": True},
                        lab_repo=str(lab), clock=lambda: 1790000000.0)
        camp = {"name": "c1", "autonomy": "off", "envelope_su": 300, "daily_cap_su": 120}
        r = X.submit(prop, actor=prop["proposed_by"], campaign=camp, ctx=ctx)
        check("price: the brain proposal is filed for approval at its computed price",
              r["status"] == "filed" and r["est_su"] == est and r["approval_id"], r)
        if r.get("approval_id"):
            approvals.decide("weed", r["approval_id"], "approve", "human:owner@example.org", "ok",
                             1790000100.0, root=ctx.approvals_root)
            e = X.execute_approved(r["approval_id"], camp, ctx,
                                   invoked_by="human:owner@example.org")
            st = B.state(camp, X.executions(ctx), None, 1790000200.0, "weed", ctx.su_base_dir)
            check("price: once approved and run, the build is charged to the campaign envelope",
                  e["status"] == "executed" and e["charged"]
                  and abs((st.get("committed_su") or 0) + (st.get("spent_su") or 0) - est) < 1e-9
                  and abs(st.get("remaining_su") - (300 - est)) < 1e-9, st)
            check("price: the cluster provenance names the parent experiment and the trigger",
                  calls and "--parent-exp pilot_v1" in calls[-1] and "--trigger D1" in calls[-1],
                  calls[-1:] and calls[-1][-300:])

    # Preconditions, on the live menu and the real diagnoses of pilot_v1.
    live = BP.load_menu()
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG
        ds = DG.detect(ev)
    except ImportError as exc:
        skip("preconditions", "diagnose.py not importable: %s" % exc)
        ds = None
    if live and ds is not None:
        by = {d["id"]: d for d in ds}
        base = M.CLUSTER_INC_DIR + "/step1/base_B.jsonl"
        rl = {"exp": "realloop_v9", "base": base, "replay_mode": "sample", "recipes": "full"}
        v = _val([_item("L6", dict(rl))], arts, live, ds)
        check("preconditions: L6 (only after D11) is refused while D11 is silent",
              not by["D11"]["fired"] and "only after D11" in _why(v), _why(v))
        v = _val([_item("L5", dict(rl))], arts, live, ds)
        check("preconditions: L5 without the size it requires is refused",
              "requires the param(s) size" in _why(v), _why(v))
        v = _val([_item("L2", dict(rl))], arts, live, ds)
        check("preconditions: no L2 while D4 fired no_recipe_tracks_truth (R4a)",
              by["D4"]["name"] == "no_recipe_tracks_truth" and "needs D4 fired as decision_slot_ready"
              in _why(v) and v["counts"]["precondition_refusals"] == 1, _why(v))
        rows = {r["id"]: r for r in BP.menu_section(live)}
        check("preconditions: the digest's menu shows only_after, requires and the D4 gate, and no "
              "price param",
              rows["L6"].get("preconditions") == {"only_after": ["D11"],
                                                  "needs_D4": "decision_slot_ready"}
              and rows["L5"]["preconditions"].get("requires") == ["size"]
              and "est_gpu_hours" not in rows["L1"]["param_bounds"]
              and "est_gpu_hours" in rows["L1"]["set_by_autopilot"], rows["L6"].get("preconditions"))

        # L4: derived manifest paths, rendered and authorised for tier2.
        if "inc_label_audit" in _POLICY["actions"]:
            v = _val([_item("L4", {"exp": "pilot_v1"})], arts, live, ds)
            p = v["menu"][0] if v["menu"] else {}
            check("filing: a brain L4 renders run_inc_audit.sh with the derived manifests, priced "
                  "by its walltime, and tier2 may file it",
                  p.get("argv", [])[:3] == ["sbatch", "run_inc_audit.sh", "--trusted"]
                  and p.get("argv", [])[3].endswith("/pilot_v1/manifests/P0.jsonl")
                  and p.get("argv", [])[-1].endswith("/pilot_v1/audit/label_audit.json")
                  and p.get("est_gpu_hours") == 2.0 and p.get("parent_exp") == "pilot_v1"
                  and p["policy_check"]["needs_approval"], (p.get("argv"), _why(v)))
        else:
            skip("filing: L4", "no inc_label_audit row in the policy table")
        if "inc_unblock_transient" in _POLICY["actions"]:
            v = _val([_item("L7", {"exp": "pilot_v1", "unit": "base", "cause": "transient"})], arts,
                     live, ds)
            check("filing: a brain L7 is dropped: tier2 may not request an unblock",
                  not v["menu"] and "may not request action 'inc_unblock_transient'" in _why(v),
                  _why(v))
        v = _val([_item("L8", {"exp": "base_b_v9", "manifest": "/ocean/projects/cis240145p/byler/"
                               "harry/weed_llm_benchmark/results/framework/inc/splits/v1/test.jsonl"})],
                 arts, live, ds)
        check("leaks: a split-manifest path (splits/v1/test.jsonl) in a param is refused",
              v["counts"]["leaks"] == 1 and "test.jsonl" in _why(v), _why(v))

        # L2 with a ready D4 and a relevance.json matching the select build. The Step 1 summaries are the
        # pinned byte copies of the real ones (tests/fixtures/inc_replay/step1_copies, MANIFEST.json sha256),
        # not the untracked results/ files, so these checks run on every checkout.
        s1 = FIXTURES / "step1_copies" / "select_summary.json"
        if s1.is_file() and "inc_build_realloop" in _POLICY["actions"]:
            r2 = inc_root(["pilot_v1"])
            (r2 / "step1").mkdir()
            sel = json.loads(s1.read_text(encoding="utf-8"))
            (r2 / "step1" / "select_summary.json").write_text(json.dumps(sel), encoding="utf-8")
            # A relevance.json made for this select build, by levers.relevance_state's rule.
            outs = sel.get("outputs") or {}
            tables = getattr(L, "RELEVANCE_TABLES", (("increment_pool", "increment_pool.jsonl"),))
            inputs = {t: {"sha256": outs[f]["sha256"]} for t, f in tables
                      if (outs.get(f) or {}).get("sha256")}
            try:
                fmt = L.protocol("relevance_format")
            except KeyError:
                fmt = "inc.relevance/1"
            # made under the protocol's rule (params), with a consistent passing check
            rule = {"min_crops": L.protocol("relevance_min_crops"),
                    "calibration_percentile": L.protocol("relevance_calibration_percentile"),
                    "tau_min": L.protocol("relevance_tau_min")}
            (r2 / "step1" / "relevance.json").write_text(json.dumps(
                {"format": fmt, "inputs": inputs, "params": rule,
                 "calibration": {"tau": 0.62, "check": {"ok": True, "tau_min": rule["tau_min"]}}}),
                encoding="utf-8")
            a2 = BP.load_artifacts(r2, ["pilot_v1"])
            ev_r = E.load_dir(r2, "pilot_v1", exps=["pilot_v1"])
            try:
                rel_state = L.relevance_state(ev_r)
            except Exception as exc:          # the levers component's own rule is mid-change
                rel_state = "unreadable (%s: %s)" % (type(exc).__name__, exc)
            ready = dict(by["D4"], name="decision_slot_ready", levers=["L2"],
                         detail=dict(by["D4"]["detail"], recipes=["full"]))
            d2 = [d for d in ds if d["id"] != "D4"] + [ready]
            check("filing: levers.relevance_state accepts the relevance.json made for this select "
                  "build", rel_state == "matching", rel_state)
            v = _val([_item("L2", {"size": 400}, trigger=["D4"],
                            cites=[ready["cites"][0]])], a2, live, d2)
            p = v["menu"][0] if v["menu"] else {}
            ev2 = E.load_dir(r2, "pilot_v1", exps=["pilot_v1"])
            want_est = L.with_build_job(*L.estimate_realloop(
                ev2, dict(p.get("params") or {}, est_gpu_hours=None), L.load_menu())[:2])[0] if p else None
            check("filing: with D4 ready, a brain L2 takes D4's replay mode and recipes and Step "
                  "1's base and relevance, keeps its own size, and is priced by estimate_realloop plus the build job",
                  p.get("params", {}).get("recipes") == "full"
                  and p["params"].get("replay_mode") == ready["detail"]["replay_mode"]
                  and p["params"].get("base", "").endswith("/step1/base_B.jsonl")
                  and p["params"].get("relevance", "").endswith("/step1/relevance.json")
                  and p["params"].get("size") == 400 and p.get("est_gpu_hours") == want_est
                  and p.get("trigger") == ["D4"] and p.get("parent_exp") == "pilot_v1",
                  (p.get("params"), _why(v)))
            v = _val([_item("L2", {"size": 400, "recipes": "lora"})], a2, live, d2)
            check("filing: a brain L2 may not pick recipes other than D4's",
                  not v["menu"] and "not the plan's to choose" in _why(v), _why(v))
            v = _val([_item("L2", {"size": 400, "increment_sources": "evidence"}, trigger=["D4"],
                            cites=[ready["cites"][0]])], a2, live, d2)
            check("filing: a brain L2 may not pick the relevance criterion (Step 1 gives --relevance here)",
                  not v["menu"] and "not the plan's to choose" in _why(v), _why(v))
            # The same select build with the cluster's failed relevance build (tau 0.0557 < 0.5) and
            # Step 1's own admit summary: the source-level species evidence criterion, whose evidenced
            # pool (1,439 images) holds N 4 x M 287 but not the default N 6 x M 393.
            a1 = FIXTURES / "step1_copies" / "admit_summary.json"
            if a1.is_file():
                (r2 / "step1" / "admit_summary.json").write_bytes(a1.read_bytes())
                (r2 / "step1" / "relevance.json").write_text(json.dumps(
                    {"format": fmt, "inputs": inputs, "params": rule,
                     "calibration": {"tau": 0.0557, "check": {"ok": False, "tau_min": 0.5}}}), encoding="utf-8")
                a3 = BP.load_artifacts(r2, ["pilot_v1"])
                ev3 = E.load_dir(r2, "pilot_v1", exps=["pilot_v1"])
                check("filing: levers.relevance_state calls the failed build calibration_failed",
                      L.relevance_state(ev3) == "calibration_failed", L.relevance_state(ev3))
                v = _val([_item("L2", {"size": 287, "n_verified": 4}, trigger=["D4"],
                                cites=[ready["cites"][0]])], a3, live, d2)
                p = v["menu"][0] if v["menu"] else {}
                check("filing: a brain L2 at N 4 x M 287 on the real evidence (1,439 >= 1,435) takes "
                      "--increment-sources evidence and no --relevance; a person approves it (R3)",
                      p.get("params", {}).get("increment_sources") == "evidence"
                      and "relevance" not in p.get("params", {}) and p["params"].get("size") == 287
                      and p["params"].get("n_verified") == 4
                      and p.get("argv", [])[10:14] == ["--recipes", "full", "--increment-sources", "evidence"]
                      and p["policy_check"]["needs_approval"], (p.get("argv"), _why(v)))
                v = _val([_item("L2", {}, trigger=["D4"], cites=[ready["cites"][0]])], a3, live, d2)
                p = v["menu"][0] if v["menu"] else {}
                check("filing: a brain L2 that names no --size or --n-verified on the real evidence gets the R4 "
                      "sizing rule's N 4 x M 287 (the deterministic L2's), rendered as --size 287 --n-verified 4 "
                      "after --increment-sources evidence, and passes the executor and policy checks",
                      p.get("params", {}).get("increment_sources") == "evidence"
                      and (p["params"].get("size"), p["params"].get("n_verified")) == (287, 4)
                      and p.get("argv", [])[12:18] == ["--increment-sources", "evidence", "--size", "287",
                                                       "--n-verified", "4"]
                      and p["policy_check"]["needs_approval"], (p.get("argv"), _why(v)))
                v = _val([_item("L2", {"size": 400}, trigger=["D4"], cites=[ready["cites"][0]])], a3, live, d2)
                check("filing: at N 6 x M 400 the evidenced pool is too small: dropped with the numbers",
                      not v["menu"] and "cannot supply this build" in _why(v) and "1439" in _why(v), _why(v))
                for bp, want_why in (({"size": 50, "n_verified": 1}, "--n-verified 1 --size 50 is below the R4 "
                                                                       "rule's floor"),
                                     ({"size": 196, "n_verified": 4}, "196.35 images"),
                                     ({"size": 150, "n_verified": 4}, "M = 150 is 3.82% of base B's 3927 images"),
                                     ({"size": 287, "n_verified": 3}, "N = 3 decides 5 increments"),
                                     ({"size": 1, "n_verified": 1}, "a person's decision")):
                    v = _val([_item("L2", bp, trigger=["D4"], cites=[ready["cites"][0]])], a3, live, d2)
                    check("filing: a brain L2 at N %d x M %d on the real evidence is below the R4 rule's floor "
                          "(M >= 5%% of B = 196.35, at least 6 decided): dropped with the numbers, never filed"
                          % (bp["n_verified"], bp["size"]),
                          not v["menu"] and want_why in _why(v) and v["counts"]["render_refusals"] == 1, _why(v))
                v = _val([_item("L2", {"size": 287, "n_verified": 4, "relevance": str(r2 / "step1" /
                                                                                   "relevance.json")},
                                trigger=["D4"], cites=[ready["cites"][0]])], a3, live, d2)
                check("filing: a brain L2 may not name the refused relevance file",
                      not v["menu"] and "not the plan's to choose" in _why(v), _why(v))
                v = _val([_item("L2", {"size": 287, "n_verified": 4, "increment_sources": "evidence",
                                       "relevance": str(r2 / "step1" / "relevance.json")},
                                trigger=["D4"], cites=[ready["cites"][0]])], a3, live, d2)
                check("filing: a brain L2 with --increment-sources evidence and --relevance is dropped with "
                      "remote.py's message (realloop refuses the pair)",
                      not v["menu"] and "does not go with it" in _why(v), _why(v))
                v = _val([_item("L3", {})], a3, live, d2)
                check("filing: a brain L3 over the calibration-failed file is dropped (never re-proposed)",
                      not v["menu"] and "L3 is not proposed" in _why(v), _why(v))
            else:
                skip("filing: L2 on source-level species evidence",
                     "step1_copies/admit_summary.json not in tests/fixtures/inc_replay")
        else:
            skip("filing: L2 with a ready D4", "step1/select_summary.json or the realloop row missing")
    else:
        skip("preconditions", "live levers.json not present")

    # Triggers are what the brain names and supports, nothing else.
    v = _val([variant(GOOD_L1, trigger=None, evidence_cites=[
        {"artifact": "pilot_v1/exp.json", "pointer": "/decision_exam", "value": "dev"}])], arts)
    p = v["menu"][0] if v["menu"] else {}
    check("trigger: an item naming no diagnosis is filed as the brain's own, attributed to none",
          p.get("trigger") == [] and p.get("basis") == "brain, no diagnosis"
          and v["counts"]["no_diagnosis"] == 1, (p.get("trigger"), p.get("basis")))
    v = _val([variant(GOOD_L1, trigger=["D1", "D4"], evidence_cites=[GOOD_L1["evidence_cites"][0]])],
             arts)
    p = v["menu"][0] if v["menu"] else {}
    check("trigger: a named diagnosis whose cites the item does not repeat, or that did not fire, "
          "is not a trigger",
          p.get("trigger") == [] and sorted(u["id"] for u in p.get("trigger_unsupported", []))
          == ["D1", "D4"], (p.get("trigger"), p.get("trigger_unsupported")))

    # Leaks the text scan alone would miss.
    raw = dict(arts)
    raw["pilot_v1/report.json"] = json.loads(
        (exp_dir("pilot_v1") / "pilot_v1" / "report.json").read_text(encoding="utf-8"))
    val0 = raw["pilot_v1/report.json"]["final"][0]["exams"]
    v = _val([variant(GOOD_L1, evidence_cites=[
        {"artifact": "pilot_v1/report.json", "pointer": "/final/0/exams", "value": val0}])], raw)
    check("leaks: a cite whose value holds a test key is refused even against an unscrubbed "
          "snapshot", "test" in val0 and not v["menu"] and v["counts"]["leaks"] == 1, _why(v))

    v = _val([variant(GOOD_L1, predicted={"metric": "agreement", "direction": "up",
                                          "magnitude": None, "test": {"twelve": 0.8}})], arts)
    check("leaks: an exam-keyed object anywhere in the item (here in predicted) is refused",
          not v["menu"] and v["counts"]["leaks"] == 1 and "/predicted/test" in _why(v), _why(v))

    # A quote of a US passage is the project's note, and is marked so.
    us = C.Corpus().line("2403.08763", 12).split(": ", 1)[1][:60]
    v = _val([variant(GOOD_L1, lit_cites=[{"paper_id": "2403.08763", "line": 12, "quote": us}])],
             arts)
    p = v["menu"][0] if v["menu"] else {}
    check("lit: a quote of a US line is marked a project note",
          p.get("lit") and p["lit"][0].get("project_note") is True
          and v["counts"]["lit_project_notes"] == 1, p.get("lit"))

    # The same request twice is one proposal.
    v = _val([GOOD_L1, variant(GOOD_L1, params={"exp": "pilot_v4"}), GOOD_L1], arts)
    check("dedup: the same build (under any new name) three times is one proposal",
          len(v["menu"]) == 1 and v["counts"]["duplicates"] == 2, v["counts"])


def _restage(dg, **changes):
    """A digest edited after staging and re-hashed as consistently as an editor could."""
    ed = copy.deepcopy(dg)
    ed.update(changes)
    ed["prompt_sha256"] = hashlib.sha256(ed["prompt"].encode("utf-8")).hexdigest()
    ed["sha256"] = BP.digest_sha256(ed)
    return ed


def test_run_and_merge(dg, arts):
    print("cluster job body, timeout and merge")
    plans = _TMP / "plans" / "camp"
    plans.mkdir(parents=True)
    inp = plans / "1.input.json"
    inp.write_text(json.dumps(dg), encoding="utf-8")
    client = MockClient(text="<think>ok</think>" + json.dumps(PLAN))
    rep = BP.run(str(inp), str(plans / "1.json"), "http://127.0.0.1:9/v1", MODEL, 32768, 60,
                 client=client)
    check("run: one call with the staged prompt, the model and num_ctx",
          client.calls == [(len(dg["prompt"]), MODEL, 32768)], (client.calls, rep.get("reason")))
    c0 = MockClient(text=json.dumps(PLAN))
    BP.run(str(inp), str(plans / "0.json"), "", MODEL, 0, 60, client=c0)
    check("run: num_ctx 0 asks for the context the digest was sized for",
          c0.calls and c0.calls[0][2] == dg["num_ctx"], c0.calls)
    cbig = MockClient(text=json.dumps(PLAN))
    rbig = BP.run(str(inp), str(plans / "over_cap.json"), "", MODEL, BP.MAX_NUM_CTX + BP.NUM_CTX_STEP, 60,
                  client=cbig)
    check("run: a num_ctx over MAX_NUM_CTX is refused before the model is called",
          not rbig["ok"] and "MAX_NUM_CTX" in rbig["reason"] and cbig.calls == []
          and json.loads((plans / "over_cap.json").read_text())["ok"] is False, rbig.get("reason"))
    check("run: the reply file is written with the parsed plan",
          rep["ok"] and json.loads((plans / "1.json").read_text())["plan"]["ranked_menu"][0]["lever"]
          == "L1" and rep["digest_sha256"] == dg["sha256"])
    sha = dg["sha256"]
    t0 = rep["started_utc"]                     # the lab submitted the job then
    at = lambda s: time.strftime("%Y-%m-%dT%H:%M:%SZ",   # noqa: E731
                                 time.gmtime(calendar.timegm(time.strptime(t0, "%Y-%m-%dT%H:%M:%SZ"))
                                             + s))
    col = BP.collect(str(plans / "1.json"), t0, at(1800), digest_sha256=sha)
    check("collect: a written reply for the staged digest, back in time, is ready",
          col["status"] == "ready", col)
    det = [M.proposal("L1", ["python", "-m", "x"], {"exp": "pilot_v2"}, "inc_build_pilot", "R3",
                      ["D1"], D1["cites"])]
    val = V.validate(col["reply"], arts, MENU, C.Corpus(), diagnoses=[D1])
    merged = BP.merge(det, col, val)
    check("merge: the deterministic proposal comes first and unchanged, the brain's after",
          merged["proposals"][0] is det[0] and len(merged["proposals"]) == 2
          and merged["proposals"][1]["proposed_by"] == "tier2:qwen3.8_27b"
          and len(merged["cards"]) == 1 and merged["brain"]["added"] == 1,
          [x["proposed_by"] for x in merged["proposals"]])
    brain = val["menu"][0]
    same = M.proposal("L1", brain["argv"][:5] + ["pilot_v2"] + brain["argv"][6:],
                      dict(brain["params"], exp="pilot_v2"), "inc_build_pilot", "R3", ["D1"],
                      D1["cites"], est_gpu_hours=brain["est_gpu_hours"])
    same["child_exp"] = "pilot_v2"
    m_same = BP.merge([same], col, val)
    got = m_same["proposals"]
    check("merge: a brain item that is the deterministic proposal's build becomes a note on it, "
          "not a second approval item",
          len(got) == 1 and got[0]["id"] == same["id"] and got[0]["argv"] == same["argv"]
          and {k: got[0][k] for k in same} == same
          and [n["proposed_by"] for n in got[0].get("brain_notes", [])] == ["tier2:qwen3.8_27b"]
          and got[0]["brain_notes"][0]["rank"] == 1 and "brain_notes" not in same
          and m_same["brain"]["noted_on_existing"] == 1,
          [(x["lever"], x.get("child_exp"), x["proposed_by"]) for x in got])
    other = V.validate(dict(col["reply"], digest_sha256="0" * 64), arts, MENU, C.Corpus(),
                       diagnoses=[D1])
    m_other = BP.merge(det, col, other)
    check("merge: a validation record of another reply is not merged",
          m_other["proposals"] == det and m_other["brain"]["status"] == "failed")

    to = MockClient(error="TimeoutError: timed out")
    rep2 = BP.run(str(inp), str(plans / "2.json"), "http://127.0.0.1:9/v1", MODEL, 32768, 60,
                  client=to)
    col2 = BP.collect(str(plans / "2.json"), t0, at(600), digest_sha256=sha)
    m2 = BP.merge(det, col2, None)
    check("timeout: a client timeout writes ok False and the lab sees 'failed' at once",
          rep2["ok"] is False and "timed out" in rep2["reason"] and col2["status"] == "failed")
    check("timeout: the deterministic path continues untouched",
          m2["proposals"] == det and m2["cards"] == [] and m2["brain"]["status"] == "failed")
    boom = MockClient(exc=RuntimeError("connection reset"))
    rep3 = BP.run(str(inp), str(plans / "3.json"), "", MODEL, 32768, 60, client=boom)
    check("run: a client that raises still leaves a reply naming the error",
          rep3["ok"] is False and "connection reset" in rep3["reason"]
          and (plans / "3.json").is_file())
    pend = BP.collect(str(plans / "9.json"), t0, at(7140), digest_sha256=sha)
    gone = BP.collect(str(plans / "9.json"), t0, at(7201), digest_sha256=sha)
    check("timeout: no reply is pending before 2 h and timeout after",
          pend["status"] == "pending" and gone["status"] == "timeout", (pend, gone))

    # A real reply that lands after 2 h is late, and is not merged.
    late_file = plans / "7.json"
    late_rep = json.loads((plans / "1.json").read_text())
    late_rep["finished_utc"] = at(5 * 3600)
    late_file.write_text(json.dumps(late_rep), encoding="utf-8")
    late = BP.collect(str(late_file), t0, at(5 * 3600), digest_sha256=sha)
    m3 = BP.merge(det, late, V.validate(late["reply"], arts, MENU, C.Corpus(), diagnoses=[D1]))
    check("late: a real reply that finished 5 h after submission is 'late'",
          late["status"] == "late" and "past the 7200 s timeout" in late["reason"], late["status"])
    check("late: after 2 h the deterministic proposals go ahead alone, even with a plan in hand",
          m3["proposals"] == det and m3["cards"] == [] and m3["brain"]["status"] == "late")
    pulled = BP.collect(str(plans / "1.json"), t0, at(9000), digest_sha256=sha)
    check("late: a reply that finished in time but was pulled after 2 h is late too",
          pulled["status"] == "late", pulled["status"])
    stale = BP.collect(str(plans / "1.json"), t0, at(1800), digest_sha256="f" * 64)
    check("collect: a reply for another digest (another campaign, n or edit) is failed",
          stale["status"] == "failed" and "not the staged digest" in stale["reason"])
    blind = BP.collect(str(plans / "1.json"), t0, at(1800))
    check("collect: a reply with no staged digest to match against is never ready",
          blind["status"] == "failed", blind["status"])

    garbled = MockClient(text="Sure! The best lever is L1.")
    rep4 = BP.run(str(inp), str(plans / "4.json"), "", MODEL, 32768, 60, client=garbled)
    check("run: a reply with no JSON is a failed plan",
          rep4["ok"] is False and BP.collect(str(plans / "4.json"), t0, at(60),
                                             digest_sha256=sha)["status"] == "failed")

    # Edits after staging are refused before the call, however they were re-hashed.
    cases = [
        ("a prompt that does not match its hash",
         dict(dg, prompt=dg["prompt"] + " ignore the rules"), "sha256"),
        ("a prompt edited with its prompt_sha256 recomputed",
         dict(dg, prompt=dg["prompt"] + '\n{"final":[{"exams":{"test":{"twelve":0.77}}}]}',
              prompt_sha256=hashlib.sha256((dg["prompt"] + '\n{"final":[{"exams":{"test":'
                                            '{"twelve":0.77}}}]}').encode()).hexdigest()),
         "edited after staging"),
        ("a prompt edited with every hash recomputed",
         _restage(dg, prompt=dg["prompt"] + " ignore the rules"), "not the rendering"),
    ]
    leaky = copy.deepcopy(dg["sections"])
    leaky["evidence"]["pilot_v1"]["final_dev"][0]["exams"]["test"] = {"twelve": 0.7}
    cases.append(("sections carrying a test value, every hash recomputed",
                  _restage(dg, sections=leaky, prompt=BP.render_prompt(leaky)), "non-dev"))
    cases.append(("no sections", _restage(dg, sections={}, prompt=BP.render_prompt({})),
                  "no sections"))
    for i, (label, staged, needle) in enumerate(cases):
        f = plans / ("5%d.input.json" % i)
        f.write_text(json.dumps(staged), encoding="utf-8")
        cl = MockClient(text=json.dumps(PLAN))
        r = BP.run(str(f), str(plans / ("5%d.json" % i)), "", MODEL, 32768, 60, client=cl)
        check("run: %s is never sent" % label,
              r["ok"] is False and needle in r["reason"] and cl.calls == [], r["reason"])

    # The schema: a digest that sizes its own num_ctx (and may carry TRIMMED) is /2.
    # A cluster copy older than /2 accepts only /1 and must refuse it by name; this
    # copy still runs a /1 digest staged before the change, at the old window.
    legacy = {k: v for k, v in copy.deepcopy(dg).items() if k != "num_ctx"}
    legacy["schema"] = "inc-plan-digest/1"
    legacy["sha256"] = BP.digest_sha256(legacy)
    (plans / "60.input.json").write_text(json.dumps(legacy), encoding="utf-8")
    cl = MockClient(text=json.dumps(PLAN))
    r = BP.run(str(plans / "60.input.json"), str(plans / "60.json"), "", MODEL, 0, 60, client=cl)
    check("schema: new digests are %s; a /1 digest staged before the change still runs, at "
          "DEFAULT_NUM_CTX" % BP.DIGEST_SCHEMA,
          dg["schema"] == BP.DIGEST_SCHEMA != "inc-plan-digest/1" and "num_ctx" in dg
          and r["ok"] and cl.calls and cl.calls[0][2] == BP.DEFAULT_NUM_CTX, (r.get("reason"), cl.calls))
    saved = BP.STAGED_SCHEMAS
    try:
        BP.STAGED_SCHEMAS = ("inc-plan-digest/1",)
        cl = MockClient(text=json.dumps(PLAN))
        r = BP.run(str(inp), str(plans / "61.json"), "", MODEL, 0, 60, client=cl)
    finally:
        BP.STAGED_SCHEMAS = saved
    check("schema: a copy that runs only /1 (the cluster before this change) refuses a /2 "
          "digest by its schema, before the call",
          r["ok"] is False and "not an INC plan digest (schema '%s'" % BP.DIGEST_SCHEMA in r["reason"]
          and cl.calls == [], r["reason"])
    paths = BP.plan_paths("camp", 3)
    argv = BP.submit_argv(paths)
    check("plan paths: input and output under CLUSTER_CAMPAIGN_DIR/plans/<campaign>/",
          paths["input"] == M.CLUSTER_CAMPAIGN_DIR + "/plans/camp/3.input.json"
          and paths["output"] == M.CLUSTER_CAMPAIGN_DIR + "/plans/camp/3.json"
          and argv[-1] == "weed_llm_benchmark/run_inc_plan.sh"
          and ("PLAN_INPUT=%s" % paths["input"]) in argv[1], argv)


# --- outcome ------------------------------------------------------------------------

def _b0(sd=0.004, mean=0.80, test_mean=0.9):
    return {"exp": "b0_v1", "type": "baseline", "final": [{
        "model": "base train_core", "runs": ["final__base__s0", "final__base__s1",
                                             "final__base__s2"],
        "exams": {"dev": {"twelve": {"mean": mean, "sd": sd, "n": 3},
                          "agnostic": {"mean": 0.85, "sd": 0.003, "n": 3}},
                  "test": {"twelve": {"mean": test_mean, "sd": 0.05, "n": 3}}}}]}


def _chain_report(flags):
    """A synthetic chain report: flags = {chain: [agree per step]}; truth says helps throughout."""
    n = len(next(iter(flags.values())))
    steps = [{"step": "S%d" % (k + 1), "tag": "s%02d_S%d" % (k + 1, k + 1), "k": k + 1,
              "truth": {"verdict": "helps"},
              "chains": {r: {"verdict": "ACCEPT" if f[k] else "REJECT", "agree": f[k]}
                         for r, f in flags.items()}} for k in range(n)]
    return {"type": "chain", "steps": steps,
            "agreement": {r: {"agree": sum(f), "compared": n, "rate": sum(f) / float(n)}
                          for r, f in flags.items()},
            "final": [{"model": "chain full: final incumbent",
                       "exams": {"dev": {"twelve": {"mean": 0.7, "sd": None, "n": 1}},
                                 "test": {"twelve": {"mean": 0.9, "sd": None, "n": 1}}}}]}


def test_outcome():
    print("outcome and track record")
    fl = O.noise_floor(_b0())
    check("floor: 2 x the dev sd of B0's three seeds, with its cite",
          abs(fl["floor"] - 0.008) < 1e-12 and fl["n"] == 3
          and fl["cite"]["pointer"] == "/final/0/exams/dev/twelve/sd", fl)
    check("floor: test values of B0 do not move it", O.noise_floor(_b0(test_mean=0.1)) == fl)
    b0_path = report_path("b0_v1")
    if b0_path is not None:
        lf = O.noise_floor(O.load_report(b0_path))
        check("floor: the real b0_v1 report gives a floor from its dev seeds",
              lf["floor"] is not None and lf["n"] >= 2, lf)
    else:
        skip("floor: real b0_v1 report", "skip-until-fixture: b0_v1/report.json is on the cluster")
    check("floor: no b0 report -> no floor", O.noise_floor(None)["floor"] is None)

    child_b = {"exp": "base_b_v1", "type": "baseline", "final": [{
        "model": "base base_B", "exams": {"dev": {"twelve": {"mean": 0.83, "sd": 0.005, "n": 3}}}}]}
    prop_b = {"id": "pb", "lever": "L8", "proposed_by": M.AUTOPILOT_ACTOR,
              "predicted": {"metric": "dev_twelve", "direction": "up"}}
    ob = O.score(prop_b, child_b, _b0(), _b0(), "base_b_v1", "b0_v1")
    check("verdict: B 0.83 vs B0 0.80 on dev, floor 0.008, n 3 -> better, prediction correct",
          ob["verdict"] == "better" and ob["correct"] is True and ob["contradicted"] is False
          and abs(ob["delta"] - 0.03) < 1e-12 and ob["result"]["n"] == 3, ob["verdict"])
    near = copy.deepcopy(child_b)
    near["final"][0]["exams"]["dev"]["twelve"]["mean"] = 0.804
    on = O.score(prop_b, near, _b0(), _b0(), "base_b_v1", "b0_v1")
    check("verdict: a 0.004 gain is within B0's noise -> unconfirmed, not contradicted",
          on["verdict"] == "within_noise" and on["correct"] is False and not on["contradicted"])
    nob0 = O.score(prop_b, child_b, _b0(), None, "base_b_v1", "b0_v1")
    check("verdict: without a b0 report the dev verdict is insufficient",
          nob0["verdict"] == "insufficient" and nob0["correct"] is None)
    leak = copy.deepcopy(child_b)
    leak["final"][0]["exams"]["test"] = {"twelve": {"mean": 0.1, "sd": 0.0, "n": 3}}
    check("verdict: the child's test values do not change the outcome",
          O.score(prop_b, leak, _b0(), _b0(), "base_b_v1", "b0_v1")["result"] == ob["result"])

    # The agreement metric: per chain, a sign test over steps paired by tag.
    up = {"id": "pu", "lever": "L1", "proposed_by": "tier2:qwen3.8_27b",
          "predicted": {"metric": "agreement", "direction": "up"}}
    par = _chain_report({"full": [True] + [False] * 9, "freeze": [True] + [False] * 9})
    kid = _chain_report({"full": [True] * 9 + [False], "freeze": [True] * 9 + [False]})
    ou = O.score(up, kid, par, None, "c", "p")
    pc = ou["measure"]["per_chain"]
    check("agreement: 1/10 -> 9/10 on every chain (8 steps gained, none lost, p 0.0078) is "
          "better, and 'up' is confirmed",
          ou["verdict"] == "better" and ou["correct"] is True and ou["n"] is None
          and pc["full"]["gained"] == 8 and pc["full"]["lost"] == 0
          and abs(pc["full"]["p"] - 2.0 / 256) < 1e-12 and ou["result"]["n"] == 0
          and ou["result"]["noise_floor"] is None
          and ou["result"]["extra"]["verdict_rule"].startswith("agreement: exact"), pc["full"])
    down = O.score(dict(up, predicted={"metric": "agreement", "direction": "down"}), kid, par,
                   None, "c", "p")
    check("agreement: a 'down' prediction on that child is contradicted (D13 evidence)",
          down["contradicted"] is True and down["correct"] is False)
    one = _chain_report({"full": [True, True] + [False] * 8, "freeze": [True, True] + [False] * 8})
    small = O.score(up, one, par, None, "c", "p")
    check("agreement: one step gained out of 10 is within the sign-test band, not 'better'",
          small["verdict"] == "within_noise" and small["correct"] is False
          and not small["contradicted"], small["measure"]["reason"])
    mixed = _chain_report({"full": [True] * 9 + [False], "freeze": [True, True] + [False] * 8})
    mx = O.score(up, mixed, par, None, "c", "p")
    check("agreement: chains that disagree give insufficient, each chain's verdict recorded",
          mx["verdict"] == "insufficient"
          and {r: v["verdict"] for r, v in mx["measure"]["per_chain"].items()}
          == {"full": "better", "freeze": "within_noise"}, mx["measure"]["reason"])
    named = O.score(dict(up, predicted={"metric": "agreement", "direction": "up",
                                        "chain": "full"}), mixed, par, None, "c", "p")
    check("agreement: a prediction naming a chain is scored on that chain",
          named["verdict"] == "better" and named["measure"]["chain"] == "full")
    try:
        from weed_optimizer_framework.tools.inc_autopilot import diagnose as D
        ev = E.from_texts({}, "pilot_v2", context={"outcomes": [down, ou]})
        d13 = [d for d in D.detect(ev) if d["id"] == "D13"][0]
        check("agreement: diagnose.D13 fires on the contradicted outcome record, not the "
              "confirmed one",
              d13["fired"] and "/outcomes/0/" in d13["cites"][0]["pointer"]
              and len(d13["cites"]) == 2, d13["summary"])
    except ImportError as exc:
        skip("agreement: D13 on outcome records", "diagnose.py not importable: %s" % exc)

    ev_path, sum_path = _TMP / "track.jsonl", _TMP / "track.json"
    ex_path = _TMP / "experiments.jsonl"
    b1, b2 = report_path("pilot_v1"), report_path("pilot_v2")
    if b1 is None or b2 is None:
        skip("verdict: pilot_v1 -> pilot_v2", "skip-until-fixture: pilot_v2 report not local")
        o = None
    else:
        parent, child = O.load_report(b1), O.load_report(b2)
        prop = {"id": "p1", "lever": "L1", "proposed_by": "tier2:qwen3.8_27b",
                "control": "pilot_v1", "params": {"exp": "pilot_v2"},
                "predicted": {"metric": "agreement", "direction": "up", "magnitude": 0.14}}
        o = O.score(prop, child, parent, None, "pilot_v2", "pilot_v1")
        pc = o["measure"]["per_chain"]
        check("verdict: L1 (pilot_v1 -> pilot_v2) is insufficient on agreement: 7 steps from a "
              "3/7 parent cannot reach the sign-test band (best p 0.125)",
              o["verdict"] == "insufficient" and o["correct"] is None and o["n"] is None
              and all(v["verdict"] == "insufficient" and abs(v["best_attainable_p"] - 0.125)
                      < 1e-12 for v in pc.values()),
              {r: (v["verdict"], v["best_attainable_p"]) for r, v in pc.items()})
        check("verdict: the per-chain deltas are recorded: full +1/7, freeze and lora -1/7",
              abs(pc["full"]["delta"] - 1.0 / 7) < 1e-12
              and abs(pc["freeze"]["delta"] + 1.0 / 7) < 1e-12
              and abs(pc["lora"]["delta"] + 1.0 / 7) < 1e-12
              and (pc["freeze"]["gained"], pc["freeze"]["lost"]) == (0, 1),
              {r: (v["delta"], v["gained"], v["lost"]) for r, v in pc.items()})
        ha = o["measure"]["helps_accepted"]
        check("verdict: the child's full chain ACCEPTs one truth-helps step, the parent none",
              ha["child"].get("full") == 1 and sum(ha["parent"].values()) == 0, ha)
        check("verdict: yet L1's success (>= 5/7) is not met: 4/7",
              o["measure"]["child"]["rate"] < 5.0 / 7)
        dv = O.score(dict(prop, predicted={"metric": "dev_twelve", "direction": "up"}), child,
                     parent, _b0(), "pilot_v2", "pilot_v1")
        check("verdict: a chain's final incumbent is one run -> dev_twelve is insufficient",
              dv["verdict"] == "insufficient" and dv["n"] == 1 and dv["correct"] is None,
              (dv["verdict"], dv["n"]))
        O.record(prop, child, parent, None, "pilot_v2", "pilot_v1", events_path=ev_path,
                 summary_path=sum_path, experiments_path=ex_path)
    rec = O.record(up, kid, par, None, "synthetic_child", "synthetic_parent", events_path=ev_path,
                   summary_path=sum_path, experiments_path=ex_path)
    O.record(prop_b, child_b, _b0(), _b0(), "base_b_v1", "b0_v1", events_path=ev_path,
             summary_path=sum_path, experiments_path=ex_path)
    O.record_validation({"proposed_by": "tier2:qwen3.8_27b", "model": MODEL,
                         "counts": {"items": 10, "valid": 1, "cards": 1, "dropped": 8,
                                    "cites_checked": 8, "cites_failed": 2, "lit_checked": 2,
                                    "lit_failed": 1, "leaks": 1, "unpriced": 1,
                                    "precondition_refusals": 2}},
                        events_path=ev_path, summary_path=sum_path)
    rows = [json.loads(ln) for ln in ex_path.read_text().splitlines()]
    real = 1 if o is not None else 0
    check("record: one experiments.record_result row per outcome, verdict and rule on it",
          len(rows) == 2 + real and rows[real]["verdict"] == "better"
          and rows[real]["lever_id"] == "L1" and rows[real]["noise_floor"] is None
          and rows[real]["extra"]["child_exp"] == "synthetic_child"
          and rows[-1]["noise_floor"] is not None, [r["verdict"] for r in rows])
    summ = json.loads(sum_path.read_text())
    tb = summ["proposers"]["tier2:qwen3.8_27b"]
    check("record: the brain's track record has prediction accuracy, insufficient outcomes kept "
          "apart, and citation failures",
          tb["scored"] == 1 + real and tb["correct"] == 1 and tb["insufficient"] == real
          and tb["prediction_accuracy"] == 1.0 and tb["plans"] == 1 and tb["unpriced"] == 1
          and tb["precondition_refusals"] == 2
          and abs(tb["citation_failure_rate"] - 0.3) < 1e-12, tb)
    check("record: levers are tracked separately",
          summ["levers"]["L1"]["correct"] == 1 and summ["levers"]["L8"]["correct"] == 1
          and summ["levers"]["L1"]["insufficient"] == real, summ["levers"])
    check("record: the outcome row is on the event log", rec["kind"] == "outcome"
          and len(O.read_events(ev_path)) == 3 + real)


# --- the sbatch job -----------------------------------------------------------------

def test_script():
    print("run_inc_plan.sh")
    path = ROOT / "run_inc_plan.sh"
    text = path.read_text(encoding="utf-8")
    rc = subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode
    check("script: bash syntax", rc == 0)
    check("script: executable", os.access(str(path), os.X_OK))
    for label, pat in (("2 h walltime", r"#SBATCH --time=02:00:00"),
                       ("per-job port", r"PORT=\$\(\( 8000 \+ \$\{SLURM_JOB_ID:-0\} % 1000 \)\)"),
                       ("planner role", r'model_router\.resolve\("planner"\)'),
                       ("warm-up before the request", r"api/generate"),
                       ("brain_plan run", r"inc_autopilot\.brain_plan run"),
                       ("inputs confined to the plans dir", r'"\$PLANS"/\*\)'),
                       ("the plans dir is resolved like the paths compared with it",
                        r'PLANS="\$\(realpath -m "\$REPO/results/framework/inc/_campaign/plans"\)"'),
                       ("log under inc/**/logs", r"inc/_campaign/plans/logs/%x_%j\.out")):
        check("script: %s" % label, re.search(pat, text) is not None)
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    gres = re.search(r"^#SBATCH --gres=gpu:([a-z0-9]+)-(\d+):1$", text, re.M)
    check("script: requests one GPU of the memory MAX_NUM_CTX is sized for, on GPU-shared",
          gres is not None and float(gres.group(2)) == BP.JOB_GPU_MEM_GB
          and re.search(r"^#SBATCH --partition=GPU-shared$", text, re.M) is not None,
          gres.group(0) if gres else None)
    check("script: one KV cache per job (OLLAMA_NUM_PARALLEL=1), residency logged after warm-up",
          "export OLLAMA_NUM_PARALLEL=1" in code and "/api/ps" in code
          and code.index("OLLAMA_NUM_PARALLEL=1") < code.index("ollama serve"))
    check("script: flash attention on before the server starts (no f32 KQ buffer of num_ctx)",
          "export OLLAMA_FLASH_ATTENTION=1" in code
          and code.index("OLLAMA_FLASH_ATTENTION=1") < code.index("ollama serve"))
    # The layout line, run with /api/show, /api/tags and nvidia-smi stubbed.
    lay = re.search(r'^curl -sf -m 120 -X POST "http://127\.0\.0\.1:\$PORT/api/show".*?'
                    r'\|\| echo "\[layout\] /api/show did not answer"$', text, re.M | re.S)
    if lay is None:
        check("script: the model's own KV layout is logged before warm-up", False)
    else:
        tags = _TMP / "layout_tags.json"
        tags.write_text(json.dumps({"models": [{"name": "m:27b", "size": 19 * 10 ** 9}]}),
                        encoding="utf-8")
        show = json.dumps({"model_info": {
            "general.architecture": "qwen3", "qwen3.block_count": 64,
            "qwen3.attention.head_count": 64, "qwen3.attention.head_count_kv": 16,
            "qwen3.attention.key_length": 128, "qwen3.attention.value_length": 128}})
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        out = {}
        for n_ctx in (BP.MAX_NUM_CTX, 131072):
            snippet = "\n".join([
                "curl() { cat <<'JSON'\n%s\nJSON\n}" % show,
                "nvidia-smi() { echo 81559; }",
                "PORT=1; PLAN_MODEL=m:27b; NUM_CTX=%d; SLURM_JOB_ID=7; PLANS=%s" % (n_ctx, _TMP),
                "cp %s \"$PLANS/logs/inc_plan_tags_7.json\" 2>/dev/null || { mkdir -p \"$PLANS/logs\"; "
                "cp %s \"$PLANS/logs/inc_plan_tags_7.json\"; }" % (tags, tags),
                lay.group(0)])
            res = subprocess.run(["bash", "-c", snippet], cwd=str(ROOT), env=env,
                                 capture_output=True, text=True)
            out[n_ctx] = res.stdout.strip()
        check("script: the model's own KV layout is logged before warm-up, against the assumed "
              "one and this GPU (a 16-KV-head model: 2x, fits at the cap, over at 131072)",
              code.index("/api/show") < code.index("api/generate")
              and "2.00x" in out[BP.MAX_NUM_CTX] and "OVER THE GPU" not in out[BP.MAX_NUM_CTX]
              and "OVER THE GPU" in out[131072] and "19.0 GB of weights" in out[131072]
              and "of 86 GB" in out[131072], out)
    guard = re.search(r"^MAX_CTX=.*?^fi$", text, re.M | re.S)
    if guard is None:
        check("script: a num_ctx cap guard", False)
    else:
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        got = {}
        for n in (BP.MAX_NUM_CTX, BP.MAX_NUM_CTX + 1, "12ab"):
            snippet = 'fail() { echo "FAIL: $1"; }\nNUM_CTX=%s\n%s\necho "PASSED max=$MAX_CTX"' % (
                n, guard.group(0))
            res = subprocess.run(["bash", "-c", snippet], cwd=str(ROOT), env=env,
                                 capture_output=True, text=True)
            got[str(n)] = (res.returncode, res.stdout.strip()[-160:])
        check("script: the cap guard reads MAX_NUM_CTX from brain_plan, passes it and refuses "
              "more (and a non-number) before the server starts",
              got[str(BP.MAX_NUM_CTX)] == (0, "PASSED max=%d" % BP.MAX_NUM_CTX)
              and got[str(BP.MAX_NUM_CTX + 1)][0] == 2 and "MAX_NUM_CTX" in got[str(BP.MAX_NUM_CTX + 1)][1]
              and got["12ab"][0] == 2
              and code.index("MAX_CTX=") < code.index("ollama serve"), got)
    check("script: no rsync over the outer package copy", "rsync" not in code)
    check("script: imports the nested (git-tracked) copy",
          'CODE="$REPO/weed_llm_benchmark"' in text and 'PYTHONPATH="$CODE' in text)

    # The script's own PLANS line and input check, run with REPO reached through a
    # symlink (GNU realpath -m emulated, as macOS realpath has no -m).
    plans_line = re.search(r"^PLANS=.*$", text, re.M).group(0)
    case = re.search(r'^case "\$\(realpath -m "\$INPUT"\)" in\n.*?^esac$', text, re.M | re.S).group(0)
    base = pathlib.Path(tempfile.mkdtemp(dir=str(_TMP), prefix="sym_"))
    (base / "real" / "results/framework/inc/_campaign/plans/c").mkdir(parents=True)
    os.symlink(str(base / "real"), str(base / "link"))
    snippet = "\n".join([
        "realpath() { python3 -c 'import os,sys; print(os.path.realpath(sys.argv[-1]))' \"$@\"; }",
        "REPO=%s" % (base / "link"),
        plans_line,
        'INPUT="$REPO/results/framework/inc/_campaign/plans/c/1.input.json"',
        case,
        "echo INSIDE"])
    res = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True)
    check("script: an input under a symlinked /ocean path passes the plans-dir check",
          "INSIDE" in res.stdout and res.returncode == 0, res.stderr[-300:])
    snippet_out = snippet.replace('INPUT="$REPO/results/framework/inc/_campaign/plans/c/1.input.json"',
                                  'INPUT="%s/elsewhere.json"' % base)
    res = subprocess.run(["bash", "-c", snippet_out], capture_output=True, text=True)
    check("script: an input outside the plans dir is still refused",
          "INSIDE" not in res.stdout and res.returncode == 2, (res.returncode, res.stderr[-200:]))


def test_validate_l9():
    """A brain L9 (D15 on pilot_v2) with the live menu: the same request as the
    deterministic one; the gate and the replay mode are the autopilot's."""
    print("validation: L9 and the gate (D15 on pilot_v2)")
    from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG
    root = inc_root(["pilot_v1", "pilot_v2"])
    if root is None:
        skip("validation: L9", "pilot_v2 evidence not local")
        return
    exps = ["pilot_v1", "pilot_v2"]
    ev = E.load_dir(root, "pilot_v2", exps=exps)
    arts = BP.load_artifacts(root, exps)
    diags = DG.detect(ev)
    by = DG.by_id(diags)
    fired = [d for d in diags if d["fired"]]
    live = BP.load_menu()
    det = [p for p in L.propose(diags, ev)["proposals"] if p["lever"] == "L9"]
    cite = by["D15"]["cites"][0]

    def val(params, diagnoses=fired):
        return V.validate({"schema": BP.REPLY_SCHEMA, "model": MODEL, "digest_sha256": "d",
                           "plan": {"ranked_menu": [_item("L9", params, cites=[cite], trigger=["D15"])],
                                    "off_menu": []}}, arts, live, C.Corpus(), diagnoses=diagnoses, exp="pilot_v2")
    # This case files through the live inc_build_pilot row (its gate_flips_mode bound);
    # the rest of this file pins its own test row.
    table = dict(_POLICY, actions=dict(_POLICY["actions"], inc_build_pilot=_LIVE_ACTIONS["inc_build_pilot"]))
    live_path = _TMP / "policy_actions_live_pilot.json"
    live_path.write_text(json.dumps(table), encoding="utf-8")
    old_env = os.environ["BRAIN_POLICY_ACTIONS"]
    os.environ["BRAIN_POLICY_ACTIONS"] = str(live_path)
    try:
        _test_validate_l9(val, det, live, arts, cite, fired)
    finally:
        os.environ["BRAIN_POLICY_ACTIONS"] = old_env


def _test_validate_l9(val, det, live, arts, cite, fired):
    v = val({})
    p = v["menu"][0] if v["menu"] else {}
    check("L9: a brain L9 on pilot_v2 materialises to the deterministic command and price",
          det and p.get("argv") == det[0]["argv"] and p.get("est_gpu_hours") == det[0]["est_gpu_hours"]
          and p.get("params", {}).get("gate_flips_mode") == "net" and p.get("parent_exp") == "pilot_v2",
          (p.get("argv"), _why(v)))
    v = val({"gate_flips_mode": "negative"})
    check("L9: a plan cannot set the gate L9 fixes", not v["menu"] and "gate_flips_mode" in _why(v), _why(v))
    v = val({"replay_mode": "sample"})
    check("L9: the replay mode is the parent's (full), not the plan's", not v["menu"] and "replay_mode" in _why(v),
          _why(v))
    v = val({}, diagnoses=[d for d in fired if d["id"] != "D15"])
    check("L9: refused unless D15 fired (only_after)", not v["menu"] and "D15" in _why(v), _why(v))
    rows = {r["id"]: r for r in BP.menu_section(live)}
    check("L9: the digest shows the gate and the replay mode as the autopilot's, D15 as its precondition",
          "gate_flips_mode" in rows["L9"]["set_by_autopilot"] and "replay_mode" in rows["L9"]["set_by_autopilot"]
          and rows["L9"].get("preconditions", {}).get("only_after") == ["D15"]
          and "gate_flips_mode" in rows["L1"]["set_by_autopilot"], rows["L9"].get("preconditions"))
    v1 = [d for d in fired if d["id"] in ("D1", "D4")]
    b1 = V.validate({"schema": BP.REPLY_SCHEMA, "model": MODEL, "digest_sha256": "d",
                     "plan": {"ranked_menu": [_item("L1", {"gate_flips_mode": "net"}, cites=[cite])],
                              "off_menu": []}}, arts, live, C.Corpus(), diagnoses=v1 or fired, exp="pilot_v2")
    check("L1: a plan cannot change a v1 parent's gate (that is L9's)",
          not b1["menu"] and "gate_flips_mode" in _why(b1), _why(b1))


def main():
    test_corpus()
    test_corpus_build()
    arts, dg, root = test_digest()
    test_num_ctx()
    test_digest_trim()
    test_parse()
    if arts is not None:
        test_validate(arts, root)
        test_validate_governance(arts, root)
        test_run_and_merge(dg, arts)
        test_validate_l9()
    else:
        skip("validation and run cases", "pilot_v1 evidence not local")
    test_outcome()
    test_script()
    print("\n%d failure(s), %d skipped" % (len(FAILURES), len(SKIPS)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
