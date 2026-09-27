"""The INC research brain: a dev-only digest in, a ranked plan out (docs/INC_AUTOPILOT.md (c)).

Why this exists
---------------
The deterministic layer (diagnose.py + levers) reproduces what it was written
for. The brain is the advisory second opinion for what it was not: D13
contradictions, residuals no rule explains, novel per-species patterns. It
runs as a cluster sbatch job (run_inc_plan.sh; brains run on cluster models),
is never on the critical path, and everything it says is checked by
validate.py before anything is filed.

The digest is dev-only by construction
--------------------------------------
Artifacts come from evidence.py (its allow-list and its scrub: every dict key
named test / ood22 / ood23 / imageweeds is removed at any depth, report.py:56
putting test into every report's final table; `dev_only()` is that same
scrub for anything else). `build_digest()` then asserts (`assert_dev_only`) that no such key, and no
pointer or score path naming such an exam, survived in any evidence section
before it renders a byte. The literature section is exempt from the string
check (a paper about test-set leakage may say "test set"); it carries no
values of ours.

Citable evidence
----------------
The brain must cite values exactly (validate.py compares them to the
snapshot). So the digest never rounds and never paraphrases: every evidence
object carries `_at` = {artifact, pointer[, line]} and is a SUBSET of the
artifact object at that address with the same keys and nesting. The address
of a value inside it is `_at.pointer + "/" + key path`. Diagnoses keep their
own full cites.

Lifecycle
---------
lab:     build_digest() -> write <n>.input.json -> stage to
         CLUSTER_CAMPAIGN_DIR/plans/<campaign>/ -> sbatch run_inc_plan.sh
cluster: run_inc_plan.sh -> `brain_plan run` -> <n>.json beside the input
lab:     pull <n>.json -> collect() -> validate.validate() -> merge()
         with the deterministic proposals. collect() matches the reply to
         the staged digest's sha256 and to the submission time: a plan that
         landed more than 2 h after submission is `late`, one for another
         digest (another campaign, n or edit) is `failed`, and only a
         `ready` plan is merged. A late, failed or unparseable plan leaves
         the deterministic path untouched. merge() files a brain item that
         renders the same command as a deterministic proposal as a note on
         that proposal, not as a second approval item.

The staged digest is checked again on the cluster before anything is sent
(run(), check_staged()): its sha256 is recomputed, the prompt must be the
rendering of its sections, and every section and the prompt text are checked
for non-dev exam data. An edit after staging is refused however consistently
it was re-hashed.

Snapshot shape this module reads
--------------------------------
An evidence.Evidence, or the map `artifacts_of()` makes of it:
{"<exp>/report.json": dict, "<exp>/exp.json": dict, ...,
"<exp>/ledger.jsonl": [entry, ...] (index i = line i+1), "step1/...": dict,
"campaign/context.json": dict}; names relative to INC_DIR, as in model.py's
Cite and evidence.py's addresses. `load_artifacts()` builds it from a local
INC tree through evidence.load_dir.

CLI:
  python -m weed_optimizer_framework.tools.inc_autopilot.brain_plan digest --inc-dir DIR --exp EXP --out F [--parent P ... --campaign C --n N --num-ctx N]
  python -m weed_optimizer_framework.tools.inc_autopilot.brain_plan run --input F --output G --endpoint URL --model M [--num-ctx N --timeout S]
  python -m weed_optimizer_framework.tools.inc_autopilot.brain_plan parse REPLY.txt
"""
from __future__ import annotations

import argparse
import calendar
import copy
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from . import model as M

# /2: the digest carries its own num_ctx (the job asks the server for it) and,
# when trimmed, a TRIMMED section in its prompt. A cluster copy that predates /2
# would render the prompt without TRIMMED and ask for a fixed window, so it must
# refuse a /2 digest, and it does: its check_staged accepts only /1 ("not an INC
# plan digest (schema 'inc-plan-digest/2')"). This copy still runs a /1 digest
# staged before the change (no TRIMMED section, DEFAULT_NUM_CTX).
DIGEST_SCHEMA = "inc-plan-digest/2"
STAGED_SCHEMAS = (DIGEST_SCHEMA, "inc-plan-digest/1")
REPLY_SCHEMA = "inc-plan-reply/1"

# The exams no decision may read (docs/INC_AUTOPILOT.md (d)). dev is the only
# decision exam (model.DECISION_EXAM).
FORBIDDEN_EXAMS = ("test", "ood22", "ood23", "imageweeds")
_FORBIDDEN_PATH_RE = re.compile(
    r"(?:scores/(?:test|ood22|ood23|imageweeds)\.json"
    r"|(?:^|/)exams/(?:test|ood22|ood23|imageweeds)(?:/|$))")

# A plan not back within this many seconds of submission is abandoned and the
# deterministic proposal goes ahead alone (contract (c): timeout 2 h, the
# sbatch walltime of run_inc_plan.sh).
PLAN_TIMEOUT_S = 7200

# --- the context a digest is sized for ---------------------------------------------
#
# build_digest chooses num_ctx from the digest's own size (choose_num_ctx) and
# records it in the digest; the job asks the server for exactly that, so the two
# cannot disagree. A caller may fix num_ctx instead (tests, the CLI's --num-ctx);
# the digest is then trimmed to fit it.
#
# The cap is what the job can hold. run_inc_plan.sh requests one H100 80 GB on
# GPU-shared (--gres=gpu:h100-80:1), serves one request at a time
# (OLLAMA_NUM_PARALLEL=1, so ollama allocates one KV cache, not one per slot)
# and turns flash attention on (OLLAMA_FLASH_ATTENTION=1). Without flash
# attention llama.cpp also allocates a KQ buffer of num_ctx x batch x query
# heads in f32, about 13 GB at 98304 tokens with a 512 batch and 64 heads,
# which RUNTIME_OVERHEAD_GB does not cover.
# The planner role's model (model_router "planner", qwen3.8:27b) declares
# context_length 262144 in its ollama manifest and loads about 19 GB of Q4
# weights. Its attention layout is not recorded in this repo. The sizing
# ASSUMES (unverified) the layout of a dense 32B such as Qwen3-32B: every layer
# full GQA attention, 64 layers x 8 KV heads x 128 dims x K and V x 2 bytes
# (f16) = 256 KiB per token. That is an assumption, not a bound: a dense 27B
# with 16 KV heads, or with head_dim 256, needs twice that; a hybrid layout
# (most layers without a KV cache) needs a fraction. So the cap must hold at
# KV_SIZING_FACTOR (2) times the assumed KV, not only at the assumption:
#   98304 tokens: 25.8 GB of KV assumed, 19 + 25.8 + 6 (runtime) = 50.8 GB of 80;
#   at twice the KV, 51.5 GB of KV and 76.5 GB in all. Fits either way.
#   106496 tokens (the next step): 80.8 GB at twice the KV. Does not fit.
#   131072 tokens: 59.4 GB assumed, but 93.7 GB at twice the KV. Not safe while
#   the layout is unverified.
#   262144 tokens (the manifest's full context): 93.7 GB even under the assumed
#   layout. It does not fit: ollama would move layers to the CPU, and the 2 h
#   walltime would be at risk.
# So MAX_NUM_CTX is 98304, 3/8 of the model's context: the largest multiple of
# NUM_CTX_STEP that fits at twice the assumed KV. It costs the brain nearly
# nothing: the ssh line (STAGED_MAX_CHARS below) already stops a digest at about
# 62-69K estimated tokens, and the cap's token budget is 65,536. Raise it only
# once the job's layout line has shown the real KV per token.
# The job settles the assumption on its first run. Before warm-up it logs the
# KV bytes per token of the model's own layout (/api/show model_info,
# layout_line) against KV_BYTES_PER_TOKEN_ASSUMED. After warm-up it logs how
# much of the model is resident on the GPU (/api/ps size_vram against size).
JOB_GPU_MEM_GB = 80.0
PLANNER_CONTEXT_LENGTH = 262144
PLANNER_WEIGHTS_GB = 19.0
KV_BYTES_PER_TOKEN_ASSUMED = 64 * 8 * 128 * 2 * 2
KV_SIZING_FACTOR = 2
RUNTIME_OVERHEAD_GB = 6.0
MAX_NUM_CTX = 98304
# The smallest window a digest is given: the fixed size every digest had before
# num_ctx was chosen from its size, so a digest that fitted then never gets less
# room now. Also the job's fallback for a digest that names no num_ctx.
DEFAULT_NUM_CTX = 49152
NUM_CTX_STEP = 8192
# The token estimate (supervisor.estimate_tokens, 2.76 chars/token) is the
# pooled measurement; the densest prompt measured ran 2.257 chars/token, i.e.
# 1.22x the estimated tokens. The chosen window covers 1.25x the estimate.
CTX_HEADROOM = 1.25
# Room the chosen window leaves for the reply. The planner is a reasoning model:
# its <think> block shares the window with the JSON reply (parse_reply strips it).
REPLY_ROOM_TOKENS = 16384
# The least a digest leaves free when the caller fixes num_ctx.
REPLY_RESERVE_TOKENS = 6000
# The largest digest (estimated tokens) whose chosen window stays within
# MAX_NUM_CTX: the budget build_digest trims to when it chooses num_ctx itself.
AUTO_BUDGET_TOKENS = int((MAX_NUM_CTX - REPLY_ROOM_TOKENS) / CTX_HEADROOM)
# The second limit a digest must fit: its staged file travels gzip+base64 inside
# one ssh command line, capped at executor.PLAN_MAX_STAGED_CHARS (Linux caps one
# argument at 128 KiB). A full digest runs 1.41 to 1.57 such characters per
# estimated token (measured on the pilots and realloop_v1), so this line stops
# a digest at roughly 62-69K tokens, about where the context budget
# (AUTO_BUDGET_TOKENS, 65,536) does. build_digest measures the
# staged form exactly (staged_chars) and trims to STAGED_BUDGET_CHARS, 1 KiB
# under the cap for serialisation drift. tests/test_inc_ap_brain.py pins that
# the two constants agree and that staged_chars is what the executor ships.
STAGED_MAX_CHARS = 96 * 1024
STAGED_BUDGET_CHARS = STAGED_MAX_CHARS - 1024
DEFAULT_LIT_K = 12
MAX_RAW_TEXT = 200000

# --- trimming a digest to its budget -------------------------------------------------
#
# Never cut (the minimal digest; if it does not fit, build_digest refuses):
#   the fired diagnoses with every cite; the deterministic proposals; the current
#   experiment's dev tables (CURRENT_KEPT: report summary, chains, per-step
#   table, final dev rows, the per-species gate rows with their per-seed values,
#   and its exp.json/state heads); the lever menu with each lever's track
#   record; the budget; the lineage summary (lever, parent, child, status); a
#   one-line summary of every older experiment; LIT_FLOOR_K literature passages.
# Cut first to last, one cut at a time, only while the prompt is over budget
# (TRIM_ORDER, stage by stage):
#   parent_steps     older experiments' per-step tables, oldest first;
#   parent_tables    older experiments' chains and final dev rows, oldest first,
#                    leaving each one's one-line summary (PARENT_LINE_KEYS);
#   literature       corpus passages from k down to LIT_FLOOR_K, lowest-ranked first;
#   residual_detail  the rest, in this order: silent diagnoses' summaries (id and
#                    name stay); Step 1 excerpts (admit, select, increments,
#                    relevance); the current experiment's build excerpt, then its
#                    increment definitions (exp_steps); lineage records to their
#                    summary; each residual to RESIDUAL_LINE_CHARS.
# A cut is made while the prompt is over its token budget ("over": "context")
# or, once it is within it, while the staged form is over STAGED_BUDGET_CHARS
# ("over": "transport"). Every cut is recorded in the digest's `trimmed` field
# (what, which limit, and the token estimate before and after it) and named in
# the prompt's TRIMMED section, so the brain knows an absent value was cut, not
# missing. The cuts read only the dev-only sections, so the trimmed digest is as
# test-blind as the full one.
TRIM_ORDER = ("parent_steps", "parent_tables", "literature", "residual_detail")
LIT_FLOOR_K = 4
# The current experiment's evidence sections no cut touches. gate_entries are
# its per-species dev deltas (and the per-seed values behind P_data): the
# per-species evidence D13 and novel per-species patterns are read from. Only
# its build excerpt and increment definitions (build, exp_steps) can be cut.
CURRENT_KEPT = ("summary", "chains", "steps", "final_dev", "gate_entries", "exp", "state")
# What an older experiment shows while there is room. "steps" (its per-step
# table) was added with the trimming: before it, older experiments showed only
# summary, chains and final_dev. It is the first thing cut (parent_steps). It
# costs about 11K estimated tokens for three older pilots (realloop_v1 with the
# last three: 45,703 tokens with, 34,815 without), so the live campaign's digest
# now sits near the ssh line and a transport cut of these tables is expected
# from the next experiment on. Dropping "steps" here restores the old content.
PARENT_SECTIONS = ("summary", "chains", "steps", "final_dev")
PARENT_LINE_KEYS = ("exp", "type", "replay_mode", "done", "agreement", "gpu_hours_total")
_STEP1_TRIM = ("admit_summary", "select_summary", "increments_summary", "relevance")
RESIDUAL_LINE_CHARS = 160
LINEAGE_SUMMARY_KEYS = ("lever", "parent_exp", "child_exp", "status")
TRIM_NOTE = ("Cut to fit the context. A value absent for this reason is not evidence of "
             "anything; cite only what is shown.")

PREDICT_METRICS = ("dev_twelve", "agreement")
PREDICT_DIRECTIONS = ("up", "down", "none")

MENU_FILE = Path(__file__).with_name("levers.json")
# The lever-row fields shown to the brain. Anything else in a levers.json row
# stays out of the prompt.
_MENU_FIELDS = ("menu", "title", "name", "kind", "policy_action", "risk", "argv",
                "derived", "fixed", "only_after", "requires", "control", "success",
                "success_criterion", "falsifier", "predicted", "lit", "hypothesis",
                "why_menu_insufficient", "required_change", "cheapest_test")

# The price param of every costed lever. The autopilot computes it from the
# evidence (validate.materialise, with the levers.py estimators); a plan's own
# price is never used (contract (b), "Cost": the autopilot computes it).
PRICE_PARAM = "est_gpu_hours"
# Params of a menu lever that the autopilot sets from the evidence
# (validate.materialise), shown to the brain with each lever. A plan may
# repeat one only with the value the evidence gives. L2 and L6 take the
# replay mode and recipe set of a ready D4 (contract (b), L2: "--replay-mode
# <D4> --recipes <D4 set>"); L5 keeps its parent loop's definition and changes
# only --size; manifest paths come from exp.json and Step 1. --gate-flips-mode
# is the pinned flips mode of the experiment a build follows (levers.gate_params),
# never the plan's choice; only L9 sets it, to net, on a v1 pilot with its own
# replay mode. The relevance criterion of a real loop (--relevance, or
# --increment-sources evidence when Step 1's relevance.json failed its own
# calibration check: levers.increment_criterion) is Step 1's, or the parent
# loop's for L5, never the plan's.
AUTOPILOT_SETS = {
    "L1": ("replay_mode", "gate_flips_mode"),
    "L2": ("base", "relevance", "increment_sources", "replay_mode", "recipes", "gate_flips_mode"),
    "L4": ("trusted", "audits", "out"),
    "L5": ("base", "relevance", "increment_sources", "replay_mode", "recipes", "n_verified", "gate_flips_mode"),
    "L6": ("base", "relevance", "increment_sources", "replay_mode", "recipes", "no_truth", "gate_flips_mode"),
    "L8": ("manifest",),
    "L9": ("replay_mode", "gate_flips_mode"),
}
# Levers that build the real loop need D4 fired as decision_slot_ready
# (contract (b) D4 and (f) R4a: no L2 while no recipe tracks the truth arm);
# L5 and L6 are L2 with a larger --size or --no-truth.
D4_GATED_ACTIONS = ("inc_build_realloop",)
D4_READY = "decision_slot_ready"

PROMPT_HEADER = """You are the research planner of an incremental-training (INC) campaign for a weed
detector (YOLO11n; 12 cwd12 species plus OtherPlant). Each INC experiment trains a base model,
adds small data increments one at a time, and a pinned statistical gate on the dev split accepts
or rejects each increment; a truth arm (cold union runs with and without the increment) says what
each increment really does. A deterministic layer has already diagnosed the latest experiment and
made its own proposal (DETERMINISTIC). You are advisory: rank the pre-registered levers for the NEXT
experiment, and state hypotheses the menu cannot test.

Rules. An item that breaks one is dropped; the rest of your plan survives.
1. Decisions use the dev split only. Do not mention, cite or reason about test, ood22, ood23,
   imageweeds or any holdout. All evidence below is dev-only.
2. ranked_menu items name a lever id from MENU whose "menu" is "menu". "params" holds only that
   lever's free params, inside its param_bounds and equal to its "fixed" values where given. The
   autopilot sets the rest from the evidence: the params under "set_by_autopilot" (repeat one only
   with the value the evidence gives), the name "exp" of a new experiment when you give none (a name
   you give must be new), and the price. Never give est_gpu_hours: every item is priced from the GPU
   time its parent experiment measured, and an item that cannot be priced is dropped.
3. A lever's "preconditions" must hold on DIAGNOSES: "only_after" diagnoses must have fired,
   "requires" params must be given, and "needs_D4" means D4 must have fired with that name. An idea
   outside the menu goes to off_menu, never to ranked_menu.
4. Every ranked_menu item has at least one evidence_cite. An evidence_cite is {"artifact", "pointer",
   "value"} (ledger rows also "line") copied EXACTLY from EVIDENCE or from a diagnosis cite. An
   object carrying "_at" is a subset of the artifact object at that address with the same keys and
   nesting: the pointer of a value inside it is _at.pointer + "/" + its key path, list positions are
   0-based indices, and a ledger row's line is _at.line. "value" is the exact value shown (full
   precision, same type). "trigger" lists the fired diagnosis ids the item answers; a diagnosis
   counts only when the item repeats at least one of that diagnosis's cites (same artifact, line and
   pointer). Without one the item is filed as your own idea, attributed to no diagnosis.
5. lit_cites are optional. Each is {"paper_id", "line", "quote"}: quote is copied character for
   character from one LITERATURE passage and holds at least 20 characters of the passage's own text
   (the "KIND [topic]: " label does not count); line is that passage's line. LITERATURE passages are
   curated notes about the papers, not the authors' own words; US passages are this project's notes.
6. predicted = {"metric": "dev_twelve" or "agreement", "direction": "up", "down" or "none",
   "magnitude": number or null, "chain": "full", "freeze" or "lora" (optional; omitted = every
   chain)}. falsifier = the observation that would show the item wrong.
7. Reply with ONE JSON object and nothing else, of this shape:
{"ranked_menu": [{"lever": "L1", "params": {}, "trigger": [], "rationale": "", "evidence_cites": [],
  "lit_cites": [], "predicted": {"metric": "agreement", "direction": "up", "magnitude": null},
  "falsifier": ""}],
 "off_menu": [{"hypothesis": "", "why_menu_insufficient": "", "required_change": "",
  "cheapest_test": "", "control": "", "success_criterion": "", "lit_cites": []}],
 "stop_recommendation": {"stop": false, "reason": ""}}
"""


# --- dev-only handling ----------------------------------------------------------

def dev_only(obj):
    """A dev-only copy of `obj`: evidence.scrub, the one scrubbing rule.

    Drops every dict key naming a non-dev exam at any depth (report.py:56 puts
    test into every report's final table), every score stamp of such an exam
    and every non-dev score path; list positions are kept (a dropped element
    becomes None), so a pointer into the scrubbed copy addresses the same
    value as a pointer into the file and as evidence.py's cites.
    """
    from . import evidence as EV
    return EV.scrub(obj)[0]


def dev_leaks(obj, path="", strings=True):
    """[path, ...] of every forbidden exam key or score stamp (and, with
    `strings`, every string naming a forbidden score path or exam pointer)
    inside `obj`."""
    found = []
    if isinstance(obj, dict):
        if isinstance(obj.get("exam"), str) and obj["exam"] in FORBIDDEN_EXAMS:
            found.append(path or "/")
        for k, v in obj.items():
            p = "%s/%s" % (path, k)
            if k in FORBIDDEN_EXAMS:
                found.append(p)
            found.extend(dev_leaks(v, p, strings))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(dev_leaks(v, "%s/%d" % (path, i), strings))
    elif strings and isinstance(obj, str) and _FORBIDDEN_PATH_RE.search(obj):
        found.append(path)
    return found


def assert_dev_only(obj, where="digest"):
    leaks = dev_leaks(obj)
    if leaks:
        raise ValueError("%s carries non-dev exam data at %s" % (where, ", ".join(leaks[:5])))


# --- artifacts ------------------------------------------------------------------

def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_artifacts(inc_dir, exps):
    """{"<exp>/<file>": dev-only content} for experiments `exps` (and step1/)
    under `inc_dir`: evidence.load_dir (its allow-list and its scrub; score
    files are never opened), as artifacts_of() returns it."""
    from . import evidence as EV
    exps = list(exps)
    return artifacts_of(EV.load_dir(inc_dir, exps[0], exps=exps))


def artifacts_of(snapshot):
    """The {artifact name: content} map this module and validate.py read.

    Accepts an evidence.Evidence (its scrubbed artifacts, and each ledger as
    "<exp>/ledger.jsonl": a list whose index i holds line i+1, None where the
    file had no entry), a dict carrying that map under "artifacts", or the map
    itself.
    """
    if hasattr(snapshot, "artifacts") and hasattr(snapshot, "ledgers"):
        arts = dict(snapshot.artifacts)
        for exp, rows in snapshot.ledgers.items():
            n = max([ln for ln, _ in rows] or [0])
            lst = [None] * n
            for ln, entry in rows:
                lst[ln - 1] = entry
            arts["%s/ledger.jsonl" % exp] = lst
        return arts
    if not isinstance(snapshot, dict):
        return {}
    if isinstance(snapshot.get("artifacts"), dict):
        return snapshot["artifacts"]
    return {k: v for k, v in snapshot.items() if isinstance(k, str) and "/" in k}


def load_menu(path=None):
    """levers.json -> {id: row}. Missing file -> {} (no menu, no ranked items)."""
    p = Path(path) if path else MENU_FILE
    if not p.is_file():
        return {}
    return normalise_menu(_read_json(p))


def normalise_menu(raw):
    """{id: row}, each row with "menu": "menu" | "off_menu".

    levers.json's shape: {"levers": {L-id: row}, "cards": {X-id: row}} --
    every "levers" row is on the menu (unless R4), every "cards" row is an
    off-menu R4 card. A row's own "kind" (build / job / driver) is left alone.
    Also accepted: {id: row} or [row with "id"], classified by id and risk.
    Keys starting with "_" are metadata.
    """
    def items(rows):
        if isinstance(rows, list):
            return [(r.get("id"), r) for r in rows if isinstance(r, dict)]
        if isinstance(rows, dict):
            return [(k, v) for k, v in rows.items() if not str(k).startswith("_")]
        return []

    if isinstance(raw, dict) and ("levers" in raw or "cards" in raw):
        blocks = [("levers", raw.get("levers")), ("cards", raw.get("cards"))]
    else:
        blocks = [(None, raw)]
    out = {}
    for block, rows in blocks:
        for lid, row in items(rows):
            if not lid or not isinstance(row, dict):
                continue
            lid = str(lid)
            off = lid.startswith("X") or row.get("risk") == "R4" or block == "cards"
            row = dict(row, id=lid, menu="off_menu" if off else "menu")
            row["fixed_params"] = dict(row.get("fixed") or row.get("fixed_params") or {})
            out[lid] = row
    return out


def param_bounds(action):
    """The policy table's param_bounds for `action`, or None if the table has no row."""
    if not action:
        return None
    from ..brain import policy
    d = policy.describe(action)
    return dict(d.get("param_bounds") or {}) if d.get("known") else None


def declared_bounds(row):
    """The param_bounds a lever row declares itself ("param_bounds", or the
    older "params"), or {} when it declares none in that form."""
    for key in ("param_bounds", "params"):
        own = (row or {}).get(key)
        if isinstance(own, dict) and own and all(isinstance(v, dict) and "type" in v
                                                 for v in own.values()):
            return dict(own)
    return {}


def lever_bounds(row):
    """(bounds, source) a proposer's params for this menu row are checked against.

    The lever row's own "param_bounds" first: levers.json keeps them in
    policy_actions.json's form, and they are the proposer-facing
    interface (a lever's derived values, such as L4's manifest paths, are read
    from the evidence at filing time, never supplied by a proposer, so the
    policy row's bounds on the rendered command are not the proposer's). Only
    a row that declares no params falls back to the policy table's row for its
    action. The executor re-authorizes the rendered command against the policy
    row either way. Neither -> (None, reason): every params dict is refused.
    """
    own = declared_bounds(row)
    if own:
        return own, "levers.json:%s" % (row.get("id"),)
    action = (row or {}).get("policy_action")
    b = param_bounds(action)
    if b is not None:
        return b, "policy_actions.json:%s" % action
    return None, "no bounds on lever %s and no policy row for %r" % ((row or {}).get("id"), action)


def actor_for(model_name):
    """'tier2:<model>' in the character set policy._ACTOR_RE accepts (':' is not)."""
    ident = re.sub(r"[^A-Za-z0-9._+@/-]", "_", str(model_name or "unknown"))[:128]
    return "tier2:%s" % (ident or "unknown")


# --- evidence projection --------------------------------------------------------

def _project(obj, spec):
    """The subset of `obj` named by `spec`, same keys and nesting.

    spec: {key: True | sub-spec}; the key "*" applies its sub-spec to every
    key of a dict. Missing keys are skipped, never invented.
    """
    if spec is True:
        return copy.deepcopy(obj)
    if not isinstance(obj, dict) or not isinstance(spec, dict):
        return None
    out = {}
    if "*" in spec:
        for k, v in obj.items():
            got = _project(v, spec["*"])
            if got is not None:
                out[k] = got
    for k, sub in spec.items():
        if k == "*" or k not in obj:
            continue
        got = _project(obj[k], sub)
        if got is not None:
            out[k] = got
    return out


def _at(artifact, pointer="", line=None):
    at = {"artifact": artifact, "pointer": pointer}
    if line is not None:
        at["line"] = line
    return at


_CHAIN_STEP_SPEC = {k: True for k in (
    "verdict", "agree", "truth_equivalent", "p_data", "p_recipe", "inc", "cand_mean",
    "cand_sd", "null_mean", "null_sd", "guards", "train_images", "d_images", "pool_images",
    "warmup_epochs_effective")}
_CHAIN_STEP_SPEC["attribution"] = {k: True for k in (
    "blame", "recipe_flag", "class_vs_loc", "map_delta", "agnostic_delta", "species_failed")}
_REPORT_TOP_SPEC = {k: True for k in (
    "exp", "type", "replay_mode", "seeds", "done", "agreement", "gpu_hours", "gpu_hours_total",
    "attribution_scope", "interventions", "blocked", "label_steps")}
_REPORT_TOP_SPEC["effective_warmup"] = {"incremental_effective_epochs": True, "note": True}
_CHAINS_SPEC = {"*": {k: True for k in (
    "phase", "accepted", "neutral", "quarantined", "final_incumbent", "bswap", "unverified",
    "rejects_without_source_attribution")}}
_CHAINS_SPEC["*"]["recipe"] = {k: True for k in (
    "trainer", "epochs", "lr0", "warmup_epochs", "freeze", "lora")}
# Per-species deltas on dev (cand - null), one row per gate entry. The step
# table above already carries each entry's verdict, P values and means, so a
# gate row adds only what the report does not: the per-species split and the
# per-seed values behind P_data.
_LEDGER_GATE_SPEC = {"id": True,
                     "decision": {"cand_values": True, "null_values": True,
                                  "attribution": {"per_species_delta": True,
                                                  "class_only_delta": True}}}
_EXP_STEP_SPEC = {"name": True, "clean": True, "planted": True, "n_images": True}
_BUILD_SPEC = {"replay_mode": True, "sequence": True,
               "bswap": {"boxes": True, "changed": True, "share": True},
               "breal": {"n": True, "sampled_per_slug": True, "class_order_verified": True},
               "size": True, "increments": True, "unverified": {"source": True, "verdicts": True,
                                                                "source_relevance": True},
               "relevance": {"excluded_sources": True}}
_STEP1_SPECS = {
    "select_summary.json": {"sizes": True, "sources": {"increment_pool": True,
                                                       "below_gate": True},
                            "retrieval": {"source_evidence": True}},
    "admit_summary.json": {"images": True, "boxes": True, "per_slug": True},
    "increments_summary.json": {"increments": True, "relevance": True},
    "relevance.json": {"calibration": {"tau": True},
                       "increment_pool": {"sources": {"*": {
                           "status": True, "p_plant_median": True,
                           "share_at_or_above_tau": True, "top_set_share": True,
                           "crops_usable": True}}, "not_passing": True}},
}


def evidence_excerpts(artifacts, exp, per_species=True):
    """{section: [rows with _at]} for experiment `exp`, dev only."""
    ev = {}
    rep_name = "%s/report.json" % exp
    rep = artifacts.get(rep_name)
    if isinstance(rep, dict):
        top = _project(rep, _REPORT_TOP_SPEC)
        top["_at"] = _at(rep_name, "")
        ev["summary"] = [top]
        if isinstance(rep.get("chains"), dict):
            ch = _project(rep["chains"], _CHAINS_SPEC)
            ch["_at"] = _at(rep_name, "/chains")
            ev["chains"] = [ch]
        rows = []
        for i, st in enumerate(rep.get("steps") or []):
            if not isinstance(st, dict):
                continue
            row = _project(st, {"step": True, "tag": True, "k": True, "clean": True,
                                "truth": True, "chains": {"*": _CHAIN_STEP_SPEC}})
            row["_at"] = _at(rep_name, "/steps/%d" % i)
            rows.append(row)
        if rows:
            ev["steps"] = rows
        finals = []
        for i, row in enumerate(rep.get("final") or []):
            if isinstance(row, dict) and isinstance((row.get("exams") or {}).get("dev"), dict):
                finals.append({"_at": _at(rep_name, "/final/%d" % i), "model": row.get("model"),
                               "exams": {"dev": copy.deepcopy(row["exams"]["dev"])}})
        if finals:
            ev["final_dev"] = finals
    exp_name = "%s/exp.json" % exp
    ex = artifacts.get(exp_name)
    if isinstance(ex, dict):
        rows = []
        for i, st in enumerate(ex.get("steps") or []):
            if isinstance(st, dict):
                row = _project(st, _EXP_STEP_SPEC)
                row["_at"] = _at(exp_name, "/steps/%d" % i)
                rows.append(row)
        if rows:
            ev["exp_steps"] = rows
        head = _project(ex, {"type": True, "builder": True, "decision_exam": True,
                             "replay_mode": True, "seeds": True})
        head["_at"] = _at(exp_name, "")
        ev["exp"] = [head]
    b_name = "%s/build_summary.json" % exp
    b = artifacts.get(b_name)
    if isinstance(b, dict):
        row = _project(b, _BUILD_SPEC)
        row["_at"] = _at(b_name, "")
        ev["build"] = [row]
    st_name = "%s/state.json" % exp
    stt = artifacts.get(st_name)
    if isinstance(stt, dict):
        row = _project(stt, {"generation": True, "done": True, "blocked": True})
        row["_at"] = _at(st_name, "")
        ev["state"] = [row]
    l_name = "%s/ledger.jsonl" % exp
    led = artifacts.get(l_name)
    if per_species and isinstance(led, list):
        rows = []
        for i, rec in enumerate(led, 1):
            if isinstance(rec, dict) and (rec.get("type") == "gate"
                                          or str(rec.get("id", "")).startswith("gate/")):
                row = _project(rec, _LEDGER_GATE_SPEC)
                row["_at"] = _at(l_name, "", line=i)
                rows.append(row)
        if rows:
            ev["gate_entries"] = rows
    return ev


def step1_excerpts(artifacts):
    ev = {}
    for fname, spec in _STEP1_SPECS.items():
        name = "step1/%s" % fname
        obj = artifacts.get(name)
        if isinstance(obj, dict):
            row = _project(obj, spec)
            row["_at"] = _at(name, "")
            ev[fname.rsplit(".", 1)[0]] = [row]
    return ev


# --- the digest -------------------------------------------------------------------

def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _dump(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def menu_section(menu, track=None, corpus=None):
    """Lever rows for the prompt: whitelisted fields, bounds, track record.

    A lever's literature reference that the corpus knows (an id or an alias
    such as ibrahim2024) is annotated with its corpus id, so the brain can
    find the passages it may quote.
    """
    levers = ((track or {}).get("levers") or {})
    rows = []
    for lid in sorted(menu):
        row = menu[lid]
        out = {"id": lid}
        for f in _MENU_FIELDS:
            if f in row and row[f] not in (None, "", [], {}):
                out[f] = copy.deepcopy(row[f])
        if row.get("menu") == "menu":
            bounds, source = lever_bounds(row)
            out["param_bounds"] = {k: v for k, v in bounds.items() if k != PRICE_PARAM} \
                if bounds is not None else "%s: any params are refused" % source
            out["set_by_autopilot"] = sorted(set(AUTOPILOT_SETS.get(lid, ())) | {PRICE_PARAM})
            pre = {}
            if row.get("only_after"):
                pre["only_after"] = list(row["only_after"])
            if row.get("requires"):
                pre["requires"] = list(row["requires"])
            if row.get("policy_action") in D4_GATED_ACTIONS:
                pre["needs_D4"] = D4_READY
            if pre:
                out["preconditions"] = pre
        if corpus is not None and isinstance(out.get("lit"), list):
            for ref in out["lit"]:
                if isinstance(ref, dict) and ref.get("paper_id"):
                    ref["corpus_id"] = corpus.canonical_id(ref["paper_id"])
        out["track_record"] = copy.deepcopy(levers.get(lid) or {"scored": 0})
        rows.append(out)
    return rows


def literature_query(diagnoses, menu, residuals):
    """The BM25 query: fired diagnoses, their levers' text, and the residuals."""
    parts, levers = [], []
    for d in diagnoses or []:
        if d.get("fired"):
            parts.append(str(d.get("name", "")).replace("_", " "))
            parts.append(str(d.get("summary", "")))
            for lv in d.get("levers") or []:
                if lv not in levers:
                    levers.append(lv)
    for lv in levers:
        row = (menu or {}).get(lv) or {}
        for f in ("title", "control", "success", "falsifier", "hypothesis"):
            if isinstance(row.get(f), str):
                parts.append(row[f])
    parts.extend(str(r) for r in (residuals or []))
    return " ".join(p for p in parts if p), levers


def literature_section(corpus, diagnoses, menu, residuals, k=DEFAULT_LIT_K):
    if corpus is None or k <= 0:
        return []
    query, levers = literature_query(diagnoses, menu, residuals)
    if not query.strip():
        query = "incremental training replay forgetting data curation label noise"
    hits = corpus.search(query, k=k, levers=levers)
    return [{"paper_id": h["paper_id"], "line": h["line"], "kind": h["kind"],
             "topic": h["topic"], "title": h["title"], "text": h["text"]} for h in hits]


def render_prompt(sections):
    parts = [PROMPT_HEADER.rstrip(), ""]
    order = (("DIAGNOSES", "diagnoses"), ("DETERMINISTIC", "deterministic"),
             ("EVIDENCE", "evidence"), ("TRIMMED", "trimmed"), ("MENU", "menu"),
             ("LINEAGE", "lineage"), ("BUDGET", "budget"), ("RESIDUALS", "residuals"),
             ("LITERATURE", "literature"))
    for title, key in order:
        if key in sections:
            parts.append("### %s" % title)
            parts.append(_dump(sections[key]))
            parts.append("")
    parts.append("Reply with the JSON object only.")
    return "\n".join(parts)


def choose_num_ctx(tokens):
    """The window for a digest of `tokens` estimated tokens: CTX_HEADROOM times
    the estimate plus REPLY_ROOM_TOKENS, rounded up to NUM_CTX_STEP, at least
    DEFAULT_NUM_CTX and at most MAX_NUM_CTX (a digest over AUTO_BUDGET_TOKENS is
    trimmed or refused before this is asked)."""
    need = int(math.ceil(int(tokens) * CTX_HEADROOM)) + REPLY_ROOM_TOKENS
    ctx = -(-need // NUM_CTX_STEP) * NUM_CTX_STEP
    return int(min(MAX_NUM_CTX, max(DEFAULT_NUM_CTX, ctx)))


def gpu_mem_needed_gb(num_ctx, kv_bytes_per_token=KV_BYTES_PER_TOKEN_ASSUMED,
                      weights_gb=PLANNER_WEIGHTS_GB):
    """GB the plan job needs to hold the planner at `num_ctx`: weights, an f16
    KV cache of `kv_bytes_per_token` (by default the assumed, unverified layout
    documented at MAX_NUM_CTX) and the runtime."""
    return float(weights_gb) + int(num_ctx) * kv_bytes_per_token / 1e9 + RUNTIME_OVERHEAD_GB


def _layout_int(v):
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


def kv_layout(model_info):
    """(bytes per token of an f16 KV cache, description) from an ollama /api/show
    `model_info`, or (None, why) when the layout cannot be read.

    Read: <arch>.block_count, <arch>.attention.head_count_kv (one number, or one
    per layer; 0 = a layer without a KV cache), key_length and value_length
    (default embedding_length / head_count), and full_attention_interval (a
    hybrid layout: only every n-th layer holds a KV cache). A sliding-window
    layer is counted as full, so the figure errs high.
    """
    info = model_info if isinstance(model_info, dict) else {}
    arch = info.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None, "no general.architecture in model_info"

    def g(key):
        return info.get("%s.%s" % (arch, key))

    layers = _layout_int(g("block_count"))
    heads = _layout_int(g("attention.head_count"))
    kv = g("attention.head_count_kv")
    if kv is None:
        kv = g("attention.head_count")
    emb = _layout_int(g("embedding_length"))
    k_len = _layout_int(g("attention.key_length")) or (emb // heads if emb and heads else None)
    v_len = _layout_int(g("attention.value_length")) or k_len
    if not layers or not k_len or not v_len or kv is None:
        return None, "%s: block_count, KV heads or head size missing from model_info" % arch
    per_layer = list(kv) if isinstance(kv, list) else [kv] * layers
    per_layer = [_layout_int(h) or 0 for h in per_layer]
    interval = _layout_int(g("full_attention_interval"))
    if interval and interval > 1:
        per_layer = [h if (i + 1) % interval == 0 else 0 for i, h in enumerate(per_layer)]
    with_kv = sum(1 for h in per_layer if h)
    total = sum(per_layer) * (k_len + v_len) * 2
    heads_kv = sorted(set(h for h in per_layer if h))
    return total, ("%s: %d layers, %d with a KV cache, x %s KV heads x (%d + %d) dims, f16"
                   % (arch, len(per_layer), with_kv, "/".join(str(h) for h in heads_kv) or "0",
                      k_len, v_len))


def layout_line(model_info, num_ctx, weights_gb=None, gpu_gb=JOB_GPU_MEM_GB):
    """The job's log line: the model's own KV bytes per token (kv_layout) and
    what `num_ctx` needs on the GPU, against the assumed layout MAX_NUM_CTX was
    sized for. Logged, never enforced: the resident size after warm-up
    (/api/ps) is the real figure."""
    per_tok, desc = kv_layout(model_info)
    if per_tok is None:
        return "[layout] unknown (%s); brain_plan assumes %d KiB per token" % (
            desc, KV_BYTES_PER_TOKEN_ASSUMED // 1024)
    w = PLANNER_WEIGHTS_GB if weights_gb is None else float(weights_gb)
    need = gpu_mem_needed_gb(num_ctx, per_tok, w)
    return ("[layout] %s: %.0f KiB per token (brain_plan assumes %d KiB, %.2fx); at num_ctx %d: "
            "%.1f GB of KV, %.1f GB with %.1f GB of weights and %.0f GB of runtime, of %.0f GB%s"
            % (desc, per_tok / 1024.0, KV_BYTES_PER_TOKEN_ASSUMED // 1024,
               per_tok / float(KV_BYTES_PER_TOKEN_ASSUMED), int(num_ctx),
               int(num_ctx) * per_tok / 1e9, need, w, RUNTIME_OVERHEAD_GB, gpu_gb,
               "" if need <= gpu_gb else " -- OVER THE GPU: expect layers on the CPU"))


def staged_chars(digest):
    """Characters `digest` takes on the ssh command line that stages it: the
    file campaign._write_json writes (indent 1, sorted keys), gzip (mtime 0),
    base64 -- what executor._plan_segment ships and checks against
    PLAN_MAX_STAGED_CHARS."""
    raw = (json.dumps(digest, indent=1, sort_keys=True, default=str) + "\n").encode("utf-8")
    return 4 * ((len(gzip.compress(raw, mtime=0)) + 2) // 3)


def _parent_line(block):
    """An older experiment's one-line summary: PARENT_LINE_KEYS of its report
    summary row, at the same address (so every value stays citable)."""
    rows = (block or {}).get("summary") or []
    if not rows or not isinstance(rows[0], dict):
        return None
    line = {k: copy.deepcopy(rows[0][k]) for k in PARENT_LINE_KEYS if k in rows[0]}
    line["_at"] = copy.deepcopy(rows[0].get("_at"))
    return line


def _trim_cuts(exp, parents, lit_k):
    """[(stage, cut)] in TRIM_ORDER. A cut is applied to the sections in place
    and returns what it removed (a short string), or None when it found nothing
    to remove (then nothing is recorded)."""
    cuts = []

    def parent_steps(p):
        def cut(s):
            rows = s["evidence"].get(p, {}).pop("steps", None)
            return ("evidence/%s/steps: the per-step table (%d rows) of an older experiment"
                    % (p, len(rows))) if rows else None
        return cut

    def parent_tables(p):
        def cut(s):
            block = s["evidence"].get(p)
            if not block:
                return None
            line = _parent_line(block)
            new = {"summary": [line]} if line else {}
            if _dump(new) == _dump(block):
                return None
            s["evidence"][p] = new
            gone = sorted(k for k in block if k != "summary")
            if line:
                return ("evidence/%s: %s cut to its one-line summary (%s)"
                        % (p, ", ".join(gone + ["the rest of its summary row"]),
                           ", ".join(PARENT_LINE_KEYS)))
            return "evidence/%s: %s cut (no report summary to keep)" % (p, ", ".join(gone))
        return cut

    def literature(s):
        lit = s.get("literature") or []
        if len(lit) <= LIT_FLOOR_K:
            return None
        h = lit.pop()
        return ("literature: passage %s line %s (rank %d of %d; %d kept)"
                % (h.get("paper_id"), h.get("line"), len(lit) + 1, lit_k, len(lit)))

    def silent_summaries(s):
        silent = s["diagnoses"].get("silent") or []
        if not any("summary" in d for d in silent):
            return None
        s["diagnoses"]["silent"] = [{"id": d.get("id"), "name": d.get("name")} for d in silent]
        return ("diagnoses/silent: the summaries of %d diagnoses that did not fire "
                "(ids and names kept)" % len(silent))

    def step1(name):
        def cut(s):
            s1 = s["evidence"].get("step1") or {}
            if name not in s1:
                return None
            s1.pop(name)
            if not s1:
                s["evidence"].pop("step1", None)
            return "evidence/step1/%s: the Step 1 excerpt" % name
        return cut

    def current(key, what):
        assert key not in CURRENT_KEPT, key

        def cut(s):
            rows = s["evidence"].get(exp, {}).pop(key, None)
            return ("evidence/%s/%s: %s (%d row%s)" % (exp, key, what, len(rows),
                                                      "" if len(rows) == 1 else "s")) if rows else None
        return cut

    def lineage(s):
        recs = s.get("lineage") or []
        short = [{k: r.get(k) for k in LINEAGE_SUMMARY_KEYS} if isinstance(r, dict) else r
                 for r in recs]
        if _dump(short) == _dump(recs):
            return None
        s["lineage"] = short
        return "lineage: %d records cut to %s" % (len(recs), ", ".join(LINEAGE_SUMMARY_KEYS))

    def residuals(s):
        res = s.get("residuals") or []
        n = sum(1 for r in res if len(r) > RESIDUAL_LINE_CHARS)
        if not n:
            return None
        s["residuals"] = [r if len(r) <= RESIDUAL_LINE_CHARS
                          else r[:RESIDUAL_LINE_CHARS - 3] + "..." for r in res]
        return "residuals: %d of %d cut to %d characters" % (n, len(res), RESIDUAL_LINE_CHARS)

    for p in parents:
        cuts.append(("parent_steps", parent_steps(p)))
    for p in parents:
        cuts.append(("parent_tables", parent_tables(p)))
    for _ in range(max(0, int(lit_k) - LIT_FLOOR_K)):
        cuts.append(("literature", literature))
    cuts.append(("residual_detail", silent_summaries))
    for name in _STEP1_TRIM:
        cuts.append(("residual_detail", step1(name)))
    cuts.append(("residual_detail", current("build", "the build_summary excerpt")))
    cuts.append(("residual_detail", current("exp_steps", "the increment definitions")))
    cuts.append(("residual_detail", lineage))
    cuts.append(("residual_detail", residuals))
    return cuts


def build_digest(artifacts, exp, diagnoses, menu, campaign="default", n=0, track=None,
                 lineage=None, budget=None, residuals=None, deterministic=None,
                 corpus=None, lit_k=DEFAULT_LIT_K, parents=(), num_ctx=None,
                 created_utc=None):
    """The brain's input: sections, the rendered prompt, and hashes.

    Raises ValueError when any evidence section carries a non-dev exam (the
    serialiser assertion of contract (d)).

    `parents` are the older experiments shown beside `exp`, oldest first; each
    shows PARENT_SECTIONS (its report summary, chains, per-step table and final
    dev rows) while there is room.

    Fitting the context: with `num_ctx` None (the default, what the campaign
    uses) the budget is AUTO_BUDGET_TOKENS and num_ctx is then chosen from the
    digest's size (choose_num_ctx, capped at MAX_NUM_CTX); with a number, the
    budget is num_ctx - REPLY_RESERVE_TOKENS (a number over MAX_NUM_CTX is
    refused). Either way the staged form must also fit the ssh line that
    carries it (STAGED_BUDGET_CHARS). A digest over either limit is trimmed in
    TRIM_ORDER, one cut at a time, until it fits; the `trimmed` field records
    every cut with the limit it was over and the token estimate before and
    after it, plus the totals before and after trimming. Only a digest that is
    over a limit with every cut made (the minimal digest) raises. Fired
    diagnoses and their cites, and the current experiment's CURRENT_KEPT
    sections, are never cut.
    """
    from ..brain import supervisor
    fixed = num_ctx not in (None, 0)
    if fixed and int(num_ctx) > MAX_NUM_CTX:
        raise ValueError("num_ctx %d is over MAX_NUM_CTX %d, the most the plan job's GPU is "
                         "sized for" % (int(num_ctx), MAX_NUM_CTX))
    artifacts = artifacts_of(artifacts) or {}
    diagnoses = list(diagnoses or [])
    evidence = {exp: evidence_excerpts(artifacts, exp)}
    older = []
    for par in parents or ():
        if par == exp or par in older or par == "step1":
            continue
        older.append(par)
        ex = evidence_excerpts(artifacts, par, per_species=False)
        evidence[par] = {k: v for k, v in ex.items() if k in PARENT_SECTIONS}
    s1 = step1_excerpts(artifacts)
    if s1:
        evidence["step1"] = s1
    sections = {
        "diagnoses": {"fired": [d for d in diagnoses if d.get("fired")],
                      "silent": [{"id": d.get("id"), "name": d.get("name"),
                                  "summary": d.get("summary")}
                                 for d in diagnoses if not d.get("fired")]},
        "deterministic": [{k: p.get(k) for k in ("lever", "params", "trigger", "risk",
                                                 "proposed_by", "control", "success")}
                          for p in (deterministic or [])],
        "evidence": evidence,
        "menu": menu_section(menu or {}, track, corpus),
        "lineage": copy.deepcopy(lineage or []),
        "budget": copy.deepcopy(budget or {}),
        "residuals": [str(r) for r in (residuals or [])],
    }
    # The staged form: what the cluster job reads back (check_staged compares
    # the prompt with a rendering of these sections, so an int key or a tuple
    # must not render differently after the JSON round trip).
    sections = json.loads(json.dumps(sections))
    for key in list(sections):
        assert_dev_only(sections[key], "digest section %r" % key)
    sections["literature"] = json.loads(json.dumps(
        literature_section(corpus, diagnoses, menu or {}, residuals, lit_k)))
    # The literature carries no values of ours, but it must not smuggle an
    # exam-keyed object in either.
    if dev_leaks(sections["literature"], strings=False):
        raise ValueError("literature section carries an exam-keyed object")

    est = supervisor.estimate_tokens
    budget_tokens = (int(num_ctx) - REPLY_RESERVE_TOKENS) if fixed else AUTO_BUDGET_TOKENS
    fired_before = _dump(sections["diagnoses"]["fired"])
    kept_before = _dump({k: sections["evidence"][exp].get(k) for k in CURRENT_KEPT})
    head = {"schema": DIGEST_SCHEMA, "campaign": str(campaign), "n": int(n), "exp": exp,
            "parents": list(parents or ()), "decision_exam": M.DECISION_EXAM,
            "created_utc": created_utc or M.utc_now()}
    cuts = []
    record = {"num_ctx_from": "caller" if fixed else "digest size", "budget_tokens": budget_tokens,
              "staged_max_chars": STAGED_MAX_CHARS, "tokens_before": None,
              "staged_chars_before": None, "tokens_after": None, "cuts": cuts}

    def assemble(prompt, tokens):
        record["tokens_after"] = tokens
        dg = dict(head, num_ctx=int(num_ctx) if fixed else choose_num_ctx(tokens),
                  sections=sections, trimmed=record,
                  evidence_sha256=_sha(sections["evidence"]), prompt=prompt,
                  prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  tokens_estimated=tokens)
        dg["sha256"] = _sha({k: v for k, v in dg.items() if k != "created_utc"})
        return dg

    prompt = render_prompt(sections)
    tokens = est(prompt)
    record["tokens_before"] = tokens
    record["staged_chars_before"] = staged_chars(assemble(prompt, tokens))
    pending = _trim_cuts(exp, older, len(sections["literature"]))
    while True:
        over, staged = "context", None
        if tokens <= budget_tokens:
            digest = assemble(prompt, tokens)
            staged = staged_chars(digest)
            if staged <= STAGED_BUDGET_CHARS:
                break
            over = "transport"
        what = None
        while pending and not what:
            stage, cut = pending.pop(0)
            what = cut(sections)
        if not what:
            if over == "context":
                raise ValueError(
                    "digest is about %d tokens even trimmed to its minimum (%d before trimming, "
                    "%d cuts), over %s" % (
                        tokens, record["tokens_before"], len(cuts),
                        "num_ctx %d minus the reply reserve" % int(num_ctx) if fixed else
                        "the %d-token budget of the largest context the plan job can hold "
                        "(num_ctx %d, MAX_NUM_CTX)" % (budget_tokens, MAX_NUM_CTX)))
            raise ValueError(
                "the staged digest is %d characters gzip+base64 even trimmed to its minimum "
                "(%d before trimming, %d cuts), over the %d one ssh command line carries "
                "(STAGED_MAX_CHARS, executor.PLAN_MAX_STAGED_CHARS)"
                % (staged, record["staged_chars_before"], len(cuts), STAGED_BUDGET_CHARS))
        rec = {"stage": stage, "cut": what, "over": over, "tokens_before": tokens}
        if staged is not None:
            rec["staged_chars_before"] = staged
        cuts.append(rec)
        sections["trimmed"] = {"note": TRIM_NOTE, "cut": [c["cut"] for c in cuts]}
        prompt = render_prompt(sections)
        tokens = est(prompt)
        rec["tokens_after"] = tokens
    # The cuts touch neither the fired diagnoses nor the dev-only guarantee;
    # checked, not assumed.
    if _dump(sections["diagnoses"]["fired"]) != fired_before:
        raise ValueError("trimming changed a fired diagnosis; refused")
    if _dump({k: sections["evidence"][exp].get(k) for k in CURRENT_KEPT}) != kept_before:
        raise ValueError("trimming changed the current experiment's dev tables; refused")
    for key in sections:
        if key != "literature":
            assert_dev_only(sections[key], "trimmed digest section %r" % key)
    return digest


# --- the reply ----------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_reply(text):
    """(plan, problems). plan is None when no JSON object can be read.

    Tolerates a reasoning model's <think> block and a ``` fence around the
    object. Normalises the three top-level keys; each item's own checks are
    validate.py's job.
    """
    problems = []
    if not isinstance(text, str) or not text.strip():
        return None, ["empty reply"]
    body = _THINK_RE.sub("", text).strip()
    obj = None
    candidates = [body] + [m.group(1) for m in _FENCE_RE.finditer(body)]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            break
        except Exception:
            pass
    if obj is None:
        dec = json.JSONDecoder()
        for i, ch in enumerate(body):
            if ch == "{":
                try:
                    obj, _ = dec.raw_decode(body[i:])
                    break
                except Exception:
                    continue
    if not isinstance(obj, dict):
        return None, ["reply carries no JSON object"]
    plan = {}
    for key in ("ranked_menu", "off_menu"):
        v = obj.get(key, [])
        if not isinstance(v, list):
            problems.append("%s is not a list; ignored" % key)
            v = []
        plan[key] = v
    plan["stop_recommendation"] = obj.get("stop_recommendation")
    extra = sorted(set(obj) - {"ranked_menu", "off_menu", "stop_recommendation"})
    if extra:
        problems.append("unknown top-level keys ignored: %s" % ", ".join(extra))
    return plan, problems


def _atomic_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


# A non-dev exam as a JSON key in the rendered prompt (an escaped quote inside
# a literature string is not a key).
_PROMPT_KEY_LEAK_RE = re.compile(r'(?<!\\)"(?:%s)"\s*:' % "|".join(FORBIDDEN_EXAMS))


def digest_sha256(digest):
    """The sha256 build_digest records: every key but created_utc (and sha256 itself)."""
    return _sha({k: v for k, v in digest.items() if k not in ("created_utc", "sha256")})


def check_staged(digest):
    """The prompt of a staged digest, or ValueError when it is not exactly what
    build_digest staged.

    The recorded sha256 is recomputed over the digest's content, the prompt
    must be render_prompt(sections) and match its prompt_sha256, every
    evidence section passes assert_dev_only, the literature carries no
    exam-keyed object, and the prompt text names no non-dev exam as a key. A
    digest edited after staging is refused however consistently its hashes
    were recomputed.
    """
    if digest.get("schema") not in STAGED_SCHEMAS:
        raise ValueError("not an INC plan digest (schema %r; this copy runs %s)"
                         % (digest.get("schema"), ", ".join(STAGED_SCHEMAS)))
    sections = digest.get("sections")
    if not isinstance(sections, dict) or not sections:
        raise ValueError("the staged digest has no sections")
    if digest_sha256(digest) != digest.get("sha256"):
        raise ValueError("the staged digest does not match its sha256: it was edited after staging")
    prompt = digest.get("prompt")
    if not isinstance(prompt, str) or prompt != render_prompt(sections):
        raise ValueError("the staged prompt is not the rendering of the digest's sections")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != digest.get("prompt_sha256"):
        raise ValueError("the staged prompt does not match its prompt_sha256")
    for key, sec in sections.items():
        if key == "literature":
            if dev_leaks(sec, strings=False):
                raise ValueError("staged digest section 'literature' carries non-dev exam data "
                                 "(an exam-keyed object)")
        else:
            assert_dev_only(sec, "staged digest section %r" % key)
    m = _PROMPT_KEY_LEAK_RE.search(prompt)
    if m:
        raise ValueError("the staged prompt carries non-dev exam data (%s)" % m.group(0))
    return prompt


def run(input_path, output_path, endpoint="", model="", num_ctx=0, timeout_s=3600,
        client=None):
    """The cluster job body: one completion for one staged digest.

    `num_ctx` 0 (the default) asks for the context the digest was sized for;
    one over MAX_NUM_CTX is refused (the job's GPU is sized for that). Always writes `output_path` (an `ok: False` reply names its reason), so the
    lab sees a failure at once rather than waiting out PLAN_TIMEOUT_S.
    """
    started = time.time()
    out = {"schema": REPLY_SCHEMA, "ok": False, "reason": "", "model": model,
           "proposed_by": actor_for(model), "endpoint": endpoint,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "place": "cluster" if os.environ.get("SLURM_JOB_ID") else "local",
           "started_utc": M.utc_now(), "plan": None, "parse_problems": []}
    try:
        digest = _read_json(input_path)
        if not isinstance(digest, dict):
            raise ValueError("the staged digest is not a JSON object")
        out.update({"campaign": digest.get("campaign"), "n": digest.get("n"),
                    "exp": digest.get("exp"), "digest_sha256": digest.get("sha256")})
        prompt = check_staged(digest)
        num_ctx = int(num_ctx or digest.get("num_ctx") or DEFAULT_NUM_CTX)
        out["num_ctx"] = num_ctx
        if num_ctx > MAX_NUM_CTX:
            raise ValueError("num_ctx %d is over MAX_NUM_CTX %d, the most this job's GPU is "
                             "sized for" % (num_ctx, MAX_NUM_CTX))
        if client is None:
            from ..brain import supervisor
            client = supervisor.OpenAICompatClient(endpoint=endpoint, model=model,
                                                   timeout_s=timeout_s, api="ollama")
        res = client(prompt, model, int(num_ctx)) or {}
        text = str(res.get("text") or "")
        out.update({"tokens_in": res.get("tokens_in"), "tokens_out": res.get("tokens_out"),
                    "latency_s": res.get("latency_s"), "truncated": bool(res.get("truncated")),
                    "model_used": res.get("model_used"), "raw_text": text[:MAX_RAW_TEXT]})
        if res.get("error"):
            out["reason"] = str(res["error"])[:1000]
        else:
            plan, problems = parse_reply(text)
            out["plan"], out["parse_problems"] = plan, problems
            out["ok"] = plan is not None
            if plan is None:
                out["reason"] = "; ".join(problems)
    except Exception as exc:
        out["reason"] = "%s: %s" % (type(exc).__name__, exc)
    out["elapsed_s"] = round(time.time() - started, 2)
    out["finished_utc"] = M.utc_now()
    _atomic_write(output_path, out)
    return out


# --- lab side: where plans live, when to give up, how they merge -------------------

def plan_paths(campaign, n):
    """Cluster input/output (CLUSTER_CAMPAIGN_DIR/plans/<campaign>/) and the lab copy."""
    base = "%s/plans/%s" % (M.CLUSTER_CAMPAIGN_DIR, campaign)
    return {"input": "%s/%d.input.json" % (base, int(n)),
            "output": "%s/%d.json" % (base, int(n)),
            "log_dir": "%s/plans/logs" % M.CLUSTER_CAMPAIGN_DIR,
            "local_input": str(M.PLAN_DIR / str(campaign) / ("%d.input.json" % int(n))),
            "local": str(M.PLAN_DIR / str(campaign) / ("%d.json" % int(n)))}


def submit_argv(paths, model=None):
    """The sbatch command that runs the brain on a staged digest (run from CLUSTER_REPO)."""
    export = "ALL,PLAN_INPUT=%s,PLAN_OUTPUT=%s" % (paths["input"], paths["output"])
    if model:
        export += ",PLAN_MODEL=%s" % model
    return ["sbatch", "--export=%s" % export, "weed_llm_benchmark/run_inc_plan.sh"]


def _utc_seconds(stamp):
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def _stamp_seconds(stamp):
    try:
        return _utc_seconds(stamp)
    except (TypeError, ValueError):
        return None


def collect(reply, submitted_utc, now_utc=None, timeout_s=PLAN_TIMEOUT_S, digest_sha256=None,
            pulled_utc=None):
    """{"status": "ready" | "late" | "failed" | "pending" | "timeout", "reply", "reason"}.

    `reply` is the pulled <n>.json (a path, a dict, or None when nothing came
    back yet); `digest_sha256` is the sha256 of the digest the lab staged for
    this submission, and `pulled_utc` when the lab pulled the reply (default
    `now_utc`). A reply is:
      * `failed` when it is not a plan reply, when no staged sha256 is given to
        match it against, when its digest_sha256 is not the staged digest's (a
        stale reply, or one for another campaign or n), or when it carries no
        plan;
      * `late` when it landed (pulled, or finished on the cluster, whichever is
        later) more than `timeout_s` after submission: the deterministic
        proposals went ahead alone by then (contract (a), "merged if back in
        time");
      * `ready` otherwise.
    With no reply: `timeout` after `timeout_s`, else `pending`. merge() merges
    only `ready`.
    """
    rep = reply
    if isinstance(reply, (str, Path)):
        rep = _read_json(reply) if Path(reply).is_file() else None
    submitted = _utc_seconds(submitted_utc)
    now = _utc_seconds(now_utc or M.utc_now())
    if isinstance(rep, dict):
        if rep.get("schema") != REPLY_SCHEMA:
            return {"status": "failed", "reply": rep, "reason": "not a plan reply"}
        if not digest_sha256:
            return {"status": "failed", "reply": rep,
                    "reason": "no staged digest sha256 to match the reply against; a reply that "
                              "cannot be matched to its digest is never merged"}
        if rep.get("digest_sha256") != digest_sha256:
            return {"status": "failed", "reply": rep,
                    "reason": "the reply answers digest %s, not the staged digest %s"
                              % (str(rep.get("digest_sha256"))[:12], str(digest_sha256)[:12])}
        if not (rep.get("ok") and isinstance(rep.get("plan"), dict)):
            return {"status": "failed", "reply": rep, "reason": rep.get("reason") or "no plan"}
        landed = _stamp_seconds(pulled_utc) if pulled_utc else now
        finished = _stamp_seconds(rep.get("finished_utc"))
        landed = max(x for x in (landed, finished) if x is not None)
        if landed - submitted > timeout_s:
            return {"status": "late", "reply": rep,
                    "reason": "the plan landed %d s after submission, past the %d s timeout; the "
                              "deterministic proposals went ahead alone" % (landed - submitted,
                                                                           timeout_s)}
        return {"status": "ready", "reply": rep, "reason": ""}
    if now - submitted > timeout_s:
        return {"status": "timeout", "reply": None,
                "reason": "no plan within %d s of submission; the deterministic "
                          "proposal goes ahead alone" % timeout_s}
    return {"status": "pending", "reply": None, "reason": ""}


def proposal_key(p):
    """What makes two proposals the same request: the policy action and the
    exact command, with a new build's own experiment name set aside (the same
    build under two names is one build)."""
    child = (p or {}).get("child_exp")
    argv = tuple("<child>" if child and str(a) == str(child) else str(a)
                 for a in ((p or {}).get("argv") or []))
    return ((p or {}).get("policy_action"), argv)


_NOTE_FIELDS = ("id", "proposed_by", "rank", "rationale", "predicted", "falsifier", "trigger",
                "basis", "cites", "lit", "params", "child_exp")


def merge(deterministic, collected, validated=None):
    """The proposals to file this tick.

    The deterministic proposals are always there, in order, with their ids,
    commands and params unchanged. Brain proposals and R4 cards are added only
    for a `ready` plan whose validation record is of that same reply (the same
    digest_sha256); anything else is recorded, not merged. A brain proposal
    that renders the same request as one already in the list (proposal_key)
    is not filed a second time: its rank, rationale and prediction are
    attached to the earlier proposal as a `brain_notes` entry (on a copy).
    """
    status = (collected or {}).get("status", "pending")
    out = {"proposals": list(deterministic or []), "cards": [],
           "brain": {"status": status, "reason": (collected or {}).get("reason", "")}}
    if status != "ready" or not isinstance(validated, dict):
        return out
    want = ((collected or {}).get("reply") or {}).get("digest_sha256")
    if not want or validated.get("digest_sha256") != want:
        out["brain"].update(status="failed",
                            reason="the validation record is of another reply (digest %s, the "
                                   "reply's %s)" % (str(validated.get("digest_sha256"))[:12],
                                                    str(want)[:12]))
        return out
    index = {}
    for i, p in enumerate(out["proposals"]):
        index.setdefault(proposal_key(p), i)
    added, noted = 0, 0
    for p in validated.get("menu") or []:
        k = proposal_key(p)
        if k in index:
            i = index[k]
            base = out["proposals"][i]
            note = {f: copy.deepcopy(p.get(f)) for f in _NOTE_FIELDS if f in p}
            out["proposals"][i] = dict(base, brain_notes=list(base.get("brain_notes") or []) + [note])
            noted += 1
            continue
        index[k] = len(out["proposals"])
        out["proposals"].append(p)
        added += 1
    out["cards"].extend(validated.get("cards") or [])
    out["brain"].update({"valid": len(validated.get("menu") or []), "added": added,
                         "noted_on_existing": noted,
                         "cards": len(validated.get("cards") or []),
                         "dropped": len(validated.get("dropped") or [])})
    return out


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="inc_autopilot.brain_plan",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("digest", help="build a dev-only digest from a local INC tree")
    d.add_argument("--inc-dir", required=True)
    d.add_argument("--exp", required=True)
    d.add_argument("--parent", action="append", default=[])
    d.add_argument("--diagnoses", default=None, help="JSON list of Diagnosis records")
    d.add_argument("--menu", default=None, help="levers.json (default: beside this module)")
    d.add_argument("--campaign", default="default")
    d.add_argument("--n", type=int, default=0)
    d.add_argument("--num-ctx", type=int, default=0,
                   help="0 = chosen from the digest's size (at most MAX_NUM_CTX); a number "
                        "fixes it and the digest is trimmed to fit")
    d.add_argument("--out", required=True)
    r = sub.add_parser("run", help="cluster job body: one completion for a staged digest")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--endpoint", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--num-ctx", type=int, default=0,
                   help="0 = the num_ctx the digest was sized for")
    r.add_argument("--timeout", type=float, default=3600)
    p = sub.add_parser("parse", help="parse a saved reply text")
    p.add_argument("reply")
    args = ap.parse_args(argv)
    if args.cmd == "digest":
        from .corpus import Corpus
        arts = load_artifacts(args.inc_dir, [args.exp] + list(args.parent))
        diags = _read_json(args.diagnoses) if args.diagnoses else []
        try:
            corpus = Corpus()
        except FileNotFoundError:
            corpus = None
        dg = build_digest(arts, args.exp, diags, load_menu(args.menu), campaign=args.campaign,
                          n=args.n, corpus=corpus, parents=args.parent,
                          num_ctx=args.num_ctx or None)
        _atomic_write(args.out, dg)
        tr = dg["trimmed"]
        print("digest %s: %d est. tokens (%d before trimming), num_ctx %d, %d cuts%s -> %s"
              % (dg["sha256"][:12], dg["tokens_estimated"], tr["tokens_before"], dg["num_ctx"],
                 len(tr["cuts"]), "".join("\n  cut %s: %s" % (c["stage"], c["cut"])
                                          for c in tr["cuts"]), args.out))
        return 0
    if args.cmd == "run":
        rep = run(args.input, args.output, args.endpoint, args.model, args.num_ctx, args.timeout)
        plan = rep.get("plan") or {}
        print("[plan] ok=%s model=%s ranked=%d off_menu=%d tokens_in=%s %.1fs reason=%s"
              % (rep["ok"], args.model, len(plan.get("ranked_menu") or []),
                 len(plan.get("off_menu") or []), rep.get("tokens_in"), rep.get("elapsed_s", 0),
                 rep.get("reason") or "-"))
        return 0 if rep["ok"] else 1
    if args.cmd == "parse":
        with open(args.reply, "r", encoding="utf-8") as fh:
            plan, problems = parse_reply(fh.read())
        print(json.dumps({"plan": plan, "problems": problems}, indent=1, sort_keys=True))
        return 0 if plan is not None else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(_main())
