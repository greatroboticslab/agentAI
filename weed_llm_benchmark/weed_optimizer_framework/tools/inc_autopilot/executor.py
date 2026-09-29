"""The one path that executes an INC autopilot action (docs/INC_AUTOPILOT.md, component 7).

The ticker, approved items and the page's manual buttons all come through
here. Nothing else in the autopilot submits, unblocks or writes on the cluster.

What happens to a request
-------------------------
1. It is resolved against `brain/policy_actions.json`. An unknown action, an
   R4 row or an R4 request (an off-menu X lever) is refused: R4 is a human card
   and is never executed, whoever asks.
2. Its parameters are resolved. When the request carries an `argv` (a lever
   proposal's exact builder command, as on its card), the policy parameters are
   read out of that argv by the builder's grammar (`params_from_argv`); a value
   the request also states must agree with it, and a request parameter the
   argv does not carry is kept as metadata, never rendered. Without an argv the
   request's parameters are the policy parameters as given. `est_gpu_hours` is
   a price: it is authorised only for an action priced by it (the builds).
3. The command is rendered from those parameters by fixed code per action
   (`render`); with an argv, the rendering must reproduce it token for token.
   So what runs is exactly what `policy.authorize` checked.
4. `policy.authorize` runs at execution time with a real `budget_state` (the
   campaign envelope, `budget.py`) and real `resources` (`mongo_ok`, cluster
   reachability). Nothing is authorised from a stale answer. An action that
   costs GPU time is refused when the resources do not state Mongo's health
   (`mongo_ok` / `mongo_down`): unknown is not healthy. An unknown cluster
   reachability is left to the ssh call, whose failure means nothing ran.
5. By actor and risk:
   * R0-R2 for `round-scheduler:inc-autopilot`: direct, when authorize says so.
     A non-human unblock (L7) must name a cause on thresholds.json's D5
     `transient_cause_kinds` and, when the ticker gave its fired diagnoses,
     be one D5 lists as transient for that unit.
   * Anything authorize sends to approval (a tier2 brain proposal, a tier0
     R1): filed with `approvals.propose`.
   * R3 from the autopilot: its ceiling has no R3 cell, so the item is filed
     with `approvals.propose` and waits for a person; the approval then runs
     authorised as that person (`human:<decided_by>`).
   * Except under the owner's 2026-09-27 envelope rule (section d): when the
     campaign config says `autonomy: "envelope"`, the replay tests have passed
     on the current code (`replay_status`, every required case), the lever is
     L1, L2, L5, L8 or L9 and its parameters are that lever's (levers.json bounds,
     fixed and required values), the proposal is the autopilot's own (an
     item a brain or anyone else filed waits for a person), every trigger is
     a diagnosis the ticker reports as fired that names this lever (or
     supports it, levers.json `supports`) and whose cites the proposal
     carries, no stop-loss diagnosis is firing, and its estimate fits the
     envelope and the daily cap, the executor records a grant by the
     autopilot (lever, trigger diagnoses, cites, envelope balance) and runs
     the item at once, authorised as the person who enabled the envelope
     (`campaign.autonomy_granted_by`). Every other rule still applies.
   * A budget escalation (authorize's own `needs_approval` for an actor with
     direct authority) is a refusal here: the envelope is a hard cap, and the
     way past it is a person raising the envelope, not an approval the same
     check would stop again at execution.
6. An approved item runs at most once: `approvals.record_executed(started)` is
   appended under a lock before anything runs, and a second call for the same
   approval id is refused. The one exception is a call that certainly never
   reached the cluster (`never_ran`): the claim is `released` and the approval
   stays executable. A direct request runs at most once per proposal id: a
   proposal that ran, or may have run, is refused the second time.
7. Before anything runs, a `started` record (charged, with a `run_id`) is
   appended to the execution log under a lock; if it cannot be written,
   nothing runs. The outcome record follows with the same `run_id`. A run
   whose outcome was never written stays charged (`budget.fold`).

Remote verbs go through the injected `slurm_sh(shell, timeout)` hook (the
round scheduler's, `dashboard_server.py:17720`), which returns
`{ok, stdout, stderr, returncode}`. Each verb line is wrapped in segment
markers so several share one ssh call; `remote.py` prints one `INCAP <json>`
line per verb (model.REMOTE_MARK) and exits 0 only when that record is ok. A
verb certainly did not run only when the call never reached the cluster: the
remote preamble failed, the hook refused to send it (`NotSent`), or ssh
failed to connect (no output at all, and ssh's own connect error). Any other
verb with no INCAP line may have run -- one that started and printed none,
and one whose call timed out or dropped with no output (the dashboard's
slurm_sh returns an empty stdout on a timeout, so segment markers the cluster
already printed are lost) -- and its estimate is charged to the envelope as
an outcome that is unknown. An approved item whose call certainly never
reached the cluster is released (approvals `released`), so the approval is
not used up by a connection that failed. Builds,
relevance and audit go through `remote.py submit BUILDER --parent-exp
--trigger --approval-id --decided-by -- ARGS`, so the provenance record on
/ocean names the approval and who decided it.

The research brain's plan job (docs/INC_AUTOPILOT.md (c)) has no remote.py
verb: inc_plan_submit (R2, one H100 for up to 2 h) runs a segment built here
(_plan_segment) that ships the lab's staged digest, gzip+base64, only when it
hashes to the authorised `digest_sha256`, writes it once under
INC_DIR/_campaign/plans/ and sbatches run_inc_plan.sh; inc_plan_pull (R0)
reads the reply back, batched into the campaign ticker's snapshot call
(campaign_snapshot's `plan_pull`). campaign_snapshot's `ledger_from` takes
N or (N, through_sha256), so a rewritten ledger prefix is re-read from line 0
on the cluster rather than spliced.

Every call leaves one record in `executions.jsonl` next to the campaign
ledger (two lines for a call that ran: `started` and its outcome, folded into
one by `executions()`): what was asked, by whom, as whom it was authorised,
and what happened. The budget fold, the per-lever submission count, the
once-per-proposal rule and the one-unblock-per-unit rule read that file.

Campaign config read here (the ticker owns it):
    {"name": "<campaign>", "autonomy": "off" | "envelope",
     "autonomy_granted_by": "human:<email>", "envelope_su": 300,
     "daily_cap_su": 120, "paused_reason": null}

The ticker's fired diagnoses reach the envelope rule through the Context's
`diagnoses` hook (a list of model.diagnosis records, or a callable taking the
campaign name). With no hook, the envelope rule cannot check a trigger and
every R3 build waits for a person.

The replay gate: `record_replay_result` stores the per-case outcome of
tests/test_inc_ap_replay.py and tests/test_inc_ap_governance.py against a hash
of this package and the governance files it depends on (`GOVERNANCE_FILES`).
`python -m weed_optimizer_framework.tools.inc_autopilot.executor record-replay`
runs both scripts and records the result.
"""
from __future__ import annotations

import base64
import datetime
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import budget as B
from . import levers as LV
from . import model as M
from ..brain import approvals as AP
from ..brain import policy as POL

# R3 levers the envelope rule covers, and the policy action each may use
# (docs/INC_AUTOPILOT.md section d: L1, L2, L5, L8, and L9, the pre-registered
# v2 pilot; L6, --no-truth, is not one).
ENVELOPE_LEVERS = {"L1": ("inc_build_pilot",), "L2": ("inc_build_realloop",),
                   "L5": ("inc_build_realloop",), "L8": ("inc_build_baseline",),
                   "L9": ("inc_build_pilot",)}
# Stream mode (docs/CONTINUOUS_LOOP.md 6.5): the R3 levers the envelope may
# approve for a stream campaign, and their actions. L21 only when its target is
# the pool the last 'hurts' milestone recommended (the campaign's
# last_milestone_pool, read from the stream summary's rollback_pending); LI
# (the stream's creation) with Stage A's recorded arms; L23's splits build and
# lock always wait for a person. Their parameters are checked against
# stream_levers.json (levers_stream.check_params) and their counts against its
# limits, which replace the blanket 3-per-campaign rule for these levers.
STREAM_ENVELOPE_LEVERS = {"L18": ("inc_build_segment",), "L20": ("inc_build_consolidation",),
                          "L21": ("inc_stream_rollback",), "L22": ("inc_build_segment",),
                          "L23B": ("inc_build_baseline_v2",), "L25": ("inc_build_pilot4",),
                          "L27": ("inc_build_consolidation",), "L28": ("inc_build_segment",),
                          "LI": ("inc_stream_init",)}
# The R2 data levers that run directly only with data_autonomy on, a stream
# replay pass, the floors set (L16) and the caps (6.5, 6.6); otherwise they are
# filed for a person. Keyed by policy action (a sub-lever shares its family's).
GATED_R2_ACTIONS = {"inc_stream_collect": "L16", "inc_stream_collect_lab": "L16", "inc_stream_intake": "L16",
                    "inc_stream_admit": "L17", "inc_stream_quarantine": "L24"}
# Stream actions of a limited family that are neither a job nor a download.
STREAM_NON_JOB_ACTIONS = ("inc_stream_sync",)
# Stop-loss: more than 3 submissions of the same lever per campaign pauses it.
# The executor will not self-approve a 4th.
MAX_LEVER_SUBMISSIONS = 3
RECIPE_ORDER = ("full", "freeze", "lora")          # inc.pilot.inc_recipes() table order
REMOTE_MODULE = "weed_optimizer_framework.tools.inc_autopilot.remote"
# A submit is a login (up to about 50 s on Bridges-2), the bench env, squeue and
# sbatch: a timeout there loses whatever the cluster printed, and the outcome
# becomes unknown, so the submits get room.
REMOTE_TIMEOUT_S = {"inc_snapshot": 300, "inc_report": 300, "inc_advance": 240,
                    "inc_plan_submit": 180, "inc_plan_pull": 120,
                    "inc_campaign_snapshot": 600, "inc_cancel_exp": 180, "inc_sync_outer": 300,
                    "inc_build_pilot": 300, "inc_build_realloop": 300, "inc_build_baseline": 300,
                    "inc_relevance_build": 300, "inc_label_audit": 300,
                    "inc_unblock_transient": 180, "inc_funnel_audit": 300, "inc_funnel_map": 300,
                    "inc_funnel_recover": 300, "inc_funnel_dev_scores": 300,
                    "inc_stream_snapshot": 600, "inc_stream_collect": 300, "inc_stream_collect_review": 300,
                    "inc_stream_intake": 300, "inc_stream_probe": 300, "inc_stream_admit": 300,
                    "inc_build_segment": 300, "inc_build_consolidation": 300, "inc_splits_build": 300,
                    "inc_build_baseline_v2": 300, "inc_build_pilot4": 300, "inc_stream_commit": 600,
                    "inc_stream_rollback": 600, "inc_stream_quarantine": 300, "inc_stream_release": 300,
                    "inc_stream_init": 300, "inc_stream_choose_arm": 600, "inc_stream_compare": 600,
                    "inc_stream_verdict": 600}
DEFAULT_REMOTE_TIMEOUT_S = 120
# ssh's own messages for a connection that was never made (read off the last
# stderr line of a call that printed nothing): the remote command never ran.
# A connection that drops after it was made ("Connection to X closed by
# remote host", "Broken pipe", a plain "Connection reset by peer") is not one.
_NEVER_CONNECTED_RE = re.compile(
    r"(ssh: connect to host |Could not resolve hostname|kex_exchange_identification|"
    r"Connection closed by \S+ port \d+|Permission denied \(|Host key verification failed|"
    r"No route to host|Network is unreachable|Connection timed out during banner exchange|"
    r"Too many authentication failures|No such file or directory: 'ssh')")
MAX_BATCH_TIMEOUT_S = 900
CONDA_SH = "/jet/home/byler/miniconda3/etc/profile.d/conda.sh"   # as run_inc_job.sh
AUTO_REASON = "auto: "                                             # remote.AUTO_PREFIX + space
# Diagnoses that are stop-losses (section d: any D7, D10 or D14). While one of
# them, or any diagnosis asking to pause or halt, is firing, nothing is
# self-approved.
STOP_DIAGNOSES = ("D7", "D10", "D14")
STOP_OPERATIONS = ("OP_PAUSE", "OP_HALT")
# The replay cases (contract section f) a pass must report. Required cases
# must pass; R2 and R4b wait on cluster fixtures and may be recorded as
# skipped, but only explicitly. R5 is pilot_v2's case (D15 -> L9); R6 is
# pilot_v3's (D1 fires but does not block D4 -> D4 ready -> L2 on gate net);
# R7 is pilot_v3 with a Step 1 whose relevance.json failed its own
# calibration check (D2 does not escalate -> L2 with --increment-sources
# evidence); R8 is pilot_v3 with the real Step 1 summaries, whose evidenced
# pool cannot hold the default loop (D2 does not escalate: the R4 sizing
# rule -> L2 with --increment-sources evidence --size 287 --n-verified 4).
# R9-R14 and the funnel's negative controls, mutation harness and domain-free
# test (docs/FUNNEL_AUDIT.md 8.9; runner 5.5.9): envelope autonomy needs them too.
FUNNEL_REPLAY_CASES = ("R9", "R9_early", "R9b", "R10", "R11", "R12", "R13", "R14", "funnel_negative_controls")
REPLAY_REQUIRED = ("R1", "R3", "R4a", "R5", "R6", "R7", "R8", "negative_controls", "test_blindness",
                   "earliest_fire", "governance") + FUNNEL_REPLAY_CASES + ("funnel_mutations", "domain_free")
REPLAY_MAY_SKIP = ("R2", "R4b")
# The stream-mode scenario cases (docs/CONTINUOUS_LOOP.md 6.8) and the mutation
# harness over D20-D33. run_replay_tests runs their scripts with the others
# (REPLAY_SCRIPTS) and records each case; any case recorded as a failure fails
# the whole record (check_replay_cases), so a failing S-case blocks envelope
# autonomy for every campaign, the funnel's and weed_inc_v1's included. A stream
# campaign's own autonomy (its envelope builds and its gated R2 data levers)
# needs every one of them to pass (stream_replay_status).
STREAM_REPLAY_CASES = ("S1", "S1b", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13",
                       "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "S24", "S25", "S26",
                       "S27", "S28", "stream_prospective", "stream_r0")
STREAM_MUTATION_CASE = "stream_mutations"
CODE_ROOT = Path(__file__).resolve().parents[3]       # the directory holding weed_optimizer_framework/
# Files outside this package that decide what the executor may do; a replay
# pass recorded before any of them changed does not count.
GOVERNANCE_FILES = ("weed_optimizer_framework/tools/brain/policy_actions.json",
                    "weed_optimizer_framework/tools/brain/policy.py",
                    "weed_optimizer_framework/tools/brain/approvals.py",
                    "weed_optimizer_framework/tools/brain/su_ledger.py",
                    "tests/test_inc_ap_replay.py",
                    "tests/test_inc_ap_governance.py",
                    "tests/test_funnel_ap_replay.py",
                    "tests/test_funnel_ap_mutations.py",
                    "tests/test_funnel_domain_free.py",
                    "weed_optimizer_framework/tools/funnel/domains/weed.json",
                    # stream mode (docs/CONTINUOUS_LOOP.md 6.5 review): the collector's
                    # policy file decides what L16 may do by itself; the stream's
                    # scenario and mutation scripts decide what a pass means.
                    "weed_optimizer_framework/tools/collect/domains/weed.json",
                    "tests/test_stream_ap_replay.py",
                    "tests/test_stream_ap_mutations.py")
REPLAY_SCRIPTS = {"replay": "tests/test_inc_ap_replay.py",
                  "governance": "tests/test_inc_ap_governance.py",
                  "funnel": "tests/test_funnel_ap_replay.py",
                  "funnel_mutations": "tests/test_funnel_ap_mutations.py",
                  "domain_free": "tests/test_funnel_domain_free.py",
                  "stream": "tests/test_stream_ap_replay.py",
                  "stream_mutations": "tests/test_stream_ap_mutations.py"}
# How test_inc_ap_replay.py names a skipped check, mapped to the case it
# belongs to. A skip under any other name makes the recorded result a fail.
REPLAY_SKIP_CASES = (("R2", "R2"), ("R4b", "R4b"), ("remote snapshot", "live_path"))

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TRIGGER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")        # remote.TRIGGER_RE
_APPROVAL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")                # remote.APPROVAL_RE
_PROV_ACTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._:+-]{0,127}$")   # remote.ACTOR_RE
_INT_RE = re.compile(r"^(0|[1-9][0-9]{0,8})$")
_SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
_LOG_PAYLOAD_CHARS = 4000
PKG_DIR = Path(__file__).resolve().parent
THRESHOLDS_FILE = PKG_DIR / "thresholds.json"

# The builder grammar of every action that has one, in rendering order:
# (first token, command or None, ((flag, param, kind), ...), required params).
# Kinds: str, int, flag (param = 1), list (comma-joined NAME=PATH values),
# auto (the unblock reason "auto: <cause>", param = cause).
ARGV_FORMS = {
    "inc_build_pilot": ("inc.pilot", "build",
                        (("--exp", "exp", "str"), ("--replay-mode", "replay_mode", "str"),
                         ("--gate-flips-mode", "gate_flips_mode", "str")),
                        ("exp", "replay_mode")),
    "inc_build_baseline": ("inc.pilot", "build-baseline",
                           (("--exp", "exp", "str"), ("--manifest", "manifest", "str"), ("--seeds", "seeds", "str")),
                           ("exp", "manifest")),
    "inc_build_realloop": ("inc.realloop", "build",
                           (("--exp", "exp", "str"), ("--base", "base", "str"),
                            ("--replay-mode", "replay_mode", "str"), ("--recipes", "recipes", "str"),
                            ("--increment-sources", "increment_sources", "str"),
                            ("--step1-overlay", "step1_overlay", "str"),
                            ("--relevance", "relevance", "str"), ("--size", "size", "int"),
                            ("--n-verified", "n_verified", "int"), ("--no-truth", "no_truth", "flag"),
                            ("--gate-flips-mode", "gate_flips_mode", "str")),
                           ("exp", "replay_mode", "recipes")),
    "inc_relevance_build": ("run_inc_relevance.sh", "build",
                            (("--sample", "sample", "int"), ("--seed", "seed", "int")), ()),
    "inc_label_audit": ("run_inc_audit.sh", None,
                        (("--trusted", "trusted", "str"), ("--audit", "audit", "list"),
                         ("--out", "out", "str")),
                        ("trusted", "audit", "out")),
    "inc_unblock_transient": ("inc.driver", "unblock",
                              (("--exp", "exp", "str"), ("--unit", "unit", "str"),
                               ("--reason", "cause", "auto")),
                              ("exp", "unit", "cause")),
    # The funnel audit (docs/FUNNEL_AUDIT.md 8.5): run_inc_funnel.sh VERB; the
    # command "{verb}" is the positional verb param. The verb's extra sbatch
    # flags stand before the script in a lever's argv; they are the cluster's
    # to add (remote.py submit funnel, funnel/__main__.py SBATCH_RESOURCES).
    "inc_funnel_audit": ("run_inc_funnel.sh", "{verb}",
                         (("--prereg", "prereg", "str"), ("--out", "out", "str"), ("--rl", "rl", "flag")),
                         ("verb", "prereg", "out")),
    "inc_funnel_map": ("run_inc_funnel.sh", "map",
                       (("--prereg", "prereg", "str"), ("--out", "out", "str"), ("--part", "part", "str")),
                       ("prereg", "out", "part")),
    "inc_funnel_recover": ("run_inc_funnel.sh", "recover",
                           (("--prereg", "prereg", "str"), ("--audit", "audit", "str"), ("--maps", "maps", "str"),
                            ("--policy", "policy", "str"), ("--out", "out", "str")),
                           ("prereg", "audit", "maps", "policy", "out")),
    # L11a and L12 run the funnel CLI's fetch on the lab (a lab hook runs it).
    "inc_funnel_fetch": ("funnel", "fetch",
                         (("--prereg", "prereg", "str"), ("--what", "what", "str"),
                          ("--names-from", "names_from", "str"), ("--out", "out", "str")),
                         ("prereg", "what", "out")),
    # Stream mode (docs/CONTINUOUS_LOOP.md 6.3; stream_levers.json). '{pkg}' in the
    # first token is the campaign's protocol package, the policy parameter 'pkg'.
    # One flag order serves every verb of a shared action (L18 build, L22 fork,
    # L28 feasibility), since a verb renders only the flags it is given. The
    # grammar is inc2.stream.build_parser()'s (group E).
    "inc_build_segment": ("{pkg}.stream", "{verb}",
                          (("--stream", "stream", "str"), ("--k", "k", "int"), ("--exp", "exp", "str"),
                           ("--holdout", "holdout", "str"), ("--m", "m", "int"), ("--recipes", "recipes", "str")),
                          ("pkg", "verb", "stream")),
    "inc_stream_init": ("{pkg}.stream", "init", (("--stream", "stream", "str"), ("--stage-b", "stage_b", "str")),
                        ("pkg", "stream", "stage_b")),
    "inc_stream_choose_arm": ("{pkg}.stream", "choose-arm", (("--stream", "stream", "str"),), ("pkg", "stream")),
    "inc_stream_compare": ("{pkg}.stream", "compare", (("--exp", "exp", "str"),), ("pkg", "exp")),
    "inc_stream_verdict": ("{pkg}.{module}", "{verb}", (("--exp", "exp", "str"),), ("pkg", "module", "verb")),
    "inc_build_consolidation": ("{pkg}.stream", "{verb}",
                                (("--stream", "stream", "str"), ("--from", "from_pool", "str")),
                                ("pkg", "verb", "stream")),
    "inc_stream_commit": ("{pkg}.stream", "commit", (("--exp", "exp", "str"),), ("pkg", "exp")),
    "inc_stream_rollback": ("{pkg}.stream", "rollback", (("--stream", "stream", "str"), ("--to", "to", "str")),
                            ("pkg", "stream", "to")),
    "inc_stream_quarantine": ("{pkg}.stream", "quarantine",
                              (("--source", "source", "str"), ("--cite", "cite", "str")), ("pkg", "source", "cite")),
    "inc_stream_release": ("{pkg}.stream", "release", (("--stream", "stream", "str"), ("--hold", "hold", "str")),
                           ("pkg", "stream", "hold")),
    "inc_splits_build": ("{pkg}.splits", "{verb}", (), ("pkg", "verb")),
    "inc_build_baseline_v2": ("{pkg}.baseline", "build",
                              (("--exp", "exp", "str"), ("--manifest", "manifest", "str"), ("--union", "union", "str"),
                               ("--seeds", "seeds", "str"), ("--arm", "arm", "str"), ("--role", "role", "str")),
                              ("pkg", "exp", "seeds", "arm", "role")),
    "inc_build_pilot4": ("{pkg}.pilot4", "build",
                         (("--exp", "exp", "str"), ("--from", "from_exp", "str"), ("--recipes", "recipes", "str")),
                         ("pkg", "exp", "from_exp", "recipes")),
    "inc_stream_collect": ("run_inc_collect.sh", "fetch",
                           (("--source", "source", "str"), ("--max-bytes", "max_bytes", "bigint"),
                            ("--candidates", "candidates", "str")),
                           ("source", "max_bytes")),
    "inc_stream_collect_review": ("run_inc_collect.sh", "fetch",
                                  (("--source", "source", "str"), ("--max-bytes", "max_bytes", "bigint"),
                                   ("--candidates", "candidates", "str")),
                                  ("source", "max_bytes")),
    "inc_stream_intake": ("run_inc_collect.sh", "intake", (("--source", "source", "str"),), ("source",)),
    "inc_stream_probe": ("run_inc_collect.sh", "probe", (), ()),
    "inc_stream_admit": ("run_inc2_stream.sh", "{verb}", (("--intake", "intake", "str"), ("--hold", "hold", "str")),
                         ("verb",)),
    "inc_stream_discover": ("collect", "plan",
                            (("--config", "config", "str"), ("--classes", "classes", "str"), ("--out", "out", "str")),
                            ("config", "classes", "out")),
    "inc_stream_names": ("collect", "names", (("--source", "source", "str"), ("--out", "out", "str")),
                         ("source", "out")),
    "inc_stream_collect_lab": ("collect", "fetch",
                               (("--source", "source", "str"), ("--max-bytes", "max_bytes", "bigint"),
                                ("--out", "out", "str")),
                               ("source", "max_bytes", "out")),
}
# Stream actions and how the cluster runs them: a job script through remote.py
# stream-submit KIND, or a login-node verb through stream-run.
STREAM_REMOTE = {"inc_build_segment": "build", "inc_build_consolidation": "build", "inc_splits_build": "build",
                 "inc_build_baseline_v2": "build", "inc_build_pilot4": "build",
                 "inc_stream_collect": "collect", "inc_stream_collect_review": "collect",
                 "inc_stream_intake": "collect", "inc_stream_probe": "collect", "inc_stream_admit": "admit",
                 "inc_stream_commit": "run", "inc_stream_rollback": "run", "inc_stream_quarantine": "run",
                 "inc_stream_release": "run", "inc_stream_init": "build", "inc_stream_choose_arm": "run",
                 "inc_stream_compare": "run", "inc_stream_verdict": "run"}
# The modules whose R0 verdicts a stream campaign records (inc_stream_verdict):
# inc2.baseline canary-verdict / capacity-verdict and inc2.pilot4 verdict.
STREAM_VERDICT_MODULES = ("baseline", "pilot4")
# A stream build's child experiment: <stream>_s|m|c|b<NNN> (contract 3.5, 3.6).
STREAM_CHILD_RE = r"^%s_[smcb][0-9]{3}$"
_BIGINT_RE = re.compile(r"^[1-9][0-9]{0,12}$")
# Lab-side actions: the executor calls the hook the caller registered for each
# (Context.local_hooks) and refuses when none is.
LAB_ACTIONS = ("inc_lit_fetch", "inc_funnel_fetch", "inc_verify_queue", "inc_funnel_sync",
               "inc_stream_discover", "inc_stream_names", "inc_stream_collect_lab", "inc_stream_sync")
# The research brain's plan job (docs/INC_AUTOPILOT.md (c)): staged and
# submitted, then pulled back, by segments the executor builds itself
# (_plan_segment), not by remote.py verbs.
PLAN_ACTIONS = ("inc_plan_submit", "inc_plan_pull")
# The staged digest travels gzip+base64 inside the one ssh command line, which
# is a single argument on both hosts (Linux caps one argument at 128 KiB).
PLAN_MAX_STAGED_CHARS = 96 * 1024
# remote.py submit BUILDER and the module named first in its ARGS.
SUBMIT_FORMS = {"inc_build_pilot": ("build", "pilot"), "inc_build_baseline": ("build", "pilot"),
                "inc_build_realloop": ("build", "realloop"),
                "inc_relevance_build": ("relevance", None), "inc_label_audit": ("audit", None),
                "inc_funnel_audit": ("funnel", None), "inc_funnel_map": ("funnel", None),
                "inc_funnel_recover": ("funnel", None)}


class ExecError(Exception):
    """A request the executor cannot render or resolve; always turned into a refusal."""


class NotSent(RuntimeError):
    """Raised by a slurm_sh hook that refuses to make the call at all (the
    campaign ticker's one-ssh-per-tick budget): nothing reached the cluster."""


# --- context -------------------------------------------------------------------
class Context(object):
    """Where the executor reads and writes, and the hooks it calls.

    `slurm_sh(shell, timeout) -> {ok, stdout, stderr, returncode}` reaches the
    cluster. `resources` is a dict or a callable returning one, with any of
    `mongo_ok`, `mongo_down`, `cluster_reachable`. `local_hooks` maps a lab-side
    action (inc_lit_fetch) to a callable taking its params and returning
    `{"ok": bool, ...}`. `domain_budget` is the domain config's `budget` block
    (db.DEFAULT_DOMAIN_CONFIG's when omitted). `lab_repo` moves every file this
    module touches (tests pass a temporary directory). `diagnoses` is the
    ticker's latest diagnosis records for the campaign (model.diagnosis
    dicts, as diagnose.detect returns them), or a callable taking the
    campaign name and returning them; the envelope rule checks triggers
    against the fired ones and refuses when there is none.
    """

    def __init__(self, slurm_sh=None, resources=None, domain_budget=None, local_hooks=None,
                 clock=None, domain=M.DOMAIN, lab_repo=None, preamble=None, diagnoses=None):
        self.slurm_sh = slurm_sh
        self._resources = resources
        self._diagnoses = diagnoses
        self.domain_budget = domain_budget
        self.local_hooks = dict(local_hooks or {})
        self.clock = clock or time.time
        self.domain = domain
        self.preamble = preamble
        if lab_repo is not None:
            repo = Path(lab_repo)
            brain = repo / "results" / "framework" / "_brain"
            self.su_base_dir = str(brain)
        else:
            repo, brain = M.LAB_REPO, M.BRAIN_DIR
            self.su_base_dir = os.environ.get("BRAIN_SU_LEDGER_DIR") or str(brain)
        self.lab_repo = repo
        self.campaign_dir = brain / domain / "inc"
        self.exec_log = self.campaign_dir / "executions.jsonl"
        self.replay_result = self.campaign_dir / "replay" / "replay_result.json"

    @property
    def approvals_root(self):
        return str(self.lab_repo)

    def resources(self):
        r = self._resources() if callable(self._resources) else self._resources
        r = dict(r) if isinstance(r, dict) else {}
        if "mongo_ok" in r and "mongo_down" not in r and r["mongo_ok"] is not None:
            r["mongo_down"] = not bool(r["mongo_ok"])
        return r

    def fired_diagnoses(self, campaign_name=None):
        """The fired diagnosis records the ticker reported, or None when it gave none."""
        d = self._diagnoses
        if d is None:
            return None
        if callable(d):
            try:
                d = d(campaign_name)
            except Exception:
                return None
        if not isinstance(d, (list, tuple)):
            return None
        return [x for x in d if isinstance(x, dict) and x.get("fired") is True]


# --- replay gate ---------------------------------------------------------------
def code_hash(pkg_dir=None, extra=None):
    """sha256 over the autopilot package's code and data (.py, .json, .txt)
    and the governance files outside it (`GOVERNANCE_FILES`, relative to
    CODE_ROOT; `extra` replaces that list).

    The envelope rule needs "the replay tests passed on this code"; a pass
    recorded against other bytes of any autopilot module, of the policy table
    or gate, of the approval queue, of the SU ledger or of the two test
    scripts does not count. A missing governance file hashes as missing.
    """
    root = Path(pkg_dir) if pkg_dir else PKG_DIR
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix in (".py", ".json", ".txt")
                   and "__pycache__" not in p.parts)
    for p in files:
        h.update(str(p.relative_to(root)).encode("utf-8") + b"\0")
        h.update(p.read_bytes() + b"\0")
    for rel in (GOVERNANCE_FILES if extra is None else extra):
        p = CODE_ROOT / rel
        h.update(b"governance:" + str(rel).encode("utf-8") + b"\0")
        h.update((p.read_bytes() if p.is_file() else b"<missing>") + b"\0")
    return h.hexdigest()


def check_replay_cases(cases):
    """[problems] with a replay result's cases; [] when they make a pass.

    `cases` maps a case id to "pass", "skip" or "fail". Every id in
    REPLAY_REQUIRED must be "pass"; every id in REPLAY_MAY_SKIP must be
    listed, as "pass" or "skip"; no case may be "fail" or anything else.
    """
    if not isinstance(cases, dict):
        return ["the cases are not an object"]
    out = []
    for c in REPLAY_REQUIRED:
        if cases.get(c) != "pass":
            out.append("case %s is %r, not pass" % (c, cases.get(c)))
    for c in REPLAY_MAY_SKIP:
        if cases.get(c) not in ("pass", "skip"):
            out.append("case %s is %r, not pass or an explicit skip" % (c, cases.get(c)))
    for c, v in sorted(cases.items()):
        if c not in REPLAY_REQUIRED and c not in REPLAY_MAY_SKIP and v not in ("pass", "skip"):
            out.append("case %s is %r" % (c, v))
    return out


def record_replay_result(status, cases=None, ctx=None, code=None, detail=None):
    """Write the replay result the envelope rule checks (`run_replay_tests` calls this).

    A "pass" must carry cases that pass `check_replay_cases`; one that does
    not is refused (ValueError) rather than recorded as a pass.
    """
    ctx = ctx or Context()
    if status not in ("pass", "fail"):
        raise ValueError("status must be 'pass' or 'fail'")
    if status == "pass":
        bad = check_replay_cases(cases)
        if bad:
            raise ValueError("a replay pass needs every required case: %s" % "; ".join(bad))
    rec = {"status": status, "code_hash": code or code_hash(), "recorded_utc": M.utc_now(),
           "cases": cases if cases is not None else {}, "writer": "inc_autopilot.executor"}
    if detail is not None:
        rec["detail"] = detail
    p = ctx.replay_result
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(p))
    return rec


def replay_status(ctx=None):
    """{"passed": bool, "reason": str, "recorded": dict|None, "code_hash_now": str}."""
    ctx = ctx or Context()
    now_hash = code_hash()
    out = {"passed": False, "reason": "", "recorded": None, "code_hash_now": now_hash,
           "path": str(ctx.replay_result)}
    try:
        rec = json.loads(ctx.replay_result.read_text(encoding="utf-8"))
    except FileNotFoundError:
        out["reason"] = "no replay result is recorded"
        return out
    except Exception as e:
        out["reason"] = "the replay result is unreadable (%s)" % type(e).__name__
        return out
    out["recorded"] = rec if isinstance(rec, dict) else None
    if not isinstance(rec, dict) or rec.get("status") != "pass":
        out["reason"] = "the recorded replay result is %r, not pass" % (
            rec.get("status") if isinstance(rec, dict) else rec)
        return out
    if rec.get("code_hash") != now_hash:
        out["reason"] = ("the replay pass was recorded for other autopilot code (%s..., now %s...)"
                         % (str(rec.get("code_hash"))[:12], now_hash[:12]))
        return out
    bad = check_replay_cases(rec.get("cases"))
    if bad:
        out["reason"] = "the recorded replay pass is incomplete: " + "; ".join(bad)
        return out
    out["passed"] = True
    out["reason"] = "replay passed on the current code (%s)" % rec.get("recorded_utc")
    return out


def stream_replay_status(ctx=None):
    """replay_status, and every stream case (STREAM_REPLAY_CASES and the
    mutation harness) recorded as a pass: what a stream campaign's envelope
    builds and gated R2 data levers need (docs/CONTINUOUS_LOOP.md 6.5)."""
    st = replay_status(ctx)
    if not st["passed"]:
        return st
    cases = (st.get("recorded") or {}).get("cases") or {}
    missing = [c for c in STREAM_REPLAY_CASES + (STREAM_MUTATION_CASE,) if cases.get(c) != "pass"]
    if missing:
        st = dict(st, passed=False,
                  reason="the recorded replay pass does not pass the stream case(s) %s" % ", ".join(missing[:8]))
    return st


_CASE_RE = re.compile(r"^case (\S+): (pass|fail)$")


def _last_line(text, prefix_re):
    for line in reversed((text or "").splitlines()):
        m = re.match(prefix_re, line.strip())
        if m:
            return m
    return None


def run_replay_tests(ctx=None, python=None, scripts=None, timeout=3600, code_root=None):
    """Run the replay and governance scripts (and the funnel's replay, mutation
    and domain-free scripts, REPLAY_SCRIPTS) and record the result.

    The replay script's cases are read from its exit code and its closing
    line "N failure(s), M skipped: <names>": exit 0 passes every case it
    never skips; a skipped check counts against its case (REPLAY_SKIP_CASES),
    and a skip under a name not listed there fails the result. The governance
    script passes on exit 0 with "ALL PASS". The code hash is taken before
    and after; a change in between records a fail. Returns the record.
    """
    ctx = ctx or Context()
    root = Path(code_root) if code_root else CODE_ROOT
    scripts = dict(REPLAY_SCRIPTS if scripts is None else scripts)
    before = code_hash()
    runs, cases, notes = {}, {}, []
    for key in ("replay", "governance"):
        path = root / scripts[key]
        try:
            p = subprocess.run([python or sys.executable, str(path)], cwd=str(root),
                               capture_output=True, text=True, timeout=timeout)
            runs[key] = {"rc": p.returncode, "tail": (p.stdout or "")[-2000:],
                         "stderr_tail": (p.stderr or "")[-500:]}
        except (OSError, subprocess.TimeoutExpired) as e:
            runs[key] = {"rc": None, "tail": "", "stderr_tail": "%s: %s" % (type(e).__name__, e)}
    rp = runs["replay"]
    m = _last_line(rp["tail"], r"^(\d+) failure\(s\), (\d+) skipped: (.*)$")
    replay_cases = [c for c in REPLAY_REQUIRED if c != "governance" and c not in FUNNEL_REPLAY_CASES
                    and c not in ("funnel_mutations", "domain_free")]
    if rp["rc"] == 0 and m and m.group(1) == "0":
        skipped = [] if m.group(3).strip() == "none" else [s.strip() for s in m.group(3).split(",")]
        for c in replay_cases + list(REPLAY_MAY_SKIP):
            cases[c] = "pass"
        for s in skipped:
            hit = [case for prefix, case in REPLAY_SKIP_CASES if s.startswith(prefix)]
            if not hit:
                notes.append("unrecognised skip %r" % s)
                cases["unrecognised_skip"] = "fail"
                continue
            cases[hit[0]] = "skip"
    else:
        notes.append("the replay script exited %s%s" % (rp["rc"], "" if m else
                                                        " with no closing summary line"))
        for c in replay_cases:
            cases[c] = "fail"
    gv = runs["governance"]
    cases["governance"] = "pass" if gv["rc"] == 0 and "ALL PASS" in gv["tail"] else "fail"
    # The funnel audit's scripts (docs/FUNNEL_AUDIT.md 8.9): each passes its
    # cases only on exit 0 with "0 failure(s), 0 skipped: none" (a skip there
    # is a case not run, never a pass).
    for key, keyed in (("funnel", FUNNEL_REPLAY_CASES), ("funnel_mutations", ("funnel_mutations",)),
                       ("domain_free", ("domain_free",))):
        if key not in scripts:
            notes.append("the %s script was not given: its case(s) %s are not run" % (key, ", ".join(keyed)))
            for c in keyed:
                cases[c] = "fail"
            continue
        path = root / scripts[key]
        try:
            p = subprocess.run([python or sys.executable, str(path)], cwd=str(root),
                               capture_output=True, text=True, timeout=timeout)
            runs[key] = {"rc": p.returncode, "tail": (p.stdout or "")[-2000:],
                         "stderr_tail": (p.stderr or "")[-500:]}
        except (OSError, subprocess.TimeoutExpired) as e:
            runs[key] = {"rc": None, "tail": "", "stderr_tail": "%s: %s" % (type(e).__name__, e)}
        m = _last_line(runs[key]["tail"], r"^(\d+) failure\(s\), (\d+) skipped: (.*)$")
        ok = runs[key]["rc"] == 0 and m is not None and m.group(1) == "0" and m.group(2) == "0"
        if not ok:
            notes.append("the %s script exited %s%s" % (key, runs[key]["rc"], "" if m else
                                                          " with no closing summary line"))
        for c in keyed:
            cases[c] = "pass" if ok else "fail"
    # The stream's scenario and mutation scripts (docs/CONTINUOUS_LOOP.md 6.8):
    # each case reports its own line "case <id>: pass|fail"; a script that does
    # not end clean ("0 failure(s), 0 skipped") fails every case it did not
    # report as a pass. A caller's own script set without them records none.
    for key, keyed in (("stream", STREAM_REPLAY_CASES), ("stream_mutations", (STREAM_MUTATION_CASE,))):
        if key not in scripts:
            continue
        path = root / scripts[key]
        out_all = ""
        try:
            p = subprocess.run([python or sys.executable, str(path)], cwd=str(root),
                               capture_output=True, text=True, timeout=timeout)
            out_all = p.stdout or ""
            runs[key] = {"rc": p.returncode, "tail": out_all[-2000:], "stderr_tail": (p.stderr or "")[-500:]}
        except (OSError, subprocess.TimeoutExpired) as e:
            runs[key] = {"rc": None, "tail": "", "stderr_tail": "%s: %s" % (type(e).__name__, e)}
        m = _last_line(runs[key]["tail"], r"^(\d+) failure\(s\), (\d+) skipped: (.*)$")
        clean = runs[key]["rc"] == 0 and m is not None and m.group(1) == "0" and m.group(2) == "0"
        said = {}
        for line in out_all.splitlines():
            mm = _CASE_RE.match(line.strip())
            if mm:
                said[mm.group(1)] = mm.group(2) if said.get(mm.group(1)) != "fail" else "fail"
        if not clean:
            notes.append("the %s script exited %s%s" % (key, runs[key]["rc"], "" if m else
                                                          " with no closing summary line"))
        if key == "stream_mutations":
            cases[STREAM_MUTATION_CASE] = "pass" if clean else "fail"
            continue
        for c in keyed:
            cases[c] = "pass" if (m is not None and said.get(c) == "pass") else "fail"
        # a failure the script reports outside every case still fails the record
        cases["stream_script"] = "pass" if clean else "fail"
    after = code_hash()
    if after != before:
        notes.append("the code changed while the tests ran")
        cases["code_stable"] = "fail"
    status = "fail" if check_replay_cases(cases) else "pass"
    return record_replay_result(status, cases, ctx=ctx, code=before,
                                detail={"notes": notes, "runs": runs})


# --- parameters and rendering ------------------------------------------------------
def canonical_recipes(names):
    """Recipe names joined in the pilot table's order, the form the policy enum holds."""
    got = [str(n).strip() for n in (names.split(",") if isinstance(names, str) else names or ())
           if str(n).strip()]
    bad = [n for n in got if n not in RECIPE_ORDER]
    if bad or not got or len(set(got)) != len(got):
        raise ValueError("recipes must be distinct names among %s, got %r" % (RECIPE_ORDER, names))
    return ",".join(r for r in RECIPE_ORDER if r in got)


def _builder_tail(argv, first):
    """The tokens after the builder module or script `first` in `argv`, or None."""
    toks = [str(a) for a in argv]
    for i, t in enumerate(toks):
        short = t[len("weed_optimizer_framework.tools."):] \
            if t.startswith("weed_optimizer_framework.tools.") else t
        if short == first or t.rsplit("/", 1)[-1] == first:
            return toks[i + 1:]
    return None


def params_from_argv(action, argv):
    """The policy parameters a builder argv carries, by the builder's grammar."""
    form = ARGV_FORMS.get(action)
    if form is None:
        raise ExecError("%s is not a builder command; it takes no argv" % action)
    first, command, spec, _required = form
    out = {}
    if "{pkg}" in first:
        # a stream action: the protocol package is read off the module token
        form_first = first
        first, pkg = _pkg_token(argv, first)
        out["pkg"] = pkg
        if form_first.endswith(".{module}"):
            out["module"] = first.split(".", 1)[1]
    tail = _builder_tail(argv, first)
    if tail is None:
        raise ExecError("the argv %r does not run %s" % (" ".join(map(str, argv)), first))
    if command and command.startswith("{") and command.endswith("}"):
        # a positional param (run_inc_funnel.sh VERB)
        if not tail or tail[0].startswith("--"):
            raise ExecError("the argv does not name the %s of %s" % (command[1:-1], first))
        out[command[1:-1]] = tail[0]
        tail = tail[1:]
    elif command:
        if not tail or tail[0] != command:
            raise ExecError("the argv does not run %s %s" % (first, command))
        tail = tail[1:]
    flags = {f: (p, k) for f, p, k in spec}
    i = 0
    while i < len(tail):
        tok = tail[i]
        if tok not in flags:
            raise ExecError("%r is not a flag %s accepts here (%s)" % (tok, action, sorted(flags)))
        param, kind = flags[tok]
        if param in out:
            raise ExecError("%s is given twice" % tok)
        if kind == "flag":
            out[param] = 1
            i += 1
            continue
        if kind == "list":
            vals, i = [], i + 1
            while i < len(tail) and not tail[i].startswith("--"):
                vals.append(tail[i])
                i += 1
            if not vals:
                raise ExecError("%s needs at least one value" % tok)
            out[param] = ",".join(vals)
            continue
        if i + 1 >= len(tail) or tail[i + 1].startswith("--"):
            raise ExecError("%s needs a value" % tok)
        val = tail[i + 1]
        i += 2
        if kind == "int":
            if not _INT_RE.match(val):
                raise ExecError("%s %r is not a whole number" % (tok, val))
            val = int(val)
        elif kind == "bigint":
            if not _BIGINT_RE.match(val):
                raise ExecError("%s %r is not a whole number" % (tok, val))
            val = int(val)
        elif kind == "auto":
            if not val.startswith(AUTO_REASON) or not val[len(AUTO_REASON):]:
                raise ExecError("an autopilot %s must read %r<cause>" % (tok, AUTO_REASON))
            val = val[len(AUTO_REASON):]
        out[param] = val
    return out


def _pkg_token(argv, first):
    """(the resolved first token, the protocol package) of a stream action's
    argv: the module token '<pkg>.<module>' (optionally under
    weed_optimizer_framework.tools.) that '{pkg}.<module>' stands for. A form
    '{pkg}.{module}' (the R0 verdicts) takes any module of STREAM_VERDICT_MODULES."""
    mod = first.split(".", 1)[1]
    mods = STREAM_VERDICT_MODULES if mod == "{module}" else (mod,)
    for t in [str(a) for a in argv]:
        short = t[len("weed_optimizer_framework.tools."):] \
            if t.startswith("weed_optimizer_framework.tools.") else t
        m = re.match(r"^([a-z][a-z0-9_]{0,31})\.(%s)$" % "|".join(re.escape(x) for x in mods), short)
        if m:
            return short, m.group(1)
    raise ExecError("the argv %r runs no <package>.%s module" % (" ".join(map(str, argv)), mod))


def _stream_meta_tokens(meta, run=False):
    toks = []
    keys = (("approval_id", "--approval-id"), ("decided_by", "--decided-by")) if run else (
        ("parent_exp", "--parent-exp"), ("child_exp", "--child-exp"), ("trigger", "--trigger"),
        ("approval_id", "--approval-id"), ("decided_by", "--decided-by"))
    for key, flag in keys:
        if meta.get(key):
            toks += [flag, str(meta[key])]
    return toks


def _flag_tokens(spec, p):
    toks = []
    for flag, param, kind in spec:
        v = p.get(param)
        if v is None or v == "":
            continue
        if kind == "flag":
            if v == 1 and not isinstance(v, bool):
                toks.append(flag)
        elif kind == "list":
            toks += [flag] + [x for x in str(v).split(",") if x]
        elif kind == "auto":
            toks += [flag, AUTO_REASON + str(v)]
        else:
            toks += [flag, str(v)]
    return toks


def _submit_meta_tokens(meta):
    toks = []
    for key, flag in (("parent_exp", "--parent-exp"), ("trigger", "--trigger"),
                      ("approval_id", "--approval-id"), ("decided_by", "--decided-by")):
        if meta.get(key):
            toks += [flag, str(meta[key])]
    return toks


def render(action, params, meta=None):
    """{"builder": [...] | None, "remote": [...] | None, "local": bool}.

    `builder` is the human form of the command (a lever card's argv, e.g.
    `inc.pilot build --exp pilot_v2 --replay-mode full`); `remote` is the
    remote.py verb line that runs it on the cluster. `meta` carries the submit
    provenance flags (parent_exp, trigger, approval_id, decided_by). Raises
    ExecError.
    """
    p = params if isinstance(params, dict) else {}

    def need(k):
        if p.get(k) in (None, ""):
            raise ExecError("%s needs the parameter %r" % (action, k))
        return str(p[k])

    if action == "inc_cancel_exp":
        return {"builder": None, "remote": ["cancel", "--exp", need("exp")], "local": False}
    if action == "inc_sync_outer":
        return {"builder": None, "remote": ["sync-outer"], "local": False}
    if action in ("inc_snapshot", "inc_report", "inc_advance"):
        remote = [action[len("inc_"):], "--exp", need("exp")]
        if action == "inc_snapshot":
            if p.get("ledger_from") is not None:
                remote += ["--ledger-from", str(p["ledger_from"])]
            if p.get("no_step1") == 1:
                remote += ["--no-step1"]
        return {"builder": None, "remote": remote, "local": False}
    if action == "inc_lit_fetch":
        need("arxiv_id")
        need("paper_id")
        return {"builder": None, "remote": None, "local": True}
    if action in ("inc_verify_queue", "inc_funnel_sync", "inc_stream_sync"):
        return {"builder": None, "remote": None, "local": True}
    if action in PLAN_ACTIONS:
        need("campaign")
        if p.get("n") in (None, ""):
            raise ExecError("%s needs the parameter 'n'" % action)
        if action == "inc_plan_submit":
            need("digest_sha256")
        # The segment is built from the lab's staged digest when the executor
        # plans the run (_plan_segment), not from the parameters alone.
        return {"builder": None, "remote": {"plan": action, "params": dict(p)}, "local": False}
    form = ARGV_FORMS.get(action)
    if form is None:
        raise ExecError("the executor has no rendering for action %r" % (action,))
    first, command, spec, required = form
    for k in required:
        need(k)
    if action == "inc_build_realloop" and LV.evidence_with_relevance(p):
        raise ExecError(M.EVIDENCE_WITH_RELEVANCE)
    if command and command.startswith("{") and command.endswith("}"):
        command = need(command[1:-1])
    if "{" in first:
        first = re.sub(r"\{([a-z_]+)\}", lambda m: need(m.group(1)), first)
    flags = _flag_tokens(spec, p)
    builder = [first] + ([command] if command else []) + flags
    if action == "inc_funnel_fetch" or action in LAB_ACTIONS:
        return {"builder": builder, "remote": None, "local": True}
    if action in STREAM_REMOTE:
        kind = STREAM_REMOTE[action]
        if kind == "run":
            remote = ["stream-run", first, command] + _stream_meta_tokens(meta or {}, run=True) + ["--"] + flags
        else:
            args = ([first] if kind == "build" else []) + ([command] if command else []) + flags
            remote = ["stream-submit", kind] + _stream_meta_tokens(meta or {}) + ["--"] + args
        return {"builder": builder, "remote": remote, "local": False}
    if action == "inc_unblock_transient":
        return {"builder": builder, "remote": ["unblock"] + flags, "local": False}
    sub, module = SUBMIT_FORMS[action]
    args = ([module] if module else []) + ([command] if command else []) + flags
    remote = ["submit", sub] + _submit_meta_tokens(meta or {}) + ["--"] + args
    return {"builder": builder, "remote": remote, "local": False}


def argv_check(rendered, argv):
    """(ok, reason): a proposal's argv must be the command the executor will run."""
    if not argv:
        return True, ""
    builder = rendered.get("builder")
    if not builder:
        return True, ""
    tail = _builder_tail(argv, builder[0])
    if tail is None or tail != builder[1:]:
        return False, ("the proposal's argv %r is not the command its parameters render (%r); "
                       "the executor runs only what the policy checked"
                       % (" ".join(map(str, argv)), " ".join(builder)))
    return True, ""


def _norm(v):
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def resolve_params(action, row, params, argv=None, est_gpu_hours=None):
    """(policy params, metadata params) of one request. Raises ExecError."""
    given = dict(params or {})
    if est_gpu_hours is not None:
        if "est_gpu_hours" in given and _norm(given["est_gpu_hours"]) != _norm(est_gpu_hours):
            raise ExecError("the proposal states two prices: est_gpu_hours %s at the top level "
                            "and %s in params" % (est_gpu_hours, given["est_gpu_hours"]))
        given.setdefault("est_gpu_hours", est_gpu_hours)
    if argv:
        policy = params_from_argv(action, argv)
        for k, v in given.items():
            if k in policy and _norm(policy[k]) != _norm(v):
                raise ExecError("params say %s=%r but the argv says %r" % (k, v, policy[k]))
    else:
        policy = {k: v for k, v in given.items() if k != "est_gpu_hours"}
    if action == "inc_build_realloop" and LV.evidence_with_relevance(policy):
        # realloop refuses the pair; so does remote.py submit, but the lab
        # refuses it here, before anything is queued or sent
        raise ExecError(M.EVIDENCE_WITH_RELEVANCE)
    meta = {k: v for k, v in given.items() if k not in policy and k != "est_gpu_hours"}
    if "est_gpu_hours" in given:
        f = row.get("est_su") if isinstance(row, dict) else {}
        if isinstance(f, dict) and f.get("hours_param") == "est_gpu_hours":
            policy["est_gpu_hours"] = given["est_gpu_hours"]
        else:
            meta["est_gpu_hours"] = given["est_gpu_hours"]
    return policy, meta


def _preamble(ctx):
    if ctx.preamble:
        return ctx.preamble
    env = os.environ.get("INCAP_REMOTE_PREAMBLE")
    if env:
        return env
    return ("source %s && conda activate bench && export REPO=%s && cd %s"
            % (shlex.quote(CONDA_SH), shlex.quote(M.CLUSTER_REPO),
               shlex.quote(M.CLUSTER_REPO + "/weed_llm_benchmark")))


def remote_script(argvs, preamble):
    """One shell script running each remote.py verb line in its own marked segment."""
    lines = ["%s || { echo INCAP_PREAMBLE_FAILED; exit 0; }" % preamble]
    for i, a in enumerate(argvs):
        if isinstance(a, str):           # a segment the executor built itself (the plan job's)
            cmd = a
        else:
            cmd = "python -u -m %s %s" % (REMOTE_MODULE, " ".join(shlex.quote(str(x)) for x in a))
        lines.append('echo "INCAP_SEG %d"; %s; echo "INCAP_SEG_END %d $?"' % (i, cmd, i))
    return "\n".join(lines)


def never_reached(res, preamble_failed=False):
    """"" when a transport result may have reached the cluster, else why it
    certainly did not: the remote preamble failed, the hook refused to send
    the call (NotSent), or ssh never connected (no output at all, ssh's exit
    status 255 or a hook exception, and a connect error on the last stderr
    line). A timeout (the dashboard's slurm_sh returns an empty stdout then,
    whatever the cluster already ran) or a connection that dropped is not."""
    res = res if isinstance(res, dict) else {}
    if preamble_failed:
        return "the remote preamble failed"
    tail = ((res.get("stderr") or "").strip().splitlines() or [""])[-1][:300]
    if res.get("not_sent"):
        return "the call was not sent (%s)" % (tail or "refused by the hook")
    if not (res.get("stdout") or "").strip() and res.get("returncode") in (255, -2) \
            and _NEVER_CONNECTED_RE.search(tail):
        return "the connection was never made (%s)" % tail
    return ""


def parse_remote(res, n):
    """Per segment: {"started", "ok", "known", "may_have_run", "payload", "rc",
    "job_ids", "error"}. `may_have_run` is False only for a verb that certainly
    did not run (`never_reached`, or a known outcome that says it failed)."""
    res = res if isinstance(res, dict) else {}
    segs = [{"started": False, "rc": None, "raw": None, "lines": []} for _ in range(n)]
    cur, preamble_failed = None, False
    mark = M.REMOTE_MARK + " "
    for line in (res.get("stdout") or "").splitlines():
        s = line.strip()
        if s == "INCAP_PREAMBLE_FAILED":
            preamble_failed = True
            continue
        m = re.match(r"^INCAP_SEG (\d+)$", s)
        if m and int(m.group(1)) < n:
            cur = int(m.group(1))
            segs[cur]["started"] = True
            continue
        m = re.match(r"^INCAP_SEG_END (\d+) (-?\d+)$", s)
        if m and int(m.group(1)) < n:
            segs[int(m.group(1))]["rc"] = int(m.group(2))
            cur = None
            continue
        if cur is not None:
            segs[cur]["lines"].append(s)
            if s.startswith(mark):
                segs[cur]["raw"] = s[len(mark):]
    tail = ((res.get("stderr") or "").strip().splitlines() or [""])[-1][:300]
    unsent = never_reached(res, preamble_failed)
    out = []
    for sg in segs:
        payload = None
        if sg["raw"] is not None:
            try:
                payload = json.loads(sg["raw"])
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            payload = None
        ok = bool(payload) and payload.get("ok", True) is True and sg["rc"] == 0
        jobs = []
        if payload:
            if isinstance(payload.get("job_ids"), list):
                jobs = [str(j) for j in payload["job_ids"]]
            elif payload.get("job_id") not in (None, ""):
                jobs = [str(payload["job_id"])]
        if not jobs:
            jobs = _SUBMITTED_RE.findall("\n".join(sg["lines"]))
        may = True
        if ok:
            err = ""
        elif payload is not None:
            err = str(payload.get("error") or "the verb exited %s" % sg["rc"])
            may = False
        elif sg["started"]:
            err = "the verb printed no %s line; its outcome is unknown" % M.REMOTE_MARK
        elif unsent:
            err = (unsent if preamble_failed else "the verb never started: %s" % unsent)
            may = False
        else:
            err = ("the call ended with no output from the verb (%s): it may have run on the "
                   "cluster, and its outcome is unknown" % (tail or "no output"))
        out.append({"started": sg["started"], "ok": ok, "known": payload is not None,
                    "may_have_run": may, "payload": payload, "rc": sg["rc"], "job_ids": jobs,
                    "error": err})
    return out


# --- the research brain's plan job ----------------------------------------------------
# Cluster side of inc_plan_submit, run in the bench env from the nested copy.
# argv: input output raw_sha256 model job_name cluster_repo payload [role]; a
# devil's-advocate digest adds role "adversary", exported to the job as
# PLAN_ROLE (docs/FUNNEL_AUDIT.md 8.6). It writes
# the staged digest once (identical bytes are accepted again), refuses when a
# reply already exists, and sbatches run_inc_plan.sh from the repo root.
_PLAN_SUBMIT_PY = """
import base64, gzip, hashlib, json, os, re, subprocess, sys
inp, out, want, model, name, repo, payload = sys.argv[1:8]
role = sys.argv[8] if len(sys.argv) > 8 else ""
rec = {"verb": "plan-submit", "ok": False, "input": inp, "output": out}
if role not in ("", "adversary"):
    rec["error"] = "role %r is not adversary" % role
    print("INCAP " + json.dumps(rec, sort_keys=True))
    sys.exit(1)
def done(**kw):
    rec.update(kw)
    print("INCAP " + json.dumps(rec, sort_keys=True))
    sys.exit(0 if rec["ok"] else 1)
plans = os.path.realpath(os.path.join(repo, "results/framework/inc/_campaign/plans"))
for f in (inp, out):
    if not os.path.realpath(f).startswith(plans + os.sep):
        done(error="%s is not under %s" % (f, plans))
try:
    raw = gzip.decompress(base64.b64decode(payload))
except Exception as e:
    done(error="the staged digest does not decode: %s" % e)
if hashlib.sha256(raw).hexdigest() != want:
    done(error="the staged digest does not match the sha256 the lab sent")
if os.path.exists(out):
    done(error="%s already exists: this plan was submitted before" % out)
if os.path.exists(inp):
    with open(inp, "rb") as fh:
        if fh.read() != raw:
            done(error="%s exists with other content; a staged digest is written once" % inp)
else:
    os.makedirs(os.path.dirname(inp), exist_ok=True)
    tmp = "%s.tmp.%d" % (inp, os.getpid())
    with open(tmp, "wb") as fh:
        fh.write(raw)
    os.replace(tmp, inp)
os.makedirs(os.path.join(plans, "logs"), exist_ok=True)
export = "ALL,PLAN_INPUT=%s,PLAN_OUTPUT=%s" % (inp, out) + (",PLAN_MODEL=%s" % model if model else "") \
    + (",PLAN_ROLE=%s" % role if role else "")
try:
    p = subprocess.run(["sbatch", "--parsable", "--job-name=" + name, "--export=" + export,
                        "weed_llm_benchmark/run_inc_plan.sh"], cwd=repo, capture_output=True,
                       text=True, timeout=120)
except Exception as e:
    done(error="sbatch did not run: %s" % e)
if p.returncode != 0:
    done(error="sbatch exited %d: %s" % (p.returncode, (p.stderr or p.stdout)[-300:]))
jid = ((p.stdout or "").strip().splitlines() or [""])[-1].split(";")[0].strip()
if not re.match(r"^[0-9]+$", jid):
    done(error="sbatch printed no job id: %r" % (p.stdout or "")[-200:])
done(ok=True, job_id=jid)
"""
# Cluster side of inc_plan_pull. argv: output cluster_repo. Reads the reply
# when it exists; writes nothing.
_PLAN_PULL_PY = """
import json, os, sys
out, repo = sys.argv[1:3]
rec = {"verb": "plan-pull", "ok": True, "output": out, "exists": False, "reply": None}
plans = os.path.realpath(os.path.join(repo, "results/framework/inc/_campaign/plans"))
if not os.path.realpath(out).startswith(plans + os.sep):
    rec.update(ok=False, error="%s is not under %s" % (out, plans))
elif os.path.isfile(out):
    rec["exists"] = True
    rec["bytes"] = os.path.getsize(out)
    if rec["bytes"] > (4 << 20):
        rec.update(ok=False, error="the reply is larger than 4 MiB")
    else:
        try:
            with open(out) as fh:
                rec["reply"] = json.load(fh)
        except ValueError as e:
            rec["error"] = "the reply is not JSON: %s" % e
print("INCAP " + json.dumps(rec, sort_keys=True))
sys.exit(0 if rec["ok"] else 1)
"""


def plan_input_path(ctx, campaign, n):
    """The lab's staged digest of plan n of a campaign (what inc_plan_submit ships)."""
    return ctx.campaign_dir / "plans" / str(campaign) / ("%d.input.json" % int(n))


def _plan_segment(ctx, action, params):
    """The shell line of one plan-job segment. Raises ExecError.

    inc_plan_submit ships the lab's staged digest only when it is the digest
    the request names: its recorded sha256 and a recomputation of it both
    equal `digest_sha256`, and it is plan `n` of `campaign`. The cluster side
    checks the bytes it received against the lab's sha256 of them.
    """
    from . import brain_plan as BP
    campaign, n = str(params.get("campaign")), int(params.get("n"))
    paths = BP.plan_paths(campaign, n)
    if action == "inc_plan_pull":
        args = [paths["output"], M.CLUSTER_REPO]
        return "python -u -c %s %s" % (shlex.quote(_PLAN_PULL_PY), " ".join(shlex.quote(a) for a in args))
    local = plan_input_path(ctx, campaign, n)
    try:
        raw = local.read_bytes()
        digest = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise ExecError("the staged digest %s cannot be read (%s)" % (local, type(e).__name__))
    want = str(params.get("digest_sha256"))
    if not isinstance(digest, dict) or digest.get("sha256") != want \
            or BP.digest_sha256(digest) != want:
        raise ExecError("the staged digest %s is not the digest %s... this request names"
                        % (local, want[:12]))
    if str(digest.get("campaign")) != campaign or digest.get("n") != n:
        raise ExecError("the staged digest %s is plan %r of %r, not plan %d of %s"
                        % (local, digest.get("n"), digest.get("campaign"), n, campaign))
    adversary = digest.get("schema") == BP.DA_DIGEST_SCHEMA
    if adversary and digest.get("role") != "adversary":
        raise ExecError("the staged DA digest %s names role %r" % (local, digest.get("role")))
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    if len(payload) > PLAN_MAX_STAGED_CHARS:
        raise ExecError("the staged digest compresses to %d characters, over the %d one ssh command "
                        "line carries" % (len(payload), PLAN_MAX_STAGED_CHARS))
    args = [paths["input"], paths["output"], hashlib.sha256(raw).hexdigest(),
            str(params.get("model") or ""), ("inc_da_%s_%d" if adversary else "inc_plan_%s_%d") % (campaign, n),
            M.CLUSTER_REPO, payload] + (["adversary"] if adversary else [])
    return "python -u -c %s %s" % (shlex.quote(_PLAN_SUBMIT_PY), " ".join(shlex.quote(a) for a in args))


# --- small helpers -------------------------------------------------------------------
def _tier(actor):
    m = POL._ACTOR_RE.match(actor) if isinstance(actor, str) else None
    return m.group("tier") if m else None


def _is_human(actor):
    return _tier(actor) == "human" and ":" in str(actor)


def _campaign(c):
    if c is None:
        return None
    if not isinstance(c, dict):
        raise ExecError("campaign config must be an object")
    name = c.get("name")
    if not _NAME_RE.match(str(name or "")):
        raise ExecError("campaign name %r does not match %s" % (name, _NAME_RE.pattern))
    out = dict(c)
    out["autonomy"] = str(c.get("autonomy") or "off")
    return out


def _costed(row):
    f = row.get("est_su") if isinstance(row, dict) else None
    return isinstance(f, dict) and "gpu_type" in f


def _resources_known(resources, costed):
    """"" or why a costed action may not run on these resources.

    policy._check_resources refuses R1+ only when `mongo_down` is set, so an
    empty resources dict reads as healthy there. GPU time is not spent on an
    unknown Mongo state: the caller must say (`mongo_ok` or `mongo_down`).
    """
    if not costed:
        return ""
    if not isinstance(resources.get("mongo_down"), bool):
        return ("this action costs GPU time and the resources do not state Mongo's health "
                "(mongo_ok / mongo_down); unknown is not healthy, so it is refused")
    return ""


def _transient_kinds():
    """(kinds, "") from thresholds.json D5.transient_cause_kinds, or (None, why)."""
    try:
        th = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
        kinds = th["D5"]["transient_cause_kinds"]["value"]
    except Exception as e:
        return None, "thresholds.json D5.transient_cause_kinds is unreadable (%s)" % type(e).__name__
    if not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds):
        return None, "thresholds.json D5.transient_cause_kinds is not a list of names"
    return kinds, ""


def _unblock_check(ctx, camp, params):
    """[reasons] an automatic (non-human) unblock may not run; [] when it may.

    The cause must be on D5's transient allow-list; the policy row's pattern
    admits any lower-case word, so without this the cause would be whatever
    the request declared. When the ticker gave its diagnoses, a fired D5 of
    that experiment must list the unit as blocked by that transient cause.
    """
    kinds, err = _transient_kinds()
    if kinds is None:
        return [err + "; an automatic unblock is refused"]
    cause, unit, exp = params.get("cause"), params.get("unit"), params.get("exp")
    if cause not in kinds:
        return ["cause %r is not on the transient allow-list %s (thresholds.json D5); a unit "
                "blocked for any other cause waits for a person" % (cause, kinds)]
    fired = ctx.fired_diagnoses((camp or {}).get("name"))
    if fired is None:
        return []
    for d in fired:
        if d.get("id") != "D5" or d.get("exp") != exp:
            continue
        for u in (d.get("detail") or {}).get("units") or []:
            if isinstance(u, dict) and u.get("unit") == unit and u.get("cause") == cause \
                    and u.get("transient") is True:
                return []
    return ["no fired D5 of %s lists unit %s as blocked by a transient %r cause" % (exp, unit, cause)]


def _cite_ok(c):
    """True for a model.cite shape: an artifact and a ledger line or a JSON pointer."""
    if not isinstance(c, dict) or "value" not in c:
        return False
    art = c.get("artifact")
    if not isinstance(art, str) or not art.strip() or len(art) > 512 or "\0" in art:
        return False
    line, ptr = c.get("line"), c.get("pointer")
    if line is None and ptr is None:
        return False
    if line is not None and (not isinstance(line, int) or isinstance(line, bool) or line < 1):
        return False
    if ptr is not None and (not isinstance(ptr, str) or (ptr and not ptr.startswith("/"))):
        return False
    return True


def _canon(v):
    return json.dumps(v, sort_keys=True, default=str)


def _trim(obj):
    txt = json.dumps(obj, sort_keys=True, default=str)
    if len(txt) <= _LOG_PAYLOAD_CHARS:
        return obj
    return {"truncated": True, "chars": len(txt), "head": txt[:_LOG_PAYLOAD_CHARS]}


def executions(ctx=None, raw=False):
    """Every execution-log record, oldest first; a torn line is skipped.

    Folded by `budget.fold` into one record per run (its outcome, or its
    `started` record when no outcome was written) unless `raw`.
    """
    ctx = ctx or Context()
    if not raw:
        return B.fold(executions(ctx, raw=True))
    out = []
    try:
        with open(str(ctx.exec_log), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return []
    return out


def _log(ctx, res):
    """Append one record; True when written. A failure is kept on the result
    (`warnings`, `log_error`): a run's charge is already on disk in its
    `started` record, so a lost outcome line overstates the spend, never
    understates it."""
    rec = {k: v for k, v in res.items() if not k.startswith("_")}
    rec["remote"] = _trim(rec.get("remote")) if rec.get("remote") is not None else None
    try:
        ctx.exec_log.parent.mkdir(parents=True, exist_ok=True)
        with open(str(ctx.exec_log), "ab") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str).encode("utf-8") + b"\n")
        return True
    except Exception as e:
        res.setdefault("warnings", []).append("the execution log could not be written: %s" % e)
        res["log_error"] = str(e)
        return False


def _log_writable(ctx):
    """"" when the execution log can be appended to, else why not (nothing is
    written: an append of zero bytes)."""
    try:
        ctx.exec_log.parent.mkdir(parents=True, exist_ok=True)
        with open(str(ctx.exec_log), "ab"):
            pass
        return ""
    except Exception as e:
        return ("the execution log %s cannot be written (%s); nothing runs that the log "
                "cannot record, charge and count" % (ctx.exec_log, e))


def _normalize(request):
    r = dict(request or {})
    params = dict(r.get("params") or {})
    action = r.get("policy_action") or r.get("action")
    success = r.get("success")
    return {"id": r.get("id"), "action": action, "params": params,
            "risk": (str(r.get("risk")).upper() if r.get("risk") else None),
            "lever": r.get("lever"), "trigger": [str(t) for t in (r.get("trigger") or [])],
            "cites": list(r.get("cites") or []), "argv": [str(a) for a in (r.get("argv") or [])],
            "reason": str(r.get("reason") or (("success: %s" % success) if success else "")).strip(),
            "parent_exp": r.get("parent_exp"), "child_exp": r.get("child_exp"),
            "est_gpu_hours": r.get("est_gpu_hours"), "proposed_by": r.get("proposed_by"),
            "meta_params": {}}


def _new_result(req, actor, camp, now):
    return {"ts": datetime.datetime.fromtimestamp(now, datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"), "epoch": now,
            "campaign": (camp or {}).get("name"), "actor": actor, "authorized_as": None,
            "action": req["action"], "risk": req["risk"], "params": req["params"],
            "meta_params": req.get("meta_params") or {},
            "status": None, "ok": False, "reasons": [], "approval_id": None, "basis": None,
            "lever": req["lever"], "trigger": req["trigger"], "proposal_id": req["id"],
            "parent_exp": req["parent_exp"], "child_exp": req["child_exp"], "est_su": None,
            "charged": False, "job_ids": [], "remote": None, "argv": None, "decided_by": None,
            "run_id": None}


def _finish(ctx, res, status, reasons=()):
    res["status"] = status
    res["reasons"] = list(res.get("reasons") or []) + [str(r) for r in reasons]
    res["ok"] = status in ("executed", "filed")
    quiet = res.pop("_quiet_repeat", False)
    if quiet and status == "refused" and res.get("approval_id"):
        # run_approved retries an approved item every tick; the same refusal
        # again is not a new event.
        last = [r for r in executions(ctx) if r.get("approval_id") == res["approval_id"]]
        if last and last[-1].get("status") == "refused" \
                and last[-1].get("reasons") == res["reasons"]:
            res["repeat_of"] = last[-1].get("ts")
            return res
    _log(ctx, res)
    return res


def _budget(ctx, camp, now):
    if camp is None:
        return None
    return B.state(camp, executions(ctx), ctx.domain_budget, now, ctx.domain, ctx.su_base_dir)


def _resolve(req, row, res):
    """Resolve the request's policy parameters in place; returns the rendering."""
    policy, meta = resolve_params(req["action"], row, req["params"], req["argv"],
                                  req["est_gpu_hours"])
    req["params"], req["meta_params"] = policy, meta
    res["params"], res["meta_params"] = policy, meta
    if req["action"] in ("inc_build_segment", "inc_build_consolidation"):
        # A stream build names no --exp: the experiment it builds is the
        # stream's next <stream>_s|m|c|b<NNN>, which the ticker states as
        # child_exp; it must be one of that stream's names (the budget releases
        # the estimate when that experiment reports its spend). A fork builds
        # no experiment.
        child = req["child_exp"]
        if policy.get("verb") == "fork":
            if child not in (None, ""):
                raise ExecError("a fork builds no experiment; child_exp %r is not one" % (child,))
        elif not re.match(STREAM_CHILD_RE % re.escape(str(policy.get("stream") or "")), str(child or "")):
            raise ExecError("child_exp %r is not an experiment of stream %r (<stream>_s|m|c|b<NNN>)"
                            % (child, policy.get("stream")))
        elif policy.get("exp") not in (None, "") and policy["exp"] != child:
            raise ExecError("child_exp %r is not the segment --exp names (%r)" % (child, policy["exp"]))
    elif str(req["action"]).startswith("inc_build_"):
        # The budget releases a build's estimate when the experiment it builds
        # reports its spend; a child_exp naming any other experiment would
        # release it early, so it is not the caller's to choose.
        exp = policy.get("exp")
        if req["child_exp"] not in (None, "") and req["child_exp"] != exp:
            raise ExecError("child_exp %r is not the experiment this build makes (%r)"
                            % (req["child_exp"], exp))
        req["child_exp"] = exp
        res["child_exp"] = exp
    rendered = render(req["action"], policy)
    ok, why = argv_check(rendered, req["argv"])
    if not ok:
        raise ExecError(why)
    res["argv"] = rendered["builder"]
    return rendered


def _submit_meta(res):
    """The provenance flags remote.py submit records, validated by its own rules."""
    meta = {}
    parent = res.get("parent_exp")
    if parent:
        if not _NAME_RE.match(str(parent)):
            raise ExecError("parent_exp %r is not an experiment name" % (parent,))
        meta["parent_exp"] = str(parent)
    trig = [str(t) for t in res.get("trigger") or []]
    if trig:
        bad = [t for t in trig if not _TRIGGER_RE.match(t)]
        if bad:
            raise ExecError("trigger ids %r do not match %s" % (bad, _TRIGGER_RE.pattern))
        meta["trigger"] = ",".join(trig)
    if res.get("approval_id"):
        if not _APPROVAL_RE.match(str(res["approval_id"])):
            raise ExecError("approval id %r cannot be recorded" % res["approval_id"])
        meta["approval_id"] = str(res["approval_id"])
    who = res.get("decided_by") or res.get("authorized_as")
    if who:
        if not _PROV_ACTOR_RE.match(str(who)):
            raise ExecError("decided_by %r cannot be recorded" % who)
        meta["decided_by"] = str(who)
    return meta


def _plan(res, action, params, rendered, ctx=None):
    """A direct execution, with the remote.py line rendered with its provenance
    (a plan-job action: the segment built from the lab's staged digest)."""
    remote = None
    if action in PLAN_ACTIONS:
        remote = _plan_segment(ctx, action, params)
    elif not rendered["local"]:
        meta = _submit_meta(res) if action in SUBMIT_FORMS or action in STREAM_REMOTE else {}
        if STREAM_REMOTE.get(action) == "build" and res.get("child_exp"):
            if not _NAME_RE.match(str(res["child_exp"])):
                raise ExecError("child_exp %r is not an experiment name" % res["child_exp"])
            meta["child_exp"] = str(res["child_exp"])
        remote = render(action, params, meta)["remote"]
    return {"res": res, "action": action, "params": params, "local": rendered["local"],
            "remote": remote}


# --- preparing one request -----------------------------------------------------------
def _prepare(request, actor, campaign, ctx):
    """Either a finished result, or a plan for direct execution."""
    now = ctx.clock()
    req = _normalize(request)
    try:
        camp = _campaign(campaign)
    except ExecError as e:
        return _finish(ctx, _new_result(req, actor, None, now), "refused", [str(e)]), None
    res = _new_result(req, actor, camp, now)
    if req["risk"] == "R4":
        return _finish(ctx, res, "refused", [
            "R4 is a human card only: the executor never runs it, and the approval queue "
            "never holds it"]), None
    if not req["action"]:
        return _finish(ctx, res, "refused", ["the request names no policy action"]), None
    row = POL.describe(req["action"])
    if not row.get("known"):
        return _finish(ctx, res, "refused", [row.get("reason") or "unknown action"]), None
    risk = row["risk"]
    if risk == "R4":
        return _finish(ctx, res, "refused", ["%s is R4: a human card only, never executed"
                                             % req["action"]]), None
    if req["risk"] and req["risk"] != risk:
        return _finish(ctx, res, "refused", [
            "the request says %s but the policy table says %s for %s"
            % (req["risk"], risk, req["action"])]), None
    res["risk"] = risk
    tier = _tier(actor)
    if tier is None:
        return _finish(ctx, res, "refused", ["actor %r is not a recognised actor" % (actor,)]), None
    if req["proposed_by"] and req["proposed_by"] != actor and tier != "human":
        # A brain proposal submitted under the autopilot's name would borrow its
        # R2 authority; only a person may act on someone else's proposal.
        return _finish(ctx, res, "refused", [
            "the request was proposed by %s; it is submitted as that actor, not as %s"
            % (req["proposed_by"], actor)]), None
    try:
        rendered = _resolve(req, row, res)
    except ExecError as e:
        return _finish(ctx, res, "refused", [str(e)]), None
    est = POL.estimate_su(req["action"], req["params"])
    res["est_su"] = est["su"]
    costed = _costed(row)
    if costed and est["su"] is None:
        return _finish(ctx, res, "refused", [
            "%s costs GPU time and its estimate is unknown (%s); an unpriced action cannot be "
            "charged to an envelope" % (req["action"], est["reason"])]), None
    if costed and camp is None:
        return _finish(ctx, res, "refused", [
            "%s costs GPU time and no campaign envelope was given to charge it to"
            % req["action"]]), None
    if tier == "round-scheduler" and camp and camp.get("paused_reason") and risk != "R0":
        return _finish(ctx, res, "refused", ["the campaign is paused: %s"
                                             % camp["paused_reason"]]), None
    resources = ctx.resources()
    if not rendered["local"] and resources.get("cluster_reachable") is False:
        return _finish(ctx, res, "refused", ["the cluster is not reachable"]), None
    bud = _budget(ctx, camp, now)
    if risk == "R3" and tier == "round-scheduler":
        return _file_r3(ctx, req, actor, camp, row, est, bud, resources, res), None
    auth = POL.authorize(actor, req["action"], req["params"], B.budget_state(bud), resources)
    if not auth["allowed"]:
        return _finish(ctx, res, "refused", auth["reasons"]), None
    if auth["needs_approval"]:
        if POL._decide(tier, risk) == "direct":
            return _finish(ctx, res, "refused", ["budget: " + "; ".join(auth["reasons"])]), None
        return _file(ctx, req, actor, camp, risk, est, bud, res, auth["reasons"]), None
    if req["action"] in GATED_R2_ACTIONS and tier != "human":
        # A stream data lever runs directly only under its gates (6.5, 6.6);
        # otherwise a person decides (shadow mode files every one).
        gate = _gated_r2(ctx, req, camp)
        if gate:
            return _file(ctx, req, actor, camp, risk, est, bud, res,
                         ["awaiting a person: " + w for w in gate]), None
    res["authorized_as"], res["decided_by"], res["basis"] = actor, actor, "direct"
    why = _resources_known(resources, costed)
    if why:
        return _finish(ctx, res, "refused", [why]), None
    if req["action"] == "inc_unblock_transient" and tier != "human":
        bad = _unblock_check(ctx, camp, req["params"])
        if bad:
            return _finish(ctx, res, "refused", bad), None
        prior = [r for r in executions(ctx)
                 if r.get("action") == "inc_unblock_transient"
                 and r.get("status") in ("executed", "failed", "started") and not never_ran(r)
                 and (r.get("params") or {}).get("exp") == req["params"].get("exp")
                 and (r.get("params") or {}).get("unit") == req["params"].get("unit")]
        if prior:
            return _finish(ctx, res, "refused", [
                "unit %s of %s was already unblocked automatically at %s; at most one automatic "
                "unblock per unit, the next is a human card"
                % (req["params"].get("unit"), req["params"].get("exp"), prior[0].get("ts"))]), None
    if costed:
        ok, why = B.fits(bud, est["su"])
        if not ok:
            return _finish(ctx, res, "refused", ["budget: " + r for r in why]), None
    try:
        plan = _plan(res, req["action"], req["params"], rendered, ctx)
    except ExecError as e:
        return _finish(ctx, res, "refused", [str(e)]), None
    return None, plan


def _find_filed(ctx, proposal_id):
    if not proposal_id:
        return None
    for item in AP.state(ctx.domain, root=ctx.approvals_root).values():
        if (item.get("context") or {}).get("proposal_id") == proposal_id:
            return item
    return None


def _context(req, camp, rendered_argv, bud, est):
    return {"campaign": (camp or {}).get("name"), "proposal_id": req["id"],
            "lever": req["lever"], "trigger": req["trigger"], "cites": req["cites"],
            "argv": req["argv"] or rendered_argv, "parent_exp": req["parent_exp"],
            "child_exp": req["child_exp"], "proposed_by": req["proposed_by"],
            "meta_params": req["meta_params"], "est_su": est["su"],
            "budget_at_filing": {k: (bud or {}).get(k) for k in
                                 ("envelope_su", "spent_su", "committed_su", "remaining_su",
                                  "daily_remaining_su")}}


def _reason(req, est, bud):
    parts = []
    if req["lever"]:
        parts.append("lever %s" % req["lever"])
    if req["trigger"]:
        parts.append("trigger %s" % ", ".join(req["trigger"]))
    if est["su"] is not None:
        money = "est %.4g SU" % est["su"]
        if bud and bud.get("remaining_su") is not None:
            money += " of %.4g SU left in the campaign envelope" % bud["remaining_su"]
        parts.append(money)
    head = req["reason"] or "%s requested" % req["action"]
    return head + (" (%s)" % "; ".join(parts) if parts else "")


def _propose(ctx, req, actor, camp, risk, est, bud, res):
    """approvals.propose, then a check that the id it returned is this request's.

    An approval id hashes (action, params, requester, ts); two identical requests
    filed at the same instant would share one, and the second would silently
    read as the first.
    """
    r = AP.propose(ctx.domain, req["action"], req["params"], risk, actor, _reason(req, est, bud),
                   res["epoch"], est_su=est["su"], root=ctx.approvals_root,
                   context=_context(req, camp, res.get("argv"), bud, est))
    if not r.get("ok"):
        return None, r.get("reason") or "the approval could not be filed"
    item = AP.state(ctx.domain, root=ctx.approvals_root).get(r["item"]["id"])
    if item is None or (item.get("context") or {}).get("proposal_id") != req["id"] \
            or item.get("action") != req["action"] or item.get("ts") != res["epoch"]:
        return None, "approval id %s collides with another request" % r["item"]["id"]
    return item, ""


def _file(ctx, req, actor, camp, risk, est, bud, res, why=()):
    existing = _find_filed(ctx, req["id"])
    if existing is not None:
        res["approval_id"] = existing["id"]
        return _finish(ctx, res, "filed", ["already filed as %s (%s)"
                                           % (existing["id"], existing.get("status"))])
    item, err = _propose(ctx, req, actor, camp, risk, est, bud, res)
    if item is None:
        return _finish(ctx, res, "refused", [err])
    res["approval_id"] = item["id"]
    return _finish(ctx, res, "filed", list(why) or ["filed for approval"])


def _lever_count(ctx, name, lever):
    """Submissions of `lever` in the campaign that ran or may have run."""
    return sum(1 for r in executions(ctx)
               if r.get("campaign") == name and r.get("lever") == lever
               and (r.get("status") == "executed" or r.get("charged")))


def _own_item(item, actor=None):
    """"" when a filed item is the autopilot's own envelope build, else why not.

    Requested by a grantee (`actor` itself when given), not carrying another
    proposer's proposal, and an action the envelope covers. A brain's item, or
    any other tier's, waits for a person even when the autopilot resubmits it.
    """
    cx = item.get("context") if isinstance(item.get("context"), dict) else {}
    by = item.get("requested_by")
    if by not in AP.ENVELOPE_GRANTEES or (actor is not None and by != actor):
        return ("approval %s was requested by %r; the envelope grants only the autopilot's own "
                "proposals, so it waits for a person" % (item.get("id"), by))
    prop = cx.get("proposed_by")
    if prop not in (None, "") and prop != by:
        return ("approval %s carries a proposal by %r; its autonomy is never self-granted, so it "
                "waits for a person" % (item.get("id"), prop))
    if item.get("action") not in AP.ENVELOPE_ACTIONS:
        return "%s is not an envelope build" % item.get("action")
    return ""


def _pcanon(params):
    return _canon({k: _norm(v) for k, v in (params or {}).items()})


def _differs(item, req):
    """What a resubmitted request states differently from the item filed under its id."""
    cx = item.get("context") if isinstance(item.get("context"), dict) else {}
    out = []
    if item.get("action") != req["action"]:
        out.append("action")
    if _pcanon(item.get("params")) != _pcanon(req["params"]):
        out.append("params")
    for k in ("lever", "trigger", "cites", "parent_exp", "child_exp"):
        if _canon(cx.get(k)) != _canon(req[k]):
            out.append(k)
    return out


def _gated_r2(ctx, req, camp):
    """[reasons] a gated stream data lever (GATED_R2_ACTIONS) may not run
    directly now; [] when it may: a stream campaign with data_autonomy 'on'
    (a person's flag), a stream replay pass, L16's floors set by a person
    (floor_gb, floor_su: a placeholder refuses), L24 citing a firing D28 or
    D31, and the stream limits (levers_stream.limits)."""
    from . import levers_stream as LS
    c = camp or {}
    fam = GATED_R2_ACTIONS[req["action"]]
    why = []
    if c.get("mode") != "stream":
        why.append("%s is a stream data lever; this campaign is not in stream mode" % fam)
    if c.get("data_autonomy") != "on":
        why.append("data_autonomy is %r, not 'on': shadow mode (diagnose and propose only); a person sets it"
                   % (c.get("data_autonomy"),))
    rp = stream_replay_status(ctx)
    if not rp["passed"]:
        why.append("stream replay: " + rp["reason"])
    if fam == "L16":
        try:
            th = LS.load_thresholds()
            floors = (LS.t(th, "D21", "floor_gb"), LS.t(th, "D21", "floor_su"))
        except Exception as e:
            floors = (None, None)
            why.append("stream_thresholds.json is unreadable (%s)" % type(e).__name__)
        if None in floors:
            why.append("floor_gb / floor_su are placeholders in stream_thresholds.json: L16 is not autonomous until "
                       "a person sets them from the measured first wave")
    if fam == "L24":
        cite = req["params"].get("cite")
        fired = ctx.fired_diagnoses(c.get("name")) or []
        if cite not in req["trigger"] or not any(d.get("id") == cite for d in fired):
            why.append("L24 needs a firing %s in its trigger" % cite)
    why += stream_limits(ctx, c, fam, req)
    return why


def _epoch(r):
    try:
        return float(r.get("epoch"))
    except (TypeError, ValueError):
        return None


def stream_limits(ctx, camp, lever, req):
    """[reasons] lever family `lever` is over a stream limit (stream_levers.json
    'limits', docs/CONTINUOUS_LOOP.md 6.6) with this request; [] otherwise.
    Counts come from the execution log (runs that ran or may have run); the
    in-flight and per-milestone / per-rollback / per-version counts come from
    the ticker (campaign 'in_flight', 'limit_counts')."""
    from . import levers_stream as LS
    lim = LS.limits(lever)
    if not lim:
        return []
    c = camp or {}
    name, now = c.get("name"), ctx.clock()
    recs = [r for r in executions(ctx) if r.get("campaign") == name and r.get("lever")
            and LS.family(r["lever"]) == lever and (r.get("status") == "executed" or r.get("charged"))]
    day = [r for r in recs if (_epoch(r) or 0.0) >= now - 86400.0]
    # a lab -> cluster sync (L16S) moves files already fetched: it is no job
    # and no download, so it does not use the day's job allowance
    day_jobs = [r for r in day if r.get("action") not in STREAM_NON_JOB_ACTIONS]
    p = req.get("params") or {}
    why = []
    for key in ("jobs_per_day", "per_day"):
        if key in lim and len(day_jobs) >= int(lim[key]):
            why.append("%s already ran %d time(s) in the last 24 h (limit %d)" % (lever, len(day_jobs), lim[key]))
    if "total" in lim and len(recs) >= int(lim["total"]):
        why.append("%s already ran %d time(s) in this campaign (limit %d)" % (lever, len(recs), lim["total"]))
    fetch = ("inc_stream_collect", "inc_stream_collect_lab", "inc_stream_collect_review")
    src = p.get("source")
    if src and req.get("action") in fetch:
        mine = [r for r in recs if r.get("action") in fetch and (r.get("params") or {}).get("source") == src]
        if "attempts_per_source" in lim and len(mine) >= int(lim["attempts_per_source"]):
            why.append("source %s was attempted %d time(s) (limit %d)" % (src, len(mine), lim["attempts_per_source"]))
        want = float(p.get("max_bytes") or 0) / 1e9
        if "gb_per_source" in lim:
            got = sum(float((r.get("params") or {}).get("max_bytes") or 0) for r in mine) / 1e9
            if got + want > float(lim["gb_per_source"]) + 1e-9:
                why.append("source %s would reach %.1f GB (limit %g GB unless a person approves)"
                           % (src, got + want, lim["gb_per_source"]))
        daily = [float(x) for x in (lim.get("gb_per_day"), c.get("collect_gb_daily")) if x is not None]
        if daily:
            got = sum(float((r.get("params") or {}).get("max_bytes") or 0) for r in day
                      if r.get("action") in fetch) / 1e9
            if got + want > min(daily) + 1e-9:
                why.append("today's fetches would reach %.1f GB (limit %g GB)" % (got + want, min(daily)))
        if c.get("collect_gb_envelope") is not None:
            got = sum(float((r.get("params") or {}).get("max_bytes") or 0) for r in recs
                      if r.get("action") in fetch) / 1e9
            if got + want > float(c["collect_gb_envelope"]) + 1e-9:
                why.append("the campaign's fetches would reach %.1f GB (collect_gb_envelope %g GB)"
                           % (got + want, float(c["collect_gb_envelope"])))
    if "in_flight" in lim:
        n = int(((c.get("in_flight") or {}).get(lever)) or 0)
        if n >= int(lim["in_flight"]):
            why.append("%d %s item(s) in flight (limit %d)" % (n, lever, lim["in_flight"]))
    counts = (c.get("limit_counts") or {}).get(lever) or {}
    for key in ("per_milestone", "per_rollback", "per_stream_version"):
        if key in lim and int(counts.get(key) or 0) >= int(lim[key]):
            why.append("%s already ran %d time(s) %s (limit %d)" % (lever, counts.get(key), key.replace("_", " "),
                                                                 lim[key]))
    return why


def _lever_params_check(lever, params):
    """[reasons] the resolved parameters are not lever `lever`'s (levers.json;
    stream_levers.json for a stream envelope lever)."""
    if lever in STREAM_ENVELOPE_LEVERS:
        from . import levers_stream as LS
        try:
            ok, bad = LS.check_params(lever, params)
        except Exception as e:
            return ["lever %s cannot be checked against stream_levers.json (%s)" % (lever, e)]
        return [] if ok else ["the parameters are not lever %s's: %s" % (lever, "; ".join(bad))]
    try:
        r = LV.row(lever)
        ok, bad = LV.check_params(lever, params)
    except Exception as e:
        return ["lever %s cannot be checked against levers.json (%s)" % (lever, e)]
    out = [] if ok else ["the parameters are not lever %s's: %s" % (lever, "; ".join(bad))]
    for k, v in sorted((r.get("fixed") or {}).items()):
        if params.get(k) != v:
            out.append("lever %s fixes %s = %r; the request has %r" % (lever, k, v, params.get(k)))
    missing = [k for k in r.get("requires") or [] if params.get(k) is None]
    if missing:
        out.append("lever %s requires %s" % (lever, ", ".join(missing)))
    return out


def _trigger_check(ctx, req, camp):
    """(reasons, matched diagnoses, cites verified): the proposal's trigger
    against the fired diagnoses the ticker reported.

    Every trigger id must be a fired diagnosis that names the lever (or
    supports it, levers.json `supports`) and whose every cite the proposal
    carries; at least one must name the lever itself; every cite must have the
    model.cite shape; and no stop-loss diagnosis may be firing. Cites the
    proposal carries beyond its diagnoses' (the cost estimate's) are counted
    in the grant as not verified here.
    """
    why, matched, verified = [], [], set()
    lever, cites, trig = req["lever"], req["cites"], req["trigger"]
    if not trig or not cites:
        why.append("the proposal carries no cited diagnosis (trigger and cites are required)")
    bad = [i for i, c in enumerate(cites) if not _cite_ok(c)]
    if bad:
        why.append("cites %s do not have the cite shape (artifact, line or JSON pointer, value)"
                   % bad)
    fired = ctx.fired_diagnoses((camp or {}).get("name"))
    if fired is None:
        why.append("the ticker gave no fired-diagnosis record, so the trigger cannot be checked")
        return why, matched, 0
    stops = sorted({str(d.get("id")) for d in fired if d.get("id") in STOP_DIAGNOSES
                    or set(d.get("levers") or []) & set(STOP_OPERATIONS)})
    if stops:
        why.append("stop-loss diagnosis %s is firing" % ", ".join(stops))
    try:
        supports = LV.load_menu().get("supports") or {}
    except Exception as e:
        supports = {}
        why.append("levers.json is unreadable (%s)" % type(e).__name__)
    have = {_canon(c) for c in cites}
    direct_any = False
    for t in trig:
        good = []
        for d in fired:
            if d.get("id") != t:
                continue
            dc = d.get("cites") or []
            if not dc or not all(_canon(c) in have for c in dc):
                continue
            direct = lever in (d.get("levers") or [])
            if direct or lever in (supports.get(t) or []):
                good.append((not direct, d))
        if not good:
            why.append("trigger %s: no fired diagnosis %s names lever %s and has every cite "
                       "carried by the proposal" % (t, t, lever))
            continue
        indirect, d = sorted(good, key=lambda x: x[0])[0]
        direct_any = direct_any or not indirect
        verified |= {_canon(c) for c in d.get("cites") or []}
        matched.append({"id": d.get("id"), "name": d.get("name"), "exp": d.get("exp"),
                        "levers": list(d.get("levers") or []),
                        "cites": len(d.get("cites") or []),
                        "via": "supports" if indirect else "levers",
                        "summary": str(d.get("summary") or "")[:300]})
    if matched and not direct_any:
        why.append("no trigger diagnosis names lever %s itself; a supporting diagnosis alone "
                   "does not trigger a build" % lever)
    return why, matched, len(verified & have)


def _autonomy(ctx, req, camp, row, est, bud, resources, item=None):
    """(ok, reasons, grant): may the envelope rule approve this R3 item now?"""
    why = []
    c = camp or {}
    if c.get("autonomy") != "envelope":
        why.append("autonomy is off for this campaign (autonomy=%r)" % c.get("autonomy"))
    granted_by = c.get("autonomy_granted_by")
    if not _is_human(granted_by):
        why.append("the campaign records no person who granted the envelope "
                   "(autonomy_granted_by=%r)" % (granted_by,))
    if c.get("paused_reason"):
        why.append("the campaign is paused: %s" % c["paused_reason"])
    if item is not None:
        own = _own_item(item)
        if own:
            why.append(own)
    lever, action = req["lever"], req["action"]
    stream_lever = c.get("mode") == "stream"
    table = STREAM_ENVELOPE_LEVERS if stream_lever else ENVELOPE_LEVERS
    rp = stream_replay_status(ctx) if stream_lever else replay_status(ctx)
    if not rp["passed"]:
        why.append("replay tests: " + rp["reason"])
    if lever not in table or action not in table[lever]:
        why.append("lever %r with %s is not covered by the envelope (covered: %s)"
                   % (lever, action, ", ".join(sorted(table))))
    else:
        why += _lever_params_check(lever, req["params"])
    if stream_lever and lever == "L21" and req["params"].get("to") != c.get("last_milestone_pool"):
        why.append("L21's envelope covers a rollback to the last milestone pool (%s) only; a rollback to %s "
                   "waits for a person" % (c.get("last_milestone_pool"), req["params"].get("to")))
    if action == "inc_build_realloop" and req["params"].get("no_truth") == 1:
        why.append("--no-truth is lever L6, which the envelope does not cover")
    tw, matched, n_verified = _trigger_check(ctx, req, camp)
    why += tw
    free = stream_lever and isinstance(row.get("est_su"), dict) and row["est_su"].get("fixed_su") == 0.0
    if free and est["su"] == 0.0:
        pass                  # a stream login-node verb (L21's rollback) costs no allocation by its policy row
    elif est["su"] is None or est["su"] <= 0:
        why.append("no positive SU estimate (%s)" % est["su"])
    else:
        _ok, r = B.fits(bud, est["su"], need_daily=True)
        why += ["budget: " + x for x in r]
    unknown = _resources_known(resources, True)
    if unknown:
        why.append(unknown)
    if c.get("name") and lever and stream_lever:
        from . import levers_stream as LS
        why += stream_limits(ctx, c, LS.family(lever), req)
    elif c.get("name") and lever:
        n = _lever_count(ctx, c["name"], lever)
        if n >= MAX_LEVER_SUBMISSIONS:
            why.append("lever %s already ran %d times in this campaign (stop-loss at more than %d)"
                       % (lever, n, MAX_LEVER_SUBMISSIONS))
    if not why:
        auth = POL.authorize(granted_by, action, req["params"], B.budget_state(bud), resources)
        if not auth["allowed"] or auth["needs_approval"]:
            why.append("authorize as %s: %s" % (granted_by, "; ".join(auth["reasons"])))
    b = bud or {}
    grant = {"lever": lever, "trigger": req["trigger"], "cites": req["cites"],
             "authority": granted_by, "campaign": c.get("name"), "proposal_id": req["id"],
             "diagnoses": matched,
             "checks": {"trigger_diagnoses_fired": bool(matched) and not tw,
                        "cites_from_diagnoses": n_verified,
                        "cites_not_verified_here": len(req["cites"]) - n_verified,
                        "lever_params": "levers.json %s bounds, fixed and required values" % lever},
             "replay": {"code_hash": rp["code_hash_now"],
                        "recorded_utc": (rp["recorded"] or {}).get("recorded_utc")},
             "envelope": {"envelope_su": b.get("envelope_su"), "spent_su": b.get("spent_su"),
                          "committed_su": b.get("committed_su"),
                          "remaining_su": b.get("remaining_su"),
                          "daily_cap_su": b.get("daily_cap_su"),
                          "daily_remaining_su": b.get("daily_remaining_su"),
                          "est_su": est["su"],
                          "balance_after_su": (None if b.get("remaining_su") is None
                                               or est["su"] is None
                                               else round(b["remaining_su"] - est["su"], 6))}}
    return (not why), why, grant


def _file_r3(ctx, req, actor, camp, row, est, bud, resources, res):
    """An R3 request from the autopilot: file it, and run it under the envelope rule."""
    if "round-scheduler" not in row["allowed_tiers"]:
        return _finish(ctx, res, "refused", ["round-scheduler may not request %s (allowed: %s)"
                                             % (req["action"], ", ".join(row["allowed_tiers"]))])
    ok, bad = POL._check_params(req["params"], row["param_bounds"])
    if not ok:
        return _finish(ctx, res, "refused", bad)
    item = _find_filed(ctx, req["id"])
    if item is None:
        item, err = _propose(ctx, req, actor, camp, "R3", est, bud, res)
        if item is None:
            return _finish(ctx, res, "refused", [err])
    else:
        diff = _differs(item, req)
        if diff:
            res["approval_id"] = item["id"]
            return _finish(ctx, res, "refused", [
                "proposal %s was filed as %s with other %s; a changed proposal is a new proposal "
                "(a new id)" % (req["id"], item["id"], ", ".join(diff))])
    res["approval_id"] = item["id"]
    status = item.get("status")
    if status == "denied":
        return _finish(ctx, res, "refused", ["denied by %s: %s" % (item.get("decided_by"),
                                                                   item.get("decision_reason"))])
    if status == "approved":
        if item.get("execution") is not None:
            return _finish(ctx, res, "refused", ["approval %s was already executed" % item["id"]])
        return execute_approved(item["id"], camp, ctx, invoked_by=actor)
    own = _own_item(item, actor)
    if own:
        return _finish(ctx, res, "filed", ["awaiting approval: " + own])
    unwritable = _log_writable(ctx)
    if unwritable:
        return _finish(ctx, res, "filed", ["awaiting approval: " + unwritable])
    auto, why, grant = _autonomy(ctx, req, camp, row, est, bud, resources, item=item)
    if not auto:
        return _finish(ctx, res, "filed", ["awaiting approval: " + w for w in why])
    g = AP.approve_within_envelope(ctx.domain, item["id"], actor, grant,
                                   "within the campaign envelope: " + _reason(req, est, bud),
                                   ctx.clock(), root=ctx.approvals_root)
    if not g.get("ok"):
        return _finish(ctx, res, "filed", ["awaiting approval: the envelope grant was refused: "
                                           + str(g.get("reason"))])
    return execute_approved(item["id"], camp, ctx, invoked_by=actor)


# --- running -----------------------------------------------------------------------------
def _call_slurm(ctx, script, timeout):
    try:
        try:
            r = ctx.slurm_sh(script, timeout)
        except TypeError:          # an older hook without a timeout parameter
            r = ctx.slurm_sh(script)
    except NotSent as e:
        return {"ok": False, "stdout": "", "stderr": "NotSent: %s" % e, "returncode": -3,
                "not_sent": True}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": "%s: %s" % (type(e).__name__, e),
                "returncode": -2}
    return r if isinstance(r, dict) else {}


def _apply_outcome(res, seg):
    may = seg.get("may_have_run", seg["started"] or seg["known"])
    res["remote"] = {"ok": seg["ok"], "known": seg["known"], "started": seg["started"],
                     "may_have_run": bool(may), "rc": seg["rc"], "payload": seg["payload"],
                     "error": seg["error"]}
    res["job_ids"] = seg["job_ids"]
    # Charged against the envelope when it ran, or may have run (an outcome
    # that is unknown, whether or not its segment marker came back).
    res["charged"] = bool(seg["ok"] or (not seg["known"] and may))
    if not seg["ok"]:
        res["reasons"] = list(res.get("reasons") or []) + [seg["error"]]
    return "executed" if seg["ok"] else "failed"


def never_ran(rec):
    """True for an execution record whose call certainly never reached the
    cluster (a connection that was never made, a call the hook did not send,
    a remote preamble that failed): failed, not charged, no outcome from the
    verb. Such a run may be retried under the same proposal id, and it
    releases an approval it had claimed."""
    rem = (rec or {}).get("remote")
    return (isinstance(rem, dict) and (rec or {}).get("status") == "failed"
            and not rec.get("charged") and not rem.get("known") and rem.get("may_have_run") is False)


def uncertain(rec):
    """True for an execution record whose outcome is unknown: it may have run
    (charged) and no verb outcome came back, or only its `started` record was
    written."""
    rec = rec or {}
    if rec.get("status") == "started":
        return True
    return (rec.get("status") == "failed" and bool(rec.get("charged"))
            and not (rec.get("remote") or {}).get("known"))


def _prior_run(folded, proposal_id):
    """The record of a run of this proposal that ran or may have run, else None."""
    for r in folded:
        if r.get("proposal_id") == proposal_id and (
                r.get("status") in ("executed", "started") or r.get("charged")):
            return r
    return None


def _claim(ctx, plans):
    """Write each plan's `started` record before anything runs; per plan None
    (claimed) or the reason it may not run.

    Under a lock on the execution log: a plan whose proposal id already ran,
    or may have run, is refused (a proposal runs once; a retry after an ssh
    timeout must not run and charge it again), and a plan whose record cannot
    be written does not run (a run the log cannot record is a charge the
    envelope would never see).
    """
    out = [None] * len(plans)
    try:
        with AP._LogLock(str(ctx.exec_log)):
            folded = executions(ctx)
            seen = set()
            for i, pl in enumerate(plans):
                res = pl["res"]
                pid = res.get("proposal_id")
                if pid:
                    prior = _prior_run(folded, pid)
                    if prior is not None or pid in seen:
                        out[i] = ("proposal %s already ran (%s at %s); a proposal runs once"
                                  % (pid, (prior or {}).get("status", "in this batch"),
                                     (prior or {}).get("ts", "now")))
                        continue
                    seen.add(pid)
                res["run_id"] = uuid.uuid4().hex
                rec = dict(res, status="started", ok=False, charged=True)
                if not _log(ctx, rec):
                    res["run_id"] = None
                    out[i] = ("the execution log could not be written (%s); nothing ran"
                              % rec.get("log_error"))
    except OSError as e:
        for i, pl in enumerate(plans):
            pl["res"]["run_id"] = None
            out[i] = "the execution log could not be locked (%s); nothing ran" % e
    return out


def _run_plans(ctx, plans):
    """Execute prepared plans: lab hooks one by one, remote verbs in one ssh call."""
    results = [None] * len(plans)
    ready = []
    for i, pl in enumerate(plans):
        if pl["local"] and not callable(ctx.local_hooks.get(pl["action"])):
            results[i] = _finish(ctx, pl["res"], "refused",
                                 ["no lab hook is registered for %s" % pl["action"]])
        elif not pl["local"] and not callable(ctx.slurm_sh):
            results[i] = _finish(ctx, pl["res"], "refused",
                                 ["no slurm_sh hook: the executor cannot reach the cluster"])
        else:
            ready.append(i)
    for i, why in zip(ready, _claim(ctx, [plans[i] for i in ready])):
        if why:
            results[i] = _finish(ctx, plans[i]["res"], "refused", [why])
    plans = [pl if results[i] is None else None for i, pl in enumerate(plans)]
    remote = []
    for i, pl in enumerate(plans):
        if pl is None:
            continue
        if pl["local"]:
            hook = ctx.local_hooks[pl["action"]]
            try:
                out = hook(dict(pl["params"]))
            except Exception as e:
                out = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
            out = out if isinstance(out, dict) else {"ok": False, "error": "hook returned %r" % out}
            seg = {"ok": out.get("ok") is True, "known": True, "started": True, "rc": 0,
                   "payload": out, "job_ids": [], "error": str(out.get("error") or "")}
            results[i] = _finish(ctx, pl["res"], _apply_outcome(pl["res"], seg))
        else:
            remote.append(i)
    if remote:
        script = remote_script([plans[i]["remote"] for i in remote], _preamble(ctx))
        timeout = min(MAX_BATCH_TIMEOUT_S, sum(REMOTE_TIMEOUT_S.get(plans[i]["action"],
                                                                    DEFAULT_REMOTE_TIMEOUT_S)
                                               for i in remote))
        segs = parse_remote(_call_slurm(ctx, script, timeout), len(remote))
        for i, seg in zip(remote, segs):
            results[i] = _finish(ctx, plans[i]["res"], _apply_outcome(plans[i]["res"], seg))
    return results


# --- public entry points -------------------------------------------------------------
def submit(request, actor=M.AUTOPILOT_ACTOR, campaign=None, ctx=None):
    """Authorise and run (or file) one request. Returns the result record.

    `request` is a Proposal (model.proposal) or a plain
    `{"policy_action"/"action", "params", ...}`. `status` is one of
    executed | failed | filed | refused; `ok` is true for executed and filed.
    """
    ctx = ctx or Context()
    done, plan = _prepare(request, actor, campaign, ctx)
    if plan is None:
        return done
    return _run_plans(ctx, [plan])[0]


def submit_many(requests, actor=M.AUTOPILOT_ACTOR, campaign=None, ctx=None):
    """`submit` for several requests; every direct remote verb shares one ssh call."""
    ctx = ctx or Context()
    out, plans, where = [], [], []
    for req in requests or ():
        done, plan = _prepare(req, actor, campaign, ctx)
        out.append(done)
        if plan is not None:
            plans.append(plan)
            where.append(len(out) - 1)
    for i, r in zip(where, _run_plans(ctx, plans)):
        out[i] = r
    return out


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _ledger_pos(v):
    """(line, sha256 or None) of a ledger_from value, or None when malformed.

    A value is a whole line count N, or (N, SHA256) where SHA256 is the
    `through_sha256` the previous read of that ledger returned: remote.py then
    checks that the file's first N lines still hash to it, and re-reads the
    ledger from line 0 with `prefix_mismatch` when they do not (a rewritten
    prefix is never spliced with new lines)."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        n, sha = v
        if sha is not None and not (isinstance(sha, str) and _SHA256_RE.match(sha)):
            return None
    else:
        n, sha = v, None
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        return None
    return n, sha


def campaign_snapshot(exps, advance=False, report="auto", ledger_from=None, no_step1=False,
                      actor=M.AUTOPILOT_ACTOR, campaign=None, ctx=None, plan_pull=None, funnel=False):
    """remote.py campaign-snapshot: advance, report and snapshot in one verb.

    Composite, so it has no policy row of its own: each part is authorised
    through its own row (inc_advance when `advance`, inc_report unless
    report='never', inc_snapshot) for every experiment, and the verb runs only
    when every part may run directly. The payload is returned whole to the
    caller and trimmed in the log.

    `ledger_from` maps an experiment to N or (N, through_sha256) (_ledger_pos).
    `plan_pull` ({"campaign", "n"}) adds the research brain's reply read
    (inc_plan_pull, R0) to the same ssh call; its own result, logged on its
    own, comes back under "plan_pull" (None when it was not asked for). A
    list of such dicts (the brain's and the devil's advocate's replies) comes
    back as a list under "plan_pulls", one result per pull, in order.
    `funnel` adds the funnel summary (inc_funnel_summary, R0: remote.py
    campaign-snapshot --funnel) to the snapshot record, under "funnel".
    """
    ctx = ctx or Context()
    now = ctx.clock()
    exps = [str(e) for e in (exps or [])]
    ledger_from = dict(ledger_from or {})
    params = {"exps": exps, "advance": bool(advance), "report": report,
              "ledger_from": ledger_from, "no_step1": bool(no_step1)}
    req = _normalize({"policy_action": "inc_campaign_snapshot", "params": params})
    try:
        camp = _campaign(campaign)
    except ExecError as e:
        return _finish(ctx, _new_result(req, actor, None, now), "refused", [str(e)])
    res = _new_result(req, actor, camp, now)
    res["risk"] = "R1" if advance else "R0"
    if not exps or len(set(exps)) != len(exps) or not all(_NAME_RE.match(e) for e in exps):
        return _finish(ctx, res, "refused", ["exps must be distinct experiment names, got %r"
                                             % (exps,)])
    if report not in ("auto", "always", "never"):
        return _finish(ctx, res, "refused", ["report must be auto, always or never"])
    bad = [k for k, v in ledger_from.items() if k not in exps or _ledger_pos(v) is None]
    if bad:
        return _finish(ctx, res, "refused", ["ledger_from names %r, not a listed experiment "
                                             "with a whole line count (and optionally the "
                                             "sha256 of those lines)" % bad])
    pos = {k: _ledger_pos(v) for k, v in ledger_from.items()}
    tier = _tier(actor)
    if tier is None:
        return _finish(ctx, res, "refused", ["actor %r is not a recognised actor" % (actor,)])
    if tier == "round-scheduler" and camp and camp.get("paused_reason") and advance:
        return _finish(ctx, res, "refused", ["the campaign is paused: %s" % camp["paused_reason"]])
    resources = ctx.resources()
    if resources.get("cluster_reachable") is False:
        return _finish(ctx, res, "refused", ["the cluster is not reachable"])
    parts = [("inc_funnel_summary", {})] if funnel else []
    for e in exps:
        sp = {"exp": e}
        if e in pos:
            sp["ledger_from"] = pos[e][0]
        if no_step1:
            sp["no_step1"] = 1
        parts.append(("inc_snapshot", sp))
        if report != "never":
            parts.append(("inc_report", {"exp": e}))
        if advance:
            parts.append(("inc_advance", {"exp": e}))
    for action, p in parts:
        auth = POL.authorize(actor, action, p, None, resources)
        if not auth["allowed"] or auth["needs_approval"]:
            return _finish(ctx, res, "refused", ["%s %s: %s" % (action, p.get("exp"),
                                                                "; ".join(auth["reasons"]))])
    res["authorized_as"], res["decided_by"], res["basis"] = actor, actor, "direct"
    argv = ["campaign-snapshot"]
    for e in exps:
        argv += ["--exp", e]
    if advance:
        argv.append("--advance")
    argv += ["--report", report]
    for e in exps:
        if e in pos:
            n, sha = pos[e]
            argv += ["--ledger-from", ("%s=%d:%s" % (e, n, sha)) if sha else ("%s=%d" % (e, n))]
    if no_step1:
        argv.append("--no-step1")
    if funnel:
        argv.append("--funnel")
    plans = [{"res": res, "action": "inc_campaign_snapshot", "params": params, "local": False,
              "remote": argv}]
    pulls = plan_pull if isinstance(plan_pull, list) else ([plan_pull] if plan_pull is not None else [])
    done = []
    for pp in pulls:
        preq = _normalize({"policy_action": "inc_plan_pull", "params": dict(pp)})
        pres = _new_result(preq, actor, camp, now)
        pres["risk"] = POL.risk_of("inc_plan_pull")
        auth = POL.authorize(actor, "inc_plan_pull", preq["params"], None, resources)
        if not auth["allowed"] or auth["needs_approval"]:
            done.append(_finish(ctx, pres, "refused", auth["reasons"]))
            continue
        try:
            seg = _plan_segment(ctx, "inc_plan_pull", preq["params"])
        except (ExecError, TypeError, ValueError) as e:
            done.append(_finish(ctx, pres, "refused", [str(e)]))
            continue
        pres["authorized_as"], pres["decided_by"], pres["basis"] = actor, actor, "direct"
        plans.append({"res": pres, "action": "inc_plan_pull", "params": preq["params"],
                      "local": False, "remote": seg})
        done.append(None)
    results = _run_plans(ctx, plans)
    out = results[0]
    it = iter(results[1:])
    pulled = [d if d is not None else next(it, None) for d in done]
    if isinstance(plan_pull, list):
        out["plan_pulls"] = pulled
    elif plan_pull is not None:
        out["plan_pull"] = pulled[0] if pulled else None
    return out


def stream_snapshot(sid, exps=(), advance=(), report="auto", ledger_from=None, dev_scores=(), sacct=(),
                    largest=False, actor=M.AUTOPILOT_ACTOR, campaign=None, ctx=None):
    """remote.py stream-snapshot: the one call of a stream campaign's tick that
    observes every lane (docs/CONTINUOUS_LOOP.md 6.2 step 2). Composite, like
    campaign_snapshot: each part is authorised through its own policy row
    (inc_stream_status; inc_snapshot and inc_report per experiment; inc_advance
    for each experiment in `advance`), and the verb runs only when every part
    may run directly. The payload comes back whole; the log keeps it trimmed."""
    ctx = ctx or Context()
    now = ctx.clock()
    exps = [str(e) for e in (exps or [])]
    advance = [str(e) for e in (advance or [])]
    ledger_from = dict(ledger_from or {})
    params = {"sid": sid, "exps": exps, "advance": advance, "report": report, "ledger_from": ledger_from,
              "dev_scores": list(dev_scores or []), "sacct": [str(j) for j in sacct or []], "largest": bool(largest)}
    req = _normalize({"policy_action": "inc_stream_snapshot", "params": params})
    try:
        camp = _campaign(campaign)
    except ExecError as e:
        return _finish(ctx, _new_result(req, actor, None, now), "refused", [str(e)])
    res = _new_result(req, actor, camp, now)
    res["risk"] = "R1" if advance else "R0"
    names = exps + advance + list(params["dev_scores"]) + [str(sid)]
    if not all(_NAME_RE.match(e) for e in names) or len(set(exps)) != len(exps) \
            or any(a not in exps for a in advance):
        return _finish(ctx, res, "refused", ["sid, exps, advance and dev_scores must be names, advance a subset of "
                                             "exps; got %r" % (names,)])
    if report not in ("auto", "always", "never"):
        return _finish(ctx, res, "refused", ["report must be auto, always or never"])
    if not all(re.match(r"^[0-9]+(_[0-9]+)?$", j) for j in params["sacct"]):
        return _finish(ctx, res, "refused", ["sacct job ids must be numeric"])
    bad = [k for k, v in ledger_from.items() if k not in exps or _ledger_pos(v) is None]
    if bad:
        return _finish(ctx, res, "refused", ["ledger_from names %r, not a listed experiment with a line count" % bad])
    tier = _tier(actor)
    if tier is None:
        return _finish(ctx, res, "refused", ["actor %r is not a recognised actor" % (actor,)])
    if tier == "round-scheduler" and camp and camp.get("paused_reason") and advance:
        return _finish(ctx, res, "refused", ["the campaign is paused: %s" % camp["paused_reason"]])
    resources = ctx.resources()
    if resources.get("cluster_reachable") is False:
        return _finish(ctx, res, "refused", ["the cluster is not reachable"])
    parts = [("inc_stream_status", {"sid": str(sid)})]
    for e in exps:
        parts.append(("inc_snapshot", {"exp": e}))
        if report != "never":
            parts.append(("inc_report", {"exp": e}))
        if e in advance:
            parts.append(("inc_advance", {"exp": e}))
    for action, p in parts:
        auth = POL.authorize(actor, action, p, None, resources)
        if not auth["allowed"] or auth["needs_approval"]:
            return _finish(ctx, res, "refused", ["%s %s: %s" % (action, p.get("exp") or p.get("sid"),
                                                                "; ".join(auth["reasons"]))])
    res["authorized_as"], res["decided_by"], res["basis"] = actor, actor, "direct"
    argv = ["stream-snapshot", "--sid", str(sid)]
    for e in exps:
        argv += ["--exp", e]
    for e in advance:
        argv += ["--advance", e]
    argv += ["--report", report]
    for e in exps:
        if e in ledger_from:
            n, sha = _ledger_pos(ledger_from[e])
            argv += ["--ledger-from", ("%s=%d:%s" % (e, n, sha)) if sha else ("%s=%d" % (e, n))]
    for e in params["dev_scores"]:
        argv += ["--dev-scores", e]
    for j in params["sacct"]:
        argv += ["--sacct", j]
    if largest:
        argv.append("--largest")
    plan = {"res": res, "action": "inc_stream_snapshot", "params": params, "local": False, "remote": argv}
    return _run_plans(ctx, [plan])[0]


def execute_approved(item_id, campaign=None, ctx=None, invoked_by=M.AUTOPILOT_ACTOR,
                     quiet_repeat=False, note=None):
    """Run one approved item once, authorised as whoever approved it.

    `quiet_repeat` (run_approved's retries): a refusal identical to the last
    one logged for this approval is returned but not logged again. `note`:
    a person's statement the execution log keeps with the result (the INC
    page's acknowledgement of a non-prospective real loop).
    """
    ctx = ctx or Context()
    now = ctx.clock()
    item = AP.state(ctx.domain, root=ctx.approvals_root).get(item_id)
    cx = (item or {}).get("context") or {}
    req = _normalize({"id": cx.get("proposal_id"), "policy_action": (item or {}).get("action"),
                      "params": (item or {}).get("params"), "risk": (item or {}).get("risk"),
                      "lever": cx.get("lever"), "trigger": cx.get("trigger"),
                      "cites": cx.get("cites"), "argv": cx.get("argv"),
                      "parent_exp": cx.get("parent_exp"), "child_exp": cx.get("child_exp"),
                      "proposed_by": cx.get("proposed_by")})
    try:
        camp = _campaign(campaign)
    except ExecError as e:
        return _finish(ctx, _new_result(req, invoked_by, None, now), "refused", [str(e)])
    res = _new_result(req, invoked_by, camp, now)
    res["approval_id"] = item_id
    res["_quiet_repeat"] = bool(quiet_repeat)
    if note:
        res["note"] = str(note)[:900]
    if item is None:
        return _finish(ctx, res, "refused", ["no such approval item %r" % (item_id,)])
    if item.get("status") != "approved":
        return _finish(ctx, res, "refused", ["approval %s is %s, not approved"
                                             % (item_id, item.get("status"))])
    if item.get("execution") is not None:
        ex = item["execution"]
        return _finish(ctx, res, "refused", [
            "approval %s was already executed (%s by %s); one approval runs once"
            % (item_id, ex.get("phase"), ex.get("executed_by"))])
    if cx.get("campaign") and (camp is None or camp["name"] != cx["campaign"]):
        return _finish(ctx, res, "refused", ["approval %s was filed for campaign %r"
                                             % (item_id, cx["campaign"])])
    row = POL.describe(req["action"])
    if not row.get("known"):
        return _finish(ctx, res, "refused", [row.get("reason") or "unknown action"])
    if row["risk"] == "R4" or req["risk"] == "R4":
        return _finish(ctx, res, "refused", ["R4 is a human card only, never executed"])
    if req["risk"] != row["risk"]:
        return _finish(ctx, res, "refused", ["the item says %s but the policy table says %s"
                                             % (req["risk"], row["risk"])])
    inv_tier = _tier(invoked_by)
    if inv_tier is None:
        return _finish(ctx, res, "refused", ["invoker %r is not a recognised actor" % (invoked_by,)])
    if camp and camp.get("paused_reason") and inv_tier != "human":
        return _finish(ctx, res, "refused", ["the campaign is paused: %s" % camp["paused_reason"]])
    try:
        rendered = _resolve(req, row, res)
    except ExecError as e:
        return _finish(ctx, res, "refused", [str(e)])
    est = POL.estimate_su(req["action"], req["params"])
    res["est_su"] = est["su"]
    costed = _costed(row)
    if costed and (est["su"] is None or camp is None):
        return _finish(ctx, res, "refused", [
            "%s costs GPU time; it needs a known estimate and a campaign envelope"
            % req["action"]])
    resources = ctx.resources()
    if not rendered["local"] and resources.get("cluster_reachable") is False:
        return _finish(ctx, res, "refused", ["the cluster is not reachable"])
    why = _resources_known(resources, costed)
    if why:
        return _finish(ctx, res, "refused", [why])
    bud = _budget(ctx, camp, now)
    basis = item.get("decision_basis") or "approval"
    res["basis"], res["decided_by"] = basis, item.get("decided_by")
    if basis == "envelope":
        if item.get("decided_by") not in AP.ENVELOPE_GRANTEES:
            return _finish(ctx, res, "refused", ["approval %s has an envelope basis but was "
                                                 "granted by %r" % (item_id, item.get("decided_by"))])
        auto, why, _grant = _autonomy(ctx, req, camp, row, est, bud, resources, item=item)
        if not auto:
            return _finish(ctx, res, "refused", ["the envelope rule no longer holds: " + w
                                                 for w in why])
        authority = camp["autonomy_granted_by"]
    else:
        authority = item.get("decided_by")
        if not _is_human(authority):
            return _finish(ctx, res, "refused", ["approval %s was decided by %r, not a person"
                                                 % (item_id, authority)])
    auth = POL.authorize(authority, req["action"], req["params"], B.budget_state(bud), resources)
    res["authorized_as"] = authority
    if not auth["allowed"]:
        return _finish(ctx, res, "refused", auth["reasons"])
    if auth["needs_approval"]:
        return _finish(ctx, res, "refused", ["budget: " + "; ".join(auth["reasons"])])
    if costed:
        ok, why = B.fits(bud, est["su"])
        if not ok:
            return _finish(ctx, res, "refused", ["budget: " + r for r in why])
    if rendered["local"]:
        if not callable(ctx.local_hooks.get(req["action"])):
            return _finish(ctx, res, "refused", ["no lab hook is registered for %s" % req["action"]])
    elif not callable(ctx.slurm_sh):
        return _finish(ctx, res, "refused", ["no slurm_sh hook: the executor cannot reach the "
                                             "cluster"])
    try:
        plan = _plan(res, req["action"], req["params"], rendered, ctx)
    except ExecError as e:
        return _finish(ctx, res, "refused", [str(e)])
    unwritable = _log_writable(ctx)
    if unwritable:
        return _finish(ctx, res, "refused", [unwritable])
    claim = AP.record_executed(ctx.domain, item_id, "started", invoked_by, ctx.clock(),
                               root=ctx.approvals_root)
    if not claim.get("ok"):
        return _finish(ctx, res, "refused", [claim.get("reason") or "could not claim the item"])
    out = _run_plans(ctx, [plan])[0]
    # A call that certainly never reached the cluster (a connection that was
    # never made) did not use the approval: it is released and stays
    # executable. Anything that ran, or may have run, closes it.
    phase = "done" if out["status"] == "executed" else "released" if never_ran(out) else "failed"
    outcome = {"status": out["status"], "job_ids": out["job_ids"],
               "error": (out.get("remote") or {}).get("error")
               or ("; ".join(out.get("reasons") or []) or None),
               "authorized_as": authority, "basis": basis, "run_id": out.get("run_id")}
    rec = AP.record_executed(ctx.domain, item_id, phase, invoked_by, ctx.clock(),
                             outcome=outcome, root=ctx.approvals_root)
    if phase == "released":
        out["released"] = bool(rec.get("ok"))
        if not rec.get("ok"):
            # Never left open: an execution with no outcome reads as "may have run".
            AP.record_executed(ctx.domain, item_id, "failed", invoked_by, ctx.clock(),
                               outcome=outcome, root=ctx.approvals_root)
            out.setdefault("warnings", []).append(
                "approval %s could not be released (%s); it is closed as failed"
                % (item_id, rec.get("reason")))
    return out


def run_approved(campaign, ctx=None, invoked_by=M.AUTOPILOT_ACTOR):
    """Execute every approved, not yet executed item filed for this campaign."""
    ctx = ctx or Context()
    camp = _campaign(campaign)
    out = []
    for item in AP.awaiting_execution(ctx.domain, root=ctx.approvals_root):
        if (item.get("context") or {}).get("campaign") != camp["name"]:
            continue
        out.append(execute_approved(item["id"], camp, ctx, invoked_by=invoked_by,
                                    quiet_repeat=True))
    return out


def budget_now(campaign, ctx=None):
    """The campaign's budget state as the executor sees it (for the ticker and the page)."""
    ctx = ctx or Context()
    return _budget(ctx, _campaign(campaign), ctx.clock())


def main(argv=None):
    """`record-replay`: run the replay and governance tests and record the result
    the envelope rule reads; `replay-status`: print what it reads now."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["record-replay"]:
        rec = run_replay_tests()
        print(json.dumps({k: rec[k] for k in ("status", "code_hash", "cases", "recorded_utc")},
                         indent=1, sort_keys=True))
        print(json.dumps((rec.get("detail") or {}).get("notes") or [], indent=1))
        return 0 if rec["status"] == "pass" else 1
    if argv[:1] == ["replay-status"]:
        st = replay_status()
        print(json.dumps({k: st[k] for k in ("passed", "reason", "code_hash_now", "path")},
                         indent=1, sort_keys=True))
        return 0 if st["passed"] else 1
    print("usage: python -m weed_optimizer_framework.tools.inc_autopilot.executor "
          "record-replay | replay-status", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())


# --- lab hooks of the funnel audit (docs/FUNNEL_AUDIT.md 8.5, runner 6.3) --------------------
# The campaign registers these in Context.local_hooks; each takes the action's
# params and returns {"ok": bool, ...}. `runner` is subprocess.run (tests
# inject a fake); nothing here runs at import.

FUNNEL_MODULE = "weed_optimizer_framework.tools.funnel"
# The fixed list the lab pushes to the cluster (runner 6.3, lab -> cluster),
# INC_DIR-relative; a directory entry ends in "/". prereg_v1.json and
# census_v0.json reach the cluster through git, not through the sync.
FUNNEL_SYNC_FILES = ("funnel/taxonomy_cache.json", "funnel/known_items_v1.json", "funnel/fetch_manifest.json",
                     "funnel/cards/", "funnel/kt7/", "funnel/refetch/", "funnel/prospective_da.json",
                     "funnel/rl_answers/RL-A/")
# The fixed list the lab pulls back from the cluster (runner 6.3, cluster ->
# lab): the aggregates, the qualification files, the sample and the sheets the
# lab's reference labeller answers (F7b), and the recovery record. Nothing
# else moves: no key file, no cluster-only sheet, no row-level ledger, no gold,
# no embedding or judge score, nothing under step1/, no evaluation image.
FUNNEL_PULL_FILES = ("funnel/census_v1.json", "funnel/name_status_v2.json", "funnel/funnel_ledger.json",
                     "funnel/audit_v1.json", "funnel/audit_v1.md", "funnel/class_maps.json",
                     "funnel/relation_geometry_v1.json", "funnel/relation_audit_v1.json",
                     "funnel/judge_qualification.json", "funnel/rl_qualification.json", "funnel/leak_v1.json",
                     "funnel/frames_v1.json", "funnel/sample_v1.csv", "funnel/sheets_v1/", "step1_r1/recovery.json")
FUNNEL_PULL_RECOVERY = "step1_r1/recovery.json"      # shipped to the evidence as funnel/recovery.json
# The pre-registration comes back only when the cluster's copy has grown by
# amendments (the sample lock) over the same core; a person commits it.
FUNNEL_PREREG_REL = "funnel/prereg_v1.json"


def funnel_fetch_argv(params, python=None):
    """The funnel CLI's fetch command of an inc_funnel_fetch request (L11a, L12)."""
    argv = [python or sys.executable, "-m", FUNNEL_MODULE, "fetch", "--prereg", str(params["prereg"]),
            "--what", str(params["what"])]
    if params.get("names_from"):
        argv += ["--names-from", str(params["names_from"])]
    return argv + ["--out", str(params["out"])]


def funnel_fetch_hook(runner=None, python=None, cwd=None, timeout=3600):
    """The lab hook of inc_funnel_fetch: runs the funnel CLI's fetch (it writes
    hashed files and fetch_manifest.json under the lab's funnel/)."""
    def hook(params):
        argv = funnel_fetch_argv(params, python)
        try:
            p = (runner or subprocess.run)(argv, cwd=str(cwd or CODE_ROOT), capture_output=True, text=True,
                                           timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "argv": argv}
        return {"ok": p.returncode == 0, "rc": p.returncode, "argv": argv,
                "tail": (p.stdout or "")[-1500:], "stderr_tail": (p.stderr or "")[-500:],
                "error": "" if p.returncode == 0 else "fetch exited %d" % p.returncode}
    return hook


def funnel_sync_list(lab_inc):
    """[INC_DIR-relative path] of the lab's files the sync may push: every file
    under an entry of FUNNEL_SYNC_FILES that exists in `lab_inc`."""
    root = Path(lab_inc)
    out = []
    for rel in FUNNEL_SYNC_FILES:
        p = root / rel
        if rel.endswith("/"):
            if p.is_dir():
                out += sorted(str(f.relative_to(root)) for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            out.append(rel)
    return out


def funnel_sync_refusals(paths, allowed=FUNNEL_SYNC_FILES):
    """Paths outside the fixed list (runner 6.3: the sync refuses them)."""
    bad = []
    for x in paths:
        if ".." in Path(x).parts or not any(x == r or (r.endswith("/") and x.startswith(r)) for r in allowed):
            bad.append(x)
    return bad


def funnel_pull_refusals(paths):
    """Paths outside the pull list (runner 6.3, cluster -> lab)."""
    return funnel_sync_refusals(paths, FUNNEL_PULL_FILES)


def _sha_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def funnel_pull_local(lab_inc):
    """{INC_DIR-relative path: sha256} of the lab's copies of the pull list's
    files (and of the pre-registration), for the ticker's context: what the
    lab already holds of what the cluster sends back."""
    root = Path(lab_inc)
    out = {}
    for rel in FUNNEL_PULL_FILES + (FUNNEL_PREREG_REL,):
        p = root / rel
        if rel.endswith("/"):
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and not f.is_symlink():
                        out[str(f.relative_to(root))] = _sha_file(f)
        elif p.is_file():
            out[rel] = _sha_file(p)
    return out


# Cluster side of the pull's listing: stdin the pull list (entries ending in
# "/" are directories), argv the cluster INC_DIR. Prints the sha256 of every
# file the list names that exists (never a symlink), and the pre-registration's
# raw and core sha256 and amendment count. Reads nothing outside the list.
_PULL_LIST_PY = r"""
import hashlib, json, os, sys
inc = sys.argv[1]
entries = json.loads(sys.stdin.read())
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
files = {}
for rel in entries:
    p = os.path.join(inc, rel)
    if rel.endswith("/"):
        for root, dirs, fs in os.walk(p):
            dirs.sort()
            for f in sorted(fs):
                fp = os.path.join(root, f)
                if not os.path.islink(fp) and os.path.isfile(fp):
                    files[os.path.relpath(fp, inc)] = sha(fp)
    elif os.path.isfile(p) and not os.path.islink(p):
        files[rel] = sha(p)
prereg = None
pp = os.path.join(inc, "funnel", "prereg_v1.json")
if os.path.isfile(pp):
    raw = open(pp, "rb").read()
    obj = json.loads(raw.decode("utf-8"))
    core = {k: v for k, v in obj.items() if k != "amendments"}
    prereg = {"sha256": hashlib.sha256(raw).hexdigest(), "amendments": len(obj.get("amendments") or []),
              "core_sha256": hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False).encode("utf-8")).hexdigest()}
print("INCAP " + json.dumps({"verb": "funnel-pull-list", "files": files, "prereg": prereg}, sort_keys=True))
"""


def _prereg_state(path):
    """(core sha256, amendments) of a pre-registration file."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    core = {k: v for k, v in obj.items() if k != "amendments"}
    return (hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                           .encode("utf-8")).hexdigest(), list(obj.get("amendments") or []))


def _remote_py(repo, script, args):
    return ("cd %s/weed_llm_benchmark && source %s && conda activate bench && python -c %s %s"
            % (shlex.quote(repo), shlex.quote(CONDA_SH), shlex.quote(script),
               " ".join(shlex.quote(str(a)) for a in args)))


def funnel_pull(lab_inc, target, cluster_inc=None, runner=None, repo=None, data_target=None):
    """Cluster -> lab (runner 6.3): list the pull list's files on the cluster
    with their sha256 (one ssh), rsync the ones the lab lacks or holds another
    version of into a staging directory, check every staged file against the
    listed sha256, and only then move them into place. The pre-registration
    comes back only when the cluster's copy has the lab's core and grew by
    amendments; another core is refused. Refuses any listed path outside the
    list. The listing runs over ssh on target; the copy runs rsync against
    data_target (the data-transfer node; default target). Returns {"ok",
    "pulled", "prereg", "error"}."""
    cluster_inc = cluster_inc or M.CLUSTER_INC_DIR
    repo = repo or M.CLUSTER_REPO
    run = runner or subprocess.run
    lab = Path(lab_inc)
    try:
        c = run(["ssh", target, _remote_py(repo, _PULL_LIST_PY, [cluster_inc])],
                input=json.dumps(list(FUNNEL_PULL_FILES)), capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "pulled": [], "error": "pull listing: %s: %s" % (type(e).__name__, e)}
    out = c.stdout or ""
    if c.returncode != 0 or "INCAP " not in out:
        return {"ok": False, "pulled": [], "error": "pull listing failed (exit %s): %s"
                % (c.returncode, (out + (c.stderr or ""))[-300:])}
    try:
        rec = json.loads(out.split("INCAP ", 1)[1].splitlines()[0])
        listed = dict(rec.get("files") or {})
    except (ValueError, IndexError, AttributeError) as e:
        return {"ok": False, "pulled": [], "error": "pull listing unreadable: %s" % e}
    bad = funnel_pull_refusals(listed)
    if bad:
        return {"ok": False, "pulled": [], "error": "the cluster listed paths outside the pull list: %s" % bad[:5]}
    want = {rel: sha for rel, sha in listed.items()
            if not (lab / rel).is_file() or _sha_file(lab / rel) != sha}
    prereg_note = "not on the cluster"
    cp = rec.get("prereg")
    lp = lab / FUNNEL_PREREG_REL
    if isinstance(cp, dict):
        if not lp.is_file():
            return {"ok": False, "pulled": [], "error": "the lab has no %s to compare the cluster's with"
                    % FUNNEL_PREREG_REL}
        core, amends = _prereg_state(lp)
        if cp.get("core_sha256") != core:
            return {"ok": False, "pulled": [], "error": "the cluster's %s has another core (%s, the lab's %s): a "
                    "different pre-registration; nothing is pulled" % (FUNNEL_PREREG_REL,
                                                                       str(cp.get("core_sha256"))[:12], core[:12])}
        if int(cp.get("amendments") or 0) > len(amends):
            want[FUNNEL_PREREG_REL] = cp.get("sha256")
            prereg_note = "amendments grew (%d -> %d): pulled for a person to commit" % (len(amends),
                                                                                      cp.get("amendments"))
        else:
            prereg_note = "unchanged"
    if not want:
        return {"ok": True, "pulled": [], "prereg": prereg_note, "error": ""}
    staging = lab / ".funnel_pull" / uuid.uuid4().hex
    staging.mkdir(parents=True)
    try:
        argv = ["rsync", "-a", "--files-from=-", "--", "%s:%s/" % (data_target or target, cluster_inc),
                str(staging) + "/"]
        try:
            p = run(argv, input="\n".join(sorted(want)) + "\n", capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "pulled": [], "error": "rsync: %s: %s" % (type(e).__name__, e)}
        if p.returncode != 0:
            return {"ok": False, "pulled": [], "error": "rsync exited %d: %s" % (p.returncode, (p.stderr or "")[-300:])}
        wrong = []
        for rel, sha in sorted(want.items()):
            f = staging / rel
            got = _sha_file(f) if f.is_file() else None
            if got != sha:
                wrong.append([rel, got, sha])
        if wrong:
            return {"ok": False, "pulled": [], "error": "changed in transit, nothing moved into place: %s" % wrong[:3]}
        if FUNNEL_PREREG_REL in want:
            core, amends = _prereg_state(lp)
            ncore, namends = _prereg_state(staging / FUNNEL_PREREG_REL)
            if ncore != core or namends[:len(amends)] != amends:
                return {"ok": False, "pulled": [], "error": "the cluster's pre-registration does not extend the "
                        "lab's by amendments alone; nothing moved into place"}
        for rel in sorted(want):
            dest = lab / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(staging / rel), str(dest))
        return {"ok": True, "pulled": sorted(want), "prereg": prereg_note, "error": ""}
    finally:
        import shutil
        shutil.rmtree(str(staging), ignore_errors=True)
        try:
            staging.parent.rmdir()
        except OSError:
            pass


# Cluster side of the sync's arrival check (runner 6.3: every transfer is
# verified on arrival). stdin: {INC_DIR-relative path: sha256} of what the lab
# pushed; argv: the cluster INC_DIR and "1" when a fetch manifest was pushed.
# Every pushed file must hash as it did on the lab (prospective_da.json and the
# RL-A answers are in no fetch manifest); then funnel.fetch.check_manifest
# checks the fetched files against fetch_manifest.json and rebuilds their
# machine-local tables. Exit 1 on any mismatch.
_SYNC_CHECK_PY = r"""
import hashlib, json, os, sys
inc, manifest = sys.argv[1], sys.argv[2] == "1"
want = json.loads(sys.stdin.read())
bad = []
for rel, sha in sorted(want.items()):
    h = hashlib.sha256()
    try:
        with open(os.path.join(inc, rel), "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        got = h.hexdigest()
    except OSError as e:
        got = "unreadable (%s)" % type(e).__name__
    if got != sha:
        bad.append([rel, got, sha])
rec = {"verb": "funnel-sync-check", "checked": len(want), "mismatched": bad, "manifest": None}
if manifest and not bad:
    try:
        from weed_optimizer_framework.tools.funnel import fetch
        fetch.check_manifest(os.path.join(inc, "funnel"))
        rec["manifest"] = "ok"
    except Exception as e:
        rec["manifest"] = "%s: %s" % (type(e).__name__, e)
        bad.append(["funnel/fetch_manifest.json", rec["manifest"], "fetch.check_manifest"])
print("INCAP " + json.dumps(rec, sort_keys=True))
sys.exit(1 if bad else 0)
"""


def funnel_sync_hook(lab_inc, target, cluster_inc=None, runner=None, repo=None, data_target=None):
    """The lab hook of inc_funnel_sync (runner 6.3), both directions over the
    dashboard's ssh target. Lab -> cluster: rsync the fixed file list from the
    lab's INC tree to the cluster's, then verify the arrival there
    (_SYNC_CHECK_PY): every pushed file hashes as on the lab, and when a fetch
    manifest was pushed, funnel.fetch.check_manifest (StaleInput on a
    mismatch). Cluster -> lab: funnel_pull (the pull list, verified before it
    is moved into place). Refuses a path outside either list. rsync runs
    against data_target (the cluster's data-transfer node: the login node has
    no rsync; default target), ssh commands against target."""
    cluster_inc = cluster_inc or M.CLUSTER_INC_DIR
    repo = repo or M.CLUSTER_REPO

    def hook(params):
        files = funnel_sync_list(lab_inc)
        bad = funnel_sync_refusals(files)
        if bad:
            return {"ok": False, "error": "outside the sync's fixed list: %s" % bad[:5]}
        if not target:
            return {"ok": False, "error": "no cluster ssh target (CLUSTER_SSH) to sync with"}
        res = push(files) if files else {"ok": True, "pushed": [], "note": "nothing to push"}
        if not res.get("ok"):
            return res
        pull = funnel_pull(lab_inc, target, cluster_inc, runner, repo, data_target=data_target)
        res.update(pulled=pull.get("pulled") or [], prereg=pull.get("prereg"))
        if not pull.get("ok"):
            res.update(ok=False, error="pull: %s" % pull.get("error"))
        return res

    def push(files):
        shas = {rel: hashlib.sha256((Path(lab_inc) / rel).read_bytes()).hexdigest() for rel in files}
        run = runner or subprocess.run
        argv = ["rsync", "-a", "--files-from=-", "--", str(Path(lab_inc)) + "/",
                "%s:%s/" % (data_target or target, cluster_inc)]
        try:
            p = run(argv, input="\n".join(files) + "\n", capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "error": "rsync: %s: %s" % (type(e).__name__, e), "files": files}
        if p.returncode != 0:
            return {"ok": False, "error": "rsync exited %d: %s" % (p.returncode, (p.stderr or "")[-300:]),
                    "files": files}
        manifest = "1" if "funnel/fetch_manifest.json" in shas else "0"
        check = ("cd %s/weed_llm_benchmark && source %s && conda activate bench && python -c %s %s %s"
                 % (shlex.quote(repo), shlex.quote(CONDA_SH), shlex.quote(_SYNC_CHECK_PY),
                    shlex.quote(cluster_inc), manifest))
        try:
            c = run(["ssh", target, check], input=json.dumps(shas, sort_keys=True), capture_output=True, text=True,
                    timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "error": "arrival check: %s: %s" % (type(e).__name__, e), "files": files}
        return {"ok": c.returncode == 0, "pushed": files, "rc": c.returncode, "sha256": shas,
                "error": "" if c.returncode == 0 else "the arrival check on the cluster failed: %s"
                % ((c.stdout or "") + (c.stderr or ""))[-400:]}
    return hook


def write_verify_queue(rows, path):
    """Append verify tasks (lever L14) to the person's queue file, once each
    (a task's id is the sha256 of its content); returns {"ok", "added", "path"}."""
    path = Path(path)
    have = set()
    if path.is_file():
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                have.add(json.loads(ln).get("task_id"))
            except ValueError:
                continue
    added = []
    for r in rows or []:
        task = dict(r, kind="class_map_verify", question="Does this source class hold the proposed class?")
        tid = hashlib.sha256(json.dumps(task, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        if tid in have:
            continue
        have.add(tid)
        added.append(dict(task, task_id=tid, queued_utc=M.utc_now()))
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for t in added:
                fh.write(json.dumps(t, sort_keys=True) + "\n")
    return {"ok": True, "added": len(added), "path": str(path)}
