"""INC autopilot, cluster side: the fixed verbs the lab calls over ssh
(docs/INC_AUTOPILOT.md, components 4 and 5).

Run on the Bridges-2 LOGIN node, in the bench env, from the git-tracked
nested copy (the lab batches every verb of a tick into one ssh, because the
login node throttles repeated connections):

    cd $REPO/weed_llm_benchmark && \\
    python -m weed_optimizer_framework.tools.inc_autopilot.remote VERB ...

Verbs. Each prints exactly one line "INCAP <json>" (model.REMOTE_MARK) on
stdout; everything else it or the INC modules print goes to stderr. Exit
status 0 when the record says ok, else 1.

    status
        every experiment under INC_DIR (type, done, generation, blocked
        units, chain phases, run counts, report staleness, abandoned), the
        inc_* jobs in squeue, the last build attempt of each provenance
        record, and "outer_drift": the outer package copy the jobs import
        against the git-tracked nested copy, per protected module (below).
    snapshot --exp X [--ledger-from N[:SHA256]] [--no-step1]
        the experiment's small files with the sha256 of each, split in two:
        "decision" (dev only; the only part decision code may read) and
        "display_only" (report.json's final table, every exam). Never opens
        runs/ (so never a scores/<exam>.json) and never ships a file over
        INCAP_MAX_FILE_BYTES; the Step 1 CSV and manifests are aggregated
        per source instead (below). --ledger-from N ships ledger lines N+1..
        with the sha256 of lines 1..N ("prefix_sha256"), so the lab can
        fetch only new lines; every read also returns "through_sha256", the
        sha256 of lines 1..next_line, which is the prefix_sha256 the next
        read from next_line must show. With N:SHA256 the check is made here:
        a prefix that does not hash to SHA256 (the file was rewritten) is
        re-read from line 0 and flagged "prefix_mismatch". The label audit
        is taken from <exp>/audit/*.json, or else from the path the pilot_v1
        audit was written to by hand, INC_DIR/audit/<exp>_audit.json, which
        is shipped as <exp>/audit/label_audit.json (its real path in files).
    advance --exp X
        the pinned driver's advance() (inc/driver.py), its result as data; a
        DriverError comes back as ok false with error_kind (code_drift,
        code_pin, not_built, testing_env, submit, locked, other). Refused
        (abandoned) after cancel, and refused as code_drift, before the
        driver runs, when the outer copy differs from the nested one in a
        protected module: this verb runs from the nested copy, where the
        driver's own drift check never compares, while every job the
        advance submits imports the outer copy and would fail on it.
    report --exp X
        inc.report.build(X): writes report.json / report.md in the experiment,
        returns only their sha256 and sizes (the values come back through
        snapshot, split as above).
    unblock --exp X --unit U --reason "auto: <cause>"
        lever L7: the driver's unblock of one blocked unit, only when the
        live state.json names a block cause on thresholds.json's
        D5.transient_cause_kinds, at most D5.max_auto_unblocks_per_unit
        times per unit (earlier unblocks whose reason starts with
        D5.auto_reason_prefix, or that the driver marked auto, count);
        refused for an abandoned experiment and on outer-copy drift.
    cancel --exp X [--approval-id ID] [--decided-by ACTOR] [--dry-run]
        policy action inc_cancel_exp (R3): first writes the abandonment
        marker INC_DIR/_campaign/abandoned/<exp>.json (who, approval, when),
        then scancels every job of the experiment in squeue (inc_<exp>_NNNN
        arrays, inc_build_<exp>, inc_audit_<exp>) and, after
        INCAP_CANCEL_SETTLE seconds, any array an in-job advance submitted
        meanwhile. The marker makes advance, campaign-snapshot --advance and
        unblock refuse the experiment; it cannot stop a person's own
        `driver watch` or `driver advance`, which must be stopped first. A
        person removes the marker to resume the experiment.
    sync-outer [--approval-id ID] [--decided-by ACTOR] [--dry-run]
        policy action inc_sync_outer (lever X5, R3): copy the nested package
        over the outer copy the INC jobs import, only changed files, deleting
        nothing. Refused while squeue shows any inc_* job that is not an
        experiment's run array (a build, audit, relevance, verify or plan job
        imports modules lazily mid-job), and refused when, for an unfinished
        experiment, it would change a protected module: a module the
        experiment pinned (state.code.modules) to anything but its pinned
        hash, or any other protected module at all.
    cancel and sync-outer (not dry runs) append who authorised them to
    INC_DIR/_campaign/provenance/_actions.jsonl.
    submit BUILDER [--parent-exp P] [--trigger D1,D3] [--approval-id ID]
                   [--decided-by ACTOR] [--dry-run] -- ARGS ...
        sbatch of run_inc_build.sh (BUILDER build: ARGS = pilot build ... |
        pilot build-baseline ... | realloop build ...; BUILDER pilot or
        realloop: the same without repeating the module, executor.render's
        form), run_inc_relevance.sh (relevance: ARGS = build ...) or
        run_inc_audit.sh (audit: ARGS = --trusted ... --audit ... --out ...).
        A lever's argv is accepted as written (a leading "sbatch <script>" or
        "python -m <module>" is dropped). ARGS are checked against the
        builder's own flag grammar (below), every path must lie under
        INC_DIR, and the parameters must be admitted by the menu (below).
        Returns the job id. The options before "--" go to the job as INCAP_*
        variables and into INC_DIR/_campaign/provenance/<exp>.json (written
        by run_inc_build.sh). Refused while a job of the same name is queued,
        or one under the script's own default name (inc_build, inc_audit: a
        submission by hand, whose experiment cannot be told); an audit is
        refused when one of that experiment already exists.
    campaign-snapshot --exp X [--exp Y ...] [--advance] [--report {auto,always,never}]
                      [--ledger-from X=N[:SHA256] ...] [--no-step1] [--funnel] [--sacct JOBID ...]
        several verbs in ONE call: per experiment advance (when asked, built
        and not abandoned), report (auto: when state.json says done and
        report.json is missing or older than state.json), snapshot; then
        Step 1 once and status last. One record, one line. --sacct adds
        sacct's record of the named jobs (job_states: state, elapsed, GPUs;
        no log is read), under "sacct": a job that ended is gone from squeue,
        and the campaign reads the final state of its funnel jobs there.
    funnel summary [--derive-ledger]
        the funnel audit's aggregates (docs/FUNNEL_AUDIT.md 8.2; runner
        5.5.6): funnel/{funnel_ledger, audit_v1, class_maps, prospective_da}.json,
        step1_r1/recovery.json (shipped as funnel/recovery.json),
        step1/{pool_summary, calibration}.json, the verifier fit record's
        projection (step1/verifier_fit_info.json, derived) and the listing of
        INC_DIR/funnel/ (funnel/files.json: names, sha256, sizes, never
        content), all through dev_only. Never ships conflicts.csv,
        pool_verdicts.npz, ledger.jsonl, key files, cluster sheets or
        evaluation descriptors (FUNNEL_NEVER). With --derive-ledger and no
        funnel_ledger.json on disk, the summaries-derived ledger computed in a
        temporary directory (derived.funnel_ledger). campaign-snapshot --funnel
        adds this record under "funnel".
    funnel ledger-summaries [--write]
        F2a: adapters.inc_step1.ledger_from_summaries on the Step 1 summaries
        and funnel/census_v0.json, returned; --write also writes it when no
        ledger file exists (a census-derived one is never replaced here).
    funnel dev-scores --exp E [--exp ...] [--truth-step STEP]
        runs/<run>/scores/dev.json of base runs (or of a real loop step's
        truth "with" runs), for inc_autopilot/panel.py; a score not stamped
        dev is refused.
    submit funnel [...] -- VERB FLAGS
        sbatch of run_inc_funnel.sh VERB (levers L10, L11 and L13): census,
        leak, embed-judges, qualify [--rl], draw, sheets, rl-b, ingest,
        estimate (--prereg, --out), map (--part geometry|relation), recover
        (--audit, --maps, --policy, --out under INC_DIR); the verb's extra
        sbatch flags come from funnel/__main__.py SBATCH_RESOURCES.
    stream-snapshot | stream-submit | stream-run
        stream mode (docs/CONTINUOUS_LOOP.md 6.2): see stream_remote.py.
    fixture --from FILE --out DIR
        (either host) the INCAP line in FILE (a snapshot or campaign-snapshot)
        written out as files named as on the cluster (relative to INC_DIR),
        dev only, with a MANIFEST.json of their sha256 (write_fixture). DIR is
        its own tree; files are then copied into tests/fixtures/inc_replay and
        pinned in that tree's MANIFEST.json (tests/test_inc_ap_fixtures.py).

Test blindness. Decision code must never read a test, ood22, ood23 or
imageweeds value (docs/INC_AUTOPILOT.md, (d)). The snapshot keeps report.json's
"final" table out of "decision" (it goes to display_only whole); the decision
part carries only its dev column ("derived.report_final_dev"). As a backstop,
every dict key named after a non-dev exam is dropped from "decision" and its
JSON pointer listed in "redacted". File sha256s are provenance, kept under
"files" outside "decision" (report.json's hash depends on its test values).

Step 1 per-source aggregates. select_clusters.csv has one row per verified
image and no source column; it is joined by key with base_selected.jsonl
and increment_pool.jsonl (select's own outputs) and counted per source and
set: images, status counts (no_feature, no_evidence, below_gate, selected,
...), species and OtherPlant boxes. The result is cached in
INC_DIR/_campaign/cache/ keyed by the inputs' size and mtime.

The menu. submit reads levers.json (component 3) and the policy table: rows
under "levers" (a dict id -> row, or a list of rows with "id"; a top-level
dict of rows also works), each with "policy_action" and "param_bounds" (or
"bounds") in policy_actions.json's format. The policy table is the
authority: an action without a valid policy_actions.json row is refused. A
request is admitted when at least one lever row of its policy action admits
every parameter, the row's bounds being its own param_bounds over its
policy_actions.json row's (the lever narrows what it declares, e.g. L1's
replay_mode ["full"]; the policy row bounds the rest, e.g. the audit paths);
a parameter neither declares is refused (policy._check_params); a value the
row's "fixed" names must be the value. The policy row alone must also admit
what it declares. A flag the command leaves out is checked at the builder's
own default (BUILDER_DEFAULTS: pilot build runs --replay-mode sample when
none is given, both builds --gate-flips-mode negative, and realloop build
--increment-sources relevance) wherever a bound declares it, so leaving a
flag out cannot pass a value its bound refuses (L9 fixes --gate-flips-mode
net: a pilot build without the flag is not L9). --increment-sources is an
enum of select.SOURCE_MODES (relevance, evidence); 'evidence' with
--relevance is refused (the builder refuses the pair), and --min-evidence
is not accepted (not on the menu: the protocol's default). Parameter names are the builder flags
without the dashes, "-" as "_" (--replay-mode -> replay_mode; --no-truth ->
no_truth, value 1, absent 0; the audit list comma-joined, as the policy row
writes it).

Protected modules (sync-outer, advance, unblock, status): inc/driver.py
PINNED_MODULES (pinned per experiment) and inc/train.py CODE_MODULES (every
run records their hashes and fails when the copy it runs differs from the
nested one), read from those modules, not copied here.

The builder grammar (flags each builder accepts) is a fact of the builders'
CLIs (inc/pilot.py, inc/realloop.py, inc/relevance.py, inc/audit.py), not a
tunable; production only: --testing, --testing-settings and --force are not
accepted.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import model as M

FORMAT = "inc_autopilot.remote/1"
AGGREGATE_FORMAT = "inc-autopilot/select-clusters-by-source/1"      # evidence.AGGREGATE_FORMAT
STATE_READ_BYTES = 64 << 20          # state.json is read up to this size: only its compact form is shipped
PROVENANCE_FORMAT = "inc_autopilot.provenance/1"
ABANDONED_FORMAT = "inc_autopilot.abandoned/1"
# The domain config's non-decision exams (common.EVAL_SPLITS minus dev) and
# every split no decision may read (those plus its extra non-decision splits,
# the weed domain's H10d domain dev): model.exam_splits / model.non_dev_exams.
NON_DEV_EXAMS = M.exam_splits()["non_decision"]
BLOCKED_SPLITS = M.non_dev_exams()
AUTO_PREFIX = "auto:"               # thresholds.json D5.auto_reason_prefix (tests check they agree); unblock reads the file
THRESHOLDS_JSON = Path(__file__).with_name("thresholds.json")   # D5: what L7 may unblock (diagnose.py, executor.py)
LEGACY_AUDIT = "audit/%s_audit.json"    # where pilot_v1's label audit was written by hand (INC_DIR-relative)
LABEL_AUDIT = "%s/audit/label_audit.json"   # the name evidence.py and diagnose.py read (INCREMENTAL_PROTOCOL.md:127)
UNCERTIFIED = ("no_feature", "no_evidence", "below_gate")   # select.py statuses without domain evidence
UNMATCHED = "<no source: key in neither manifest>"
DRIFT_ENVS = ("INC_ALLOW_DRIFT", "INC_BUILD_ALLOW_DRIFT", "INC_AUDIT_ALLOW_DRIFT",
              "INC_RELEVANCE_ALLOW_DRIFT", "INC_VERIFY_ALLOW_DRIFT")

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")          # experiment names (policy pattern)
AUDIT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")    # inc/audit.py NAME_RE, bounded
TRIGGER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
APPROVAL_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
ACTOR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._:+-]{0,127}\Z")
UNIT_RE = re.compile(r"(base|truth|final|chain:[A-Za-z0-9][A-Za-z0-9_-]*)\Z")
BAD_TOKEN_RE = re.compile(r"[\x00-\x20\x7f;&|`$<>'\"\\*?!#(){}]")
INT_RE = re.compile(r"[1-9][0-9]{0,8}\Z")
INT0_RE = re.compile(r"(0|[1-9][0-9]{0,8})\Z")
SEEDS_RE = re.compile(r"(0|[1-9][0-9]{0,8})(,(0|[1-9][0-9]{0,8}))*\Z")
MODULE_RE = re.compile(r"(?:(?:weed_optimizer_framework\.)?tools\.)?(?:inc\.)?(pilot|realloop|relevance|audit)\Z")
PYTHON_RE = re.compile(r"python[0-9.]*\Z")
RELEVANCE_OUT_RE = re.compile(r"relevance[A-Za-z0-9_.-]*\.json\Z")

STEP1_JSON = ("select_summary.json", "admit_summary.json", "relevance.json")
STEP1_BIG = ("select_clusters.csv", "base_selected.jsonl", "increment_pool.jsonl")

# The job script per submit builder, and the policy action of each builder form
# (docs/INC_AUTOPILOT.md (g) step 6).
SCRIPTS = {"build": "run_inc_build.sh", "relevance": "run_inc_relevance.sh", "audit": "run_inc_audit.sh",
           "funnel": "run_inc_funnel.sh"}
# The #SBATCH --job-name each script runs under when submitted by hand: remote.py
# names its jobs per experiment, but a hand submission of the same experiment
# carries only this name, so it counts as a possible duplicate.
SCRIPT_JOB_NAMES = {"build": "inc_build", "relevance": "inc_relevance", "audit": "inc_audit",
                    "funnel": "inc_funnel"}
BUILDER_ALIASES = {"pilot": "build", "realloop": "build"}      # executor.render's 'submit pilot|realloop ...'
ACTIONS = {("build", "pilot", "build"): "inc_build_pilot",
           ("build", "pilot", "build-baseline"): "inc_build_baseline",
           ("build", "realloop", "build"): "inc_build_realloop",
           ("relevance", "relevance", "build"): "inc_relevance_build",
           ("audit", "audit", None): "inc_label_audit"}
# The funnel audit's cluster jobs (docs/FUNNEL_AUDIT.md 8.5; runner 5.6): run_inc_funnel.sh VERB.
FUNNEL_AUDIT_VERBS = ("census", "leak", "embed-judges", "qualify", "draw", "sheets", "rl-b", "ingest", "estimate")
for _v in FUNNEL_AUDIT_VERBS:
    ACTIONS[("funnel", "funnel", _v)] = "inc_funnel_audit"
ACTIONS[("funnel", "funnel", "map")] = "inc_funnel_map"
ACTIONS[("funnel", "funnel", "recover")] = "inc_funnel_recover"
FUNNEL_PARTS = ("geometry", "relation")
FUNNEL_POLICY_RE = re.compile(r"R-[ACTVJF](,R-[ACTVJF]){0,5}\Z")
# flag -> (param, kind). Kinds: name, replay_mode, recipes, flips_mode, increment_sources, int (>= 1),
# int0 (>= 0), seeds, flag, path_in (an existing file), dir_in (an existing dir),
# out_json, audit_list (NAME=MANIFEST ...). Paths are PATH_KINDS.
FORMS = {
    "inc_build_pilot": {"flags": {"--exp": ("exp", "name"), "--replay-mode": ("replay_mode", "replay_mode"),
                                  "--gate-flips-mode": ("gate_flips_mode", "flips_mode")},
                        "required": ("--exp",)},
    "inc_build_baseline": {"flags": {"--exp": ("exp", "name"), "--manifest": ("manifest", "path_in"),
                                     "--seeds": ("seeds", "seeds")},
                           "required": ("--exp", "--manifest")},
    "inc_build_realloop": {"flags": {"--exp": ("exp", "name"), "--replay-mode": ("replay_mode", "replay_mode"),
                                     "--recipes": ("recipes", "recipes"), "--base": ("base", "path_in"),
                                     "--n-verified": ("n_verified", "int"), "--size": ("size", "int"),
                                     "--no-truth": ("no_truth", "flag"), "--relevance": ("relevance", "path_in"),
                                     "--increment-sources": ("increment_sources", "increment_sources"),
                                     "--step1-overlay": ("step1_overlay", "dir_in"),
                                     "--gate-flips-mode": ("gate_flips_mode", "flips_mode")},
                           "required": ("--exp", "--replay-mode", "--recipes")},
    "inc_relevance_build": {"flags": {"--sample": ("sample", "int"), "--seed": ("seed", "int0"),
                                      "--base-dir": ("base_dir", "dir_in"), "--out": ("out", "out_json")},
                            "required": ()},
    "inc_label_audit": {"flags": {"--trusted": ("trusted", "path_in"), "--audit": ("audit", "audit_list"),
                                  "--out": ("out", "out_json"), "--nshards": ("nshards", "int")},
                        "required": ("--trusted", "--audit", "--out")},
}
# The funnel builder's grammar (run_inc_funnel.sh VERB FLAGS: a positional verb,
# then flags), apart from FORMS, whose builders all name a module or command.
FUNNEL_FORMS = {
    "inc_funnel_audit": {"flags": {"--prereg": ("prereg", "path_in"), "--out": ("out", "dir_in"),
                                   "--rl": ("rl", "flag")},
                         "required": ("--prereg", "--out")},
    "inc_funnel_map": {"flags": {"--prereg": ("prereg", "path_in"), "--out": ("out", "dir_in"),
                                 "--part": ("part", "funnel_part")},
                       "required": ("--prereg", "--out", "--part")},
    "inc_funnel_recover": {"flags": {"--prereg": ("prereg", "path_in"), "--audit": ("audit", "path_in"),
                                     "--maps": ("maps", "path_in"), "--policy": ("policy", "funnel_policy"),
                                     "--out": ("out", "dir_out")},
                           "required": ("--prereg", "--audit", "--maps", "--policy", "--out")},
}
PATH_KINDS = ("path_in", "dir_in", "out_json", "audit_list", "dir_out")
# What each builder runs when a flag is left out (their argparse defaults; tests
# check each against its module). A value that depends on the inputs (realloop
# --size, --base, --relevance; relevance --out, --base-dir) is not listed: those
# are paths under INC_DIR or a share of the base, and no lever bounds them to a
# value the default could miss. --no-truth absent is 0.
REALLOOP_N_VERIFIED = 6        # inc/realloop.py N_VERIFIED
# realloop build --increment-sources: select.SOURCE_MODES and its default,
# select.SOURCES_RELEVANCE (tests check both against the module). 'evidence'
# reads no relevance file, so it never goes with --relevance (the builder
# refuses the two together; submit refuses them first).
INCREMENT_SOURCES = ("relevance", "evidence")
INCREMENT_SOURCES_DEFAULT = "relevance"
INCREMENT_SOURCES_EVIDENCE = "evidence"
# realloop build's own choices (realloop.INCREMENT_SOURCE_MODES): select's two
# criteria and the funnel's recovered overlay (realloop_v2, --step1-overlay).
INCREMENT_SOURCES_RECOVERED = "recovered"
REALLOOP_INCREMENT_SOURCES = INCREMENT_SOURCES + (INCREMENT_SOURCES_RECOVERED,)
RELEVANCE_SAMPLE = 300         # inc/relevance.py SAMPLE
RELEVANCE_SEED = 0             # inc/relevance.py build --seed default


def builder_defaults(action):
    D = _D()
    return {"inc_build_pilot": {"replay_mode": D.DEFAULT_REPLAY_MODE, "gate_flips_mode": D.DEFAULT_FLIPS_MODE},
            "inc_build_baseline": {"seeds": ",".join(str(s) for s in D.SEEDS)},
            "inc_build_realloop": {"n_verified": REALLOOP_N_VERIFIED, "no_truth": 0,
                                   "gate_flips_mode": D.DEFAULT_FLIPS_MODE,
                                   "increment_sources": INCREMENT_SOURCES_DEFAULT},
            "inc_relevance_build": {"sample": RELEVANCE_SAMPLE, "seed": RELEVANCE_SEED},
            "inc_label_audit": {}}.get(action, {})


class Refused(Exception):
    """A request remote.py will not carry out; the message says why."""


# ------------------------------------------------------------------ plumbing
def _C():
    from ..inc import common as C
    return C


def _D():
    from ..inc import driver as D
    return D


def inc_dir():
    return Path(_C().INC_DIR)


def campaign_dir():
    """INC_DIR/_campaign: provenance/<exp>.json and cache/ (model.CLUSTER_CAMPAIGN_DIR on the cluster)."""
    return inc_dir() / "_campaign"


def provenance_path(exp):
    return campaign_dir() / "provenance" / ("%s.json" % exp)


def actions_log_path():
    """Who authorised each R3 cluster action that is not a build (cancel,
    sync-outer): one JSON line per call, beside the builds' provenance."""
    return campaign_dir() / "provenance" / "_actions.jsonl"


def abandoned_path(exp):
    return campaign_dir() / "abandoned" / ("%s.json" % exp)


def abandoned(exp):
    """The abandonment marker cancel wrote for exp, or None. The file's
    existence is what counts: an unreadable marker still abandons."""
    p = abandoned_path(exp)
    if not p.exists():
        return None
    rec = _read_json_or_none(p)
    return rec if isinstance(rec, dict) else {"exp": exp, "unreadable": True}


def script_dir():
    """Where the job scripts are: the root of the package copy this runs from (on the
    cluster the nested, git-tracked $REPO/weed_llm_benchmark); INCAP_SCRIPT_DIR overrides."""
    return Path(os.environ.get("INCAP_SCRIPT_DIR") or Path(__file__).resolve().parents[3])


def levers_path():
    return Path(os.environ.get("INCAP_LEVERS_JSON") or Path(__file__).with_name("levers.json"))


def _max_file_bytes():
    return int(os.environ.get("INCAP_MAX_FILE_BYTES", str(4 << 20)))


def _max_ledger_bytes():
    return int(os.environ.get("INCAP_MAX_LEDGER_BYTES", str(8 << 20)))


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _short(text, n=400):
    text = str(text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


def _write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def code_info():
    here = Path(__file__).resolve()
    pkg = here.parents[2]
    nested = (_C().REPO / "weed_llm_benchmark" / "weed_optimizer_framework")
    try:
        is_nested = nested.resolve() == pkg
    except OSError:
        is_nested = False
    return {"package_dir": str(pkg), "nested": is_nested, "remote_sha256": _sha256_file(here)}


def base_record(verb):
    return {"verb": verb, "ok": True, "format": FORMAT, "utc": M.utc_now(), "host": socket.gethostname(),
            "inc_dir": str(inc_dir()), "code": code_info()}


def _user():
    try:
        return os.environ.get("USER") or getpass.getuser()
    except (OSError, KeyError):
        return None


def _append_line(path, obj):
    """One JSON line appended in a single write (O_APPEND)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, sort_keys=True, default=str) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def log_action(rec, meta, detail):
    """Append one record of an R3 action (cancel, sync-outer) to
    actions_log_path(); a failure to write is reported in rec, not raised."""
    entry = {"format": PROVENANCE_FORMAT, "verb": rec.get("verb"), "utc": rec.get("utc"), "host": rec.get("host"),
             "user": _user(), "exp": rec.get("exp"), "ok": bool(rec.get("ok")), "error": rec.get("error"),
             "approval_id": meta.get("approval_id"), "decided_by": meta.get("decided_by"), "detail": detail}
    try:
        _append_line(actions_log_path(), entry)
        rec["action_log"] = str(actions_log_path())
    except OSError as e:
        rec["action_log_error"] = "%s: %s" % (type(e).__name__, _short(e, 200))


# ------------------------------------------------------ protected code
def protected_modules():
    """The modules whose outer copy must equal the nested one while an
    experiment is unfinished: driver.PINNED_MODULES (pinned per experiment)
    and train.CODE_MODULES (every run hashes them and fails on drift)."""
    from ..inc import train as T
    return tuple(sorted(set(_D().PINNED_MODULES) | set(T.CODE_MODULES)))


def package_copies():
    """(nested git-tracked copy, outer copy the INC jobs import)."""
    repo = Path(_C().REPO)
    return repo / "weed_llm_benchmark" / "weed_optimizer_framework", repo / "weed_optimizer_framework"


def outer_drift(modules=None):
    """The outer copy against the nested one, per protected module, whichever
    copy this process runs from (the driver's own check compares only when the
    running copy is not the nested one, and remote.py runs from the nested
    one). {checked, drift: [module], modules: {module: {nested, outer}}}."""
    nested, outer = package_copies()
    out = {"nested_dir": str(nested), "outer_dir": str(outer), "checked": False, "drift": [], "modules": {}}
    if not nested.is_dir() or not outer.is_dir():
        out["why"] = "no %s copy at %s" % (("nested", nested) if not nested.is_dir() else ("outer", outer))
        return out
    for m in protected_modules() if modules is None else modules:
        a, b = nested / m, outer / m
        ns = _sha256_file(a) if a.is_file() else None
        os_ = _sha256_file(b) if b.is_file() else None
        if ns != os_:
            out["drift"].append(m)
            out["modules"][m] = {"nested": ns, "outer": os_}
    out["checked"] = True
    return out


def drift_message(d):
    # Worded to match thresholds.json D7.error_patterns ("differs from the git-tracked copy").
    return ("the outer package copy %s (what the INC jobs import) differs from the git-tracked copy %s in %s: "
            "a job would fail on it; a person syncs the outer copy (sync-outer, R3) or repins; nothing was done"
            % (d["outer_dir"], d["nested_dir"], d["drift"]))


def unblock_rules(path=None):
    """(transient cause kinds, max automatic unblocks per unit, reason prefix)
    from thresholds.json D5, the values diagnose.py and executor.py decide L7
    with. Raises Refused when the file or a key is missing: no compiled
    fallback, so the cluster side cannot drift from the pre-registered rule."""
    p = Path(path or THRESHOLDS_JSON)
    try:
        with open(p) as fh:
            d5 = json.load(fh)["D5"]
        kinds = d5["transient_cause_kinds"]["value"]
        most = d5["max_auto_unblocks_per_unit"]["value"]
        prefix = d5["auto_reason_prefix"]["value"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused("thresholds.json D5 unreadable at %s (%s: %s): no automatic unblock"
                      % (p, type(e).__name__, _short(e, 200)))
    if (not isinstance(kinds, list) or not all(isinstance(k, str) and k for k in kinds)
            or not isinstance(most, int) or isinstance(most, bool) or most < 0
            or not isinstance(prefix, str) or not prefix.strip()):
        raise Refused("thresholds.json D5 at %s is malformed (transient_cause_kinds %r, max_auto_unblocks_per_unit "
                      "%r, auto_reason_prefix %r): no automatic unblock" % (p, kinds, most, prefix))
    return tuple(kinds), most, prefix


def fail(rec, error, error_type="Refused", error_kind=None):
    rec.update(ok=False, error=str(error), error_type=error_type)
    if error_kind:
        rec["error_kind"] = error_kind
    return rec


# ------------------------------------------------------------ the marker line
def emit(rec, out=None):
    """Print rec as the one marked line; json.dumps escapes every newline."""
    out = out or sys.stdout
    out.write("%s %s\n" % (M.REMOTE_MARK, json.dumps(rec, sort_keys=True, separators=(",", ":"),
                                                     default=str)))
    out.flush()


def parse_marked(text):
    """Every "INCAP {json}" record in a remote command's output (login banners
    and other lines are ignored), in order."""
    out = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if line.startswith(M.REMOTE_MARK + " "):
            rec = json.loads(line[len(M.REMOTE_MARK) + 1:])
            if not isinstance(rec, dict):
                raise ValueError("an %s line holds %s, not an object" % (M.REMOTE_MARK, type(rec).__name__))
            out.append(rec)
    return out


def parse_one(text, verb=None):
    """The single record of one verb's output; raises ValueError when there is
    not exactly one, or it is another verb's."""
    recs = parse_marked(text)
    if len(recs) != 1:
        raise ValueError("expected exactly one %s line, found %d" % (M.REMOTE_MARK, len(recs)))
    if verb is not None and recs[0].get("verb") != verb:
        raise ValueError("expected verb %r, got %r" % (verb, recs[0].get("verb")))
    return recs[0]


# ------------------------------------------------------------- test blindness
def _ptr(parts):
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def dev_only(obj, _path=(), blocked=None):
    """(copy of obj without any dict key named after a non-dev split, [JSON
    pointers of what was dropped]). `blocked`: the split names (default
    BLOCKED_SPLITS; model.non_dev_exams(domain) for another domain)."""
    dropped = []
    names = BLOCKED_SPLITS if blocked is None else tuple(blocked)

    def walk(o, path):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k in names:
                    dropped.append(_ptr(path + (k,)))
                    continue
                out[k] = walk(v, path + (k,))
            return out
        if isinstance(o, list):
            return [walk(v, path + (i,)) for i, v in enumerate(o)]
        return o

    return walk(obj, tuple(_path)), dropped


def non_dev_keys(obj, blocked=None):
    """JSON pointers of every dict key named after a non-dev split (tests and the lab's check)."""
    return dev_only(obj, blocked=blocked)[1]


# --------------------------------------------------------------- small files
def read_small(path, cap=None):
    """(parsed JSON or None, info). info: sha256, bytes, shipped, why."""
    path = Path(path)
    st = path.stat()
    cap = _max_file_bytes() if cap is None else cap
    if st.st_size > cap:
        return None, {"bytes": st.st_size, "sha256": _sha256_file(path), "shipped": False,
                      "why": "larger than %d bytes (INCAP_MAX_FILE_BYTES)" % cap}
    data = path.read_bytes()
    info = {"bytes": len(data), "sha256": _sha256_bytes(data), "shipped": True}
    try:
        return json.loads(data.decode("utf-8")), info
    except (UnicodeDecodeError, ValueError) as e:
        info.update(shipped=False, why="not JSON: %s" % _short(e, 200))
        return None, info


def big_file_info(path):
    st = Path(path).stat()
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "shipped": False,
            "why": "large input; aggregated per source, never shipped"}


class _Collector:
    """What one snapshot gathers: files (provenance), decision artifacts,
    derived tables, omitted parts, display-only values, missing files."""

    def __init__(self):
        self.files, self.artifacts, self.derived, self.omitted = {}, {}, {}, {}
        self.display, self.missing, self.cache, self.ledger = {}, [], {}, None

    def take(self, rel, cap=None):
        p = inc_dir() / rel
        if not p.is_file():
            self.missing.append(rel)
            return None
        obj, info = read_small(p, cap)
        self.files[rel] = info
        return obj

    def put(self, rel, obj):
        if obj is not None:
            self.artifacts[rel] = obj


def summarize_runs(runs):
    by_owner = collections.defaultdict(collections.Counter)
    failed, live = [], set()
    for rid, r in sorted((runs or {}).items()):
        o, s = r.get("owner", "?"), r.get("status", "?")
        by_owner[o][s] += 1
        if s == "failed":
            failed.append({"run_id": rid, "owner": o, "attempt": r.get("attempt")})
        if s in ("submitting", "submitted") and r.get("job"):
            live.add(str(r["job"]).split("_")[0])
    return {"n_runs": len(runs or {}), "by_owner": {o: dict(sorted(c.items())) for o, c in sorted(by_owner.items())},
            "failed": failed, "live_job_ids": sorted(live)}


def compact_state(st):
    """state.json without runs{} and without each submission's task list (both
    summarised in derived.state_runs); every kept key sits at its own pointer."""
    out = {k: v for k, v in st.items() if k != "runs"}
    out["submissions"] = [{k: v for k, v in s.items() if k != "tasks"} for s in st.get("submissions") or []]
    return out


def report_final_dev(final):
    """The dev column of report.json's final table, nothing else of it."""
    rows = []
    for row in final or []:
        ex = (row.get("exams") or {}) if isinstance(row, dict) else {}
        rows.append({"model": row.get("model"), "runs": row.get("runs"),
                     "exams": {M.DECISION_EXAM: ex.get(M.DECISION_EXAM)}})
    return rows


def _lines_sha256(lines):
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln)
        h.update(b"\n")
    return h.hexdigest()


def read_ledger(path, rel, from_line=0, max_bytes=None, expect_prefix=None):
    """Ledger lines from_line+1 .. (1-based line numbers), up to max_bytes per
    call, with the sha256 of lines 1..from_line ("prefix_sha256") so the
    reader can check that its earlier copy is this file's prefix, and the
    sha256 of lines 1..next_line ("through_sha256"), the prefix_sha256 the
    next read from next_line must show. With expect_prefix (the reader's
    stored through_sha256) the check is made here: on a mismatch the ledger
    is read again from line 0 and "prefix_mismatch" says so, so a rewritten
    prefix is never spliced with new lines. A partial last line (an advance
    killed mid-append; the next advance repairs it) is not shipped."""
    max_bytes = _max_ledger_bytes() if max_bytes is None else max_bytes
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()                    # b"" when the file ends with a newline
    out = {"artifact": rel, "from_line": int(from_line), "n_lines": len(lines), "partial_tail": bool(tail.strip()),
           "entries": [], "complete": True}
    if from_line < 0 or from_line > len(lines):
        out.update(complete=False, next_line=len(lines),
                   error="from_line %d is outside the ledger's %d complete line(s); re-read it from 0"
                         % (from_line, len(lines)))
        return out
    out["prefix_sha256"] = _lines_sha256(lines[:from_line])
    if expect_prefix is not None and from_line and out["prefix_sha256"] != expect_prefix:
        out["prefix_mismatch"] = {"from_line": int(from_line), "expected": expect_prefix,
                                  "found": out["prefix_sha256"]}
        from_line = 0
        out.update(from_line=0, prefix_sha256=_lines_sha256([]))
    size = 0
    for i in range(from_line, len(lines)):
        raw = lines[i]
        if out["entries"] and size + len(raw) > max_bytes:
            out["complete"] = False
            break
        size += len(raw)
        try:
            out["entries"].append({"line": i + 1, "entry": json.loads(raw.decode("utf-8"))})
        except (UnicodeDecodeError, ValueError) as e:
            out["entries"].append({"line": i + 1, "entry": None, "error": "not JSON: %s" % _short(e, 120)})
    out["next_line"] = from_line + len(out["entries"])
    out["through_sha256"] = _lines_sha256(lines[:out["next_line"]])
    return out


# ------------------------------------------------------- Step 1 aggregation
def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def clusters_by_source(step1, recorded_sha256=None):
    """(aggregate, cache status) of step1/select_clusters.csv per source and
    set, or (None, None) without the CSV."""
    step1 = Path(step1)
    csv_p = step1 / "select_clusters.csv"
    if not csv_p.is_file():
        return None, None
    mans = [(step1 / "base_selected.jsonl", "selected"), (step1 / "increment_pool.jsonl", "increment_pool")]
    inputs = {}
    for p in [csv_p] + [m for m, _ in mans]:
        if p.is_file():
            st = p.stat()
            inputs[str(p)] = [st.st_size, st.st_mtime_ns]
    cache = campaign_dir() / "cache" / "select_clusters_by_source.json"
    try:
        with open(cache) as fh:
            c = json.load(fh)
        if c.get("format") == FORMAT and c.get("inputs") == inputs:
            result = dict(c["result"])
            result["sha256_matches_select_summary"] = (None if recorded_sha256 is None
                                                       else result["sha256"] == recorded_sha256)
            return result, "hit"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    where = {}
    for p, tag in mans:
        if p.is_file():
            with open(p) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        r = json.loads(ln)
                        where[r["key"]] = (str(r.get("source") or ""), tag)
    blank = lambda: {"images": 0, "status": collections.Counter(), "species_boxes": 0, "other_boxes": 0}  # noqa: E731
    by = collections.defaultdict(lambda: collections.defaultdict(blank))
    n = unmatched = 0
    with open(csv_p, newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            src, tag = where.get(row.get("key"), (None, None))
            if src is None:
                unmatched += 1
                src, tag = UNMATCHED, "unmatched"
            b = by[src][tag]
            b["images"] += 1
            b["status"][row.get("status") or ""] += 1
            b["species_boxes"] += _int(row.get("species_boxes"))
            b["other_boxes"] += _int(row.get("other_boxes"))
    sources = {}
    for src in sorted(by):
        sources[src] = {}
        for tag in sorted(by[src]):
            b = by[src][tag]
            sources[src][tag] = {"images": b["images"], "status": dict(sorted(b["status"].items())),
                                 "uncertified": sum(b["status"][s] for s in UNCERTIFIED),
                                 "species_boxes": b["species_boxes"], "other_boxes": b["other_boxes"]}
    sha = _sha256_file(csv_p)
    result = {"format": AGGREGATE_FORMAT, "artifact": "step1/select_clusters.csv", "sha256": sha,
              "sha256_matches_select_summary": (None if recorded_sha256 is None else sha == recorded_sha256),
              "joined_with": ["step1/base_selected.jsonl", "step1/increment_pool.jsonl"],
              "rows": n, "unmatched_keys": unmatched, "uncertified_statuses": list(UNCERTIFIED),
              "sources": sources}
    with contextlib.suppress(OSError):
        _write_json_atomic(cache, {"format": FORMAT, "inputs": inputs, "result": result})
    return result, "miss"


def collect_step1(col):
    d = inc_dir() / "step1"
    summary = None
    for name in STEP1_JSON:
        obj = col.take("step1/" + name)
        col.put("step1/" + name, obj)
        if name == "select_summary.json":
            summary = obj
    for name in STEP1_BIG:
        p = d / name
        if p.is_file():
            col.files["step1/" + name] = big_file_info(p)
            rec = ((summary or {}).get("outputs") or {}).get(name)
            if isinstance(rec, dict) and rec.get("sha256"):
                col.files["step1/" + name]["sha256_recorded_by_select"] = rec["sha256"]
        else:
            col.missing.append("step1/" + name)
    recorded = col.files.get("step1/select_clusters.csv", {}).get("sha256_recorded_by_select")
    agg, cache = clusters_by_source(d, recorded)
    if agg is not None:
        col.derived["select_clusters_by_source"] = agg
        col.cache["select_clusters_by_source"] = cache


def _finish(rec, col):
    decision = {"artifacts": col.artifacts, "derived": col.derived, "omitted": col.omitted}
    if col.ledger is not None:
        decision["ledger"] = col.ledger
    decision, redacted = dev_only(decision)
    rec.update(decision_exam=M.DECISION_EXAM, decision=decision, redacted=redacted, files=col.files,
               missing=col.missing, display_only=col.display, cache=col.cache)
    return rec


# ------------------------------------------------------------------- verbs
def _check_exp(exp):
    if not isinstance(exp, str) or not NAME_RE.match(exp):
        raise Refused("experiment %r is not a name ([A-Za-z0-9][A-Za-z0-9_-]{0,63})" % (exp,))
    return exp


def snapshot(exp, ledger_from=0, step1=True, ledger_sha256=None):
    rec = base_record("snapshot")
    try:
        _check_exp(exp)
    except Refused as e:
        return fail(rec, e, error_kind="bad_name")
    rec["exp"] = exp
    col = _Collector()
    root = inc_dir() / exp
    rec["built"] = (root / "exp.json").is_file()
    rec["abandoned"] = abandoned(exp)
    col.put(exp + "/exp.json", col.take(exp + "/exp.json"))

    rel = exp + "/state.json"
    st = col.take(rel, cap=max(_max_file_bytes(), STATE_READ_BYTES))     # shipped compacted (no runs{})
    if isinstance(st, dict):
        col.put(rel, compact_state(st))
        col.omitted[rel] = ["/runs", "/submissions/*/tasks"]
        col.derived["state_runs"] = summarize_runs(st.get("runs"))

    rel = exp + "/report.json"
    rep = col.take(rel)
    if isinstance(rep, dict):
        final = rep.get("final")
        col.put(rel, {k: v for k, v in rep.items() if k != "final"})
        col.omitted[rel] = ["/final"]
        if final is not None:
            col.display[rel + "#/final"] = final
            col.derived["report_final_dev"] = report_final_dev(final)

    for rel in (exp + "/build_summary.json", exp + "/manifests/increments_summary.json"):
        col.put(rel, col.take(rel))
    audit = root / "audit"
    if audit.is_dir():
        for p in sorted(audit.glob("*.json")):
            rel = "%s/audit/%s" % (exp, p.name)
            col.put(rel, col.take(rel))
    as_rel, legacy = LABEL_AUDIT % exp, LEGACY_AUDIT % exp
    if as_rel not in col.artifacts and (inc_dir() / legacy).is_file():
        # pilot_v1's audit was run by hand with --out INC_DIR/audit/<exp>_audit.json;
        # evidence and diagnose D3 read <exp>/audit/label_audit.json only.
        obj = col.take(legacy)
        col.files[legacy]["as"] = as_rel
        col.put(as_rel, obj)
    rel = "_campaign/provenance/%s.json" % exp
    col.put(rel, col.take(rel))

    rel = exp + "/ledger.jsonl"
    lp = inc_dir() / rel
    if lp.is_file():
        st_ = lp.stat()
        col.files[rel] = {"bytes": st_.st_size, "sha256": _sha256_file(lp), "shipped": True}
        col.ledger = read_ledger(lp, rel, from_line=int(ledger_from), expect_prefix=ledger_sha256)
    else:
        col.missing.append(rel)
    if step1:
        collect_step1(col)
    return _finish(rec, col)


def step1_snapshot():
    rec = base_record("step1")
    col = _Collector()
    collect_step1(col)
    return _finish(rec, col)


def _read_json_or_none(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def squeue_jobs(prefix="inc_"):
    cmd = os.environ.get("INCAP_SQUEUE", "squeue")
    user = os.environ.get("USER") or getpass.getuser()
    try:
        p = subprocess.run([cmd, "-h", "-u", user, "-o", "%i|%j|%T|%M|%V"], capture_output=True, text=True,
                           timeout=90)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, _short(e, 200))}
    if p.returncode != 0:
        return {"ok": False, "error": "squeue exited %d: %s" % (p.returncode, _short(p.stderr or p.stdout, 200))}
    jobs = []
    for ln in p.stdout.splitlines():
        parts = [x.strip() for x in ln.split("|")]
        if len(parts) >= 3 and parts[1].startswith(prefix):
            jobs.append({"id": parts[0], "name": parts[1], "state": parts[2],
                         "elapsed": parts[3] if len(parts) > 3 else None,
                         "submit": parts[4] if len(parts) > 4 else None})
    return {"ok": True, "jobs": jobs}


def exp_status(exp):
    root = inc_dir() / exp
    ab = abandoned(exp)
    e = {"exp": exp, "built": (root / "exp.json").is_file(),
         "abandoned": None if ab is None else {k: ab.get(k) for k in ("utc", "approval_id", "decided_by",
                                                                      "unreadable") if k in ab}}
    st = _read_json_or_none(root / "state.json")
    if not isinstance(st, dict):
        e["state"] = "missing or unreadable"
        return e
    runs = st.get("runs") or {}
    counts = collections.Counter(r.get("status", "?") for r in runs.values())
    e.update(type=st.get("type"), testing=bool(st.get("testing")), done=bool(st.get("done")),
             done_utc=st.get("done_utc"), generation=st.get("generation"), updated_utc=st.get("updated_utc"),
             blocked={u: {"utc": b.get("utc"), "cause": b.get("cause"), "error": _short(b.get("error"), 300)}
                      for u, b in sorted((st.get("blocked") or {}).items())},
             transient=sorted((st.get("transient") or {}).keys()),
             chains={r: {"phase": c.get("phase"), "k": c.get("k")}
                     for r, c in sorted((st.get("chains") or {}).items())},
             runs=dict(sorted(counts.items())), n_unblocks=len(st.get("unblocks") or []),
             n_repins=len(st.get("repins") or []))
    rp, sp = root / "report.json", root / "state.json"
    e["report"] = ("missing" if not rp.is_file() else
                   "stale" if rp.stat().st_mtime < sp.stat().st_mtime else "current")
    return e


def status():
    rec = base_record("status")
    d = inc_dir()
    exps = []
    if d.is_dir():
        for p in sorted(d.iterdir()):            # INC_DIR's top level only (never recursive)
            if p.is_dir() and NAME_RE.match(p.name) and (p / "exp.json").is_file():
                exps.append(exp_status(p.name))
    rec["experiments"] = exps
    builds = {}
    pdir = campaign_dir() / "provenance"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.json")):
            pr = _read_json_or_none(p)
            att = (pr or {}).get("attempts") or []
            last = att[-1] if att and isinstance(att[-1], dict) else {}
            builds[p.stem] = {k: last.get(k) for k in ("job_id", "status", "started_utc", "updated_utc",
                                                         "build_rc", "advance_rc", "refusal")}
            builds[p.stem]["attempts"] = len(att)
    rec["builds"] = builds
    rec["squeue"] = squeue_jobs()
    try:
        rec["outer_drift"] = outer_drift()
    except Exception as e:                      # a status never fails on it; advance refuses instead
        rec["outer_drift"] = {"checked": False, "drift": [], "error": "%s: %s" % (type(e).__name__, _short(e, 300))}
    return rec


def classify_driver_error(e):
    name, msg = type(e).__name__, str(e)
    if "differs from the git-tracked copy" in msg:
        return "code_drift"
    if "code changed since experiment" in msg:
        return "code_pin"
    if "build the experiment and run init first" in msg or "does not exist; run init" in msg:
        return "not_built"
    if "is marked testing" in msg:
        return "testing_env"
    if name in ("SubmitNotStarted", "SubmitUncertain") or "sbatch" in msg:
        return "submit"
    if name == "ConcurrentPass" or "another advance holds" in msg:
        return "locked"
    if "not path-safe" in msg:
        return "bad_name"
    return "other"


def _abandoned_error(exp, ab):
    return ("%s was abandoned (cancel at %s, approval %s, by %s; %s): nothing is advanced or unblocked; a person "
            "removes the marker to resume it" % (exp, ab.get("utc"), ab.get("approval_id"), ab.get("decided_by"),
                                                abandoned_path(exp)))


def _drift_guard(rec):
    """None when the outer copy matches the nested one in every protected
    module (or there is no outer copy to compare), else the refusal message.
    Sets rec["outer_drift"]."""
    d = outer_drift()
    rec["outer_drift"] = {"checked": d["checked"], "drift": d["drift"]}
    return drift_message(d) if d["drift"] else None


def advance(exp, backend=None):
    rec = base_record("advance")
    try:
        _check_exp(exp)
    except Refused as e:
        return fail(rec, e, error_kind="bad_name")
    rec["exp"] = exp
    ab = abandoned(exp)
    if ab is not None:
        return fail(rec, _abandoned_error(exp, ab), error_kind="abandoned")
    try:
        why = _drift_guard(rec)
    except Exception as e:                      # cannot tell: do not submit jobs that may fail on drift
        return fail(rec, "the outer-copy drift check failed: %s: %s" % (type(e).__name__, _short(e, 300)),
                    type(e).__name__, "other")
    if why:
        return fail(rec, why, "DriftError", "code_drift")
    D = _D()
    try:
        rec["result"] = D.advance(exp, backend=backend, quiet=True)
    except D.DriverError as e:
        fail(rec, e, type(e).__name__, classify_driver_error(e))
    except (OSError, ValueError, KeyError) as e:
        fail(rec, e, type(e).__name__, "io" if isinstance(e, OSError) else "other")
    return rec


def report(exp):
    rec = base_record("report")
    try:
        _check_exp(exp)
    except Refused as e:
        return fail(rec, e, error_kind="bad_name")
    rec["exp"] = exp
    from ..inc import report as R
    D = _D()
    try:
        rep = R.build(exp)
    except (OSError, ValueError, KeyError, D.DriverError) as e:
        return fail(rec, e, type(e).__name__, "not_built" if isinstance(e, FileNotFoundError) else "other")
    root = inc_dir() / exp
    rec.update(done=bool(rep.get("done")), testing=bool(rep.get("testing")), generated_utc=rep.get("generated_utc"),
               files={"%s/%s" % (exp, n): {"bytes": (root / n).stat().st_size, "sha256": _sha256_file(root / n)}
                      for n in ("report.json", "report.md") if (root / n).is_file()})
    return rec


def unblock(exp, unit, reason, backend=None):
    """Lever L7 on the live state.json: the block's own cause must be on
    thresholds.json D5.transient_cause_kinds (the lab decided on a snapshot
    that may be stale; a failed_run or gate block needs a person), and the
    unit may have had at most D5.max_auto_unblocks_per_unit automatic
    unblocks before (the driver's record: reason prefix, or its auto flag)."""
    rec = base_record("unblock")
    try:
        _check_exp(exp)
        rec.update(exp=exp, unit=unit)
        if not isinstance(unit, str) or not UNIT_RE.match(unit):
            raise Refused("unit %r is not base, truth, final or chain:<recipe>" % (unit,))
        kinds, most, prefix = unblock_rules()
        reason = str(reason or "").strip()
        if not reason.startswith(prefix) or len(reason) > 400 or "\n" in reason:
            raise Refused("an autopilot unblock reason starts with %r, is one line and at most 400 "
                          "characters" % prefix)
        ab = abandoned(exp)
        if ab is not None:
            return fail(rec, _abandoned_error(exp, ab), error_kind="abandoned")
        st = _read_json_or_none(inc_dir() / exp / "state.json")
        if not isinstance(st, dict):
            raise Refused("%s has no readable state.json" % exp)
        blocked = st.get("blocked") or {}
        if unit not in blocked:
            raise Refused("%s is not blocked (blocked: %s)" % (unit, sorted(blocked) or "none"))
        b = blocked[unit] if isinstance(blocked[unit], dict) else {}
        cause = b.get("cause") if isinstance(b.get("cause"), dict) else {}
        rec["cause"] = cause or None
        if cause.get("kind") not in kinds:
            raise Refused("%s is blocked by cause %r, not a transient one (thresholds.json D5.transient_cause_kinds "
                          "%s): an automatic unblock is refused, a person decides (block: %s)"
                          % (unit, cause.get("kind"), list(kinds), _short(b.get("error"), 200)))
        prior = [u for u in st.get("unblocks") or [] if isinstance(u, dict) and u.get("unit") == unit
                 and (u.get("auto") is True or str(u.get("reason", "")).startswith(prefix))]
        if len(prior) >= most:
            raise Refused("%s was already unblocked automatically (%d time(s), last at %s); at most %d automatic "
                          "unblock(s) per unit (a human decides the next)"
                          % (unit, len(prior), prior[-1].get("utc") if prior else None, most))
        try:
            why = _drift_guard(rec)
        except Exception as e:                  # cannot tell: do not resubmit runs that may fail on drift
            return fail(rec, "the outer-copy drift check failed: %s: %s" % (type(e).__name__, _short(e, 300)),
                        type(e).__name__, "other")
        if why:
            return fail(rec, why, "DriftError", "code_drift")
    except Refused as e:
        return fail(rec, e, error_kind="refused")
    D = _D()
    try:
        rec["result"] = D.Driver(exp, backend=backend, quiet=True).unblock([unit], reason)
    except D.DriverError as e:
        fail(rec, e, type(e).__name__, classify_driver_error(e))
    except (OSError, ValueError, KeyError) as e:
        fail(rec, e, type(e).__name__, "other")
    return rec


def exp_job_re(exp):
    """The Slurm job names that belong to an experiment: the driver's arrays
    (inc_<exp>_NNNN, driver.py _submit_queued), its build (inc_build_<exp>) and
    its audit (inc_audit_<exp>), both named by submit."""
    e = re.escape(exp)
    return re.compile(r"(inc_%s_[0-9]{4}|inc_build_%s|inc_audit_%s)\Z" % (e, e, e))


def _job_ids(jobs):
    return sorted({j["id"].split("_")[0] for j in jobs}, key=lambda s: (len(s), s))


def _scancel(ids):
    """(ok, rc or None, error text)."""
    cmd = os.environ.get("INCAP_SCANCEL", "scancel")
    try:
        p = subprocess.run([cmd] + list(ids), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, None, "scancel could not be run: %s: %s" % (type(e).__name__, _short(e, 200))
    if p.returncode != 0:
        return False, p.returncode, "scancel exited %d: %s" % (p.returncode, _short(p.stderr or p.stdout, 300))
    return True, 0, None


def _mark_abandoned(exp, meta, ids):
    """Write (or extend) INC_DIR/_campaign/abandoned/<exp>.json; raises OSError."""
    p = abandoned_path(exp)
    now = M.utc_now()
    this = {"utc": now, "approval_id": meta.get("approval_id"), "decided_by": meta.get("decided_by"),
            "user": _user(), "host": socket.gethostname(), "job_ids": list(ids)}
    rec = _read_json_or_none(p) if p.exists() else None
    if not isinstance(rec, dict):
        rec = {"format": ABANDONED_FORMAT, "exp": exp, "utc": now, "approval_id": this["approval_id"],
               "decided_by": this["decided_by"], "cancels": [],
               "effect": "remote.py advance, campaign-snapshot --advance and unblock refuse this experiment. A "
                         "person's own `driver watch` / `driver advance` is not stopped by this file and must be "
                         "stopped by hand. A person removes this file to resume the experiment."}
    rec.setdefault("cancels", []).append(this)
    _write_json_atomic(p, rec)
    return p


def cancel(exp, dry_run=False, meta=None):
    """Abandon the experiment and scancel every queued or running job of it
    (policy action inc_cancel_exp, R3). The abandonment marker is written
    first: the driver counts a cancelled task as a failed run and resubmits
    it on the next advance, and runs still queued in state.json would be
    submitted by it, so every advance this module makes is refused from then
    on. After the scancel, squeue is read again (INCAP_CANCEL_SETTLE seconds
    later, twice at most) and an array an in-job advance submitted meanwhile
    is cancelled too."""
    rec = base_record("cancel")
    try:
        _check_exp(exp)
    except Refused as e:
        return fail(rec, e, error_kind="bad_name")
    try:
        prov = _meta(meta or {})
    except Refused as e:
        return fail(rec, e, error_kind="refused")
    rec.update(exp=exp, dry_run=bool(dry_run), provenance=prov)
    q = squeue_jobs()
    if not q["ok"]:
        return fail(rec, "squeue unavailable (%s): nothing cancelled" % q["error"], error_kind="squeue")
    rx = exp_job_re(exp)
    jobs = [j for j in q["jobs"] if rx.match(j["name"])]
    ids = _job_ids(jobs)
    rec.update(jobs=jobs, job_ids=ids, late_job_ids=[])
    if dry_run:
        return rec
    try:
        rec["abandoned"] = str(_mark_abandoned(exp, prov, ids))
    except OSError as e:
        fail(rec, "the abandonment marker could not be written (%s: %s): nothing cancelled"
             % (type(e).__name__, _short(e, 200)), type(e).__name__, "io")
        log_action(rec, prov, {"job_ids": ids})
        return rec
    if ids:
        ok, rc, err = _scancel(ids)
        rec["scancel_rc"] = rc
        if not ok:
            fail(rec, err, "CancelFailed", "cancel")
            log_action(rec, prov, {"job_ids": ids, "abandoned": rec["abandoned"]})
            return rec
    settle = float(os.environ.get("INCAP_CANCEL_SETTLE", "3"))
    done = set(ids)
    for _ in range(2):
        if settle > 0:
            time.sleep(settle)
        q2 = squeue_jobs()
        if not q2["ok"]:
            rec["recheck_error"] = q2["error"]
            break
        late = [i for i in _job_ids([j for j in q2["jobs"] if rx.match(j["name"])]) if i not in done]
        if not late:
            break
        ok, rc, err = _scancel(late)
        done.update(late)
        rec["late_job_ids"] += late
        if not ok:
            fail(rec, "late jobs %s: %s" % (late, err), "CancelFailed", "cancel")
            break
    log_action(rec, prov, {"job_ids": ids, "late_job_ids": rec["late_job_ids"], "abandoned": rec["abandoned"]})
    return rec


def _tree_files(root):
    """{relative path: absolute path} of every file of a package copy, without
    __pycache__, compiled files and dotfiles (a small tree: ~160 files)."""
    out = {}
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not d.startswith("."))
        for f in sorted(filenames):
            if f.startswith(".") or f.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / f
            out[p.relative_to(root).as_posix()] = p
    return out


ARRAY_RE = re.compile(r"inc_(.+)_[0-9]{4}\Z")          # the driver's run arrays, inc_<exp>_NNNN


def sync_outer(dry_run=False, meta=None):
    """Copy the nested (git-tracked) weed_optimizer_framework onto the outer
    copy the INC jobs import (policy action inc_sync_outer, R3; lever X5).
    Only files whose content differs are written (tmp + rename, mode and mtime
    kept); nothing is deleted from the outer copy. Refused, with nothing
    written:
      * while squeue shows an inc_* job that is not an experiment's run array
        (inc_<exp>_NNNN): builds, audits, relevance, verify and plan jobs
        import modules lazily and would load the new code mid-job while their
        provenance records the old hashes (squeue down: refused too);
      * when it would change a protected module (protected_modules():
        driver.PINNED_MODULES and train.CODE_MODULES) while an experiment is
        unfinished: one the experiment pinned in state.code.modules to
        anything but its pinned hash (a change back to the pin is allowed: it
        repairs the outer copy), any other protected one at all (its remaining
        runs would run it, and train.py's drift check would pass on it).
    An abandoned experiment (cancel) with no run array in squeue holds
    nothing. Not a dry run: who authorised it goes to _actions.jsonl."""
    rec = base_record("sync-outer")
    rec["dry_run"] = bool(dry_run)
    try:
        prov = _meta(meta or {})
    except Refused as e:
        return fail(rec, e, error_kind="refused")
    rec["provenance"] = prov
    _sync_outer(rec, dry_run)
    if not dry_run:
        log_action(rec, prov, {"changes": len(rec.get("changes") or []), "written": rec.get("written"),
                               "conflicts": rec.get("conflicts"), "in_flight": rec.get("in_flight")})
    return rec


def _sync_outer(rec, dry_run):
    nested, outer = package_copies()
    if not nested.is_dir() or not outer.is_dir():
        return fail(rec, "need both copies: nested %s (%s), outer %s (%s)"
                    % (nested, nested.is_dir(), outer, outer.is_dir()), error_kind="refused")
    src, changes = _tree_files(nested), []
    for rel, p in sorted(src.items()):
        q = outer / rel
        ns = _sha256_file(p)
        os_ = _sha256_file(q) if q.is_file() else None
        if ns != os_:
            changes.append({"module": rel, "nested": ns, "outer": os_})
    rec["changes"] = changes
    changed = {c["module"]: c for c in changes}
    protected = set(protected_modules())
    rec["protected"] = sorted(protected)
    d = inc_dir()
    exps = sorted(e.name for e in d.iterdir() if e.is_dir() and NAME_RE.match(e.name)) if d.is_dir() else []
    q = squeue_jobs()
    if not q["ok"]:
        return fail(rec, "squeue unavailable (%s): in-flight INC jobs cannot be ruled out; nothing written"
                    % q["error"], error_kind="squeue")
    arrays, in_flight = collections.defaultdict(list), []
    for j in q["jobs"]:
        m = ARRAY_RE.match(j["name"])
        if m and m.group(1) in exps:
            arrays[m.group(1)].append(j["id"])
        else:
            in_flight.append(j)
    conflicts, running, skipped = [], [], []
    for name in exps:
        st = _read_json_or_none(d / name / "state.json")
        if not isinstance(st, dict) or st.get("done"):
            continue
        if abandoned(name) is not None and not arrays.get(name):
            skipped.append(name)
            continue
        running.append(name)
        pins = (st.get("code") or {}).get("modules") or {}
        for m, c in sorted(changed.items()):
            if m in pins:
                if c["nested"] != pins[m]:
                    conflicts.append({"exp": name, "module": m, "pinned": pins[m], "nested": c["nested"],
                                      "why": "pinned in state.code.modules"})
            elif m in protected:
                conflicts.append({"exp": name, "module": m, "outer": c["outer"], "nested": c["nested"],
                                  "why": "a protected module (driver.PINNED_MODULES / train.CODE_MODULES): the "
                                         "experiment's remaining runs import the outer copy and record its hash"})
    rec.update(running=running, abandoned=skipped, conflicts=conflicts, in_flight=in_flight)
    reasons = []
    if in_flight:
        reasons.append("INC jobs other than experiment run arrays are queued or running (%s): they import modules "
                       "mid-job; wait until they end" % ", ".join("%s %s" % (j["name"], j["id"]) for j in in_flight))
    if conflicts:
        reasons.append("the sync would change code an unfinished experiment depends on: %s (a person decides: "
                       "repin, or wait until it is done)"
                       % ", ".join(sorted({"%s:%s" % (c["exp"], c["module"]) for c in conflicts})))
    if reasons:
        return fail(rec, "; ".join(reasons), error_kind="refused")
    if dry_run:
        return rec
    written = []
    for c in changes:
        s, t = nested / c["module"], outer / c["module"]
        t.parent.mkdir(parents=True, exist_ok=True)
        tmp = t.with_name(".%s.%d.sync" % (t.name, os.getpid()))
        try:
            shutil.copy2(s, tmp)
            os.replace(tmp, t)
        except OSError as e:
            with contextlib.suppress(OSError):
                tmp.unlink()
            rec["written"] = written
            return fail(rec, "copying %s failed after %d file(s): %s" % (c["module"], len(written), e),
                        type(e).__name__, "io")
        written.append(c["module"])
    rec["written"] = written
    return rec


# ------------------------------------------------------------------ submit
def _under(path, root):
    rp, rr = os.path.realpath(path), os.path.realpath(root)
    return rp == rr or rp.startswith(rr.rstrip(os.sep) + os.sep)


def _check_path(kind, value, flag):
    if not os.path.isabs(value):
        raise Refused("%s %s: give an absolute path" % (flag, value))
    if ".." in Path(value).parts:
        raise Refused("%s %s: no '..' in paths" % (flag, value))
    root = inc_dir()
    if not _under(value, root):
        raise Refused("%s %s lies outside INC_DIR %s" % (flag, value, root))
    if kind == "path_in" and not os.path.isfile(value):
        raise Refused("%s %s: no such file" % (flag, value))
    if kind == "dir_in" and not os.path.isdir(value):
        raise Refused("%s %s: no such directory" % (flag, value))
    if kind == "out_json" and not value.endswith(".json"):
        raise Refused("%s %s: must end in .json" % (flag, value))
    if kind == "dir_out" and not os.path.isdir(os.path.dirname(value.rstrip("/")) or "/"):
        raise Refused("%s %s: its parent directory does not exist" % (flag, value))
    return value


def _value(kind, value, flag):
    D = _D()
    if kind == "name":
        if not NAME_RE.match(value):
            raise Refused("%s %r is not a name ([A-Za-z0-9][A-Za-z0-9_-]{0,63})" % (flag, value))
        return value
    if kind == "replay_mode":
        if value not in D.REPLAY_MODES:
            raise Refused("%s %r is not one of %s" % (flag, value, list(D.REPLAY_MODES)))
        return value
    if kind == "flips_mode":
        if value not in D.FLIPS_MODES:
            raise Refused("%s %r is not one of %s" % (flag, value, list(D.FLIPS_MODES)))
        return value
    if kind == "increment_sources":
        if value not in REALLOOP_INCREMENT_SOURCES:
            raise Refused("%s %r is not one of %s" % (flag, value, list(REALLOOP_INCREMENT_SOURCES)))
        return value
    if kind == "funnel_part":
        if value not in FUNNEL_PARTS:
            raise Refused("%s %r is not one of %s" % (flag, value, list(FUNNEL_PARTS)))
        return value
    if kind == "funnel_policy":
        if not FUNNEL_POLICY_RE.match(value):
            raise Refused("%s %r: comma-separated recovery policies R-A, R-C, R-T, R-V, R-J, R-F" % (flag, value))
        return value
    if kind == "recipes":
        parts = value.split(",")
        if not value or any(p not in D.TRAINERS for p in parts) or len(set(parts)) != len(parts):
            raise Refused("%s %r: a comma-separated subset of %s without repeats" % (flag, value, list(D.TRAINERS)))
        return value
    if kind == "int":
        if not INT_RE.match(value):
            raise Refused("%s %r is not a positive integer" % (flag, value))
        return int(value)
    if kind == "int0":
        if not INT0_RE.match(value):
            raise Refused("%s %r is not a non-negative integer" % (flag, value))
        return int(value)
    if kind == "seeds":
        if not SEEDS_RE.match(value) or len(set(value.split(","))) != len(value.split(",")):
            raise Refused("%s %r: distinct non-negative integers, comma-separated" % (flag, value))
        return value
    if kind == "audit_list":
        name, sep, path = value.partition("=")
        if not sep or not AUDIT_NAME_RE.match(name):
            raise Refused("%s %r: NAME=MANIFEST with NAME in [A-Za-z0-9_.-]" % (flag, value))
        _check_path("path_in", path, "%s %s=" % (flag, name))
        return value
    if kind in ("path_in", "dir_in", "out_json", "dir_out"):
        return _check_path(kind, value, flag)
    raise Refused("internal: unknown kind %r" % kind)


def _strip_prefix(builder, args):
    """Accept a lever's argv as written: a leading 'sbatch' and the job script,
    and for build a leading 'python -m'."""
    args = list(args)
    if args and args[0] == "sbatch":
        args = args[1:]
    if args and (args[0] == SCRIPTS[builder] or args[0].endswith("/" + SCRIPTS[builder])):
        args = args[1:]
    if builder == "build" and len(args) >= 2 and PYTHON_RE.match(args[0]) and args[1] == "-m":
        args = args[2:]
    return args


def parse_builder_args(builder, args):
    """The validated request: {builder, action, module, command, script_args,
    params, path_params}. Raises Refused."""
    args = list(args)
    if builder in BUILDER_ALIASES:              # 'submit pilot build ...' = 'submit build pilot build ...'
        args, builder = [builder] + args, BUILDER_ALIASES[builder]
    if builder not in SCRIPTS:
        raise Refused("builder %r is not one of %s" % (builder, sorted(SCRIPTS) + sorted(BUILDER_ALIASES)))
    args = _strip_prefix(builder, args)
    for a in args:
        if not isinstance(a, str) or not a or BAD_TOKEN_RE.search(a) or len(a) > 1024:
            raise Refused("argument %r holds whitespace, a shell metacharacter or is empty / too long" % (a,))
    if builder == "funnel":
        return _parse_funnel_args(args)
    m = MODULE_RE.match(args[0]) if args else None
    if builder == "build":
        # run_inc_build.sh runs a module: the arguments start with it
        if not m or m.group(1) not in ("pilot", "realloop"):
            raise Refused("build: the arguments start with the module, pilot or realloop (got %r)" % (args[:1],))
        module = m.group(1)
    else:
        # run_inc_relevance.sh / run_inc_audit.sh run one module; naming it is optional
        module = "relevance" if builder == "relevance" else "audit"
        if m and m.group(1) != module:
            raise Refused("%s runs inc.%s, not inc.%s" % (builder, module, m.group(1)))
    if m:
        args = args[1:]
    command = None
    if builder != "audit":
        if not args or args[0].startswith("--"):
            raise Refused("%s %s: a command is missing" % (builder, module))
        command, args = args[0], args[1:]
    action = ACTIONS.get((builder, module, command))
    if action is None:
        raise Refused("%s does not run %s %s (it runs %s)"
                      % (builder, module, command or "",
                         ", ".join(" ".join(x for x in k[1:] if x) for k in ACTIONS if k[0] == builder)))
    form = FORMS[action]
    params, seen, i = {}, set(), 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith("--"):
            raise Refused("unexpected argument %r (%s takes flags only)" % (tok, action))
        flag, eq, val = tok.partition("=")
        spec = form["flags"].get(flag)
        if spec is None:
            raise Refused("%s is not accepted for %s (accepted: %s)" % (flag, action, sorted(form["flags"])))
        param, kind = spec
        if kind != "audit_list" and flag in seen:
            raise Refused("%s given twice" % flag)
        seen.add(flag)
        if kind == "flag":
            if eq:
                raise Refused("%s takes no value" % flag)
            params[param] = True
            i += 1
            continue
        if kind == "audit_list":
            vals = [val] if eq else []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i])
                i += 1
            if not vals or any(not v for v in vals):
                raise Refused("%s needs NAME=MANIFEST values" % flag)
            params.setdefault(param, []).extend(_value(kind, v, flag) for v in vals)
            continue
        if eq:
            i += 1
        else:
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raise Refused("%s needs a value" % flag)
            val = args[i + 1]
            i += 2
        if not val:
            raise Refused("%s needs a value" % flag)
        params[param] = _value(kind, val, flag)
    missing = [f for f in form["required"] if f not in seen]
    if missing:
        raise Refused("%s needs %s" % (action, ", ".join(missing)))
    if params.get("increment_sources") == INCREMENT_SOURCES_EVIDENCE and "relevance" in params:
        raise Refused(M.EVIDENCE_WITH_RELEVANCE)
    if (params.get("increment_sources") == INCREMENT_SOURCES_RECOVERED) != ("step1_overlay" in params):
        raise Refused("--increment-sources recovered and --step1-overlay go together (realloop build refuses "
                      "one without the other)")
    if params.get("increment_sources") == INCREMENT_SOURCES_RECOVERED:
        bad = [f for f, k in (("--relevance", "relevance"), ("--n-verified", "n_verified"), ("--no-truth", "no_truth"))
               if k in params]
        if bad or "size" not in params:
            raise Refused("--increment-sources recovered needs --size and refuses %s (realloop build)"
                          % (", ".join(bad) or "--relevance, --n-verified and --no-truth"))
    if "audit" in params:
        names = [v.partition("=")[0] for v in params["audit"]]
        if len(set(names)) != len(names):
            raise Refused("an audit name is given twice: %s" % names)
    path_params = sorted(p for p, k in form["flags"].values() if k in PATH_KINDS and p in params)
    script_args = ([module, command] if builder == "build" else [command] if builder == "relevance" else []) + args
    return {"builder": builder, "action": action, "module": module, "command": command,
            "script_args": script_args, "params": params, "path_params": path_params}


def _parse_funnel_args(args):
    """The validated request of 'submit funnel -- VERB FLAGS' (run_inc_funnel.sh VERB)."""
    if not args or args[0].startswith("--"):
        raise Refused("funnel: the arguments start with the verb (%s, map, recover)" % ", ".join(FUNNEL_AUDIT_VERBS))
    command, rest = args[0], args[1:]
    action = ACTIONS.get(("funnel", "funnel", command))
    if action is None:
        raise Refused("funnel does not run %r as a job (it runs %s, map, recover)"
                      % (command, ", ".join(FUNNEL_AUDIT_VERBS)))
    form = FUNNEL_FORMS[action]
    params, seen, i = {}, set(), 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("--"):
            raise Refused("unexpected argument %r (%s takes flags only)" % (tok, action))
        flag, eq, val = tok.partition("=")
        spec = form["flags"].get(flag)
        if spec is None:
            raise Refused("%s is not accepted for %s (accepted: %s)" % (flag, action, sorted(form["flags"])))
        if flag in seen:
            raise Refused("%s given twice" % flag)
        seen.add(flag)
        param, kind = spec
        if kind == "flag":
            if eq:
                raise Refused("%s takes no value" % flag)
            params[param] = True
            i += 1
            continue
        if eq:
            i += 1
        else:
            if i + 1 >= len(rest) or rest[i + 1].startswith("--"):
                raise Refused("%s needs a value" % flag)
            val = rest[i + 1]
            i += 2
        if not val:
            raise Refused("%s needs a value" % flag)
        params[param] = _value(kind, val, flag)
    missing = [f for f in form["required"] if f not in seen]
    if missing:
        raise Refused("%s needs %s" % (action, ", ".join(missing)))
    if params.get("rl") and command != "qualify":
        raise Refused("--rl belongs to qualify (the reference labeller's qualification), not %s" % command)
    if action == "inc_funnel_audit":
        params["verb"] = command
    path_params = sorted(p for p, k in form["flags"].values() if k in PATH_KINDS and p in params)
    return {"builder": "funnel", "action": action, "module": "funnel", "command": command,
            "script_args": list(args), "params": params, "path_params": path_params}


def funnel_resources(verb):
    """The extra sbatch flags of a funnel verb: funnel/__main__.py
    SBATCH_RESOURCES[VERB_CLASS[verb]] (the CLI's one table; runner 5.6.1)."""
    from ..funnel import __main__ as FM
    cls = FM.VERB_CLASS.get(verb)
    if cls is None or cls not in FM.SBATCH_RESOURCES:
        raise Refused("funnel verb %r has no resource class in the CLI's table" % (verb,))
    return list(FM.SBATCH_RESOURCES[cls])


def load_levers(path=None):
    """(path, [(lever id, row)]) of levers.json; raises Refused when it is missing
    or unreadable (no menu, no submission)."""
    p = Path(path or levers_path())
    try:
        with open(p) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise Refused("levers.json unreadable at %s (%s): no menu, nothing is submitted" % (p, _short(e, 200)))
    rows = data.get("levers", data) if isinstance(data, dict) else data
    if isinstance(rows, dict):
        items = [(str(k), v) for k, v in rows.items() if isinstance(v, dict)]
    elif isinstance(rows, list):
        items = [(str(r.get("id")), r) for r in rows if isinstance(r, dict)]
    else:
        raise Refused("levers.json at %s holds no lever rows" % p)
    return p, items


def _policy():
    from ..brain import policy
    return policy


def _typed(value, bound):
    """A CLI value in the type its declared bound expects (policy._check_value
    never coerces)."""
    if not isinstance(bound, dict):
        return value
    t, vt = bound.get("type"), bound.get("value_type")
    wants_int = t == "int" or (t == "enum" and vt == "int")
    if value is True:
        return 1 if wants_int else ("true" if t in ("str", "enum") else True)
    if wants_int and isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if t == "float" and isinstance(value, (str, int)) and not isinstance(value, bool):
        with contextlib.suppress(ValueError):
            return float(value)
    return value


def _check_bounds(params, bounds):
    """(ok, reasons) of params against one bounds table (policy._check_params:
    no coercion beyond the CLI string's declared type; an undeclared param is
    refused). A NAME=MANIFEST list is checked as policy_actions.json writes it,
    comma-joined."""
    typed = {k: _typed(",".join(v) if isinstance(v, list) else v, bounds.get(k)) for k, v in params.items()}
    return _policy()._check_params(typed, bounds)


def _fmt_defaults(defaults, keys):
    used = {k: defaults[k] for k in keys if k in defaults}
    return " [not given, checked at the builder's default: %s]" % ", ".join(
        "%s=%s" % (k, v) for k, v in sorted(used.items())) if used else ""


def check_menu(req, levers=None):
    """{levers, policy_row, defaults_checked} for a request the menu admits;
    raises Refused.

    The policy table is the authority: an action with no valid
    policy_actions.json row (policy.describe known false: absent, or its row
    failed validation) is refused. Admitted = at least one levers.json row
    with the request's policy action admits every parameter, where a row's
    bounds are its own param_bounds over its policy row's (the lever narrows
    what it declares; the policy row bounds the rest), and every value the
    row's "fixed" names equals it. A parameter neither declares is refused.
    The policy row alone must also admit every parameter it declares. A flag
    the command leaves out is checked at the builder's default
    (builder_defaults) wherever a bound (or "fixed") declares it: pilot build
    without --replay-mode builds sample replay, which L1's ["full"] refuses."""
    policy = _policy()
    lpath, items = levers if levers is not None else load_levers()
    action = req["action"]
    prow = policy.describe(action)
    if not prow.get("known"):
        raise Refused("policy_actions.json has no valid row for %s (%s): the policy table is the authority, "
                      "nothing is submitted" % (action, prow.get("reason") or "unknown"))
    pbounds = prow.get("param_bounds")
    if not isinstance(pbounds, dict):
        raise Refused("policy_actions.json %s has no param_bounds object: nothing is submitted" % action)
    rows = [(lid, r) for lid, r in items if r.get("policy_action") == action]
    if not rows:
        raise Refused("no lever in %s runs %s: the command is not on the menu" % (lpath, action))
    given = req["params"]
    defaults = {k: v for k, v in builder_defaults(action).items() if k not in given}
    admitted, why, checked = [], [], set()
    for lid, r in rows:
        own = r.get("param_bounds", r.get("bounds"))
        if own is not None and not isinstance(own, dict):
            why.append("%s: param_bounds is not an object" % lid)
            continue
        fixed = r.get("fixed") or {}
        if not isinstance(fixed, dict):
            why.append("%s: fixed is not an object" % lid)
            continue
        merged = dict(pbounds)
        merged.update(own or {})
        eff = {k: v for k, v in defaults.items() if k in merged or k in fixed}
        eff.update(given)
        ok, reasons = _check_bounds(eff, merged)
        for k, v in sorted(fixed.items()):
            got = eff.get(k)
            if got is None or str(_typed(got, merged.get(k))) != str(v):
                ok = False
                reasons.append("%r must be %r (the lever fixes it), got %r" % (k, v, got))
        if ok:
            checked.update(k for k in eff if k in defaults)
            admitted.append({"lever": lid, "declared_by_lever": sorted(k for k in given if k in (own or {})),
                             "declared_by_policy": sorted(k for k in given if k not in (own or {}))})
        else:
            why.append("%s: %s%s" % (lid, "; ".join(reasons), _fmt_defaults(defaults, eff)))
    if not admitted:
        raise Refused("no %s lever admits these arguments: %s" % (action, " | ".join(why)))
    declared = {k: v for k, v in defaults.items() if k in pbounds}
    declared.update({k: v for k, v in given.items() if k in pbounds})
    ok, reasons = _check_bounds(declared, pbounds)
    if not ok:
        raise Refused("policy_actions.json %s refuses: %s%s" % (action, "; ".join(reasons),
                                                                _fmt_defaults(defaults, declared)))
    checked.update(k for k in declared if k in defaults)
    return {"levers": admitted, "policy_row": "present",
            "defaults_checked": {k: defaults[k] for k in sorted(checked)}}


def _meta(meta):
    out = {}
    for key, rx, what in (("parent_exp", NAME_RE, "an experiment name"), ("approval_id", APPROVAL_RE, "an id"),
                          ("decided_by", ACTOR_RE, "an actor id")):
        v = meta.get(key)
        if v is not None:
            if not rx.match(v):
                raise Refused("--%s %r is not %s" % (key.replace("_", "-"), v, what))
            out[key] = v
    t = meta.get("trigger")
    if t is not None:
        parts = t.split(",")
        if not t or not all(TRIGGER_RE.match(p) for p in parts):
            raise Refused("--trigger %r: comma-separated diagnosis ids" % t)
        out["trigger"] = parts
    return out


def submission_env(prov):
    """The login environment for sbatch (the job exports it): no SLURM_*/SBATCH_*
    of an enclosing job, no test-mode scoring, no drift override; the
    provenance fields as INCAP_* variables."""
    from ..inc.scorer import TEST_ENV
    env = _D().submission_env(False)
    env.pop(TEST_ENV, None)
    for k in DRIFT_ENVS:
        env.pop(k, None)
    for k in ("INCAP_PARENT_EXP", "INCAP_TRIGGER", "INCAP_APPROVAL_ID", "INCAP_DECIDED_BY"):
        env.pop(k, None)
    for k, v in prov.items():
        env["INCAP_" + k.upper()] = ",".join(v) if isinstance(v, list) else str(v)
    env["INCAP_REQUESTED_UTC"] = M.utc_now()
    return env


def _job_name(req):
    p = req["params"]
    if req["builder"] == "funnel":
        return "inc_funnel_%s" % (req["command"] if req["command"] != "map" else "map_%s" % p.get("part"))
    if req["builder"] == "build":
        return "inc_build_%s" % p["exp"]
    if req["builder"] == "audit":
        rel = os.path.relpath(os.path.realpath(p["out"]), os.path.realpath(inc_dir())).split(os.sep)
        return "inc_audit_%s" % rel[0]
    return "inc_relevance"


def submit(builder, args, meta=None, dry_run=False, levers=None):
    rec = base_record("submit")
    rec.update(builder=builder, dry_run=bool(dry_run))
    try:
        prov = _meta(meta or {})
        req = parse_builder_args(builder, args)
        builder = req["builder"]
        rec.update(action=req["action"], script=SCRIPTS[builder], script_args=req["script_args"],
                   params=req["params"], provenance=prov)
        rec["menu"] = check_menu(req, levers)
        p, root = req["params"], inc_dir()
        if builder == "build":
            e = root / p["exp"]
            if (e / "exp.json").exists() or (e / "state.json").exists():
                raise Refused("experiment %s is already built at %s; an experiment is built once" % (p["exp"], e))
        if builder == "audit":
            rel = os.path.relpath(os.path.realpath(p["out"]), os.path.realpath(root)).split(os.sep)
            if len(rel) != 3 or rel[1] != "audit" or not (root / rel[0] / "exp.json").is_file():
                raise Refused("--out %s: the autopilot writes an audit only as INC_DIR/<built exp>/audit/<name>.json"
                              % p["out"])
            # L4 runs once per experiment; a rerun over an existing audit is a person's decision.
            have = [str(x) for x in (Path(p["out"]), root / (LABEL_AUDIT % rel[0]), root / (LEGACY_AUDIT % rel[0]))
                    if x.is_file()]
            if have:
                raise Refused("a label audit of %s already exists (%s): the autopilot does not run it again"
                              % (rel[0], ", ".join(sorted(set(have)))))
        if builder == "relevance" and "out" in p and not RELEVANCE_OUT_RE.match(os.path.basename(p["out"])):
            raise Refused("--out %s: a relevance file is named relevance*.json" % p["out"])
        script = script_dir() / SCRIPTS[builder]
        if not script.is_file():
            raise Refused("job script %s not found" % script)
        name = _job_name(req)
        sbatch = os.environ.get("INCAP_SBATCH", "sbatch")
        extra = funnel_resources(req["command"]) if builder == "funnel" else []
        argv = [sbatch, "--parsable", "--job-name=%s" % name] + extra + [str(script)] + req["script_args"]
        rec.update(job_name=name, sbatch_argv=argv)
        if dry_run:
            return rec
        q = squeue_jobs()
        if not q["ok"]:
            raise Refused("squeue unavailable (%s): cannot rule out a duplicate job, nothing submitted" % q["error"])
        dup = [j for j in q["jobs"] if j["name"] == name]
        if dup and builder != "relevance":
            raise Refused("%s is already queued or running as job %s" % (name, dup[0]["id"]))
        if dup:
            raise Refused("a relevance build is already queued or running as job %s" % dup[0]["id"])
        hand = [j for j in q["jobs"] if j["name"] == SCRIPT_JOB_NAMES[builder]]
        if hand:
            raise Refused("%s is queued or running as job %s under the script's own name (a submission by hand, "
                          "its experiment unknown): a duplicate cannot be ruled out, nothing submitted"
                          % (SCRIPT_JOB_NAMES[builder], hand[0]["id"]))
    except Refused as e:
        return fail(rec, e, error_kind="refused")
    for d in (inc_dir() / "logs", inc_dir() / "step1" / "logs", inc_dir() / "funnel" / "logs"):   # Slurm opens --output before the script runs
        d.mkdir(parents=True, exist_ok=True)
    C = _C()
    try:
        pr = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=submission_env(prov),
                            cwd=str(C.REPO) if Path(C.REPO).is_dir() else None)
    except (OSError, subprocess.TimeoutExpired) as e:
        return fail(rec, "sbatch could not be run: %s: %s" % (type(e).__name__, _short(e, 200)),
                    type(e).__name__, "submit")
    rec.update(sbatch_rc=pr.returncode, sbatch_stderr=_short(pr.stderr, 400))
    if pr.returncode != 0:
        return fail(rec, "sbatch exited %d: %s" % (pr.returncode, _short(pr.stderr or pr.stdout, 300)),
                    "SubmitFailed", "submit")
    D = _D()
    try:
        rec["job_id"] = D.parse_parsable(pr.stdout)
    except D.DriverError as e:
        return fail(rec, e, type(e).__name__, "submit")
    return rec


# ---------------------------------------------------------- batched snapshot
def _report_due(exp, mode):
    if mode == "never":
        return False
    root = inc_dir() / exp
    if not (root / "exp.json").is_file() or not (root / "state.json").is_file():
        return False
    if mode == "always":
        return True
    st = _read_json_or_none(root / "state.json")
    if not (isinstance(st, dict) and st.get("done")):
        return False
    rp = root / "report.json"
    return not rp.is_file() or rp.stat().st_mtime < (root / "state.json").stat().st_mtime


def _guard(verb, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:                      # one experiment's failure never sinks the batch
        rec = base_record(verb)
        return fail(rec, e, type(e).__name__, "other")


def job_states(job_ids):
    """sacct of the named jobs, {"ok", "jobs": {id: {"state", "elapsed_s",
    "gpu_count", "gpu_type"}}} (stream_remote.sacct; INCAP_SACCT replaces the
    command in tests). Only a stream data job's log is ever read there, so a
    funnel job ships its state alone."""
    from . import stream_remote as SR
    rec = SR.sacct(job_ids)
    return dict(rec, verb="sacct") if isinstance(rec, dict) else rec


def campaign_snapshot(exps, do_advance=False, report_mode="auto", ledger_from=None, step1=True, backend=None,
                      funnel=False, sacct=None):
    rec = base_record("campaign-snapshot")
    ledger_from = ledger_from or {}
    out, seen = {}, []
    for exp in exps:
        if exp in seen:
            continue
        seen.append(exp)
        sub = {}
        if do_advance:
            if not (inc_dir() / str(exp) / "exp.json").is_file():
                sub["advance"] = {"verb": "advance", "ok": True, "skipped": "not built"}
            elif NAME_RE.match(str(exp)) and abandoned(exp) is not None:
                sub["advance"] = {"verb": "advance", "ok": True, "skipped": "abandoned",
                                  "abandoned": str(abandoned_path(exp))}
            else:
                sub["advance"] = _guard("advance", advance, exp, backend=backend)
        if report_mode not in ("auto", "always", "never"):
            sub["report"] = fail(base_record("report"), "report mode %r" % report_mode, error_kind="refused")
        else:
            try:
                due = _report_due(exp, report_mode)
            except (OSError, ValueError) as e:
                due = False
                sub["report"] = fail(base_record("report"), e, type(e).__name__, "io")
            if due:
                sub["report"] = _guard("report", report, exp)
        lf, lsha = _ledger_pos(ledger_from.get(exp, 0))
        sub["snapshot"] = _guard("snapshot", snapshot, exp, ledger_from=lf, step1=False, ledger_sha256=lsha)
        out[exp] = sub
    rec["experiments"] = out
    if step1:
        rec["step1"] = _guard("step1", step1_snapshot)
    if funnel:
        rec["funnel"] = _guard("funnel-summary", funnel_summary, derive_ledger=True)
    rec["status"] = _guard("status", status)
    if sacct:
        # read-only, and never sinks the snapshot: a failed sacct only leaves
        # the job states unread this tick
        rec["sacct"] = _guard("sacct", job_states, [str(j) for j in sacct])
    subs = [r for s in out.values() for r in s.values() if isinstance(r, dict)]
    subs += [rec[k] for k in ("step1", "funnel", "status") if k in rec]
    rec["ok"] = all(r.get("ok", False) for r in subs)
    return rec


# ------------------------------------------------------------ the funnel audit
# docs/FUNNEL_AUDIT.md 8.2 and runner 5.5.6: the funnel verbs print aggregates
# only. FUNNEL_SHIP is the whole list of files a summary ships (as named for the
# lab's evidence: recovery.json lives in step1_r1/ and is shipped as
# funnel/recovery.json); FUNNEL_NEVER names what is never read for shipping,
# whatever it holds (row-level files, key files, evaluation descriptors). The
# funnel directory's listing gives names, sha256 and sizes only, never content.
FUNNEL_SHIP = (("funnel/funnel_ledger.json", "funnel/funnel_ledger.json"),
               ("funnel/audit_v1.json", "funnel/audit_v1.json"),
               ("funnel/class_maps.json", "funnel/class_maps.json"),
               ("funnel/recovery.json", "step1_r1/recovery.json"),
               ("funnel/prospective_da.json", "funnel/prospective_da.json"),
               ("step1/pool_summary.json", "step1/pool_summary.json"),
               ("step1/calibration.json", "step1/calibration.json"))
FUNNEL_NEVER = ("step1/conflicts.csv", "step1/pool_verdicts.npz", "funnel/ledger.jsonl", "funnel/sample_v1_key.jsonl",
                "funnel/sheets_v1_key/", "funnel/sheets_v1_cluster/", "funnel/leak_eval_desc.npz",
                "funnel/leak_pairs_v1.csv")
FUNNEL_LISTING_DEPTH = 3                 # funnel/<a>/<b>/<file>: rl_answers/RL-B/<sheet>.json
FUNNEL_HASH_MAX = 64 << 20               # a bigger file is listed with its size, sha256 null
FIT_INFO = "step1/verifier/fit_info.json"


def funnel_listing():
    """{INC_DIR-relative path: {"sha256", "bytes"}} of INC_DIR/funnel/ to depth
    FUNNEL_LISTING_DEPTH (one os.listdir per directory; sha256 cached by size
    and mtime in _campaign/cache/funnel_listing.json)."""
    root = inc_dir() / "funnel"
    cache_path = campaign_dir() / "cache" / "funnel_listing.json"
    cache = _read_json_or_none(cache_path) or {}
    out, new_cache = {}, {}

    def go(d, depth):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return
        for n in names:
            p = d / n
            rel = str(p.relative_to(inc_dir()))
            if p.is_dir():
                if depth < FUNNEL_LISTING_DEPTH:
                    go(p, depth + 1)
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            key = "%s|%d|%d" % (rel, st.st_size, st.st_mtime_ns)
            sha = cache.get(key)
            if sha is None and st.st_size <= FUNNEL_HASH_MAX:
                sha = _sha256_file(p)
            if sha is not None:
                new_cache[key] = sha
            out[rel] = {"sha256": sha, "bytes": st.st_size}
    if root.is_dir():
        go(root, 1)
    if new_cache != cache:
        with contextlib.suppress(OSError):
            _write_json_atomic(cache_path, new_cache)
    return out


def _derive_ledger():
    """The summaries-derived funnel ledger of this Step 1 (adapters.inc_step1.
    ledger_from_summaries on step1/*_summary.json, calibration.json and
    funnel/census_v0.json), built in a temporary directory: nothing under
    INC_DIR is written. Returns (ledger, None) or (None, why)."""
    import tempfile
    from ..funnel.adapters import inc_step1 as AD
    from ..funnel import domain as FD
    census = inc_dir() / "funnel" / "census_v0.json"
    if not census.is_file():
        return None, "no funnel/census_v0.json on the cluster"
    proj = None
    with contextlib.suppress(Exception):
        proj = AD.verifier_fit_info_projection(inc_dir() / "step1")
    with tempfile.TemporaryDirectory(prefix="funnel_ledger_") as td:
        out = Path(td) / "funnel_ledger.json"
        AD.ledger_from_summaries(FD.load(M.DOMAIN), inc_dir() / "step1", census, out, fit_info_projection=proj)
        with open(out) as fh:
            return json.load(fh), None


def funnel_summary(derive_ledger=False):
    """INCAP funnel-summary: the funnel's aggregates (FUNNEL_SHIP), the verifier
    fit record's projection (derived: step1/verifier_fit_info.json), and the
    listing of INC_DIR/funnel (derived: funnel/files.json); with derive_ledger
    and no funnel_ledger.json on disk, the summaries-derived ledger computed in
    memory (derived.funnel_ledger). Everything in "decision" passes dev_only."""
    rec = base_record("funnel-summary")
    arts, derived, files, missing, notes = {}, {}, {}, [], []
    for name, rel in FUNNEL_SHIP:
        p = inc_dir() / rel
        if not p.is_file():
            missing.append(rel)
            continue
        obj, info = read_small(p)
        files[name] = dict(info, path=rel)
        if obj is None:
            notes.append("%s not shipped: %s" % (rel, info.get("why")))
            continue
        arts[name] = obj
    fp = inc_dir() / FIT_INFO
    if fp.is_file():
        try:
            from ..funnel.adapters import inc_step1 as AD
            arts["step1/verifier_fit_info.json"] = AD.verifier_fit_info_projection(inc_dir() / "step1")
            files["step1/verifier_fit_info.json"] = {"sha256": _sha256_file(fp), "bytes": fp.stat().st_size,
                                                    "path": FIT_INFO, "shipped": False,
                                                    "why": "projected to its OtherPlant sample"}
        except Exception as e:
            notes.append("verifier fit record not projected: %s: %s" % (type(e).__name__, _short(e, 200)))
    else:
        missing.append(FIT_INFO)
    arts["funnel/files.json"] = {"format": "funnel-files/1", "root": "funnel/", "files": funnel_listing()}
    if derive_ledger and "funnel/funnel_ledger.json" not in arts and "funnel/funnel_ledger.json" not in files:
        try:
            led, why = _derive_ledger()
        except Exception as e:
            led, why = None, "%s: %s" % (type(e).__name__, _short(e, 300))
        if led is not None:
            derived["funnel_ledger"] = led
        else:
            notes.append("summaries-derived ledger not built: %s" % why)
    decision, redacted = dev_only({"artifacts": arts, "derived": derived})
    rec.update(decision_exam=M.DECISION_EXAM, decision=decision, redacted=redacted, files=files,
               missing=missing, notes=notes, never_shipped=list(FUNNEL_NEVER))
    return rec


def funnel_ledger_summaries(write=False):
    """INCAP funnel-ledger-summaries (F2a): the summaries-derived ledger, built
    in memory; with write, also written to INC_DIR/funnel/funnel_ledger.json
    when no ledger is there (a census-derived ledger is never replaced here)."""
    rec = base_record("funnel-ledger-summaries")
    led, why = _derive_ledger()
    if led is None:
        return fail(rec, why, error_kind="missing")
    target = inc_dir() / "funnel" / "funnel_ledger.json"
    if write:
        if target.exists():
            cur = _read_json_or_none(target) or {}
            if cur.get("derivation") != "summaries" or cur.get("fingerprint") != led.get("fingerprint"):
                return fail(rec, "%s exists (derivation %r); a census-derived or other ledger is not replaced here"
                            % (target, cur.get("derivation")), error_kind="refused")
        else:
            _write_json_atomic(target, led)
            rec["written"] = str(target)
    rec["ledger"] = dev_only(led)[0]
    rec["fingerprint"] = led.get("fingerprint")
    return rec


FUNNEL_BASE_RUN_RE = re.compile(r"base__s[0-9]+\Z")


def funnel_dev_scores(exps, truth_step=None):
    """INCAP funnel-dev-scores: runs/<run>/scores/dev.json of each experiment's
    base runs (base__s<seed>), or with truth_step of a real loop step's truth
    'with' runs (truth__s<k>_<step>__union__s<seed>), and nothing else. A score
    not stamped dev is refused (the panel reads dev only)."""
    rec = base_record("funnel-dev-scores")
    if truth_step is not None and not NAME_RE.match(truth_step):
        return fail(rec, "--truth-step %r is not a step name" % truth_step, error_kind="refused")
    rx = re.compile(r"truth__s[0-9]+_%s__union__s[0-9]+\Z" % re.escape(truth_step)) if truth_step \
        else FUNNEL_BASE_RUN_RE
    scores, files = {}, {}
    for exp in exps:
        try:
            _check_exp(exp)
        except Refused as e:
            return fail(rec, e, error_kind="refused")
        runs = inc_dir() / exp / "runs"
        got = {}
        try:
            names = sorted(os.listdir(runs))
        except OSError:
            names = []
        for rid in names:
            if not rx.match(rid):
                continue
            p = runs / rid / "scores" / ("%s.json" % M.DECISION_EXAM)
            if not p.is_file():
                continue
            obj, info = read_small(p)
            if not isinstance(obj, dict) or obj.get("exam") != M.DECISION_EXAM:
                return fail(rec, "%s/runs/%s/scores/%s.json is not a dev score" % (exp, rid, M.DECISION_EXAM),
                            error_kind="refused")
            got[rid] = obj
            files["%s/runs/%s/scores/%s.json" % (exp, rid, M.DECISION_EXAM)] = info
        scores[exp] = got
    rec.update(scores=scores, files=files, truth_step=truth_step, exams_read=[M.DECISION_EXAM])
    return rec


# --------------------------------------------------------------- fixtures
def write_fixture(rec, out_dir):
    """Write a snapshot (or each snapshot of a campaign-snapshot, and its Step 1)
    as files under out_dir, named as on the cluster (relative to INC_DIR): the
    decision artifacts (state.json without runs; report.json with its final
    table's dev column only, derived.report_final_dev put back at /final, as
    evidence.from_snapshot does), <exp>/ledger.jsonl when the record holds the
    whole ledger, <exp>/derived/state_runs.json and
    step1/select_clusters_by_source.json (evidence.py's names). Returns {name:
    sha256}, also written to out_dir/MANIFEST.json: out_dir is its own tree,
    not tests/fixtures/inc_replay (whose MANIFEST.json pins what is copied in)."""
    out_dir = Path(out_dir)
    snaps = []
    if rec.get("verb") == "snapshot":
        snaps.append(rec)
    elif rec.get("verb") == "campaign-snapshot":
        snaps += [s["snapshot"] for s in (rec.get("experiments") or {}).values() if s.get("snapshot")]
        if rec.get("step1"):
            snaps.append(rec["step1"])
    else:
        raise ValueError("a fixture comes from a snapshot or campaign-snapshot record, not %r" % rec.get("verb"))
    written, sources = {}, {}

    def put(name, text):
        p = out_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        written[name] = _sha256_bytes(text.encode("utf-8"))

    for s in snaps:
        if not s.get("ok"):
            raise ValueError("record %s is not ok: %s" % (s.get("verb"), s.get("error")))
        dec = s.get("decision") or {}
        derived = dec.get("derived") or {}
        for name, obj in sorted((dec.get("artifacts") or {}).items()):
            if name.endswith("/report.json") and isinstance(obj, dict) and "final" not in obj \
                    and isinstance(derived.get("report_final_dev"), list):
                obj = dict(obj, final=derived["report_final_dev"])
            put(name, json.dumps(obj, indent=1, sort_keys=True) + "\n")
            if name.endswith("/state.json") and isinstance(derived.get("state_runs"), dict):
                put(name[:-len("state.json")] + "derived/state_runs.json",
                    json.dumps(derived["state_runs"], indent=1, sort_keys=True) + "\n")
        led = dec.get("ledger")
        if led and led.get("from_line") == 0 and led.get("complete") and not led.get("partial_tail"):
            put(led["artifact"], "".join(json.dumps(e["entry"], sort_keys=True) + "\n" for e in led["entries"]))
        agg = derived.get("select_clusters_by_source")
        if agg:
            put("step1/select_clusters_by_source.json", json.dumps(agg, indent=1, sort_keys=True) + "\n")
        sources.update(s.get("files") or {})
    man = {"format": FORMAT, "written_utc": M.utc_now(), "from_utc": rec.get("utc"), "host": rec.get("host"),
           "note": "decision sections only (dev): state.json without runs (derived/state_runs.json instead), "
                   "report.json's final table with the dev column only",
           "files": written, "cluster_files": sources}
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return written


# ---------------------------------------------------------------------- CLI
class _ArgError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _ArgError(message)

    def exit(self, status=0, message=None):
        raise _ArgError(message or "exit %s" % status)

    def print_help(self, file=None):
        super().print_help(sys.stderr)


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _ledger_pos(v):
    """(line, expected prefix sha256 or None) of a ledger_from value: N, (N, SHA)."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), v[1]
    return int(v or 0), None


def _ledger_arg(text, what="--ledger-from"):
    """N or N:SHA256 -> (N, SHA256 or None)."""
    n, sep, sha = str(text).partition(":")
    if not INT0_RE.match(n) or (sep and not SHA256_RE.match(sha)):
        raise _ArgError("%s takes N or N:SHA256 (64 lowercase hex), got %r" % (what, text))
    return int(n), (sha if sep else None)


def _ledger_map(items):
    out = {}
    for it in items or []:
        exp, sep, pos = it.partition("=")
        if not sep or not NAME_RE.match(exp):
            raise _ArgError("--ledger-from takes EXP=N or EXP=N:SHA256, got %r" % it)
        out[exp] = _ledger_arg(pos, "--ledger-from %s=" % exp)
    return out


def _split_submit(argv):
    """(builder, meta, dry_run, builder args) from the words after 'submit'."""
    if not argv:
        raise _ArgError("submit BUILDER [--parent-exp P] [--trigger T] [--approval-id ID] [--decided-by A] "
                        "[--dry-run] -- ARGS ...")
    builder, rest = argv[0], argv[1:]
    meta, dry = {}, False
    if "--" not in rest:
        return builder, meta, dry, rest
    k = rest.index("--")
    head, args = rest[:k], rest[k + 1:]
    i = 0
    keys = {"--parent-exp": "parent_exp", "--trigger": "trigger", "--approval-id": "approval_id",
            "--decided-by": "decided_by"}
    while i < len(head):
        tok = head[i]
        if tok == "--dry-run":
            dry = True
            i += 1
            continue
        flag, eq, val = tok.partition("=")
        if flag not in keys:
            raise _ArgError("submit option %r is not one of %s, --dry-run" % (tok, sorted(keys)))
        if not eq:
            if i + 1 >= len(head):
                raise _ArgError("%s needs a value" % flag)
            val = head[i + 1]
            i += 1
        if keys[flag] in meta:
            raise _ArgError("%s given twice" % flag)
        meta[keys[flag]] = val
        i += 1
    return builder, meta, dry, args


def dispatch(argv):
    if not argv:
        raise _ArgError("a verb is needed: status, snapshot, advance, report, unblock, cancel, sync-outer, "
                        "submit, campaign-snapshot, funnel, fixture, stream-snapshot, stream-submit, stream-run")
    verb, rest = argv[0], argv[1:]
    if verb in ("stream-snapshot", "stream-submit", "stream-run"):
        # Stream mode (docs/CONTINUOUS_LOOP.md 6.2): the verbs live in
        # stream_remote.py, which reuses this module's snapshot, report,
        # advance, status and dev-only scrub.
        from . import stream_remote as SR
        return SR.dispatch(verb, rest)
    if verb == "submit":
        builder, meta, dry, args = _split_submit(rest)
        return submit(builder, args, meta=meta, dry_run=dry)
    ap = _Parser(prog="inc_autopilot.remote %s" % verb)
    if verb == "status":
        ap.parse_args(rest)
        return status()
    if verb in ("cancel", "sync-outer"):
        if verb == "cancel":
            ap.add_argument("--exp", required=True)
        ap.add_argument("--approval-id", default=None)
        ap.add_argument("--decided-by", default=None)
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args(rest)
        meta = {k: v for k, v in (("approval_id", a.approval_id), ("decided_by", a.decided_by)) if v is not None}
        if verb == "cancel":
            return cancel(a.exp, dry_run=a.dry_run, meta=meta)
        return sync_outer(dry_run=a.dry_run, meta=meta)
    if verb in ("snapshot", "advance", "report", "unblock"):
        ap.add_argument("--exp", required=True)
        if verb == "snapshot":
            ap.add_argument("--ledger-from", default="0")
            ap.add_argument("--no-step1", action="store_true")
        if verb == "unblock":
            ap.add_argument("--unit", required=True)
            ap.add_argument("--reason", required=True)
        a = ap.parse_args(rest)
        if verb == "snapshot":
            lf, lsha = _ledger_arg(a.ledger_from)
            return snapshot(a.exp, ledger_from=lf, step1=not a.no_step1, ledger_sha256=lsha)
        if verb == "advance":
            return advance(a.exp)
        if verb == "report":
            return report(a.exp)
        return unblock(a.exp, a.unit, a.reason)
    if verb == "campaign-snapshot":
        ap.add_argument("--exp", action="append", required=True)
        ap.add_argument("--advance", action="store_true")
        ap.add_argument("--report", choices=("auto", "always", "never"), default="auto")
        ap.add_argument("--ledger-from", action="append", default=[])
        ap.add_argument("--no-step1", action="store_true")
        ap.add_argument("--funnel", action="store_true")
        ap.add_argument("--sacct", action="append", default=[])
        a = ap.parse_args(rest)
        bad = [j for j in a.sacct if not re.match(r"^[0-9]+(_[0-9]+)?$", j)]
        if bad:
            raise _ArgError("--sacct takes Slurm job ids, got %r" % bad)
        return campaign_snapshot(a.exp, do_advance=a.advance, report_mode=a.report,
                                 ledger_from=_ledger_map(a.ledger_from), step1=not a.no_step1, funnel=a.funnel,
                                 sacct=a.sacct)
    if verb == "funnel":
        if not rest or rest[0] not in ("summary", "dev-scores", "ledger-summaries"):
            raise _ArgError("funnel summary [--derive-ledger] | dev-scores --exp E [--exp ...] [--truth-step S] | "
                            "ledger-summaries [--write]")
        sub, rest = rest[0], rest[1:]
        ap = _Parser(prog="inc_autopilot.remote funnel %s" % sub)
        if sub == "summary":
            ap.add_argument("--derive-ledger", action="store_true")
            a = ap.parse_args(rest)
            return funnel_summary(derive_ledger=a.derive_ledger)
        if sub == "dev-scores":
            ap.add_argument("--exp", action="append", required=True)
            ap.add_argument("--truth-step", default=None)
            a = ap.parse_args(rest)
            return funnel_dev_scores(a.exp, truth_step=a.truth_step)
        ap.add_argument("--write", action="store_true")
        a = ap.parse_args(rest)
        return funnel_ledger_summaries(write=a.write)
    if verb == "fixture":
        ap.add_argument("--from", dest="src", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args(rest)
        rec = base_record("fixture")
        try:
            with open(a.src) as fh:
                src = parse_one(fh.read())
            rec["written"] = write_fixture(src, a.out)
        except (OSError, ValueError) as e:
            fail(rec, e, type(e).__name__, "other")
        return rec
    raise _ArgError("unknown verb %r" % verb)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    verb = argv[0] if argv else ""
    real_out = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            rec = dispatch(argv)
    except _ArgError as e:
        rec = fail({"verb": verb or None, "format": FORMAT, "utc": M.utc_now()}, e, "UsageError", "usage")
    except Exception as e:                      # the one line is printed whatever happened
        rec = fail({"verb": verb or None, "format": FORMAT, "utc": M.utc_now()}, e, type(e).__name__, "crash")
        rec["trace"] = _short(traceback.format_exc(), 2000)
    emit(rec, real_out)
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
